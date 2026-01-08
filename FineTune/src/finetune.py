#!/usr/bin/env python3
"""
Central Fine-Tuning Engine (Gold Standard: Experiment 4 - Decoupled Stop).
Supports loading standalone supervised Stop Modules as an option.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from datetime import datetime
from torch.distributions import Categorical

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import shared logic from Experiment 4 and StopModule
try:
    from Experiment4_separate_stop_v2.src.train import (
        CurveEnvUnified, update_ppo, load_curve_config, 
        DEVICE, ACTIONS_MOVEMENT, ACTION_STOP_IDX, N_MOVEMENT_ACTIONS,
        fixed_window_history
    )
    from Experiment4_separate_stop_v2.src.models import DecoupledStopBackboneActorCritic
    from StopModule.src.models import StandaloneStopDetector
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

class HybridFinetuneModel(nn.Module):
    """
    A wrapper that allows swapping the RL stop head with a Standalone Supervised one.
    """
    def __init__(self, base_model, standalone_stop=None):
        super().__init__()
        self.base = base_model
        self.standalone_stop = standalone_stop # StandaloneStopDetector instance

    def forward(self, actor_obs, critic_gt, ahist_onehot, hc_actor=None, hc_critic=None):
        # 1. Standard Movement and Critic pass from Experiment 4
        # movement_logits, stop_logit, value, hc_actor, hc_critic
        m_logits, base_stop_logit, val, h_a, h_c = self.base(actor_obs, critic_gt, ahist_onehot, hc_actor, hc_critic)
        
        # 2. If standalone stop is provided, override the logit
        if self.standalone_stop is not None:
            # Standalone takes [image, path_mask] which are channels 0 and 3 of actor_obs
            stop_input = torch.cat([actor_obs[:, 0:1, :, :], actor_obs[:, 3:4, :, :]], dim=1)
            stop_logit = self.standalone_stop(stop_input)
        else:
            stop_logit = base_stop_logit
            
        return m_logits, stop_logit, val, h_a, h_c

def run_finetune(config_path=None, **kwargs):
    # 1. Load Configuration
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    champion_name = config.get("base_champion", kwargs.get("champion"))
    stop_weights = config.get("standalone_stop_weights", kwargs.get("stop_weights"))
    stage_id = config.get("target_stage_id", kwargs.get("stage", 11))
    episodes = config.get("episodes", kwargs.get("episodes", 5000))
    exp_name_base = config.get("experiment_name", kwargs.get("name", "finetune"))
    
    hp = config.get("hyperparameters", {})
    lr = hp.get("learning_rate", kwargs.get("lr", 1e-5))
    lambda_stop = hp.get("lambda_stop", kwargs.get("lambda_stop", 5.0))

    # 2. Setup Paths and Model
    finetune_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    champion_dir = os.path.join(finetune_dir, "base_weights", champion_name)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "runs", f"FT_Hybrid_{exp_name_base}_{timestamp}")
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    
    # Initialize Hybrid Model
    base_model = DecoupledStopBackboneActorCritic(n_movement_actions=N_MOVEMENT_ACTIONS).to(DEVICE)
    if os.path.exists(os.path.join(champion_dir, "weights.pth")):
        base_model.load_state_dict(torch.load(os.path.join(champion_dir, "weights.pth"), map_location=DEVICE))
        print(f"📥 Loaded Movement Champion: {champion_name}")

    standalone_stop = None
    if stop_weights:
        standalone_stop = StandaloneStopDetector().to(DEVICE)
        standalone_stop.load_state_dict(torch.load(stop_weights, map_location=DEVICE))
        print(f"🛑 Loaded Standalone Stop Module: {stop_weights}")
    
    model = HybridFinetuneModel(base_model, standalone_stop).to(DEVICE)

    # 3. Setup Environment
    curve_config, _ = load_curve_config()
    env = CurveEnvUnified(h=128, w=128, stage_id=stage_id, curve_config=curve_config)
    
    # 4. Training Loop (Standard PPO loop adjusted for Hybrid model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batch_buffer = []
    ep_returns, ep_successes = [], []
    
    print(f"🚀 Fine-Tune session: {run_dir}")
    
    for ep in range(1, episodes + 1):
        obs_dict = env.reset()
        done, ahist = False, []
        ep_traj = {"obs":{'actor':[], 'critic_gt':[]}, "ahist":[], "act":[], "logp":[], "val":[], "rew":[], "stop_label":[]}
        
        while not done:
            obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
            obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
            A = fixed_window_history(ahist, 8, N_MOVEMENT_ACTIONS)[None, ...]
            A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                move_logits, stop_logit, value, _, _ = model(obs_a, obs_c, A_t)
                stop_prob = torch.sigmoid(stop_logit).view(-1)
                
                stop_sample = torch.bernoulli(stop_prob).item()
                if stop_sample >= 0.5:
                    action = ACTION_STOP_IDX
                    logp = torch.log(stop_prob + 1e-8).item()
                else:
                    dist = Categorical(logits=torch.clamp(move_logits, -20, 20))
                    action = dist.sample().item()
                    logp = torch.log1p(-stop_prob + 1e-8).item() + dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                val = value.item()

            next_obs, r, done, info = env.step(action)
            
            ep_traj["obs"]['actor'].append(obs_dict['actor'])
            ep_traj["obs"]['critic_gt'].append(obs_dict['critic_gt'])
            ep_traj["ahist"].append(A[0]); ep_traj["act"].append(action)
            ep_traj["logp"].append(logp); ep_traj["val"].append(val); ep_traj["rew"].append(r)
            ep_traj["stop_label"].append(1 if (info.get('stopped_correctly') or info.get('reached_end')) else 0)
            
            a_onehot = np.zeros(N_MOVEMENT_ACTIONS)
            if action < N_MOVEMENT_ACTIONS: a_onehot[action] = 1.0
            ahist.append(a_onehot); obs_dict = next_obs
            
        if len(ep_traj["rew"]) > 2:
            rews = np.array(ep_traj["rew"])
            vals = np.array(ep_traj["val"] + [0.0])
            delta = rews + 0.9 * vals[1:] - vals[:-1]
            adv = np.zeros_like(rews)
            acc = 0
            for t in reversed(range(len(rews))):
                acc = delta[t] + 0.9 * 0.95 * acc
                adv[t] = acc
            
            batch_buffer.append({
                "obs": {"actor": np.array(ep_traj["obs"]['actor']), "critic_gt": np.array(ep_traj["obs"]['critic_gt'])},
                "ahist": np.array(ep_traj["ahist"]), "act": np.array(ep_traj["act"]),
                "logp": np.array(ep_traj["logp"]), "adv": adv, "ret": adv + vals[:-1],
                "stop_label": np.array(ep_traj["stop_label"])
            })
            ep_returns.append(sum(rews))
            ep_successes.append(1 if info.get('stopped_correctly') or info.get('reached_end') else 0)

        if len(batch_buffer) >= 32:
            update_ppo(opt, model, batch_buffer, lambda_stop=lambda_stop)
            batch_buffer = []

        if ep % 100 == 0:
            avg_r, succ = np.mean(ep_returns[-100:]), np.mean(ep_successes[-100:])
            print(f"[FT-Hybrid] Ep {ep} | Rew: {avg_r:.2f} | Succ: {succ:.2f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(run_dir, "weights", "finetuned_hybrid_FINAL.pth"))
    print(f"✅ Hybrid Fine-tune complete. Saved to: {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to fine-tune JSON config")
    parser.add_argument("--champion", type=str, help="Champion name override")
    parser.add_argument("--stop_weights", type=str, help="Path to standalone stop weights (.pth)")
    parser.add_argument("--stage", type=int, help="Stage ID override")
    args = parser.parse_args()
    
    run_finetune(config_path=args.config, champion=args.champion, stop_weights=args.stop_weights, stage=args.stage)
