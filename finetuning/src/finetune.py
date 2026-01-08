#!/usr/bin/env python3
"""
Central Fine-Tuning Engine (Gold Standard: Experiment 4 - Decoupled Stop).
Supports configuration files for running multiple refinement experiments.
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import shutil
import numpy as np
from datetime import datetime
from torch.distributions import Categorical

# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import shared logic from Experiment 4 (Gold Standard: Decoupled Stop)
try:
    from Experiment4_separate_stop_v2.src.train import (
        CurveEnvUnified, update_ppo, load_curve_config, 
        DEVICE, ACTIONS_MOVEMENT, ACTION_STOP_IDX, N_MOVEMENT_ACTIONS,
        fixed_window_history
    )
    from Experiment4_separate_stop_v2.src.models import DecoupledStopBackboneActorCritic
except ImportError:
    print("❌ Error: Could not import training logic from Experiment4_separate_stop_v2.")
    sys.exit(1)

def run_finetuning(config_path=None, **kwargs):
    # 1. Load Configuration
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded fine-tuning config: {config_path}")
    
    # Merge with command line overrides
    champion_name = config.get("base_champion", kwargs.get("champion"))
    stage_id = config.get("target_stage_id", kwargs.get("stage", 11))
    episodes = config.get("episodes", kwargs.get("episodes", 5000))
    exp_name_base = config.get("experiment_name", kwargs.get("name", "finetune"))
    
    hp = config.get("hyperparameters", {})
    lr = hp.get("learning_rate", kwargs.get("lr", 1e-5))
    lambda_stop = hp.get("lambda_stop", kwargs.get("lambda_stop", 5.0))
    ppo_clip = hp.get("ppo_clip", 0.2)
    entropy_coef = hp.get("entropy_coef", 0.01)
    minibatch = hp.get("minibatch_size", 32)

    # 2. Setup Paths
    finetune_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    champion_dir = os.path.join(finetune_dir, "base_weights", champion_name)
    
    if not os.path.exists(champion_dir):
        print(f"❌ Champion '{champion_name}' not found in {champion_dir}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(project_root, "runs", f"FT_{exp_name_base}_{timestamp}")
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    
    # Save a copy of the config for recording
    with open(os.path.join(run_dir, "finetune_config.json"), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"🚀 Fine-Tuning session: {run_dir}")

    # 3. Initialize Model and Load Weights
    model = DecoupledStopBackboneActorCritic(n_movement_actions=N_MOVEMENT_ACTIONS).to(DEVICE)
    weights_path = os.path.join(champion_dir, "weights.pth")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    # 4. Setup Environment
    curve_config, _ = load_curve_config()
    env = CurveEnvUnified(h=128, w=128, stage_id=stage_id, curve_config=curve_config)
    
    # Apply Stage Overrides
    stages_cfg = curve_config.get('training_stages', [])
    target_stage = next((s for s in stages_cfg if s['stage_id'] == stage_id), None)
    stage_settings = target_stage.get('training', {}).copy() if target_stage else {}
    
    overrides = config.get("stage_overrides", {})
    if overrides:
        print(f"🛠️  Applying Stage Overrides: {overrides}")
        stage_settings.update(overrides)
    
    env.set_stage(stage_settings)

    # 5. Training Loop
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    batch_buffer = []
    ep_returns, ep_successes = [], []
    all_metrics = {"config": config, "stages": [{"name": f"FT_Stage{stage_id}", "rewards": [], "success_rates": []}]}
    
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
            update_ppo(opt, model, batch_buffer, clip=ppo_clip, minibatch=minibatch, lambda_stop=lambda_stop)
            batch_buffer = []

        if ep % 100 == 0:
            avg_r, succ = np.mean(ep_returns[-100:]), np.mean(ep_successes[-100:])
            print(f"[FT] Ep {ep} | Rew: {avg_r:.2f} | Succ: {succ:.2f} | LR: {lr}")
            all_metrics["stages"][0]["rewards"].append(float(avg_r))
            all_metrics["stages"][0]["success_rates"].append(float(succ))

    # 6. Save results
    final_path = os.path.join(run_dir, "weights", "finetuned_FINAL.pth")
    torch.save(model.state_dict(), final_path)
    with open(os.path.join(run_dir, "metrics.json"), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✅ Finished. Results in: {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to fine-tuning JSON config")
    parser.add_argument("--champion", type=str, help="Champion name override")
    parser.add_argument("--stage", type=int, help="Stage ID override")
    args = parser.parse_args()
    
    run_finetuning(config_path=args.config, champion=args.champion, stage=args.stage)
