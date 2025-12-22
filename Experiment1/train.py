#!/usr/bin/env python3
"""
Training script for DSA RL Experiment
Trains an agent to follow curves using PPO with curriculum learning.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
import os
import sys
import cv2
import glob
import shutil
import json
from datetime import datetime

# ---------- GLOBALS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 8 Movement Actions + 1 Stop Action = 9 Total
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
ACTION_STOP_IDX = 8
N_ACTIONS = 9

STEP_ALPHA = 2.0
CROP = 33
EPSILON = 1e-6

# ---------- HELPERS ----------
def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32(img: np.ndarray, cy: int, cx: int, size=CROP):
    h, w = img.shape
    corners = [img[0,0], img[0, w-1], img[h-1, 0], img[h-1, w-1]]
    bg_estimate = np.median(corners)
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0
    
    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    
    out = np.full((size, size), pad_val, dtype=img.dtype)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    
    oy0 = sy0 - y0; ox0 = sx0 - x0
    sh  = sy1 - sy0; sw  = sx1 - sx0
    
    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]
    return out

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def get_distance_to_poly(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return np.sqrt(np.min(d2))

def nearest_gt_index(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return int(np.argmin(d2))

@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray

# ---------- UNIFIED ENVIRONMENT ----------
class CurveEnvUnified:
    def __init__(self, h=128, w=128, max_steps=200, curves_dir=None):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.curves_dir = curves_dir
        
        # Load pre-generated curves if directory provided
        if curves_dir is not None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.curves_dir = os.path.join(script_dir, curves_dir)
            if not os.path.exists(self.curves_dir):
                raise FileNotFoundError(f"Curves directory not found: {self.curves_dir}")
            
            # Find all curve files
            curve_files = sorted(glob.glob(os.path.join(self.curves_dir, "curve_*.png")))
            if len(curve_files) == 0:
                raise ValueError(f"No curve files found in {self.curves_dir}. Please generate curves first using curve_generator.py")
            
            self.curve_files = curve_files
            self.num_curves = len(self.curve_files)
            self.curve_idx = 0
            print(f"[ENV] Loaded {self.num_curves} pre-generated curves from {self.curves_dir}")
        else:
            raise ValueError("curves_dir must be provided. Training requires pre-generated curves.")
        
        self.stage_config = {
            'stage_id': 1,
            'width': (2, 4),
            'noise': 0.0,
            'invert': 0.5,
            'tissue': False,
            'strict_stop': False,
            'mixed_start': False
        }
        self.reset()

    def set_stage(self, config):
        self.stage_config.update(config)
        print(f"\n[ENV] Config Updated for Stage {self.stage_config.get('stage_id')}:")
        print(f"      Width: {self.stage_config['width']}, Noise: {self.stage_config['noise']}")
        print(f"      Tissue: {self.stage_config['tissue']}, Strict Stop: {self.stage_config['strict_stop']}")

    def generate_tissue_noise(self):
        noise = np.random.randn(self.h, self.w)
        tissue = gaussian_filter(noise, sigma=np.random.uniform(2.0, 5.0))
        tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min())
        return tissue * np.random.uniform(0.2, 0.4)

    def reset(self):
        # Load a random pre-generated curve
        curve_idx = np.random.randint(0, self.num_curves)
        curve_file = self.curve_files[curve_idx]
        
        # Extract index from filename (e.g., "curve_00042.png" -> 42)
        base_name = os.path.basename(curve_file)
        idx_str = base_name.replace("curve_", "").replace(".png", "")
        
        # Load image
        img = cv2.imread(curve_file, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        
        # Load mask
        mask_file = os.path.join(self.curves_dir, f"mask_{idx_str}.png")
        if os.path.exists(mask_file):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
        else:
            # Fallback: create mask from image if not found
            mask = (img > 0.5).astype(np.uint8)
        
        # Load points
        pts_file = os.path.join(self.curves_dir, f"points_{idx_str}.npy")
        if os.path.exists(pts_file):
            gt_poly = np.load(pts_file).astype(np.float32)
        else:
            raise FileNotFoundError(f"Points file not found: {pts_file}. Please regenerate curves.")

        if self.stage_config['tissue']:
            tissue = self.generate_tissue_noise()
            is_white_bg = np.mean([img[0,0], img[0,-1]]) > 0.5
            if is_white_bg:
                img = np.clip(img - tissue, 0.0, 1.0)
            else:
                img = np.clip(img + tissue, 0.0, 1.0)

        self.gt_map = np.zeros_like(img)
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0<=r<self.h and 0<=c<self.w:
                self.gt_map[r,c] = 1.0
        
        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly)

        use_cold_start = False
        if self.stage_config['mixed_start']:
            use_cold_start = (np.random.rand() < 0.5)

        if use_cold_start:
            curr = gt_poly[0]
            self.history_pos = [tuple(curr)] * 3
            self.prev_idx = 0
            self.agent = (float(curr[0]), float(curr[1]))
        else:
            start_idx = 5 if len(gt_poly) > 10 else 0
            curr = gt_poly[start_idx]
            p1 = gt_poly[max(0, start_idx-1)]
            p2 = gt_poly[max(0, start_idx-2)]
            self.history_pos = [tuple(p2), tuple(p1), tuple(curr)]
            self.prev_idx = start_idx
            self.agent = (float(curr[0]), float(curr[1]))

        self.steps = 0
        self.prev_action = -1
        self.path_mask = np.zeros_like(mask, dtype=np.float32)
        self.path_points = [self.agent]
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
        self.L_prev = get_distance_to_poly(self.agent, self.ep.gt_poly)
        return self.obs()

    def obs(self):
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        ch0 = crop32(self.ep.img, int(curr[0]), int(curr[1]))
        ch1 = crop32(self.ep.img, int(p1[0]), int(p1[1]))
        ch2 = crop32(self.ep.img, int(p2[0]), int(p2[1]))
        ch3 = crop32(self.path_mask, int(curr[0]), int(curr[1]))
        
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        gt_crop = crop32(self.gt_map, int(curr[0]), int(curr[1]))
        gt_obs = gt_crop[None, ...]
        
        return {"actor": actor_obs, "critic_gt": gt_obs}

    def step(self, a_idx: int):
        self.steps += 1
        dist_to_end = np.sqrt(
            (self.agent[0] - self.ep.gt_poly[-1][0])**2 + 
            (self.agent[1] - self.ep.gt_poly[-1][1])**2
        )

        if a_idx == ACTION_STOP_IDX:
            if dist_to_end < 5.0:
                return self.obs(), 50.0, True, {"reached_end": True, "stopped_correctly": True}
            else:
                return self.obs(), -2.0, False, {"reached_end": False, "stopped_correctly": False}

        dy, dx = ACTIONS_MOVEMENT[a_idx]
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)
        self.agent = (ny, nx)
        
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0

        L_t = get_distance_to_poly(self.agent, self.ep.gt_poly)
        dist_diff = abs(L_t - self.L_prev)
        best_idx = nearest_gt_index(self.agent, self.ep.gt_poly)
        progress_delta = best_idx - self.prev_idx
        
        sigma = 1.5 if self.stage_config['stage_id'] == 1 else 1.0
        precision_score = np.exp(-(L_t**2) / (2 * sigma**2))
        
        if L_t < self.L_prev:
            r = np.log(EPSILON + dist_diff)
        else:
            r = -np.log(EPSILON + dist_diff)
        r = float(np.clip(r, -2.0, 2.0))

        if progress_delta > 0:
            r += precision_score * 2.0
        elif progress_delta <= 0:
            r -= 0.1

        if self.stage_config['stage_id'] >= 2:
            lookahead_idx = min(best_idx + 4, len(self.ep.gt_poly) - 1)
            gt_vec = self.ep.gt_poly[lookahead_idx] - self.ep.gt_poly[best_idx]
            act_vec = np.array([dy, dx])
            norm_gt = np.linalg.norm(gt_vec)
            norm_act = np.linalg.norm(act_vec)
            if norm_gt > 1e-6 and norm_act > 1e-6:
                cos_sim = np.dot(gt_vec, act_vec) / (norm_gt * norm_act)
                if cos_sim > 0: r += cos_sim * 0.5

        if self.prev_action != -1 and self.prev_action != a_idx:
            r -= 0.05
        self.prev_action = a_idx
        r -= 0.05

        self.L_prev = L_t
        self.prev_idx = max(self.prev_idx, best_idx)

        done = False
        reached_end = (dist_to_end < 5.0)
        
        off_track_limit = 10.0 if self.stage_config['stage_id'] == 1 else 8.0
        off_track = L_t > off_track_limit
        
        if off_track:
            r -= 5.0
            done = True
        
        if self.steps >= self.max_steps:
            done = True

        if self.stage_config['strict_stop'] is False:
            if reached_end:
                r += 20.0
                done = True
        else:
            if reached_end:
                r += 0.5 

        return self.obs(), r, done, {"reached_end": reached_end, "stopped_correctly": False}

# ---------- NETWORK ARCHITECTURE ----------
# Import from shared models file
from models import AsymmetricActorCritic

# ---------- PPO UPDATE ----------
def update_ppo(ppo_opt, model, buf_list, clip=0.2, epochs=4, minibatch=32):
    obs_a = torch.tensor(np.concatenate([x['obs']['actor'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    obs_c = torch.tensor(np.concatenate([x['obs']['critic_gt'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ahist = torch.tensor(np.concatenate([x['ahist'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    act   = torch.tensor(np.concatenate([x['act'] for x in buf_list]), dtype=torch.long, device=DEVICE)
    logp  = torch.tensor(np.concatenate([x['logp'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    adv   = torch.tensor(np.concatenate([x['adv'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ret   = torch.tensor(np.concatenate([x['ret'] for x in buf_list]), dtype=torch.float32, device=DEVICE)

    if adv.numel() > 1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    N = obs_a.shape[0]
    idxs = np.arange(N)
    for _ in range(epochs):
        np.random.shuffle(idxs)
        for s in range(0, N, minibatch):
            mb = idxs[s:s+minibatch]
            if len(mb) == 0: continue
            
            logits, val, _, _ = model(obs_a[mb], obs_c[mb], ahist[mb])
            logits = torch.clamp(logits, -20, 20)
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(act[mb])
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_logp - logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0-clip, 1.0+clip) * adv[mb]
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(val, ret[mb])
            
            loss = p_loss + 0.5 * v_loss - 0.01 * entropy
            
            ppo_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            ppo_opt.step()

# ---------- MAIN CURRICULUM MANAGER ----------
def run_unified_training(run_dir, curves_base_dir, clean_previous=False, experiment_name=None, resume_from=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Handle resume from checkpoint
    resume_stage_idx = None
    resume_episode = None
    if resume_from is not None:
        resume_path = os.path.abspath(resume_from)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        
        # Extract run directory from checkpoint path
        # Checkpoint path: runs/.../checkpoints/ckpt_StageX_epY.pth
        checkpoint_dir = os.path.dirname(resume_path)
        run_dir = os.path.dirname(checkpoint_dir)  # Go up from checkpoints/ to run_dir
        
        # Load config to determine stage
        config_file = os.path.join(run_dir, "config.json")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found in run directory: {config_file}")
        
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        # Extract stage name and episode from checkpoint filename
        checkpoint_name = os.path.basename(resume_path)
        # Format: ckpt_Stage1_Bootstrap_ep2000.pth
        if "_ep" in checkpoint_name:
            parts = checkpoint_name.replace("ckpt_", "").replace(".pth", "").split("_ep")
            stage_name = parts[0]
            resume_episode = int(parts[1])
            
            # Find which stage index this corresponds to
            stage_names = ['Stage1_Bootstrap', 'Stage2_Robustness', 'Stage3_Realism']
            if stage_name in stage_names:
                resume_stage_idx = stage_names.index(stage_name)
        
        print(f"\n🔄 RESUMING TRAINING FROM CHECKPOINT")
        print(f"   Checkpoint: {resume_path}")
        print(f"   Run Directory: {run_dir}")
        if resume_stage_idx is not None:
            print(f"   Resuming from: {stage_names[resume_stage_idx]}, Episode {resume_episode}")
        print()
    else:
        # Create new timestamped run directory
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if experiment_name:
                # Format: runs/experiment_name_TIMESTAMP/
                run_dir = os.path.join(script_dir, "runs", f"{experiment_name}_{timestamp}")
            else:
                run_dir = os.path.join(script_dir, "runs", timestamp)
        else:
            run_dir = os.path.join(script_dir, run_dir)
    
    # Clean previous runs if requested
    runs_base = os.path.join(script_dir, "runs")
    if clean_previous and os.path.exists(runs_base):
        print(f"\n⚠️  Cleaning previous runs from {runs_base}...")
        try:
            shutil.rmtree(runs_base)
            print("✅ Previous runs cleaned successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not clean previous runs: {e}")
    
    # Create run directory structure
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    weights_dir = os.path.join(run_dir, "weights")
    logs_dir = os.path.join(run_dir, "logs")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(logs_dir, "training.log")
    
    # Save training configuration
    config_file = os.path.join(run_dir, "config.json")
    training_config = {
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "n_actions": N_ACTIONS,
        "curves_base_dir": curves_base_dir,
        "stages": []
    }
    
    print("=== STARTING UNIFIED RL TRAINING (3 STAGES) ===")
    print(f"Device: {DEVICE} | Actions: {N_ACTIONS} (Inc. STOP)")
    print(f"Run Directory: {run_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Weights: {weights_dir}")
    print(f"Logs: {log_file}")
    print(f"Curves Base Directory: {curves_base_dir}")
    
    # Redirect stdout to both console and log file
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
        def __getattr__(self, name):
            # Forward any other attributes to the first file (original stdout)
            return getattr(self.files[0], name)
    
    log_fp = open(log_file, 'w')
    original_stdout = sys.stdout
    sys.stdout = TeeOutput(original_stdout, log_fp)

    stages = [
        {
            'name': 'Stage1_Bootstrap',
            'episodes': 8000,
            'lr': 1e-4,
            'curves_dir': os.path.join(curves_base_dir, 'stage1'),
            'config': {
                'stage_id': 1, 'width': (2, 4), 'noise': 0.0, 
                'tissue': False, 'strict_stop': False, 'mixed_start': False
            }
        },
        {
            'name': 'Stage2_Robustness',
            'episodes': 12000,
            'lr': 5e-5,
            'curves_dir': os.path.join(curves_base_dir, 'stage2'),
            'config': {
                'stage_id': 2, 'width': (2, 8), 'noise': 0.5, 
                'tissue': False, 'strict_stop': True, 'mixed_start': True
            }
        },
        {
            'name': 'Stage3_Realism',
            'episodes': 15000,
            'lr': 1e-5,
            'curves_dir': os.path.join(curves_base_dir, 'stage3'),
            'config': {
                'stage_id': 3, 'width': (1, 10), 'noise': 0.8, 
                'tissue': True, 'strict_stop': True, 'mixed_start': True
            }
        }
    ]

    model = AsymmetricActorCritic(n_actions=N_ACTIONS).to(DEVICE)
    K = 8
    
    # Load checkpoint if resuming
    if resume_from is not None:
        print(f"Loading checkpoint: {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location=DEVICE))
        print("✅ Checkpoint loaded successfully")
        
        # Load existing metrics if available
        metrics_file = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
            print("✅ Loaded existing metrics")
        else:
            all_metrics = {"stages": []}
    else:
        # Initialize metrics storage
        all_metrics = {
            "stages": []
        }

    for stage_idx, stage in enumerate(stages):
        # Skip completed stages if resuming
        if resume_stage_idx is not None and stage_idx < resume_stage_idx:
            print(f"\n⏭️  Skipping {stage['name']} (already completed)")
            continue
        
        # Check if we need to resume mid-stage
        start_episode = 1
        if resume_stage_idx is not None and stage_idx == resume_stage_idx and resume_episode is not None:
            start_episode = resume_episode + 1
            print(f"\n🔄 Resuming {stage['name']} from episode {start_episode}")
            resume_stage_idx = None  # Clear resume flag after using it
        # Add stage config to training config
        stage_config_entry = {
            "name": stage['name'],
            "episodes": stage['episodes'],
            "lr": stage['lr'],
            "curves_dir": stage['curves_dir'],
            "config": stage['config']
        }
        training_config["stages"].append(stage_config_entry)
        
        # Initialize stage metrics
        stage_metrics = {
            "name": stage['name'],
            "episodes": [],
            "rewards": [],
            "success_rates": [],
            "avg_rewards": []
        }
        print(f"\n=============================================")
        print(f"STARTING {stage['name']}")
        print(f"Episodes: {stage['episodes']} | LR: {stage['lr']}")
        print(f"Curves Directory: {stage['curves_dir']}")
        print(f"=============================================")
        
        # Create environment with stage-specific curves
        env = CurveEnvUnified(h=128, w=128, curves_dir=stage['curves_dir'])
        env.set_stage(stage['config'])
        opt = torch.optim.Adam(model.parameters(), lr=stage['lr'])
        
        batch_buffer = []
        ep_returns = []
        ep_successes = []
        
        # Load existing metrics for this stage if resuming
        stage_metrics = None
        if resume_from is not None and len(all_metrics.get("stages", [])) > stage_idx:
            stage_metrics = all_metrics["stages"][stage_idx].copy()  # Make a copy to avoid modifying original
            # Restore episode returns and successes from metrics if available
            if "rewards" in stage_metrics and len(stage_metrics["rewards"]) > 0:
                ep_returns = stage_metrics["rewards"][:start_episode-1]
            if "successes" in stage_metrics and len(stage_metrics["successes"]) > 0:
                ep_successes = stage_metrics["successes"][:start_episode-1]
            # Ensure lists exist for appending
            if "rewards" not in stage_metrics:
                stage_metrics["rewards"] = []
            if "successes" not in stage_metrics:
                stage_metrics["successes"] = []
            if "episodes" not in stage_metrics:
                stage_metrics["episodes"] = []
            if "avg_rewards" not in stage_metrics:
                stage_metrics["avg_rewards"] = []
            if "success_rates" not in stage_metrics:
                stage_metrics["success_rates"] = []
        else:
            # Initialize stage metrics
            stage_metrics = {
                "name": stage['name'],
                "episodes": [],
                "rewards": [],
                "successes": [],
                "success_rates": [],
                "avg_rewards": []
            }
        
        for ep in range(start_episode, stage['episodes'] + 1):
            obs_dict = env.reset()
            done = False
            
            ahist = []
            ep_traj = {
                "obs":{'actor':[], 'critic_gt':[]}, "ahist":[], 
                "act":[], "logp":[], "val":[], "rew":[]
            }
            
            while not done:
                obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
                obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
                
                A = fixed_window_history(ahist, K, N_ACTIONS)[None, ...]
                A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    logits, value, _, _ = model(obs_a, obs_c, A_t)
                    logits = torch.clamp(logits, -20, 20)
                    dist = Categorical(logits=logits)
                    action = dist.sample().item()
                    logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                    val = value.item()

                next_obs, r, done, info = env.step(action)

                ep_traj["obs"]['actor'].append(obs_dict['actor'])
                ep_traj["obs"]['critic_gt'].append(obs_dict['critic_gt'])
                ep_traj["ahist"].append(A[0])
                ep_traj["act"].append(action)
                ep_traj["logp"].append(logp)
                ep_traj["val"].append(val)
                ep_traj["rew"].append(r)
                
                a_onehot = np.zeros(N_ACTIONS); a_onehot[action] = 1.0
                ahist.append(a_onehot)
                obs_dict = next_obs

            if len(ep_traj["rew"]) > 2:
                rews = np.array(ep_traj["rew"])
                vals = np.array(ep_traj["val"] + [0.0])
                delta = rews + 0.9 * vals[1:] - vals[:-1]
                adv = np.zeros_like(rews)
                acc = 0
                for t in reversed(range(len(rews))):
                    acc = delta[t] + 0.9 * 0.95 * acc
                    adv[t] = acc
                ret = adv + vals[:-1]
                
                final_ep_data = {
                    "obs": {"actor": np.array(ep_traj["obs"]['actor']), "critic_gt": np.array(ep_traj["obs"]['critic_gt'])},
                    "ahist": np.array(ep_traj["ahist"]),
                    "act": np.array(ep_traj["act"]),
                    "logp": np.array(ep_traj["logp"]),
                    "adv": adv, "ret": ret
                }
                batch_buffer.append(final_ep_data)
                ep_return = sum(rews)
                ep_returns.append(ep_return)
                
                success_val = 0
                if stage['config']['strict_stop']:
                    success_val = 1 if info.get('stopped_correctly') else 0
                else:
                    success_val = 1 if info.get('reached_end') else 0
                ep_successes.append(success_val)
                
                # Update metrics
                stage_metrics["rewards"].append(float(ep_return))
                stage_metrics["successes"].append(success_val)

            if len(batch_buffer) >= 32:
                update_ppo(opt, model, batch_buffer)
                batch_buffer = []

            if ep % 100 == 0:
                avg_r = np.mean(ep_returns[-100:])
                succ_rate = np.mean(ep_successes[-100:]) if ep_successes else 0.0
                print(f"[{stage['name']}] Ep {ep} | Avg Rew: {avg_r:.2f} | Success: {succ_rate:.2f}")
                
                # Save metrics
                stage_metrics["episodes"].append(ep)
                stage_metrics["avg_rewards"].append(float(avg_r))
                stage_metrics["success_rates"].append(float(succ_rate))
                
                # Save metrics periodically
                if ep % 1000 == 0:
                    metrics_file = os.path.join(run_dir, "metrics.json")
                    # Update or append stage metrics
                    if stage_idx < len(all_metrics["stages"]):
                        all_metrics["stages"][stage_idx] = stage_metrics
                    else:
                        all_metrics["stages"].append(stage_metrics)
                    with open(metrics_file, 'w') as f:
                        json.dump(all_metrics, f, indent=2)

            if ep % 2000 == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{stage['name']}_ep{ep}.pth")
                torch.save(model.state_dict(), ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")
                
                # Also save actor-only weights for inference
                actor_weights = {k: v for k, v in model.state_dict().items() if k.startswith('actor_')}
                actor_path = os.path.join(checkpoint_dir, f"actor_{stage['name']}_ep{ep}.pth")
                torch.save(actor_weights, actor_path)
                print(f"Actor-only weights saved: {actor_path}")

        final_path = os.path.join(weights_dir, f"model_{stage['name']}_FINAL.pth")
        torch.save(model.state_dict(), final_path)
        print(f"Finished {stage['name']}. Saved model to {final_path}")
        
        # Also save actor-only weights for inference
        actor_weights = {k: v for k, v in model.state_dict().items() if k.startswith('actor_')}
        actor_final_path = os.path.join(weights_dir, f"actor_{stage['name']}_FINAL.pth")
        torch.save(actor_weights, actor_final_path)
        print(f"Actor-only weights saved: {actor_final_path}")
        
        # Add final metrics
        if ep_returns:
            stage_metrics["final_avg_reward"] = float(np.mean(ep_returns))
            stage_metrics["final_success_rate"] = float(np.mean(ep_successes)) if ep_successes else 0.0
            stage_metrics["total_episodes"] = len(ep_returns)
        
        # Update or append stage metrics
        if stage_idx < len(all_metrics["stages"]):
            all_metrics["stages"][stage_idx] = stage_metrics
        else:
            all_metrics["stages"].append(stage_metrics)
    
    # Save configuration and metrics
    with open(config_file, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Restore stdout and close log file
    sys.stdout = original_stdout
    log_fp.close()
    
    print("\n=== TRAINING COMPLETE ===")
    print(f"All results saved to: {run_dir}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Final weights: {weights_dir}")
    print(f"  - Training log: {log_file}")
    print(f"  - Configuration: {config_file}")
    print(f"  - Metrics: {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DSA RL agent")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory (default: runs/TIMESTAMP or runs/EXPERIMENT_NAME_TIMESTAMP). All results (checkpoints, weights, logs) will be saved here.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name. Creates runs/EXPERIMENT_NAME_TIMESTAMP/ directory. Ignored if --run_dir is specified.")
    parser.add_argument("--curves_base_dir", type=str, default="generated_curves",
                        help="Base directory containing stage-specific curve folders (stage1/, stage2/, stage3/)")
    parser.add_argument("--clean_previous", action="store_true",
                        help="Delete all previous runs before starting new training")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume training from a checkpoint file (e.g., runs/20251222_143022/checkpoints/ckpt_Stage1_Bootstrap_ep2000.pth)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.resume_from and args.clean_previous:
        print("⚠️  Warning: --clean_previous is ignored when using --resume_from")
    
    if args.resume_from and args.run_dir:
        print("⚠️  Warning: --run_dir is ignored when using --resume_from (using run directory from checkpoint)")
    
    run_unified_training(args.run_dir, args.curves_base_dir, args.clean_previous, 
                        args.experiment_name, args.resume_from)
