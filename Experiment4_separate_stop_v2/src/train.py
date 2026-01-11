#!/usr/bin/env python3
"""
Training script for DSA RL Experiment
Trains an agent to follow curves using PPO with curriculum learning.
Curves are generated on-the-fly with reproducible seeds.
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
import scipy.ndimage
import os
import sys
import cv2
import shutil
import json
from datetime import datetime

# Add parent directory to path so 'src' package can be imported
# This allows imports to work both when running as script and when imported as module
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Add project root to path to import central CurveGeneratorModule
_project_root = os.path.dirname(_parent_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Import central curve generator module
from CurveGeneratorModule import load_curve_config, CenterlineMask5PointsGenerator

# ---------- GLOBALS ----------
# Auto-detect device: use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print("⚠️  GPU not available, using CPU (training will be slower)")

# 8 Movement Actions; Stop is handled separately by stop head
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
N_MOVEMENT_ACTIONS = len(ACTIONS_MOVEMENT)  # 8
ACTION_STOP_IDX = 8  # Reserve 8 for stop when talking to env

STEP_ALPHA = 2.0
CROP = 33
EPSILON = 1e-6

# Base seed for reproducibility (can be overridden)
BASE_SEED = 42

# ---------- CONFIG LOADING ----------
def load_curve_config(config_path=None):
    """Load curve generation configuration from JSON file.
    
    Looks for config in the following order:
    1. Explicit path provided (if config_path is given)
    2. config/curve_config.json (relative to Experiment4_separate_stop_v2 directory)
    3. curve_config.json (relative to Experiment4_separate_stop_v2 directory)
    
    Returns:
        tuple: (config_dict, actual_config_path)
        - config_dict: The loaded configuration dictionary (or empty dict if not found)
        - actual_config_path: The path that was actually used (or None if not found)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    experiment_dir = os.path.dirname(script_dir)  # Experiment4_separate_stop_v2 directory
    config_dir = os.path.join(experiment_dir, "config")  # Experiment4_separate_stop_v2/config/
    
    if config_path is None:
        # Default: look in config/ directory first, then experiment directory for backward compatibility
        default_paths = [
            os.path.join(config_dir, "curve_config.json"),
            os.path.join(experiment_dir, "curve_config.json")
        ]
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            config_path = default_paths[0]  # Use config/ path even if doesn't exist (for error message)
    
    # Convert to absolute path if relative
    if not os.path.isabs(config_path):
        # Try relative to config directory first
        test_path = os.path.join(config_dir, config_path)
        if os.path.exists(test_path):
            config_path = test_path
        else:
            # Try relative to experiment directory
            test_path = os.path.join(experiment_dir, config_path)
            if os.path.exists(test_path):
                config_path = test_path
            else:
                # Try relative to script directory (for backward compatibility)
                test_path = os.path.join(script_dir, config_path)
                if os.path.exists(test_path):
                    config_path = test_path
                else:
                    # Use the original path (will show error if not found)
                    config_path = os.path.join(config_dir, config_path) if not os.path.isabs(config_path) else config_path
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded experiment configuration from: {config_path}")
        return config, config_path
    else:
        print(f"⚠️  Config file not found: {config_path}")
        print(f"   Expected locations:")
        print(f"     - {os.path.join(config_dir, 'curve_config.json')}")
        print(f"     - {os.path.join(experiment_dir, 'curve_config.json')}")
        print("   Using default configuration")
        return {}, None

# ---------- CURVE GENERATION (On-The-Fly) ----------
# Using central CurveGeneratorModule with CenterlineMask5PointsGenerator
# Config loaded from strong_foundation_config.json

# Load strong_foundation config from central module
_curve_config_path = os.path.join(_project_root, "CurveGeneratorModule", "ExampleConfigs", "strong_foundation_config.json")
STRONG_FOUNDATION_CONFIG, _ = load_curve_config(_curve_config_path)

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

# ---------- UNIFIED ENVIRONMENT (On-The-Fly Generation) ----------
class CurveEnvUnified:
    def __init__(self, h=128, w=128, max_steps=200, base_seed=BASE_SEED, stage_id=1, curve_config=None):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.base_seed = base_seed
        self.current_episode = 0
        self.curve_config = curve_config or {}
        
        # Stage-specific curve generation parameters
        self.stage_config = {
            'stage_id': stage_id,
            'width': (2, 4),
            'noise': 0.0,
            'invert': 0.5,
            'tissue': False,
            'strict_stop': False,
            'mixed_start': False,
            'curvature_factor': 0.5,
            'min_intensity': 0.6,
            'max_intensity': None,
            'branches': False,
            'width_variation': 'none',
            'start_width': None,
            'end_width': None,
            'intensity_variation': 'none',
            'start_intensity': None,
            'end_intensity': None,
            'background_intensity': None
        }
        
        print(f"[ENV] On-the-fly curve generation enabled (base_seed={base_seed})")
        self.reset()

    def set_stage(self, config):
        self.stage_config.update(config)
        
        # Load stage-specific parameters from curve_config if available
        stages_cfg = self.curve_config.get('stages', {})
        stage_key = str(self.stage_config['stage_id'])
        
        if stage_key in stages_cfg:
            stage_curve_cfg = stages_cfg[stage_key]
            # Update from config file
            if 'width_range' in stage_curve_cfg:
                self.stage_config['width'] = tuple(stage_curve_cfg['width_range'])
            if 'curvature_factor' in stage_curve_cfg:
                self.stage_config['curvature_factor'] = stage_curve_cfg['curvature_factor']
            if 'min_intensity' in stage_curve_cfg:
                self.stage_config['min_intensity'] = stage_curve_cfg['min_intensity']
            if 'max_intensity' in stage_curve_cfg:
                self.stage_config['max_intensity'] = stage_curve_cfg['max_intensity']
            if 'branches' in stage_curve_cfg:
                self.stage_config['branches'] = stage_curve_cfg['branches']
            if 'width_variation' in stage_curve_cfg:
                self.stage_config['width_variation'] = stage_curve_cfg['width_variation']
            if 'start_width' in stage_curve_cfg:
                self.stage_config['start_width'] = stage_curve_cfg['start_width']
            if 'end_width' in stage_curve_cfg:
                self.stage_config['end_width'] = stage_curve_cfg['end_width']
            if 'intensity_variation' in stage_curve_cfg:
                self.stage_config['intensity_variation'] = stage_curve_cfg['intensity_variation']
            if 'start_intensity' in stage_curve_cfg:
                self.stage_config['start_intensity'] = stage_curve_cfg['start_intensity']
            if 'end_intensity' in stage_curve_cfg:
                self.stage_config['end_intensity'] = stage_curve_cfg['end_intensity']
            if 'background_intensity' in stage_curve_cfg:
                self.stage_config['background_intensity'] = stage_curve_cfg['background_intensity']
        
        # Config update logging removed (as requested)

    def generate_tissue_noise(self):
        tissue_cfg = self.curve_config.get('tissue_noise', {})
        sigma_range = tuple(tissue_cfg.get('sigma_range', [2.0, 5.0]))
        intensity_range = tuple(tissue_cfg.get('intensity_range', [0.2, 0.4]))
        
        noise = np.random.randn(self.h, self.w)
        tissue = gaussian_filter(noise, sigma=np.random.uniform(sigma_range[0], sigma_range[1]))
        tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min())
        return tissue * np.random.uniform(intensity_range[0], intensity_range[1])

    def reset(self, episode_number=None):
        """Reset environment and generate a new curve on-the-fly.
        
        Args:
            episode_number: Episode number for seed calculation. If None, uses self.current_episode.
        """
        if episode_number is not None:
            self.current_episode = episode_number
        
        # Calculate seed for this episode: base_seed + episode_number
        # This ensures reproducibility: same episode number = same curve
        episode_seed = self.base_seed + self.current_episode
        
        # Create curve generator using central CurveGeneratorModule with 5-point centerline generator
        # Use strong_foundation config merged with experiment-specific config
        merged_config = STRONG_FOUNDATION_CONFIG.copy() if STRONG_FOUNDATION_CONFIG else {}
        merged_config.update(self.curve_config)
        curve_maker = CenterlineMask5PointsGenerator(h=self.h, w=self.w, seed=episode_seed, config=merged_config)
        
        # Generate curve with stage-specific parameters
        img, mask, pts_all = curve_maker.sample_curve(
            width_range=self.stage_config['width'],
            noise_prob=0.0,  # Noise applied during training, not generation
            invert_prob=self.stage_config['invert'],
            min_intensity=self.stage_config['min_intensity'],
            max_intensity=self.stage_config.get('max_intensity', None),
            branches=self.stage_config['branches'],
            curvature_factor=self.stage_config['curvature_factor'],
            allow_self_cross=self.stage_config.get('allow_self_cross', False),
            self_cross_prob=self.stage_config.get('self_cross_prob', 0.0),
            width_variation=self.stage_config.get('width_variation', 'none'),
            start_width=self.stage_config.get('start_width', None),
            end_width=self.stage_config.get('end_width', None),
            intensity_variation=self.stage_config.get('intensity_variation', 'none'),
            start_intensity=self.stage_config.get('start_intensity', None),
            end_intensity=self.stage_config.get('end_intensity', None),
            background_intensity=self.stage_config.get('background_intensity', None)
        )
        
        # Extract main curve points
        gt_poly = pts_all[0].astype(np.float32)
        
        # Apply tissue noise if enabled
        if self.stage_config['tissue']:
            # Use a deterministic seed for tissue noise based on episode
            np.random.seed(episode_seed + 10000)  # Offset to avoid conflicts
            tissue = self.generate_tissue_noise()
            is_white_bg = np.mean([img[0,0], img[0,-1]]) > 0.5
            if is_white_bg:
                img = np.clip(img - tissue, 0.0, 1.0)
            else:
                img = np.clip(img + tissue, 0.0, 1.0)
            np.random.seed()  # Reset to random state

        # Create ground truth map
        self.gt_map = np.zeros_like(img)
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0<=r<self.h and 0<=c<self.w:
                self.gt_map[r,c] = 1.0
        
        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly)

        # Determine start position (with deterministic randomness)
        np.random.seed(episode_seed + 20000)  # Offset for start position
        use_cold_start = False
        if self.stage_config['mixed_start']:
            use_cold_start = (np.random.rand() < 0.5)
        np.random.seed()  # Reset to random state

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
        
        # Track if we're at the very beginning (for bootstrap rewards)
        self.is_at_start = (self.prev_idx <= 5)
        self.initial_steps = 0  # Count first few steps for bootstrap rewards
        
        # Increment episode counter for next reset
        self.current_episode += 1
        
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
                # Reward correct stop more heavily for strict_stop stages to align reward/success
                stop_bonus = 30.0 if self.stage_config['strict_stop'] else 20.0
                return self.obs(), stop_bonus, True, {"reached_end": True, "stopped_correctly": True}
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
        
        # Track initial steps for bootstrap rewards
        if self.is_at_start:
            self.initial_steps += 1
            # Stop tracking after 5 steps or when we've made progress
            if self.initial_steps >= 5 or progress_delta > 0:
                self.is_at_start = False
        
        sigma = 1.5 if self.stage_config['stage_id'] == 1 else 1.0
        precision_score = np.exp(-(L_t**2) / (2 * sigma**2))
        
        if L_t < self.L_prev:
            r = np.log(EPSILON + dist_diff)
        else:
            r = -np.log(EPSILON + dist_diff)
        r = float(np.clip(r, -2.0, 2.0))
        
        # Direction hint reward: encourage moving in the direction of the curve (after r is initialized)
        if self.is_at_start and self.initial_steps <= 5 and len(self.ep.gt_poly) > 1:
            # Get direction vector from start point
            start_pt = self.ep.gt_poly[0]
            # Look ahead a few points to get curve direction
            lookahead_idx = min(3, len(self.ep.gt_poly) - 1)
            direction_pt = self.ep.gt_poly[lookahead_idx]
            curve_dir = np.array([direction_pt[0] - start_pt[0], direction_pt[1] - start_pt[1]])
            curve_dir_norm = np.linalg.norm(curve_dir)
            
            if curve_dir_norm > 1e-6:
                curve_dir = curve_dir / curve_dir_norm
                action_dir = np.array([dy, dx])
                action_dir_norm = np.linalg.norm(action_dir)
                
                if action_dir_norm > 1e-6:
                    action_dir = action_dir / action_dir_norm
                    # Cosine similarity: 1.0 = same direction, -1.0 = opposite
                    direction_alignment = np.dot(curve_dir, action_dir)
                    # Reward alignment with curve direction (stronger for first steps)
                    direction_bonus = direction_alignment * (3.0 - self.initial_steps * 0.4) * 0.3
                    r += direction_bonus

        if progress_delta > 0:
            r += precision_score * 2.0
        elif progress_delta <= 0:
            r -= 0.1
        
        # Bootstrap reward for initial steps when starting from the beginning
        # Problem: At the start, all history positions are the same, action history is empty,
        #          so the model must infer direction from the image alone (very hard!)
        # Solution: Provide stronger rewards and direction hints for the first few steps
        #           to guide the model toward the correct initial direction
        if self.is_at_start and self.initial_steps <= 5:
            bootstrap_multiplier = max(1.0, 3.0 - self.initial_steps * 0.4)  # 3.0, 2.6, 2.2, 1.8, 1.4, 1.0
            if progress_delta > 0:
                # Strong reward for making progress from the start
                r += precision_score * bootstrap_multiplier * 1.5
            elif L_t < self.L_prev:
                # Reward for getting closer to the curve (even without progress yet)
                r += abs(r) * bootstrap_multiplier * 0.5
            # Less penalty for wrong moves at the start (encourages exploration)
            if progress_delta < 0:
                r = max(r, -0.5)  # Cap penalty at -0.5 instead of -0.1

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
                # Small shaping reward for getting to the end; must stop to finish
                r += 5.0

        return self.obs(), r, done, {"reached_end": reached_end, "stopped_correctly": False}

# ---------- NETWORK ARCHITECTURE ----------
# Import from shared models file
# Use absolute import - works both as module and script
from src.models import DecoupledStopBackboneActorCritic

# ---------- PPO UPDATE ----------
def update_ppo(ppo_opt, model, buf_list, clip=0.2, epochs=4, minibatch=32, lambda_stop=5.0):
    obs_a = torch.tensor(np.concatenate([x['obs']['actor'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    obs_c = torch.tensor(np.concatenate([x['obs']['critic_gt'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ahist = torch.tensor(np.concatenate([x['ahist'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    act   = torch.tensor(np.concatenate([x['act'] for x in buf_list]), dtype=torch.long, device=DEVICE)
    logp  = torch.tensor(np.concatenate([x['logp'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    adv   = torch.tensor(np.concatenate([x['adv'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ret   = torch.tensor(np.concatenate([x['ret'] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    stop_labels = torch.tensor(np.concatenate([x['stop_label'] for x in buf_list]), dtype=torch.float32, device=DEVICE)

    if adv.numel() > 1: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    
    N = obs_a.shape[0]
    idxs = np.arange(N)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        np.random.shuffle(idxs)
        for s in range(0, N, minibatch):
            mb = idxs[s:s+minibatch]
            if len(mb) == 0: continue
            
            # Decoupled model returns 5 values
            movement_logits, stop_logit, val, _, _ = model(obs_a[mb], obs_c[mb], ahist[mb])
            movement_logits = torch.clamp(movement_logits, -20, 20)
            stop_prob = torch.sigmoid(stop_logit)
            move_logp_all = F.log_softmax(movement_logits, dim=1)

            stop_logp = torch.log(stop_prob + 1e-8)
            cont_logp = torch.log1p(-stop_prob + 1e-8)

            act_mb = act[mb]
            stop_mask = (act_mb == ACTION_STOP_IDX)
            move_mask = ~stop_mask

            new_logp = torch.zeros_like(stop_prob, dtype=torch.float32)
            # Stop actions: log p(stop)
            if stop_mask.any():
                new_logp[stop_mask] = stop_logp[stop_mask]
            # Movement actions: log p(continue) + log p(move | continue)
            if move_mask.any():
                chosen_moves = act_mb[move_mask]
                move_lp = move_logp_all[move_mask, :].gather(1, chosen_moves.view(-1,1)).squeeze(1)
                new_logp[move_mask] = cont_logp[move_mask] + move_lp

            # Entropy: Bernoulli + categorical
            bern_entropy = -(stop_prob * stop_logp + (1 - stop_prob) * cont_logp)
            move_entropy = Categorical(logits=movement_logits).entropy()
            entropy = (bern_entropy + move_entropy).mean()

            ratio = torch.exp(new_logp - logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0-clip, 1.0+clip) * adv[mb]
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(val, ret[mb])
            stop_loss = bce(stop_logit, stop_labels[mb])
            
            loss = p_loss + 0.5 * v_loss + lambda_stop * stop_loss - 0.01 * entropy
            
            ppo_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            ppo_opt.step()

# ---------- MAIN CURRICULUM MANAGER ----------
def run_unified_training(run_dir, base_seed=BASE_SEED, clean_previous=False, experiment_name=None, resume_from=None, curve_config_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Experiment1 directory
    runs_base = os.path.join(parent_dir, "runs")
    
    # Load curve configuration
    curve_config, actual_config_path = load_curve_config(curve_config_path)
    
    # Get image dimensions from config if available
    img_cfg = curve_config.get('image', {})
    img_h = img_cfg.get('height', 128)
    img_w = img_cfg.get('width', 128)
    
    # Handle resume from checkpoint
    resume_stage_idx = None
    resume_episode = None
    saved_config = None
    if resume_from is not None:
        resume_path = os.path.abspath(resume_from)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        
        # Extract run directory from checkpoint path
        # Handle both checkpoints/ and weights/ directories
        checkpoint_dir = os.path.dirname(resume_path)
        if os.path.basename(checkpoint_dir) == "checkpoints":
            run_dir = os.path.dirname(checkpoint_dir)
        elif os.path.basename(checkpoint_dir) == "weights":
            run_dir = os.path.dirname(checkpoint_dir)
        else:
            # Fallback: assume checkpoint is in run_dir/checkpoints/
            run_dir = os.path.dirname(checkpoint_dir)
        
        # Load config to determine stage and seed
        config_file = os.path.join(run_dir, "curve_config.json")
        saved_config = {}
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                saved_config = json.load(f)
            print(f"✓ Loaded existing configuration from: {config_file}")
        else:
            # Fallback to old name if it exists
            old_config_file = os.path.join(run_dir, "config.json")
            if os.path.exists(old_config_file):
                with open(old_config_file, 'r') as f:
                    saved_config = json.load(f)
                print(f"✓ Loaded existing configuration from: {old_config_file}")
            else:
                print(f"⚠️  Config file not found in run directory")
                print("   Using defaults and command line arguments")
        
        # Get base_seed from config or use default
        base_seed = saved_config.get('base_seed', BASE_SEED)
        
        # If config.json exists and has stages, we can use them directly
        if saved_config and 'stages' in saved_config and len(saved_config['stages']) > 0:
            print("✓ Will use training stages defined in config.json")
        else:
            print("⚠️  No stages found in config.json, using provided curve_config")
        
        img_cfg = curve_config.get('image', {})
        img_h = img_cfg.get('height', 128)
        img_w = img_cfg.get('width', 128)
        
        # Extract stage name and episode from checkpoint filename
        checkpoint_name = os.path.basename(resume_path)
        resume_episode = None
        
        # Handle both "ckpt_StageName_epXXXX.pth" and "model_StageName_FINAL.pth" formats
        if "_ep" in checkpoint_name:
            parts = checkpoint_name.replace("ckpt_", "").replace(".pth", "").split("_ep")
            stage_name = parts[0]
            resume_episode = int(parts[1])
        elif "FINAL" in checkpoint_name:
            # Extract stage name from FINAL checkpoint (e.g., "model_Stage2_Curvature_FINAL.pth")
            parts = checkpoint_name.replace("model_", "").replace("_FINAL.pth", "").split("_")
            # Reconstruct stage name (e.g., "Stage2_Curvature")
            if len(parts) >= 2:
                stage_name = "_".join(parts)  # "Stage2_Curvature"
            else:
                stage_name = parts[0] if parts else "Unknown"
            # For FINAL checkpoints, resume from the next stage (episode 0 means start of next stage)
            resume_episode = None  # Will start from beginning of next stage
        
        # Extract stage number from stage name (e.g., "Stage2_Curvature" -> stage_num=2)
        # Stage numbers in names are 1-indexed, but resume_stage_idx is 0-indexed
        import re
        stage_num_match = re.search(r'Stage(\d+)_', stage_name)
        if stage_num_match:
            stage_num = int(stage_num_match.group(1))  # 1-indexed (Stage2 -> 2)
            if "FINAL" in checkpoint_name:
                # FINAL checkpoint means this stage is complete, so resume from NEXT stage
                resume_stage_idx = stage_num  # Stage2_FINAL -> resume from Stage3 (index 2)
                resume_episode = None  # Start from beginning of next stage
            else:
                # Regular checkpoint: resume from same stage, next episode
                resume_stage_idx = stage_num - 1  # Stage2 -> index 1 (0-indexed)
        else:
            # Fallback: try to map known stage names
            stage_name_mapping = {
                'Stage1_Bootstrap': 0, 'Stage1_Foundation': 0,
                'Stage2_Robustness': 1, 'Stage2_Curvature': 1,
                'Stage3_Realism': 2, 'Stage3_ThinPaths': 2
            }
            if stage_name in stage_name_mapping:
                resume_stage_idx = stage_name_mapping[stage_name]
                if "FINAL" in checkpoint_name:
                    resume_stage_idx += 1  # Move to next stage
                    resume_episode = None
        
        print(f"\n🔄 RESUMING TRAINING FROM CHECKPOINT")
        print(f"   Checkpoint: {resume_path}")
        print(f"   Run Directory: {run_dir}")
        print(f"   Base Seed: {base_seed}")
        if resume_stage_idx is not None:
            if resume_episode is not None:
                print(f"   Resuming from Stage {resume_stage_idx + 1}, Episode {resume_episode}")
            else:
                print(f"   Resuming from Stage {resume_stage_idx + 1} (start of stage)")
        print()
    else:
        # Create new timestamped run directory
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if experiment_name:
                run_dir = os.path.join(runs_base, f"{experiment_name}_{timestamp}")
            else:
                run_dir = os.path.join(runs_base, timestamp)
        else:
            if os.path.isabs(run_dir):
                run_dir = run_dir
            else:
                # Place relative run_dir under Experiment1/runs for cleanliness
                run_dir = os.path.join(runs_base, run_dir)
    
    # Clean previous runs if requested
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
    
    # Determine curve config source for metadata
    if resume_from is None:
        stored_config_path = actual_config_path if actual_config_path else curve_config_path
    else:
        # Use saved path if available
        if saved_config:
            stored_config_path = saved_config.get('curve_config_path', None)
        else:
            stored_config_path = actual_config_path if actual_config_path else curve_config_path
    
    # Save training configuration
    config_file = os.path.join(run_dir, "curve_config.json")
    training_config = {
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "n_movement_actions": N_MOVEMENT_ACTIONS,
        "has_stop_head": True,
        "base_seed": base_seed,
        "curve_generation": "on_the_fly",
        "generator_type": "CenterlineMask5PointsGenerator",
        "curve_config_path": stored_config_path,
        "image_height": img_h,
        "image_width": img_w,
        "stages": []
    }
    
    # Load training stages from config
    training_stages_cfg = curve_config.get('training_stages', None)
    
    if training_stages_cfg:
        # Load stages from config
        stages = []
        for stage_cfg in training_stages_cfg:
            stage_id = stage_cfg.get('stage_id')
            curve_gen = stage_cfg.get('curve_generation', {})
            training = stage_cfg.get('training', {})
            
            # Build stage config
            stage = {
                'name': stage_cfg.get('name', f'Stage{stage_id}'),
                'episodes': stage_cfg.get('episodes', 8000),
                'lr': stage_cfg.get('learning_rate', 1e-4),
                'config': {
                    'stage_id': stage_id,
                    'width': tuple(curve_gen.get('width_range', [2, 4])),
                    'noise': training.get('noise', 0.0),
                    'tissue': training.get('tissue', False),
                    'strict_stop': training.get('strict_stop', False),
                    'mixed_start': training.get('mixed_start', False),
                    'invert': curve_gen.get('invert_prob', 0.5),
                    'min_intensity': curve_gen.get('min_intensity', 0.6),
                    'max_intensity': curve_gen.get('max_intensity', None),
                    'branches': curve_gen.get('branches', False),
                    'curvature_factor': curve_gen.get('curvature_factor', 1.0),
                    'allow_self_cross': curve_gen.get('allow_self_cross', False),
                    'self_cross_prob': curve_gen.get('self_cross_prob', 0.0),
                    'width_variation': curve_gen.get('width_variation', 'none'),
                    'start_width': curve_gen.get('start_width', None),
                    'end_width': curve_gen.get('end_width', None),
                    'intensity_variation': curve_gen.get('intensity_variation', 'none'),
                    'start_intensity': curve_gen.get('start_intensity', None),
                    'end_intensity': curve_gen.get('end_intensity', None),
                    'background_intensity': curve_gen.get('background_intensity', None)
                }
            }
            stages.append(stage)
        
        num_stages = len(stages)
        print(f"✓ Loaded {num_stages} training stages from config")
    else:
        # Fallback to hardcoded defaults
        stages = [
            {
                'name': 'Stage1_Bootstrap',
                'episodes': 8000,
                'lr': 1e-4,
                'config': {
                    'stage_id': 1, 'width': (2, 4), 'noise': 0.0, 
                    'tissue': False, 'strict_stop': False, 'mixed_start': False
                }
            },
            {
                'name': 'Stage2_Robustness',
                'episodes': 12000,
                'lr': 5e-5,
                'config': {
                    'stage_id': 2, 'width': (2, 8), 'noise': 0.5, 
                    'tissue': False, 'strict_stop': True, 'mixed_start': True
                }
            },
            {
                'name': 'Stage3_Realism',
                'episodes': 15000,
                'lr': 1e-5,
                'config': {
                    'stage_id': 3, 'width': (1, 10), 'noise': 0.8, 
                    'tissue': True, 'strict_stop': True, 'mixed_start': True
                }
            }
        ]
        num_stages = len(stages)
        print(f"⚠️  No training_stages in config, using default {num_stages} stages")
    
    # Record training start time
    training_start_time = datetime.now()
    
    print("=== STARTING UNIFIED RL TRAINING ===")
    print(f"Number of Stages: {num_stages}")
    print(f"Device: {DEVICE} | Movement Actions: {N_MOVEMENT_ACTIONS} (+ separate STOP head)")
    print(f"Base Seed: {base_seed} (for reproducibility)")
    print(f"Curve Generation: On-The-Fly (CenterlineMask5PointsGenerator)")
    print(f"Run Directory: {run_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Weights: {weights_dir}")
    print(f"Logs: {log_file}")
    print(f"Training Start Time: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Anti-forgetting: Initialize parameters (defined early for print statement)
    anti_forgetting_prob = 0.15  # 15% chance to sample from previous stages
    eval_previous_every = 500  # Evaluate on previous stages every N episodes
    
    print(f"\n🛡️  Anti-Forgetting Mechanisms Enabled:")
    print(f"   - Mixed Curriculum: {anti_forgetting_prob*100:.0f}% chance to sample from previous stages")
    print(f"   - Periodic Evaluation: Every {eval_previous_every} episodes on previous stages")
    print(f"   - This prevents catastrophic forgetting during curriculum learning")
    
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

    # Use the new decoupled model class for Experiment 4
    model = DecoupledStopBackboneActorCritic(n_movement_actions=N_MOVEMENT_ACTIONS).to(DEVICE)
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
        all_metrics = {"stages": []}

    # Track global episode counter across stages for seed calculation
    global_episode_offset = 0
    
    # Anti-forgetting: Track all previous stage configs for mixed curriculum training
    previous_stages = []  # List of (stage_config, stage_name) tuples
    # Note: anti_forgetting_prob and eval_previous_every are defined earlier (before print statement)
    
    def evaluate_on_stage(model, stage_config, stage_name, num_episodes=10, base_seed=42):
        """Evaluate model performance on a previous stage to detect forgetting."""
        eval_env = CurveEnvUnified(h=img_h, w=img_w, base_seed=base_seed, 
                                   stage_id=stage_config['stage_id'], curve_config=curve_config)
        eval_env.set_stage(stage_config)
        
        reached_ends = []
        stopped_correctly = []
        returns = []
        
        for eval_ep in range(num_episodes):
            obs_dict = eval_env.reset(episode_number=base_seed + eval_ep + 100000)  # Offset seed
            done = False
            ahist = []
            ep_return = 0.0
            
            while not done:
                obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
                obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
                A = fixed_window_history(ahist, K, N_MOVEMENT_ACTIONS)[None, ...]
                A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
                
                with torch.no_grad():
                    movement_logits, stop_logit, _, _, _ = model(obs_a, obs_c, A_t)
                    movement_logits = torch.clamp(movement_logits, -20, 20)
                    stop_prob = torch.sigmoid(stop_logit).view(-1)
                    stop_sample = torch.bernoulli(stop_prob).item()
                    if stop_sample >= 0.5:
                        action = ACTION_STOP_IDX
                    else:
                        dist = Categorical(logits=movement_logits)
                        action = dist.sample().item()
                
                next_obs, r, done, info = eval_env.step(action)
                ep_return += r
                
                a_onehot = np.zeros(N_MOVEMENT_ACTIONS)
                if action != ACTION_STOP_IDX and action < N_MOVEMENT_ACTIONS:
                    a_onehot[action] = 1.0
                ahist.append(a_onehot)
                obs_dict = next_obs
            
            returns.append(ep_return)
            reached_ends.append(1 if info.get('reached_end') else 0)
            stopped_correctly.append(1 if info.get('stopped_correctly') else 0)
        
        return {
            'avg_return': float(np.mean(returns)),
            'reached_end_rate': float(np.mean(reached_ends)),
            'stop_success_rate': float(np.mean(stopped_correctly)),
            'stage_name': stage_name
        }

    for stage_idx, stage in enumerate(stages):
        # Skip completed stages if resuming
        if resume_stage_idx is not None and stage_idx < resume_stage_idx:
            print(f"\n⏭️  Skipping {stage['name']} (already completed)")
            # Update global episode offset
            global_episode_offset += stage['episodes']
            # Anti-forgetting: Add completed stages to previous_stages when resuming
            previous_stages.append((stage['config'].copy(), stage['name']))
            continue
        
        # Check if we need to resume mid-stage
        start_episode = 1
        if resume_stage_idx is not None and stage_idx == resume_stage_idx and resume_episode is not None:
            start_episode = resume_episode + 1
            print(f"\n🔄 Resuming {stage['name']} from episode {start_episode}")
            resume_stage_idx = None
        
        # Add stage config to training config
        stage_config_entry = {
            "name": stage['name'],
            "episodes": stage['episodes'],
            "lr": stage['lr'],
            "config": stage['config']
        }
        training_config["stages"].append(stage_config_entry)
        
        # Initialize stage metrics
        stage_metrics = {
            "name": stage['name'],
            "episodes": [],
            "rewards": [],
            "reached_end": [],
            "stopped_correctly": [],
            "avg_rewards": [],
            "reached_end_rates": [],
            "stop_success_rates": []
        }
        print(f"\n=============================================")
        print(f"STARTING {stage['name']}")
        print(f"Episodes: {stage['episodes']} | LR: {stage['lr']}")
        print(f"Global Episode Offset: {global_episode_offset}")
        print(f"=============================================")
        
        # Create environment with on-the-fly generation
        env = CurveEnvUnified(h=img_h, w=img_w, base_seed=base_seed, stage_id=stage['config']['stage_id'], curve_config=curve_config)
        env.set_stage(stage['config'])
        opt = torch.optim.Adam(model.parameters(), lr=stage['lr'])
        
        batch_buffer = []
        ep_returns = []
        ep_reached_end = []
        ep_stopped_correctly = []
        
        # Load existing metrics for this stage if resuming
        if resume_from is not None and len(all_metrics.get("stages", [])) > stage_idx:
            stage_metrics = all_metrics["stages"][stage_idx].copy()  # Make a copy to avoid modifying original
            # Restore episode returns and successes from metrics if available
            if "rewards" in stage_metrics and len(stage_metrics["rewards"]) > 0:
                ep_returns = stage_metrics["rewards"][:start_episode-1]
            if "reached_end" in stage_metrics and len(stage_metrics["reached_end"]) > 0:
                ep_reached_end = stage_metrics["reached_end"][:start_episode-1]
            if "stopped_correctly" in stage_metrics and len(stage_metrics["stopped_correctly"]) > 0:
                ep_stopped_correctly = stage_metrics["stopped_correctly"][:start_episode-1]
            
            # Ensure lists exist for appending
            for k in ["rewards", "reached_end", "stopped_correctly", "episodes", "avg_rewards", "reached_end_rates", "stop_success_rates"]:
                if k not in stage_metrics:
                    stage_metrics[k] = []
        else:
            # Already initialized stage_metrics above
            pass
        
        for ep in range(start_episode, stage['episodes'] + 1):
            # Calculate global episode number for seed
            global_episode = global_episode_offset + ep
            
            # Anti-forgetting: Periodically sample from previous stages (mixed curriculum)
            use_previous_stage = False
            if previous_stages and np.random.rand() < anti_forgetting_prob:
                # Sample from a random previous stage
                prev_idx = np.random.randint(0, len(previous_stages))
                prev_stage_config, prev_stage_name = previous_stages[prev_idx]
                env.set_stage(prev_stage_config)
                use_previous_stage = True
                eval_seed_offset = 50000  # Offset to avoid seed collision
                if ep % 100 == 0:  # Only print occasionally to avoid spam
                    print(f"   🔄 Anti-forgetting: Training on previous stage '{prev_stage_name}'")
            else:
                # Use current stage
                env.set_stage(stage['config'])
                use_previous_stage = False
                eval_seed_offset = 0
            
            obs_dict = env.reset(episode_number=global_episode + eval_seed_offset)
            done = False
            
            ahist = []
            ep_traj = {
                "obs":{'actor':[], 'critic_gt':[]}, "ahist":[], 
                "act":[], "logp":[], "val":[], "rew":[], "stop_label":[]
            }
            
            while not done:
                obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
                obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
                
                A = fixed_window_history(ahist, K, N_MOVEMENT_ACTIONS)[None, ...]
                A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    movement_logits, stop_logit, value, _, _ = model(obs_a, obs_c, A_t)
                    movement_logits = torch.clamp(movement_logits, -20, 20)
                    stop_prob = torch.sigmoid(stop_logit).view(-1)
                    stop_sample = torch.bernoulli(stop_prob).item()
                    if stop_sample >= 0.5:
                        action = ACTION_STOP_IDX
                        logp = torch.log(stop_prob + 1e-8).item()
                    else:
                        dist = Categorical(logits=movement_logits)
                        action = dist.sample().item()
                        logp_move = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                        logp = torch.log1p(-stop_prob + 1e-8).item() + logp_move
                    val = value.item()

                next_obs, r, done, info = env.step(action)

                ep_traj["obs"]['actor'].append(obs_dict['actor'])
                ep_traj["obs"]['critic_gt'].append(obs_dict['critic_gt'])
                ep_traj["ahist"].append(A[0])
                ep_traj["act"].append(action)
                ep_traj["logp"].append(logp)
                ep_traj["val"].append(val)
                ep_traj["rew"].append(r)
                # Label as stop whenever the agent IS near the end, regardless of whether it stopped
                # This provides necessary supervision for the stop head to learn
                stop_label = 1 if info.get('reached_end') else 0
                ep_traj["stop_label"].append(stop_label)
                
                a_onehot = np.zeros(N_MOVEMENT_ACTIONS)
                if action != ACTION_STOP_IDX and action < N_MOVEMENT_ACTIONS:
                    a_onehot[action] = 1.0
                ahist.append(a_onehot)
                obs_dict = next_obs
            
            # Reset to current stage if we used a previous stage
            if use_previous_stage:
                env.set_stage(stage['config'])

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
                    "adv": adv, "ret": ret,
                    "stop_label": np.array(ep_traj["stop_label"])
                }
                batch_buffer.append(final_ep_data)
                ep_return = sum(rews)
                ep_returns.append(ep_return)
                
                reached_end_val = 1 if info.get('reached_end') else 0
                stopped_correctly_val = 1 if info.get('stopped_correctly') else 0
                ep_reached_end.append(reached_end_val)
                ep_stopped_correctly.append(stopped_correctly_val)

                # Save image, mask, and path for the last 5 episodes of this stage
                if ep >= stage['episodes'] - 4:
                    sample_dir = os.path.join(run_dir, "episode_samples", stage['name'])
                    os.makedirs(sample_dir, exist_ok=True)
                    ep_tag = f"ep_{ep:05d}"
                    img_u8 = np.clip(env.ep.img * 255.0, 0, 255).astype(np.uint8)
                    mask_u8 = np.clip(env.ep.mask * 255.0, 0, 255).astype(np.uint8)
                    path_u8 = np.clip(env.path_mask * 255.0, 0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(sample_dir, f"{ep_tag}_img.png"), img_u8)
                    cv2.imwrite(os.path.join(sample_dir, f"{ep_tag}_mask.png"), mask_u8)
                    cv2.imwrite(os.path.join(sample_dir, f"{ep_tag}_path.png"), path_u8)
                    np.save(os.path.join(sample_dir, f"{ep_tag}_path_points.npy"),
                            np.array(env.path_points, dtype=np.float32))
                
                # Update metrics
                stage_metrics["rewards"].append(float(ep_return))
                stage_metrics["reached_end"].append(reached_end_val)
                stage_metrics["stopped_correctly"].append(stopped_correctly_val)

            if len(batch_buffer) >= 32:
                update_ppo(opt, model, batch_buffer)
                batch_buffer = []

            if ep % 100 == 0:
                avg_r = np.mean(ep_returns[-100:])
                reached_rate = np.mean(ep_reached_end[-100:]) if ep_reached_end else 0.0
                stopped_rate = np.mean(ep_stopped_correctly[-100:]) if ep_stopped_correctly else 0.0
                
                prev_stage_info = f" [Prev: {sum(1 for _ in previous_stages)}]" if previous_stages else ""
                print(f"[{stage['name']}] Ep {ep} | Avg Rew: {avg_r:.2f} | Reached End: {reached_rate:.2f} | Stopped Correctly: {stopped_rate:.2f}{prev_stage_info}")
                
                # Save metrics
                stage_metrics["episodes"].append(ep)
                stage_metrics["avg_rewards"].append(float(avg_r))
                stage_metrics["reached_end_rates"].append(float(reached_rate))
                stage_metrics["stop_success_rates"].append(float(stopped_rate))
                
                # Anti-forgetting: Evaluate on previous stages periodically
                if previous_stages and ep % eval_previous_every == 0:
                    print(f"\n🔍 Evaluating on {len(previous_stages)} previous stage(s) to check for forgetting...")
                    eval_results = []
                    for prev_stage_config, prev_stage_name in previous_stages:
                        eval_result = evaluate_on_stage(model, prev_stage_config, prev_stage_name, 
                                                       num_episodes=10, base_seed=base_seed)
                        eval_results.append(eval_result)
                        print(f"   {prev_stage_name}: Reached End={eval_result['reached_end_rate']:.2f}, "
                              f"Stopped Correctly={eval_result['stop_success_rate']:.2f}, Return={eval_result['avg_return']:.2f}")
                    
                    # Store evaluation results in metrics
                    if "previous_stage_evaluations" not in stage_metrics:
                        stage_metrics["previous_stage_evaluations"] = []
                    stage_metrics["previous_stage_evaluations"].append({
                        "episode": ep,
                        "evaluations": eval_results
                    })
                    print()  # Empty line for readability
                
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
                
                # Also save actor/stop/shared weights for inference (exclude critic)
                actor_weights = {k: v for k, v in model.state_dict().items() if not k.startswith('critic_')}
                actor_path = os.path.join(checkpoint_dir, f"actor_{stage['name']}_ep{ep}.pth")
                torch.save(actor_weights, actor_path)
                print(f"Actor-only weights saved: {actor_path}")

        final_path = os.path.join(weights_dir, f"model_{stage['name']}_FINAL.pth")
        torch.save(model.state_dict(), final_path)
        print(f"Finished {stage['name']}. Saved model to {final_path}")
        
        # Also save actor/stop/shared weights for inference (exclude critic)
        actor_weights = {k: v for k, v in model.state_dict().items() if not k.startswith('critic_')}
        actor_final_path = os.path.join(weights_dir, f"actor_{stage['name']}_FINAL.pth")
        torch.save(actor_weights, actor_final_path)
        
        # Anti-forgetting: Add completed stage to previous_stages for mixed curriculum training
        previous_stages.append((stage['config'].copy(), stage['name']))
        print(f"✅ Added {stage['name']} to previous stages pool (total: {len(previous_stages)} stages)")
        
        print(f"Actor-only weights saved: {actor_final_path}")
        
        # Add final metrics
        if ep_returns:
            stage_metrics["final_avg_reward"] = float(np.mean(ep_returns))
            stage_metrics["final_reached_end_rate"] = float(np.mean(ep_reached_end)) if ep_reached_end else 0.0
            stage_metrics["final_stop_success_rate"] = float(np.mean(ep_stopped_correctly)) if ep_stopped_correctly else 0.0
            stage_metrics["total_episodes"] = len(ep_returns)
        
        # Update or append stage metrics
        if stage_idx < len(all_metrics["stages"]):
            all_metrics["stages"][stage_idx] = stage_metrics
        else:
            all_metrics["stages"].append(stage_metrics)
        
        # Update global episode offset for next stage
        global_episode_offset += stage['episodes']
    
    # Calculate total training time
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time
    total_seconds = training_duration.total_seconds()
    
    # Add training time to config and metrics
    training_config["training_start_time"] = training_start_time.isoformat()
    training_config["training_end_time"] = training_end_time.isoformat()
    training_config["training_duration_seconds"] = total_seconds
    
    all_metrics["training_start_time"] = training_start_time.isoformat()
    all_metrics["training_end_time"] = training_end_time.isoformat()
    all_metrics["training_duration_seconds"] = total_seconds
    
    # Save configuration and metrics
    with open(config_file, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    metrics_file = os.path.join(run_dir, "metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Calculate total training time
    training_end_time = datetime.now()
    training_duration = training_end_time - training_start_time
    total_seconds = training_duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    # Log training time to file before restoring stdout
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Training Start Time: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training End Time: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s ({total_seconds:.1f} seconds)")
    print(f"All results saved to: {run_dir}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Final weights: {weights_dir}")
    print(f"  - Training log: {log_file}")
    print(f"  - Configuration: {config_file}")
    print(f"  - Metrics: {metrics_file}")
    
    # Delete checkpoints after successful completion
    if os.path.exists(checkpoint_dir):
        print(f"🧹 Cleaning up checkpoints directory: {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)
        
    # Restore stdout and close log file
    sys.stdout = original_stdout
    log_fp.close()
    
    # Also print to console
    print("\n=== TRAINING COMPLETE ===")
    print(f"Training Start Time: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training End Time: {training_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Training Time: {hours}h {minutes}m {seconds}s ({total_seconds:.1f} seconds)")
    print(f"All results saved to: {run_dir}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"  - Final weights: {weights_dir}")
    print(f"  - Training log: {log_file}")
    print(f"  - Configuration: {config_file}")
    print(f"  - Metrics: {metrics_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DSA RL agent with on-the-fly curve generation")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run directory (default: runs/TIMESTAMP or runs/EXPERIMENT_NAME_TIMESTAMP). All results (checkpoints, weights, logs) will be saved here.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Experiment name. Creates runs/EXPERIMENT_NAME_TIMESTAMP/ directory. Ignored if --run_dir is specified.")
    parser.add_argument("--base_seed", type=int, default=BASE_SEED,
                        help=f"Base seed for reproducibility (default: {BASE_SEED})")
    parser.add_argument("--curve_config", type=str, default=None,
                        help="Path to curve configuration JSON file (default: curve_config.json)")
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
    
    run_unified_training(args.run_dir, args.base_seed, args.clean_previous, 
                        args.experiment_name, args.resume_from, args.curve_config)
