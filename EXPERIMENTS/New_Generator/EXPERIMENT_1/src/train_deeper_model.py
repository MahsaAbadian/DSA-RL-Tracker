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
import shutil
import json
from datetime import datetime

# Add paths for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_curve_gen_path = os.path.join(_project_root, "CurveGeneratorModule")
if _curve_gen_path not in sys.path:
    sys.path.insert(0, _curve_gen_path)

from generator import CurveMaker

# ---------- GLOBALS ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  GPU not available, using CPU")

ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
ACTION_STOP_IDX = 8
N_ACTIONS = 9

STEP_ALPHA = 1.0
CROP = 33
EPSILON = 1e-6
BASE_SEED = 50

# ---------- CONFIG LOADING ----------
def load_curve_config(config_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    config_dir = os.path.join(parent_dir, "config")
    
    if config_path is None:
        default_paths = [
            os.path.join(config_dir, "curve_config.json"),
            os.path.join(parent_dir, "curve_config.json")
        ]
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            config_path = default_paths[0]
            
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded curve configuration from: {config_path}")
        return config, config_path
    else:
        return {}, None

# ---------- CURVE GENERATION ----------
def _rng(seed=None):
    return np.random.default_rng(seed)

def get_distance_to_poly(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return np.sqrt(np.min(d2))

def nearest_gt_index(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return int(np.argmin(d2))

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

def clamp(v, lo, hi): return max(lo, min(v, hi))

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray

# ---------- ENVIRONMENT ----------
class CurveEnvUnified:
    def __init__(self, h=128, w=128, max_steps=200, base_seed=BASE_SEED, stage_id=1, curve_config=None):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.base_seed = base_seed
        self.current_episode = 0
        self.curve_config = curve_config or {}
        
        # FIX: Provide comprehensive defaults so reset() doesn't fail on missing keys
        self.stage_config = {
            'stage_id': stage_id,
            'width': (2, 4),
            'noise_prob': 0.0,
            'tissue_noise_prob': 0.0,
            'invert_prob': 0.0,
            'curvature_factor': 0.5,
            'branches': False,
            'strict_stop': False,
            'mixed_start': False,
            'min_intensity': 0.5,
            'max_intensity': 1.0,
            'background_intensity': 0.0,
            'topology': 'random',
            'num_control_points': 4,
            'num_segments': 1,
            'centerline_mask': True,
            'width_variation': 'none',
            'intensity_variation': 'none',
            'allow_self_cross': False,
            'self_cross_prob': 0.0
        }
        self.reset()

    def set_stage(self, config):
        """
        Updates the stage configuration.
        Updates ALL keys so that _range parameters are properly passed.
        """
        self.stage_config.update(config)
        
        stages_cfg = self.curve_config.get('training_stages', [])
        stage_id = self.stage_config['stage_id']

        stage_curve_cfg = None
        for stage in stages_cfg:
            if stage.get('stage_id') == stage_id:
                stage_curve_cfg = stage.get('curve_generation', {})
                break
        
        if stage_curve_cfg:
            # Handle the one key that needs translation
            if 'width_range' in stage_curve_cfg: 
                self.stage_config['width'] = tuple(stage_curve_cfg['width_range'])
            
            # Update EVERYTHING else directly.
            self.stage_config.update(stage_curve_cfg)

    def reset(self, episode_number=None):
        if episode_number is not None:
            self.current_episode = episode_number
        
        episode_seed = self.base_seed + self.current_episode
        curve_maker = CurveMaker(h=self.h, w=self.w, seed=episode_seed, config=self.curve_config)
        
        # Sample ranges
        curv = self._sample_range('curvature_range', 'curvature_factor')
        noise_p = self._sample_range('noise_range', 'noise_prob')
        bg_i = self._sample_range('background_intensity_range', 'background_intensity')
        min_i = self._sample_range('min_intensity_range', 'min_intensity')
        max_i = self._sample_range('max_intensity_range', 'max_intensity')
        
        if max_i is not None and max_i < min_i: max_i = min_i + 0.1

        img, mask, pts_all = curve_maker.sample_curve(
            width_range=self.stage_config['width'],
            noise_prob=noise_p,
            tissue_noise_prob=self.stage_config.get('tissue_noise_prob', 0.0),
            invert_prob=self.stage_config.get('invert_prob', 0.5),
            min_intensity=min_i,
            max_intensity=max_i,
            background_intensity=bg_i,
            branches=self.stage_config['branches'],
            curvature_factor=curv,
            num_control_points=self.stage_config.get('num_control_points'),
            num_segments=self.stage_config.get('num_segments'),
            topology=self.stage_config.get('topology', 'random'),
            centerline_mask=self.stage_config.get('centerline_mask', True),
            width_variation=self.stage_config.get('width_variation', 'none'),
            intensity_variation=self.stage_config.get('intensity_variation', 'none'),
            allow_self_cross=self.stage_config.get('allow_self_cross', False),
            self_cross_prob=self.stage_config.get('self_cross_prob', 0.0)
        )
        
        gt_poly = pts_all[0].astype(np.float32)
        self.gt_map = np.zeros_like(img)
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0<=r<self.h and 0<=c<self.w:
                self.gt_map[r,c] = 1.0
        
        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly)

        # Start pos
        np.random.seed(episode_seed + 20000)
        use_cold_start = (np.random.rand() < 0.5) if self.stage_config['mixed_start'] else False
        np.random.seed() 

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
        self.is_at_start = (self.prev_idx == 0)
        self.initial_steps = 0
        self.current_episode += 1
        
        return self.obs()

    def _sample_range(self, range_key, val_key):
        if self.stage_config.get(range_key) is not None:
            r = self.stage_config[range_key]
            return np.random.uniform(r[0], r[1])
        return self.stage_config.get(val_key)

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
        dist_to_end = np.sqrt((self.agent[0]-self.ep.gt_poly[-1][0])**2 + (self.agent[1]-self.ep.gt_poly[-1][1])**2)

        if a_idx == ACTION_STOP_IDX:
            if dist_to_end < 5.0:
                bonus = 30.0 if self.stage_config['strict_stop'] else 20.0
                return self.obs(), bonus, True, {"reached_end": True, "stopped_correctly": True}
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
        
        if self.is_at_start:
            self.initial_steps += 1
            if self.initial_steps >= 5 or progress_delta > 0:
                self.is_at_start = False
        
        sigma = 1.5 if self.stage_config['stage_id'] == 1 else 1.0
        precision_score = np.exp(-(L_t**2) / (2 * sigma**2))
        
        if L_t < self.L_prev:
            r = np.log(EPSILON + dist_diff)
        else:
            r = -np.log(EPSILON + dist_diff)
        r = float(np.clip(r, -2.0, 2.0))
        
        # Direction Alignment Reward
        if self.is_at_start and self.initial_steps <= 5:
            start_pt = self.ep.gt_poly[0]
            lookahead_idx = min(3, len(self.ep.gt_poly) - 1)
            direction_pt = self.ep.gt_poly[lookahead_idx]
            curve_dir = np.array([direction_pt[0] - start_pt[0], direction_pt[1] - start_pt[1]])
            norm = np.linalg.norm(curve_dir) + 1e-8
            curve_dir /= norm
            
            # FIX: Explicit float dtype to avoid casting error
            act_dir = np.array([dy, dx], dtype=np.float32)
            act_dir /= (np.linalg.norm(act_dir) + 1e-8)
            
            alignment = np.dot(curve_dir, act_dir)
            r += alignment * (3.0 - self.initial_steps * 0.4) * 0.3

        if progress_delta > 0:
            r += precision_score * 2.0
        elif progress_delta <= 0:
            r -= 0.1
        
        if self.is_at_start and self.initial_steps <= 5:
            bm = max(1.0, 3.0 - self.initial_steps * 0.4)
            if progress_delta > 0: r += precision_score * bm * 1.5
            elif L_t < self.L_prev: r += abs(r) * bm * 0.5
            if progress_delta < 0: r = max(r, -0.5)

        # General Cosine Similarity Reward
        lookahead_idx = min(best_idx + 4, len(self.ep.gt_poly) - 1)
        gt_vec = self.ep.gt_poly[lookahead_idx] - self.ep.gt_poly[best_idx]
        act_vec = np.array([dy, dx])
        norm_gt = np.linalg.norm(gt_vec)
        norm_act = np.linalg.norm(act_vec)
        if norm_gt > 1e-6 and norm_act > 1e-6:
            cos_sim = np.dot(gt_vec, act_vec) / (norm_gt * norm_act)
            if cos_sim > 0: r += cos_sim * 0.5

        # Wiggly Penalty (Reduced for sharp turn stages)
        if self.prev_action != -1 and self.prev_action != a_idx:
            # If sharp turns are expected, penalize less
            if self.stage_config.get('topology') in ['zigzag', 'ribbon', 'hairpin']:
                r -= 0.02
            else:
                r -= 0.05
        
        self.prev_action = a_idx
        r -= 0.05 # Step penalty

        self.L_prev = L_t
        self.prev_idx = max(self.prev_idx, best_idx)

        done = False
        reached_end = (dist_to_end < 5.0)
        off_track_limit = 10.0 if self.stage_config['stage_id'] == 1 else 8.0
        if L_t > off_track_limit:
            r -= 5.0
            done = True
        
        if self.steps >= self.max_steps: done = True

        if not self.stage_config['strict_stop'] and reached_end:
            r += 20.0
            done = True
        elif reached_end:
            r += 5.0

        return self.obs(), r, done, {"reached_end": reached_end, "stopped_correctly": False}

# Import model
from models_deeper import AsymmetricActorCritic

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
            
            loss = p_loss + 0.5 * v_loss - 0.005 * entropy
            
            ppo_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            ppo_opt.step()

# ---------- TRAINING LOOP ----------
def run_unified_training(run_dir, base_seed=BASE_SEED, clean_previous=False, experiment_name=None, resume_from=None, curve_config_path=None):
    curve_config, _ = load_curve_config(curve_config_path)
    img_h = curve_config.get('image', {}).get('height', 128)
    img_w = curve_config.get('image', {}).get('width', 128)
    
    # [Directory setup]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    runs_base = os.path.join(parent_dir, "runs")
    
    # Handle clean_previous
    if clean_previous and os.path.exists(runs_base):
        print(f"⚠️  Cleaning previous runs in {runs_base}...")
        try:
            shutil.rmtree(runs_base)
        except Exception as e:
            print(f"Error cleaning: {e}")

    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(runs_base, f"{experiment_name or 'Exp'}_{timestamp}")
    
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
    
    model = AsymmetricActorCritic(n_actions=N_ACTIONS).to(DEVICE)
    K = 16
    
    # Resume Logic
    if resume_from:
        if os.path.exists(resume_from):
            print(f"🔄 Resuming from {resume_from}")
            model.load_state_dict(torch.load(resume_from, map_location=DEVICE))
        else:
            print(f"❌ Resume path not found: {resume_from}")
            return

    previous_stages = []
    global_episode_offset = 0

    stages = curve_config.get('training_stages', [])
    
    for stage in stages:
        print(f"\n=== STARTING {stage['name']} ===")
        print(f"Episodes: {stage['episodes']} | LR: {stage['learning_rate']}")
        
        env = CurveEnvUnified(h=img_h, w=img_w, base_seed=base_seed, stage_id=stage['stage_id'], curve_config=curve_config)
        env.set_stage(stage['curve_generation'])
        env.stage_config.update(stage['training']) 
        
        opt = torch.optim.Adam(model.parameters(), lr=stage['learning_rate'])
        
        # Anti-forgetting Config
        anti_forgetting_prob = stage.get('training', {}).get('anti_forgetting_prob', 0.15)
        print(f"🛡️  Anti-Forgetting Probability: {anti_forgetting_prob}")

        batch_buffer = []
        ep_returns = []
        ep_successes = [] # Added for success tracking
        
        for ep in range(1, stage['episodes'] + 1):
            global_ep = global_episode_offset + ep
            
            # Anti-forgetting logic
            use_prev = False
            if previous_stages and np.random.rand() < anti_forgetting_prob:
                prev_idx = np.random.randint(0, len(previous_stages))
                prev_cfg, prev_name = previous_stages[prev_idx]
                env.set_stage(prev_cfg)
                # Ensure training params (strict_stop etc) are also applied
                env.stage_config.update(prev_cfg) 
                use_prev = True
            else:
                env.set_stage(stage['curve_generation'])
                env.stage_config.update(stage['training'])

            obs_dict = env.reset(episode_number=global_ep)
            done = False
            ahist = []
            ep_traj = {'obs':{'actor':[], 'critic_gt':[]}, 'ahist':[], 'act':[], 'logp':[], 'val':[], 'rew':[]}
            
            while not done:
                obs_a = torch.tensor(obs_dict['actor'][None], dtype=torch.float32, device=DEVICE)
                obs_c = torch.tensor(obs_dict['critic_gt'][None], dtype=torch.float32, device=DEVICE)
                A = fixed_window_history(ahist, K, N_ACTIONS)[None, ...]
                A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    logits, value, _, _ = model(obs_a, obs_c, A_t)
                    dist = Categorical(logits=torch.clamp(logits, -20, 20))
                    action = dist.sample().item()
                    logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                    val = value.item()

                next_obs, r, done, info = env.step(action)
                
                ep_traj['obs']['actor'].append(obs_dict['actor'])
                ep_traj['obs']['critic_gt'].append(obs_dict['critic_gt'])
                ep_traj['ahist'].append(A[0])
                ep_traj['act'].append(action)
                ep_traj['logp'].append(logp)
                ep_traj['val'].append(val)
                ep_traj['rew'].append(r)
                
                a_oh = np.zeros(N_ACTIONS); a_oh[action] = 1.0
                ahist.append(a_oh)
                obs_dict = next_obs
            
            # Calculate Success
            if env.stage_config['strict_stop']:
                succ = 1.0 if info.get('stopped_correctly') else 0.0
            else:
                succ = 1.0 if info.get('reached_end') else 0.0
            ep_successes.append(succ)

            # GAE Calculation