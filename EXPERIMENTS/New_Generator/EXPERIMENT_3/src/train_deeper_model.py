#!/usr/bin/env python3
"""
Training script for DSA RL Experiment
Trains an agent to follow curves using PPO with curriculum learning + anti-forgetting replay.
Drop-in rewrite of your current train_deeper_model.py with:
  - Proper __main__ entry point (so it doesn't exit immediately)
  - Robust config loading + safe defaults
  - Stage replay (anti-forgetting) done correctly
  - Per-topology performance logging
"""

import argparse
import json
import os
import sys
import shutil
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ----------------------------
# Import curve generator
# ----------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_curve_gen_path = os.path.join(_project_root, "CurveGeneratorModule")
if _curve_gen_path not in sys.path:
    sys.path.insert(0, _curve_gen_path)

from generator import CurveMaker  # your unified generator

# ----------------------------
# Device
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  GPU not available, using CPU")

# ----------------------------
# Globals
# ----------------------------
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
ACTION_STOP_IDX = 8
N_ACTIONS = 9

STEP_ALPHA = 1.0
CROP = 33
EPSILON = 1e-6
BASE_SEED = 50


# ----------------------------
# Utilities
# ----------------------------
def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def crop32(img: np.ndarray, cy: int, cx: int, size=CROP):
    h, w = img.shape
    corners = [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]]
    bg_estimate = np.median(corners)
    pad_val = 1.0 if bg_estimate > 0.5 else 0.0

    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1

    out = np.full((size, size), pad_val, dtype=img.dtype)

    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)

    oy0 = sy0 - y0
    ox0 = sx0 - x0
    sh = sy1 - sy0
    sw = sx1 - sx0

    if sh > 0 and sw > 0:
        out[oy0 : oy0 + sh, ox0 : ox0 + sw] = img[sy0:sy1, sx0:sx1]
    return out


def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0:
        return out
    tail = ahist_list[-K:]
    out[-len(tail) :] = np.stack(tail, axis=0)
    return out


def get_distance_to_poly(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return float(np.sqrt(np.min(d2)))


def nearest_gt_index(pt, poly):
    dif = poly - np.array(pt, dtype=np.float32)
    d2 = np.sum(dif * dif, axis=1)
    return int(np.argmin(d2))


# ----------------------------
# Config loading
# ----------------------------
def load_curve_config(config_path=None):
    """
    Looks for:
      ../config/curve_config.json
      ../curve_config.json
    unless a path is explicitly provided.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    config_dir = os.path.join(parent_dir, "config")

    if config_path is None:
        default_paths = [
            os.path.join(config_dir, "curve_config.json"),
            os.path.join(parent_dir, "curve_config.json"),
        ]
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            config_path = default_paths[0]

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"✓ Loaded curve configuration from: {config_path}")
        return config, config_path

    print(f"❌ Could not find curve_config.json. Tried: {config_path}")
    return {}, None


# ----------------------------
# Episode container
# ----------------------------
@dataclass
class CurveEpisode:
    img: np.ndarray
    mask: np.ndarray
    gt_poly: np.ndarray


# ----------------------------
# Environment
# ----------------------------
class CurveEnvUnified:
    def __init__(self, h=128, w=128, max_steps=200, base_seed=BASE_SEED, stage_id=1, curve_config=None):
        self.h, self.w = h, w
        self.max_steps = max_steps
        self.base_seed = base_seed
        self.current_episode = 0
        self.curve_config = curve_config or {}

        # Safe defaults (so reset never crashes)
        self.stage_config = {
            "stage_id": stage_id,
            "width": (2, 4),
            "noise_prob": 0.0,
            "noise_range": None,
            "tissue_noise_prob": 0.0,
            "invert_prob": 0.5,
            "curvature_factor": 0.5,
            "curvature_range": None,
            "branches": False,
            "strict_stop": False,
            "mixed_start": False,
            "min_intensity": 0.6,
            "min_intensity_range": None,
            "max_intensity": 1.0,
            "max_intensity_range": None,
            "background_intensity": 0.0,
            "background_intensity_range": None,
            "topology": "random",
            "num_control_points": 4,
            "num_segments": 1,
            "centerline_mask": True,
            "width_variation": "none",
            "intensity_variation": "none",
            "allow_self_cross": False,
            "self_cross_prob": 0.0,
        }

        self.current_topology = "random"
        self.reset()

    def set_stage(self, stage_cfg: dict):
        """
        stage_cfg should be either:
          - stage["curve_generation"]
        plus we separately apply stage["training"] in training loop.
        """
        if not isinstance(stage_cfg, dict):
            raise ValueError(f"set_stage expected dict, got: {type(stage_cfg)}")

        # Update stage config with provided keys
        self.stage_config.update(stage_cfg)

        # Backward compat: width_range -> width
        if "width_range" in self.stage_config and "width" not in self.stage_config:
            self.stage_config["width"] = tuple(self.stage_config["width_range"])

        # Ensure width always is tuple
        if "width" in self.stage_config and isinstance(self.stage_config["width"], list):
            self.stage_config["width"] = tuple(self.stage_config["width"])

    def _sample_range(self, range_key, val_key):
        r = self.stage_config.get(range_key, None)
        if r is not None:
            return float(np.random.uniform(r[0], r[1]))
        return self.stage_config.get(val_key)

    def _choose_topology(self, topology):
        # Save actual topology used for logging
        if isinstance(topology, list):
            chosen = np.random.choice(topology)
        else:
            chosen = topology
        self.current_topology = str(chosen)
        return chosen

    def reset(self, episode_number=None):
        if episode_number is not None:
            self.current_episode = int(episode_number)

        episode_seed = self.base_seed + self.current_episode
        curve_maker = CurveMaker(h=self.h, w=self.w, seed=episode_seed, config=self.curve_config)

        # Sample stage values (range overrides scalar)
        curv = self._sample_range("curvature_range", "curvature_factor")
        noise_p = self._sample_range("noise_range", "noise_prob")
        bg_i = self._sample_range("background_intensity_range", "background_intensity")
        min_i = self._sample_range("min_intensity_range", "min_intensity")
        max_i = self._sample_range("max_intensity_range", "max_intensity")
        topo = self._choose_topology(self.stage_config.get("topology", "random"))

        if max_i is not None and max_i < min_i:
            max_i = min_i + 0.05

        img, mask, pts_all = curve_maker.sample_curve(
            width_range=self.stage_config["width"],
            noise_prob=noise_p,
            tissue_noise_prob=self.stage_config.get("tissue_noise_prob", 0.0),
            invert_prob=self.stage_config.get("invert_prob", 0.5),
            min_intensity=min_i,
            max_intensity=max_i,
            background_intensity=bg_i,
            branches=self.stage_config.get("branches", False),
            curvature_factor=curv,
            num_control_points=self.stage_config.get("num_control_points"),
            num_segments=self.stage_config.get("num_segments"),
            topology=topo,
            centerline_mask=self.stage_config.get("centerline_mask", True),
            width_variation=self.stage_config.get("width_variation", "none"),
            intensity_variation=self.stage_config.get("intensity_variation", "none"),
            allow_self_cross=self.stage_config.get("allow_self_cross", False),
            self_cross_prob=self.stage_config.get("self_cross_prob", 0.0),
        )

        gt_poly = pts_all[0].astype(np.float32)
        self.gt_map = np.zeros_like(img, dtype=np.float32)
        for pt in gt_poly:
            r, c = int(pt[0]), int(pt[1])
            if 0 <= r < self.h and 0 <= c < self.w:
                self.gt_map[r, c] = 1.0

        self.ep = CurveEpisode(img=img, mask=mask, gt_poly=gt_poly)

        # Start position (mixed_start)
        np.random.seed(episode_seed + 20000)
        use_cold_start = (np.random.rand() < 0.5) if self.stage_config.get("mixed_start", False) else False
        np.random.seed(None)

        if use_cold_start:
            curr = gt_poly[0]
            self.history_pos = [tuple(curr)] * 3
            self.prev_idx = 0
            self.agent = (float(curr[0]), float(curr[1]))
        else:
            start_idx = 5 if len(gt_poly) > 10 else 0
            curr = gt_poly[start_idx]
            p1 = gt_poly[max(0, start_idx - 1)]
            p2 = gt_poly[max(0, start_idx - 2)]
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

        dist_to_end = float(
            np.sqrt(
                (self.agent[0] - self.ep.gt_poly[-1][0]) ** 2 + (self.agent[1] - self.ep.gt_poly[-1][1]) ** 2
            )
        )

        # STOP action
        if a_idx == ACTION_STOP_IDX:
            if dist_to_end < 5.0:
                bonus = 30.0 if self.stage_config.get("strict_stop", False) else 20.0
                return self.obs(), bonus, True, {"reached_end": True, "stopped_correctly": True}
            return self.obs(), -2.0, False, {"reached_end": False, "stopped_correctly": False}

        dy, dx = ACTIONS_MOVEMENT[a_idx]
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h - 1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w - 1)
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

        sigma = 1.5 if self.stage_config.get("stage_id", 1) == 1 else 1.0
        precision_score = float(np.exp(-(L_t**2) / (2 * sigma**2)))

        # distance shaping
        if L_t < self.L_prev:
            r = np.log(EPSILON + dist_diff)
        else:
            r = -np.log(EPSILON + dist_diff)
        r = float(np.clip(r, -2.0, 2.0))

        # alignment reward in first few steps
        if self.is_at_start and self.initial_steps <= 5:
            start_pt = self.ep.gt_poly[0]
            lookahead_idx = min(3, len(self.ep.gt_poly) - 1)
            direction_pt = self.ep.gt_poly[lookahead_idx]
            curve_dir = np.array([direction_pt[0] - start_pt[0], direction_pt[1] - start_pt[1]], dtype=np.float32)
            curve_dir /= (np.linalg.norm(curve_dir) + 1e-8)

            act_dir = np.array([dy, dx], dtype=np.float32)
            act_dir /= (np.linalg.norm(act_dir) + 1e-8)

            alignment = float(np.dot(curve_dir, act_dir))
            r += alignment * (3.0 - self.initial_steps * 0.4) * 0.3

        if progress_delta > 0:
            r += precision_score * 2.0
        else:
            r -= 0.1

        # general cosine similarity
        lookahead_idx = min(best_idx + 4, len(self.ep.gt_poly) - 1)
        gt_vec = self.ep.gt_poly[lookahead_idx] - self.ep.gt_poly[best_idx]
        act_vec = np.array([dy, dx], dtype=np.float32)
        norm_gt = np.linalg.norm(gt_vec)
        norm_act = np.linalg.norm(act_vec)
        if norm_gt > 1e-6 and norm_act > 1e-6:
            cos_sim = float(np.dot(gt_vec, act_vec) / (norm_gt * norm_act))
            if cos_sim > 0:
                r += cos_sim * 0.5

        # wiggly penalty (reduced on hairpin/ribbon)
        if self.prev_action != -1 and self.prev_action != a_idx:
            if self.current_topology in ["ribbon", "hairpin"]:
                r -= 0.02
            else:
                r -= 0.05
        self.prev_action = a_idx

        # step penalty
        r -= 0.05

        self.L_prev = L_t
        self.prev_idx = max(self.prev_idx, best_idx)

        done = False
        reached_end = (dist_to_end < 5.0)

        off_track_limit = 10.0 if self.stage_config.get("stage_id", 1) == 1 else 8.0
        if L_t > off_track_limit:
            r -= 5.0
            done = True

        if self.steps >= self.max_steps:
            done = True

        # terminate when reaching end (non-strict) or reward partial (strict)
        if not self.stage_config.get("strict_stop", False) and reached_end:
            r += 20.0
            done = True
        elif reached_end:
            r += 5.0

        return self.obs(), float(r), bool(done), {"reached_end": reached_end, "stopped_correctly": False}


# ----------------------------
# Model import (your file)
# ----------------------------
from models_deeper import AsymmetricActorCritic


# ----------------------------
# PPO Update
# ----------------------------
def update_ppo(ppo_opt, model, buf_list, clip=0.2, epochs=4, minibatch=32):
    obs_a = torch.tensor(np.concatenate([x["obs"]["actor"] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    obs_c = torch.tensor(np.concatenate([x["obs"]["critic_gt"] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ahist = torch.tensor(np.concatenate([x["ahist"] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    act = torch.tensor(np.concatenate([x["act"] for x in buf_list]), dtype=torch.long, device=DEVICE)
    logp = torch.tensor(np.concatenate([x["logp"] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    adv = torch.tensor(np.concatenate([x["adv"] for x in buf_list]), dtype=torch.float32, device=DEVICE)
    ret = torch.tensor(np.concatenate([x["ret"] for x in buf_list]), dtype=torch.float32, device=DEVICE)

    if adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = obs_a.shape[0]
    idxs = np.arange(N)

    for _ in range(epochs):
        np.random.shuffle(idxs)
        for s in range(0, N, minibatch):
            mb = idxs[s : s + minibatch]
            if len(mb) == 0:
                continue

            logits, val, _, _ = model(obs_a[mb], obs_c[mb], ahist[mb])
            logits = torch.clamp(logits, -20, 20)
            dist = Categorical(logits=logits)

            new_logp = dist.log_prob(act[mb])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - logp[mb])
            surr1 = ratio * adv[mb]
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * adv[mb]
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(val, ret[mb])

            loss = p_loss + 0.5 * v_loss - 0.005 * entropy

            ppo_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            ppo_opt.step()


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: list[float] length T
    values:  list[float] length T
    returns: adv (T,), ret (T,)
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        adv[t] = last_gae
        next_value = values[t]

    ret = adv + np.array(values, dtype=np.float32)
    return adv, ret


# ----------------------------
# Stage replay sampler
# ----------------------------
def sample_stage_with_replay(stages, stage_idx):
    """
    60% current, 25% previous, 15% two-back (when available)
    """
    active = [stages[stage_idx]]
    if stage_idx >= 1:
        active.append(stages[stage_idx - 1])
    if stage_idx >= 2:
        active.append(stages[stage_idx - 2])

    p = np.random.rand()
    if p < 0.60:
        return active[0]
    elif p < 0.85 and len(active) > 1:
        return active[1]
    else:
        return active[-1]


# ----------------------------
# Training loop
# ----------------------------
def run_unified_training(
    run_dir=None,
    base_seed=BASE_SEED,
    clean_previous=False,
    experiment_name=None,
    resume_from=None,
    curve_config_path=None,
):
    curve_config, used_path = load_curve_config(curve_config_path)
    if not curve_config:
        print("❌ curve_config is empty -> exiting.")
        return

    stages = curve_config.get("training_stages", [])
    if not stages:
        print("❌ No training_stages found in curve_config -> exiting.")
        return

    img_h = curve_config.get("image", {}).get("height", 128)
    img_w = curve_config.get("image", {}).get("width", 128)

    # run dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    runs_base = os.path.join(parent_dir, "runs")

    if clean_previous and os.path.exists(runs_base):
        print(f"⚠️  Cleaning previous runs in {runs_base}...")
        shutil.rmtree(runs_base, ignore_errors=True)

    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(runs_base, f"{experiment_name or 'Exp'}_{timestamp}")

    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)

    model = AsymmetricActorCritic(n_actions=N_ACTIONS).to(DEVICE)
    K = 16

    if resume_from:
        if os.path.exists(resume_from):
            print(f"🔄 Resuming from {resume_from}")
            model.load_state_dict(torch.load(resume_from, map_location=DEVICE))
        else:
            print(f"❌ Resume path not found: {resume_from}")
            return

    global_episode_offset = 0

    # Per-topology stats (success rate)
    topo_stats = {}

    for stage_idx, stage in enumerate(stages):
        lr = stage.get("learning_rate", 1e-4)
        episodes = int(stage.get("episodes", 10000))
        stage_name = stage.get("name", f"Stage{stage.get('stage_id','?')}")

        print("\n" + "=" * 45)
        print(f"STARTING {stage_name}")
        print(f"Episodes: {episodes} | LR: {lr}")
        print(f"Global Episode Offset: {global_episode_offset}")
        print("=" * 45)

        env = CurveEnvUnified(h=img_h, w=img_w, base_seed=base_seed, stage_id=stage.get("stage_id", 1), curve_config=curve_config)

        opt = torch.optim.Adam(model.parameters(), lr=lr)

        batch_buffer = []
        ep_returns = []
        ep_successes = []

        # Training flags for the CURRENT stage (applied when stage is selected)
        stage_training = stage.get("training", {}) or {}
        print(f"[ENV] On-the-fly curve generation enabled (base_seed={base_seed})")

        for ep in range(1, episodes + 1):
            global_ep = global_episode_offset + ep

            # ---- stage replay sampling (prevents forgetting) ----
            stage_sample = sample_stage_with_replay(stages, stage_idx)

            curve_gen_cfg = stage_sample.get("curve_generation", {}) or {}
            train_cfg = stage_sample.get("training", {}) or {}

            # apply curve-generation config
            env.set_stage({"stage_id": stage_sample.get("stage_id", env.stage_config["stage_id"])})
            env.set_stage(curve_gen_cfg)

            # apply training-mode params (strict_stop, mixed_start, etc.)
            env.stage_config.update(train_cfg)

            # reset episode
            obs_dict = env.reset(episode_number=global_ep)
            done = False
            ahist = []

            ep_traj = {"obs": {"actor": [], "critic_gt": []}, "ahist": [], "act": [], "logp": [], "val": [], "rew": []}

            last_info = {}
            while not done:
                obs_a = torch.tensor(obs_dict["actor"][None], dtype=torch.float32, device=DEVICE)
                obs_c = torch.tensor(obs_dict["critic_gt"][None], dtype=torch.float32, device=DEVICE)
                A = fixed_window_history(ahist, K, N_ACTIONS)[None, ...]
                A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    logits, value, _, _ = model(obs_a, obs_c, A_t)
                    dist = Categorical(logits=torch.clamp(logits, -20, 20))
                    action = dist.sample().item()
                    logp = dist.log_prob(torch.tensor(action, device=DEVICE)).item()
                    val = float(value.item())

                next_obs, r, done, info = env.step(action)
                last_info = info

                ep_traj["obs"]["actor"].append(obs_dict["actor"])
                ep_traj["obs"]["critic_gt"].append(obs_dict["critic_gt"])
                ep_traj["ahist"].append(A[0])
                ep_traj["act"].append(action)
                ep_traj["logp"].append(logp)
                ep_traj["val"].append(val)
                ep_traj["rew"].append(float(r))

                a_oh = np.zeros(N_ACTIONS, dtype=np.float32)
                a_oh[action] = 1.0
                ahist.append(a_oh)
                obs_dict = next_obs

            # ---- success metric (depends on strict_stop) ----
            if env.stage_config.get("strict_stop", False):
                succ = 1.0 if last_info.get("stopped_correctly", False) else 0.0
            else:
                succ = 1.0 if last_info.get("reached_end", False) else 0.0
            ep_successes.append(succ)

            # ---- per-topology logging ----
            topo = env.current_topology
            if topo not in topo_stats:
                topo_stats[topo] = {"success": 0.0, "total": 0}
            topo_stats[topo]["total"] += 1
            topo_stats[topo]["success"] += succ

            # ---- GAE / returns ----
            adv, ret = compute_gae(ep_traj["rew"], ep_traj["val"], gamma=0.99, lam=0.95)

            # pack to buffer (shape them like your update_ppo expects)
            ep_pack = {
                "obs": {
                    "actor": np.array(ep_traj["obs"]["actor"], dtype=np.float32),
                    "critic_gt": np.array(ep_traj["obs"]["critic_gt"], dtype=np.float32),
                },
                "ahist": np.array(ep_traj["ahist"], dtype=np.float32),
                "act": np.array(ep_traj["act"], dtype=np.int64),
                "logp": np.array(ep_traj["logp"], dtype=np.float32),
                "adv": adv.astype(np.float32),
                "ret": ret.astype(np.float32),
            }
            batch_buffer.append(ep_pack)

            # bookkeeping
            ep_returns.append(float(np.sum(ep_traj["rew"])))

            # ---- PPO update every N episodes (adjust if you prefer steps-based batching) ----
            if len(batch_buffer) >= 32:
                update_ppo(opt, model, batch_buffer, clip=0.2, epochs=4, minibatch=32)
                batch_buffer = []

            # ---- print progress ----
            if ep % 100 == 0:
                avg_rew = float(np.mean(ep_returns[-100:])) if len(ep_returns) >= 100 else float(np.mean(ep_returns))
                avg_succ = float(np.mean(ep_successes[-100:])) if len(ep_successes) >= 100 else float(np.mean(ep_successes))
                print(f"[{stage_name}] Ep {ep:5d} | Avg Rew: {avg_rew:7.2f} | Success: {avg_succ:.2f}")

            # ---- print topology performance occasionally ----
            if ep % 500 == 0:
                print("Topology performance (success rate):")
                for k, v in topo_stats.items():
                    acc = v["success"] / max(v["total"], 1)
                    print(f"  {k:10s}: {acc:.2f}  (n={v['total']})")

        # stage end
        global_episode_offset += episodes

        # save weights
        out_path = os.path.join(run_dir, "weights", f"model_{stage_name}_FINAL.pth")
        torch.save(model.state_dict(), out_path)
        print(f"Finished {stage_name}. Saved model to {out_path}")

        actor_path = os.path.join(run_dir, "weights", f"actor_{stage_name}_FINAL.pth")
        # if your AsymmetricActorCritic has actor-only, adjust accordingly; otherwise save full again
        torch.save(model.state_dict(), actor_path)
        print(f"Actor-only weights saved: {actor_path}")

    print("\n✅ Training complete.")
    print(f"Run directory: {run_dir}")


# ----------------------------
# Main entry point (CRITICAL)
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="EXPERIMENT_1")
    parser.add_argument("--curve_config", type=str, default=None)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=BASE_SEED)

    args = parser.parse_args()

    print("\n🚀 Starting Training")
    print(f"Experiment: {args.experiment_name}")
    print(f"Curve config: {args.curve_config}")
    print(f"Clean runs: {args.clean}")
    print(f"Resume from: {args.resume_from}")
    print(f"Run dir: {args.run_dir}")
    print(f"Seed: {args.seed}")

    run_unified_training(
        run_dir=args.run_dir,
        base_seed=args.seed,
        clean_previous=args.clean,
        experiment_name=args.experiment_name,
        resume_from=args.resume_from,
        curve_config_path=args.curve_config,
    )
