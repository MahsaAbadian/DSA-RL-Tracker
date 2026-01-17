#!/usr/bin/env python3
"""
Interactive Step-by-Step Debugger.
Updated: Press 'I' to toggle Invert Background (Dark vs Light).
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys

# Add parent directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from src.models_deeper import ActorOnly
from src.train_deeper_model import CurveMakerFlexible, load_curve_config

# --- CONSTANTS ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
ACTION_NAMES = ["U", "D", "L", "R", "UL", "UR", "DL", "DR"] 
N_ACTIONS = 8 
K = 16
CROP = 33

# --- HELPERS ---
def clamp(v, lo, hi): return max(lo, min(v, hi))

def crop32_inference(img, cy, cx, size=CROP):
    h, w = img.shape
    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    out = np.zeros((size, size), dtype=np.float32)
    
    # Smart padding: detect background color
    corners = [img[0,0], img[0,w-1], img[h-1,0], img[h-1,w-1]]
    bg_color = np.median(corners)
    # If background is bright (>0.5), pad with 1.0. If dark, pad with 0.0
    pad_val = 1.0 if bg_color > 0.5 else 0.0
    out.fill(pad_val)
        
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    oy0, ox0 = sy0 - y0, sx0 - x0
    sh, sw = sy1 - sy0, sx1 - sx0
    
    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]
    return out

def get_closest_action(dy, dx):
    best_idx = -1
    max_dot = -float("inf")
    mag = np.sqrt(dy**2 + dx**2) + 1e-6
    uy, ux = dy / mag, dx / mag
    for i, (ay, ax) in enumerate(ACTIONS_MOVEMENT):
        amag = np.sqrt(ay**2 + ax**2) + 1e-6
        ny, nx = ay / amag, ax / amag
        dot = uy * ny + ux * nx
        if dot > max_dot:
            max_dot = dot
            best_idx = i
    return best_idx

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

# --- INTERACTIVE SESSION ---
class InteractiveSession:
    def __init__(self, model, config, stage_id=1):
        self.model = model
        self.config = config
        self.stage_id = stage_id
        self.step_alpha = 1.0 
        
        # Toggle state for Inversion
        self.force_invert = False
        
        self.fig = plt.figure(figsize=(14, 8))
        gs = gridspec.GridSpec(2, 3, width_ratios=[3, 1, 1])
        
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_crop0 = self.fig.add_subplot(gs[0, 1])
        self.ax_crop1 = self.fig.add_subplot(gs[0, 2])
        self.ax_probs = self.fig.add_subplot(gs[1, 1:])
        
        self.ax_main.set_title("Main View (Press 'N' or 'SPACE' to step)")
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.reset_env()
        self.update_plot()
        
        print("\n--- CONTROLS ---")
        print(" [SPACE] / [N] : Step Forward")
        print(" [R]           : Reset (New Curve)")
        print(" [I]           : Toggle Invert (White/Black BG)")
        print(" [Q]           : Quit")
        print("----------------")
        plt.show()

    def reset_env(self):
        h = self.config.get('image', {}).get('height', 128)
        w = self.config.get('image', {}).get('width', 128)
        
        stages = self.config.get('training_stages', [])
        target_stage = next((s for s in stages if s['stage_id'] == self.stage_id), stages[0])
        gen_cfg = target_stage.get('curve_generation', {})
        
        maker = CurveMakerFlexible(h=h, w=w, seed=np.random.randint(0, 100000), config=self.config)
        
        c_range = gen_cfg.get('curvature_range', [0.5, 0.5])
        curv = np.random.uniform(c_range[0], c_range[1])
        w_range = gen_cfg.get('width_range', [3, 3])
        
        # Logic: If user pressed 'I', force invert. Else use config.
        current_invert_prob = 1.0 if self.force_invert else gen_cfg.get('invert_prob', 0.0)

        img, _, pts_all = maker.sample_curve(
            width_range=tuple(w_range),
            curvature_factor=curv,
            allow_self_cross=gen_cfg.get('allow_self_cross', False),
            self_cross_prob=gen_cfg.get('self_cross_prob', 0.0),
            branches=False,
            noise_prob=gen_cfg.get('noise_prob', 0.0),
            
            # Appearance
            min_intensity=gen_cfg.get('min_intensity', 0.6),
            max_intensity=gen_cfg.get('max_intensity', 1.0),
            background_intensity=gen_cfg.get('background_intensity', 0.0),
            intensity_variation=gen_cfg.get('intensity_variation', 'none'),
            start_intensity=gen_cfg.get('start_intensity', None),
            end_intensity=gen_cfg.get('end_intensity', None),
            
            # INVERT
            invert_prob=current_invert_prob
        )
        
        self.img = img
        self.gt_poly = pts_all[0]
        
        start_pt = self.gt_poly[0]
        p_next = self.gt_poly[min(5, len(self.gt_poly)-1)]
        vec = p_next - start_pt
        
        self.agent = (float(start_pt[0]), float(start_pt[1]))
        
        mag = np.linalg.norm(vec) + 1e-6
        dy, dx = (vec[0]/mag), (vec[1]/mag)
        p_prev1 = (self.agent[0] - dy, self.agent[1] - dx)
        p_prev2 = (self.agent[0] - 2*dy, self.agent[1] - 2*dx)
        
        self.history_pos = [p_prev2, p_prev1, self.agent]
        self.path = [self.agent]
        self.path_mask = np.zeros_like(img)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        
        start_action = get_closest_action(dy, dx)
        self.ahist = [np.zeros(N_ACTIONS)] * K
        self.ahist[-1][start_action] = 1.0
        
        self.step_count = 0
        self.done = False
        self.last_probs = np.zeros(N_ACTIONS)
        self.last_action = start_action
        
        self.update_plot()

    def step(self):
        if self.done: return

        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        ch0 = crop32_inference(self.img, int(curr[0]), int(curr[1]))
        ch1 = crop32_inference(self.img, int(p1[0]), int(p1[1]))
        ch2 = crop32_inference(self.img, int(p2[0]), int(p2[1]))
        ch3 = crop32_inference(self.path_mask, int(curr[0]), int(curr[1]))
        
        obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
        
        ahist_arr = fixed_window_history(self.ahist, K, N_ACTIONS)
        hist_t = torch.tensor(ahist_arr[None], dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            logits, _ = self.model(obs_t, hist_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        self.last_probs = probs
        action = np.argmax(probs)
        self.last_action = action
        
        dy, dx = ACTIONS_MOVEMENT[action]
        ny = self.agent[0] + dy * self.step_alpha
        nx = self.agent[1] + dx * self.step_alpha
        
        h, w = self.img.shape
        if not (0 <= ny < h and 0 <= nx < w):
            print("Hit wall.")
            self.done = True
            return

        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0
        
        oh = np.zeros(N_ACTIONS)
        oh[action] = 1.0
        self.ahist.append(oh)
        
        self.step_count += 1
        self.update_plot()

    def update_plot(self):
        self.ax_main.clear()
        
        # Use vmin/vmax to force full dynamic range rendering
        self.ax_main.imshow(self.img, cmap='gray', vmin=0, vmax=1)
        
        gt_y = self.gt_poly[:, 0]
        gt_x = self.gt_poly[:, 1]
        self.ax_main.plot(gt_x, gt_y, 'g--', linewidth=1, alpha=0.5, label="GT")
        
        path = np.array(self.path)
        if len(path) > 0:
            self.ax_main.plot(path[:, 1], path[:, 0], 'c-', linewidth=2, label="Agent")
            self.ax_main.plot(path[-1, 1], path[-1, 0], 'ro', markersize=6)
        
        bg_status = "Inverted (White BG)" if self.force_invert else "Standard (Dark BG)"
        self.ax_main.set_title(f"Step: {self.step_count} | {bg_status}")
        self.ax_main.legend()

        curr = self.history_pos[-1]
        ch0 = crop32_inference(self.img, int(curr[0]), int(curr[1]))
        self.ax_crop0.clear()
        self.ax_crop0.imshow(ch0, cmap='gray', vmin=0, vmax=1)
        self.ax_crop0.set_title("Agent View")
        self.ax_crop0.axis('off')
        
        ch3 = crop32_inference(self.path_mask, int(curr[0]), int(curr[1]))
        self.ax_crop1.clear()
        self.ax_crop1.imshow(ch3, cmap='bone')
        self.ax_crop1.set_title("Memory Mask")
        self.ax_crop1.axis('off')
        
        self.ax_probs.clear()
        bars = self.ax_probs.bar(ACTION_NAMES, self.last_probs, color='skyblue')
        if self.last_action is not None:
            bars[self.last_action].set_color('orange')
        self.ax_probs.set_ylim(0, 1.0)
        self.ax_probs.set_title("Action Probabilities")
        
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key in [' ', 'n', 'N']: self.step()
        elif event.key in ['r', 'R']: self.reset_env()
        elif event.key in ['q', 'Q', 'escape']: plt.close()
        # TOGGLE INVERT
        elif event.key in ['i', 'I']: 
            self.force_invert = not self.force_invert
            print(f"Invert Mode: {self.force_invert}")
            self.reset_env()

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to .pth model weights")
    parser.add_argument("--config", type=str, default="config/curve_config.json")
    parser.add_argument("--stage_id", type=int, default=1)
    args = parser.parse_args()

    config, _ = load_curve_config(args.config)
    
    print(f"Initializing Model with N_ACTIONS={N_ACTIONS}...")
    model = ActorOnly(n_actions=N_ACTIONS, K=K).to(DEVICE)
    
    try:
        print(f"Loading weights from {args.weights}...")
        loaded = torch.load(args.weights, map_location=DEVICE)
        
        clean_weights = {}
        for k, v in loaded.items():
            if k.startswith('actor_'):
                clean_weights[k] = v
            elif 'critic' not in k and 'actor_' not in k:
                clean_weights[f"actor_{k}"] = v
        
        model.load_state_dict(clean_weights, strict=True)
        model.eval()
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR Loading Weights: {e}")
        return

    InteractiveSession(model, config, stage_id=args.stage_id)

if __name__ == "__main__":
    main()