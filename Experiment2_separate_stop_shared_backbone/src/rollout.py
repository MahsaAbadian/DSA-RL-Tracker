#!/usr/bin/env python3
"""
Inference script for DSA RL Experiment.
Uses ActorOnlyWithStop model (no critic) for efficient inference.
"""
import argparse
import os
import sys
import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Add parent directory to path so 'src' package can be imported
# This allows imports to work both when running as script and when imported as module
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Use absolute import - works both as module and script
from src.models import ActorOnlyWithStop

# Import constants and utilities
# (These are also defined in train.py, but we define them here for standalone use)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
N_MOVEMENT_ACTIONS = len(ACTIONS_MOVEMENT)
ACTION_STOP_IDX = 8
CROP = 33

def clamp(v, lo, hi): 
    return max(lo, min(v, hi))

def fixed_window_history(ahist_list, K, n_actions):
    """Create fixed-size window of action history."""
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: 
        return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

# ---------- UTILITIES ----------
def preprocess_full_image(path):
    """Load and preprocess DSA image for inference."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img = clahe.apply(img)
    
    # Normalize
    img = img.astype(np.float32) / 255.0
    
    # Auto-invert if light background
    if np.median(img) > 0.5:
        print("Detected light background. Inverting image...")
        img = 1.0 - img
    
    return img

def crop32_inference(img: np.ndarray, cy: int, cx: int, size=CROP):
    """Crop 33x33 patch from full image with smart padding."""
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
    
    oy0 = sy0 - y0
    ox0 = sx0 - x0
    sh = sy1 - sy0
    sw = sx1 - sx0
    
    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]
    
    return out

def get_closest_action(dy, dx):
    """Match click vector to discrete movement action (0-7)."""
    best_idx = -1
    max_dot = -float('inf')
    mag = np.sqrt(dy**2 + dx**2) + 1e-6
    uy, ux = dy/mag, dx/mag
    
    for i, (ay, ax) in enumerate(ACTIONS_MOVEMENT):
        amag = np.sqrt(ay**2 + ax**2)
        ny, nx = ay/amag, ax/amag
        dot = uy*ny + ux*nx
        if dot > max_dot:
            max_dot = dot
            best_idx = i
    return best_idx

# ---------- INFERENCE ENVIRONMENT ----------
class InferenceEnv:
    """Environment for inference on full-size DSA images."""
    def __init__(self, full_img, start_pt, start_vector, max_steps=1000):
        self.img = full_img
        self.h, self.w = full_img.shape
        self.max_steps = max_steps
        
        # Initialize agent position
        self.agent = tuple(start_pt)  # (y, x)
        
        # Setup momentum from start vector
        dy, dx = start_vector
        mag = np.sqrt(dy**2 + dx**2) + 1e-6
        dy, dx = (dy/mag) * 2.0, (dx/mag) * 2.0
        
        # History for momentum (3 positions)
        p_prev1 = (self.agent[0] - dy, self.agent[1] - dx)
        p_prev2 = (self.agent[0] - dy*2, self.agent[1] - dx*2)
        self.history_pos = [p_prev2, p_prev1, self.agent]
        
        self.path_points = [self.agent]
        self.steps = 0
        self.path_mask = np.zeros_like(full_img, dtype=np.float32)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        self.visited = set()
        self.visited.add((int(self.agent[0]), int(self.agent[1])))

    def obs(self):
        """Get observation for current position."""
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        ch0 = crop32_inference(self.img, int(curr[0]), int(curr[1]))
        ch1 = crop32_inference(self.img, int(p1[0]), int(p1[1]))
        ch2 = crop32_inference(self.img, int(p2[0]), int(p2[1]))
        ch3 = crop32_inference(self.path_mask, int(curr[0]), int(curr[1]))
        
        actor_obs = np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
        return actor_obs

    def step(self, a_idx):
        """Execute action and return (done, reason)."""
        self.steps += 1
        
        # Handle STOP action
        if a_idx == ACTION_STOP_IDX:
            return True, "Stopped by Agent"
        
        # Movement action
        dy, dx = ACTIONS_MOVEMENT[a_idx]
        STEP_ALPHA = 2.0
        ny = self.agent[0] + dy * STEP_ALPHA
        nx = self.agent[1] + dx * STEP_ALPHA
        
        # Check boundaries
        if not (0 <= ny < self.h and 0 <= nx < self.w):
            return True, "Hit Image Border"
        
        iy, ix = int(ny), int(nx)
        
        # Loop detection
        if (iy, ix) in self.visited and self.steps > 20:
            return True, "Loop Detected"
        self.visited.add((iy, ix))
        
        # Update state
        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[iy, ix] = 1.0
        
        if self.steps >= self.max_steps:
            return True, "Max Steps Reached"
        
        return False, "Running"

# ---------- INTERACTIVE CLICK HANDLER ----------
coords = []
def onclick(event):
    """Handle mouse clicks for start and direction selection."""
    global coords
    if event.xdata and event.ydata:
        ix, iy = int(event.xdata), int(event.ydata)
        coords.append((ix, iy))
        
        plt.plot(ix, iy, 'ro', markersize=5)
        if len(coords) > 1:
            plt.plot([coords[-2][0], coords[-1][0]], 
                    [coords[-2][1], coords[-1][1]], 'r-', linewidth=2)
        plt.draw()
        
        if len(coords) == 2:
            print("Direction set. Starting tracking...")
            plt.pause(0.5)
            plt.close()

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser(description="DSA RL Inference - Track curves in real DSA images")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to DSA image file")
    parser.add_argument("--actor_weights", type=str, required=True,
                        help="Path to actor-only weights (e.g., actor_Stage3_Realism_FINAL.pth)")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum steps before stopping")
    args = parser.parse_args()
    
    # 1. Load and preprocess image
    raw_img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        print(f"Error: Could not load image from {args.image_path}")
        return
    
    processed_img = preprocess_full_image(args.image_path)
    
    # 2. Load ActorOnly model
    K = 8
    model = ActorOnlyWithStop(n_movement_actions=N_MOVEMENT_ACTIONS, K=K).to(DEVICE)
    
    try:
        actor_weights = torch.load(args.actor_weights, map_location=DEVICE)
        model.load_state_dict(actor_weights)
        print(f"✓ Loaded actor weights from: {args.actor_weights}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("\nMake sure you're using actor-only weights (actor_*.pth), not full model weights.")
        print("Actor weights are automatically saved during training.")
        return
    
    model.eval()
    
    # 3. Get start point and direction from user clicks
    print("\n--- INSTRUCTIONS ---")
    print("1. Click START point")
    print("2. Click DIRECTION point")
    print("--------------------\n")
    
    plt.figure(figsize=(12, 12))
    plt.imshow(raw_img, cmap='gray')
    plt.title("Select Start & Direction")
    plt.connect('button_press_event', onclick)
    plt.show()
    
    if len(coords) < 2:
        print("Error: Need 2 clicks (start and direction)")
        return
    
    # Convert matplotlib (x,y) to numpy (y,x)
    p1_x, p1_y = coords[0]
    p2_x, p2_y = coords[1]
    vec_y = p2_y - p1_y
    vec_x = p2_x - p1_x
    
    # 4. Initialize inference environment
    env = InferenceEnv(processed_img, start_pt=(p1_y, p1_x), 
                       start_vector=(vec_y, vec_x), max_steps=args.max_steps)
    
    # 5. Prime action history
    start_action = get_closest_action(vec_y, vec_x)
    a_onehot = np.zeros(N_MOVEMENT_ACTIONS)
    a_onehot[start_action] = 1.0
    ahist = [a_onehot] * K
    
    # 6. Run inference
    done = False
    print(f"Tracking started... (Max steps: {args.max_steps})")
    
    while not done:
        obs = env.obs()
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
        A = fixed_window_history(ahist, K, N_MOVEMENT_ACTIONS)[None, ...]
        A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            movement_logits, stop_logit, _ = model(obs_t, A_t)
            stop_prob = torch.sigmoid(stop_logit).view(-1)
            # Greedy stop decision
            if stop_prob.item() > 0.5:
                action = ACTION_STOP_IDX
            else:
                action = torch.argmax(movement_logits, dim=1).item()
        
        done, reason = env.step(action)
        
        # Update history
        new_onehot = np.zeros(N_MOVEMENT_ACTIONS)
        if action != ACTION_STOP_IDX and action < N_MOVEMENT_ACTIONS:
            new_onehot[action] = 1.0
        ahist.append(new_onehot)
    
    print(f"Finished: {reason} ({env.steps} steps)")
    
    # 7. Visualize result
    path = env.path_points
    try:
        y = [p[0] for p in path]
        x = [p[1] for p in path]
        tck, u = splprep([y, x], s=20.0)
        new = splev(np.linspace(0, 1, len(path)*3), tck)
        sy, sx = new[0], new[1]
    except:
        sy, sx = [p[0] for p in path], [p[1] for p in path]
    
    plt.figure(figsize=(12, 12))
    plt.imshow(raw_img, cmap='gray')
    plt.plot(sx, sy, 'cyan', linewidth=2, label='Tracked Path')
    plt.plot(p1_x, p1_y, 'go', markersize=8, label='Start')
    if "Stopped" in reason:
        plt.plot(path[-1][1], path[-1][0], 'rx', markersize=10, 
                markeredgewidth=3, label='Stop')
    plt.legend()
    plt.title(f"Result: {reason}")
    plt.show()

if __name__ == "__main__":
    main()

