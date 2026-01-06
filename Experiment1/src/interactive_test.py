#!/usr/bin/env python3
"""
Interactive testing script for DSA RL model.
Generates synthetic curves and visualizes tracking step-by-step with keyboard control.
"""
import argparse
import numpy as np
import torch
import cv2
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

# Add parent directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from src.models import ActorOnly
from src.curve_generator import CurveMakerFlexible, load_curve_config

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0,-1), (0, 1), (-1,-1), (-1,1), (1,-1), (1,1)]
ACTION_STOP_IDX = 8
N_ACTIONS = 9
K = 8
STEP_ALPHA = 1.0
CROP = 33

ACTION_NAMES = [
    "UP", "DOWN", "LEFT", "RIGHT",
    "UP-LEFT", "UP-RIGHT", "DOWN-LEFT", "DOWN-RIGHT", "STOP"
]

def clamp(v, lo, hi): 
    return max(lo, min(v, hi))

def crop32(img, cy, cx):
    """Crop 33x33 patch centered at (cy, cx)."""
    h, w = img.shape
    y0, y1 = cy - CROP//2, cy + CROP//2 + 1
    x0, x1 = cx - CROP//2, cx + CROP//2 + 1
    out = np.zeros((CROP, CROP), dtype=img.dtype)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)
    oy0 = sy0 - y0; ox0 = sx0 - x0
    sh = sy1 - sy0; sw = sx1 - sx0
    if sh > 0 and sw > 0:
        out[oy0:oy0+sh, ox0:ox0+sw] = img[sy0:sy1, sx0:sx1]
    return out

def fixed_window_history(ahist_list, K, n_actions):
    """Create fixed-size window of action history."""
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0: 
        return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

class InteractiveEnv:
    """Simple environment for interactive testing."""
    def __init__(self, img, gt_poly, start_idx=5):
        self.img = img
        self.h, self.w = img.shape
        self.gt_poly = gt_poly
        
        # Start position
        if start_idx >= len(gt_poly):
            start_idx = len(gt_poly) - 1
        start_pt = gt_poly[start_idx]
        self.agent = (float(start_pt[0]), float(start_pt[1]))
        
        # History
        if start_idx >= 2:
            p2 = gt_poly[start_idx-2]
            p1 = gt_poly[start_idx-1]
        elif start_idx >= 1:
            p2 = gt_poly[0]
            p1 = gt_poly[start_idx-1]
        else:
            p2 = p1 = gt_poly[0]
        self.history_pos = [tuple(p2), tuple(p1), tuple(self.agent)]
        
        self.path_points = [self.agent]
        self.path_mask = np.zeros_like(img, dtype=np.float32)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        self.steps = 0
        self.max_steps = 1000
        
    def obs(self):
        """Get observation."""
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]
        
        ch0 = crop32(self.img, int(curr[0]), int(curr[1]))
        ch1 = crop32(self.img, int(p1[0]), int(p1[1]))
        ch2 = crop32(self.img, int(p2[0]), int(p2[1]))
        ch3 = crop32(self.path_mask, int(curr[0]), int(curr[1]))
        
        return np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)
    
    def step(self, action):
        """Take a step."""
        if action == ACTION_STOP_IDX:
            return True, "Stopped"
        
        dy, dx = ACTIONS_MOVEMENT[action]
        ny = clamp(self.agent[0] + dy * STEP_ALPHA, 0, self.h-1)
        nx = clamp(self.agent[1] + dx * STEP_ALPHA, 0, self.w-1)
        self.agent = (ny, nx)
        
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[int(ny), int(nx)] = 1.0
        self.steps += 1
        
        # Check if reached end
        dist_to_end = np.sqrt(
            (self.agent[0] - self.gt_poly[-1][0])**2 + 
            (self.agent[1] - self.gt_poly[-1][1])**2
        )
        if dist_to_end < 5.0:
            return True, "Reached end"
        
        if self.steps >= self.max_steps:
            return True, "Max steps"
        
        return False, ""

def visualize_step(img, gt_poly, path_points, agent_pos, action_name, step_num, done=False):
    """Visualize current state."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Full image with path
    axes[0].imshow(img, cmap='gray')
    
    # Ground truth path
    gt_y = [p[0] for p in gt_poly]
    gt_x = [p[1] for p in gt_poly]
    axes[0].plot(gt_x, gt_y, 'g-', linewidth=2, alpha=0.5, label='Ground Truth')
    
    # Tracked path
    if len(path_points) > 1:
        path_y = [p[0] for p in path_points]
        path_x = [p[1] for p in path_points]
        axes[0].plot(path_x, path_y, 'cyan', linewidth=2, label='Tracked Path')
    
    # Current agent position
    axes[0].plot(agent_pos[1], agent_pos[0], 'ro', markersize=10, label='Current Position')
    
    # Start and end points
    axes[0].plot(gt_x[0], gt_y[0], 'go', markersize=8, label='Start')
    axes[0].plot(gt_x[-1], gt_y[-1], 'gx', markersize=10, markeredgewidth=3, label='End')
    
    axes[0].set_title(f'Step {step_num} - Action: {action_name}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].axis('off')
    
    # Right: Agent's view (33x33 crop)
    obs = crop32(img, int(agent_pos[0]), int(agent_pos[1]))
    axes[1].imshow(obs, cmap='gray')
    axes[1].set_title('Agent View (33x33 crop)', fontsize=12)
    axes[1].axis('off')
    
    # Add info text
    info_text = f"Step: {step_num}\nAction: {action_name}\nPosition: ({int(agent_pos[0])}, {int(agent_pos[1])})"
    if done:
        info_text += f"\n\nStatus: {done}"
    info_text += "\n\nPress ANY KEY to continue\nPress 'q' to quit"
    axes[1].text(0.5, -0.15, info_text, transform=axes[1].transAxes, 
                 ha='center', va='top', fontsize=10, 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Interactive testing with synthetic curves")
    parser.add_argument("--actor_weights", type=str, required=True,
                        help="Path to actor-only weights")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to curve config (default: curve_config.json)")
    parser.add_argument("--width", type=int, nargs=2, default=[1, 2],
                        help="Width range (default: 1 2)")
    parser.add_argument("--intensity", type=float, nargs=2, default=[0.15, 0.25],
                        help="Intensity range (default: 0.15 0.25)")
    parser.add_argument("--noise", type=float, default=0.5,
                        help="Noise probability (default: 0.3)")
    parser.add_argument("--background", type=float, default=0.1,
                        help="Background intensity (default: 0.1)")
    parser.add_argument("--curvature", type=float, default=5,
                        help="Curvature factor (default: 1.5)")
    parser.add_argument("--seed", type=int, default=34,
                        help="Random seed for curve generation (default: 42)")
    parser.add_argument("--start_idx", type=int, default=5,
                        help="Starting index on path (default: 5)")
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = ActorOnly(n_actions=N_ACTIONS, K=K).to(DEVICE)

    try:
        actor_weights = torch.load(args.actor_weights, map_location=DEVICE)
        model.load_state_dict(actor_weights)
        print(f"✓ Loaded weights from: {args.actor_weights}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    
    model.eval()
    
    # Load config and generate curve
    print("Generating synthetic curve...")
    print(f"  Parameters: width={args.width}, intensity={args.intensity}, noise={args.noise}")
    if args.background is not None:
        print(f"  Background intensity: {args.background}")
    print(f"  Curvature: {args.curvature}, Seed: {args.seed}")
    
    curve_config, _ = load_curve_config(args.config)
    curve_maker = CurveMakerFlexible(h=128, w=128, seed=args.seed, config=curve_config)
    
    img, mask, pts_all = curve_maker.sample_curve(
        width_range=tuple(args.width),
        noise_prob=args.noise,
        invert_prob=0.5,
        min_intensity=args.intensity[0],
        max_intensity=args.intensity[1],
        branches=False,
        curvature_factor=args.curvature,
        allow_self_cross=False,
        self_cross_prob=0.0,
        background_intensity=args.background
    )
    
    gt_poly = pts_all[0].astype(np.float32)
    print(f"✓ Generated curve with {len(gt_poly)} points")
    
    # Initialize environment
    env = InteractiveEnv(img, gt_poly, start_idx=args.start_idx)
    
    # Initialize action history
    ahist = []
    for _ in range(K):
        a_onehot = np.zeros(N_ACTIONS)
        ahist.append(a_onehot)
    
    print("\n" + "="*60)
    print("INTERACTIVE TESTING")
    print("="*60)
    print("Click on the plot window, then:")
    print("  - Press ANY KEY (or Enter/Space) to take next step")
    print("  - Press 'q' to quit")
    print("="*60 + "\n")
    
    step_num = 0
    done = False
    quit_flag = [False]  # Use list to allow modification in nested function
    
    def on_key_press(event):
        """Handle key press events."""
        if event.key == 'q' or event.key == 'Q':
            quit_flag[0] = True
            plt.close('all')
        elif event.key is not None:
            # Any other key (including Enter, Space) continues to next step
            # Close the figure to unblock plt.show()
            plt.close(event.canvas.figure)
    
    while not done and not quit_flag[0]:
        # Get observation
        obs = env.obs()
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
        A = fixed_window_history(ahist, K, N_ACTIONS)[None, ...]
        A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)
        
        # Get action
        with torch.no_grad():
            logits, _ = model(obs_t, A_t)
            probs = torch.softmax(logits, dim=1)
            action = torch.argmax(probs, dim=1).item()
            action_probs = probs[0].cpu().numpy()
        
        action_name = ACTION_NAMES[action]
        
        # Visualize
        fig = visualize_step(img, gt_poly, env.path_points, env.agent, 
                            action_name, step_num, done)
        
        # Connect key press event
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Print info
        print(f"Step {step_num}: Action = {action_name} (prob: {action_probs[action]:.3f})")
        print(f"  Position: ({int(env.agent[0])}, {int(env.agent[1])})")
        print("  (Click on plot window, then press any key to continue, 'q' to quit)")
        
        # Show plot and wait for key press (blocks until figure is closed)
        plt.show(block=True)
        
        if quit_flag[0]:
            print("Quitting...")
            break
        
        # Take step
        done, reason = env.step(action)
        
        # Update history
        new_onehot = np.zeros(N_ACTIONS)
        new_onehot[action] = 1.0
        ahist.append(new_onehot)
        
        step_num += 1
        
        if done:
            print(f"\nFinished: {reason}")
            fig = visualize_step(img, gt_poly, env.path_points, env.agent, 
                                 action_name, step_num, done=reason)
            fig.canvas.mpl_connect('key_press_event', on_key_press)
            print("Press any key to exit...")
            plt.show(block=True)
            break
    
    plt.close('all')

if __name__ == "__main__":
    main()

