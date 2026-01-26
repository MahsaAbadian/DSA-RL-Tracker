#!/usr/bin/env python3
"""
Inference test for separate stop module + 8-action movement weights.
Loads movement weights from any path and uses a standalone stop module
to decide when to stop (instead of the RL stop head).
"""
import argparse
import os
import sys
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Add project root + archive to sys.path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
_archive_root = os.path.join(_project_root, "archive")
for _path in (_project_root, _archive_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Imports from Experiment4 and StopModule
from Experiment4_separate_stop_v2.src.models import DecoupledStopActorOnly
from StopModule.src.models import StandaloneStopDetector

# Constants (mirrors Experiment4 rollout)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIONS_MOVEMENT = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
N_MOVEMENT_ACTIONS = len(ACTIONS_MOVEMENT)
ACTION_STOP_IDX = 8
CROP = 33

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def fixed_window_history(ahist_list, K, n_actions):
    out = np.zeros((K, n_actions), dtype=np.float32)
    if len(ahist_list) == 0:
        return out
    tail = ahist_list[-K:]
    out[-len(tail):] = np.stack(tail, axis=0)
    return out

def preprocess_full_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img = clahe.apply(img)
    img = img.astype(np.float32) / 255.0
    if np.median(img) > 0.5:
        print("Detected light background. Inverting image...")
        img = 1.0 - img
    return img

def crop32_inference(img: np.ndarray, cy: int, cx: int, size=CROP):
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
        out[oy0:oy0 + sh, ox0:ox0 + sw] = img[sy0:sy1, sx0:sx1]
    return out

def get_closest_action(dy, dx):
    best_idx = -1
    max_dot = -float("inf")
    mag = np.sqrt(dy ** 2 + dx ** 2) + 1e-6
    uy, ux = dy / mag, dx / mag

    for i, (ay, ax) in enumerate(ACTIONS_MOVEMENT):
        amag = np.sqrt(ay ** 2 + ax ** 2)
        ny, nx = ay / amag, ax / amag
        dot = uy * ny + ux * nx
        if dot > max_dot:
            max_dot = dot
            best_idx = i
    return best_idx

class InferenceEnv:
    def __init__(self, full_img, start_pt, start_vector, max_steps=1000):
        self.img = full_img
        self.h, self.w = full_img.shape
        self.max_steps = max_steps
        self.agent = tuple(start_pt)

        dy, dx = start_vector
        mag = np.sqrt(dy ** 2 + dx ** 2) + 1e-6
        dy, dx = (dy / mag) * 2.0, (dx / mag) * 2.0

        p_prev1 = (self.agent[0] - dy, self.agent[1] - dx)
        p_prev2 = (self.agent[0] - dy * 2, self.agent[1] - dx * 2)
        self.history_pos = [p_prev2, p_prev1, self.agent]

        self.path_points = [self.agent]
        self.steps = 0
        self.path_mask = np.zeros_like(full_img, dtype=np.float32)
        self.path_mask[int(self.agent[0]), int(self.agent[1])] = 1.0
        self.visited = {(int(self.agent[0]), int(self.agent[1]))}

    def obs(self):
        curr = self.history_pos[-1]
        p1 = self.history_pos[-2]
        p2 = self.history_pos[-3]

        ch0 = crop32_inference(self.img, int(curr[0]), int(curr[1]))
        ch1 = crop32_inference(self.img, int(p1[0]), int(p1[1]))
        ch2 = crop32_inference(self.img, int(p2[0]), int(p2[1]))
        ch3 = crop32_inference(self.path_mask, int(curr[0]), int(curr[1]))

        return np.stack([ch0, ch1, ch2, ch3], axis=0).astype(np.float32)

    def step(self, a_idx):
        self.steps += 1
        if a_idx == ACTION_STOP_IDX:
            return True, "Stopped by Stop Module"

        dy, dx = ACTIONS_MOVEMENT[a_idx]
        STEP_ALPHA = 2.0
        ny = self.agent[0] + dy * STEP_ALPHA
        nx = self.agent[1] + dx * STEP_ALPHA

        if not (0 <= ny < self.h and 0 <= nx < self.w):
            return True, "Hit Image Border"

        iy, ix = int(ny), int(nx)
        if (iy, ix) in self.visited and self.steps > 20:
            return True, "Loop Detected"
        self.visited.add((iy, ix))

        self.agent = (ny, nx)
        self.history_pos.append(self.agent)
        self.path_points.append(self.agent)
        self.path_mask[iy, ix] = 1.0

        if self.steps >= self.max_steps:
            return True, "Max Steps Reached"

        return False, "Running"

coords = []
def onclick(event):
    global coords
    if event.xdata and event.ydata:
        ix, iy = int(event.xdata), int(event.ydata)
        coords.append((ix, iy))
        plt.plot(ix, iy, "ro", markersize=5)
        if len(coords) > 1:
            plt.plot([coords[-2][0], coords[-1][0]], [coords[-2][1], coords[-1][1]], "r-", linewidth=2)
        plt.draw()
        if len(coords) == 2:
            print("Direction set. Starting tracking...")
            plt.pause(0.5)
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Hybrid rollout: movement + standalone stop module")
    parser.add_argument("--image_path", type=str, required=True, help="Path to DSA image file")
    parser.add_argument("--movement_weights", type=str, required=True, help="Path to 8-action movement weights (.pth)")
    parser.add_argument("--stop_weights", type=str, required=True, help="Path to standalone stop module weights (.pth)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps before stopping")
    parser.add_argument("--stop_threshold", type=float, default=0.7, help="Stop probability threshold")
    args = parser.parse_args()

    raw_img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        print(f"Error: Could not load image from {args.image_path}")
        return
    processed_img = preprocess_full_image(args.image_path)

    K = 8
    movement_model = DecoupledStopActorOnly(n_movement_actions=N_MOVEMENT_ACTIONS, K=K).to(DEVICE)
    stop_model = StandaloneStopDetector().to(DEVICE)

    try:
        movement_state = torch.load(args.movement_weights, map_location=DEVICE)
        movement_model.load_state_dict(movement_state, strict=False)
        print(f"✓ Loaded movement weights from: {args.movement_weights}")
    except Exception as e:
        print(f"Error loading movement weights: {e}")
        return

    try:
        stop_state = torch.load(args.stop_weights, map_location=DEVICE)
        stop_model.load_state_dict(stop_state)
        print(f"✓ Loaded stop weights from: {args.stop_weights}")
    except Exception as e:
        print(f"Error loading stop weights: {e}")
        return

    movement_model.eval()
    stop_model.eval()

    print("\n--- INSTRUCTIONS ---")
    print("1. Click START point")
    print("2. Click DIRECTION point")
    print("--------------------\n")

    plt.figure(figsize=(12, 12))
    plt.imshow(raw_img, cmap="gray")
    plt.title("Select Start & Direction")
    plt.connect("button_press_event", onclick)
    plt.show()

    if len(coords) < 2:
        print("Error: Need 2 clicks (start and direction)")
        return

    p1_x, p1_y = coords[0]
    p2_x, p2_y = coords[1]
    vec_y = p2_y - p1_y
    vec_x = p2_x - p1_x

    env = InferenceEnv(processed_img, start_pt=(p1_y, p1_x),
                       start_vector=(vec_y, vec_x), max_steps=args.max_steps)

    start_action = get_closest_action(vec_y, vec_x)
    a_onehot = np.zeros(N_MOVEMENT_ACTIONS)
    a_onehot[start_action] = 1.0
    ahist = [a_onehot] * K

    done = False
    print(f"Tracking started... (Max steps: {args.max_steps})")

    while not done:
        obs = env.obs()
        obs_t = torch.tensor(obs[None], dtype=torch.float32, device=DEVICE)
        A = fixed_window_history(ahist, K, N_MOVEMENT_ACTIONS)[None, ...]
        A_t = torch.tensor(A, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            movement_logits, _, _ = movement_model(obs_t, A_t)
            stop_input = torch.cat([obs_t[:, 0:1, :, :], obs_t[:, 3:4, :, :]], dim=1)
            stop_logit = stop_model(stop_input)
            stop_prob = torch.sigmoid(stop_logit).view(-1)

            if stop_prob.item() > args.stop_threshold:
                action = ACTION_STOP_IDX
            else:
                action = torch.argmax(movement_logits, dim=1).item()

        done, reason = env.step(action)

        new_onehot = np.zeros(N_MOVEMENT_ACTIONS)
        if action != ACTION_STOP_IDX and action < N_MOVEMENT_ACTIONS:
            new_onehot[action] = 1.0
        ahist.append(new_onehot)

    print(f"Finished: {reason} ({env.steps} steps)")

    path = env.path_points
    try:
        y = [p[0] for p in path]
        x = [p[1] for p in path]
        tck, u = splprep([y, x], s=20.0)
        new = splev(np.linspace(0, 1, len(path) * 3), tck)
        sy, sx = new[0], new[1]
    except Exception:
        sy, sx = [p[0] for p in path], [p[1] for p in path]

    plt.figure(figsize=(12, 12))
    plt.imshow(raw_img, cmap="gray")
    plt.plot(sx, sy, "cyan", linewidth=2, label="Tracked Path")
    plt.plot(p1_x, p1_y, "go", markersize=8, label="Start")
    if "Stopped" in reason:
        plt.plot(path[-1][1], path[-1][0], "rx", markersize=10,
                 markeredgewidth=3, label="Stop")
    plt.legend()
    plt.title(f"Result: {reason}")
    plt.show()

if __name__ == "__main__":
    main()
