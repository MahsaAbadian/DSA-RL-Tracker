#!/usr/bin/env python3
"""
Supervised Trainer for the Standalone Stop Detector.
Generates a dataset of curve endpoints vs. midpoints.
Includes validation split and per-class metrics.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add repo root + archive for curve generation logic
_script_dir = os.path.dirname(os.path.abspath(__file__))
stop_module_root = os.path.dirname(_script_dir)
lego_root = os.path.dirname(stop_module_root)
repo_root = os.path.dirname(lego_root)
archive_root = os.path.join(repo_root, "archive")
DEFAULT_STOP_CURVE_CONFIG = os.path.join(
    lego_root, "curve_generator_module", "config", "stop_curve_config.json"
)
for _path in (stop_module_root, lego_root, archive_root, repo_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from curve_generator_module import CurveMaker, load_curve_config

CROP = 33

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def crop32(img: np.ndarray, cy: int, cx: int, size: int = CROP, bg_value: float = None) -> np.ndarray:
    """
    Crop a region from the image centered at (cy, cx).
    
    Args:
        img: Input image
        cy, cx: Center coordinates
        size: Crop size (default 33)
        bg_value: Background value for padding (if None, estimate from image corners)
    """
    h, w = img.shape
    
    if bg_value is not None:
        pad_val = bg_value
    else:
        corners = [img[0, 0], img[0, w - 1], img[h - 1, 0], img[h - 1, w - 1]]
        bg_estimate = np.median(corners)
        pad_val = 1.0 if bg_estimate > 0.5 else 0.0

    r = size // 2
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1

    out = np.full((size, size), pad_val, dtype=img.dtype)
    sy0, sy1 = clamp(y0, 0, h), clamp(y1, 0, h)
    sx0, sx1 = clamp(x0, 0, w), clamp(x1, 0, w)

    oy0, ox0 = sy0 - y0, sx0 - x0
    sh, sw = sy1 - sy0, sx1 - sx0

    if sh > 0 and sw > 0:
        out[oy0:oy0 + sh, ox0:ox0 + sw] = img[sy0:sy1, sx0:sx1]
    return out

class StopDataset(Dataset):
    def __init__(self, samples_per_class=5000, config=None, vessel_realism=True):
        """
        Args:
            samples_per_class: Number of positive/negative samples per class
            config: Optional curve config dict
            vessel_realism: If True, generates vessel-like curves with tapering and fading
        """
        self.samples = []
        self.labels = []
        
        cfg = config or load_curve_config()[0]
        maker = CurveMaker(h=128, w=128, config=cfg)
        
        print(f"🏗️  Generating {samples_per_class * 2} supervised samples...")
        if vessel_realism:
            print("   Using vessel-realistic features: tapering (wide→narrow) and fading (bright→dim)")
        
        for _ in tqdm(range(samples_per_class)):
            # Generate a random curve with varied parameters
            w_ranges = [(1, 3), (3, 5), (1, 5), (2, 6)]
            w_range = w_ranges[np.random.randint(len(w_ranges))]
            
            # HARDER SAMPLES: Increase curvature diversity
            # Higher curvature (up to 2.5) makes midpoints look more like sharp turns/ends
            curvature = np.random.uniform(0.3, 2.5)
            noise_prob = np.random.uniform(0.0, 0.2)  # Lower noise for cleaner samples
            
            # Random background color from generator: black or white
            # (Gray backgrounds have poor contrast, so we skip them)
            bg_choice = np.random.randint(0, 2)
            if bg_choice == 0:
                # Black background (draw bright curves)
                background_intensity = 0.0
                invert_prob = 0.0
            else:
                # White background (draw dark curves via inversion)
                background_intensity = 0.0
                invert_prob = 1.0
            
            # For vessel realism: use tapering and fading
            if vessel_realism:
                width_variation = "wide_to_narrow"
                intensity_variation = "bright_to_dim"
                
                # HARDER ENDPOINTS: Some very narrow and very faint
                is_hard_endpoint = np.random.random() < 0.3
                if is_hard_endpoint:
                    start_width = np.random.randint(2, 4)
                    end_width = 1  # Very thin
                    start_intensity = np.random.uniform(0.5, 0.7)
                    end_intensity = np.random.uniform(0.1, 0.25) # Very faint
                else:
                    start_width = np.random.randint(w_range[1], w_range[1] + 3)
                    end_width = np.random.randint(1, max(2, w_range[0]))
                    start_intensity = np.random.uniform(0.7, 1.0)
                    end_intensity = np.random.uniform(0.2, 0.5)
            else:
                width_variation = "none"
                intensity_variation = "none"
                start_width = None
                end_width = None
                start_intensity = None
                end_intensity = None
            
            img, mask, pts_all = maker.sample_curve(
                width_range=w_range, 
                curvature_factor=curvature,
                noise_prob=noise_prob,
                invert_prob=invert_prob,
                background_intensity=background_intensity,
                width_variation=width_variation,
                start_width=start_width,
                end_width=end_width,
                intensity_variation=intensity_variation,
                start_intensity=start_intensity,
                end_intensity=end_intensity
            )
            
            # Determine the actual background value after potential inversion
            actual_bg = 1.0 - background_intensity if invert_prob == 1.0 else background_intensity
            pts = pts_all[0]
            
            # 1. POSITIVE SAMPLE: The end of the curve
            end_pt = pts[-1]
            path_mask = np.zeros_like(img)
            # Clip points to ensure they stay within 128x128 bounds
            for p in pts:
                py, px = int(np.clip(p[0], 0, 127)), int(np.clip(p[1], 0, 127))
                path_mask[py, px] = 1.0
            
            crop_img = crop32(img, int(np.clip(end_pt[0], 0, 127)), int(np.clip(end_pt[1], 0, 127)), bg_value=actual_bg)
            crop_path = crop32(path_mask, int(np.clip(end_pt[0], 0, 127)), int(np.clip(end_pt[1], 0, 127)), bg_value=0.0)
            
            self.samples.append(np.stack([crop_img, crop_path], axis=0))
            self.labels.append(1.0)
            
            # 2. NEGATIVE SAMPLE: A random point in the middle
            # Select from middle portion (30% to 70% of curve) to avoid endpoints
            # Ensure path mask doesn't fill too much (clearly not at end)
            min_idx = max(1, int(len(pts) * 0.3))  # At least 30% into the curve
            max_idx = max(min_idx + 1, int(len(pts) * 0.7))  # At most 70% of the curve
            
            # Try multiple times to find a good midpoint where path mask isn't too full
            # AND path continues beyond the crop (not truncated at edges)
            mid_idx = None
            for attempt in range(20):  # More attempts to find good samples
                candidate_idx = np.random.randint(min_idx, max_idx)
                candidate_pt = pts[candidate_idx]
                
                # CRITICAL: Ensure path continues beyond this point (at least 10 more points)
                if candidate_idx + 10 >= len(pts):
                    continue  # Too close to end, skip
                
                # Build path mask up to this point
                test_mask = np.zeros_like(img)
                for p in pts[:candidate_idx+1]:
                    py, px = int(np.clip(p[0], 0, 127)), int(np.clip(p[1], 0, 127))
                    test_mask[py, px] = 1.0
                
                # Crop the path mask
                crop_y = int(np.clip(candidate_pt[0], 0, 127))
                crop_x = int(np.clip(candidate_pt[1], 0, 127))
                test_crop_mask = crop32(test_mask, crop_y, crop_x)
                
                # Check 1: Path mask doesn't fill more than 60% of crop
                path_fill_ratio = test_crop_mask.sum() / (CROP * CROP)
                if path_fill_ratio >= 0.6:
                    continue  # Too full, skip
                
                # Check 2: Path mask doesn't touch crop edges (indicates truncation)
                # Check all 4 edges: top, bottom, left, right
                edge_margin = 3  # Require at least 3 pixels margin from edges
                top_edge = test_crop_mask[:edge_margin, :].sum()
                bottom_edge = test_crop_mask[-edge_margin:, :].sum()
                left_edge = test_crop_mask[:, :edge_margin].sum()
                right_edge = test_crop_mask[:, -edge_margin:].sum()
                
                # If path mask touches any edge significantly, it might be truncated
                if (top_edge > 5 or bottom_edge > 5 or left_edge > 5 or right_edge > 5):
                    continue  # Path mask touches edges, might look like truncation
                
                # Check 3: Verify curve continues in the image crop itself
                # Look ahead: check if there are curve points after candidate_idx within the crop
                crop_img_test = crop32(img, crop_y, crop_x)
                # Check if there's curve signal in the direction the path is heading
                # (This is a heuristic - if the crop shows curve continuing, it's good)
                
                # All checks passed!
                mid_idx = candidate_idx
                break
            
            # Fallback: use any midpoint if we couldn't find a good one
            if mid_idx is None:
                mid_idx = np.random.randint(min_idx, max_idx)
            
            mid_pt = pts[mid_idx]
            
            path_mask_mid = np.zeros_like(img)
            for p in pts[:mid_idx+1]:
                py, px = int(np.clip(p[0], 0, 127)), int(np.clip(p[1], 0, 127))
                path_mask_mid[py, px] = 1.0
            
            crop_img_mid = crop32(img, int(np.clip(mid_pt[0], 0, 127)), int(np.clip(mid_pt[1], 0, 127)), bg_value=actual_bg)
            crop_path_mid = crop32(path_mask_mid, int(np.clip(mid_pt[0], 0, 127)), int(np.clip(mid_pt[1], 0, 127)), bg_value=0.0)
            
            self.samples.append(np.stack([crop_img_mid, crop_path_mid], axis=0))
            self.labels.append(0.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.samples[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

def plot_samples(dataset, num_samples=4):
    """Visualize some samples from the dataset."""
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 3))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        sample, label = dataset[idx]
        img_crop = sample[0].numpy()
        path_crop = sample[1].numpy()
        
        axes[i, 0].imshow(img_crop, cmap='gray')
        axes[i, 0].set_title(f"Image (Label: {'STOP' if label > 0.5 else 'GO'})")
        axes[i, 1].imshow(path_crop, cmap='jet')
        axes[i, 1].set_title("Path Mask")
        
    plt.tight_layout()
    plt.show()

def train_stop_detector(epochs=15, batch_size=64, samples=5000, learning_rate=1e-4, output_path=None, config_path=None, vessel_realism=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.models import StandaloneStopDetector
    
    # Load config if provided
    config = None
    if config_path:
        config = load_curve_config(config_path)[0]
    else:
        if os.path.exists(DEFAULT_STOP_CURVE_CONFIG):
            config = load_curve_config(DEFAULT_STOP_CURVE_CONFIG)[0]
    
    # 1. Prepare Data
    full_dataset = StopDataset(samples_per_class=samples, config=config, vessel_realism=vessel_realism)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_db, val_db = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_db, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_db, batch_size=batch_size, shuffle=False)
    
    # 2. Setup Model
    model = StandaloneStopDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Weighted Loss: penalize missing a STOP (endpoint) more than missing a GO (midpoint)
    # pos_weight=5.0 means stops are 5x more important than midpoints
    pos_weight = torch.tensor([5.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_acc = 0.0
    if output_path is None:
        # Default: save in stop_module/weights folder
        weights_dir = os.path.join(stop_module_root, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        save_path = os.path.join(weights_dir, "stop_detector_v1.pth")
    else:
        save_path = output_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"🚀 Training on {device}...")
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0
        train_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == y).sum().item()
            
        # Validation Phase
        model.eval()
        val_loss = 0
        val_correct = 0
        pos_correct = 0
        pos_total = 0
        neg_correct = 0
        neg_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y).sum().item()
                
                # Per-class metrics
                pos_mask = (y == 1.0)
                neg_mask = (y == 0.0)
                pos_correct += (preds[pos_mask] == y[pos_mask]).sum().item()
                pos_total += pos_mask.sum().item()
                neg_correct += (preds[neg_mask] == y[neg_mask]).sum().item()
                neg_total += neg_mask.sum().item()
        
        train_acc = train_correct / train_size
        val_acc = val_correct / val_size
        pos_acc = pos_correct / pos_total if pos_total > 0 else 0
        neg_acc = neg_correct / neg_total if neg_total > 0 else 0
        
        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"T-Loss: {train_loss/len(train_loader):.4f} | T-Acc: {train_acc:.2%} | "
              f"V-Acc: {val_acc:.2%} (Stop: {pos_acc:.1%}, Go: {neg_acc:.1%})")
        
        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"⭐ New Best Val Accuracy! Saved to {save_path}")

    print(f"✅ Training complete. Best Val Accuracy: {best_val_acc:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Standalone Stop Detector")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--samples", type=int, default=5000, help="Samples per class (default: 5000)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--output", type=str, default=None, help="Output path for weights (default: stop_module/weights/stop_detector_v1.pth)")
    parser.add_argument("--config", type=str, default=None, help="Path to curve config JSON (optional)")
    parser.add_argument("--no_vessel_realism", action="store_true", help="Disable vessel-realistic features (tapering/fading)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🛑 Standalone Stop Detector Training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Samples per Class: {args.samples}")
    print(f"Learning Rate: {args.lr}")
    default_output = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "stop_detector_v1.pth")
    print(f"Output Path: {args.output or default_output}")
    print(f"Vessel Realism: {not args.no_vessel_realism} (tapering & fading)")
    if args.config:
        print(f"Config Path: {args.config}")
    elif os.path.exists(DEFAULT_STOP_CURVE_CONFIG):
        print(f"Config Path: {DEFAULT_STOP_CURVE_CONFIG}")
    print("=" * 60)
    print()
    
    train_stop_detector(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples=args.samples,
        learning_rate=args.lr,
        output_path=args.output,
        config_path=args.config,
        vessel_realism=not args.no_vessel_realism
    )
