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

# Add project root for curve generation logic
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import from Experiment 5 (latest generation logic)
try:
    from Experiment5_reward_config.src.train import CurveMakerFlexible, crop32, load_curve_config
except ImportError:
    # Fallback for colab or other environments
    from src.train import CurveMakerFlexible, crop32, load_curve_config

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
        maker = CurveMakerFlexible(h=128, w=128, config=cfg)
        
        print(f"🏗️  Generating {samples_per_class * 2} supervised samples...")
        if vessel_realism:
            print("   Using vessel-realistic features: tapering (wide→narrow) and fading (bright→dim)")
        
        for _ in tqdm(range(samples_per_class)):
            # Generate a random curve with varied parameters
            w_range = np.random.choice([(1, 3), (3, 5), (1, 5), (2, 6)])
            curvature = np.random.uniform(0.3, 1.5)
            
            # For vessel realism: use tapering and fading
            if vessel_realism:
                # Vessel-like: wide at start, narrow at end (like real vessels)
                width_variation = "wide_to_narrow"
                # Vessel-like: bright at start, dim/faded at end
                intensity_variation = "bright_to_dim"
                
                # Ensure end is narrow (1-2 pixels) for realistic vessel termination
                start_width = np.random.randint(w_range[1], w_range[1] + 3)  # Wide start
                end_width = np.random.randint(1, max(2, w_range[0]))  # Narrow end (1-2 pixels)
                
                # Ensure end fades significantly
                start_intensity = np.random.uniform(0.7, 1.0)  # Bright start
                end_intensity = np.random.uniform(0.2, 0.5)  # Dim end
            else:
                # Original uniform behavior
                width_variation = "none"
                intensity_variation = "none"
                start_width = None
                end_width = None
                start_intensity = None
                end_intensity = None
            
            img, mask, pts_all = maker.sample_curve(
                width_range=w_range, 
                curvature_factor=curvature,
                noise_prob=0.3,
                invert_prob=0.5,
                width_variation=width_variation,
                start_width=start_width,
                end_width=end_width,
                intensity_variation=intensity_variation,
                start_intensity=start_intensity,
                end_intensity=end_intensity
            )
            pts = pts_all[0]
            
            # 1. POSITIVE SAMPLE: The end of the curve
            end_pt = pts[-1]
            path_mask = np.zeros_like(img)
            for p in pts: path_mask[int(p[0]), int(p[1])] = 1.0
            
            crop_img = crop32(img, int(end_pt[0]), int(end_pt[1]))
            crop_path = crop32(path_mask, int(end_pt[0]), int(end_pt[1]))
            
            self.samples.append(np.stack([crop_img, crop_path], axis=0))
            self.labels.append(1.0)
            
            # 2. NEGATIVE SAMPLE: A random point in the middle
            # Stay away from the very end to avoid ambiguity
            mid_idx = np.random.randint(0, max(1, len(pts) - 15))
            mid_pt = pts[mid_idx]
            
            path_mask_mid = np.zeros_like(img)
            for p in pts[:mid_idx+1]: path_mask_mid[int(p[0]), int(p[1])] = 1.0
            
            crop_img_mid = crop32(img, int(mid_pt[0]), int(mid_pt[1]))
            crop_path_mid = crop32(path_mask_mid, int(mid_pt[0]), int(mid_pt[1]))
            
            self.samples.append(np.stack([crop_img_mid, crop_path_mid], axis=0))
            self.labels.append(0.0)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

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
    from StopModule.src.models import StandaloneStopDetector
    
    # Load config if provided
    config = None
    if config_path:
        from Experiment5_reward_config.src.train import load_curve_config
        config = load_curve_config(config_path)[0]
    
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
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_acc = 0.0
    save_path = output_path or "StopModule/weights/stop_detector_v1.pth"
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
    parser.add_argument("--output", type=str, default=None, help="Output path for weights (default: StopModule/weights/stop_detector_v1.pth)")
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
    print(f"Output Path: {args.output or 'StopModule/weights/stop_detector_v1.pth'}")
    print(f"Vessel Realism: {not args.no_vessel_realism} (tapering & fading)")
    if args.config:
        print(f"Config Path: {args.config}")
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
