#!/usr/bin/env python3
"""
Supervised Trainer for the Standalone Stop Detector.
Generates a dataset of curve endpoints vs. midpoints.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add project root for curve generation logic
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from Experiment5_reward_config.src.train import CurveMakerFlexible, crop32, load_curve_config

class StopDataset(Dataset):
    def __init__(self, samples_per_class=5000):
        self.samples = []
        self.labels = []
        
        cfg, _ = load_curve_config()
        maker = CurveMakerFlexible(h=128, w=128, config=cfg)
        
        print(f"🏗️  Generating {samples_per_class * 2} supervised samples...")
        
        for _ in tqdm(range(samples_per_class)):
            # Generate a random curve
            img, mask, pts_all = maker.sample_curve(width_range=(1, 5), curvature_factor=1.2)
            pts = pts_all[0]
            
            # 1. POSITIVE SAMPLE: The end of the curve
            end_pt = pts[-1]
            # Mock a path mask ending there
            path_mask = np.zeros_like(img)
            for p in pts: path_mask[int(p[0]), int(p[1])] = 1.0
            
            crop_img = crop32(img, int(end_pt[0]), int(end_pt[1]))
            crop_path = crop32(path_mask, int(end_pt[0]), int(end_pt[1]))
            
            self.samples.append(np.stack([crop_img, crop_path], axis=0))
            self.labels.append(1.0)
            
            # 2. NEGATIVE SAMPLE: A random point in the middle
            mid_idx = np.random.randint(0, len(pts) - 10)
            mid_pt = pts[mid_idx]
            
            # Mock a path mask that only goes up to the middle
            path_mask_mid = np.zeros_like(img)
            for p in pts[:mid_idx+1]: path_mask_mid[int(p[0]), int(p[1])] = 1.0
            
            crop_img_mid = crop32(img, int(mid_pt[0]), int(mid_pt[1]))
            crop_path_mid = crop32(path_mask_mid, int(mid_pt[0]), int(mid_pt[1]))
            
            self.samples.append(np.stack([crop_img_mid, crop_path_mid], axis=0))
            self.labels.append(0.0)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

def train_stop_detector(epochs=10, batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from StopModule.src.models import StandaloneStopDetector
    
    model = StandaloneStopDetector().to(device)
    dataset = StopDataset(samples_per_class=5000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    print(f"🚀 Training on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            
        acc = correct / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.2%}")
        
    # Save Weights
    save_path = "StopModule/weights/stop_detector_v1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training complete. Model saved to {save_path}")

if __name__ == "__main__":
    train_stop_detector()
