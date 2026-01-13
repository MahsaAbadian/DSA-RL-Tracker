#!/usr/bin/env python3
"""
Standalone Stop Detector Model.
Trained via Supervised Learning to detect the end of a curve.
"""
import torch
import torch.nn as nn

def gn(c): 
    return nn.GroupNorm(4, c)

class StandaloneStopDetector(nn.Module):
    """
    A vision-only binary classifier that takes a 2-channel crop:
    Channel 0: The DSA Image crop (33x33)
    Channel 1: The Path Mask (where the agent has already been)
    """
    def __init__(self):
        super().__init__()
        
        # Deeper CNN than the RL version to capture more visual nuance
        self.backbone = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), gn(32), nn.PReLU(),
            nn.Conv2d(32, 64, 3, padding=2, dilation=2), gn(64), nn.PReLU(),
            nn.Conv2d(64, 128, 3, padding=3, dilation=3), gn(128), nn.PReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 1) # Output: Logit for "is terminal"
        )

    def forward(self, x):
        # x shape: (B, 2, 33, 33)
        features = self.backbone(x).flatten(1)
        logit = self.head(features).squeeze(-1)
        return logit

if __name__ == "__main__":
    # Quick Test
    model = StandaloneStopDetector()
    dummy_input = torch.randn(8, 2, 33, 33)
    output = model(dummy_input)
    print(f"Model Test - Input: {dummy_input.shape} -> Output: {output.shape}")
