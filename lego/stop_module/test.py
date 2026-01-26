#!/usr/bin/env python3
"""
Test script for the Standalone Stop Detector.

Tests the model with dummy inputs and verifies it works correctly.
"""
import os
import sys
import torch
import numpy as np

# Add paths for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
stop_module_root = _script_dir
lego_root = os.path.dirname(stop_module_root)
repo_root = os.path.dirname(lego_root)

for _path in (stop_module_root, lego_root, repo_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.models import StandaloneStopDetector

def test_model_architecture():
    """Test that the model architecture is correct."""
    print("=" * 60)
    print("Testing Model Architecture")
    print("=" * 60)
    
    model = StandaloneStopDetector()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Test input shape: (batch, channels, height, width) = (1, 2, 33, 33)
    dummy_input = torch.randn(1, 2, 33, 33, device=device)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Input shape: {dummy_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output logit: {output.item():.3f}")
    print(f"✓ Output probability: {torch.sigmoid(output).item():.4f}")
    
    assert output.shape == (1,), f"Expected output shape (1,), got {output.shape}"
    print("✓ Model architecture test passed!\n")
    return model, device

def test_model_with_weights(weights_path=None):
    """Test loading and using trained weights."""
    print("=" * 60)
    print("Testing Model with Weights")
    print("=" * 60)
    
    if weights_path is None:
        weights_path = os.path.join(_script_dir, "weights", "stop_detector_v1.pth")
    
    if not os.path.exists(weights_path):
        print(f"⚠️  Weights not found at: {weights_path}")
        print("   Skipping weights test. Train the model first with:")
        print("   python src/train_standalone.py")
        return None
    
    model = StandaloneStopDetector()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    try:
        state_dict = torch.load(weights_path, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        
        print(f"✓ Loaded weights from: {weights_path}")
        
        # Test with dummy input
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 2, 33, 33, device=device)
            output = model(dummy_input)
            prob = torch.sigmoid(output)
        
        print(f"✓ Test logit: {output.item():.3f}")
        print(f"✓ Test probability: {prob.item():.4f}")
        
        # Test with multiple inputs
        batch_input = torch.randn(4, 2, 33, 33, device=device)
        batch_output = model(batch_input)
        batch_probs = torch.sigmoid(batch_output)
        
        print(f"✓ Batch input shape: {batch_input.shape}")
        print(f"✓ Batch output shape: {batch_output.shape}")
        print(f"✓ Batch probabilities: {batch_probs.cpu().numpy()}")
        
        print("✓ Weights test passed!\n")
        return model
        
    except Exception as e:
        print(f"❌ Error loading weights: {e}\n")
        return None

def test_simulated_scenarios():
    """Test with simulated realistic scenarios."""
    print("=" * 60)
    print("Testing Simulated Scenarios")
    print("=" * 60)
    
    model = StandaloneStopDetector()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Scenario 1: Endpoint (should have high stop probability)
    # Simulate: bright curve ending, path mask reaching the end
    endpoint_img = torch.zeros(1, 2, 33, 33, device=device)
    endpoint_img[0, 0, 15:18, 15:18] = 0.8  # Bright curve in center
    endpoint_img[0, 1, :16, :] = 1.0  # Path mask covering most of the crop
    
    # Scenario 2: Midpoint (should have low stop probability)
    # Simulate: curve continuing, path mask in middle
    midpoint_img = torch.zeros(1, 2, 33, 33, device=device)
    midpoint_img[0, 0, 10:20, 10:20] = 0.6  # Curve continuing
    midpoint_img[0, 1, :10, :] = 1.0  # Path mask only in top half
    
    with torch.no_grad():
        endpoint_logit = model(endpoint_img)
        midpoint_logit = model(midpoint_img)
        
        endpoint_prob = torch.sigmoid(endpoint_logit)
        midpoint_prob = torch.sigmoid(midpoint_logit)
    
    print(f"Endpoint scenario:")
    print(f"  Logit: {endpoint_logit.item():.3f}")
    print(f"  Probability: {endpoint_prob.item():.4f}")
    
    print(f"\nMidpoint scenario:")
    print(f"  Logit: {midpoint_logit.item():.3f}")
    print(f"  Probability: {midpoint_prob.item():.4f}")
    
    print("\nNote: These are untrained model outputs.")
    print("After training, endpoint should have higher probability than midpoint.\n")

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Stop Module Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Architecture
    model, device = test_model_architecture()
    
    # Test 2: Weights loading
    weights_path = os.path.join(_script_dir, "weights", "stop_detector_v1.pth")
    trained_model = test_model_with_weights(weights_path)
    
    # Test 3: Simulated scenarios
    test_simulated_scenarios()
    
    print("=" * 60)
    print("All Tests Complete")
    print("=" * 60)
    
    if trained_model is None:
        print("\n⚠️  Model weights not found. To train the model:")
        print("   python src/train_standalone.py")
    else:
        print("\n✓ Model is ready to use!")

if __name__ == "__main__":
    main()
