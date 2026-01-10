#!/usr/bin/env python3
"""
Example usage of the Curve Generator Module.

This demonstrates how to:
1. Load configuration from JSON
2. Create a curve generator
3. Generate curves
4. Save config snapshots
"""
import sys
import os
import numpy as np
import cv2

# Add parent directory to path to import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CurveGeneratorModule import load_curve_config, CurveMakerFlexible, CurveMakerSixPoint, save_config_snapshot


def main():
    # Example 1: Load config (will search default locations if None)
    # You can also provide an explicit path: load_curve_config("path/to/config.json")
    config, config_path = load_curve_config()
    
    if config_path:
        print(f"Loaded config from: {config_path}")
    else:
        print("No config found, using defaults")
    
    # Example 2: Create curve generator
    img_cfg = config.get('image', {})
    h = img_cfg.get('height', 128)
    w = img_cfg.get('width', 128)
    
    generator = CurveMakerFlexible(h=h, w=w, seed=42, config=config)
    
    # Example 3: Generate a curve with default parameters
    print("\nGenerating a simple curve...")
    img, mask, pts_all = generator.sample_curve(
        width_range=(2, 4),
        noise_prob=0.0,
        invert_prob=0.5,
        min_intensity=0.6,
        max_intensity=0.8,
        branches=False,
        curvature_factor=1.0
    )
    
    # Save the generated curve
    output_dir = "example_output"
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "example_curve.png"), (img * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, "example_mask.png"), (mask * 255).astype(np.uint8))
    print(f"Saved example curve to {output_dir}/")
    
    # Example 4: Generate curve using stage config from JSON
    training_stages = config.get('training_stages', [])
    if training_stages:
        print("\nGenerating curve using stage 1 config...")
        stage1 = training_stages[0]
        cg = stage1.get('curve_generation', {})
        
        img2, mask2, pts_all2 = generator.sample_curve(
            width_range=tuple(cg.get("width_range", [2, 4])),
            noise_prob=cg.get("noise_prob", 0.0),
            invert_prob=cg.get("invert_prob", 0.5),
            min_intensity=cg.get("min_intensity", 0.6),
            max_intensity=cg.get("max_intensity", 0.8),
            branches=cg.get("branches", False),
            curvature_factor=cg.get("curvature_factor", 1.0),
            allow_self_cross=cg.get("allow_self_cross", False),
            self_cross_prob=cg.get("self_cross_prob", 0.0),
            width_variation=cg.get("width_variation", "none"),
            start_width=cg.get("start_width", None),
            end_width=cg.get("end_width", None),
            intensity_variation=cg.get("intensity_variation", "none"),
            start_intensity=cg.get("start_intensity", None),
            end_intensity=cg.get("end_intensity", None),
            background_intensity=cg.get("background_intensity", None),
        )
        
        cv2.imwrite(os.path.join(output_dir, "stage1_curve.png"), (img2 * 255).astype(np.uint8))
        print(f"Saved stage 1 curve to {output_dir}/")
    
    # Example 5: Generate six-point curve (more complex curves through 6 points)
    print("\nGenerating six-point curve...")
    sixpoint_generator = CurveMakerSixPoint(h=h, w=w, seed=42, config=config)
    img3, mask3, pts_all3 = sixpoint_generator.sample_curve(
        width_range=(2, 4),
        noise_prob=0.0,
        invert_prob=0.0,
        min_intensity=0.6,
        max_intensity=0.8,
        branches=False,
        curvature_factor=1.0
    )
    cv2.imwrite(os.path.join(output_dir, "sixpoint_curve.png"), (img3 * 255).astype(np.uint8))
    print(f"Saved six-point curve to {output_dir}/")
    print(f"  Curve passes through 6 points, total points: {len(pts_all3[0])}")
    
    # Example 6: Save config snapshot (for reproducibility)
    if config_path:
        snapshot_path = os.path.join(output_dir, "curve_config_snapshot.json")
        save_config_snapshot(config, snapshot_path)
        print(f"\nSaved config snapshot to: {snapshot_path}")
    
    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()

