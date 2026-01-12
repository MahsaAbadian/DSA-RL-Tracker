#!/usr/bin/env python3
"""
Visual Sampler for Curve Configurations.
Generates 10x7 grids (35 image-mask pairs) for each stage in a config.
"""
import os
import sys
import json
import numpy as np
import cv2
import argparse
from pathlib import Path

# Add project root to path to import CurveGeneratorModule
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from CurveGeneratorModule import load_curve_config, CurveMaker

def create_paired_grid(images, masks, cols=7, rows=10):
    """
    Creates a grid where each image is followed by its mask.
    A 10x7 grid means 70 slots total -> 35 pairs.
    """
    if not images or not masks:
        return None
    
    h, w = images[0].shape[:2]
    # Create a grid with some padding between pairs if needed, but let's keep it tight
    grid = np.zeros((h * rows, w * cols), dtype=np.float32)
    
    for idx in range(min(len(images), (rows * cols) // 2)):
        # Pair index i -> slots 2*i and 2*i + 1
        img_slot = 2 * idx
        mask_slot = 2 * idx + 1
        
        # Image position
        r_i, c_i = divmod(img_slot, cols)
        grid[r_i*h:(r_i+1)*h, c_i*w:(c_i+1)*w] = images[idx]
        
        # Mask position
        r_m, c_m = divmod(mask_slot, cols)
        grid[r_m*h:(r_m+1)*h, c_m*w:(c_m+1)*w] = masks[idx]
        
    return grid

def main():
    # Use a directory relative to this script for default output
    script_dir = Path(__file__).parent.resolve()
    default_out = script_dir / "sample_curve"

    parser = argparse.ArgumentParser(description="Generate visual samples for a curve config")
    parser.add_argument("config_path", type=str, help="Path to the config JSON file")
    parser.add_argument("--output_root", type=str, default=str(default_out), help="Root folder for samples")
    args = parser.parse_args()

    # 1. Load Config
    config_path = Path(args.config_path).resolve()
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
        
    config, _ = load_curve_config(str(config_path))
    
    # 2. Determine Output Directory
    # Name: <experiment_folder>_<config_name>
    exp_folder = config_path.parent.parent.name # e.g. Experiment4_separate_stop_v2
    config_name = config_path.stem # e.g. curve_config
    folder_name = f"{exp_folder}_{config_name}"
    out_dir = Path(args.output_root) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating samples in: {out_dir}")
    
    # 3. Save Config Snapshot
    with open(out_dir / "config_snapshot.json", 'w') as f:
        json.dump(config, f, indent=2)

    # 4. Generate Grids for each Stage
    stages = config.get('training_stages', [])
    if not stages:
        print("No training stages found in config.")
        return

    h = config.get('image', {}).get('height', 128)
    w = config.get('image', {}).get('width', 128)
    
    generator = CurveMaker(h=h, w=w, config=config)

    for stage in stages:
        s_id = stage.get('stage_id', 'X')
        s_name = stage.get('name', f"Stage{s_id}")
        cg = stage.get('curve_generation', {})
        
        print(f"Processing Stage {s_id}: {s_name}...")
        
        images = []
        masks = []
        
        # 35 pairs for a 10x7 grid
        for i in range(35):
            params = {
                "width_range": tuple(cg.get("width_range", [2, 4])),
                "noise_prob": cg.get("noise_prob", 0.0),
                "tissue_noise_prob": cg.get("tissue_noise_prob", 0.0),
                "invert_prob": cg.get("invert_prob", 0.5),
                "min_intensity": cg.get("min_intensity", 0.6),
                "max_intensity": cg.get("max_intensity", 1.0),
                "branches": cg.get("branches", False),
                "curvature_factor": cg.get("curvature_factor", 1.0),
                "num_control_points": cg.get("num_control_points", 5),
                "num_segments": cg.get("num_segments", 1),
                "centerline_mask": True # Assuming experiment preference for centerline
            }
            # Add extra params from config if present
            for key in ['allow_self_cross', 'self_cross_prob', 'width_variation', 'intensity_variation']:
                if key in cg: params[key] = cg[key]

            img, mask, _ = generator.sample_curve(**params)
            images.append(img)
            masks.append(mask.astype(np.float32))

        grid = create_paired_grid(images, masks, cols=7, rows=10)
        if grid is not None:
            save_path = out_dir / f"stage_{s_id}_{s_name.replace(' ', '_')}.png"
            cv2.imwrite(str(save_path), (grid * 255).astype(np.uint8))

    print(f"Done! Samples generated for {len(stages)} stages.")

if __name__ == "__main__":
    main()
