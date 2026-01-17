#!/usr/bin/env python3
"""
Generate datasets sorted into bins with full control over Intensity, Contrast, and Inversion.
"""
import os
import sys
import cv2
import numpy as np
import argparse
import json
from tqdm import tqdm

# Import existing classes
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from train_deeper_model import CurveMakerFlexible, load_curve_config

def generate_dataset_bins(output_dir="data_bins", n_images=100, config_path="../config/curve_bins.json", force_invert=False):
    # Load the bins config
    config, _ = load_curve_config(config_path)
    stages = config.get('training_stages', [])
    
    h = config.get('image', {}).get('height', 128)
    w = config.get('image', {}).get('width', 128)

    print(f"Generating {n_images} images per bin into '{output_dir}'...")

    for stage in stages:
        bin_name = stage['name']
        bin_id = stage['stage_id']
        
        # Extract generation config
        gen_cfg = stage.get('curve_generation', {})
        
        save_path = os.path.join(output_dir, bin_name)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"  > Bin {bin_id}: {bin_name} | Intensity: {gen_cfg.get('min_intensity', 'Def')}-{gen_cfg.get('max_intensity', 'Def')}")
        
        # Initialize maker with specific seed for reproducibility
        curve_maker = CurveMakerFlexible(h=h, w=w, seed=bin_id*999, config=config)
        
        for i in tqdm(range(n_images)):
            # 1. Sample Randomized Parameters from Ranges
            
            # Curvature
            c_range = gen_cfg.get('curvature_range', [1.0, 1.0])
            curv = np.random.uniform(c_range[0], c_range[1])
            
            # Width
            w_range = gen_cfg.get('width_range', [3, 3])
            
            # Noise
            n_range = gen_cfg.get('noise_range', [0.0, 0.0]) 
            # If noise_range exists, use it, otherwise use noise_prob fixed value
            noise_val = np.random.uniform(n_range[0], n_range[1]) if 'noise_range' in gen_cfg else gen_cfg.get('noise_prob', 0.0)

            # Inversion (White Background vs Black Background)
            # If force_invert CLI arg is set, always invert. Otherwise check config.
            invert_p = 1.0 if force_invert else gen_cfg.get('invert_prob', 0.0)

            # 2. Generate
            img, mask, _ = curve_maker.sample_curve(
                width_range=tuple(w_range),
                curvature_factor=curv,
                
                # Topolgoy
                allow_self_cross=gen_cfg.get('allow_self_cross', False),
                self_cross_prob=gen_cfg.get('self_cross_prob', 0.0),
                branches=gen_cfg.get('branches', False),
                
                # Appearance & Intensity (THIS WAS MISSING BEFORE)
                min_intensity=gen_cfg.get('min_intensity', 0.6),
                max_intensity=gen_cfg.get('max_intensity', 1.0),
                background_intensity=gen_cfg.get('background_intensity', 0.0),
                
                # Gradients
                intensity_variation=gen_cfg.get('intensity_variation', 'none'),
                start_intensity=gen_cfg.get('start_intensity', None),
                end_intensity=gen_cfg.get('end_intensity', None),
                
                # Noise & Inversion
                noise_prob=noise_val,
                invert_prob=invert_p
            )

            # 3. Save
            # Scale 0.0-1.0 float to 0-255 uint8
            filename = f"{bin_name}_{i:05d}.png"
            cv2.imwrite(os.path.join(save_path, filename), (img * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data_generated")
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--config", type=str, default="../config/curve_bins.json")
    parser.add_argument("--invert", action="store_true", help="Force white background (Dark curves on White)")
    args = parser.parse_args()
    
    generate_dataset_bins(args.out, args.num, args.config, args.invert)