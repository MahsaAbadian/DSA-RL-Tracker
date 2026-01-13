#!/usr/bin/env python3
"""
Visualize sample curves for each training stage in Experiment3_Refine.
Shows what the model sees during training for each stage.
"""
import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths for imports - CurveGeneratorModule is in the main project directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_script_dir))))
_curve_gen_path = os.path.join(_project_root, "CurveGeneratorModule")
if _curve_gen_path not in sys.path:
    sys.path.insert(0, _curve_gen_path)

from generator import CurveMaker
from train_deeper_model import load_curve_config

def format_params(cfg):
    """Format curve generation parameters for display."""
    params = []
    # Width
    if 'width_range' in cfg: params.append(f"W:{cfg['width_range']}")
    
    # Intensity
    min_i = cfg.get('min_intensity_range', cfg.get('min_intensity', 0))
    max_i = cfg.get('max_intensity_range', cfg.get('max_intensity', 1))
    params.append(f"Int:{min_i}-{max_i}")
    
    # Topology/Curvature
    params.append(f"Top:{cfg.get('topology', 'random')}")
    
    if 'curvature_range' in cfg:
        params.append(f"Curv:{cfg['curvature_range']}")
    else:
        params.append(f"Curv:{cfg.get('curvature_factor', 1.0):.1f}")
    
    # Noise
    params.append(f"Noise:{cfg.get('noise_prob', 0.0):.1f}")
    params.append(f"Tiss:{cfg.get('tissue_noise_prob', 0.0):.1f}")
    
    if cfg.get('background_intensity') is not None or 'background_intensity_range' in cfg:
        bg = cfg.get('background_intensity_range', cfg.get('background_intensity'))
        params.append(f"BG:{bg}")
    
    return ", ".join(params)

def _sample_val(cfg, key, range_key=None):
    """Helper to sample a value if it's a range in the config, or return the fixed value."""
    if range_key and range_key in cfg:
        r = cfg[range_key]
        return np.random.uniform(r[0], r[1])
    return cfg.get(key)

def visualize_stage_samples(config_path, samples_per_stage=6, seed=42, save_dir=None):
    """
    Generate and visualize sample curves for each training stage.
    """
    # Load config
    curve_config, config_file_path = load_curve_config(config_path)
    if curve_config is None:
        print(f"Error: Could not load config from {config_path}")
        return
    
    stages = curve_config.get('training_stages', [])
    if not stages:
        print("Error: No training stages found in config")
        return
    
    print(f"✅ Loaded config: {config_file_path}")
    print(f"✅ Found {len(stages)} training stages")
    print(f"✅ Generating {samples_per_stage} samples per stage\n")
    
    # Create curve generator
    h = curve_config.get('image', {}).get('height', 128)
    w = curve_config.get('image', {}).get('width', 128)
    
    # Visualize each stage
    for stage_idx, stage in enumerate(stages):
        stage_id = stage.get('stage_id', stage_idx + 1)
        stage_name = stage.get('name', f'Stage{stage_id}')
        curve_cfg = stage.get('curve_generation', {})
        
        print(f"📊 Stage {stage_id}: {stage_name}")
        print(f"   {format_params(curve_cfg)}")
        
        # Create figure for this stage
        n_cols = 3
        n_rows = (samples_per_stage + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # Generate samples
        for sample_idx in range(samples_per_stage):
            # Use different seed for each sample
            sample_seed = seed + stage_id * 1000 + sample_idx
            
            # Re-init CurveMaker to handle seed properly for ranges
            curve_maker = CurveMaker(h=h, w=w, seed=sample_seed, config=curve_config)
            
            # --- Extract Parameters (Handling Ranges for Grandmaster Stage) ---
            width_range = tuple(curve_cfg.get('width_range', [3, 5]))
            
            # Sample probabilistic parameters if ranges exist (for Stage 5)
            noise_prob = _sample_val(curve_cfg, 'noise_prob', 'noise_range') or 0.0
            min_intensity = _sample_val(curve_cfg, 'min_intensity', 'min_intensity_range') or 0.6
            max_intensity = _sample_val(curve_cfg, 'max_intensity', 'max_intensity_range')
            background_intensity = _sample_val(curve_cfg, 'background_intensity', 'background_intensity_range') or 0.0
            curvature_factor = _sample_val(curve_cfg, 'curvature_factor', 'curvature_range') or 1.0
            
            # Fixed or Probability parameters
            tissue_noise_prob = curve_cfg.get('tissue_noise_prob', 0.0)
            invert_prob = curve_cfg.get('invert_prob', 0.0)
            branches = curve_cfg.get('branches', False)
            allow_self_cross = curve_cfg.get('allow_self_cross', False)
            self_cross_prob = curve_cfg.get('self_cross_prob', 0.0)
            
            # Structural parameters
            topology = curve_cfg.get('topology', 'random')
            num_segments = curve_cfg.get('num_segments')
            num_control_points = curve_cfg.get('num_control_points')
            
            # Variation parameters
            width_variation = curve_cfg.get('width_variation', 'none')
            intensity_variation = curve_cfg.get('intensity_variation', 'none')

            # Generate curve
            img, mask, pts_all = curve_maker.sample_curve(
                width_range=width_range,
                noise_prob=noise_prob,
                tissue_noise_prob=tissue_noise_prob,  # PASSED CORRECTLY NOW
                invert_prob=invert_prob,
                min_intensity=min_intensity,
                max_intensity=max_intensity,
                background_intensity=background_intensity, # ADDED
                branches=branches,
                curvature_factor=curvature_factor,
                topology=topology,                         # ADDED
                num_segments=num_segments,                 # ADDED
                num_control_points=num_control_points,     # ADDED
                width_variation=width_variation,           # ADDED
                intensity_variation=intensity_variation,   # ADDED
                allow_self_cross=allow_self_cross,
                self_cross_prob=self_cross_prob
            )
            
            # Display
            ax = axes[sample_idx]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Sample {sample_idx + 1}', fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for idx in range(samples_per_stage, len(axes)):
            axes[idx].axis('off')
        
        # Add main title
        fig.suptitle(
            f'Stage {stage_id}: {stage_name}\n{format_params(curve_cfg)}',
            fontsize=14,
            y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save or show
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'stage_{stage_id:02d}_{stage_name}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   💾 Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
        print()
    
    print("✅ Visualization complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize sample curves for each training stage"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../config/curve_config.json',
        help='Path to curve_config.json'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=6,
        help='Number of samples per stage (default: 6)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        default=None,
        help='Directory to save images (default: None, displays instead)'
    )
    args = parser.parse_args()
    
    visualize_stage_samples(
        config_path=args.config,
        samples_per_stage=args.samples,
        seed=args.seed,
        save_dir=args.save_dir
    )

if __name__ == '__main__':
    main()