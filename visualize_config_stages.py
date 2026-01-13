#!/usr/bin/env python3
"""
Visualize sample images for each stage in a curve config file.
Shows what the model will see during training.
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add paths for imports
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "archive" / "Experiment3_Refine"))

from src.train import CurveMakerFlexible

def visualize_stages(config_path, samples_per_stage=6, seed=42):
    """Generate sample images for each stage."""
    
    # Load config
    config_path = Path(config_path)
    if not config_path.exists():
        # Try relative to script directory
        config_path = script_dir / config_path
    
    print(f"📂 Loading: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    stages = config.get('training_stages', [])
    n_stages = len(stages)
    print(f"✅ Found {n_stages} stages")
    
    # Image dimensions
    h = config.get('image', {}).get('height', 128)
    w = config.get('image', {}).get('width', 128)
    
    # Create figure - taller rows to fit stacked labels
    fig, axes = plt.subplots(n_stages, samples_per_stage, 
                             figsize=(samples_per_stage * 1.8, n_stages * 4))
    
    if n_stages == 1:
        axes = axes.reshape(1, -1)
    
    print(f"\n🎨 Generating {samples_per_stage} samples per stage...\n")
    
    for stage_idx, stage in enumerate(stages):
        stage_id = stage.get('stage_id', stage_idx + 1)
        stage_name = stage.get('name', f'Stage{stage_id}')
        cfg = stage.get('curve_generation', {})
        
        # Print stage info
        curv_range = cfg.get('curvature_range', None)
        curv_info = f"Range: {curv_range}" if curv_range else f"Fixed: {cfg.get('curvature_factor', 1.0)}"
        noise_range = cfg.get('noise_range', None)
        noise_info = f"Range: {noise_range}" if noise_range else f"Fixed: {cfg.get('noise_prob', 0):.2f}"
        bg_range = cfg.get('background_intensity_range', None)
        bg_info = f"Range: {bg_range}" if bg_range else f"Fixed: {cfg.get('background_intensity', 0) or 0:.2f}"
        min_int_range = cfg.get('min_intensity_range', None)
        max_int_range = cfg.get('max_intensity_range', None)
        if min_int_range or max_int_range:
            min_info = f"Range: {min_int_range}" if min_int_range else f"Fixed: {cfg.get('min_intensity', 0.6):.2f}"
            max_info = f"Range: {max_int_range}" if max_int_range else f"Fixed: {cfg.get('max_intensity', 1.0) or 1.0:.2f}"
            int_info = f"Min: {min_info}, Max: {max_info}"
        else:
            int_info = f"{cfg.get('min_intensity', 0.6):.2f}-{cfg.get('max_intensity', 1.0) or 1.0:.2f}"
        print(f"Stage {stage_id}: {stage_name}")
        print(f"   Curvature: {curv_info}")
        print(f"   Width: {cfg.get('width_range', [3,5])}, Intensity: {int_info}")
        print(f"   Noise: {noise_info}, BG: {bg_info}")
        
        for sample_idx in range(samples_per_stage):
            # Retry until we get valid intensity > background
            for attempt in range(100):
                # Use different seeds for each attempt
                curve_seed = seed + stage_id * 1000 + sample_idx + attempt * 10000
                curv_seed = seed * 7 + stage_id * 100 + sample_idx * 13 + attempt * 777
                
                curve_maker = CurveMakerFlexible(h=h, w=w, seed=curve_seed, config=config)
                
                # Sample from ranges if available (with its own seed)
                np.random.seed(curv_seed)
                
                # Curvature range
                if 'curvature_range' in cfg:
                    curv = np.random.uniform(cfg['curvature_range'][0], cfg['curvature_range'][1])
                else:
                    curv = cfg.get('curvature_factor', 1.0)
                
                # Noise range
                if 'noise_range' in cfg:
                    noise_prob = np.random.uniform(cfg['noise_range'][0], cfg['noise_range'][1])
                else:
                    noise_prob = cfg.get('noise_prob', 0.0)
                
                # Background intensity range
                if 'background_intensity_range' in cfg:
                    bg_intensity = np.random.uniform(cfg['background_intensity_range'][0], cfg['background_intensity_range'][1])
                else:
                    bg_intensity = cfg.get('background_intensity', None) or 0.0
                
                # Get intensity bounds
                if 'min_intensity_range' in cfg:
                    min_int_bound = np.random.uniform(cfg['min_intensity_range'][0], cfg['min_intensity_range'][1])
                else:
                    min_int_bound = cfg.get('min_intensity', 0.6)
                
                if 'max_intensity_range' in cfg:
                    max_int_bound = np.random.uniform(cfg['max_intensity_range'][0], cfg['max_intensity_range'][1])
                    max_int_bound = max(max_int_bound, min_int_bound + 0.05)
                else:
                    max_int_bound = cfg.get('max_intensity', 1.0) or 1.0
                
                # Sample a single intensity value
                actual_intensity = np.random.uniform(min_int_bound, max_int_bound)
                
                # Ensure intensity > background (curve must be visible)
                bg_val = bg_intensity if bg_intensity is not None else 0.0
                if actual_intensity > bg_val + 0.05:
                    break  # Valid combination found
            
            # Generate image
            try:
                img, _, _ = curve_maker.sample_curve(
                    width_range=tuple(cfg.get('width_range', [3, 5])),
                    noise_prob=noise_prob,
                    invert_prob=cfg.get('invert_prob', 0.5),
                    min_intensity=actual_intensity,
                    max_intensity=actual_intensity,
                    branches=cfg.get('branches', False),
                    curvature_factor=curv,
                    allow_self_cross=cfg.get('allow_self_cross', False),
                    self_cross_prob=cfg.get('self_cross_prob', 0.0),
                    width_variation=cfg.get('width_variation', 'none'),
                    intensity_variation=cfg.get('intensity_variation', 'none'),
                    background_intensity=bg_intensity
                )
            except Exception as e:
                print(f"   ⚠️ Error generating sample: {e}")
                img = np.zeros((h, w))
            
            # Display
            ax = axes[stage_idx, sample_idx]
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            
            # Title for first column only
            if sample_idx == 0:
                ax.set_ylabel(f"S{stage_id}", fontsize=10, rotation=0, ha='right', va='center')
            
            # Show all sampled parameters - stacked on separate lines
            bg_str = f"{bg_intensity:.2f}" if bg_intensity is not None else "0"
            title = f"c={curv:.1f}\ni={actual_intensity:.2f}\nn={noise_prob:.2f}\nbg={bg_str}"
            ax.set_title(title, fontsize=5, pad=2, linespacing=0.9)
        
        print()
    
    # Main title
    config_name = config_path.stem
    fig.suptitle(f'Stage Samples: {config_name}', fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=0.15)
    
    # Save
    output_path = script_dir / f'{config_name}_samples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    plt.show()
    print("\n✅ Done!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize curve config stages")
    parser.add_argument('config', nargs='?',
                       default='archive/Experiment3_Refine/config/curve_config.json',
                       help='Path to config file (default: archive/Experiment3_Refine/config/curve_config.json)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Samples per stage (default: 6)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    visualize_stages(args.config, samples_per_stage=args.samples, seed=args.seed)

