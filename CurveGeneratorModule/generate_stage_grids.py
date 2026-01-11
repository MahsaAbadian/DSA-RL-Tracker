#!/usr/bin/env python3
"""
Generate 7x7 grids (49 curves) for each stage configuration in a curve config JSON.

This helps visualize what each training stage produces.
"""
import sys
import os
import json
import numpy as np
import cv2
import argparse
from pathlib import Path

# Add current directory to path to import the module
script_dir = os.path.dirname(os.path.abspath(__file__))
module_dir = script_dir  # Script is in CurveGeneratorModule directory
# Add parent to import CurveGeneratorModule as a package
sys.path.insert(0, os.path.dirname(module_dir))

from CurveGeneratorModule import (
    load_curve_config, 
    CurveMakerFlexible, 
    CurveMakerMultiSegment, 
    CurveMakerSixPoint,
    CenterlineMask5PointsGenerator
)


def create_grid(images, cols=7, rows=7):
    """Create a grid image from a list of images."""
    if not images:
        return None
    
    h, w = images[0].shape[:2]
    grid = np.zeros((h * rows, w * cols), dtype=np.float32)
    
    for idx, img in enumerate(images[:rows * cols]):
        row = idx // cols
        col = idx % cols
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
    return grid


def generate_stage_grids(config_path, output_dir, grid_size=7, seed=42, 
                         use_multi_segment=False, use_six_point=False, 
                         use_centerline_5point=False,
                         num_segments=None, segment_length_factor=1.0):
    """Generate 7x7 grids for each stage in the config.
    
    Args:
        config_path: Path to curve config JSON file
        output_dir: Directory to save output grids
        grid_size: Size of grid (grid_size x grid_size = total curves)
        seed: Base seed for reproducibility
    """
    # Load config
    config, actual_config_path = load_curve_config(config_path)
    
    if not config:
        print(f"❌ Failed to load config from {config_path}")
        return
    
    print(f"✓ Loaded config from: {actual_config_path}")
    
    # Get image dimensions
    img_cfg = config.get('image', {})
    h = img_cfg.get('height', 128)
    w = img_cfg.get('width', 128)
    
    # Get training stages
    training_stages = config.get('training_stages', [])
    if not training_stages:
        print("⚠️  No training_stages found in config")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy config to output directory for reference
    config_copy_path = os.path.join(output_dir, "curve_config.json")
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved config copy to: {config_copy_path}")
    
    # Generate grids for each stage
    total_curves = grid_size * grid_size
    
    for stage in training_stages:
        stage_id = stage.get('stage_id', 'X')
        stage_name = stage.get('name', f"Stage{stage_id}")
        cg = stage.get('curve_generation', {})
        
        print(f"\n📊 Generating {total_curves} curves for {stage_name}...")
        
        # Create stage directory
        stage_dir = os.path.join(output_dir, f"stage{stage_id}_{stage_name.replace(' ', '_')}")
        os.makedirs(stage_dir, exist_ok=True)
        
        # Create generator with unique seed for this stage
        if use_six_point:
            generator = CurveMakerSixPoint(h=h, w=w, seed=seed + int(stage_id) * 1000, config=config)
        elif use_multi_segment:
            generator = CurveMakerMultiSegment(h=h, w=w, seed=seed + int(stage_id) * 1000, config=config)
        elif use_centerline_5point:
            generator = CenterlineMask5PointsGenerator(h=h, w=w, seed=seed + int(stage_id) * 1000, config=config)
        else:
            generator = CurveMakerFlexible(h=h, w=w, seed=seed + int(stage_id) * 1000, config=config)
        
        # Generate curves
        images = []
        masks = []
        
        for i in range(total_curves):
            # Base parameters
            curve_params = {
                "width_range": tuple(cg.get("width_range", [2, 4])),
                "noise_prob": cg.get("noise_prob", 0.0),
                "invert_prob": cg.get("invert_prob", 0.5),
                "min_intensity": cg.get("min_intensity", 0.6),
                "max_intensity": cg.get("max_intensity", 0.8),
                "branches": cg.get("branches", False),
                "curvature_factor": cg.get("curvature_factor", 1.0),
                "allow_self_cross": cg.get("allow_self_cross", False),
                "self_cross_prob": cg.get("self_cross_prob", 0.0),
                "width_variation": cg.get("width_variation", "none"),
                "start_width": cg.get("start_width", None),
                "end_width": cg.get("end_width", None),
                "intensity_variation": cg.get("intensity_variation", "none"),
                "start_intensity": cg.get("start_intensity", None),
                "end_intensity": cg.get("end_intensity", None),
                "background_intensity": cg.get("background_intensity", None),
            }
            
            # Add multi-segment parameters if using multi-segment generator
            if use_multi_segment:
                # Default to 2 segments if not specified
                curve_params["num_segments"] = num_segments if num_segments is not None else cg.get("num_segments", 2)
                curve_params["segment_length_factor"] = cg.get("segment_length_factor", segment_length_factor)
            
            img, mask, pts_all = generator.sample_curve(**curve_params)
            
            images.append(img)
            masks.append(mask)
        
        # Create and save grid
        grid = create_grid(images, cols=grid_size, rows=grid_size)
        if grid is not None:
            grid_path = os.path.join(stage_dir, "grid_7x7.png")
            cv2.imwrite(grid_path, (grid * 255).astype(np.uint8))
            print(f"  ✓ Saved grid to: {grid_path}")
        
        # Create mask grid
        mask_grid = create_grid(masks, cols=grid_size, rows=grid_size)
        if mask_grid is not None:
            mask_grid_path = os.path.join(stage_dir, "mask_grid_7x7.png")
            cv2.imwrite(mask_grid_path, (mask_grid * 255).astype(np.uint8))
            print(f"  ✓ Saved mask grid to: {mask_grid_path}")
        
        # Create comparison grid (side-by-side)
        if grid is not None and mask_grid is not None:
            comparison_grid = np.hstack([grid, mask_grid])
            comp_path = os.path.join(stage_dir, "comparison_7x7.png")
            cv2.imwrite(comp_path, (comparison_grid * 255).astype(np.uint8))
            print(f"  ✓ Saved comparison grid to: {comp_path}")
        
        # Save stage info
        stage_info = {
            "stage_id": stage_id,
            "stage_name": stage_name,
            "curve_generation": cg,
            "num_curves": total_curves,
            "grid_size": f"{grid_size}x{grid_size}",
        }
        info_path = os.path.join(stage_dir, "stage_info.json")
        with open(info_path, 'w') as f:
            json.dump(stage_info, f, indent=2)
    
    print(f"\n✅ Done! Generated grids for {len(training_stages)} stages")
    print(f"   Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate 7x7 curve grids for each stage")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to curve config JSON file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: ExampleConfigs/outputs/<config_name>)")
    parser.add_argument("--grid-size", type=int, default=7,
                        help="Grid size (default: 7, creates 7x7=49 curves)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--multi-segment", action="store_true",
                        help="Use multi-segment Bezier curve generator (longer curves)")
    parser.add_argument("--six-point", action="store_true",
                        help="Use six-point curve generator (more complex curves through 6 points)")
    parser.add_argument("--centerline-5point", action="store_true",
                        help="Use 5-point Bezier generator with mandatory 1px centerline mask")
    parser.add_argument("--num-segments", type=int, default=None,
                        help="Number of segments for multi-segment curves (default: random from config or 2-4)")
    parser.add_argument("--segment-length-factor", type=float, default=1.0,
                        help="Factor to control segment length (default: 1.0, >1.0 = longer segments)")
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output is None:
        config_name = Path(args.config).stem
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, "outputs", config_name)
    
    generate_stage_grids(
        config_path=args.config,
        output_dir=args.output,
        grid_size=args.grid_size,
        seed=args.seed,
        use_multi_segment=args.multi_segment,
        use_six_point=args.six_point,
        use_centerline_5point=args.centerline_5point,
        num_segments=args.num_segments,
        segment_length_factor=args.segment_length_factor
    )


if __name__ == "__main__":
    main()

