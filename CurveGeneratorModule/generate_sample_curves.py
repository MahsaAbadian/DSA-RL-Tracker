#!/usr/bin/env python3
"""
Generate sample curves to visualize the different generators.
Creates a grid showing single-segment, 2-segment multi-segment, and 6-point curves.
"""
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CurveGeneratorModule import (
    CurveMakerFlexible, 
    CurveMakerMultiSegment, 
    CurveMakerSixPoint,
    CenterlineMask5PointsGenerator
)


def create_grid(images, cols=5, rows=5):
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


def main():
    output_dir = "sample_curves_output"
    os.makedirs(output_dir, exist_ok=True)
    
    h, w = 128, 128
    seed = 42
    num_samples = 25  # 5x5 grid
    
    print("Generating sample curves...")
    print(f"Output directory: {output_dir}")
    
    # 1. Single-segment curves (4-point Bezier)
    print("\n1. Generating single-segment curves (4-point Bezier)...")
    single_gen = CurveMakerFlexible(h=h, w=w, seed=seed)
    single_images = []
    single_masks = []
    for i in range(num_samples):
        img, mask, pts = single_gen.sample_curve(
            width_range=(2, 3),
            min_intensity=0.7,
            max_intensity=0.9,
            curvature_factor=1.0
        )
        single_images.append(img)
        single_masks.append(mask.astype(np.float32))
    
    single_grid = create_grid(single_images, cols=5, rows=5)
    single_mask_grid = create_grid(single_masks, cols=5, rows=5)
    if single_grid is not None:
        cv2.imwrite(os.path.join(output_dir, "single_segment_4point.png"), 
                   (single_grid * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, "single_segment_4point_mask.png"), 
                   (single_mask_grid * 255).astype(np.uint8))
        print(f"  ✓ Saved: {output_dir}/single_segment_4point.png")
        print(f"  ✓ Saved: {output_dir}/single_segment_4point_mask.png")
    
    # 2. Multi-segment curves (2 segments)
    print("\n2. Generating multi-segment curves (2 segments)...")
    multi_gen = CurveMakerMultiSegment(h=h, w=w, seed=seed)
    multi_images = []
    multi_masks = []
    for i in range(num_samples):
        img, mask, pts = multi_gen.sample_curve(
            width_range=(2, 3),
            min_intensity=0.7,
            max_intensity=0.9,
            curvature_factor=1.0,
            num_segments=2  # Fixed to 2 segments
        )
        multi_images.append(img)
        multi_masks.append(mask.astype(np.float32))
    
    multi_grid = create_grid(multi_images, cols=5, rows=5)
    multi_mask_grid = create_grid(multi_masks, cols=5, rows=5)
    if multi_grid is not None:
        cv2.imwrite(os.path.join(output_dir, "multi_segment_2segments.png"), 
                   (multi_grid * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, "multi_segment_2segments_mask.png"), 
                   (multi_mask_grid * 255).astype(np.uint8))
        print(f"  ✓ Saved: {output_dir}/multi_segment_2segments.png")
        print(f"  ✓ Saved: {output_dir}/multi_segment_2segments_mask.png")
    
    # 3. Six-point curves (degree-5 Bezier)
    print("\n3. Generating six-point curves (degree-5 Bezier)...")
    sixpoint_gen = CurveMakerSixPoint(h=h, w=w, seed=seed)
    sixpoint_images = []
    sixpoint_masks = []
    for i in range(num_samples):
        img, mask, pts = sixpoint_gen.sample_curve(
            width_range=(2, 3),
            min_intensity=0.7,
            max_intensity=0.9,
            curvature_factor=1.0
        )
        sixpoint_images.append(img)
        sixpoint_masks.append(mask.astype(np.float32))
    
    sixpoint_grid = create_grid(sixpoint_images, cols=5, rows=5)
    sixpoint_mask_grid = create_grid(sixpoint_masks, cols=5, rows=5)
    if sixpoint_grid is not None:
        cv2.imwrite(os.path.join(output_dir, "six_point_degree5.png"), 
                   (sixpoint_grid * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, "six_point_degree5_mask.png"), 
                   (sixpoint_mask_grid * 255).astype(np.uint8))
        print(f"  ✓ Saved: {output_dir}/six_point_degree5.png")
        print(f"  ✓ Saved: {output_dir}/six_point_degree5_mask.png")

    # 4. Centerline 5-point curves (degree-4 Bezier + 1px mask)
    print("\n4. Generating centerline 5-point curves (degree-4 Bezier)...")
    centerline_gen = CenterlineMask5PointsGenerator(h=h, w=w, seed=seed)
    centerline_images = []
    centerline_masks = []
    for i in range(num_samples):
        # Using a slightly thicker width to emphasize the centerline mask
        img, mask, pts = centerline_gen.sample_curve(
            width_range=(4, 6),
            min_intensity=0.7,
            max_intensity=0.9,
            curvature_factor=1.0
        )
        centerline_images.append(img)
        centerline_masks.append(mask.astype(np.float32))
    
    centerline_grid = create_grid(centerline_images, cols=5, rows=5)
    centerline_mask_grid = create_grid(centerline_masks, cols=5, rows=5)
    
    if centerline_grid is not None:
        cv2.imwrite(os.path.join(output_dir, "centerline_5point_img.png"), 
                   (centerline_grid * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, "centerline_5point_mask.png"), 
                   (centerline_mask_grid * 255).astype(np.uint8))
        print(f"  ✓ Saved: {output_dir}/centerline_5point_img.png")
        print(f"  ✓ Saved: {output_dir}/centerline_5point_mask.png")
    
    # Create comparison grid (side by side)
    print("\n5. Creating comparison grid...")
    # Add labels row
    label_h = 30
    comparison_w = w * 5 * 4
    comparison_h = h * 5 + label_h
    
    # Simple stacking
    comparison = np.hstack([
        single_grid if single_grid is not None else np.zeros((h*5, w*5)),
        multi_grid if multi_grid is not None else np.zeros((h*5, w*5)),
        sixpoint_grid if sixpoint_grid is not None else np.zeros((h*5, w*5)),
        centerline_grid if centerline_grid is not None else np.zeros((h*5, w*5))
    ])
    
    # Also create a mask comparison specifically to show the centerline difference
    print("   Creating mask comparison...")
    
    mask_comparison = np.hstack([
        single_mask_grid if single_mask_grid is not None else np.zeros((h*5, w*5)),
        multi_mask_grid if multi_mask_grid is not None else np.zeros((h*5, w*5)),
        sixpoint_mask_grid if sixpoint_mask_grid is not None else np.zeros((h*5, w*5)),
        centerline_mask_grid if centerline_mask_grid is not None else np.zeros((h*5, w*5))
    ])
    
    # Final mega-comparison: Images on top, Masks on bottom
    mega_comparison = np.vstack([comparison, mask_comparison])
    
    cv2.imwrite(os.path.join(output_dir, "comparison_all_types.png"), 
               (mega_comparison * 255).astype(np.uint8))
    print(f"  ✓ Saved: {output_dir}/comparison_all_types.png")
    
    print(f"\n✅ Done! Generated {num_samples} curves of each type.")
    print(f"   Check {output_dir}/ for all output images.")


if __name__ == "__main__":
    main()

