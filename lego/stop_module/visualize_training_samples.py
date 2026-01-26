#!/usr/bin/env python3
"""
Visualize training samples for the stop module.
Shows what the model sees: positive samples (endpoints) vs negative samples (midpoints).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add paths for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
stop_module_root = _script_dir
lego_root = os.path.dirname(stop_module_root)
repo_root = os.path.dirname(lego_root)
DEFAULT_STOP_CURVE_CONFIG = os.path.join(
    lego_root, "curve_generator_module", "config", "stop_curve_config.json"
)

for _path in (stop_module_root, lego_root, repo_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from curve_generator_module import CurveMaker, load_curve_config
from src.train_standalone import crop32, CROP

def generate_samples(num_samples=100, vessel_realism=True):
    """Generate positive and negative samples like the training does."""
    if os.path.exists(DEFAULT_STOP_CURVE_CONFIG):
        cfg = load_curve_config(DEFAULT_STOP_CURVE_CONFIG)[0]
    else:
        cfg = load_curve_config()[0]
    maker = CurveMaker(h=128, w=128, config=cfg)
    
    positive_samples = []  # Endpoints (label=1.0)
    negative_samples = []  # Midpoints (label=0.0)
    
    print(f"Generating {num_samples} curves...")
    for i in range(num_samples):
        # Same curve generation logic as training
        w_ranges = [(1, 3), (3, 5), (1, 5), (2, 6)]
        w_range = w_ranges[np.random.randint(len(w_ranges))]
        
        # HARDER SAMPLES: Match training logic
        curvature = np.random.uniform(0.3, 2.5)
        noise_prob = np.random.uniform(0.0, 0.2)  # Lower noise for cleaner samples
        
        # Random background color from generator: black or white
        # (Gray backgrounds have poor contrast, so we skip them)
        bg_choice = np.random.randint(0, 2)
        if bg_choice == 0:
            # Black background (draw bright curves)
            background_intensity = 0.0
            invert_prob = 0.0
        else:
            # White background (draw dark curves via inversion)
            background_intensity = 0.0
            invert_prob = 1.0
        
        if vessel_realism:
            width_variation = "wide_to_narrow"
            intensity_variation = "bright_to_dim"
            
            is_hard_endpoint = np.random.random() < 0.3
            if is_hard_endpoint:
                start_width = np.random.randint(2, 4)
                end_width = 1
                start_intensity = np.random.uniform(0.5, 0.7)
                end_intensity = np.random.uniform(0.1, 0.25)
            else:
                start_width = np.random.randint(w_range[1], w_range[1] + 3)
                end_width = np.random.randint(1, max(2, w_range[0]))
                start_intensity = np.random.uniform(0.7, 1.0)
                end_intensity = np.random.uniform(0.2, 0.5)
        else:
            width_variation = "none"
            intensity_variation = "none"
            start_width = None
            end_width = None
            start_intensity = None
            end_intensity = None
        
        img, mask, pts_all = maker.sample_curve(
            width_range=w_range,
            curvature_factor=curvature,
            noise_prob=noise_prob,
            invert_prob=invert_prob,
            background_intensity=background_intensity,
            width_variation=width_variation,
            start_width=start_width,
            end_width=end_width,
            intensity_variation=intensity_variation,
            start_intensity=start_intensity,
            end_intensity=end_intensity
        )
        pts = pts_all[0]
        
        # Determine actual background after potential inversion
        actual_bg = 1.0 - background_intensity if invert_prob == 1.0 else background_intensity
        
        # POSITIVE SAMPLE: Endpoint
        end_pt = pts[-1]
        path_mask = np.zeros_like(img)
        for p in pts:
            py, px = int(np.clip(p[0], 0, 127)), int(np.clip(p[1], 0, 127))
            path_mask[py, px] = 1.0
        
        crop_img = crop32(img, int(np.clip(end_pt[0], 0, 127)), int(np.clip(end_pt[1], 0, 127)), bg_value=actual_bg)
        crop_path = crop32(path_mask, int(np.clip(end_pt[0], 0, 127)), int(np.clip(end_pt[1], 0, 127)), bg_value=0.0)
        
        # Stack: [image, path_mask] - this is what the model sees
        positive_samples.append(np.stack([crop_img, crop_path], axis=0))
        
        # NEGATIVE SAMPLE: Midpoint
        # Select from middle portion (30% to 70% of curve) to avoid endpoints
        # Ensure path mask doesn't fill too much (clearly not at end)
        min_idx = max(1, int(len(pts) * 0.3))  # At least 30% into the curve
        max_idx = max(min_idx + 1, int(len(pts) * 0.7))  # At most 70% of the curve
        
        # Try multiple times to find a good midpoint where path mask isn't too full
        # AND path continues beyond the crop (not truncated at edges)
        mid_idx = None
        for attempt in range(20):  # More attempts to find good samples
            candidate_idx = np.random.randint(min_idx, max_idx)
            candidate_pt = pts[candidate_idx]
            
            # CRITICAL: Ensure path continues beyond this point (at least 10 more points)
            if candidate_idx + 10 >= len(pts):
                continue  # Too close to end, skip
            
            # Build path mask up to this point
            test_mask = np.zeros_like(img)
            for p in pts[:candidate_idx+1]:
                py, px = int(np.clip(p[0], 0, 127)), int(np.clip(p[1], 0, 127))
                test_mask[py, px] = 1.0
            
            # Crop the path mask
            crop_y = int(np.clip(candidate_pt[0], 0, 127))
            crop_x = int(np.clip(candidate_pt[1], 0, 127))
            test_crop_mask = crop32(test_mask, crop_y, crop_x)
            
            # Check 1: Path mask doesn't fill more than 60% of crop
            path_fill_ratio = test_crop_mask.sum() / (CROP * CROP)
            if path_fill_ratio >= 0.6:
                continue  # Too full, skip
            
            # Check 2: Path mask doesn't touch crop edges (indicates truncation)
            # Check all 4 edges: top, bottom, left, right
            edge_margin = 3  # Require at least 3 pixels margin from edges
            top_edge = test_crop_mask[:edge_margin, :].sum()
            bottom_edge = test_crop_mask[-edge_margin:, :].sum()
            left_edge = test_crop_mask[:, :edge_margin].sum()
            right_edge = test_crop_mask[:, -edge_margin:].sum()
            
            # If path mask touches any edge significantly, it might be truncated
            if (top_edge > 5 or bottom_edge > 5 or left_edge > 5 or right_edge > 5):
                continue  # Path mask touches edges, might look like truncation
            
            # All checks passed!
            mid_idx = candidate_idx
            break
        
        # Fallback: use any midpoint if we couldn't find a good one
        if mid_idx is None:
            mid_idx = np.random.randint(min_idx, max_idx)
        
        mid_pt = pts[mid_idx]
        
        path_mask_mid = np.zeros_like(img)
        for p in pts[:mid_idx+1]:
            py, px = int(np.clip(p[0], 0, 127)), int(np.clip(p[1], 0, 127))
            path_mask_mid[py, px] = 1.0
        
        crop_img_mid = crop32(img, int(np.clip(mid_pt[0], 0, 127)), int(np.clip(mid_pt[1], 0, 127)), bg_value=actual_bg)
        crop_path_mid = crop32(path_mask_mid, int(np.clip(mid_pt[0], 0, 127)), int(np.clip(mid_pt[1], 0, 127)), bg_value=0.0)
        
        negative_samples.append(np.stack([crop_img_mid, crop_path_mid], axis=0))
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_samples} curves...")
    
    return positive_samples, negative_samples

def create_grid_visualization(positive_samples, negative_samples, output_path="stop_module_training_samples.png"):
    """Create a grid visualization of all samples."""
    num_samples = len(positive_samples)
    grid_size = int(np.ceil(np.sqrt(num_samples)))  # 10x10 for 100 samples
    
    # Create figure with two subplots: positive (top) and negative (bottom)
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(2, 1, figure=fig, hspace=0.3, height_ratios=[1, 1])
    
    # POSITIVE SAMPLES (Endpoints - Should STOP)
    ax_pos = fig.add_subplot(gs[0])
    ax_pos.set_title(f"POSITIVE SAMPLES (Endpoints - Label=1.0, Should STOP)\n{num_samples} samples", 
                     fontsize=16, fontweight='bold', pad=20)
    
    # Create grid for positive samples
    pos_grid = np.zeros((grid_size * CROP, grid_size * CROP))
    for idx, sample in enumerate(positive_samples):
        row = idx // grid_size
        col = idx % grid_size
        y_start = row * CROP
        y_end = y_start + CROP
        x_start = col * CROP
        x_end = x_start + CROP
        # Show the image channel (channel 0)
        pos_grid[y_start:y_end, x_start:x_end] = sample[0]
    
    im1 = ax_pos.imshow(pos_grid, cmap='gray', vmin=0, vmax=1)
    ax_pos.axis('off')
    
    # NEGATIVE SAMPLES (Midpoints - Should NOT STOP)
    ax_neg = fig.add_subplot(gs[1])
    ax_neg.set_title(f"NEGATIVE SAMPLES (Midpoints - Label=0.0, Should NOT STOP)\n{num_samples} samples",
                     fontsize=16, fontweight='bold', pad=20)
    
    # Create grid for negative samples
    neg_grid = np.zeros((grid_size * CROP, grid_size * CROP))
    for idx, sample in enumerate(negative_samples):
        row = idx // grid_size
        col = idx % grid_size
        y_start = row * CROP
        y_end = y_start + CROP
        x_start = col * CROP
        x_end = x_start + CROP
        # Show the image channel (channel 0)
        neg_grid[y_start:y_end, x_start:x_end] = sample[0]
    
    im2 = ax_neg.imshow(neg_grid, cmap='gray', vmin=0, vmax=1)
    ax_neg.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.close()

def create_overlay_visualization(positive_samples, negative_samples, output_path="stop_module_training_samples_overlay.png"):
    """Create visualization showing both image and path mask channels."""
    num_samples = len(positive_samples)
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Positive: Image channel
    pos_img_grid = np.zeros((grid_size * CROP, grid_size * CROP))
    for idx, sample in enumerate(positive_samples):
        row, col = idx // grid_size, idx % grid_size
        y_start, y_end = row * CROP, (row + 1) * CROP
        x_start, x_end = col * CROP, (col + 1) * CROP
        pos_img_grid[y_start:y_end, x_start:x_end] = sample[0]
    
    axes[0, 0].imshow(pos_img_grid, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f"POSITIVE: Image Channel (Endpoints)", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Positive: Path mask channel
    pos_mask_grid = np.zeros((grid_size * CROP, grid_size * CROP))
    for idx, sample in enumerate(positive_samples):
        row, col = idx // grid_size, idx % grid_size
        y_start, y_end = row * CROP, (row + 1) * CROP
        x_start, x_end = col * CROP, (col + 1) * CROP
        pos_mask_grid[y_start:y_end, x_start:x_end] = sample[1]
    
    axes[0, 1].imshow(pos_mask_grid, cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title(f"POSITIVE: Path Mask Channel (Endpoints)", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Negative: Image channel
    neg_img_grid = np.zeros((grid_size * CROP, grid_size * CROP))
    for idx, sample in enumerate(negative_samples):
        row, col = idx // grid_size, idx % grid_size
        y_start, y_end = row * CROP, (row + 1) * CROP
        x_start, x_end = col * CROP, (col + 1) * CROP
        neg_img_grid[y_start:y_end, x_start:x_end] = sample[0]
    
    axes[1, 0].imshow(neg_img_grid, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f"NEGATIVE: Image Channel (Midpoints)", fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Negative: Path mask channel
    neg_mask_grid = np.zeros((grid_size * CROP, grid_size * CROP))
    for idx, sample in enumerate(negative_samples):
        row, col = idx // grid_size, idx % grid_size
        y_start, y_end = row * CROP, (row + 1) * CROP
        x_start, x_end = col * CROP, (col + 1) * CROP
        neg_mask_grid[y_start:y_end, x_start:x_end] = sample[1]
    
    axes[1, 1].imshow(neg_mask_grid, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title(f"NEGATIVE: Path Mask Channel (Midpoints)", fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved overlay visualization to: {output_path}")
    plt.close()

def main():
    print("=" * 60)
    print("Stop Module Training Samples Visualization")
    print("=" * 60)
    print("\nGenerating 100 positive (endpoint) and 100 negative (midpoint) samples...")
    print("This matches what the model sees during training.\n")
    
    positive_samples, negative_samples = generate_samples(num_samples=100, vessel_realism=True)
    
    print(f"\n✓ Generated {len(positive_samples)} positive samples (endpoints)")
    print(f"✓ Generated {len(negative_samples)} negative samples (midpoints)")
    
    output_dir = _script_dir
    output_path1 = os.path.join(output_dir, "stop_module_training_samples.png")
    output_path2 = os.path.join(output_dir, "stop_module_training_samples_overlay.png")
    
    print("\nCreating visualizations...")
    create_grid_visualization(positive_samples, negative_samples, output_path1)
    create_overlay_visualization(positive_samples, negative_samples, output_path2)
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    print(f"\nFiles saved:")
    print(f"  1. {output_path1} - Side-by-side comparison")
    print(f"  2. {output_path2} - Image and path mask channels separately")
    print("\nThese show what the stop module sees during training.")
    print("Compare positive (endpoints) vs negative (midpoints) to diagnose issues.\n")

if __name__ == "__main__":
    main()
