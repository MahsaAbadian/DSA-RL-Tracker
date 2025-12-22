#!/usr/bin/env python3
"""
Curve Generator for DSA RL Experiment
Generates synthetic curves for training/testing.
"""
import numpy as np
import cv2
import scipy.ndimage
import os
import argparse

def _rng(seed=None):
    return np.random.default_rng(seed)

def _cubic_bezier(p0, p1, p2, p3, t):
    """Standard Cubic Bezier formula."""
    omt = 1.0 - t
    return (omt**3)*p0 + 3*omt*omt*t*p1 + 3*omt*t*t*p2 + (t**3)*p3

class CurveMakerFlexible:
    def __init__(self, h=128, w=128, seed=None):
        self.h = h
        self.w = w
        self.rng = _rng(seed)

    def _random_point(self, margin=10):
        y = self.rng.integers(margin, self.h - margin)
        x = self.rng.integers(margin, self.w - margin)
        return np.array([y, x], dtype=np.float32)

    def _generate_bezier_points(self, p0=None, n_samples=1000, curvature_factor=1.0):
        """Generates a list of (y,x) points forming a smooth bezier curve.
        
        Args:
            p0: Starting point (if None, random)
            n_samples: Number of points to sample
            curvature_factor: Controls curve complexity (1.0 = normal, <1.0 = straighter, >1.0 = more curved)
        """
        if p0 is None:
            p0 = self._random_point(margin=10)
        
        # Ensure p3 is far enough from p0 (prevents blobs)
        for _ in range(20): 
            p3 = self._random_point(margin=10)
            dist = np.linalg.norm(p0 - p3)
            if dist > 40.0: 
                break
        else:
            p3 = np.array([self.h - p0[0], self.w - p0[1]], dtype=np.float32)

        # Control points - curvature_factor affects how much control points deviate
        center = (p0 + p3) / 2.0
        spread = np.array([self.h, self.w], dtype=np.float32) * 0.3 * curvature_factor
        p1 = center + self.rng.normal(0, 1, 2) * spread * 0.6
        p2 = center + self.rng.normal(0, 1, 2) * spread * 0.6
        
        ts = np.linspace(0, 1, n_samples, dtype=np.float32)
        pts = np.stack([_cubic_bezier(p0, p1, p2, p3, t) for t in ts], axis=0)
        
        return pts

    def _draw_aa_curve(self, img, pts, thickness, intensity):
        # OpenCV draw
        pts_xy = pts[:, ::-1] * 16 
        pts_int = pts_xy.astype(np.int32).reshape((-1, 1, 2))
        canvas = np.zeros((self.h, self.w), dtype=np.uint8)
        cv2.polylines(canvas, [pts_int], isClosed=False, color=255, 
                      thickness=int(thickness), lineType=cv2.LINE_AA, shift=4)
        canvas_float = canvas.astype(np.float32) / 255.0
        img[:] = np.maximum(img, canvas_float * intensity)

    def sample_curve(self, 
                     width_range=(2, 2),    
                     noise_prob=0.0,        
                     invert_prob=0.0,
                     min_intensity=0.6,
                     branches=False,
                     curvature_factor=1.0):       
        """Generate a curve with specified parameters.
        
        Args:
            width_range: (min, max) thickness of the curve
            noise_prob: Probability of adding DSA noise
            invert_prob: Probability of inverting the image
            min_intensity: Minimum intensity value (higher = more contrast)
            branches: Whether to add branch curves
            curvature_factor: Controls curve complexity (1.0 = normal, <1.0 = straighter, >1.0 = more curved)
        """
        img = np.zeros((self.h, self.w), dtype=np.float32)
        mask = np.zeros_like(img) 
        
        thickness = self.rng.integers(width_range[0], width_range[1] + 1)
        thickness = max(1, int(thickness))
        intensity = self.rng.uniform(min_intensity, 1.0)

        pts_main = self._generate_bezier_points(curvature_factor=curvature_factor)
        self._draw_aa_curve(img, pts_main, thickness, intensity)
        self._draw_aa_curve(mask, pts_main, thickness, 1.0)
        pts_all = [pts_main]

        if branches:
            num_branches = self.rng.integers(1, 3)
            for _ in range(num_branches):
                idx = self.rng.integers(int(len(pts_main)*0.2), int(len(pts_main)*0.8))
                p0 = pts_main[idx]
                pts_branch = self._generate_bezier_points(p0=p0, curvature_factor=curvature_factor)
                b_thick = max(1, int(thickness * 0.7))
                self._draw_aa_curve(img, pts_branch, b_thick, intensity)
                self._draw_aa_curve(mask, pts_branch, b_thick, 1.0)
                pts_all.append(pts_branch)

        if self.rng.random() < noise_prob:
            self._apply_dsa_noise(img)

        if self.rng.random() < invert_prob:
            img = 1.0 - img

        img = np.clip(img, 0.0, 1.0)
        mask = (mask > 0.1).astype(np.uint8)

        return img, mask, pts_all

    def _apply_dsa_noise(self, img):
        num_blobs = self.rng.integers(1, 4)
        for _ in range(num_blobs):
            y, x = self._random_point(margin=0)
            sigma = self.rng.uniform(2, 8) 
            yy, xx = np.ogrid[:self.h, :self.w]
            dist_sq = (yy - y)**2 + (xx - x)**2
            blob = np.exp(-dist_sq / (2 * sigma**2))
            blob_int = self.rng.uniform(0.05, 0.2)
            img[:] = np.maximum(img, blob * blob_int)

        # Gaussian Static
        noise_level = self.rng.uniform(0.05, 0.15)
        noise = self.rng.normal(0, noise_level, img.shape)
        img[:] += noise

        # Gaussian Blur
        if self.rng.random() < 0.5:
            sigma = self.rng.uniform(0.5, 1.0)
            img[:] = scipy.ndimage.gaussian_filter(img, sigma=sigma)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic curves")
    parser.add_argument("--output_dir", type=str, default="generated_curves", 
                        help="Output directory for generated curves")
    parser.add_argument("--num_curves", type=int, default=100, 
                        help="Number of curves to generate")
    parser.add_argument("--h", type=int, default=128, help="Image height")
    parser.add_argument("--w", type=int, default=128, help="Image width")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                        help="Stage number (1=simple, 2=medium, 3=complex). If None, generates for all stages.")
    parser.add_argument("--all_stages", action="store_true",
                        help="Generate curves for all 3 stages")
    
    args = parser.parse_args()
    
    # Get absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, args.output_dir)
    
    # Stage-specific configurations
    stage_configs = {
        1: {
            'name': 'Stage1_Bootstrap',
            'width_range': (2, 4),
            'noise_prob': 0.0,
            'invert_prob': 0.5,
            'min_intensity': 0.6,
            'branches': False,
            'curvature_factor': 0.5,  # Straighter curves
            'num_curves': args.num_curves
        },
        2: {
            'name': 'Stage2_Robustness',
            'width_range': (2, 8),
            'noise_prob': 0.0,  # Noise applied during training, not generation
            'invert_prob': 0.5,
            'min_intensity': 0.4,
            'branches': False,
            'curvature_factor': 1.0,  # Normal curvature
            'num_curves': args.num_curves
        },
        3: {
            'name': 'Stage3_Realism',
            'width_range': (1, 10),
            'noise_prob': 0.0,  # Noise applied during training, not generation
            'invert_prob': 0.5,
            'min_intensity': 0.2,
            'branches': True,  # Add branches for complexity
            'curvature_factor': 1.5,  # More curved
            'num_curves': args.num_curves
        }
    }
    
    # Determine which stages to generate
    if args.all_stages:
        stages_to_generate = [1, 2, 3]
    elif args.stage is not None:
        stages_to_generate = [args.stage]
    else:
        # Default: generate for all stages
        stages_to_generate = [1, 2, 3]
    
    for stage_num in stages_to_generate:
        config = stage_configs[stage_num]
        output_dir = os.path.join(base_output_dir, f"stage{stage_num}")
        os.makedirs(output_dir, exist_ok=True)
        
        cm = CurveMakerFlexible(h=args.h, w=args.w, seed=args.seed)
        print(f"\n=== Generating {config['num_curves']} curves for {config['name']} ===")
        print(f"  Width: {config['width_range']}, Curvature: {config['curvature_factor']:.1f}")
        print(f"  Branches: {config['branches']}, Intensity: {config['min_intensity']:.1f}")
        print(f"  Output: {output_dir}")
        
        for i in range(config['num_curves']):
            img, mask, pts_all = cm.sample_curve(
                width_range=config['width_range'],
                noise_prob=config['noise_prob'],
                invert_prob=config['invert_prob'],
                min_intensity=config['min_intensity'],
                branches=config['branches'],
                curvature_factor=config['curvature_factor']
            )
            # Save image
            img_path = os.path.join(output_dir, f"curve_{i:05d}.png")
            cv2.imwrite(img_path, (img * 255).astype(np.uint8))
            
            # Save mask
            mask_path = os.path.join(output_dir, f"mask_{i:05d}.png")
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
            
            # Save points (ground truth polyline)
            pts_path = os.path.join(output_dir, f"points_{i:05d}.npy")
            np.save(pts_path, pts_all[0])  # Save the main curve points
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{config['num_curves']} curves")
        
        print(f"Done! Saved {config['num_curves']} curves to {output_dir}")
        print(f"  - Images: curve_*.png")
        print(f"  - Masks: mask_*.png")
        print(f"  - Points: points_*.npy")
    
    print(f"\n=== All curve generation complete ===")
