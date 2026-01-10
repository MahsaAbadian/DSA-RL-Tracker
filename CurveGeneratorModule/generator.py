#!/usr/bin/env python3
"""
Curve Generator for DSA RL Experiment
Generates synthetic curves for training/testing.
"""
import numpy as np
import cv2
import scipy.ndimage
from typing import Optional, Tuple, List, Dict, Any


def _rng(seed=None):
    """Create a numpy random number generator."""
    return np.random.default_rng(seed)


def _cubic_bezier(p0, p1, p2, p3, t):
    """Standard Cubic Bezier formula."""
    omt = 1.0 - t
    return (omt**3)*p0 + 3*omt*omt*t*p1 + 3*omt*t*t*p2 + (t**3)*p3


class CurveMakerFlexible:
    """Flexible curve generator that reads configuration from a dict.
    
    The config dict is typically loaded from a JSON file using load_curve_config()
    from CurveGeneratorModule.config_loader. If config is None, defaults are used.
    
    Example:
        from CurveGeneratorModule import load_curve_config, CurveMakerFlexible
        
        config, _ = load_curve_config("path/to/curve_config.json")
        generator = CurveMakerFlexible(h=128, w=128, seed=42, config=config)
    """
    
    def __init__(self, h=128, w=128, seed=None, config=None):
        """Initialize the curve generator.
        
        Args:
            h: Image height
            w: Image width
            seed: Random seed for reproducibility
            config: Configuration dictionary (typically loaded from JSON via load_curve_config()).
                   If None, uses default values. The dict structure should include:
                   - 'image': {height, width}
                   - 'bezier': {n_samples, margin, min_distance, ...}
                   - 'branches': {num_branches_range, start_range, ...}
                   - 'noise': {num_blobs_range, blob_sigma_range, ...}
        """
        self.h = h
        self.w = w
        self.rng = _rng(seed)
        self.config = config or {}
        
        # Extract bezier config with defaults
        bezier_cfg = self.config.get('bezier', {})
        self.bezier_n_samples = bezier_cfg.get('n_samples', 1000)
        self.bezier_margin = bezier_cfg.get('margin', 10)
        self.bezier_min_distance = bezier_cfg.get('min_distance', 40.0)
        self.bezier_spread = bezier_cfg.get('control_point_spread', 0.3)
        self.bezier_factor = bezier_cfg.get('control_point_factor', 0.6)
        
        # Extract branch config with defaults
        branch_cfg = self.config.get('branches', {})
        if not isinstance(branch_cfg, dict):
            branch_cfg = {}
        self.branch_num_range = tuple(branch_cfg.get('num_branches_range', [1, 3]))
        self.branch_start_range = tuple(branch_cfg.get('start_range', [0.2, 0.8]))
        self.branch_thickness_factor = branch_cfg.get('thickness_factor', 0.7)
        
        # Extract noise config with defaults
        noise_cfg = self.config.get('noise', {})
        self.noise_num_blobs_range = tuple(noise_cfg.get('num_blobs_range', [1, 4]))
        self.noise_blob_sigma_range = tuple(noise_cfg.get('blob_sigma_range', [2.0, 8.0]))
        self.noise_blob_intensity_range = tuple(noise_cfg.get('blob_intensity_range', [0.05, 0.2]))
        self.noise_level_range = tuple(noise_cfg.get('noise_level_range', [0.05, 0.15]))
        self.noise_gaussian_blur_prob = noise_cfg.get('gaussian_blur_prob', 0.5)
        self.noise_gaussian_blur_sigma_range = tuple(noise_cfg.get('gaussian_blur_sigma_range', [0.5, 1.0]))

    def _effective_margin(self, max_thickness=10):
        """Calculate effective margin accounting for line thickness."""
        return self.bezier_margin + max_thickness
    
    def _clip_point(self, p, margin=None):
        """Clip a point to stay within image bounds with margin."""
        if margin is None:
            margin = self.bezier_margin
        y = np.clip(p[0], margin, self.h - margin - 1)
        x = np.clip(p[1], margin, self.w - margin - 1)
        return np.array([y, x], dtype=np.float32)
    
    def _clip_points(self, pts, margin=None):
        """Clip all points to stay within image bounds with margin."""
        if margin is None:
            margin = self.bezier_margin
        pts_clipped = pts.copy()
        pts_clipped[:, 0] = np.clip(pts_clipped[:, 0], margin, self.h - margin - 1)
        pts_clipped[:, 1] = np.clip(pts_clipped[:, 1], margin, self.w - margin - 1)
        return pts_clipped
    
    def _random_point(self, margin=None):
        """Generate a random point within the image bounds."""
        margin = margin if margin is not None else self.bezier_margin
        # Ensure valid range
        y_max = max(margin + 1, self.h - margin)
        x_max = max(margin + 1, self.w - margin)
        y = self.rng.integers(margin, y_max)
        x = self.rng.integers(margin, x_max)
        return np.array([y, x], dtype=np.float32)

    def _check_curve_bounds(self, p0, p1, p2, p3, margin, num_test_points=50):
        """Check if a bezier curve would stay within bounds.
        
        Uses the convex hull property: bezier curves stay within the convex hull
        of their control points. We also sample a few points to be safe.
        
        Returns:
            bool: True if curve appears to stay within bounds, False otherwise
        """
        # Check 1: All control points within bounds?
        control_points = np.array([p0, p1, p2, p3])
        if np.any(control_points[:, 0] < margin) or np.any(control_points[:, 0] >= self.h - margin):
            return False
        if np.any(control_points[:, 1] < margin) or np.any(control_points[:, 1] >= self.w - margin):
            return False
        
        # Check 2: Sample points along the curve
        test_ts = np.linspace(0, 1, num_test_points)
        for t in test_ts:
            pt = _cubic_bezier(p0, p1, p2, p3, t)
            if pt[0] < margin or pt[0] >= self.h - margin:
                return False
            if pt[1] < margin or pt[1] >= self.w - margin:
                return False
        
        return True

    def _generate_bezier_points(self, p0=None, n_samples=None, curvature_factor=1.0,
                                allow_self_cross=False, self_cross_prob=0.0):
        """Generates a list of (y,x) points forming a smooth bezier curve.
        
        Uses rejection sampling to ensure curves stay within bounds while maintaining curvature.
        All points are guaranteed to stay within image bounds.
        """
        n_samples = n_samples if n_samples is not None else self.bezier_n_samples
        
        # Use effective margin to account for potential line thickness
        m = self._effective_margin()
        
        # Rejection sampling: try up to 50 times to generate a valid curve
        for attempt in range(50):
            if p0 is None:
                p0_candidate = self._random_point(margin=m)
            else:
                p0_candidate = self._clip_point(p0, margin=m)
            
            # Ensure p3 is far enough from p0 (prevents blobs)
            p3_candidate = None
            for _ in range(20): 
                p3_test = self._random_point(margin=m)
                dist = np.linalg.norm(p0_candidate - p3_test)
                if dist > self.bezier_min_distance: 
                    p3_candidate = p3_test
                    break
            
            if p3_candidate is None:
                # Fallback: place p3 opposite to p0
                p3_candidate = np.array([self.h - p0_candidate[0], self.w - p0_candidate[1]], dtype=np.float32)
                p3_candidate = self._clip_point(p3_candidate, margin=m)
                # Ensure minimum distance
                if np.linalg.norm(p0_candidate - p3_candidate) < self.bezier_min_distance:
                    direction = p3_candidate - p0_candidate
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction = direction / norm
                        p3_candidate = p0_candidate + direction * self.bezier_min_distance
                        p3_candidate = self._clip_point(p3_candidate, margin=m)

            # Control points - curvature_factor affects how much control points deviate
            center = (p0_candidate + p3_candidate) / 2.0
            
            # Calculate maximum available spread from center to bounds
            max_spread_y = min(center[0] - m, self.h - m - center[0])
            max_spread_x = min(center[1] - m, self.w - m - center[1])
            max_spread = np.array([max_spread_y, max_spread_x], dtype=np.float32)
            
            # Calculate desired spread
            desired_spread = np.array([self.h, self.w], dtype=np.float32) * self.bezier_spread * curvature_factor
            
            # Adaptive spread calculation based on center position
            # If center is near image center → use larger spread (more curvy)
            # If center is near edges → use smaller but still reasonable spread
            image_center = np.array([self.h / 2.0, self.w / 2.0], dtype=np.float32)
            center_offset = np.abs(center - image_center)
            max_offset = np.array([self.h / 2.0 - m, self.w / 2.0 - m], dtype=np.float32)
            
            # Normalize offset: 0 = at center, 1 = at edge
            normalized_offset = np.minimum(center_offset / (max_offset + 1e-6), 1.0)
            
            # Adaptive safety factor: 0.9 near center, 0.7 near edges
            # This allows curves near center to be more curvy, while edge curves are more conservative
            adaptive_factor = 0.9 - (normalized_offset * 0.2)  # Range: 0.9 (center) to 0.7 (edge)
            
            # Apply adaptive factor to max_spread
            spread = np.minimum(desired_spread, max_spread * adaptive_factor)

            if allow_self_cross and self.rng.random() < self_cross_prob:
                # Force control points to opposite sides of the center to encourage a self-cross
                dir_vec = self.rng.normal(0, 1, 2)
                norm = np.linalg.norm(dir_vec) + 1e-8
                dir_unit = dir_vec / norm
                p1_candidate = center + dir_unit * spread * self.bezier_factor
                p2_candidate = center - dir_unit * spread * self.bezier_factor
            else:
                p1_candidate = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor
                p2_candidate = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor
            
            # Check if this curve would stay within bounds
            if self._check_curve_bounds(p0_candidate, p1_candidate, p2_candidate, p3_candidate, m):
                # Good curve! Generate all points
                ts = np.linspace(0, 1, n_samples, dtype=np.float32)
                pts = np.stack([_cubic_bezier(p0_candidate, p1_candidate, p2_candidate, p3_candidate, t) for t in ts], axis=0)
                
                # Final safety check: verify all points are within bounds
                if np.all(pts[:, 0] >= m) and np.all(pts[:, 0] < self.h - m) and \
                   np.all(pts[:, 1] >= m) and np.all(pts[:, 1] < self.w - m):
                    return pts
                else:
                    # If somehow points are out of bounds, clip them (shouldn't happen with good check)
                    pts = self._clip_points(pts, margin=m)
                    return pts
        
        # Fallback: if rejection sampling fails after 50 attempts, generate a simple safe curve
        # This should rarely happen, but ensures we always return something
        p0_safe = self._random_point(margin=m)
        p3_safe = self._random_point(margin=m)
        # Ensure minimum distance
        if np.linalg.norm(p0_safe - p3_safe) < self.bezier_min_distance:
            direction = p3_safe - p0_safe
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
                p3_safe = p0_safe + direction * self.bezier_min_distance
                p3_safe = self._clip_point(p3_safe, margin=m)
        
        # Use very conservative control points for fallback
        center_safe = (p0_safe + p3_safe) / 2.0
        max_spread_safe = np.array([
            min(center_safe[0] - m, self.h - m - center_safe[0]),
            min(center_safe[1] - m, self.w - m - center_safe[1])
        ], dtype=np.float32)
        spread_safe = max_spread_safe * 0.5  # Very conservative
        
        p1_safe = center_safe + self.rng.normal(0, 1, 2) * spread_safe * self.bezier_factor
        p2_safe = center_safe + self.rng.normal(0, 1, 2) * spread_safe * self.bezier_factor
        p1_safe = self._clip_point(p1_safe, margin=m)
        p2_safe = self._clip_point(p2_safe, margin=m)
        
        ts = np.linspace(0, 1, n_samples, dtype=np.float32)
        pts = np.stack([_cubic_bezier(p0_safe, p1_safe, p2_safe, p3_safe, t) for t in ts], axis=0)
        pts = self._clip_points(pts, margin=m)
        return pts

    def _draw_aa_curve(self, img, pts, thickness, intensity, width_variation="none", start_width=None, end_width=None,
                       intensity_variation="none", start_intensity=None, end_intensity=None):
        """Draw a curve with optional variable width and intensity.
        
        Args:
            img: Image to draw on
            pts: Array of (y, x) points
            thickness: Base thickness (or end thickness if width_variation is used)
            intensity: Base intensity value
            width_variation: "none", "wide_to_narrow", "narrow_to_wide", or "custom"
            start_width: Starting width (for custom or wide_to_narrow/narrow_to_wide)
            end_width: Ending width (for custom or wide_to_narrow/narrow_to_wide)
            intensity_variation: "none", "bright_to_dim", "dim_to_bright", or "custom"
            start_intensity: Starting intensity (for variable intensity)
            end_intensity: Ending intensity (for variable intensity)
        """
        # Determine intensity values
        if intensity_variation == "none":
            start_i = end_i = intensity
        elif intensity_variation == "bright_to_dim":
            start_i = start_intensity if start_intensity is not None else intensity * 1.5
            end_i = end_intensity if end_intensity is not None else intensity * 0.5
        elif intensity_variation == "dim_to_bright":
            start_i = start_intensity if start_intensity is not None else intensity * 0.5
            end_i = end_intensity if end_intensity is not None else intensity * 1.5
        elif intensity_variation == "custom":
            start_i = start_intensity if start_intensity is not None else intensity
            end_i = end_intensity if end_intensity is not None else intensity
        else:
            start_i = end_i = intensity
        
        if width_variation == "none":
            # Constant width, but may have variable intensity
            if intensity_variation == "none":
                # Original constant width and intensity drawing
                pts_xy = pts[:, ::-1] * 16 
                pts_int = pts_xy.astype(np.int32).reshape((-1, 1, 2))
                canvas = np.zeros((self.h, self.w), dtype=np.uint8)
                cv2.polylines(canvas, [pts_int], isClosed=False, color=255, 
                              thickness=int(thickness), lineType=cv2.LINE_AA, shift=4)
                canvas_float = canvas.astype(np.float32) / 255.0
                img[:] = np.maximum(img, canvas_float * intensity)
            else:
                # Constant width but variable intensity - draw segment by segment
                n_pts = len(pts)
                canvas = np.zeros((self.h, self.w), dtype=np.uint8)
                segment_length = max(1, n_pts // 50)
                
                for i in range(0, n_pts - 1, segment_length):
                    end_idx = min(i + segment_length, n_pts - 1)
                    t_start = i / (n_pts - 1) if n_pts > 1 else 0.0
                    t_end = end_idx / (n_pts - 1) if n_pts > 1 else 1.0
                    intensity_start = start_i + (end_i - start_i) * t_start
                    intensity_end = start_i + (end_i - start_i) * t_end
                    intensity_avg = (intensity_start + intensity_end) / 2.0
                    
                    segment_pts = pts[i:end_idx+1]
                    pts_xy = segment_pts[:, ::-1] * 16
                    pts_int = pts_xy.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [pts_int], isClosed=False, color=255,
                                  thickness=int(thickness), lineType=cv2.LINE_AA, shift=4)
                    # Apply variable intensity to this segment
                    segment_mask = np.zeros((self.h, self.w), dtype=np.float32)
                    cv2.polylines(segment_mask, [pts_int], isClosed=False, color=1.0,
                                  thickness=int(thickness), lineType=cv2.LINE_AA, shift=4)
                    img[:] = np.maximum(img, segment_mask * intensity_avg)
                
                return  # Early return for variable intensity case
        else:
            # Variable width: draw segment by segment (with optional variable intensity)
            if width_variation == "wide_to_narrow":
                start_w = start_width if start_width is not None else thickness * 2.0
                end_w = end_width if end_width is not None else thickness
            elif width_variation == "narrow_to_wide":
                start_w = start_width if start_width is not None else thickness
                end_w = end_width if end_width is not None else thickness * 2.0
            elif width_variation == "custom":
                start_w = start_width if start_width is not None else thickness
                end_w = end_width if end_width is not None else thickness
            else:
                start_w = end_w = thickness
            
            # Draw curve in segments with interpolated thickness and intensity
            n_pts = len(pts)
            canvas = np.zeros((self.h, self.w), dtype=np.uint8)
            segment_length = max(1, n_pts // 50)  # ~50 segments for smooth variation
            
            for i in range(0, n_pts - 1, segment_length):
                end_idx = min(i + segment_length, n_pts - 1)
                
                # Interpolate thickness and intensity
                t_start = i / (n_pts - 1) if n_pts > 1 else 0.0
                t_end = end_idx / (n_pts - 1) if n_pts > 1 else 1.0
                thickness_start = start_w + (end_w - start_w) * t_start
                thickness_end = start_w + (end_w - start_w) * t_end
                thickness_avg = (thickness_start + thickness_end) / 2.0
                intensity_start = start_i + (end_i - start_i) * t_start
                intensity_end = start_i + (end_i - start_i) * t_end
                intensity_avg = (intensity_start + intensity_end) / 2.0
                
                # Draw segment
                segment_pts = pts[i:end_idx+1]
                pts_xy = segment_pts[:, ::-1] * 16
                pts_int = pts_xy.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(canvas, [pts_int], isClosed=False, color=255,
                              thickness=max(1, int(thickness_avg)), lineType=cv2.LINE_AA, shift=4)
                # Apply variable intensity to this segment
                segment_mask = np.zeros((self.h, self.w), dtype=np.float32)
                cv2.polylines(segment_mask, [pts_int], isClosed=False, color=1.0,
                              thickness=max(1, int(thickness_avg)), lineType=cv2.LINE_AA, shift=4)
                img[:] = np.maximum(img, segment_mask * intensity_avg)

    def sample_curve(self, 
                     width_range=(2, 2),    
                     noise_prob=0.0,        
                     invert_prob=0.0,
                     min_intensity=0.6,
                     max_intensity=None,
                     branches=False,
                     curvature_factor=1.0,
                     width_variation="none",
                     start_width=None,
                     end_width=None,
                     intensity_variation="none",
                     start_intensity=None,
                     end_intensity=None,
                     background_intensity=None,
                     allow_self_cross=False,
                     self_cross_prob=0.0):       
        """Generate a curve with specified parameters.
        
        Args:
            width_range: (min, max) thickness of the curve
            noise_prob: Probability of adding DSA noise
            invert_prob: Probability of inverting the image
            min_intensity: Minimum intensity value (lower = more transparent/faint)
            max_intensity: Maximum intensity value (if None, uses 1.0)
            branches: Whether to add branch curves
            curvature_factor: Controls curve complexity (1.0 = normal, <1.0 = straighter, >1.0 = more curved)
            width_variation: "none", "wide_to_narrow", "narrow_to_wide", or "custom"
            start_width: Starting width (for variable width curves)
            end_width: Ending width (for variable width curves)
            intensity_variation: "none", "bright_to_dim", "dim_to_bright", or "custom"
            start_intensity: Starting intensity (for variable intensity curves)
            end_intensity: Ending intensity (for variable intensity curves)
            background_intensity: Background intensity (0.0-1.0). If None, uses 0.0 (black)
            allow_self_cross: Whether to allow self-crossing curves
            self_cross_prob: Probability of self-crossing (if allow_self_cross is True)
        
        Returns:
            tuple: (img, mask, pts_all)
            - img: Generated image as float32 array (0.0-1.0)
            - mask: Binary mask of the curve as uint8 array
            - pts_all: List of point arrays (main curve + branches)
        """
        # Set background intensity
        bg_intensity = background_intensity if background_intensity is not None else 0.0
        img = np.full((self.h, self.w), bg_intensity, dtype=np.float32)
        mask = np.zeros_like(img) 
        
        # Determine base thickness
        if width_variation == "none":
            thickness = self.rng.integers(width_range[0], width_range[1] + 1)
            thickness = max(1, int(thickness))
            start_w = end_w = thickness
        elif width_variation == "wide_to_narrow":
            start_w = self.rng.integers(width_range[1], width_range[1] * 2 + 1) if width_range[1] > width_range[0] else width_range[0] * 2
            end_w = self.rng.integers(width_range[0], max(width_range[0] + 1, width_range[1]))
            if start_width is not None:
                start_w = start_width
            if end_width is not None:
                end_w = end_width
            thickness = end_w  # Use end width as base
        elif width_variation == "narrow_to_wide":
            start_w = self.rng.integers(width_range[0], max(width_range[0] + 1, width_range[1]))
            end_w = self.rng.integers(width_range[1], width_range[1] * 2 + 1) if width_range[1] > width_range[0] else width_range[0] * 2
            if start_width is not None:
                start_w = start_width
            if end_width is not None:
                end_w = end_width
            thickness = start_w  # Use start width as base
        elif width_variation == "custom":
            start_w = start_width if start_width is not None else width_range[0]
            end_w = end_width if end_width is not None else width_range[1]
            thickness = (start_w + end_w) / 2  # Average for base
        else:
            thickness = self.rng.integers(width_range[0], width_range[1] + 1)
            start_w = end_w = thickness
        
        max_int = max_intensity if max_intensity is not None else 1.0
        intensity = self.rng.uniform(min_intensity, max_int)

        # Determine intensity variation parameters
        if intensity_variation == "bright_to_dim":
            start_i = start_intensity if start_intensity is not None else max_int
            end_i = end_intensity if end_intensity is not None else min_intensity
        elif intensity_variation == "dim_to_bright":
            start_i = start_intensity if start_intensity is not None else min_intensity
            end_i = end_intensity if end_intensity is not None else max_int
        elif intensity_variation == "custom":
            start_i = start_intensity if start_intensity is not None else intensity
            end_i = end_intensity if end_intensity is not None else intensity
        else:
            start_i = end_i = intensity

        pts_main = self._generate_bezier_points(
            curvature_factor=curvature_factor,
            allow_self_cross=allow_self_cross,
            self_cross_prob=self_cross_prob
        )
        self._draw_aa_curve(img, pts_main, thickness, intensity, width_variation, start_w, end_w,
                           intensity_variation, start_i, end_i)
        self._draw_aa_curve(mask, pts_main, thickness, 1.0, width_variation, start_w, end_w)
        pts_all = [pts_main]

        if branches:
            num_branches = self.rng.integers(self.branch_num_range[0], self.branch_num_range[1])
            for _ in range(num_branches):
                start_min = int(len(pts_main) * self.branch_start_range[0])
                start_max = int(len(pts_main) * self.branch_start_range[1])
                idx = self.rng.integers(start_min, start_max)
                p0 = pts_main[idx]
                pts_branch = self._generate_bezier_points(p0=p0, curvature_factor=curvature_factor)
                b_thick = max(1, int(thickness * self.branch_thickness_factor))
                b_start_w = max(1, int(start_w * self.branch_thickness_factor)) if width_variation != "none" else b_thick
                b_end_w = max(1, int(end_w * self.branch_thickness_factor)) if width_variation != "none" else b_thick
                # Branches use same intensity variation pattern
                self._draw_aa_curve(img, pts_branch, b_thick, intensity, width_variation, b_start_w, b_end_w,
                                   intensity_variation, start_i, end_i)
                self._draw_aa_curve(mask, pts_branch, b_thick, 1.0, width_variation, b_start_w, b_end_w)
                pts_all.append(pts_branch)

        if self.rng.random() < noise_prob:
            self._apply_dsa_noise(img)

        if self.rng.random() < invert_prob:
            img = 1.0 - img

        img = np.clip(img, 0.0, 1.0)
        mask = (mask > 0.1).astype(np.uint8)

        return img, mask, pts_all

    def _apply_dsa_noise(self, img):
        """Apply DSA-style noise to the image."""
        num_blobs = self.rng.integers(self.noise_num_blobs_range[0], self.noise_num_blobs_range[1])
        for _ in range(num_blobs):
            y, x = self._random_point(margin=0)
            sigma = self.rng.uniform(self.noise_blob_sigma_range[0], self.noise_blob_sigma_range[1])
            yy, xx = np.ogrid[:self.h, :self.w]
            dist_sq = (yy - y)**2 + (xx - x)**2
            blob = np.exp(-dist_sq / (2 * sigma**2))
            blob_int = self.rng.uniform(self.noise_blob_intensity_range[0], self.noise_blob_intensity_range[1])
            img[:] = np.maximum(img, blob * blob_int)

        # Gaussian Static
        noise_level = self.rng.uniform(self.noise_level_range[0], self.noise_level_range[1])
        noise = self.rng.normal(0, noise_level, img.shape)
        img[:] += noise

        # Gaussian Blur
        if self.rng.random() < self.noise_gaussian_blur_prob:
            sigma = self.rng.uniform(self.noise_gaussian_blur_sigma_range[0], self.noise_gaussian_blur_sigma_range[1])
            img[:] = scipy.ndimage.gaussian_filter(img, sigma=sigma)

