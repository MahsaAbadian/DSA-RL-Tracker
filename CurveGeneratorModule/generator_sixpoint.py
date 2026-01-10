#!/usr/bin/env python3
"""
Six-Point Bezier Curve Generator for DSA RL Experiment
Generates more complex curves by passing through 6 control points using piecewise cubic Bezier curves.
"""
import numpy as np
import cv2
import scipy.ndimage
from typing import Optional, Tuple, List, Dict, Any
from .generator import CurveMakerFlexible, _cubic_bezier


def _degree5_bezier(p0, p1, p2, p3, p4, p5, t):
    """Evaluate a degree-5 Bezier curve at parameter t.
    
    A degree-5 Bezier curve uses 6 control points. The curve passes through
    p0 (at t=0) and p5 (at t=1), with p1-p4 controlling the shape.
    
    Formula: B(t) = Σ(i=0 to 5) C(5,i) * t^i * (1-t)^(5-i) * P_i
    
    Args:
        p0, p1, p2, p3, p4, p5: Six control points (y, x)
        t: Parameter value in [0, 1]
    
    Returns:
        Point on the curve at parameter t
    """
    omt = 1.0 - t
    # Binomial coefficients for degree 5: C(5,0)=1, C(5,1)=5, C(5,2)=10, C(5,3)=10, C(5,4)=5, C(5,5)=1
    return (omt**5)*p0 + 5*(omt**4)*t*p1 + 10*(omt**3)*t*t*p2 + 10*(omt**2)*t*t*t*p3 + 5*omt*t*t*t*t*p4 + (t**5)*p5


class CurveMakerSixPoint(CurveMakerFlexible):
    """Six-point curve generator that creates a single degree-5 Bezier curve.
    
    Uses 6 control points to define a single smooth Bezier curve segment.
    The curve passes through the first and last control points (p0 and p5),
    with the middle 4 control points (p1-p4) shaping the curve.
    This creates slightly more complex curves than the standard 4-point cubic Bezier.
    
    Example:
        from CurveGeneratorModule import load_curve_config, CurveMakerSixPoint
        
        config, _ = load_curve_config("path/to/curve_config.json")
        generator = CurveMakerSixPoint(h=128, w=128, seed=42, config=config)
        img, mask, pts = generator.sample_curve(...)
    """
    
    def __init__(self, h=128, w=128, seed=None, config=None):
        """Initialize the six-point curve generator.
        
        Args:
            h: Image height
            w: Image width
            seed: Random seed for reproducibility
            config: Configuration dictionary (same as CurveMakerFlexible)
        """
        super().__init__(h, w, seed, config)
        
        # Extract six-point specific config
        sixpoint_cfg = self.config.get('six_point', {}) if self.config else {}
        self.point_spacing_factor = sixpoint_cfg.get('point_spacing_factor', 0.8)
        self.tension = sixpoint_cfg.get('tension', 0.5)  # Catmull-Rom tension parameter
        self.min_point_distance = sixpoint_cfg.get('min_point_distance', 20.0)
    
    def _generate_six_point_curve(self, curvature_factor=1.0, allow_self_cross=False, 
                                   self_cross_prob=0.0, n_samples=None):
        """Generate a single degree-5 Bezier curve using 6 control points.
        
        Similar to the 4-point cubic Bezier, but uses 6 control points for slightly
        more complex shapes. The curve passes through p0 and p5, with p1-p4 shaping it.
        
        Args:
            curvature_factor: Controls how curvy the curve is
            allow_self_cross: Whether to allow self-crossing curves
            self_cross_prob: Probability of self-crossing
            n_samples: Number of samples along the curve (if None, uses default)
        
        Returns:
            Array of (y, x) points forming the complete smooth curve
        """
        n_samples = n_samples if n_samples is not None else self.bezier_n_samples
        m = self._effective_margin()
        
        # Rejection sampling: try up to 50 times to generate a valid curve
        for attempt in range(50):
            # Generate start and end points (p0 and p5) - these the curve passes through
            p0 = self._random_point(margin=m)
            p5 = None
            
            # Ensure p5 is far enough from p0
            for _ in range(20):
                p5_candidate = self._random_point(margin=m)
                dist = np.linalg.norm(p0 - p5_candidate)
                if dist >= self.bezier_min_distance:
                    p5 = p5_candidate
                    break
            
            if p5 is None:
                # Fallback: place p5 opposite to p0
                p5 = np.array([self.h - p0[0], self.w - p0[1]], dtype=np.float32)
                p5 = self._clip_point(p5, margin=m)
                if np.linalg.norm(p0 - p5) < self.bezier_min_distance:
                    direction = p5 - p0
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction = direction / norm
                    else:
                        direction = np.array([1.0, 0.0])
                    p5 = p0 + direction * self.bezier_min_distance
                    p5 = self._clip_point(p5, margin=m)
            
            # Calculate center and spread (similar to 4-point Bezier)
            center = (p0 + p5) / 2.0
            
            # Calculate maximum available spread
            max_spread_y = min(center[0] - m, self.h - m - center[0])
            max_spread_x = min(center[1] - m, self.w - m - center[1])
            max_spread = np.array([max_spread_y, max_spread_x], dtype=np.float32)
            
            # Calculate desired spread
            desired_spread = np.array([self.h, self.w], dtype=np.float32) * self.bezier_spread * curvature_factor
            
            # Adaptive spread calculation (same as parent class)
            image_center = np.array([self.h / 2.0, self.w / 2.0], dtype=np.float32)
            center_offset = np.abs(center - image_center)
            max_offset = np.array([self.h / 2.0 - m, self.w / 2.0 - m], dtype=np.float32)
            normalized_offset = np.minimum(center_offset / (max_offset + 1e-6), 1.0)
            adaptive_factor = 0.9 - (normalized_offset * 0.2)
            spread = np.minimum(desired_spread, max_spread * adaptive_factor)
            
            # Generate 4 intermediate control points (p1, p2, p3, p4)
            # These shape the curve but don't necessarily lie on it
            if allow_self_cross and self.rng.random() < self_cross_prob:
                # Force control points to opposite sides to encourage self-cross
                dir_vec = self.rng.normal(0, 1, 2)
                norm = np.linalg.norm(dir_vec) + 1e-8
                dir_unit = dir_vec / norm
                p1 = center + dir_unit * spread * self.bezier_factor
                p2 = center - dir_unit * spread * self.bezier_factor * 0.8
                p3 = center + dir_unit * spread * self.bezier_factor * 0.6
                p4 = center - dir_unit * spread * self.bezier_factor * 0.4
            else:
                # Generate control points around center with some variation
                p1 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor
                p2 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor * 0.8
                p3 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor * 0.6
                p4 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor * 0.4
            
            # Clip control points to bounds
            p0 = self._clip_point(p0, margin=m)
            p1 = self._clip_point(p1, margin=m)
            p2 = self._clip_point(p2, margin=m)
            p3 = self._clip_point(p3, margin=m)
            p4 = self._clip_point(p4, margin=m)
            p5 = self._clip_point(p5, margin=m)
            
            # Check if curve would stay within bounds (sample a few points)
            test_ts = np.linspace(0, 1, 50)
            valid = True
            for t in test_ts:
                pt = _degree5_bezier(p0, p1, p2, p3, p4, p5, t)
                if pt[0] < m or pt[0] >= self.h - m or pt[1] < m or pt[1] >= self.w - m:
                    valid = False
                    break
            
            if valid:
                # Generate all points along the curve
                ts = np.linspace(0, 1, n_samples, dtype=np.float32)
                curve_points = np.stack([
                    _degree5_bezier(p0, p1, p2, p3, p4, p5, t) 
                    for t in ts
                ], axis=0)
                
                # Final safety check: clip all points to bounds
                curve_points = self._clip_points(curve_points, margin=m)
                return curve_points
        
        # Fallback: if rejection sampling fails, generate a simpler curve
        # Use the parent class's 4-point Bezier as fallback
        return super()._generate_bezier_points(
            curvature_factor=curvature_factor,
            allow_self_cross=allow_self_cross,
            self_cross_prob=self_cross_prob
        )
    
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
        """Generate a six-point curve with specified parameters.
        
        Args:
            All parameters same as CurveMakerFlexible.sample_curve()
        
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
            thickness = end_w
        elif width_variation == "narrow_to_wide":
            start_w = self.rng.integers(width_range[0], max(width_range[0] + 1, width_range[1]))
            end_w = self.rng.integers(width_range[1], width_range[1] * 2 + 1) if width_range[1] > width_range[0] else width_range[0] * 2
            if start_width is not None:
                start_w = start_width
            if end_width is not None:
                end_w = end_width
            thickness = start_w
        elif width_variation == "custom":
            start_w = start_width if start_width is not None else width_range[0]
            end_w = end_width if end_width is not None else width_range[1]
            thickness = (start_w + end_w) // 2
        else:
            start_w = end_w = thickness = width_range[0]
        
        # Determine intensity
        max_i = max_intensity if max_intensity is not None else 1.0
        intensity = self.rng.uniform(min_intensity, max_i)
        
        # Determine intensity variation parameters
        if intensity_variation == "bright_to_dim":
            start_i = start_intensity if start_intensity is not None else max_i
            end_i = end_intensity if end_intensity is not None else min_intensity
        elif intensity_variation == "dim_to_bright":
            start_i = start_intensity if start_intensity is not None else min_intensity
            end_i = end_intensity if end_intensity is not None else max_i
        elif intensity_variation == "custom":
            start_i = start_intensity if start_intensity is not None else intensity
            end_i = end_intensity if end_intensity is not None else intensity
        else:
            start_i = end_i = intensity
        
        # Generate six-point curve
        pts_main = self._generate_six_point_curve(
            curvature_factor=curvature_factor,
            allow_self_cross=allow_self_cross,
            self_cross_prob=self_cross_prob
        )
        
        pts_all = [pts_main]
        
        # Draw main curve
        self._draw_aa_curve(img, pts_main, thickness, intensity,
                           width_variation, start_width, end_width,
                           intensity_variation, start_intensity, end_intensity)
        
        # Create mask
        pts_xy = pts_main[:, ::-1] * 16
        pts_int = pts_xy.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [pts_int], isClosed=False, color=255,
                     thickness=max(1, int(thickness)), lineType=cv2.LINE_AA, shift=4)
        
        # Add branches if requested
        if branches:
            branch_cfg = self.config.get('branches', {}) if self.config else {}
            num_branches_range = branch_cfg.get('num_branches_range', [0, 2])
            start_range = branch_cfg.get('start_range', [0.2, 0.8])
            thickness_factor = branch_cfg.get('thickness_factor', 0.7)
            
            num_branches = self.rng.integers(num_branches_range[0], num_branches_range[1] + 1)
            
            for _ in range(num_branches):
                # Pick a random point along the main curve
                t_start = self.rng.uniform(start_range[0], start_range[1])
                idx = int(t_start * (len(pts_main) - 1))
                branch_start = pts_main[idx]
                
                # Generate branch curve (use parent's 4-point Bezier for branches)
                branch_thickness = max(1, int(thickness * thickness_factor))
                branch_intensity = intensity * 0.8
                
                branch_pts = self._generate_bezier_points(
                    p0=branch_start,
                    curvature_factor=curvature_factor * 0.7,
                    allow_self_cross=False,
                    self_cross_prob=0.0
                )
                
                pts_all.append(branch_pts)
                
                # Draw branch
                self._draw_aa_curve(img, branch_pts, branch_thickness, branch_intensity,
                                   "none", None, None, "none", None, None)
                
                # Update mask
                branch_pts_xy = branch_pts[:, ::-1] * 16
                branch_pts_int = branch_pts_xy.astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(mask, [branch_pts_int], isClosed=False, color=255,
                             thickness=max(1, int(branch_thickness)), lineType=cv2.LINE_AA, shift=4)
        
        # Add noise if requested
        if noise_prob > 0.0 and self.rng.random() < noise_prob:
            self._apply_dsa_noise(img)
        
        # Invert if requested
        if invert_prob > 0.0 and self.rng.random() < invert_prob:
            img = 1.0 - img
        
        img = np.clip(img, 0.0, 1.0)
        mask = (mask > 0.1).astype(np.uint8)
        
        return img, mask, pts_all

