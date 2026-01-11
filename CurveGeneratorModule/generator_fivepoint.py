#!/usr/bin/env python3
"""
Five-Point Bezier Curve Generator for DSA RL Experiment
Generates curves using 5 control points using a degree-4 Bezier curve.
"""
import numpy as np
import cv2
import scipy.ndimage
from typing import Optional, Tuple, List, Dict, Any
from .generator import CurveMakerFlexible, _cubic_bezier


def _degree4_bezier(p0, p1, p2, p3, p4, t):
    """Evaluate a degree-4 Bezier curve at parameter t.
    
    A degree-4 Bezier curve uses 5 control points. The curve passes through
    p0 (at t=0) and p4 (at t=1), with p1-p3 controlling the shape.
    
    Formula: B(t) = (1-t)^4*p0 + 4*(1-t)^3*t*p1 + 6*(1-t)^2*t^2*p2 + 4*(1-t)*t^3*p3 + t^4*p4
    
    Args:
        p0, p1, p2, p3, p4: Five control points (y, x)
        t: Parameter value in [0, 1]
    
    Returns:
        Point on the curve at parameter t
    """
    omt = 1.0 - t
    return (omt**4)*p0 + 4*(omt**3)*t*p1 + 6*(omt**2)*t**2*p2 + 4*omt*t**3*p3 + (t**4)*p4


class CurveMakerFivePoint(CurveMakerFlexible):
    """Five-point curve generator that creates a single degree-4 Bezier curve.
    
    Uses 5 control points to define a single smooth Bezier curve segment.
    The curve passes through the first and last control points (p0 and p4),
    with the middle 3 control points (p1-p3) shaping the curve.
    
    Example:
        from CurveGeneratorModule import CurveMakerFivePoint
        generator = CurveMakerFivePoint(h=128, w=128)
        img, mask, pts = generator.sample_curve()
    """
    
    def _generate_five_point_curve(self, curvature_factor=1.0, allow_self_cross=False, 
                                   self_cross_prob=0.0, n_samples=None):
        """Generate a single degree-4 Bezier curve using 5 control points."""
        n_samples = n_samples if n_samples is not None else self.bezier_n_samples
        m = self._effective_margin()
        
        for attempt in range(50):
            p0 = self._random_point(margin=m)
            p4 = None
            
            for _ in range(20):
                p4_candidate = self._random_point(margin=m)
                if np.linalg.norm(p0 - p4_candidate) >= self.bezier_min_distance:
                    p4 = p4_candidate
                    break
            
            if p4 is None:
                p4 = np.array([self.h - p0[0], self.w - p0[1]], dtype=np.float32)
                p4 = self._clip_point(p4, margin=m)
            
            center = (p0 + p4) / 2.0
            
            max_spread = np.array([
                min(center[0] - m, self.h - m - center[0]),
                min(center[1] - m, self.w - m - center[1])
            ], dtype=np.float32)
            
            desired_spread = np.array([self.h, self.w], dtype=np.float32) * self.bezier_spread * curvature_factor
            
            image_center = np.array([self.h / 2.0, self.w / 2.0], dtype=np.float32)
            center_offset = np.abs(center - image_center)
            max_offset = np.array([self.h / 2.0 - m, self.w / 2.0 - m], dtype=np.float32)
            normalized_offset = np.minimum(center_offset / (max_offset + 1e-6), 1.0)
            adaptive_factor = 0.9 - (normalized_offset * 0.2)
            spread = np.minimum(desired_spread, max_spread * adaptive_factor)
            
            if allow_self_cross and self.rng.random() < self_cross_prob:
                dir_vec = self.rng.normal(0, 1, 2)
                norm = np.linalg.norm(dir_vec) + 1e-8
                dir_unit = dir_vec / norm
                p1 = center + dir_unit * spread * self.bezier_factor
                p2 = center - dir_unit * spread * self.bezier_factor
                p3 = center + dir_unit * spread * self.bezier_factor * 0.5
            else:
                p1 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor
                p2 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor * 0.8
                p3 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor * 0.6
            
            p0 = self._clip_point(p0, margin=m)
            p1 = self._clip_point(p1, margin=m)
            p2 = self._clip_point(p2, margin=m)
            p3 = self._clip_point(p3, margin=m)
            p4 = self._clip_point(p4, margin=m)
            
            test_ts = np.linspace(0, 1, 50)
            valid = True
            for t in test_ts:
                pt = _degree4_bezier(p0, p1, p2, p3, p4, t)
                if pt[0] < m or pt[0] >= self.h - m or pt[1] < m or pt[1] >= self.w - m:
                    valid = False
                    break
            
            if valid:
                ts = np.linspace(0, 1, n_samples, dtype=np.float32)
                curve_points = np.stack([
                    _degree4_bezier(p0, p1, p2, p3, p4, t) 
                    for t in ts
                ], axis=0)
                return self._clip_points(curve_points, margin=m)
        
        return super()._generate_bezier_points(
            curvature_factor=curvature_factor,
            allow_self_cross=allow_self_cross,
            self_cross_prob=self_cross_prob
        )

    def sample_curve(self, *args, **kwargs):
        """Generate a five-point degree-4 Bezier curve.
        
        Args:
            Same as CurveMakerFlexible.sample_curve().
        """
        # We need to override sample_curve to use _generate_five_point_curve instead of _generate_bezier_points
        # However, the base sample_curve calls _generate_bezier_points directly.
        # So we should probably override _generate_bezier_points in this class.
        return super().sample_curve(*args, **kwargs)

    def _generate_bezier_points(self, *args, **kwargs):
        """Override to use 5-point curve generation."""
        return self._generate_five_point_curve(*args, **kwargs)


class CenterlineMask5PointsGenerator(CurveMakerFivePoint):
    """Generator that uses 5 control points and always produces a 1-pixel thick centerline mask.
    
    This matches the user request for a "centerline mask 5 points generator".
    """
    def sample_curve(self, *args, **kwargs):
        """Generate a 5-point curve with a 1-pixel thick centerline mask."""
        kwargs['centerline_mask'] = True
        return super().sample_curve(*args, **kwargs)
