#!/usr/bin/env python3
"""
Unified Curve Generator for DSA RL Experiment
A single, config-driven class that generates any degree Bezier curve,
supporting multiple segments and flexible ground truth masks.
"""
import numpy as np
import cv2
import scipy.ndimage
from scipy.special import comb
from typing import Optional, Tuple, List, Dict, Any


def _rng(seed=None):
    """Create a numpy random number generator."""
    return np.random.default_rng(seed)


def _general_bezier(control_points: np.ndarray, n_samples: int) -> np.ndarray:
    """Evaluate a Bezier curve of any degree defined by control_points.
    
    Args:
        control_points: Array of shape (N, 2) where N is number of control points.
        n_samples: Number of points to sample along the curve.
        
    Returns:
        Array of shape (n_samples, 2) containing the curve points.
    """
    n = len(control_points) - 1
    t = np.linspace(0, 1, n_samples)[:, np.newaxis]
    
    # Precompute binomial coefficients and basis
    # B(t) = sum_{i=0}^{n} comb(n, i) * (1-t)^(n-i) * t^i * P_i
    curve = np.zeros((n_samples, 2), dtype=np.float32)
    for i in range(n + 1):
        basis = comb(n, i) * ((1 - t) ** (n - i)) * (t ** i)
        curve += basis * control_points[i]
        
    return curve


class CurveMaker(object):
    """Unified, config-driven curve generator.
    
    Handles geometry generation (Bezier of any degree), multi-segment chaining,
    and rendering (noise, variable width, centerline masks).
    """
    
    def __init__(self, h=128, w=128, seed=None, config=None):
        """Initialize the unified generator.
        
        Args:
            h, w: Image dimensions.
            seed: Random seed.
            config: Configuration dictionary.
        """
        self.h = h
        self.w = w
        self.rng = _rng(seed)
        self.config = config or {}
        
        # --- Bezier Geometry Config ---
        bezier_cfg = self.config.get('bezier', {})
        self.n_samples = bezier_cfg.get('n_samples', 1000)
        self.margin = bezier_cfg.get('margin', 10)
        self.min_dist = bezier_cfg.get('min_distance', 40.0)
        self.spread = bezier_cfg.get('control_point_spread', 0.3)
        self.cp_factor = bezier_cfg.get('control_point_factor', 0.6)
        
        # Unified parameters: degree and segments
        self.default_num_cp = bezier_cfg.get('num_control_points', 4)  # Default: cubic
        self.default_num_segments = self.config.get('multi_segment', {}).get('num_segments_range', [1, 1])[0]
        self.default_centerline_mask = bezier_cfg.get('centerline_mask', False)
        
        # --- Branch Config ---
        branch_cfg = self.config.get('branches', {})
        self.branch_num_range = tuple(branch_cfg.get('num_branches_range', [1, 3]))
        self.branch_start_range = tuple(branch_cfg.get('start_range', [0.2, 0.8]))
        self.branch_thickness_factor = branch_cfg.get('thickness_factor', 0.7)
        
        # --- Noise Config ---
        noise_cfg = self.config.get('noise', {})
        self.noise_num_blobs_range = tuple(noise_cfg.get('num_blobs_range', [1, 4]))
        self.noise_blob_sigma_range = tuple(noise_cfg.get('blob_sigma_range', [2.0, 8.0]))
        self.noise_blob_intensity_range = tuple(noise_cfg.get('blob_intensity_range', [0.05, 0.2]))
        self.noise_level_range = tuple(noise_cfg.get('noise_level_range', [0.05, 0.15]))
        self.noise_blur_prob = noise_cfg.get('gaussian_blur_prob', 0.5)
        self.noise_blur_sigma_range = tuple(noise_cfg.get('gaussian_blur_sigma_range', [0.5, 1.0]))

        # --- Tissue Noise Config ---
        tissue_cfg = self.config.get('tissue_noise', {})
        self.tissue_sigma_range = tuple(tissue_cfg.get('sigma_range', [2.0, 5.0]))
        self.tissue_intensity_range = tuple(tissue_cfg.get('intensity_range', [0.2, 0.4]))

    def _effective_margin(self, max_thickness=10):
        return self.margin + max_thickness
    
    def _random_point(self, margin=None):
        m = margin if margin is not None else self.margin
        y_max = max(m + 1, self.h - m)
        x_max = max(m + 1, self.w - m)
        return np.array([self.rng.integers(m, y_max), self.rng.integers(m, x_max)], dtype=np.float32)

    def _clip_point(self, p, margin=None):
        m = margin if margin is not None else self.margin
        return np.array([np.clip(p[0], m, self.h - m - 1), np.clip(p[1], m, self.w - m - 1)], dtype=np.float32)

    def _generate_single_segment(self, p0=None, num_cp=None, curvature_factor=1.0, allow_self_cross=False, self_cross_prob=0.0, topology="random"):
        """Generates a single Bezier segment and returns (points, control_points)."""
        num_cp = num_cp if num_cp is not None else self.default_num_cp
        m = self._effective_margin()
        
        for attempt in range(50):
            # 1. Generate Control Points based on topology
            if topology == "ribbon":
                # Deterministic self-crossing loop (lemniscate-style) to avoid U-shapes.
                # Generate a figure-eight parametric curve, then rotate/scale into the safe drawable area.
                inner_h = self.h - 2*m
                inner_w = self.w - 2*m

                t = np.linspace(0, 2*np.pi, self.n_samples)
                # Lemniscate of Bernoulli (normalized)
                x = np.cos(t) / (1 + np.sin(t)**2 + 1e-8)
                y = (np.sin(t) * np.cos(t)) / (1 + np.sin(t)**2 + 1e-8)
                pts = np.stack([y, x], axis=1)  # y first, then x to match image coords

                # Normalize to [-0.5, 0.5] range and optionally shrink to avoid margin
                max_abs = np.max(np.abs(pts)) + 1e-6
                pts = (pts / max_abs) * 0.45

                # Random rotation to diversify shapes
                angle = self.rng.uniform(0, 2*np.pi)
                c, s = np.cos(angle), np.sin(angle)
                rot = np.array([[c, -s], [s, c]])
                pts = pts @ rot.T

                # Shift to [0,1] then to pixel coords inside margins
                pts = (pts + 0.5) * np.array([inner_h, inner_w])
                pts = pts + m

                # If chained, anchor start point to p0
                if p0 is not None:
                    offset = p0 - pts[0]
                    pts = pts + offset

                # Small jitter without leaving bounds
                pts = np.array([self._clip_point(pt + self.rng.normal(0, 2, 2), m) for pt in pts])
                control_points = [] 
            elif topology == "hairpin":
                start = p0 if p0 is not None else self._random_point(m)

                # choose forward direction
                theta = self.rng.uniform(0, 2*np.pi)
                d = np.array([np.cos(theta), np.sin(theta)])

                # forward then reverse
                p1 = start + d * self.rng.uniform(30, 60)
                p2 = p1 + self.rng.normal(0, 1, 2) * 20
                p3 = start + self.rng.normal(0, 1, 2) * 10  # return near start

                control_points = np.array([
                    self._clip_point(start, m),
                    self._clip_point(p1, m),
                    self._clip_point(p2, m),
                    self._clip_point(p3, m)
                ])

                pts = _general_bezier(control_points, self.n_samples)
            
            elif topology == "zigzag":
                # Forces a sharp "S" or "Z" shape by pulling CPs in opposite directions
                start = p0 if p0 is not None else self._random_point(m)
                
                # Pick end point reasonably far away
                for _ in range(20):
                    candidate = self._random_point(m)
                    dist = np.linalg.norm(start - candidate)
                    if dist > self.min_dist and dist < min(self.h, self.w) * 0.8:
                        end = candidate
                        break
                else:
                    end = self._clip_point(np.array([self.h - start[0], self.w - start[1]]), m)
                
                # Vector from start to end
                vec = end - start
                dist = np.linalg.norm(vec)
                normal = np.array([-vec[1], vec[0]]) / (dist + 1e-8)
                
                # Amplitude of the zig-zag (sharpness)
                amp = dist * self.rng.uniform(0.4, 0.8) * curvature_factor
                
                # P1 pulls left, P2 pulls right (or vice versa)
                p1 = start + vec * 0.33 + normal * amp
                p2 = start + vec * 0.66 - normal * amp
                
                control_points = np.array([
                    self._clip_point(start, m),
                    self._clip_point(p1, m),
                    self._clip_point(p2, m),
                    self._clip_point(end, m)
                ])
                
                pts = _general_bezier(control_points, self.n_samples)
            # Parametric recipe; no discrete cps
            else:
                # Default "random" topology
                start = p0 if p0 is not None else self._random_point(m)
                end = None
                for _ in range(20):
                    candidate = self._random_point(m)
                    if np.linalg.norm(start - candidate) > self.min_dist:
                        end = candidate
                        break
                if end is None:
                    end = self._clip_point(np.array([self.h - start[0], self.w - start[1]]), m)

                center = (start + end) / 2.0
                image_center = np.array([self.h/2, self.w/2], dtype=np.float32)
                normalized_offset = np.minimum(np.abs(center - image_center) / (image_center - m + 1e-6), 1.0)
                adaptive_factor = 0.9 - (normalized_offset * 0.2)
                
                max_spread = np.array([min(center[0]-m, self.h-m-center[0]), min(center[1]-m, self.w-m-center[1])])
                desired_spread = np.array([self.h, self.w]) * self.spread * curvature_factor
                spread = np.minimum(desired_spread, max_spread * adaptive_factor)

                control_points = [start]
                for i in range(num_cp - 2):
                    if allow_self_cross and self.rng.random() < self_cross_prob:
                        dir_vec = self.rng.normal(0, 1, 2)
                        dir_vec /= (np.linalg.norm(dir_vec) + 1e-8)
                        sign = 1 if i % 2 == 0 else -1
                        cp = center + sign * dir_vec * spread * self.cp_factor
                    else:
                        cp = center + self.rng.normal(0, 1, 2) * spread * self.cp_factor
                    control_points.append(self._clip_point(cp, m))
                control_points.append(end)
                control_points = np.array(control_points)
                pts = _general_bezier(control_points, self.n_samples)
            
            # Simple bounds check
            if np.all(pts >= m) and np.all(pts[:, 0] < self.h - m) and np.all(pts[:, 1] < self.w - m):
                return pts, control_points
                
        # Fallback
        ts = np.linspace(0, 1, self.n_samples)
        start = self._random_point(m)
        end = self._clip_point(start + np.array([20, 20]), m)
        pts_fb = np.outer(1-ts, start) + np.outer(ts, end)
        return pts_fb, np.array([start, end])

    def _draw_aa_curve(self, img, pts, thickness, intensity, width_variation="none", start_width=None, end_width=None,
                       intensity_variation="none", start_intensity=None, end_intensity=None, **kwargs):
        """Standard anti-aliased drawing logic."""
        if intensity_variation == "none":
            start_i = end_i = intensity
        elif intensity_variation == "bright_to_dim":
            start_i, end_i = (start_intensity or intensity*1.5), (end_intensity or intensity*0.5)
        elif intensity_variation == "dim_to_bright":
            start_i, end_i = (start_intensity or intensity*0.5), (end_intensity or intensity*1.5)
        elif intensity_variation == "random":
            # Randomly choose a gradient direction
            if self.rng.random() < 0.5:
                # Bright to Dim
                start_i, end_i = (start_intensity or intensity*1.5), (end_intensity or intensity*0.5)
            else:
                # Dim to Bright
                start_i, end_i = (start_intensity or intensity*0.5), (end_intensity or intensity*1.5)

        else:
            start_i, end_i = (start_intensity or intensity), (end_intensity or intensity)

        if width_variation == "none":
            if intensity_variation == "none":
                pts_int = (pts[:, ::-1] * 16).astype(np.int32).reshape((-1, 1, 2))
                canvas = np.zeros((self.h, self.w), dtype=np.uint8)
                cv2.polylines(canvas, [pts_int], False, 255, int(thickness), cv2.LINE_AA, 4)
                img[:] = np.maximum(img, (canvas.astype(np.float32)/255.0) * intensity)
            else:
                self._draw_segmented(img, pts, thickness, start_i, end_i, thickness, thickness)
        else:
            if width_variation == "wide_to_narrow":
                sw, ew = (start_width or thickness*2.0), (end_width or thickness)
            elif width_variation == "narrow_to_wide":
                sw, ew = (start_width or thickness), (end_width or thickness*2.0)
            
            elif width_variation == "random":
                if self.rng.random() < 0.5:
                     sw, ew = (start_width or thickness*2.0), (end_width or thickness)
                else:
                     sw, ew = (start_width or thickness), (end_width or thickness*2.0)
            else:
                sw, ew = (start_width or thickness), (end_width or thickness)
            self._draw_segmented(img, pts, thickness, start_i, end_i, sw, ew)

    def _draw_segmented(self, img, pts, base_thickness, si, ei, sw, ew):
        n_pts = len(pts)
        seg_len = max(1, n_pts // 50)
        for i in range(0, n_pts - 1, seg_len):
            idx_e = min(i + seg_len, n_pts - 1)
            t_s, t_e = i/(n_pts-1), idx_e/(n_pts-1)
            curr_i = si + (ei - si) * (t_s + t_e)/2
            curr_w = sw + (ew - sw) * (t_s + t_e)/2
            
            pts_seg = (pts[i:idx_e+1, ::-1] * 16).astype(np.int32).reshape((-1, 1, 2))
            seg_mask = np.zeros((self.h, self.w), dtype=np.float32)
            cv2.polylines(seg_mask, [pts_seg], False, 1.0, max(1, int(curr_w)), cv2.LINE_AA, 4)
            img[:] = np.maximum(img, seg_mask * curr_i)

    def sample_curve(self, width_range=(2, 2), noise_prob=0.0, tissue_noise_prob=0.0, invert_prob=0.0, min_intensity=0.6,
                     max_intensity=None, branches=False, curvature_factor=1.0, 
                     num_control_points=None, num_segments=None, centerline_mask=None, 
                     topology=None, return_control_points=False, **kwargs):
        """Unified sample method.
        
        Args:
            num_control_points: Fixed int OR (min, max) tuple.
            num_segments: Fixed int OR (min, max) tuple.
            topology: "random", "ribbon", or "multi" (randomly pick between them).
            tissue_noise_prob: Probability of adding low-frequency background texture.
        """
        # 1. Resolve num_control_points
        cp_val = num_control_points if num_control_points is not None else self.default_num_cp
        if isinstance(cp_val, (tuple, list)):
            num_cp = int(self.rng.integers(cp_val[0], cp_val[1] + 1))
        else:
            num_cp = int(cp_val)

        # 2. Resolve num_segments
        seg_val = num_segments if num_segments is not None else self.default_num_segments
        if isinstance(seg_val, (tuple, list)):
            num_seg = int(self.rng.integers(seg_val[0], seg_val[1] + 1))
        else:
            num_seg = int(seg_val)

        # 3. Resolve Topology
        topol = topology or self.config.get('bezier', {}).get('topology', 'random')
        if topol == "multi":
        # 50% Random (General), 20% Zigzag (Sharp), 15% Ribbon (Loops), 15% Hairpin (U-turns)
            r_val = self.rng.random()
            if r_val < 0.50:
                topol = "random"
            elif r_val < 0.70:
                topol = "zigzag"
            elif r_val < 0.85:
                topol = "ribbon"
            else:
                topol = "hairpin"

        # 4. Resolve centerline_mask
        c_mask = centerline_mask if centerline_mask is not None else self.default_centerline_mask
        
        bg_i = kwargs.get('background_intensity', 0.0)
        img = np.full((self.h, self.w), bg_i, dtype=np.float32)
        mask = np.zeros_like(img)
        
        # Base thickness
        thickness = self.rng.integers(width_range[0], width_range[1] + 1)
        intensity = self.rng.uniform(min_intensity, max_intensity or 1.0)

        # Generate points (chained segments)
        all_segments = []
        all_control_points = []
        curr_p0 = None
        for _ in range(num_seg):
            seg, seg_cps = self._generate_single_segment(p0=curr_p0, num_cp=num_cp, curvature_factor=curvature_factor,
                                                         allow_self_cross=kwargs.get('allow_self_cross', False),
                                                         self_cross_prob=kwargs.get('self_cross_prob', 0.0),
                                                         topology=topol)
            if all_segments:
                seg = seg[1:] # Avoid duplicate point
            all_segments.append(seg)
            all_control_points.append(seg_cps)
            curr_p0 = seg[-1]
        pts_main = np.concatenate(all_segments, axis=0)

        # Draw Image
        self._draw_aa_curve(img, pts_main, thickness, intensity, **kwargs)
        
        # Draw Mask
        m_thick = 1.0 if c_mask else thickness
        m_var = "none" if c_mask else kwargs.get("width_variation", "none")
        self._draw_aa_curve(mask, pts_main, m_thick, 1.0, width_variation=m_var)
        
        pts_all = [pts_main]
        # Branches (always uses 4-point cubic for simplicity)
        if branches:
            n_b = self.rng.integers(self.branch_num_range[0], self.branch_num_range[1] + 1)
            for _ in range(n_b):
                idx = self.rng.integers(int(len(pts_main)*0.2), int(len(pts_main)*0.8))
                b_pts, b_cps = self._generate_single_segment(p0=pts_main[idx], num_cp=4, curvature_factor=curvature_factor*0.7)
                pts_all.append(b_pts)
                all_control_points.append(b_cps)
                b_thick = max(1, int(thickness * self.branch_thickness_factor))
                self._draw_aa_curve(img, b_pts, b_thick, intensity * 0.8, **kwargs)
                self._draw_aa_curve(mask, b_pts, 1.0 if c_mask else b_thick, 1.0, width_variation=m_var)

        # --- Post-Processing (Visual Artifacts) ---
        
        # 1. DSA Noise (Blobs, static, blur)
        if self.rng.random() < noise_prob: 
            self._apply_dsa_noise(img)
            
        # 2. Tissue Noise (Integrated)
        if self.rng.random() < tissue_noise_prob:
            self._apply_tissue_noise(img)
            
        # 3. Final Inversion (Optional)
        if self.rng.random() < invert_prob: 
            img = 1.0 - img
        
        img = np.clip(img, 0.0, 1.0)
        mask = (mask > 0.1).astype(np.uint8)
        if return_control_points:
            return img, mask, pts_all, all_control_points
        return img, mask, pts_all

    def _apply_dsa_noise(self, img):
        num_blobs = self.rng.integers(self.noise_num_blobs_range[0], self.noise_num_blobs_range[1] + 1)
        for _ in range(num_blobs):
            y, x = self.rng.integers(0, self.h), self.rng.integers(0, self.w)
            sigma = self.rng.uniform(self.noise_blob_sigma_range[0], self.noise_blob_sigma_range[1])
            yy, xx = np.ogrid[:self.h, :self.w]
            blob = np.exp(-((yy-y)**2 + (xx-x)**2)/(2*sigma**2)) * self.rng.uniform(self.noise_blob_intensity_range[0], self.noise_blob_intensity_range[1])
            img[:] = np.maximum(img, blob)
        img[:] += self.rng.normal(0, self.rng.uniform(self.noise_level_range[0], self.noise_level_range[1]), img.shape)
        if self.rng.random() < self.noise_blur_prob:
            img[:] = scipy.ndimage.gaussian_filter(img, self.rng.uniform(self.noise_blur_sigma_range[0], self.noise_blur_sigma_range[1]))

    def _apply_tissue_noise(self, img):
        """Adds low-frequency background texture (simulating overlapping tissue)."""
        num_blobs = self.rng.integers(1, 4)
        h, w = img.shape
        tissue_map = np.zeros_like(img)
        
        for _ in range(num_blobs):
            sigma = self.rng.uniform(*self.tissue_sigma_range)
            intensity = self.rng.uniform(*self.tissue_intensity_range)
            # Create a larger Gaussian blob at random location
            y = self.rng.integers(0, h)
            x = self.rng.integers(0, w)
            
            # Simple manual blob for efficiency
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            dist_sq = (yy - y)**2 + (xx - x)**2
            # Tissue blobs are much larger and softer than DSA blobs
            blob = np.exp(-dist_sq / (2 * (sigma*10)**2)) * intensity
            tissue_map = np.maximum(tissue_map, blob)
            
        img[:] = np.clip(img + tissue_map, 0, 1)


# Alias for backward compatibility
CurveMakerFlexible = CurveMaker
