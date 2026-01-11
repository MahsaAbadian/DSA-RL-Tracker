#!/usr/bin/env python3
"""
Multi-Segment Bezier Curve Generator for DSA RL Experiment
Generates longer curves by chaining multiple Bezier segments together.
"""
import numpy as np
import cv2
import scipy.ndimage
from typing import Optional, Tuple, List, Dict, Any
from .generator import CurveMakerFlexible, _cubic_bezier


class CurveMakerMultiSegment(CurveMakerFlexible):
    """Multi-segment Bezier curve generator that chains multiple Bezier curves.
    
    Extends CurveMakerFlexible to generate longer curves by connecting multiple
    Bezier segments end-to-end. Each segment shares an endpoint with the next.
    
    Example:
        from CurveGeneratorModule import load_curve_config, CurveMakerMultiSegment
        
        config, _ = load_curve_config("path/to/curve_config.json")
        generator = CurveMakerMultiSegment(h=128, w=128, seed=42, config=config)
        img, mask, pts = generator.sample_curve(num_segments=3, ...)
    """
    
    def __init__(self, h=128, w=128, seed=None, config=None):
        """Initialize the multi-segment curve generator.
        
        Args:
            h: Image height
            w: Image width
            seed: Random seed for reproducibility
            config: Configuration dictionary (same as CurveMakerFlexible)
        """
        super().__init__(h, w, seed, config)
        # Default number of segments for multi-segment curves (now fixed at 2 segments)
        self.default_num_segments = config.get('multi_segment', {}).get('num_segments_range', [2, 2]) if config else [2, 2]
    
    def _generate_multi_segment_points(self, num_segments=None, curvature_factor=1.0,
                                       allow_self_cross=False, self_cross_prob=0.0,
                                       segment_length_factor=1.0):
        """Generates a multi-segment Bezier curve by chaining segments together.
        
        Args:
            num_segments: Number of Bezier segments to chain (if None, random from default range)
            curvature_factor: Controls curve complexity per segment
            allow_self_cross: Whether to allow self-crossing curves
            self_cross_prob: Probability of self-crossing per segment
            segment_length_factor: Factor to control average segment length (1.0 = normal, >1.0 = longer)
        
        Returns:
            Array of (y, x) points forming the complete multi-segment curve
        """
        if num_segments is None:
            if isinstance(self.default_num_segments, list):
                num_segments = self.rng.integers(self.default_num_segments[0], self.default_num_segments[1] + 1)
            else:
                num_segments = self.default_num_segments
        
        num_segments = max(1, int(num_segments))
        m = self._effective_margin()
        
        all_points = []
        
        # Generate first segment
        p0 = self._random_point(margin=m)
        pts_segment = self._generate_bezier_points(
            p0=p0,
            curvature_factor=curvature_factor,
            allow_self_cross=allow_self_cross,
            self_cross_prob=self_cross_prob
        )
        all_points.append(pts_segment)
        current_end = pts_segment[-1]  # Last point of first segment
        
        # Generate subsequent segments, each starting where the previous ended
        for seg_idx in range(1, num_segments):
            # Adjust min_distance based on segment_length_factor
            original_min_dist = self.bezier_min_distance
            adjusted_min_dist = original_min_dist * segment_length_factor
            self.bezier_min_distance = adjusted_min_dist
            
            try:
                # Generate next segment starting from current_end
                pts_segment = self._generate_bezier_points(
                    p0=current_end,
                    curvature_factor=curvature_factor,
                    allow_self_cross=allow_self_cross,
                    self_cross_prob=self_cross_prob
                )
                
                # Remove first point to avoid duplicate (it's the same as current_end)
                if len(pts_segment) > 1:
                    pts_segment = pts_segment[1:]
                
                all_points.append(pts_segment)
                current_end = pts_segment[-1]
            finally:
                # Restore original min_distance
                self.bezier_min_distance = original_min_dist
        
        # Concatenate all segments
        if all_points:
            combined_points = np.concatenate(all_points, axis=0)
            # Ensure all points are within bounds
            combined_points = self._clip_points(combined_points, margin=m)
            return combined_points
        else:
            # Fallback: single segment
            return self._generate_bezier_points(
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
                     self_cross_prob=0.0,
                     num_segments=None,
                     segment_length_factor=1.0,
                     centerline_mask=False):
        """Generate a multi-segment curve with specified parameters.
        
        Args:
            All parameters same as CurveMakerFlexible.sample_curve(), plus:
            num_segments: Number of Bezier segments to chain (if None, random from config)
            segment_length_factor: Factor to control average segment length (1.0 = normal, >1.0 = longer)
            centerline_mask: If True, the mask will only be 1 pixel thick regardless of curve width.
        
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
        
        # Generate multi-segment curve
        pts_main = self._generate_multi_segment_points(
            num_segments=num_segments,
            curvature_factor=curvature_factor,
            allow_self_cross=allow_self_cross,
            self_cross_prob=self_cross_prob,
            segment_length_factor=segment_length_factor
        )
        
        pts_all = [pts_main]
        
        # Draw main curve on image
        self._draw_aa_curve(img, pts_main, thickness, intensity,
                           width_variation, start_width, end_width,
                           "none", None, None) # Intensity handled below
        
        # Re-draw with correct intensity logic if needed
        # (This replicates generator.py logic which was missing here)
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

        # Clear and redraw with correct parameters
        img.fill(bg_intensity)
        self._draw_aa_curve(img, pts_main, thickness, intensity,
                           width_variation, start_w, end_w,
                           intensity_variation, start_i, end_i)
        
        # Create mask
        m_thick = 1.0 if centerline_mask else thickness
        m_start_w = 1.0 if centerline_mask else start_w
        m_end_w = 1.0 if centerline_mask else end_w
        m_variation = "none" if centerline_mask else width_variation
        
        self._draw_aa_curve(mask, pts_main, m_thick, 1.0, 
                           m_variation, m_start_w, m_end_w)
        
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
                
                # Generate branch curve
                branch_thickness = max(1, int(thickness * thickness_factor))
                branch_intensity = intensity * 0.8
                
                # Branches use same width/intensity variation pattern but scaled
                b_start_w = max(1, int(start_w * thickness_factor)) if width_variation != "none" else branch_thickness
                b_end_w = max(1, int(end_w * thickness_factor)) if width_variation != "none" else branch_thickness
                
                branch_pts = self._generate_bezier_points(
                    p0=branch_start,
                    curvature_factor=curvature_factor * 0.7,
                    allow_self_cross=False,
                    self_cross_prob=0.0
                )
                
                pts_all.append(branch_pts)
                
                # Draw branch on image
                self._draw_aa_curve(img, branch_pts, branch_thickness, branch_intensity,
                                   width_variation, b_start_w, b_end_w,
                                   intensity_variation, start_i, end_i)
                
                # Draw branch on mask
                bm_thick = 1.0 if centerline_mask else branch_thickness
                bm_start_w = 1.0 if centerline_mask else b_start_w
                bm_end_w = 1.0 if centerline_mask else b_end_w
                bm_variation = "none" if centerline_mask else width_variation
                
                self._draw_aa_curve(mask, branch_pts, bm_thick, 1.0, 
                                   bm_variation, bm_start_w, bm_end_w)
        
        # Add noise if requested
        if noise_prob > 0.0 and self.rng.random() < noise_prob:
            noise_cfg = self.config.get('noise', {}) if self.config else {}
            num_blobs_range = noise_cfg.get('num_blobs_range', [1, 4])
            blob_sigma_range = noise_cfg.get('blob_sigma_range', [2.0, 8.0])
            blob_intensity_range = noise_cfg.get('blob_intensity_range', [0.05, 0.2])
            noise_level_range = noise_cfg.get('noise_level_range', [0.05, 0.15])
            gaussian_blur_prob = noise_cfg.get('gaussian_blur_prob', 0.5)
            gaussian_blur_sigma_range = noise_cfg.get('gaussian_blur_sigma_range', [0.5, 1.0])
            
            num_blobs = self.rng.integers(num_blobs_range[0], num_blobs_range[1] + 1)
            
            for _ in range(num_blobs):
                blob_y = self.rng.integers(0, self.h)
                blob_x = self.rng.integers(0, self.w)
                blob_sigma = self.rng.uniform(blob_sigma_range[0], blob_sigma_range[1])
                blob_intensity = self.rng.uniform(blob_intensity_range[0], blob_intensity_range[1])
                
                y, x = np.ogrid[:self.h, :self.w]
                dist_sq = (y - blob_y)**2 + (x - blob_x)**2
                blob = np.exp(-dist_sq / (2 * blob_sigma**2)) * blob_intensity
                img = np.maximum(img, blob)
            
            # Add general noise
            noise_level = self.rng.uniform(noise_level_range[0], noise_level_range[1])
            noise = self.rng.normal(0, noise_level, (self.h, self.w))
            img = np.clip(img + noise, 0, 1)
            
            # Apply Gaussian blur if requested
            if self.rng.random() < gaussian_blur_prob:
                blur_sigma = self.rng.uniform(gaussian_blur_sigma_range[0], gaussian_blur_sigma_range[1])
                img = scipy.ndimage.gaussian_filter(img, sigma=blur_sigma)
        
        # Invert if requested
        if invert_prob > 0.0 and self.rng.random() < invert_prob:
            img = 1.0 - img
        
        return img, mask, pts_all

