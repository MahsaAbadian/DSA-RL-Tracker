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
import json



def _rng(seed=None):
    return np.random.default_rng(seed)

def _cubic_bezier(p0, p1, p2, p3, t):
    """Standard Cubic Bezier formula."""
    omt = 1.0 - t
    return (omt**3)*p0 + 3*omt*omt*t*p1 + 3*omt*t*t*p2 + (t**3)*p3

class CurveMakerFlexible:
    def __init__(self, h=128, w=128, seed=None, config=None):
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

    def _random_point(self, margin=None):
        margin = margin if margin is not None else self.bezier_margin
        y = self.rng.integers(margin, self.h - margin)
        x = self.rng.integers(margin, self.w - margin)
        return np.array([y, x], dtype=np.float32)

    def _generate_bezier_points(self, p0=None, n_samples=None, curvature_factor=1.0,
                                allow_self_cross=False, self_cross_prob=0.0):
        """Generates a list of (y,x) points forming a smooth bezier curve."""
        n_samples = n_samples if n_samples is not None else self.bezier_n_samples
        
        if p0 is None:
            p0 = self._random_point()
        
        # Ensure p3 is far enough from p0 (prevents blobs)
        for _ in range(20): 
            p3 = self._random_point()
            dist = np.linalg.norm(p0 - p3)
            if dist > self.bezier_min_distance: 
                break
        else:
            p3 = np.array([self.h - p0[0], self.w - p0[1]], dtype=np.float32)

        # Control points - curvature_factor affects how much control points deviate
        center = (p0 + p3) / 2.0
        spread = np.array([self.h, self.w], dtype=np.float32) * self.bezier_spread * curvature_factor

        if allow_self_cross and self.rng.random() < self_cross_prob:
            # Force control points to opposite sides of the center to encourage a self-cross
            dir_vec = self.rng.normal(0, 1, 2)
            norm = np.linalg.norm(dir_vec) + 1e-8
            dir_unit = dir_vec / norm
            p1 = center + dir_unit * spread * self.bezier_factor
            p2 = center - dir_unit * spread * self.bezier_factor
        else:
            p1 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor
            p2 = center + self.rng.normal(0, 1, 2) * spread * self.bezier_factor
        
        ts = np.linspace(0, 1, n_samples, dtype=np.float32)
        pts = np.stack([_cubic_bezier(p0, p1, p2, p3, t) for t in ts], axis=0)
        
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


def load_curve_config(config_path=None):
    """Load curve generation configuration from JSON file.
    
    Returns:
        tuple: (config_dict, actual_config_path)
        - config_dict: The loaded configuration dictionary (or empty dict if not found)
        - actual_config_path: The path that was actually used (or None if not found)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Experiment1 directory
    config_dir = os.path.join(parent_dir, "config")  # Experiment1/config/
    
    if config_path is None:
        # Default: look in config/ directory first, then parent directory for backward compatibility
        default_paths = [
            os.path.join(config_dir, "curve_config.json"),
            os.path.join(parent_dir, "curve_config.json")
        ]
        for path in default_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            config_path = default_paths[0]  # Use config/ path even if doesn't exist
    
    # Convert to absolute path if relative
    if not os.path.isabs(config_path):
        # Try relative to config directory first
        test_path = os.path.join(config_dir, config_path)
        if os.path.exists(test_path):
            config_path = test_path
        else:
            # Try relative to parent directory
            test_path = os.path.join(parent_dir, config_path)
            if os.path.exists(test_path):
                config_path = test_path
            else:
                # Try relative to script directory (for backward compatibility)
                test_path = os.path.join(script_dir, config_path)
                if os.path.exists(test_path):
                    config_path = test_path
                else:
                    # Use config directory as default
                    config_path = os.path.join(config_dir, config_path)
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded curve configuration from: {config_path}")
        return config, config_path
    else:
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using default configuration")
        return {}, None


def _save_grid(images, grid_path, cols=5):
    """Save a list of images into a tiled grid (PNG)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; cannot save grid.")
        return
    if not images:
        return
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(3*cols, 3*rows))
    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(grid_path), exist_ok=True)
    plt.savefig(grid_path, dpi=150)
    plt.close()


def save_stage_examples(config_path="config/curve_config.json", samples_per_stage=5, output_dir="runs/stage_examples", seed=1234):
    """Generate and save example curves for each training stage."""
    config, _ = load_curve_config(config_path)
    img_cfg = config.get('image', {})
    h = img_cfg.get('height', 128)
    w = img_cfg.get('width', 128)
    training_stages = config.get('training_stages', [])
    if not training_stages:
        print("No training_stages found in config.")
        return

    for stage in training_stages:
        stage_id = stage.get('stage_id', 'X')
        stage_name = stage.get('name', f"Stage{stage_id}")
        cg = stage.get('curve_generation', {})

        stage_dir = os.path.join(output_dir, f"stage{stage_id}_{stage_name.replace(' ', '_')}")
        # Clean old pngs (including prior sample_*.png and grid.png)
        if os.path.exists(stage_dir):
            for f in os.listdir(stage_dir):
                if f.lower().endswith(".png"):
                    try:
                        os.remove(os.path.join(stage_dir, f))
                    except OSError:
                        pass
        os.makedirs(stage_dir, exist_ok=True)

        images = []
        for i in range(samples_per_stage):
            maker = CurveMakerFlexible(h=h, w=w, seed=seed + int(stage_id) * 100 + i, config=config)
            img, mask, pts_all = maker.sample_curve(
                width_range=tuple(cg.get("width_range", [2, 4])),
                noise_prob=cg.get("noise_prob", 0.0),
                invert_prob=cg.get("invert_prob", 0.5),
                min_intensity=cg.get("min_intensity", 0.6),
                max_intensity=cg.get("max_intensity", 0.8),
                branches=cg.get("branches", False),
                curvature_factor=cg.get("curvature_factor", 1.0),
                allow_self_cross=cg.get("allow_self_cross", False),
                self_cross_prob=cg.get("self_cross_prob", 0.0),
                width_variation=cg.get("width_variation", "none"),
                start_width=cg.get("start_width", None),
                end_width=cg.get("end_width", None),
                intensity_variation=cg.get("intensity_variation", "none"),
                start_intensity=cg.get("start_intensity", None),
                end_intensity=cg.get("end_intensity", None),
                background_intensity=cg.get("background_intensity", None),
            )
            images.append(img)

        grid_path = os.path.join(stage_dir, "grid.png")
        _save_grid(images, grid_path, cols=5)

    print(f"Saved grids to {output_dir} (grid.png per stage)")


def render_stage_examples(root_dir="runs/stage_examples", cols=5):
    """Render tiled previews of per-stage samples (saved by save_stage_examples)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Safe for headless
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
    except ImportError:
        print("matplotlib not installed; cannot render stage examples.")
        return

    stage_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    if not stage_dirs:
        print(f"No stage directories found under {root_dir}")
        return

    for stage_dir in stage_dirs:
        full_dir = os.path.join(root_dir, stage_dir)
        imgs = sorted([f for f in os.listdir(full_dir)
                       if f.lower().endswith(".png") and f.lower() != "grid.png"])
        if not imgs:
            continue
        rows = (len(imgs) + cols - 1) // cols
        plt.figure(figsize=(3*cols, 3*rows))
        for i, fname in enumerate(imgs):
            path = os.path.join(full_dir, fname)
            img = mpimg.imread(path)
            plt.subplot(rows, cols, i+1)
            if img.ndim == 2:
                plt.imshow(img, cmap="gray")
            else:
                plt.imshow(img)
            plt.axis("off")
            plt.title(fname, fontsize=8)
        plt.tight_layout()
        grid_path = os.path.join(full_dir, "grid.png")
        plt.savefig(grid_path, dpi=150)
        plt.close()
        print(f"Saved grid to {grid_path}")


def save_stage_examples(config_path="config/curve_config.json", samples_per_stage=5, output_dir="runs/stage_examples", seed=1234):
    """Generate and save example curves for each training stage."""
    config, _ = load_curve_config(config_path)
    img_cfg = config.get('image', {})
    h = img_cfg.get('height', 128)
    w = img_cfg.get('width', 128)
    training_stages = config.get('training_stages', [])
    if not training_stages:
        print("No training_stages found in config.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for stage in training_stages:
        stage_id = stage.get('stage_id', 'X')
        stage_name = stage.get('name', f"Stage{stage_id}")
        cg = stage.get('curve_generation', {})

        stage_dir = os.path.join(output_dir, f"stage{stage_id}_{stage_name.replace(' ', '_')}")
        os.makedirs(stage_dir, exist_ok=True)

        for i in range(samples_per_stage):
            maker = CurveMakerFlexible(h=h, w=w, seed=seed + int(stage_id) * 100 + i, config=config)
            img, mask, pts_all = maker.sample_curve(
                width_range=tuple(cg.get("width_range", [2, 4])),
                noise_prob=cg.get("noise_prob", 0.0),
                invert_prob=cg.get("invert_prob", 0.5),
                min_intensity=cg.get("min_intensity", 0.6),
                max_intensity=cg.get("max_intensity", 0.8),
                branches=cg.get("branches", False),
                curvature_factor=cg.get("curvature_factor", 1.0),
                allow_self_cross=cg.get("allow_self_cross", False),
                self_cross_prob=cg.get("self_cross_prob", 0.0),
                width_variation=cg.get("width_variation", "none"),
                start_width=cg.get("start_width", None),
                end_width=cg.get("end_width", None),
                intensity_variation=cg.get("intensity_variation", "none"),
                start_intensity=cg.get("start_intensity", None),
                end_intensity=cg.get("end_intensity", None),
                background_intensity=cg.get("background_intensity", None),
            )
            out_path = os.path.join(stage_dir, f"sample_{i}.png")
            cv2.imwrite(out_path, (img * 255).astype(np.uint8))

    print(f"Saved {samples_per_stage} samples per stage to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic curves")
    parser.add_argument("--output_dir", type=str, default="generated_curves", 
                        help="Output directory for generated curves")
    parser.add_argument("--num_curves", type=int, default=100, 
                        help="Number of curves to generate")
    parser.add_argument("--h", type=int, default=None, help="Image height (overrides config)")
    parser.add_argument("--w", type=int, default=None, help="Image width (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                        help="Stage number (1=simple, 2=medium, 3=complex). If None, generates for all stages.")
    parser.add_argument("--all_stages", action="store_true",
                        help="Generate curves for all 3 stages")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to curve configuration JSON file (default: curve_config.json)")
    parser.add_argument("--stage_examples", type=int, default=0,
                        help="If >0, generate this many example curves per stage to runs/stage_examples and exit")
    parser.add_argument("--render_stage_examples", action="store_true",
                        help="Render tiled previews from runs/stage_examples (saves grid.png in each stage dir)")
    parser.add_argument("--render_cols", type=int, default=5,
                        help="Columns when rendering grids")
    parser.add_argument("--render_dir", type=str, default="runs/stage_examples",
                        help="Directory containing stage example images")
    
    args = parser.parse_args()
    
    # Load configuration
    config, _ = load_curve_config(args.config)
    
    # Get image dimensions (from args, config, or defaults)
    img_cfg = config.get('image', {})
    h = args.h if args.h is not None else img_cfg.get('height', 128)
    w = args.w if args.w is not None else img_cfg.get('width', 128)
    
    # Get absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir, args.output_dir)
    
    # Handle stage examples (generate previews and optionally render grids)
    if args.stage_examples and args.stage_examples > 0:
        save_stage_examples(config_path=args.config or None,
                            samples_per_stage=args.stage_examples,
                            output_dir="runs/stage_examples",
                            seed=args.seed or 1234)
        if args.render_stage_examples:
            render_stage_examples(root_dir=args.render_dir, cols=args.render_cols)
        exit(0)

    # Load stage-specific configurations from training_stages
    training_stages = config.get('training_stages', [])
    
    def get_stage_config(stage_num):
        """Get stage config from training_stages, with fallback defaults."""
        for stage_cfg in training_stages:
            if int(stage_cfg.get('stage_id', -1)) == stage_num:
                cg = stage_cfg.get('curve_generation', {}).copy()
                # Convert lists to tuples for width_range
                if 'width_range' in cg:
                    cg['width_range'] = tuple(cg['width_range'])
                return {
                    'name': stage_cfg.get('name', f'Stage{stage_num}'),
                    'width_range': cg.get('width_range', (2, 4)),
                    'noise_prob': cg.get('noise_prob', 0.0),
                    'invert_prob': cg.get('invert_prob', 0.5),
                    'min_intensity': cg.get('min_intensity', 0.6),
                    'max_intensity': cg.get('max_intensity', 0.8),
                    'branches': cg.get('branches', False),
                    'curvature_factor': cg.get('curvature_factor', 1.0),
                    'allow_self_cross': cg.get('allow_self_cross', False),
                    'self_cross_prob': cg.get('self_cross_prob', 0.0),
                    'width_variation': cg.get('width_variation', 'none'),
                    'start_width': cg.get('start_width', None),
                    'end_width': cg.get('end_width', None),
                    'intensity_variation': cg.get('intensity_variation', 'none'),
                    'start_intensity': cg.get('start_intensity', None),
                    'end_intensity': cg.get('end_intensity', None),
                    'background_intensity': cg.get('background_intensity', None)
                }
        # Fallback defaults (aligned with current main config defaults)
        defaults = {
            1: {'name': 'Stage1_Bootstrap', 'width_range': (2, 4), 'noise_prob': 0.0, 
                'invert_prob': 0.5, 'min_intensity': 0.6, 'max_intensity': 0.8, 'branches': False, 
            'curvature_factor': 0.5, 'allow_self_cross': False, 'self_cross_prob': 0.0,
            'width_variation': 'none', 'start_width': None, 'end_width': None,
            'intensity_variation': 'none', 'start_intensity': None, 'end_intensity': None,
            'background_intensity': None},
            2: {'name': 'Stage2_Robustness', 'width_range': (2, 8), 'noise_prob': 0.3, 
                'invert_prob': 0.5, 'min_intensity': 0.4, 'max_intensity': 0.7, 'branches': False, 
            'curvature_factor': 1.0, 'allow_self_cross': False, 'self_cross_prob': 0.0,
            'width_variation': 'none', 'start_width': None, 'end_width': None,
            'intensity_variation': 'none', 'start_intensity': None, 'end_intensity': None,
            'background_intensity': None},
            3: {'name': 'Stage3_Realism', 'width_range': (1, 10), 'noise_prob': 0.5, 
                'invert_prob': 0.5, 'min_intensity': 0.2, 'max_intensity': 0.6, 'branches': True, 
            'curvature_factor': 1.5, 'allow_self_cross': False, 'self_cross_prob': 0.0,
            'width_variation': 'none', 'start_width': None, 'end_width': None,
            'intensity_variation': 'none', 'start_intensity': None, 'end_intensity': None,
            'background_intensity': None}
        }
        return defaults.get(stage_num, {})

    stage_configs = {
        1: get_stage_config(1),
        2: get_stage_config(2),
        3: get_stage_config(3)
    }
    
    # Add num_curves to each stage config
    for stage_num in [1, 2, 3]:
        stage_configs[stage_num]['num_curves'] = args.num_curves
    
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
        
        cm = CurveMakerFlexible(h=h, w=w, seed=args.seed, config=config)
        print(f"\n=== Generating {config['num_curves']} curves for {config['name']} ===")
        print(f"  Width: {config['width_range']}, Curvature: {config['curvature_factor']:.1f}")
        print(f"  Branches: {config['branches']}, Intensity: {config['min_intensity']:.2f}-{config.get('max_intensity', 1.0):.2f}")
        print(f"  Output: {output_dir}")
        
        for i in range(config['num_curves']):
            img, mask, pts_all = cm.sample_curve(
                width_range=config['width_range'],
                noise_prob=config['noise_prob'],
                invert_prob=config['invert_prob'],
                min_intensity=config['min_intensity'],
                max_intensity=config.get('max_intensity', None),
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
