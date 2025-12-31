# Curve Generation Configuration

All curve generation parameters are now configurable via `curve_config.json`. This allows you to easily customize curve properties without modifying code.

## Configuration File Structure

The `curve_config.json` file contains the following sections:

### 1. Image Settings
```json
"image": {
  "height": 128,
  "width": 128
}
```

### 2. Bezier Curve Generation
```json
"bezier": {
  "n_samples": 1000,              // Number of points to sample along curve
  "margin": 10,                   // Margin from image edges for random points
  "min_distance": 40.0,          // Minimum distance between start and end points
  "control_point_spread": 0.3,   // How much control points deviate (0.3 = normal)
  "control_point_factor": 0.6     // Additional factor for control point spread
}
```

### 3. Branch Parameters
```json
"branches": {
  "num_branches_range": [1, 3],   // Number of branches (min, max)
  "start_range": [0.2, 0.8],     // Where branches start (fraction of main curve)
  "thickness_factor": 0.7         // Branch thickness relative to main curve
}
```

### 4. DSA Noise Parameters
```json
"noise": {
  "num_blobs_range": [1, 4],                    // Number of noise blobs
  "blob_sigma_range": [2.0, 8.0],              // Blob size (sigma)
  "blob_intensity_range": [0.05, 0.2],         // Blob intensity
  "noise_level_range": [0.05, 0.15],           // Gaussian noise level
  "gaussian_blur_prob": 0.5,                   // Probability of applying blur
  "gaussian_blur_sigma_range": [0.5, 1.0]      // Blur sigma range
}
```

### 5. Tissue Noise Parameters
```json
"tissue_noise": {
  "sigma_range": [2.0, 5.0],      // Tissue noise blur sigma
  "intensity_range": [0.2, 0.4]   // Tissue noise intensity
}
```

### 6. Training Stages Configuration
**NEW**: Training stages are now fully configurable! You can specify:
- Number of stages (not limited to 3)
- Episodes per stage
- Learning rate per stage
- Curve generation parameters per stage
- Training difficulty parameters per stage

```json
"training_stages": [
  {
    "stage_id": 1,
    "name": "Stage1_Bootstrap",
    "episodes": 8000,                    // Number of episodes for this stage
    "learning_rate": 0.0001,            // Learning rate for this stage
    "curve_generation": {
      "width_range": [1, 2],            // Curve width in pixels (min, max)
      "noise_prob": 0.0,                // Probability of adding DSA noise during generation
      "invert_prob": 0.5,               // Probability of inverting image
      "min_intensity": 0.08,            // Minimum curve intensity (lower = more transparent)
      "max_intensity": 0.20,            // Maximum curve intensity
      "branches": false,                // Whether to add branches
      "curvature_factor": 0.5           // Curve complexity (0.5 = straighter, 1.5 = more curved)
    },
    "training": {
      "noise": 0.0,                     // Noise probability during training (0.0-1.0)
      "tissue": false,                  // Whether to add tissue noise
      "strict_stop": false,             // Whether to require precise stopping
      "mixed_start": false              // Whether to use mixed starting positions
    }
  },
  {
    "stage_id": 2,
    "name": "Stage2_Robustness",
    "episodes": 12000,
    "learning_rate": 0.00005,
    "curve_generation": { ... },
    "training": { ... }
  },
  {
    "stage_id": 3,
    "name": "Stage3_Realism",
    "episodes": 15000,
    "learning_rate": 0.00001,
    "curve_generation": { ... },
    "training": { ... }
  }
]
```

**Note**: The old `"stages"` format is still supported for backward compatibility (used only for curve generation parameters), but `"training_stages"` is the recommended format as it includes both curve generation and training parameters.

### 7. Legacy Stage-Specific Parameters (Backward Compatibility)
The old format is still supported:

```json
"stages": {
  "1": {
    "name": "Stage1_Bootstrap",
    "width_range": [1, 2],
    "noise_prob": 0.0,
    "invert_prob": 0.5,
    "min_intensity": 0.08,
    "max_intensity": 0.20,
    "branches": false,
    "curvature_factor": 0.5
  },
  "2": { ... },
  "3": { ... }
}
```

## Usage

### Default Configuration
If `curve_config.json` exists in the Experiment1 directory, it will be automatically loaded:

```bash
# Uses curve_config.json automatically
./run_train.sh
```

### Custom Configuration File
Specify a custom config file:

```bash
# Using Python directly
python3 src/train.py --curve_config my_custom_config.json

# Using bash script
./run_train.sh --curve_config my_custom_config.json
```

### Curve Generator Script
The curve generator also uses the config:

```bash
# Uses default curve_config.json
python3 src/curve_generator.py --num_curves 1000 --all_stages

# Custom config
python3 src/curve_generator.py --config my_config.json --num_curves 1000
```

## Parameter Descriptions

### Stage Parameters

- **width_range**: `[min, max]` - Curve thickness in pixels
  - Narrow: `[1, 2]` or `[1, 3]`
  - Wide: `[2, 8]` or `[1, 10]`

- **width_variation**: `"none"`, `"wide_to_narrow"`, `"narrow_to_wide"`, or `"custom"` - How width changes along the curve
  - `"none"`: Constant width (default)
  - `"wide_to_narrow"`: Starts wide, tapers to narrow
  - `"narrow_to_wide"`: Starts narrow, widens along curve
  - `"custom"`: Use `start_width` and `end_width` explicitly

- **start_width**: Starting width in pixels (for `"custom"` or overrides for `"wide_to_narrow"`/`"narrow_to_wide"`)

- **end_width**: Ending width in pixels (for `"custom"` or overrides for `"wide_to_narrow"`/`"narrow_to_wide"`)

- **intensity_variation**: `"none"`, `"bright_to_dim"`, `"dim_to_bright"`, or `"custom"` - How intensity changes along the curve
  - `"none"`: Constant intensity (default)
  - `"bright_to_dim"`: Starts bright, fades to dim
  - `"dim_to_bright"`: Starts dim, brightens along curve
  - `"custom"`: Use `start_intensity` and `end_intensity` explicitly

- **start_intensity**: Starting intensity (for variable intensity curves)

- **end_intensity**: Ending intensity (for variable intensity curves)

- **background_intensity**: Background intensity `0.0-1.0` (default: `0.0` for black background)
  - Use higher values (e.g., `0.3-0.5`) for low contrast with light backgrounds
  - Use lower values (e.g., `0.0-0.1`) for dark backgrounds

- **min_intensity / max_intensity**: Curve transparency/opacity range
  - Low (transparent): `0.05-0.20`
  - Medium: `0.2-0.5`
  - High (opaque): `0.6-1.0`

- **curvature_factor**: Controls how curved the paths are
  - Straight: `0.3-0.5`
  - Normal: `1.0`
  - Very curved: `1.5-2.0`

- **branches**: `true/false` - Whether to add branch curves

- **invert_prob**: `0.0-1.0` - Probability of inverting the image (dark on light vs light on dark)

### Bezier Parameters

- **n_samples**: Number of points along the curve (more = smoother)
- **margin**: Safe margin from image edges
- **min_distance**: Prevents curves that are too short/blobby
- **control_point_spread**: How much curves can deviate (higher = more varied shapes)
- **control_point_factor**: Additional multiplier for control point deviation

### Branch Parameters

- **num_branches_range**: How many branches to add
- **start_range**: Where branches start (0.2 = 20% along main curve, 0.8 = 80%)
- **thickness_factor**: Branch thickness relative to main curve (0.7 = 70% of main thickness)

### Noise Parameters

- **num_blobs_range**: Number of noise blobs to add
- **blob_sigma_range**: Size of noise blobs
- **blob_intensity_range**: How bright the blobs are
- **noise_level_range**: Overall noise level
- **gaussian_blur_prob**: Chance of applying blur
- **gaussian_blur_sigma_range**: Blur amount

## Training Stage Parameters

### Stage Configuration Fields

- **stage_id**: Unique identifier for the stage (1, 2, 3, ...)
- **name**: Human-readable name for the stage
- **episodes**: Number of training episodes for this stage
- **learning_rate**: Learning rate for PPO optimizer (typically decreases: 1e-4 → 5e-5 → 1e-5)

### Training Parameters

- **noise**: `0.0-1.0` - Probability of adding noise during training (increases difficulty)
- **tissue**: `true/false` - Whether to add tissue noise (simulates background texture)
- **strict_stop**: `true/false` - If true, requires precise stopping at end (harder)
- **mixed_start**: `true/false` - If true, agent starts at random positions (harder)

### Adding More Stages

You can add as many stages as you want! Just add more entries to the `training_stages` array:

```json
"training_stages": [
  { "stage_id": 1, ... },
  { "stage_id": 2, ... },
  { "stage_id": 3, ... },
  { "stage_id": 4, "name": "Stage4_Expert", "episodes": 20000, ... }
]
```

## Examples

### Very Narrow, Very Transparent Curves
```json
"stages": {
  "1": {
    "width_range": [1, 1],
    "min_intensity": 0.03,
    "max_intensity": 0.10,
    "curvature_factor": 0.3
  }
}
```

### Wide, High Contrast Curves
```json
"stages": {
  "1": {
    "width_range": [3, 6],
    "min_intensity": 0.7,
    "max_intensity": 1.0,
    "curvature_factor": 1.2
  }
}
```

### More Complex Curves with Many Branches
```json
"branches": {
  "num_branches_range": [2, 5],
  "start_range": [0.1, 0.9]
}
```

### Wide-to-Narrow Tapering Curves
```json
"curve_generation": {
  "width_range": [1, 4],
  "width_variation": "wide_to_narrow",
  "start_width": 6,
  "end_width": 1
}
```

### Narrow-to-Wide Expanding Curves
```json
"curve_generation": {
  "width_range": [1, 4],
  "width_variation": "narrow_to_wide",
  "start_width": 1,
  "end_width": 8
}
```

### Custom Width Profile
```json
"curve_generation": {
  "width_range": [1, 10],
  "width_variation": "custom",
  "start_width": 5,
  "end_width": 2
}
```

### Low Contrast with Variable Lightness
```json
"curve_generation": {
  "width_range": [1, 2],
  "min_intensity": 0.15,
  "max_intensity": 0.25,
  "intensity_variation": "bright_to_dim",
  "start_intensity": 0.25,
  "end_intensity": 0.10,
  "background_intensity": 0.20
}
```

### Very Low Contrast (Subtle Curves)
```json
"curve_generation": {
  "width_range": [1, 2],
  "min_intensity": 0.08,
  "max_intensity": 0.15,
  "intensity_variation": "dim_to_bright",
  "background_intensity": 0.10
}
```

### Variable Lightness Along Curve
```json
"curve_generation": {
  "width_range": [1, 3],
  "min_intensity": 0.10,
  "max_intensity": 0.30,
  "intensity_variation": "custom",
  "start_intensity": 0.12,
  "end_intensity": 0.28,
  "background_intensity": 0.15
}
```

## Fallback Behavior

If `curve_config.json` is not found or a parameter is missing:
- The code will use hardcoded defaults
- A warning message will be printed
- Training will continue normally

This ensures backward compatibility and prevents errors from missing config files.

