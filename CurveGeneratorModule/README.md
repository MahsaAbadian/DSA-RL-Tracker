# Curve Generator Module

A centralized module for curve generation used across all experiments. This module provides:

- **Configuration loading**: Read curve configuration from JSON files
- **Curve generation**: Generate synthetic curves on-the-fly with flexible parameters
- **Config snapshots**: Save configuration snapshots for reproducibility

## Installation

Install dependencies using a virtual environment (recommended for externally-managed Python environments):

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: If your Python environment is externally managed (common on macOS with Homebrew), you'll need to use a virtual environment. The `generate.sh` script in `ExampleConfigs/` will automatically activate the venv if it exists.

## Usage

### Basic Usage

```python
from CurveGeneratorModule import load_curve_config, CurveMakerFlexible, save_config_snapshot

# Load configuration from JSON
config, config_path = load_curve_config("path/to/curve_config.json")

# Create curve generator
img_cfg = config.get('image', {})
h = img_cfg.get('height', 128)
w = img_cfg.get('width', 128)
generator = CurveMakerFlexible(h=h, w=w, seed=42, config=config)

# Generate a curve
img, mask, pts_all = generator.sample_curve(
    width_range=(2, 4),
    noise_prob=0.0,
    invert_prob=0.5,
    min_intensity=0.6,
    max_intensity=0.8,
    branches=False,
    curvature_factor=1.0
)

# Save config snapshot for reproducibility
save_config_snapshot(config, "runs/my_run/curve_config.json")
```

### Configuration Structure

The curve config JSON typically has the following structure:

```json
{
  "image": {
    "height": 128,
    "width": 128
  },
  "bezier": {
    "n_samples": 1000,
    "margin": 10,
    "min_distance": 40.0,
    "control_point_spread": 0.3,
    "control_point_factor": 0.6
  },
  "branches": {
    "num_branches_range": [1, 3],
    "start_range": [0.2, 0.8],
    "thickness_factor": 0.7
  },
  "noise": {
    "num_blobs_range": [1, 4],
    "blob_sigma_range": [2.0, 8.0],
    "blob_intensity_range": [0.05, 0.2],
    "noise_level_range": [0.05, 0.15],
    "gaussian_blur_prob": 0.5,
    "gaussian_blur_sigma_range": [0.5, 1.0]
  },
  "training_stages": [
    {
      "stage_id": 1,
      "name": "Stage1_Foundation",
      "curve_generation": {
        "width_range": [3, 5],
        "noise_prob": 0.0,
        "invert_prob": 0.5,
        "min_intensity": 0.8,
        "max_intensity": 1.0,
        "branches": false,
        "curvature_factor": 0.3,
        ...
      }
    }
  ]
}
```

### Loading Configuration

The `load_curve_config()` function searches for config files in the following order:

1. Explicit path provided as argument
2. `config/curve_config.json` relative to calling script
3. `curve_config.json` relative to calling script

If no config is found, it returns an empty dict and uses default values.

### Saving Config Snapshots

When running training, you should save a snapshot of the config used:

```python
# After loading config
config, config_path = load_curve_config("config/curve_config.json")

# Save snapshot to run directory
run_dir = "runs/my_experiment_20240101_120000"
save_config_snapshot(config, os.path.join(run_dir, "curve_config.json"))
```

This ensures reproducibility - each run has its exact config saved.

## Module Structure

- `config_loader.py`: Functions to load and save curve configuration JSON files
- `generator.py`: The `CurveMakerFlexible` class that generates single-segment Bezier curves
- `generator_multisegment.py`: The `CurveMakerMultiSegment` class that generates multi-segment Bezier curves (longer curves)
- `__init__.py`: Module exports
- `ExampleConfigs/`: Tools to visualize and generate curve grids for experiments
- `examples/`: Example scripts demonstrating usage of both generators

## Notes

- Curve generation is always **on-the-fly** - no pre-generation needed
- Configs can differ between experiments and runs
- Always save a config snapshot for each training run for reproducibility
