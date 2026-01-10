# Curve Experiments

This folder contains tools to visualize curve generation for different stage configurations.

## Files

- `generate_stage_grids.py` - Python script that generates 7x7 grids (49 curves) for each stage
- `generate.sh` - Bash script to generate grids (single config or all configs)
- `*_config.json` - Example configuration files from different experiments

## Setup

First, install the required dependencies. Since your Python environment is externally managed, create a virtual environment:

```bash
# From the CurveGeneratorModule directory
cd CurveGeneratorModule

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The scripts will automatically activate the venv if it exists.

## Usage

### Generate grids for a single config

```bash
./generate.sh experiment1
```

Or with the `_config` suffix:

```bash
./generate.sh experiment1_config
```

**Note**: The script automatically generates **BOTH** single-segment and multi-segment curves, saving them to separate folders.

### Generate grids for all configs

```bash
./generate.sh
```

This will:

1. Find all `*_config.json` files in this directory
2. Generate 7x7 grids for each stage in each config
3. Generate **both** single-segment and multi-segment curves
4. Save outputs to separate folders: `outputs/<config_name>/single_segment/` and `outputs/<config_name>/multi_segment/`

### Customize multi-segment settings

```bash
# Use 3 segments with longer length factor
./generate.sh strong_foundation --num-segments 3 --segment-length-factor 1.2
```

## Output Structure

For each config, the script creates **both** single-segment and multi-segment outputs:

```
outputs/
└── experiment1_config/
    ├── single_segment/
    │   ├── curve_config.json          # Copy of the config used
    │   └── stage1_Stage1_Foundation/
    │       ├── grid_7x7.png           # 7x7 grid of 49 curves
    │       ├── mask_grid_7x7.png      # 7x7 grid of masks
    │       └── stage_info.json         # Stage configuration info
    │   └── stage2_Stage2_Curvature/
    │       └── ...
    └── multi_segment/
        ├── curve_config.json          # Copy of the config used
        └── stage1_Stage1_Foundation/
            ├── grid_7x7.png           # 7x7 grid of 49 multi-segment curves
            ├── mask_grid_7x7.png      # 7x7 grid of masks
            └── stage_info.json         # Stage configuration info
        └── stage2_Stage2_Curvature/
            └── ...
```

## Options

### generate.sh

- `[config_name]`: Config name (without `_config` suffix) - optional, if omitted generates for all configs
- `--num-segments N`: Number of segments for multi-segment curves (default: random from config)
- `--segment-length-factor F`: Factor to control segment length (default: 1.0, >1.0 = longer segments)

### generate_stage_grids.py

- `--config`: Path to curve config JSON file (required)
- `--output`: Output directory (default: `outputs/<config_name>`)
- `--grid-size`: Grid size, e.g., 7 creates 7x7=49 curves (default: 7)
- `--seed`: Base random seed for reproducibility (default: 42)
- `--multi-segment`: Use multi-segment Bezier curve generator (longer curves)
- `--num-segments N`: Number of segments (for multi-segment, default: random)
- `--segment-length-factor F`: Segment length factor (default: 1.0)

### Examples

```bash
# Generate both single and multi-segment for one config
./generate.sh strong_foundation

# Generate with custom multi-segment settings
./generate.sh strong_foundation --num-segments 3 --segment-length-factor 1.2

# Generate 10x10 grid (100 curves) instead of 7x7
python3 ../generate_stage_grids.py --config experiment1_config.json --grid-size 10

# Use different seed
python3 ../generate_stage_grids.py --config experiment1_config.json --seed 123

# Generate only multi-segment curves
python3 ../generate_stage_grids.py --config experiment1_config.json --multi-segment --output outputs/test/multi_segment
```

## Dependencies

The script requires:

- `numpy` - For array operations
- `scipy` - For image processing (gaussian filter)
- `opencv-python` (cv2) - For image I/O and drawing

Install with:

```bash
pip install numpy scipy opencv-python
```

Or from the CurveGeneratorModule directory:

```bash
pip install -r ../requirements.txt
```

## Purpose

This tool helps you:

- **Visualize** what each training stage produces
- **Compare** different configurations side-by-side
- **Debug** curve generation parameters
- **Document** your experiments with visual examples

Each 7x7 grid shows 49 randomly generated curves using the same stage configuration, giving you a good sense of the variety and characteristics of curves produced by that stage.
