# Configuration Files

This directory contains configuration files for the DSA RL training pipeline.

## Files

- **`curve_config.json`** - Main configuration file for:
  - Curve generation parameters (bezier curves, branches, noise)
  - Training stages (curriculum learning settings)
  - Image dimensions
  - Stage-specific curve properties

- **`example_config_variations.json`** - Example variations of the config file for different experiment setups

## Usage

The training script automatically looks for `config/curve_config.json` first, then falls back to `Experiment1/curve_config.json` for backward compatibility.

You can specify a custom config path:
```bash
python src/train.py --curve_config config/my_custom_config.json
```

## Structure

Configuration files are kept separate from source code to:
- Keep code and configs organized
- Make it easier to version control configs separately
- Allow multiple config variations without cluttering the source directory
- Follow Python project best practices

