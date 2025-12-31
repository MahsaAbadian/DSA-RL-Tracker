# Parallel Experiments Guide

Run multiple training experiments in parallel with different curve generation configurations.

## Quick Start

### Basic Usage

```bash
# Run with default variations (3 example configs)
python3 run_parallel_experiments.py --experiment_name my_parallel_exp

# Or use the bash script
./run_parallel_experiments.sh --experiment_name my_parallel_exp
```

### With Custom Configurations

```bash
# Use example config file
python3 run_parallel_experiments.py \
    --experiment_name my_exp \
    --config_file example_config_variations.json \
    --max_workers 2

# Or with bash script
./run_parallel_experiments.sh \
    --experiment_name my_exp \
    --config_file example_config_variations.json \
    --max_workers 2
```

## Configuration File Format

Create a JSON file with a list of configuration variations. Each variation specifies what to change from the base config:

```json
[
  {
    "training_stages.0.curve_generation.width_range": [3, 6],
    "training_stages.0.curve_generation.curvature_factor": 0.5,
    "training_stages.0.curve_generation.min_intensity": 0.7
  },
  {
    "training_stages.0.curve_generation.width_range": [1, 2],
    "training_stages.0.curve_generation.curvature_factor": 1.0,
    "training_stages.0.curve_generation.min_intensity": 0.5
  }
]
```

### Nested Key Syntax

Use dot notation to specify nested keys:
- `training_stages.0.curve_generation.width_range` - Stage 1 curve width
- `training_stages.1.curve_generation.curvature_factor` - Stage 2 curvature
- `training_stages.0.training.noise` - Stage 1 training noise

Array indices use numbers: `training_stages.0` = first stage

## Examples

### Example 1: Test Different Width Ranges

Create `width_test.json`:
```json
[
  {"training_stages.0.curve_generation.width_range": [1, 2]},
  {"training_stages.0.curve_generation.width_range": [2, 4]},
  {"training_stages.0.curve_generation.width_range": [3, 6]},
  {"training_stages.0.curve_generation.width_range": [4, 8]}
]
```

Run:
```bash
python3 run_parallel_experiments.py \
    --experiment_name width_test \
    --config_file width_test.json
```

### Example 2: Test Curvature Factors

Create `curvature_test.json`:
```json
[
  {"training_stages.0.curve_generation.curvature_factor": 0.5},
  {"training_stages.0.curve_generation.curvature_factor": 1.0},
  {"training_stages.0.curve_generation.curvature_factor": 1.5},
  {"training_stages.0.curve_generation.curvature_factor": 2.0}
]
```

Run:
```bash
python3 run_parallel_experiments.py \
    --experiment_name curvature_test \
    --config_file curvature_test.json \
    --max_workers 4
```

### Example 3: Test Multiple Parameters

Create `multi_param_test.json`:
```json
[
  {
    "training_stages.0.curve_generation.width_range": [2, 4],
    "training_stages.0.curve_generation.curvature_factor": 0.5,
    "training_stages.0.curve_generation.min_intensity": 0.6,
    "training_stages.0.training.noise": 0.0
  },
  {
    "training_stages.0.curve_generation.width_range": [1, 3],
    "training_stages.0.curve_generation.curvature_factor": 1.0,
    "training_stages.0.curve_generation.min_intensity": 0.4,
    "training_stages.0.training.noise": 0.5
  },
  {
    "training_stages.0.curve_generation.width_range": [1, 5],
    "training_stages.0.curve_generation.curvature_factor": 1.5,
    "training_stages.0.curve_generation.min_intensity": 0.3,
    "training_stages.0.training.noise": 0.8
  }
]
```

## Command Line Options

```bash
python3 run_parallel_experiments.py [OPTIONS]

Options:
  --experiment_name NAME    Base name for experiments (default: parallel_exp)
  --base_seed SEED          Base seed (default: 42)
  --base_config PATH         Path to base curve_config.json
  --config_file PATH         Path to JSON file with config variations
  --max_workers NUM          Max parallel workers (default: min(CPUs, 4))
  --variations JSON_STRING   JSON string with variations (alternative to --config_file)
```

## How It Works

1. **Loads base config**: Uses `curve_config.json` (or specified file)
2. **Applies variations**: Each variation modifies specific parameters
3. **Runs in parallel**: Uses multiprocessing to run multiple trainings simultaneously
4. **Saves results**: Each run goes to `runs/EXPERIMENT_NAME_configN_TIMESTAMP/`
5. **Saves config**: Each run directory contains `curve_config.json` with the exact config used
6. **Generates summary**: Creates `runs/EXPERIMENT_NAME_summary.json` with results

## Output Structure

```
runs/
├── parallel_exp_config0_20241222_120000/
│   ├── curve_config.json          # Config used for this run
│   ├── checkpoints/
│   ├── weights/
│   └── logs/
├── parallel_exp_config1_20241222_120100/
│   ├── curve_config.json
│   └── ...
└── parallel_exp_summary.json      # Summary of all runs
```

## Summary File

The summary file contains:
```json
{
  "experiment_name": "parallel_exp",
  "base_seed": 42,
  "total_experiments": 5,
  "successful": 4,
  "failed": 1,
  "total_time_minutes": 120.5,
  "results": [
    {
      "config_id": 0,
      "success": true,
      "run_dir": "runs/parallel_exp_config0_...",
      "error": null
    },
    ...
  ]
}
```

## Tips

1. **Start small**: Test with 2-3 configs first
2. **Limit workers**: Use `--max_workers 2` if you have limited GPU memory
3. **Check GPU**: Each worker uses GPU, so don't exceed GPU count
4. **Monitor progress**: Check `runs/` directory for individual run logs
5. **Compare results**: Use the summary file to identify best performing configs

## Troubleshooting

### Out of Memory
- Reduce `--max_workers` (e.g., `--max_workers 1`)
- Run fewer experiments at once

### Experiments Failing
- Check individual run logs in `runs/EXPERIMENT_NAME_configN/logs/`
- Verify config variations are valid JSON
- Check base config file exists

### Slow Performance
- Reduce number of parallel workers
- Check GPU utilization: `nvidia-smi`
- Consider running sequentially if GPU memory is limited

## Advanced Usage

### Inline Variations (without file)

```python
import json
variations = [
    {"training_stages.0.curve_generation.width_range": [2, 4]},
    {"training_stages.0.curve_generation.width_range": [1, 3]}
]
variations_json = json.dumps(variations)

# Then use --variations flag
!python3 run_parallel_experiments.py --variations '{variations_json}'
```

### Custom Base Config

```bash
python3 run_parallel_experiments.py \
    --base_config my_custom_base.json \
    --config_file my_variations.json
```

