# DSA RL Tracker

DSA Curve Tracking with Reinforcement Learning

A reinforcement learning project for training an agent to follow curves in images using PPO (Proximal Policy Optimization) with curriculum learning.

## Project Structure

```
DSA-RL-Tracker/
├── CurveGeneratorModule/    # Central curve generation module (shared)
│   ├── generator.py         # Base curve generator (4-point Bezier)
│   ├── generator_multisegment.py  # Multi-segment curve generator
│   ├── generator_sixpoint.py # Six-point curve generator (degree-5 Bezier)
│   ├── config_loader.py     # Configuration loading utilities
│   ├── ExampleConfigs/      # Example curve configurations
│   └── generate.sh          # Script to generate curve grids
├── StopModule/              # Central stop detection module (shared)
│   ├── src/                 # Stop detection utilities
│   └── README.md            # Stop module documentation
├── FineTune/                # Fine-tuning utilities (shared)
│   ├── src/                 # Fine-tuning scripts
│   └── README.md            # Fine-tuning documentation
├── Experiment1/             # Experiment 1 directory
│   ├── src/                 # Source code
│   ├── docs/                # Experiment documentation
│   └── run_train.sh         # Training script wrapper
├── Experiment4_separate_stop_v2/  # Experiment 4 (uses central modules)
│   ├── src/                 # Source code
│   ├── config/              # Experiment-specific config
│   └── run_train.sh         # Training script wrapper
├── scripts/                 # Shared, experiment-agnostic scripts
│   ├── setup_conda_env.sh   # Conda environment setup
│   ├── check_gpu_setup.py   # GPU verification
│   └── ...
├── environment.yml          # Conda environment configuration
└── README.md                # This file
```

### Shared Modules

The project now includes several **centralized, reusable modules**:

1. **CurveGeneratorModule** (`CurveGeneratorModule/`)

   - Centralized curve generation for all experiments
   - Supports multiple curve types: single-segment (4-point), multi-segment (2+ segments), and six-point (degree-5 Bezier)
   - Currently used by: **Experiment 4**
   - Provides configurable curve generation with JSON-based configuration

2. **StopModule** (`StopModule/`)

   - Centralized stop detection utilities
   - Can be used across experiments for consistent stop detection logic

3. **FineTune** (`FineTune/`)
   - Fine-tuning utilities and scripts
   - Can be used to fine-tune models from any experiment

### Module Usage

- **CurveGeneratorModule**: Currently integrated into Experiment 4, which uses the six-point generator with the `strong_foundation` configuration
- Other experiments (1, 2, 3, 5) maintain their own curve generation implementations for now

## Quick Start

### Option 1: Using the Setup Script (Recommended)

```bash
# From repository root
./scripts/setup_conda_env.sh

# Activate environment
conda activate dsa_rl

# Navigate to experiment and run training
cd Experiment1
./run_train.sh
```

### Option 2: Manual Setup

```bash
# Create conda environment from root
conda env create -f environment.yml
conda activate dsa_rl

# Navigate to experiment
cd Experiment1

# Verify GPU setup
python3 ../scripts/check_gpu_setup.py

# Run training
./run_train.sh
```

### Option 3: Google Colab

See [`COLAB_SETUP.md`](COLAB_SETUP.md) or open [`colab_training.ipynb`](colab_training.ipynb) in Google Colab.

## Key Features

- **On-the-fly curve generation**: Curves are generated during training, no pre-generation needed
- **Curriculum learning**: Three-stage training progression (Bootstrap → Robustness → Realism)
- **GPU support**: Automatic GPU detection with CPU fallback
- **Checkpoint system**: Resume training from any checkpoint
- **Experiment tracking**: Timestamped runs with metrics and logs

## Documentation

- **Quick Start**: [`Experiment1/docs/QUICK_INSTALL.md`](Experiment1/docs/QUICK_INSTALL.md)
- **Detailed Installation**: [`Experiment1/docs/INSTALL.md`](Experiment1/docs/INSTALL.md)
- **Colab Setup**: [`COLAB_SETUP.md`](COLAB_SETUP.md)
- **Experiment README**: [`Experiment1/README.md`](Experiment1/README.md)
- **Shared Scripts**: [`scripts/README.md`](scripts/README.md)

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional, CPU fallback available)
- Conda (recommended) or pip

See [`environment.yml`](environment.yml) for full dependency list.

## Training

Training runs through 3 curriculum stages:

1. **Stage 1: Bootstrap** (8k episodes) - Simple curves, clean images
2. **Stage 2: Robustness** (12k episodes) - Added noise, wider curves
3. **Stage 3: Realism** (15k episodes) - Maximum difficulty with tissue artifacts

Results are saved in `Experiment1/runs/` with checkpoints, weights, logs, and metrics.

## License

[Add your license here]
