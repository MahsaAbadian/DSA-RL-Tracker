# DSA RL Tracker

DSA Curve Tracking with Reinforcement Learning

A reinforcement learning project for training an agent to follow curves in images using PPO (Proximal Policy Optimization) with curriculum learning.

## Project Structure

```
DSA-RL-Tracker/
├── Experiment1/              # Main experiment directory
│   ├── src/                  # Source code
│   │   ├── train.py         # Training script
│   │   ├── models.py        # Neural network architectures
│   │   ├── inference.py     # Inference/testing script
│   │   └── curve_generator.py  # Curve generation utilities
│   ├── docs/                 # Experiment documentation
│   ├── run_train.sh          # Training script wrapper
│   └── requirements.txt      # Python dependencies
├── scripts/                  # Shared, experiment-agnostic scripts
│   ├── setup_conda_env.sh   # Conda environment setup
│   ├── check_gpu_setup.py   # GPU verification
│   ├── check_cuda_compatibility.sh  # CUDA compatibility check
│   └── install_for_cuda12.sh  # CUDA 12.x installation helper
├── environment.yml          # Conda environment configuration (root level)
├── colab_training.ipynb     # Google Colab training notebook
├── COLAB_SETUP.md           # Colab setup guide
└── README.md                # This file
```

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
