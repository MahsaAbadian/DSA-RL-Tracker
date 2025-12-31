# Shared Scripts

This directory contains experiment-agnostic scripts for GPU setup, CUDA compatibility checking, and environment setup.

## Scripts

- **`check_cuda_compatibility.sh`** - Check CUDA version and PyTorch compatibility
- **`check_gpu_setup.py`** - Comprehensive GPU setup verification
- **`diagnose_cuda.py`** - Quick PyTorch CUDA diagnostic
- **`install_for_cuda12.sh`** - Installation script for CUDA 12.x systems

## Usage

These scripts can be used from anywhere in the repository:

```bash
# From repository root
python3 scripts/check_gpu_setup.py

# From Experiment1 directory
python3 ../scripts/check_gpu_setup.py
```

## Why These Are Here

These scripts are **experiment-agnostic** - they work for any PyTorch/CUDA project, not just Experiment1. They were moved here to:
- Avoid duplication if you create more experiments
- Keep experiment-specific code separate from shared utilities
- Make it easier to reuse across projects

