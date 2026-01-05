# Installation Guide

## Stable PyTorch + CUDA Setup for GPU

This guide helps you set up a **stable** PyTorch + CUDA configuration for reliable GPU usage.

## Quick Start (Recommended)

**Option 1: Using automated setup script (easiest)**
```bash
# This script handles existing environments and guides you through setup
# Run from Experiment1 directory:
../scripts/setup_conda_env.sh
```

**Option 2: Using conda environment file manually**
```bash
# Check your CUDA version first
nvidia-smi

# Edit environment.yml to match your CUDA version (pytorch-cuda=11.8 or 12.1)
# environment.yml is in the repository root
# Then create environment:
conda env create -f ../environment.yml
conda activate dsa_rl

# Verify setup:
python3 ../scripts/check_gpu_setup.py
```

**If you get "DirectoryNotACondaEnvironmentError":**
```bash
# Remove the problematic directory first
conda env remove -n RL-mahsa

# Then create new environment
conda env create -f ../environment.yml
# or use the setup script:
../scripts/setup_conda_env.sh
```

**Option 2: Manual installation (more control)**
See detailed steps below.

## PyTorch Installation (Conda - Recommended)

PyTorch should be installed separately via conda with your preferred CUDA configuration. This allows you to choose the exact CUDA version and PyTorch build that matches your cluster.

### Step 1: Install PyTorch via Conda

**Check your CUDA version first:**
```bash
nvidia-smi
```

**Then install PyTorch with matching CUDA:**

For CUDA 11.8:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

For CUDA 12.1:
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

For CPU only:
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### Step 2: Install Other Dependencies

After PyTorch is installed, install the remaining dependencies:

**Option A - Using pip:**
```bash
pip install -r requirements.txt
```

**Option B - Using conda:**
```bash
conda install numpy scipy opencv matplotlib -c conda-forge
```

### Step 3: Verify Installation & CUDA Compatibility

**Quick CUDA version check:**
```bash
# From repository root
../scripts/check_cuda_compatibility.sh
```

This script will:
- ✅ Check system CUDA version (from nvidia-smi)
- ✅ Check PyTorch CUDA version
- ✅ Verify compatibility between them
- ✅ Show GPU allocation status

**Comprehensive GPU test (recommended):**
```bash
python3 ../scripts/check_gpu_setup.py
```

This script will:
- ✅ Check PyTorch installation
- ✅ Verify CUDA availability
- ✅ Check CUDA version compatibility
- ✅ Show detailed GPU information
- ✅ Test GPU computation (matrix multiplication, memory, performance)
- ✅ Confirm training script will use GPU
- ✅ Provide troubleshooting guidance

## Stable Configuration Tips

### 1. Match CUDA Versions
**Critical:** Your PyTorch CUDA version should match (or be compatible with) your system's CUDA driver version.

```bash
# Check system CUDA version
nvidia-smi

# Install matching PyTorch CUDA version
# If nvidia-smi shows CUDA 11.8, use pytorch-cuda=11.8
# If nvidia-smi shows CUDA 12.1, use pytorch-cuda=12.1
```

### 2. Use Specific PyTorch Version (Recommended for Stability)
Instead of installing the latest PyTorch, pin a stable version:

```bash
# Example: Install PyTorch 2.1.0 with CUDA 11.8
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### 3. Create a Dedicated Conda Environment
```bash
# Create environment (created in conda's central location: ~/miniconda3/envs/dsa_rl)
conda create -n dsa_rl python=3.9

# Activate environment (can be done from any directory)
conda activate dsa_rl

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Navigate to project directory and install other dependencies
cd Experiment1  # or path to your project
pip install -r requirements.txt
```

**Note:** Conda environments are stored centrally (e.g., `~/miniconda3/envs/dsa_rl`), not inside your project directory. The `environment.yml` file is in the repository root and is just a configuration file.

### 4. Verify Before Training
Always run the verification script before training:
```bash
python3 ../scripts/check_gpu_setup.py
```

## Why PyTorch is Separate?

PyTorch is commented out in `requirements.txt` because:
1. Different clusters have different CUDA versions
2. Conda provides better CUDA integration than pip
3. You may want specific PyTorch builds optimized for your hardware
4. Allows flexibility in choosing PyTorch version and CUDA configuration
5. **Stability**: You can pin specific versions that work reliably on your cluster

## Alternative: Install Everything via Conda

If you prefer to install everything via conda:

```bash
# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
conda install numpy scipy opencv matplotlib -c conda-forge
```

## Troubleshooting

**If you get "ModuleNotFoundError: No module named 'torch'":**
- Make sure you've installed PyTorch via conda first
- Verify you're in the correct conda environment: `conda activate your_env_name`
- Check installation: `python3 -c "import torch; print(torch.__version__)"`

**If CUDA is not available:**
- Check CUDA version: `nvidia-smi`
- Reinstall PyTorch with matching CUDA version
- Verify: `python3 -c "import torch; print(torch.cuda.is_available())"`

