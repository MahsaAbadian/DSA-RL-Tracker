# Installation Guide

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

### Step 3: Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Why PyTorch is Separate?

PyTorch is commented out in `requirements.txt` because:
1. Different clusters have different CUDA versions
2. Conda provides better CUDA integration than pip
3. You may want specific PyTorch builds optimized for your hardware
4. Allows flexibility in choosing PyTorch version and CUDA configuration

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

