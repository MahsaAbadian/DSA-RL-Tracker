# Quick Installation Guide

## Step-by-Step Installation for Cluster

### Step 1: Check Your CUDA Version

```bash
nvidia-smi
```

Look for "CUDA Version" in the output (e.g., 11.8, 12.1, etc.)

### Step 2: Create Conda Environment

**Note:** Conda environments are created in conda's central location (e.g., `~/miniconda3/envs/dsa_rl`), not inside the project directory. The `environment.yml` file is just a configuration file.

**Option A: Using environment.yml (Recommended)**
```bash
# Navigate to Experiment1 directory (where environment.yml is located)
cd Experiment1

# Edit environment.yml to match your CUDA version (change pytorch-cuda=11.8 if needed)
# Then create environment (environment is created in conda's central location):
conda env create -f environment.yml

# Activate environment (can be done from anywhere)
conda activate dsa_rl
```

**Option B: Manual Creation**
```bash
# Create environment (created in conda's central location, not in project)
conda create -n dsa_rl python=3.9
conda activate dsa_rl

# Install PyTorch with CUDA (adjust version based on nvidia-smi output)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1/12.6:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Navigate to project directory and install other dependencies
cd Experiment1  # or wherever your project is
pip install -r requirements.txt
```

### Step 3: Verify GPU Setup

```bash
# Quick compatibility check
./check_cuda_compatibility.sh

# Comprehensive GPU test
python3 check_gpu_setup.py
```

You should see:
- ✅ PyTorch installed
- ✅ CUDA available
- ✅ GPU computation tests passing
- ✅ Training device: CUDA (GPU)

### Step 4: Start Training

```bash
# Navigate to Experiment1 directory
cd Experiment1

# Activate environment (if not already activated)
conda activate dsa_rl

# Start training
./run_train.sh
```

**Note:** The conda environment is global - you can activate it from any directory. Just make sure you're in the `Experiment1` directory when running training scripts.

## Troubleshooting

**If you get "DirectoryNotACondaEnvironmentError":**
```bash
# Remove broken environment
conda env remove -n RL-mahsa  # or whatever name you used

# Create fresh environment
conda env create -f environment.yml
```

**If CUDA is not available:**
1. Check CUDA version: `nvidia-smi`
2. Verify PyTorch CUDA matches: `python3 -c "import torch; print(torch.version.cuda)"`
3. Reinstall with matching version:
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

**If dependencies are missing:**
```bash
pip install -r requirements.txt
```

## Summary

1. ✅ Check CUDA: `nvidia-smi`
2. ✅ Create env: `conda env create -f environment.yml`
3. ✅ Activate: `conda activate dsa_rl`
4. ✅ Verify: `python3 check_gpu_setup.py`
5. ✅ Train: `./run_train.sh`

