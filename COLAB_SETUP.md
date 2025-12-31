# Google Colab Setup Guide

## Step-by-Step Code for Running Training in Google Colab

Copy and paste these code blocks into separate cells in a Google Colab notebook.

---

### Step 0: Check CUDA Version (Optional)

```python
# Check CUDA version before installing PyTorch
# This will show CUDA version in the nvidia-smi output
!nvidia-smi | grep -i "cuda version" || echo "nvidia-smi not available - will check after PyTorch installation"
```

## Step 1: Install Dependencies

```python
# Install required packages
# Colab uses CUDA 12.4, so we use cu121 (compatible with CUDA 12.x)
# If you get CUDA errors, try: cu118 for CUDA 11.8 or cpu for CPU-only
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install numpy>=1.21.0 scipy>=1.7.0 opencv-python>=4.5.0 matplotlib>=3.4.0
```

---

### Step 2: Clone Repository and Navigate

```python
# Clone your repository (replace with your actual repo URL)
!git clone https://github.com/YOUR_USERNAME/DSA-RL-Tracker.git
# Or if you've already cloned it, skip this step

# Navigate to Experiment1 directory
import os
os.chdir('/content/DSA-RL-Tracker/Experiment1')
print(f"Current directory: {os.getcwd()}")
```

---

### Step 3: Verify GPU Setup

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("\n✅ GPU is ready for training!")
else:
    print("\n❌ ERROR: GPU is not available!")
    print("\n⚠️  Training requires GPU acceleration.")
    print("   Please enable GPU:")
    print("   1. Go to: Runtime → Change runtime type")
    print("   2. Set Hardware accelerator: GPU")
    print("   3. Click Save")
    print("   4. Re-run this cell")
    print("\n   Training will be extremely slow on CPU and may timeout.")
    raise RuntimeError("GPU not available. Please enable GPU runtime before continuing.")
```

---

### Step 4: Verify Installation

```python
# Check if all required files exist
import os
required_files = [
    'src/train.py',
    'src/models.py',
    'config/curve_config.json'  # Config is now in config/ directory
]

for file in required_files:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file} - MISSING!")
```

---

### Step 5: Run Training

```python
# Run training with default settings
# This will create a timestamped run directory in runs/
import sys
sys.path.insert(0, '/content/DSA-RL-Tracker/Experiment1')

from src.train import run_unified_training

# Start training
run_unified_training(
    run_dir=None,  # Auto-create timestamped directory
    base_seed=42,
    clean_previous=False,
    experiment_name="colab_training",
    resume_from=None,
    curve_config_path="config/curve_config.json"  # Config is now in config/ directory
)
```

---

### Alternative: Run Training with Command Line Arguments

```python
# Alternative method using command line interface
import sys
import subprocess

# Change to Experiment1 directory
os.chdir('/content/DSA-RL-Tracker/Experiment1')

# Run training script
subprocess.run([
    sys.executable, 
    'src/train.py',
    '--experiment_name', 'colab_training',
    '--curve_config', 'curve_config.json'
])
```

---

### Step 6: Monitor Training Progress

```python
# Monitor the latest training run
import glob
import os

# Find the latest run directory
run_dirs = glob.glob('runs/*/')
if run_dirs:
    latest_run = max(run_dirs, key=os.path.getctime)
    log_file = os.path.join(latest_run, 'logs', 'training.log')
    
    if os.path.exists(log_file):
        print(f"📊 Latest run: {latest_run}")
        print("\n=== Last 50 lines of training log ===")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line.rstrip())
    else:
        print(f"Log file not found: {log_file}")
else:
    print("No runs found yet")
```

---

### Step 7: Download Results (Optional)

```python
# Download checkpoints and results
from google.colab import files
import zipfile
import os

# Find latest run
run_dirs = glob.glob('runs/*/')
if run_dirs:
    latest_run = max(run_dirs, key=os.path.getctime)
    
    # Create zip file
    zip_path = f'{latest_run.rstrip("/")}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, filenames in os.walk(latest_run):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                arcname = os.path.relpath(file_path, os.path.dirname(latest_run))
                zipf.write(file_path, arcname)
    
    # Download
    files.download(zip_path)
    print(f"✅ Downloaded: {zip_path}")
```

---

## Quick Start (All-in-One Cell)

If you want to run everything in one go:

```python
# ===== COMPLETE SETUP AND TRAINING =====

# 1. Install dependencies
print("📦 Installing dependencies...")
# Using cu121 which is compatible with CUDA 12.4
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install numpy>=1.21.0 scipy>=1.7.0 opencv-python>=4.5.0 matplotlib>=3.4.0

# 2. Clone repository (if not already cloned)
import os
if not os.path.exists('/content/DSA-RL-Tracker'):
    !git clone https://github.com/YOUR_USERNAME/DSA-RL-Tracker.git

# 3. Navigate to Experiment1
os.chdir('/content/DSA-RL-Tracker/Experiment1')
print(f"📁 Current directory: {os.getcwd()}")

# 4. Verify GPU
import torch
print(f"\n🖥️  GPU Check:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# 5. Run training
print("\n🚀 Starting training...")
import sys
sys.path.insert(0, '/content/DSA-RL-Tracker/Experiment1')

from src.train import run_unified_training

run_unified_training(
    run_dir=None,
    base_seed=42,
    clean_previous=False,
    experiment_name="colab_training",
    resume_from=None,
    curve_config_path="curve_config.json"
)

print("\n✅ Training complete!")
```

---

## Notes for Colab:

1. **GPU Runtime REQUIRED**: 
   - **You MUST enable GPU before running** - training will fail without it
   - Go to: Runtime → Change runtime type → Hardware accelerator → GPU → Save
   - Colab typically provides CUDA 12.4, which is compatible with PyTorch cu121
   - The notebook will verify GPU availability and stop if GPU is not detected
   
2. **Why pip instead of conda?**
   - Colab doesn't have conda pre-installed
   - pip is simpler and works well for Colab's environment
   - PyTorch wheels from pip work perfectly with Colab's GPU setup

2. **Session Timeout**: Colab sessions timeout after ~12 hours. For long training:
   - Use checkpoints to resume training
   - Save results periodically
   - Consider using Colab Pro for longer sessions

3. **Resume Training**: If training is interrupted, you can resume:
   ```python
   run_unified_training(
       resume_from='runs/colab_training_TIMESTAMP/checkpoints/ckpt_Stage1_Bootstrap_ep2000.pth'
   )
   ```

4. **Monitor Progress**: Keep the Colab tab open and check the output periodically.

5. **Save Results**: Download checkpoints regularly to avoid losing progress.

---

## Troubleshooting:

**Issue: ModuleNotFoundError**
- Solution: Make sure you've navigated to the correct directory and added it to sys.path

**Issue: CUDA out of memory**
- Solution: Reduce batch size or use CPU (slower but works)

**Issue: File not found**
- Solution: Verify you're in the Experiment1 directory: `os.getcwd()`

