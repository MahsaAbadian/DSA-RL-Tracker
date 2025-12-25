# DSA RL Experiment 1

A structured reinforcement learning project for training an agent to follow curves in images using PPO (Proximal Policy Optimization) with curriculum learning.

## Quick Start

### 1. Install Dependencies

**Recommended - Using conda environment file (stable GPU setup):**
```bash
# Check your CUDA version first
nvidia-smi

# Edit environment.yml to match your CUDA version, then:
conda env create -f environment.yml
conda activate dsa_rl

# Verify GPU setup
python3 check_gpu_setup.py
```

**Alternative - Manual conda installation:**
```bash
# Install PyTorch with CUDA (adjust version based on nvidia-smi output)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

**See `INSTALL.md` for detailed stable setup instructions.**

### 2. Setup and Run (All-in-One)

**Automated setup: Generate curves and run training:**
```bash
./setup_and_run.sh [num_curves]
# Example: ./setup_and_run.sh 1000
```

This script:
1. ✅ Checks if dependencies are installed
2. ✅ Generates curves (or uses existing ones if found)
3. ✅ Runs training automatically

**Note:** If curves already exist, it will ask if you want to regenerate them. Default is 1000 curves if not specified.

### 3. Generate Curves (REQUIRED - Must be done before training)

**Step 1: Generate curves first**

The curve generator now creates **stage-specific curves** organized by difficulty:
- **Stage 1 (Simple)**: Wider curves (2-4px), straighter paths, no branches, high contrast
- **Stage 2 (Medium)**: Medium width (2-8px), normal curvature, moderate contrast
- **Stage 3 (Complex)**: Narrow curves (1-10px), high curvature, branches, low contrast

**Option A: Using the bash script (generates for all stages)**
```bash
./run_curve_generator.sh [num_curves] [output_dir]
# Example: ./run_curve_generator.sh 1000 generated_curves
```

**Option B: Direct Python execution**
```bash
# Generate for all stages (recommended)
python3 curve_generator.py --output_dir generated_curves --num_curves 1000 --all_stages

# Or generate for a specific stage
python3 curve_generator.py --output_dir generated_curves --num_curves 1000 --stage 1
python3 curve_generator.py --output_dir generated_curves --num_curves 1000 --stage 2
python3 curve_generator.py --output_dir generated_curves --num_curves 1000 --stage 3
```

**Important:** You must generate curves BEFORE training. The training script loads pre-generated curves from stage-specific directories:
- `generated_curves/stage1/` - Simple curves for Stage 1
- `generated_curves/stage2/` - Medium curves for Stage 2
- `generated_curves/stage3/` - Complex curves for Stage 3

**Recommended:** Generate at least 1000 curves per stage for good training diversity.

### 4. Run Training

**Option A: Using the bash script (recommended for clusters)**
```bash
# Normal run (creates new timestamped folder)
./run_train.sh

# With experiment name (creates runs/experiment_name_TIMESTAMP/)
./run_train.sh --experiment_name baseline_v1

# Clean run (deletes previous runs first)
./run_train.sh --clean

# Resume from checkpoint
./run_train.sh --resume_from runs/20251222_143022/checkpoints/ckpt_Stage1_Bootstrap_ep2000.pth

# Combine options
./run_train.sh --experiment_name test_run --clean
```

**Option B: Direct Python execution**
```bash
# Normal run (auto-creates runs/TIMESTAMP/)
python3 train.py --curves_base_dir generated_curves

# With experiment name (creates runs/EXPERIMENT_NAME_TIMESTAMP/)
python3 train.py --curves_base_dir generated_curves --experiment_name baseline_v1

# Clean run (deletes previous runs first)
python3 train.py --curves_base_dir generated_curves --clean_previous

# Resume from checkpoint
python3 train.py --curves_base_dir generated_curves \
    --resume_from runs/20251222_143022/checkpoints/ckpt_Stage1_Bootstrap_ep2000.pth

# Custom run directory
python3 train.py --run_dir runs/my_experiment --curves_base_dir generated_curves
```

**Results Organization:**
All training results are organized in timestamped folders under `runs/`:
```
runs/
  ├── 20251222_143022/          # Timestamp only: YYYYMMDD_HHMMSS
  └── baseline_v1_20251222_150315/  # With experiment name: EXPERIMENT_NAME_TIMESTAMP
      ├── checkpoints/          # Checkpoints saved every 2000 episodes
      │   ├── ckpt_Stage1_Bootstrap_ep2000.pth
      │   ├── ckpt_Stage1_Bootstrap_ep4000.pth
      │   └── ...
      ├── weights/               # Final weights after each stage
      │   ├── model_Stage1_Bootstrap_FINAL.pth
      │   ├── model_Stage2_Robustness_FINAL.pth
      │   └── model_Stage3_Realism_FINAL.pth
      ├── logs/                 # Training log
      │   └── training.log
      ├── config.json           # Training configuration
      └── metrics.json          # Training metrics (rewards, success rates)
```

**Resume Training:**
You can resume training from any checkpoint:
```bash
# Resume from a specific checkpoint
./run_train.sh --resume_from runs/20251222_143022/checkpoints/ckpt_Stage1_Bootstrap_ep2000.pth

# Training will:
# - Load the model weights from the checkpoint
# - Continue from the next episode in the same stage
# - Skip any stages that were already completed
# - Append to existing logs and metrics
```

The training will:
- Load stage-specific curves from `generated_curves/stage1/`, `stage2/`, and `stage3/` directories
- Run 3 curriculum stages:
  - **Stage 1 (Bootstrap)**: 8k episodes, simple curves, no noise, auto-complete
  - **Stage 2 (Robustness)**: 12k episodes, medium curves, noise enabled, strict stop
  - **Stage 3 (Realism)**: 15k episodes, complex curves, tissue artifacts, full difficulty
- Save checkpoints every 2000 episodes to `runs/TIMESTAMP/checkpoints/`
- Save final weights after each stage to `runs/TIMESTAMP/weights/`
- Log all output to `runs/TIMESTAMP/logs/training.log`

**Note:** Training requires curves for all 3 stages. If any stage directory is missing, training will fail with a clear error message.

### 5. Workflow Summary

**Option A: Automated (Recommended for first-time setup)**
```bash
# 1. Install dependencies (recommended: conda)
conda install pytorch numpy scipy opencv matplotlib -c pytorch -c conda-forge

# 2. Run everything (generates curves + trains)
./setup_and_run.sh 1000

# 3. Monitor progress (find latest run)
LATEST_RUN=$(ls -td runs/*/ | head -1)
tail -f $LATEST_RUN/logs/training.log
```

**Option B: Manual (More control)**
```bash
# 1. Install dependencies (recommended: conda)
conda install pytorch numpy scipy opencv matplotlib -c pytorch -c conda-forge

# 2. Generate curves for all stages separately
./run_curve_generator.sh 1000 generated_curves
# This creates: generated_curves/stage1/, stage2/, stage3/

# 3. Run training separately
./run_train.sh                    # Normal run
# OR
./run_train.sh --clean            # Delete previous runs first

# 4. Monitor progress (find latest run)
LATEST_RUN=$(ls -td runs/*/ | head -1)
tail -f $LATEST_RUN/logs/training.log
```

**Option C: Clean Start (Delete all previous results)**
```bash
# Delete all previous runs and start fresh
./run_train.sh --clean

# Or with setup script
./setup_and_run.sh 1000 --clean
```

**Option D: Named Experiments**
```bash
# Create a named experiment (easier to identify later)
./run_train.sh --experiment_name ablation_study_1

# Results will be in: runs/ablation_study_1_20251222_143022/
```

**Option E: Resume Training**
```bash
# Resume from a checkpoint (useful if training was interrupted)
./run_train.sh --resume_from runs/20251222_143022/checkpoints/ckpt_Stage2_Robustness_ep5000.pth

# Training will continue from episode 5001 in Stage 2
```

### 6. Monitor Training

**Watch the log file (latest run):**
```bash
LATEST_RUN=$(ls -td runs/*/ | head -1)
tail -f $LATEST_RUN/logs/training.log
```

**Check checkpoints (latest run):**
```bash
LATEST_RUN=$(ls -td runs/*/ | head -1)
ls -lh $LATEST_RUN/checkpoints/
```

**Check final weights (latest run):**
```bash
LATEST_RUN=$(ls -td runs/*/ | head -1)
ls -lh $LATEST_RUN/weights/
```

**List all training runs:**
```bash
ls -lht runs/
```

## Project Structure

```
Experiment1/
├── curve_generator.py         # Generate synthetic curves for all stages
├── train.py                   # Training script with curriculum learning
├── test.py                    # Testing/inference script (to be created)
├── run_curve_generator.sh     # Bash script to generate curves separately
├── run_train.sh               # Bash script to run training
├── setup_and_run.sh           # Check dependencies and run training
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── generated_curves/          # Pre-generated curves (organized by stage)
│   ├── stage1/               # Simple curves (wide, straight)
│   ├── stage2/               # Medium curves (moderate complexity)
│   └── stage3/               # Complex curves (narrow, curved, branches)
└── runs/                      # All training runs (timestamped or named)
    ├── 20251222_143022/      # Timestamp-only run
    ├── baseline_v1_20251222_150315/  # Named experiment
    │   ├── checkpoints/      # Checkpoints saved during training
    │   ├── weights/          # Final model weights per stage
    │   ├── logs/             # Training log file
    │   ├── config.json       # Training configuration
    │   └── metrics.json     # Training metrics
    └── ...
```

## Training Stages

1. **Stage 1: Bootstrap** (8,000 episodes)
   - Clean images, narrow curves (width 2-4)
   - Auto-completes when reaching end
   - Learning rate: 1e-4

2. **Stage 2: Robustness** (12,000 episodes)
   - Adds noise (50% probability)
   - Wider curves (width 2-8)
   - Strict stop required
   - Learning rate: 5e-5

3. **Stage 3: Realism** (15,000 episodes)
   - Maximum difficulty: tissue artifacts, high noise (80%), very wide curves (width 1-10)
   - Learning rate: 1e-5

## Running on a Cluster

The `run_train.sh` script is designed for cluster environments:

1. **Submit job** (example for SLURM):
   ```bash
   sbatch --job-name=dsa_train --time=48:00:00 --mem=8G --gres=gpu:1 run_train.sh
   ```

2. **Monitor progress**:
   ```bash
   tail -f logs/training_*.log
   ```

3. **Check checkpoints**:
   ```bash
   ls -lh checkpoints/
   ```

## Notes

- **Curve generation and training are completely separate processes**
- You MUST generate curves before training (use `./run_curve_generator.sh`)
- Training loads curves randomly from the `generated_curves/` directory
- Training uses GPU if available, otherwise falls back to CPU
- Checkpoints are saved every 2000 episodes to allow resuming
- All paths are relative to the script directory for portability
- Total training time: ~35,000 episodes (can take several hours to days depending on hardware)
- Recommended: Generate 1000+ curves for good training diversity
