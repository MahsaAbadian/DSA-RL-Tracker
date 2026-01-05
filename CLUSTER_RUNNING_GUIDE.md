# Running Training on a Cluster

This guide explains how to run training on a cluster so it continues even if you lose connection.

## Method 1: Using `tmux` (Recommended)

**tmux** allows you to create persistent terminal sessions that survive disconnections.

### Basic Usage:

```bash
# 1. SSH into the cluster
ssh username@cluster.com

# 2. Navigate to your project directory
cd /path/to/DSA-RL-Tracker/Experiment3_Refine

# 3. Start a new tmux session
tmux new -s training

# 4. Run your training script
cd src
python train.py --experiment_name my_training_run

# 5. Detach from tmux (training continues): Press Ctrl+B, then D
# Or simply close your SSH connection - tmux session stays alive

# 6. Reconnect later:
ssh username@cluster.com
tmux attach -t training  # Reattach to see progress
# Or: tmux ls  # List all sessions, then tmux attach -t <session-name>
```

### Useful tmux Commands:

```bash
# Create new session
tmux new -s session_name

# List all sessions
tmux ls

# Attach to session
tmux attach -t session_name

# Kill a session
tmux kill-session -t session_name

# Inside tmux:
#   Ctrl+B then D  - Detach (session keeps running)
#   Ctrl+B then %  - Split vertically
#   Ctrl+B then "  - Split horizontally
#   Ctrl+B then [  - Scroll mode
```

---

## Method 2: Using `screen` (Alternative)

Similar to tmux, but simpler interface.

### Basic Usage:

```bash
# 1. SSH into cluster
ssh username@cluster.com

# 2. Start a new screen session
screen -S training

# 3. Run training
cd /path/to/DSA-RL-Tracker/Experiment3_Refine/src
python train.py --experiment_name my_training_run

# 4. Detach: Press Ctrl+A, then D
# Training continues even after disconnection

# 5. Reconnect later:
ssh username@cluster.com
screen -r training  # Reattach
# Or: screen -ls  # List sessions
```

### Useful screen Commands:

```bash
# Create new session
screen -S session_name

# List sessions
screen -ls

# Reattach
screen -r session_name

# Inside screen:
#   Ctrl+A then D  - Detach
#   Ctrl+A then [  - Scroll mode
```

---

## Method 3: Using `nohup` (Simple Background)

Runs the process in the background and continues after disconnection.

### Basic Usage:

```bash
# 1. SSH into cluster
ssh username@cluster.com

# 2. Navigate to project
cd /path/to/DSA-RL-Tracker/Experiment3_Refine/src

# 3. Run with nohup and redirect output
nohup python train.py --experiment_name my_training_run > ../training.log 2>&1 &

# 4. Check if running
ps aux | grep train.py

# 5. Monitor output
tail -f ../training.log

# 6. Kill if needed
# Find process ID: ps aux | grep train.py
kill <PID>
```

### Enhanced nohup Script:

```bash
#!/bin/bash
# run_training_background.sh

cd /path/to/DSA-RL-Tracker/Experiment3_Refine/src

# Redirect stdout and stderr to log file
nohup python train.py \
    --experiment_name my_training_run \
    > ../runs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "Training started with PID: $!"
echo "Monitor with: tail -f ../runs/training_*.log"
```

---

## Method 4: Using SLURM (Cluster Job Scheduler)

If your cluster uses SLURM, submit a job that runs in the background.

### Basic SLURM Script:

```bash
#!/bin/bash
# train.slurm
#SBATCH --job-name=dsa_rl_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00          # Time limit (48 hours)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=16GB               # Memory request

# Load necessary modules (adjust for your cluster)
module load python/3.9
module load cuda/11.8

# Activate conda environment if needed
# conda activate your_env

# Navigate to project directory
cd $SLURM_SUBMIT_DIR/Experiment3_Refine/src

# Run training
python train.py --experiment_name my_training_run
```

### Submit Job:

```bash
# Submit job
sbatch train.slurm

# Check job status
squeue -u $USER

# Cancel job
scancel <job_id>

# Monitor output
tail -f logs/train_<job_id>.out
```

### Advanced SLURM Script with GPU Monitoring:

```bash
#!/bin/bash
# train.slurm
#SBATCH --job-name=dsa_rl_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --partition=gpu          # Adjust partition name

# Setup
module load python/3.9 cuda/11.8
cd $SLURM_SUBMIT_DIR/Experiment3_Refine/src

# Activate environment
source activate your_env  # or conda activate your_env

# Start GPU monitoring in background
nvidia-smi dmon -s u -d 5 > ../runs/gpu_monitor_%j.log &

# Run training
python train.py --experiment_name my_training_run

# Training will continue even if you disconnect
```

---

## Method 5: Using `disown` (Alternative)

```bash
# Start training in background
python train.py --experiment_name my_training_run > training.log 2>&1 &

# Get the job ID
JOB_ID=$!

# Disown the job (makes it independent of terminal)
disown $JOB_ID

# Now you can safely disconnect
```

---

## Recommended Workflow

**For most clusters, use tmux + nohup combination:**

```bash
# 1. SSH and start tmux
ssh username@cluster.com
tmux new -s training

# 2. Run with nohup (extra safety)
cd /path/to/DSA-RL-Tracker/Experiment3_Refine/src
nohup python train.py --experiment_name my_run > ../training.log 2>&1 &

# 3. Detach tmux: Ctrl+B, then D

# 4. Check status anytime:
ssh username@cluster.com
tmux attach -t training
tail -f ../training.log
```

---

## Monitoring and Checking Status

### Check if training is running:

```bash
# Using ps
ps aux | grep train.py

# Using tmux
tmux ls

# Using screen
screen -ls

# Using SLURM
squeue -u $USER
```

### Monitor training output:

```bash
# Real-time log monitoring
tail -f runs/training.log

# Last 100 lines
tail -n 100 runs/training.log

# Search for errors
grep -i error runs/training.log
```

### Check GPU usage:

```bash
# Current GPU status
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi
```

---

## Tips for Cluster Training

1. **Always redirect output** to a log file - makes debugging easier
2. **Use checkpoints** - your code already saves checkpoints, so you can resume if needed
3. **Set reasonable time limits** - know your cluster's job time limits
4. **Monitor disk space** - training generates logs, checkpoints, and metrics
5. **Use environment modules** - if available, use cluster's Python/CUDA modules
6. **Test connection first** - make sure your SSH stays connected long enough

---

## Example: Complete Cluster Setup Script

Create `scripts/run_on_cluster.sh`:

```bash
#!/bin/bash
# run_on_cluster.sh - Run training on cluster with tmux

EXPERIMENT_NAME=${1:-"cluster_run_$(date +%Y%m%d_%H%M%S)"}
PROJECT_DIR="/path/to/DSA-RL-Tracker/Experiment3_Refine"
SESSION_NAME="dsa_training"

echo "Starting training session: $SESSION_NAME"
echo "Experiment name: $EXPERIMENT_NAME"
echo "Project directory: $PROJECT_DIR"

# Create tmux session and run training
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR/src" \
    "python train.py --experiment_name $EXPERIMENT_NAME > ../training_${EXPERIMENT_NAME}.log 2>&1"

echo "Training started in tmux session: $SESSION_NAME"
echo ""
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To monitor: tail -f $PROJECT_DIR/training_${EXPERIMENT_NAME}.log"
echo "To check status: tmux ls"
```

Make it executable and run:
```bash
chmod +x scripts/run_on_cluster.sh
./scripts/run_on_cluster.sh my_experiment
```

