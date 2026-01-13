#!/bin/bash
# train.slurm - SLURM job script for training
# Submit with: sbatch train.slurm
# Check status with: squeue -u $USER
# Cancel with: scancel <job_id>

#SBATCH --job-name=dsa_rl_train
#SBATCH --output=runs/slurm_%j.out
#SBATCH --error=runs/slurm_%j.err
#SBATCH --time=72:00:00          # Time limit (72 hours)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=32GB               # Memory request
#SBATCH --partition=gpu          # Adjust partition name for your cluster

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment (if using conda)
# source ~/.bashrc
# conda activate your_env
# Or use virtualenv:
# source /path/to/venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print GPU info
echo "GPU Info:"
nvidia-smi

# Navigate to src directory (adjust path if needed)
# If submitting from Experiment3_Refine directory:
cd src
# If submitting from project root:
# cd Experiment3_Refine/src

# Experiment name (optional - can also pass via command line)
EXPERIMENT_NAME="slurm_run_${SLURM_JOB_ID}"

# Start GPU monitoring in background (optional)
nvidia-smi dmon -s u -d 10 > ../runs/gpu_monitor_${SLURM_JOB_ID}.log &
MONITOR_PID=$!

# Run training
echo "Starting training..."
python train.py --experiment_name "$EXPERIMENT_NAME"

# Training completed
TRAIN_EXIT_CODE=$?

# Stop GPU monitoring
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "Training finished with exit code: $TRAIN_EXIT_CODE"
echo "End time: $(date)"

exit $TRAIN_EXIT_CODE

