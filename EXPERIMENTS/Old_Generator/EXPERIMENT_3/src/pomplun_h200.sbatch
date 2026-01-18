#!/bin/bash
#SBATCH --job-name=pomplun_h200_test
#SBATCH --partition=pomplun
#SBATCH --account=cs_funda.durupinarbabur

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=100G
#SBATCH --time=00-10:00:00
#SBATCH --gres=gpu:h200:1

#SBATCH --output=pomplun_%j.out
#SBATCH --error=pomplun_%j.err

echo "===== JOB INFO ====="
echo "JobID:   $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Tasks:   $SLURM_NTASKS"
echo "GPUs:    $SLURM_GPUS"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "===================="

echo
echo "===== NVIDIA-SMI ====="
nvidia-smi
echo "======================"

source ~/.bashrc
conda activate RL

# Get the directory where this script is located (absolute path)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Go up one level from src/ to EXPERIMENT_3/ (ensure absolute path)
PROJECT_DIR="$(cd "$(dirname "$SCRIPT_DIR")" && pwd)"
cd "$PROJECT_DIR" || { 
    echo "ERROR: Failed to cd to $PROJECT_DIR"
    echo "Current directory: $(pwd)"
    echo "Script directory: $SCRIPT_DIR"
    echo "Attempted project directory: $PROJECT_DIR"
    exit 1
}

echo "Current directory: $(pwd)"
echo "Script directory: $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "Running training from: $(pwd)"

# Verify the Python script exists
if [ ! -f "src/train_deeper_model.py" ]; then
    echo "ERROR: train_deeper_model.py not found at src/train_deeper_model.py"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

python src/train_deeper_model.py \
  --resume_from runs/20260116_030245/weights/model_Bin4_Mixed_Random_FINAL.pth \
  --curve_config config/curve_config.json