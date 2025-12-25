#!/bin/bash
# Bash script to run training
# Curves are generated on-the-fly during training (no pre-generation needed)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Parse arguments
CLEAN_PREVIOUS=false
EXPERIMENT_NAME=""
BASE_SEED=""
CURVE_CONFIG=""
RESUME_FROM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean|-c)
            CLEAN_PREVIOUS=true
            echo "⚠️  Clean mode enabled: Previous runs will be deleted"
            shift
            ;;
        --experiment_name|-n)
            EXPERIMENT_NAME="$2"
            echo "📝 Experiment name: $EXPERIMENT_NAME"
            shift 2
            ;;
        --base_seed|-s)
            BASE_SEED="$2"
            echo "🌱 Base seed: $BASE_SEED"
            shift 2
            ;;
        --curve_config)
            CURVE_CONFIG="$2"
            echo "⚙️  Curve config: $CURVE_CONFIG"
            shift 2
            ;;
        --resume_from|-r)
            RESUME_FROM="$2"
            echo "🔄 Resume mode: Will resume from $RESUME_FROM"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_train.sh [--clean] [--experiment_name NAME] [--base_seed SEED] [--curve_config PATH] [--resume_from CHECKPOINT]"
            exit 1
            ;;
    esac
done

# Print start information
echo "=========================================="
echo "Starting DSA RL Training"
echo "Script directory: $SCRIPT_DIR"
echo "Timestamp: $(date)"
echo ""
echo "Curve Generation: On-The-Fly (no pre-generation needed)"
echo "=========================================="
echo ""

# Run training script
# Results will be saved to runs/TIMESTAMP/ or runs/EXPERIMENT_NAME_TIMESTAMP/ directory
# Using unbuffered python output for real-time logging
TRAIN_CMD="python3 -u train.py"

if [ "$CLEAN_PREVIOUS" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --clean_previous"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    TRAIN_CMD="$TRAIN_CMD --experiment_name \"$EXPERIMENT_NAME\""
fi

if [ -n "$BASE_SEED" ]; then
    TRAIN_CMD="$TRAIN_CMD --base_seed $BASE_SEED"
fi

if [ -n "$CURVE_CONFIG" ]; then
    TRAIN_CMD="$TRAIN_CMD --curve_config \"$CURVE_CONFIG\""
fi

if [ -n "$RESUME_FROM" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume_from \"$RESUME_FROM\""
fi

eval $TRAIN_CMD

# Find the most recent run directory
LATEST_RUN=$(ls -td runs/*/ 2>/dev/null | head -1)

# Print completion information
echo ""
echo "=========================================="
echo "Training completed"
if [ -n "$LATEST_RUN" ]; then
    echo "Results saved to: $LATEST_RUN"
    echo "  - Checkpoints: $LATEST_RUN/checkpoints/"
    echo "  - Final weights: $LATEST_RUN/weights/"
    echo "  - Training log: $LATEST_RUN/logs/training.log"
else
    echo "⚠️  Warning: Could not find run directory"
fi
echo "=========================================="
