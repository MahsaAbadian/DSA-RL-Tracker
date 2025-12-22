#!/bin/bash
# Bash script to run training on cluster
# This script runs training and saves all results to a timestamped run folder
# Note: Training requires pre-generated curves. Generate curves first using run_curve_generator.sh

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Parse arguments
CLEAN_PREVIOUS=false
EXPERIMENT_NAME=""
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
        --resume_from|-r)
            RESUME_FROM="$2"
            echo "🔄 Resume mode: Will resume from $RESUME_FROM"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./run_train.sh [--clean] [--experiment_name NAME] [--resume_from CHECKPOINT]"
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
echo "IMPORTANT: Training requires pre-generated curves for all stages!"
echo "Generate curves first using: ./run_curve_generator.sh"
echo ""
echo "Checking for stage-specific curve directories..."
MISSING_STAGES=""

if [ ! -d "generated_curves/stage1" ] || [ -z "$(ls -A generated_curves/stage1/curve_*.png 2>/dev/null)" ]; then
    MISSING_STAGES="${MISSING_STAGES} stage1"
fi

if [ ! -d "generated_curves/stage2" ] || [ -z "$(ls -A generated_curves/stage2/curve_*.png 2>/dev/null)" ]; then
    MISSING_STAGES="${MISSING_STAGES} stage2"
fi

if [ ! -d "generated_curves/stage3" ] || [ -z "$(ls -A generated_curves/stage3/curve_*.png 2>/dev/null)" ]; then
    MISSING_STAGES="${MISSING_STAGES} stage3"
fi

if [ -n "$MISSING_STAGES" ]; then
    echo "ERROR: Missing curves for:$MISSING_STAGES"
    echo "Please run: ./run_curve_generator.sh [num_curves]"
    exit 1
fi

STAGE1_COUNT=$(ls -1 generated_curves/stage1/curve_*.png 2>/dev/null | wc -l)
STAGE2_COUNT=$(ls -1 generated_curves/stage2/curve_*.png 2>/dev/null | wc -l)
STAGE3_COUNT=$(ls -1 generated_curves/stage3/curve_*.png 2>/dev/null | wc -l)
echo "✅ Curves found for all stages!"
echo "   Stage 1: $STAGE1_COUNT curves"
echo "   Stage 2: $STAGE2_COUNT curves"
echo "   Stage 3: $STAGE3_COUNT curves"
echo "=========================================="
echo ""

# Run training script
# Results will be saved to runs/TIMESTAMP/ or runs/EXPERIMENT_NAME_TIMESTAMP/ directory
# Using unbuffered python output for real-time logging
TRAIN_CMD="python3 -u train.py --curves_base_dir generated_curves"

if [ "$CLEAN_PREVIOUS" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --clean_previous"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    TRAIN_CMD="$TRAIN_CMD --experiment_name \"$EXPERIMENT_NAME\""
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
