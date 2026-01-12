#!/bin/bash
# run_training_cluster.sh - Run training on cluster with tmux
# Usage: ./run_training_cluster.sh [experiment_name] [experiment_dir]

set -e  # Exit on error

# Default values
EXPERIMENT_NAME=${1:-"cluster_run_$(date +%Y%m%d_%H%M%S)"}
EXPERIMENT_DIR=${2:-"Experiment3_Refine"}
SESSION_NAME="dsa_training_${EXPERIMENT_NAME}"

# Get absolute path to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="${PROJECT_ROOT}/${EXPERIMENT_DIR}"

# Check if experiment directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Experiment directory not found: $PROJECT_DIR"
    exit 1
fi

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Warning: tmux not found. Falling back to nohup..."
    USE_TMUX=false
else
    USE_TMUX=true
fi

echo "=========================================="
echo "Cluster Training Setup"
echo "=========================================="
echo "Experiment name: $EXPERIMENT_NAME"
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "Session name: $SESSION_NAME"
echo "Using tmux: $USE_TMUX"
echo "=========================================="
echo ""

# Check if session already exists
if [ "$USE_TMUX" = true ] && tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Warning: tmux session '$SESSION_NAME' already exists!"
    echo "Options:"
    echo "  1. Attach to existing session: tmux attach -t $SESSION_NAME"
    echo "  2. Kill existing session: tmux kill-session -t $SESSION_NAME"
    echo "  3. Use different experiment name"
    exit 1
fi

# Create log directory
LOG_DIR="${PROJECT_DIR}/runs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/training_${EXPERIMENT_NAME}.log"

# Change to src directory
cd "${PROJECT_DIR}/src"

if [ "$USE_TMUX" = true ]; then
    # Start training in tmux session
    echo "Starting training in tmux session: $SESSION_NAME"
    tmux new-session -d -s "$SESSION_NAME" -c "${PROJECT_DIR}/src" \
        "python train.py --experiment_name $EXPERIMENT_NAME > $LOG_FILE 2>&1"
    
    echo ""
    echo "✅ Training started in tmux session!"
    echo ""
    echo "Useful commands:"
    echo "  Attach to session:    tmux attach -t $SESSION_NAME"
    echo "  List all sessions:    tmux ls"
    echo "  Detach (inside tmux): Ctrl+B, then D"
    echo "  Kill session:         tmux kill-session -t $SESSION_NAME"
    echo ""
    echo "Monitor training:"
    echo "  tail -f $LOG_FILE"
else
    # Fallback to nohup
    echo "Starting training with nohup..."
    nohup python train.py --experiment_name "$EXPERIMENT_NAME" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    
    echo ""
    echo "✅ Training started with PID: $TRAIN_PID"
    echo ""
    echo "Monitor training:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "Check if running:"
    echo "  ps aux | grep train.py"
    echo ""
    echo "Kill if needed:"
    echo "  kill $TRAIN_PID"
fi

echo ""
echo "Log file: $LOG_FILE"
echo ""

