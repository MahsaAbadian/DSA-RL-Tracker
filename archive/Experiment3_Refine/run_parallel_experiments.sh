#!/bin/bash
# Bash wrapper for parallel experiments script

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Parallel Training Experiments"
echo "=========================================="
echo ""

# Default values
EXPERIMENT_NAME="parallel_exp"
BASE_SEED=42
MAX_WORKERS=""
CONFIG_FILE=""
BASE_CONFIG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_name|-n)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --base_seed|-s)
            BASE_SEED="$2"
            shift 2
            ;;
        --max_workers|-w)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --config_file|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --base_config|-b)
            BASE_CONFIG="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_parallel_experiments.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --experiment_name, -n NAME    Base experiment name (default: parallel_exp)"
            echo "  --base_seed, -s SEED          Base seed (default: 42)"
            echo "  --max_workers, -w NUM         Max parallel workers (default: min(CPUs, 4))"
            echo "  --config_file, -c PATH        Path to config variations JSON file"
            echo "  --base_config, -b PATH        Path to base curve_config.json"
            echo ""
            echo "Example:"
            echo "  ./run_parallel_experiments.sh --experiment_name my_exp --max_workers 2"
            echo "  ./run_parallel_experiments.sh --config_file example_config_variations.json"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python run_parallel_experiments.py"
CMD="$CMD --experiment_name \"$EXPERIMENT_NAME\""
CMD="$CMD --base_seed $BASE_SEED"

if [ -n "$MAX_WORKERS" ]; then
    CMD="$CMD --max_workers $MAX_WORKERS"
fi

if [ -n "$CONFIG_FILE" ]; then
    CMD="$CMD --config_file \"$CONFIG_FILE\""
fi

if [ -n "$BASE_CONFIG" ]; then
    CMD="$CMD --base_config \"$BASE_CONFIG\""
fi

# Run
echo "Running: $CMD"
echo ""
eval $CMD

