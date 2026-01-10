#!/bin/bash
# Helper script to run inference/rollout with weights from the latest training run

# Find the latest run directory
LATEST_RUN=$(ls -td runs/*/ 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "ERROR: No training runs found in runs/ directory"
    echo "Please run training first: ./run_train.sh"
    exit 1
fi

echo "Found latest run: $LATEST_RUN"

# Determine which weights to use (prefer Stage 3, then Stage 2, then Stage 1)
WEIGHTS_DIR="$LATEST_RUN/weights"
ACTOR_WEIGHTS=""

# Find the highest stage number available
ACTOR_WEIGHTS=$(ls -v "$WEIGHTS_DIR"/actor_Stage*_FINAL.pth 2>/dev/null | tail -n 1)

if [ -n "$ACTOR_WEIGHTS" ]; then
    STAGE_NAME=$(basename "$ACTOR_WEIGHTS" | sed 's/actor_//; s/_FINAL.pth//')
    echo "Using latest available weights: $STAGE_NAME"
else
    echo "ERROR: No final weights found in $WEIGHTS_DIR"
    echo "Available files:"
    ls -lh "$WEIGHTS_DIR" 2>/dev/null || echo "  (weights directory is empty)"
    echo ""
    echo "You can also use checkpoints:"
    echo "  python src/rollout.py --image_path <image> --actor_weights $LATEST_RUN/checkpoints/actor_Stage*_ep*.pth"
    exit 1
fi

# Check if image path is provided
if [ -z "$1" ]; then
    echo ""
    echo "Usage: $0 <image_path> [max_steps]"
    echo ""
    echo "Example:"
    echo "  $0 path/to/image.png"
    echo "  $0 path/to/image.png 2000"
    echo ""
    echo "Using weights: $ACTOR_WEIGHTS"
    exit 1
fi

IMAGE_PATH="$1"
MAX_STEPS="${2:-1000}"

# Check if image exists
if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Image file not found: $IMAGE_PATH"
    exit 1
fi

# Run inference
echo "Running inference on: $IMAGE_PATH"
echo "Max steps: $MAX_STEPS"
echo ""
KMP_DUPLICATE_LIB_OK=TRUE python src/rollout.py \
    --image_path "$IMAGE_PATH" \
    --actor_weights "$ACTOR_WEIGHTS" \
    --max_steps "$MAX_STEPS"

