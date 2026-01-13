#!/bin/bash
# Bash script to run standalone stop detector training
# This trains a supervised classifier to detect curve endpoints

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Parse arguments
EPOCHS=""
BATCH_SIZE=""
SAMPLES=""
LEARNING_RATE=""
OUTPUT_PATH=""
CONFIG_PATH=""
NO_VESSEL_REALISM=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs|-e)
            EPOCHS="$2"
            echo "📊 Epochs: $EPOCHS"
            shift 2
            ;;
        --batch_size|-b)
            BATCH_SIZE="$2"
            echo "📦 Batch Size: $BATCH_SIZE"
            shift 2
            ;;
        --samples|-s)
            SAMPLES="$2"
            echo "📈 Samples per Class: $SAMPLES"
            shift 2
            ;;
        --lr|-l)
            LEARNING_RATE="$2"
            echo "🎓 Learning Rate: $LEARNING_RATE"
            shift 2
            ;;
        --output|-o)
            OUTPUT_PATH="$2"
            echo "💾 Output Path: $OUTPUT_PATH"
            shift 2
            ;;
        --config|-c)
            CONFIG_PATH="$2"
            echo "⚙️  Config Path: $CONFIG_PATH"
            shift 2
            ;;
        --no_vessel_realism)
            NO_VESSEL_REALISM=true
            echo "⚠️  Vessel realism disabled (no tapering/fading)"
            shift
            ;;
        --help|-h)
            echo "Usage: ./run_train.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs, -e          Number of training epochs (default: 15)"
            echo "  --batch_size, -b     Batch size (default: 64)"
            echo "  --samples, -s         Samples per class (default: 5000)"
            echo "  --lr, -l              Learning rate (default: 1e-4)"
            echo "  --output, -o          Output path for weights"
            echo "  --config, -c          Path to curve config JSON"
            echo "  --no_vessel_realism   Disable vessel-realistic features (tapering/fading)"
            echo ""
            echo "Examples:"
            echo "  ./run_train.sh"
            echo "  ./run_train.sh --epochs 20 --batch_size 128"
            echo "  ./run_train.sh --samples 10000 --lr 5e-5"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print start information
echo "=========================================="
echo "Starting Standalone Stop Detector Training"
echo "Script directory: $SCRIPT_DIR"
echo "Timestamp: $(date)"
echo "=========================================="
echo ""

# Build command
TRAIN_CMD="python -u src/train_standalone.py"

if [ -n "$EPOCHS" ]; then
    TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
fi

if [ -n "$BATCH_SIZE" ]; then
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
fi

if [ -n "$SAMPLES" ]; then
    TRAIN_CMD="$TRAIN_CMD --samples $SAMPLES"
fi

if [ -n "$LEARNING_RATE" ]; then
    TRAIN_CMD="$TRAIN_CMD --lr $LEARNING_RATE"
fi

if [ -n "$OUTPUT_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --output \"$OUTPUT_PATH\""
fi

if [ -n "$CONFIG_PATH" ]; then
    TRAIN_CMD="$TRAIN_CMD --config \"$CONFIG_PATH\""
fi

if [ "$NO_VESSEL_REALISM" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --no_vessel_realism"
fi

# Run training
eval $TRAIN_CMD

# Check if training succeeded
if [ $? -eq 0 ]; then
    # Print completion information
    echo ""
    echo "=========================================="
    echo "✅ Training completed successfully"
    if [ -n "$OUTPUT_PATH" ]; then
        echo "Weights saved to: $OUTPUT_PATH"
    else
        echo "Weights saved to: StopModule/weights/stop_detector_v1.pth"
    fi
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ Training failed. Please check the error messages above."
    echo "=========================================="
    exit 1
fi

