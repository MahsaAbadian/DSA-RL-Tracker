#!/bin/bash
# Setup and run script for DSA RL Experiment
# This script: 1) Checks dependencies, 2) Generates curves, 3) Runs training

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "DSA RL Experiment Setup & Run"
echo "=========================================="
echo ""

# Check if dependencies are installed
echo "Step 1: Checking dependencies..."
python3 -c "import torch, numpy, scipy, cv2, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not found!"
    echo ""
    echo "Please install dependencies first:"
    echo ""
    echo "Option 1 - conda (recommended):"
    echo "  conda install pytorch numpy scipy opencv matplotlib -c pytorch -c conda-forge"
    echo ""
    echo "Option 2 - pip:"
    echo "  pip3 install -r requirements.txt"
    echo ""
    exit 1
fi

echo "✅ Dependencies found!"
echo ""

# Check if curves already exist
CURVES_BASE_DIR="generated_curves"
NUM_CURVES=${1:-1000}  # Default 1000 curves per stage, can override with first argument

echo "Step 2: Checking for pre-generated curves..."
# Check if all stage directories exist and have curves
STAGE1_EXISTS=false
STAGE2_EXISTS=false
STAGE3_EXISTS=false

if [ -d "$CURVES_BASE_DIR/stage1" ] && [ -n "$(ls -A $CURVES_BASE_DIR/stage1/curve_*.png 2>/dev/null)" ]; then
    STAGE1_COUNT=$(ls -1 $CURVES_BASE_DIR/stage1/curve_*.png 2>/dev/null | wc -l)
    echo "✅ Found $STAGE1_COUNT curves in $CURVES_BASE_DIR/stage1/"
    STAGE1_EXISTS=true
fi

if [ -d "$CURVES_BASE_DIR/stage2" ] && [ -n "$(ls -A $CURVES_BASE_DIR/stage2/curve_*.png 2>/dev/null)" ]; then
    STAGE2_COUNT=$(ls -1 $CURVES_BASE_DIR/stage2/curve_*.png 2>/dev/null | wc -l)
    echo "✅ Found $STAGE2_COUNT curves in $CURVES_BASE_DIR/stage2/"
    STAGE2_EXISTS=true
fi

if [ -d "$CURVES_BASE_DIR/stage3" ] && [ -n "$(ls -A $CURVES_BASE_DIR/stage3/curve_*.png 2>/dev/null)" ]; then
    STAGE3_COUNT=$(ls -1 $CURVES_BASE_DIR/stage3/curve_*.png 2>/dev/null | wc -l)
    echo "✅ Found $STAGE3_COUNT curves in $CURVES_BASE_DIR/stage3/"
    STAGE3_EXISTS=true
fi

if [ "$STAGE1_EXISTS" = true ] && [ "$STAGE2_EXISTS" = true ] && [ "$STAGE3_EXISTS" = true ]; then
    echo ""
    read -p "Regenerate curves for all stages? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Generating $NUM_CURVES curves per stage..."
        ./run_curve_generator.sh "$NUM_CURVES" "$CURVES_BASE_DIR"
    else
        echo "Using existing curves."
    fi
else
    echo ""
    echo "Missing curves for some stages. Generating $NUM_CURVES curves per stage..."
    echo ""
    ./run_curve_generator.sh "$NUM_CURVES" "$CURVES_BASE_DIR"
fi

echo ""
echo "Step 3: Starting training..."
echo ""

# Parse additional arguments (experiment name, clean, resume)
TRAIN_ARGS=""
if [ -n "$2" ]; then
    TRAIN_ARGS="$2"
    if [ -n "$3" ]; then
        TRAIN_ARGS="$TRAIN_ARGS $3"
    fi
    if [ -n "$4" ]; then
        TRAIN_ARGS="$TRAIN_ARGS $4"
    fi
fi

# Run training
./run_train.sh $TRAIN_ARGS

