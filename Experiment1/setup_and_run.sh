#!/bin/bash
# Setup and run script for DSA RL Experiment
# This script: 1) Checks dependencies, 2) Runs training (curves generated on-the-fly)

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
    echo "Step 1: Install PyTorch via conda (with your preferred CUDA config):"
    echo "  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    echo "  # Adjust pytorch-cuda version (11.8, 12.1, etc.) based on your cluster's CUDA"
    echo ""
    echo "Step 2: Install other dependencies:"
    echo "  pip install -r requirements.txt"
    echo "  # or"
    echo "  conda install numpy scipy opencv matplotlib -c conda-forge"
    echo ""
    echo "Note: PyTorch is commented out in requirements.txt to allow custom conda installation"
    exit 1
fi

echo "✅ Dependencies found!"
echo ""
echo "Step 2: Starting training..."
echo "Curves will be generated on-the-fly during training (no pre-generation needed)"
echo ""

# Pass all arguments to run_train.sh
./run_train.sh "$@"

