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
echo "Step 2: Starting training..."
echo "Curves will be generated on-the-fly during training (no pre-generation needed)"
echo ""

# Pass all arguments to run_train.sh
./run_train.sh "$@"

