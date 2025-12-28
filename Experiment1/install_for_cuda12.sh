#!/bin/bash
# Installation script for CUDA 12.6 systems (like yours with RTX 4090)
# This installs PyTorch with CUDA 12.1 (compatible with CUDA 12.6)

echo "=========================================="
echo "Installing PyTorch for CUDA 12.6 System"
echo "Detected: 4x NVIDIA GeForce RTX 4090"
echo "=========================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install conda/miniconda first."
    exit 1
fi

# Check if environment already exists
ENV_NAME="dsa_rl"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment. Activate with: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

echo "Step 1: Creating conda environment..."
conda create -n ${ENV_NAME} python=3.9 -y

echo ""
echo "Step 2: Installing PyTorch with CUDA 12.1 (compatible with CUDA 12.6)..."
conda activate ${ENV_NAME}
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo ""
echo "Step 3: Installing other dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate ${ENV_NAME}"
echo "  2. Verify GPU setup: python3 check_gpu_setup.py"
echo "  3. Start training: ./run_train.sh"
echo ""

