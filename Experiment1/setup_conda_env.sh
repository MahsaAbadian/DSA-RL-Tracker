#!/bin/bash
# Setup script for creating conda environment for DSA RL project
# Handles existing directories and provides clear instructions

ENV_NAME="dsa_rl"
ENV_FILE="environment.yml"

echo "=========================================="
echo "Conda Environment Setup for DSA RL"
echo "=========================================="
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists"
    echo ""
    echo "Options:"
    echo "  1. Remove existing environment and create new one"
    echo "  2. Use existing environment"
    echo "  3. Use a different name"
    echo ""
    read -p "Choose option (1/2/3): " choice
    
    case $choice in
        1)
            echo "Removing existing environment..."
            conda env remove -n ${ENV_NAME} -y
            echo "✅ Environment removed"
            ;;
        2)
            echo "Using existing environment '${ENV_NAME}'"
            echo "Activating environment..."
            conda activate ${ENV_NAME}
            echo "✅ Environment activated"
            echo ""
            echo "To verify GPU setup, run:"
            echo "  python3 check_gpu_setup.py"
            exit 0
            ;;
        3)
            read -p "Enter new environment name: " ENV_NAME
            echo "Will create environment: ${ENV_NAME}"
            ;;
        *)
            echo "Invalid choice. Exiting."
            exit 1
            ;;
    esac
fi

# Check if environment.yml exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Error: $ENV_FILE not found!"
    echo "Make sure you're in the Experiment1 directory"
    exit 1
fi

echo ""
echo "Step 1: Checking CUDA version..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA version from nvidia-smi:"
    nvidia-smi | grep -i "cuda version" || nvidia-smi | head -n 3
    echo ""
    echo "⚠️  IMPORTANT: Make sure environment.yml has the correct pytorch-cuda version!"
    echo "   Edit environment.yml and set pytorch-cuda=11.8 or pytorch-cuda=12.1"
    echo "   based on your CUDA version above"
    echo ""
    read -p "Press Enter to continue after editing environment.yml (or Ctrl+C to cancel)..."
else
    echo "⚠️  nvidia-smi not found. Assuming CPU-only or CUDA not accessible."
fi

echo ""
echo "Step 2: Creating conda environment '${ENV_NAME}'..."
echo "This may take a few minutes..."
conda env create -n ${ENV_NAME} -f ${ENV_FILE}

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Environment created successfully!"
    echo ""
    echo "Step 3: Activating environment..."
    echo "Run this command to activate:"
    echo "  conda activate ${ENV_NAME}"
    echo ""
    echo "Step 4: Verify GPU setup:"
    echo "  python3 check_gpu_setup.py"
    echo ""
    echo "Step 5: Start training:"
    echo "  ./run_train.sh"
else
    echo ""
    echo "❌ Environment creation failed!"
    echo ""
    echo "Common issues:"
    echo "  1. CUDA version mismatch - edit environment.yml to match your CUDA version"
    echo "  2. Network issues - try again later"
    echo "  3. Conda channels not accessible"
    echo ""
    echo "You can also create manually:"
    echo "  conda create -n ${ENV_NAME} python=3.9"
    echo "  conda activate ${ENV_NAME}"
    echo "  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    echo "  pip install -r requirements.txt"
    exit 1
fi

