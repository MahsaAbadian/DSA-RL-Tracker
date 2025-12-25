#!/bin/bash
# Script to check CUDA version and PyTorch compatibility on cluster
# Helps ensure GPU can be leveraged for training

echo "=========================================="
echo "CUDA Version & PyTorch Compatibility Check"
echo "=========================================="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found!"
    echo ""
    echo "Possible reasons:"
    echo "  1. Not on a GPU node"
    echo "  2. CUDA drivers not installed"
    echo "  3. GPU not allocated in your job"
    echo ""
    echo "Check your cluster job submission:"
    echo "  - Did you request GPU resources?"
    echo "  - Are you on the correct node type?"
    exit 1
fi

echo "Step 1: System CUDA Information"
echo "-------------------------------"
nvidia-smi
echo ""

# Extract CUDA version from nvidia-smi
CUDA_VERSION=$(nvidia-smi | grep -i "cuda version" | head -1 | sed 's/.*CUDA Version: //' | sed 's/ .*//')
if [ -n "$CUDA_VERSION" ]; then
    echo "✅ Detected CUDA Version: $CUDA_VERSION"
else
    CUDA_VERSION=$(nvidia-smi | grep -i "driver version" | head -1)
    echo "⚠️  Could not extract exact CUDA version"
    echo "   Driver info: $CUDA_VERSION"
fi
echo ""

# Check Python/PyTorch
echo "Step 2: PyTorch CUDA Compatibility"
echo "-----------------------------------"
if python3 -c "import torch" 2>/dev/null; then
    PYTORCH_CUDA=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    
    echo "PyTorch CUDA Version: $PYTORCH_CUDA"
    echo "CUDA Available: $CUDA_AVAILABLE"
    echo ""
    
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        echo "✅ PyTorch can access GPU"
        echo ""
        echo "Compatibility Check:"
        if [ -n "$CUDA_VERSION" ] && [ -n "$PYTORCH_CUDA" ]; then
            # Extract major.minor versions
            SYS_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
            SYS_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
            PYT_MAJOR=$(echo $PYTORCH_CUDA | cut -d. -f1)
            PYT_MINOR=$(echo $PYTORCH_CUDA | cut -d. -f2)
            
            if [ "$PYT_MAJOR" -lt "$SYS_MAJOR" ] || ([ "$PYT_MAJOR" -eq "$SYS_MAJOR" ] && [ "$PYT_MINOR" -le "$SYS_MINOR" ]); then
                echo "✅ Compatible: PyTorch CUDA $PYTORCH_CUDA <= System CUDA $CUDA_VERSION"
            else
                echo "⚠️  Warning: PyTorch CUDA $PYTORCH_CUDA > System CUDA $CUDA_VERSION"
                echo "   Consider reinstalling PyTorch with CUDA $CUDA_VERSION"
            fi
        fi
    else
        echo "❌ PyTorch cannot access GPU"
        echo ""
        echo "Possible fixes:"
        echo "  1. Reinstall PyTorch with matching CUDA version:"
        if [ -n "$CUDA_VERSION" ]; then
            echo "     conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia"
        else
            echo "     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
        fi
        echo ""
        echo "  2. Check GPU allocation:"
        echo "     - Are you on a GPU node?"
        echo "     - Did you request GPU in your job script?"
    fi
else
    echo "❌ PyTorch is not installed"
    echo ""
    echo "Install PyTorch with CUDA:"
    if [ -n "$CUDA_VERSION" ]; then
        echo "  conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia"
    else
        echo "  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia"
    fi
fi

echo ""
echo "Step 3: Run Comprehensive GPU Test"
echo "-----------------------------------"
echo "For detailed GPU testing, run:"
echo "  python3 check_gpu_setup.py"
echo ""

