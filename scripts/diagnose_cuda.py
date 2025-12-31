#!/usr/bin/env python3
"""
Quick diagnostic script to check PyTorch CUDA installation
"""
import sys

print("=" * 70)
print("PyTorch CUDA Diagnostic")
print("=" * 70)
print()

# Check if torch is installed
try:
    import torch
    print(f"✅ PyTorch is installed")
    print(f"   Version: {torch.__version__}")
except ImportError:
    print("❌ PyTorch is NOT installed")
    sys.exit(1)

print()

# Check CUDA availability
print("CUDA Status:")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   PyTorch CUDA Version: {torch.version.cuda}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print()
    print("❌ CUDA is NOT available")
    print()
    print("Checking PyTorch build info...")
    print(f"   PyTorch version: {torch.__version__}")
    
    # Check if it's CPU-only build
    if '+cpu' in torch.__version__:
        print("   ⚠️  This is a CPU-only build of PyTorch!")
        print("   You need to install PyTorch with CUDA support.")
    else:
        print("   ⚠️  PyTorch may have been built without CUDA support")
    
    print()
    print("To fix this, reinstall PyTorch with CUDA:")
    print("   1. Uninstall current PyTorch:")
    print("      conda remove pytorch torchvision torchaudio")
    print("      # or")
    print("      pip uninstall torch torchvision torchaudio")
    print()
    print("   2. Install PyTorch with CUDA 12.1 (compatible with CUDA 12.6):")
    print("      conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
    print()
    print("   3. Verify installation:")
    print("      python3 -c \"import torch; print('CUDA:', torch.cuda.is_available())\"")

print()
print("=" * 70)


