#!/usr/bin/env python3
"""
GPU Setup Verification Script for Cluster Environments
Comprehensive checks for PyTorch, CUDA compatibility, and GPU functionality.
"""
import sys
import subprocess
import re

def get_system_cuda_version():
    """Get CUDA version from nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Try to extract CUDA version from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    # Extract version number (e.g., "CUDA Version: 11.8")
                    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                    if match:
                        return match.group(1)
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None

def check_cuda_compatibility(pytorch_cuda_version, system_cuda_version):
    """Check if PyTorch CUDA version is compatible with system CUDA."""
    if not system_cuda_version:
        return None, "Could not detect system CUDA version"
    
    try:
        pytorch_major, pytorch_minor = map(float, pytorch_cuda_version.split('.'))
        system_major, system_minor = map(float, system_cuda_version.split('.'))
        
        # PyTorch CUDA version should be <= system CUDA version
        # PyTorch built with CUDA 11.8 works with CUDA 11.8+ drivers
        if pytorch_major < system_major:
            return True, "Compatible (PyTorch CUDA < System CUDA)"
        elif pytorch_major == system_major:
            if pytorch_minor <= system_minor:
                return True, "Compatible (same major version)"
            else:
                return False, f"Incompatible: PyTorch CUDA {pytorch_cuda_version} > System CUDA {system_cuda_version}"
        else:
            return False, f"Incompatible: PyTorch CUDA {pytorch_cuda_version} > System CUDA {system_cuda_version}"
    except Exception as e:
        return None, f"Error checking compatibility: {e}"

def check_pytorch_cuda():
    """Check PyTorch and CUDA installation with comprehensive tests."""
    print("=" * 70)
    print("GPU Setup Verification for Cluster Environment")
    print("=" * 70)
    print()
    
    # Check system CUDA version
    print("Step 1: Checking System CUDA Version...")
    system_cuda = get_system_cuda_version()
    if system_cuda:
        print(f"   ✅ System CUDA Version: {system_cuda}")
    else:
        print("   ⚠️  Could not detect CUDA version (nvidia-smi not available or no GPU)")
        print("   This might be normal if running on CPU-only node")
    print()
    
    # Check if torch is installed
    print("Step 2: Checking PyTorch Installation...")
    try:
        import torch
        print(f"   ✅ PyTorch is installed")
        print(f"   Version: {torch.__version__}")
    except ImportError:
        print("   ❌ PyTorch is NOT installed")
        print("   Install via: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        return False
    print()
    
    # Check CUDA availability
    print("Step 3: Checking CUDA Availability...")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")
    
    if cuda_available:
        pytorch_cuda = torch.version.cuda
        print(f"   PyTorch CUDA Version: {pytorch_cuda}")
        
        # Check compatibility
        if system_cuda:
            compatible, msg = check_cuda_compatibility(pytorch_cuda, system_cuda)
            if compatible is True:
                print(f"   ✅ CUDA Compatibility: {msg}")
            elif compatible is False:
                print(f"   ⚠️  CUDA Compatibility Warning: {msg}")
                print(f"   Consider reinstalling PyTorch with matching CUDA version")
            else:
                print(f"   ⚠️  {msg}")
        
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        print()
        
        # GPU Information
        print("Step 4: GPU Information...")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   GPU {i}: {props.name}")
            print(f"   Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            print(f"   Multiprocessors: {props.multi_processor_count}")
        print()
        
        # Test GPU computation
        print("Step 5: Testing GPU Computation...")
        try:
            # Simple computation test
            print("   Running simple matrix multiplication test...")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            print("   ✅ GPU computation test PASSED")
            
            # Memory test
            print("   Testing GPU memory allocation...")
            large_tensor = torch.randn(1000, 1000, device='cuda')
            del large_tensor
            torch.cuda.empty_cache()
            print("   ✅ GPU memory test PASSED")
            
            # Performance test
            print("   Running performance benchmark...")
            import time
            x = torch.randn(5000, 5000, device='cuda')
            y = torch.randn(5000, 5000, device='cuda')
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"   ✅ Performance test PASSED ({elapsed:.3f}s for 10 matrix multiplications)")
            
        except Exception as e:
            print(f"   ❌ GPU computation test FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print()
        print("Step 6: Training Script Compatibility...")
        # Check if training script will use GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Training will use: {device.upper()}")
        if device == "cuda":
            print(f"   ✅ GPU will be leveraged for training")
        else:
            print(f"   ⚠️  Training will run on CPU (slower)")
        
    else:
        print("\n⚠️  CUDA is not available. Possible reasons:")
        print("   1. PyTorch was installed without CUDA support (CPU-only)")
        print("   2. CUDA drivers are not installed on the system")
        print("   3. GPU is not accessible (check cluster GPU allocation)")
        print("   4. CUDA version mismatch between PyTorch and system")
        print("\n   To install PyTorch with CUDA:")
        if system_cuda:
            print(f"   conda install pytorch torchvision torchaudio pytorch-cuda={system_cuda} -c pytorch -c nvidia")
        else:
            print("   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")
        print("\n   Check cluster GPU allocation:")
        print("   - Are you on a GPU node?")
        print("   - Did you request GPU resources in your job script?")
        print("   - Run: nvidia-smi to see available GPUs")
        return False
    
    print()
    print("=" * 70)
    print("✅ GPU Setup Verification Complete!")
    print("=" * 70)
    print("\nSummary:")
    print(f"  - PyTorch Version: {torch.__version__}")
    if cuda_available:
        print(f"  - CUDA Version: {pytorch_cuda}")
        if system_cuda:
            print(f"  - System CUDA: {system_cuda}")
        print(f"  - GPUs Available: {torch.cuda.device_count()}")
        print(f"  - Training Device: CUDA (GPU)")
    else:
        print(f"  - Training Device: CPU")
    print()
    return True

if __name__ == "__main__":
    success = check_pytorch_cuda()
    sys.exit(0 if success else 1)

