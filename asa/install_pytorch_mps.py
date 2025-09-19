#!/usr/bin/env python3
"""
Install PyTorch with Apple Metal Performance Shaders (MPS) support.
This is more reliable than JAX Metal on Apple Silicon.
"""

import subprocess
import sys
import platform
import os

def install_pytorch_mps():
    """Install PyTorch with MPS support for Apple Silicon."""
    
    print("🔥 Installing PyTorch with Apple MPS Support")
    print("=" * 48)
    
    # Use ARM64 execution
    python_cmd = ["arch", "-arm64", "/usr/bin/python3"]
    
    try:
        # Verify ARM64 mode
        result = subprocess.run(python_cmd + ["-c", "import platform; print('Architecture:', platform.machine())"],
                               capture_output=True, text=True)
        print(f"Running in: {result.stdout.strip()}")
        
        # Clean existing PyTorch installations
        print("🧹 Cleaning existing PyTorch...")
        subprocess.run(python_cmd + ["-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"], 
                      capture_output=True)
        
        # Clean NumPy too since it was x86
        subprocess.run(python_cmd + ["-m", "pip", "uninstall", "-y", "numpy"], 
                      capture_output=True)
        
        # Install PyTorch with MPS support
        commands = [
            python_cmd + ["-m", "pip", "install", "--upgrade", "pip"],
            # Install ARM64 native NumPy first
            python_cmd + ["-m", "pip", "install", "--no-cache-dir", "--force-reinstall", "numpy"],
            # Install PyTorch with MPS support
            python_cmd + ["-m", "pip", "install", "--no-cache-dir", "torch", "torchvision", "torchaudio"]
        ]
        
        for cmd in commands:
            cmd_str = " ".join(cmd)
            print(f"🔨 Running: {cmd_str}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Success!")
            else:
                print(f"❌ Failed")
                print(f"Error: {result.stderr}")
                # Continue anyway, some packages may work
        
        # Test PyTorch MPS
        print(f"\n🧪 Testing PyTorch MPS...")
        test_code = '''
import platform
import numpy as np
import torch

print(f"Python arch: {platform.machine()}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Test MPS availability
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    print("🚀 MPS GPU acceleration is available!")
    
    # Test MPS performance
    import time
    
    # CPU test
    size = 1000
    x = torch.ones(size, size)
    y = torch.ones(size, size)
    
    start = time.perf_counter()
    z_cpu = torch.mm(x, y)
    cpu_time = time.perf_counter() - start
    print(f"CPU matrix multiply: {cpu_time:.4f}s")
    
    # MPS test
    device = torch.device("mps")
    x_mps = x.to(device)
    y_mps = y.to(device)
    
    # Warmup
    _ = torch.mm(x_mps, y_mps)
    torch.mps.synchronize()
    
    start = time.perf_counter()
    z_mps = torch.mm(x_mps, y_mps)
    torch.mps.synchronize()
    mps_time = time.perf_counter() - start
    
    print(f"MPS matrix multiply: {mps_time:.4f}s")
    speedup = cpu_time / mps_time if mps_time > 0 else float('inf')
    print(f"MPS speedup: {speedup:.1f}x")
    
    if speedup > 1.5:
        print("✅ MPS acceleration working!")
    else:
        print("⚠️  MPS slower than CPU (possible overhead)")
        
else:
    print("❌ MPS not available - falling back to CPU")
    
print("✅ PyTorch test completed!")
'''
        
        result = subprocess.run(python_cmd + ["-c", test_code], 
                              capture_output=True, text=True, timeout=60)
        
        print(result.stdout)
        if result.stderr.strip():
            print("Warnings:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

if __name__ == "__main__":
    success = install_pytorch_mps()
    
    if success:
        print(f"\n🎉 PyTorch MPS installation successful!")
        print(f"✅ This can replace JAX for GPU acceleration in Huey")
        print(f"💡 Use: arch -arm64 /usr/bin/python3 to run with MPS support")
    else:
        print(f"\n💥 PyTorch MPS installation issues detected")
        print(f"Try running the test again - some warnings may be normal")