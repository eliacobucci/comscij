#!/usr/bin/env python3
"""
Install JAX with Metal support using native ARM64 execution.
This should resolve the x86/ARM architecture mismatch.
"""

import subprocess
import sys
import os

def install_native_arm64_jax():
    """Install JAX using forced ARM64 execution."""
    
    print("🍎 Installing JAX Metal with Native ARM64 Execution")
    print("=" * 52)
    
    # Force ARM64 execution
    python_cmd = ["arch", "-arm64", "/usr/bin/python3"]
    
    try:
        # Test ARM64 mode is working
        result = subprocess.run(python_cmd + ["-c", "import platform; print('Architecture:', platform.machine())"],
                               capture_output=True, text=True)
        
        if "arm64" not in result.stdout:
            print("❌ Failed to force ARM64 execution")
            return False
        
        print("✅ Running in native ARM64 mode")
        
        # Clean any existing installation
        print("🧹 Cleaning existing JAX...")
        subprocess.run(python_cmd + ["-m", "pip", "uninstall", "-y", "jax", "jaxlib", "jax-metal"], 
                      capture_output=True)
        
        # Install JAX with Metal support in ARM64 mode
        commands = [
            python_cmd + ["-m", "pip", "install", "--upgrade", "pip"],
            python_cmd + ["-m", "pip", "install", "--no-cache-dir", "jax", "jaxlib"],
            python_cmd + ["-m", "pip", "install", "--no-cache-dir", "jax-metal"]
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
                return False
        
        # Test installation
        print(f"\n🧪 Testing JAX Metal in ARM64 mode...")
        test_code = '''
import platform
import jax
import jax.numpy as jnp

print(f"Python architecture: {platform.machine()}")
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# Test basic computation
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.dot(x, x)
print(f"Test computation: {y}")
print(f"Device: {y.device()}")

# Test performance
import time
size = 1000
A = jnp.ones((size, size))
B = jnp.ones((size, size))

# Warmup
_ = jnp.dot(A, B)

start = time.perf_counter()
C = jnp.dot(A, B)
C.block_until_ready()  # Wait for GPU computation
end = time.perf_counter()

print(f"Matrix multiply ({size}x{size}): {end-start:.4f}s")
print(f"Device used: {C.device()}")

if 'gpu' in str(C.device()).lower() or 'metal' in str(C.device()).lower():
    print("🚀 GPU/Metal acceleration detected!")
else:
    print("ℹ️  Running on CPU")
    
print("✅ JAX installation working!")
'''
        
        result = subprocess.run(python_cmd + ["-c", test_code], 
                              capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr.strip():
            print("Warnings/Errors:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

if __name__ == "__main__":
    success = install_native_arm64_jax()
    
    if success:
        print(f"\n🎉 JAX Metal ARM64 installation successful!")
        print(f"Use: arch -arm64 /usr/bin/python3 to run Huey with GPU acceleration")
    else:
        print(f"\n💥 Installation failed")
        
        # Show alternative
        print(f"\nAlternative approaches to try:")
        print(f"1. Install Homebrew Python: brew install python")
        print(f"2. Use Miniconda for ARM64: https://docs.conda.io/en/latest/miniconda.html")
        print(f"3. Use PyTorch with MPS instead of JAX")