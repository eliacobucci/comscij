#!/usr/bin/env python3
"""
Install JAX with Apple Metal GPU support for ARM64 (M-series) Macs.
This fixes the x86/ARM architecture mismatch that broke GPU acceleration.
"""

import subprocess
import sys
import platform
import os

def install_jax_metal_arm():
    """Install JAX with Metal support for ARM64."""
    
    print("🍎 Installing JAX Metal for Apple Silicon (ARM64)")
    print("=" * 50)
    
    # Verify we're on ARM
    arch = platform.machine()
    print(f"Architecture: {arch}")
    
    if arch != 'arm64':
        print("⚠️  Warning: Not running on ARM64 - this may not work correctly")
    
    try:
        # Use system Python that supports ARM natively
        python_cmd = "/usr/bin/python3"
        
        # Check if pip is available
        result = subprocess.run([python_cmd, "-m", "pip", "--version"], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ pip not available for system Python, installing...")
            # Install pip for system Python
            subprocess.run([python_cmd, "-m", "ensurepip", "--upgrade"], check=True)
        
        print("✅ Using system Python with ARM64 support")
        
        # Install JAX with Metal support
        commands = [
            # Upgrade pip first
            f"{python_cmd} -m pip install --upgrade pip",
            # Install JAX with Metal support (ARM64 compatible)
            f"{python_cmd} -m pip install --upgrade jax jaxlib",
            # Install JAX Metal plugin specifically for Apple Silicon
            f"{python_cmd} -m pip install jax-metal"
        ]
        
        for cmd in commands:
            print(f"🔨 Running: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Success!")
                if result.stdout.strip():
                    print(f"   {result.stdout.strip()}")
            else:
                print(f"❌ Failed: {cmd}")
                print(f"   Error: {result.stderr}")
                return False
        
        # Test JAX Metal installation
        print(f"\n🧪 Testing JAX Metal on ARM64...")
        test_code = '''
import platform
import jax
import jax.numpy as jnp

print(f"Python arch: {platform.machine()}")
print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# Test computation
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.dot(x, x)
print(f"Test computation result: {y}")
print(f"Computation device: {y.device()}")

# Test matrix multiply (should use Metal GPU)
import time
size = 1000
A = jnp.ones((size, size))
B = jnp.ones((size, size))

start = time.perf_counter()
C = jnp.dot(A, B)
end = time.perf_counter()

print(f"Matrix multiply ({size}x{size}): {end-start:.4f}s")
print(f"Result shape: {C.shape}, device: {C.device()}")

if 'Metal' in str(C.device()) or 'gpu' in str(C.device()).lower():
    print("🚀 JAX Metal GPU acceleration WORKING!")
else:
    print("⚠️  JAX running on CPU - Metal not activated")
'''
        
        result = subprocess.run([python_cmd, "-c", test_code], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ JAX Metal installation successful!")
            print(result.stdout)
            return True
        else:
            print("❌ JAX Metal test failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

if __name__ == "__main__":
    success = install_jax_metal_arm()
    
    if success:
        print(f"\n🎉 SUCCESS! JAX Metal is now working on ARM64")
        print(f"To use in Huey, make sure to use: /usr/bin/python3")
    else:
        print(f"\n💥 Installation failed - see errors above")