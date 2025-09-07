#!/usr/bin/env python3
"""
Install JAX with Apple Metal GPU support for Huey acceleration.
"""

import subprocess
import sys

def install_jax_metal():
    """Install JAX with Apple Metal support."""
    
    print("🚀 Installing JAX with Apple Metal GPU support...")
    print("=" * 50)
    
    try:
        # Install JAX with Metal support (Apple Silicon)
        commands = [
            "pip install --upgrade jax jaxlib",
            "pip install jax-metal"  # Apple's Metal plugin for JAX
        ]
        
        for cmd in commands:
            print(f"Running: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Success: {cmd}")
            else:
                print(f"❌ Failed: {cmd}")
                print(f"Error: {result.stderr}")
        
        # Test JAX Metal
        print("\n🧪 Testing JAX Metal installation...")
        test_code = '''
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX backend: {jax.default_backend()}")
print(f"Available devices: {jax.devices()}")

# Simple test
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.dot(x, x)
print(f"Test computation: {y}")
print("🚀 JAX Metal working!")
'''
        
        result = subprocess.run([sys.executable, "-c", test_code], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ JAX Metal installation successful!")
            print(result.stdout)
        else:
            print("❌ JAX Metal test failed:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Installation error: {e}")

if __name__ == "__main__":
    install_jax_metal()