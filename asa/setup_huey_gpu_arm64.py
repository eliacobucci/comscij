#!/usr/bin/env python3
"""
Setup script for HueyGPU with ARM64 Python and JAX Metal support
"""

import subprocess
import sys
import platform

def run_command(cmd, description):
    """Run a shell command and display results"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} successful")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {description} failed")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    return True

def main():
    print("🚀 HueyGPU ARM64 Setup")
    print("=" * 50)
    print(f"Current architecture: {platform.machine()}")
    print(f"Python executable: {sys.executable}")
    
    if platform.machine() != 'arm64':
        print("\n⚠️  This script should be run with ARM64 Python:")
        print("   arch -arm64 python3 setup_huey_gpu_arm64.py")
        return
    
    # Required packages for HueyGPU
    packages = [
        "jax[metal]",  # JAX with Metal GPU support
        "streamlit",   # Web interface
        "numpy",       # Numerical computing
        "matplotlib",  # Plotting
        "seaborn",     # Statistical visualization
        "scikit-learn", # Machine learning tools
        "nltk"         # Natural language processing
    ]
    
    print(f"\n📦 Installing {len(packages)} required packages...")
    
    for package in packages:
        print(f"\n📥 Installing {package}...")
        success = run_command(f"pip3 install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️  Failed to install {package} - continuing with others...")
    
    # Test JAX Metal functionality
    print("\n🧪 Testing JAX Metal functionality...")
    test_code = '''
import jax
import jax.numpy as jnp
import platform

print(f"Architecture: {platform.machine()}")
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Test basic JAX operation
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.dot(x, x)
print(f"JAX test computation: {y}")

if any("metal" in str(d).lower() for d in jax.devices()):
    print("🚀 JAX Metal GPU acceleration is ENABLED!")
else:
    print("💻 JAX running on CPU")
'''
    
    try:
        exec(test_code)
        print("\n✅ JAX setup successful!")
    except Exception as e:
        print(f"\n❌ JAX test failed: {e}")
        print("   Try: pip3 install --upgrade jax jax-metal")
    
    print("\n🏁 Setup complete!")
    print("\nTo launch HueyGPU with GPU acceleration:")
    print("   ./launch_huey_gpu_arm64.command")
    print("   OR")
    print("   arch -arm64 python3 -m streamlit run huey_gpu_web_interface_complete.py --server.port=8505")

if __name__ == "__main__":
    main()