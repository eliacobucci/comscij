#!/usr/bin/env python3
"""
Force install ARM64-native JAX with Metal support.
This addresses the x86/ARM library mismatch issue.
"""

import subprocess
import sys
import platform
import os

def force_arm64_jax():
    """Force install JAX with ARM64 compatibility."""
    
    print("🔧 Force Installing ARM64 JAX Metal")
    print("=" * 40)
    
    python_cmd = "/usr/bin/python3"
    
    # Set environment for ARM64
    env = os.environ.copy()
    env['ARCHFLAGS'] = '-arch arm64'
    env['HOMEBREW_ARCH'] = 'arm64'
    env['_PYTHON_HOST_PLATFORM'] = 'macosx-11.0-arm64'
    
    try:
        # First, uninstall any existing JAX
        print("🧹 Cleaning existing JAX installation...")
        subprocess.run([python_cmd, "-m", "pip", "uninstall", "-y", "jax", "jaxlib", "jax-metal"], 
                      capture_output=True, env=env)
        
        # Try installing from Apple's recommendation
        commands = [
            # Install JAX for Apple Silicon specifically
            f"{python_cmd} -m pip install --upgrade jax[metal]",
        ]
        
        for cmd in commands:
            print(f"🔨 Running: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                print(f"✅ Success!")
                print(f"   {result.stdout.strip()}")
            else:
                print(f"❌ Failed: {cmd}")
                print(f"   Error: {result.stderr}")
                
                # Try alternative approach - building from source
                print(f"🔄 Trying source build...")
                alt_cmd = f"{python_cmd} -m pip install --no-binary=jaxlib jax"
                alt_result = subprocess.run(alt_cmd.split(), capture_output=True, text=True, env=env, timeout=300)
                
                if alt_result.returncode == 0:
                    print("✅ Source build successful!")
                else:
                    print("❌ Source build also failed")
                    return False
        
        # Test the installation
        print(f"\n🧪 Testing ARM64 JAX...")
        test_code = '''
import platform
print(f"Architecture: {platform.machine()}")

try:
    import jax
    import jax.numpy as jnp
    
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    
    # Quick test
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    print(f"Test result: {y}")
    print("✅ JAX working on ARM64!")
    
except Exception as e:
    print(f"❌ JAX test failed: {e}")
    import traceback
    traceback.print_exc()
'''
        
        result = subprocess.run([python_cmd, "-c", test_code], 
                              capture_output=True, text=True, env=env)
        
        print(result.stdout)
        if result.stderr:
            print("Stderr:", result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

if __name__ == "__main__":
    success = force_arm64_jax()
    
    if success:
        print(f"\n🎉 JAX ARM64 installation successful!")
    else:
        print(f"\n🔄 Trying alternative approach...")
        
        # Alternative: try using conda or other package manager
        print("Consider using conda with conda-forge channel for ARM64 packages:")
        print("conda install -c conda-forge jax")