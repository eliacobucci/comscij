#!/usr/bin/env python3
"""
Test JAX with Huey on native ARM64 environment
"""

# Test JAX functionality first
def test_jax_setup():
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit
        
        print("🚀 JAX Import Test")
        print(f"   Version: {jax.__version__}")
        print(f"   Backend: {jax.default_backend()}")
        print(f"   Devices: {jax.devices()}")
        
        # Test JIT compilation
        @jit
        def fast_function(x):
            return jnp.dot(x, x) + jnp.sum(x)
        
        test_array = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fast_function(test_array)
        
        print(f"   JIT test result: {result}")
        print("   ✅ JAX working perfectly!")
        
        return True
        
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        return False

# Test Huey GPU interface
def test_huey_with_jax():
    try:
        # Import Huey components
        import sys
        import os
        
        # Add the asa directory to path if needed
        if '/Users/josephwoelfel/asa' not in sys.path:
            sys.path.append('/Users/josephwoelfel/asa')
        
        from huey_gpu_interface import HueyGPUInterface
        import numpy as np
        
        print("\n🧠 Huey GPU Interface Test")
        
        # Create interface - should now use JAX
        interface = HueyGPUInterface(max_neurons=100, use_gpu_acceleration=True)
        
        # Test matrix operation
        test_matrix = np.random.rand(50, 50).astype(np.float32)
        test_matrix = (test_matrix + test_matrix.T) / 2  # Make symmetric
        
        print("   Testing eigenvalue calculation...")
        import time
        start_time = time.perf_counter()
        
        eigenvals, eigenvecs = interface.calculate_eigenvalues_gpu(test_matrix)
        
        elapsed = time.perf_counter() - start_time
        
        print(f"   ✅ Eigenvalue calculation completed in {elapsed:.4f}s")
        print(f"   Top 3 eigenvalues: {eigenvals[:3]}")
        print(f"   Matrix size: {test_matrix.shape}")
        
        # Get performance stats
        stats = interface.get_performance_stats()
        print(f"   GPU acceleration: {stats['gpu_acceleration_enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Huey test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔬 Testing JAX + Huey on Native ARM64")
    print("=" * 50)
    
    # Test 1: JAX functionality
    jax_works = test_jax_setup()
    
    # Test 2: Huey with JAX
    if jax_works:
        huey_works = test_huey_with_jax()
        
        if huey_works:
            print("\n🎉 SUCCESS!")
            print("JAX is working with Huey on native ARM64!")
            print("Ready to launch Huey GUI with JAX acceleration!")
        else:
            print("\n⚠️  JAX works but Huey integration needs adjustment")
    else:
        print("\n❌ JAX setup failed")
    
    print("\n" + "=" * 50)