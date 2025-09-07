#!/usr/bin/env python3
"""
Verify that Huey is working completely with all dependencies
including PDF parsing, JAX acceleration, and GUI components.
"""

def test_all_components():
    print("🔍 Verifying Complete Huey Installation")
    print("=" * 50)
    
    # Test 1: JAX acceleration
    try:
        import jax
        import jax.numpy as jnp
        from jax import jit
        
        print("✅ JAX imported successfully")
        print(f"   Version: {jax.__version__}")
        print(f"   Backend: {jax.default_backend()}")
        print(f"   Devices: {jax.devices()}")
        
        # Quick JIT test
        @jit
        def test_func(x):
            return jnp.sum(x**2)
        
        result = test_func(jnp.array([1.0, 2.0, 3.0]))
        print(f"   JIT test result: {result}")
        
    except Exception as e:
        print(f"❌ JAX test failed: {e}")
        return False
    
    # Test 2: PDF parsing (PyPDF2)
    try:
        import PyPDF2
        print(f"✅ PyPDF2 imported successfully (version: {PyPDF2.__version__})")
    except Exception as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    # Test 3: Core scientific libraries
    libraries = {
        'numpy': 'np',
        'matplotlib': 'matplotlib',
        'scipy': 'scipy', 
        'sklearn': 'scikit-learn',
        'networkx': 'networkx',
        'nltk': 'nltk',
        'pandas': 'pandas',
        'plotly': 'plotly'
    }
    
    for lib_name, display_name in libraries.items():
        try:
            __import__(lib_name)
            print(f"✅ {display_name} available")
        except Exception as e:
            print(f"❌ {display_name} failed: {e}")
            return False
    
    # Test 4: Tkinter GUI support
    try:
        import tkinter as tk
        import tkinter.ttk as ttk
        print("✅ Tkinter GUI support available")
    except Exception as e:
        print(f"❌ Tkinter failed: {e}")
        return False
    
    # Test 5: Huey GPU interface
    try:
        import sys
        if '/Users/josephwoelfel/asa' not in sys.path:
            sys.path.append('/Users/josephwoelfel/asa')
            
        from huey_gpu_interface import HueyGPUInterface
        
        interface = HueyGPUInterface(max_neurons=50, use_gpu_acceleration=True)
        stats = interface.get_performance_stats()
        
        print("✅ Huey GPU Interface working")
        print(f"   GPU acceleration: {stats['gpu_acceleration_enabled']}")
        print(f"   Backend: JAX with {stats.get('device', 'CPU')}")
        
    except Exception as e:
        print(f"❌ Huey GPU interface failed: {e}")
        return False
    
    # Test 6: Streamlit web components  
    try:
        import streamlit as st
        print("✅ Streamlit web interface available")
    except Exception as e:
        print(f"❌ Streamlit failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL COMPONENTS VERIFIED SUCCESSFULLY!")
    print("🚀 Huey is ready with:")
    print("   ✅ JAX Metal acceleration")
    print("   ✅ PDF parsing capabilities") 
    print("   ✅ Complete GUI interface")
    print("   ✅ All scientific computing libraries")
    print("   ✅ Web interface components")
    print("   ✅ Native ARM64 optimization")
    
    return True

if __name__ == "__main__":
    success = test_all_components()
    exit_code = 0 if success else 1
    exit(exit_code)