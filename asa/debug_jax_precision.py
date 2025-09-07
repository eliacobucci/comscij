#!/usr/bin/env python3
"""
Debug JAX vs NumPy precision differences in Huey calculations
"""

import numpy as np
import sys
import os

# Add asa to path
if '/Users/josephwoelfel/asa' not in sys.path:
    sys.path.append('/Users/josephwoelfel/asa')

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
    print("✅ JAX available for comparison")
except ImportError:
    jax = None
    jnp = None
    jit = None
    JAX_AVAILABLE = False
    print("❌ JAX not available - running NumPy only")

def test_hebbian_precision():
    """Compare JAX vs NumPy for exact same calculations"""
    
    print("\n🧮 Hebbian Calculation Precision Test")
    print("=" * 50)
    
    # Create test data - same as what Huey would use
    np.random.seed(42)  # Reproducible results
    n_neurons = 10
    
    # Test activation calculation (core Hebbian operation)
    current_activations = np.random.rand(n_neurons).astype(np.float32)
    connections_matrix = np.random.rand(n_neurons, n_neurons).astype(np.float32)
    
    print(f"Test matrix size: {n_neurons}x{n_neurons}")
    print(f"NumPy input dtype: {current_activations.dtype}")
    print(f"Sample activation values: {current_activations[:3]}")
    
    # NumPy calculation (original method)
    numpy_weighted_sums = np.dot(connections_matrix, current_activations)
    numpy_result = 1.0 / (1.0 + np.exp(-numpy_weighted_sums))
    
    print(f"\n📊 NumPy Results:")
    print(f"   Weighted sums: {numpy_weighted_sums[:3]}")
    print(f"   Final activations: {numpy_result[:3]}")
    print(f"   Sum of activations: {np.sum(numpy_result)}")
    print(f"   Result dtype: {numpy_result.dtype}")
    
    if JAX_AVAILABLE:
        # JAX calculation (new method)
        import jax.numpy as jnp
        from jax import jit
        
        jax_activations = jnp.array(current_activations)
        jax_connections = jnp.array(connections_matrix)
        
        @jit
        def jax_hebbian_calculation(connections, activations):
            weighted_sums = jnp.dot(connections, activations)
            return 1.0 / (1.0 + jnp.exp(-weighted_sums))
        
        jax_result = jax_hebbian_calculation(jax_connections, jax_activations)
        jax_result_np = np.array(jax_result)  # Convert back to NumPy for comparison
        
        print(f"\n🚀 JAX Results:")
        print(f"   JAX dtype: {jax_result.dtype}")
        print(f"   Final activations: {jax_result_np[:3]}")
        print(f"   Sum of activations: {np.sum(jax_result_np)}")
        
        # Calculate differences
        absolute_diff = np.abs(numpy_result - jax_result_np)
        relative_diff = absolute_diff / (numpy_result + 1e-10)  # Avoid division by zero
        
        print(f"\n🔍 Precision Comparison:")
        print(f"   Max absolute difference: {np.max(absolute_diff):.2e}")
        print(f"   Max relative difference: {np.max(relative_diff):.2e}")
        print(f"   Mean absolute difference: {np.mean(absolute_diff):.2e}")
        
        # Check if differences are significant
        if np.max(absolute_diff) > 1e-6:
            print("   ⚠️  SIGNIFICANT PRECISION DIFFERENCE DETECTED")
            print("   This could explain the different results!")
            
            # Show where the biggest differences are
            max_diff_idx = np.argmax(absolute_diff)
            print(f"   Biggest diff at index {max_diff_idx}:")
            print(f"     NumPy: {numpy_result[max_diff_idx]:.10f}")
            print(f"     JAX:   {jax_result_np[max_diff_idx]:.10f}")
            print(f"     Diff:  {absolute_diff[max_diff_idx]:.2e}")
            
        else:
            print("   ✅ Precision differences are negligible")
    
    # Test eigenvalue calculation precision
    print(f"\n🔢 Eigenvalue Calculation Test")
    print("-" * 30)
    
    # Create symmetric test matrix
    test_matrix = np.random.rand(5, 5).astype(np.float32)
    test_matrix = (test_matrix + test_matrix.T) / 2  # Make symmetric
    
    # NumPy eigenvalues
    numpy_eigenvals, numpy_eigenvecs = np.linalg.eigh(test_matrix)
    numpy_eigenvals = numpy_eigenvals[::-1]  # Sort descending
    
    print(f"NumPy eigenvalues: {numpy_eigenvals}")
    
    if JAX_AVAILABLE:
        # JAX eigenvalues
        import jax.numpy as jnp
        jax_matrix = jnp.array(test_matrix)
        jax_eigenvals, jax_eigenvecs = jnp.linalg.eigh(jax_matrix)
        
        # Sort descending
        idx = jnp.argsort(jax_eigenvals)[::-1]
        jax_eigenvals_sorted = jax_eigenvals[idx]
        
        jax_eigenvals_np = np.array(jax_eigenvals_sorted)
        print(f"JAX eigenvalues:   {jax_eigenvals_np}")
        
        eigenval_diff = np.abs(numpy_eigenvals - jax_eigenvals_np)
        print(f"Eigenvalue differences: {eigenval_diff}")
        print(f"Max eigenvalue diff: {np.max(eigenval_diff):.2e}")
        
        if np.max(eigenval_diff) > 1e-5:
            print("⚠️  EIGENVALUE PRECISION ISSUE DETECTED")
        else:
            print("✅ Eigenvalue precision is acceptable")

def test_data_types():
    """Test different data type behaviors"""
    print(f"\n🏷️  Data Type Analysis")
    print("-" * 30)
    
    # Test float32 vs float64
    test_val = np.array([0.12345678901234567890], dtype=np.float32)
    print(f"float32 precision: {test_val[0]:.15f}")
    
    test_val64 = np.array([0.12345678901234567890], dtype=np.float64)  
    print(f"float64 precision: {test_val64[0]:.15f}")
    
    if JAX_AVAILABLE:
        jax_val32 = jnp.array([0.12345678901234567890], dtype=jnp.float32)
        jax_val64 = jnp.array([0.12345678901234567890], dtype=jnp.float64)
        
        print(f"JAX float32: {float(jax_val32[0]):.15f}")
        print(f"JAX float64: {float(jax_val64[0]):.15f}")

if __name__ == "__main__":
    test_hebbian_precision()
    test_data_types()
    
    print(f"\n💡 DIAGNOSIS:")
    print("If significant precision differences are detected above,")
    print("the issue is likely JAX using different floating point")
    print("precision or numerical algorithms than NumPy.")
    print("This would cause different network evolution patterns.")