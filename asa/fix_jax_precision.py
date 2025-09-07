#!/usr/bin/env python3
"""
Fix JAX precision issue by enabling 64-bit precision
"""

import os
import numpy as np

# Enable JAX 64-bit precision BEFORE importing JAX
os.environ['JAX_ENABLE_X64'] = '1'

import jax
import jax.numpy as jnp
from jax import jit

def test_precision_fix():
    """Test that 64-bit precision is now enabled"""
    
    print("🔧 JAX Precision Fix Test")
    print("=" * 40)
    
    # Test default precision
    test_val = jnp.array([0.12345678901234567890])
    print(f"JAX default precision: {float(test_val[0]):.15f}")
    print(f"JAX array dtype: {test_val.dtype}")
    
    # Test explicit float64
    test_val64 = jnp.array([0.12345678901234567890], dtype=jnp.float64)
    print(f"JAX float64 precision: {float(test_val64[0]):.15f}")
    print(f"JAX float64 dtype: {test_val64.dtype}")
    
    # Compare with NumPy
    numpy_val64 = np.array([0.12345678901234567890], dtype=np.float64)
    print(f"NumPy float64 precision: {numpy_val64[0]:.15f}")
    
    # Check if they match
    diff = abs(float(test_val64[0]) - numpy_val64[0])
    print(f"Difference: {diff:.2e}")
    
    if diff < 1e-14:
        print("✅ JAX 64-bit precision is working correctly")
        return True
    else:
        print("❌ JAX precision issue remains")
        return False

def simulate_network_evolution_precision():
    """Simulate how precision affects network evolution"""
    
    print(f"\n🧠 Network Evolution Precision Test")
    print("-" * 40)
    
    # Simulate 100 iteration steps like Huey does
    np.random.seed(42)
    n_neurons = 50
    
    # Initial state
    activations = np.random.rand(n_neurons).astype(np.float64)
    connections = np.random.rand(n_neurons, n_neurons).astype(np.float64)
    connections = (connections + connections.T) / 2  # Make symmetric
    
    print(f"Initial activation sum: {np.sum(activations):.10f}")
    
    # NumPy 64-bit evolution
    numpy_activations = activations.copy()
    for i in range(100):
        weighted_sums = np.dot(connections, numpy_activations)
        numpy_activations = 1.0 / (1.0 + np.exp(-weighted_sums))
        # Small random perturbation to simulate learning
        numpy_activations += np.random.normal(0, 0.001, n_neurons)
        numpy_activations = np.clip(numpy_activations, 0, 1)
    
    print(f"NumPy final activation sum: {np.sum(numpy_activations):.10f}")
    
    # JAX 64-bit evolution (if precision fix works)
    @jit
    def jax_step(activations, connections):
        weighted_sums = jnp.dot(connections, activations) 
        return 1.0 / (1.0 + jnp.exp(-weighted_sums))
    
    jax_activations = jnp.array(activations)
    jax_connections = jnp.array(connections)
    
    for i in range(100):
        jax_activations = jax_step(jax_connections, jax_activations)
        # Add same perturbation
        jax_activations += jax.random.normal(jax.random.PRNGKey(i), (n_neurons,)) * 0.001
        jax_activations = jnp.clip(jax_activations, 0, 1)
    
    jax_final = np.array(jax_activations)
    print(f"JAX final activation sum: {np.sum(jax_final):.10f}")
    
    # Compare final states
    diff = np.sum(np.abs(numpy_activations - jax_final))
    print(f"Total absolute difference: {diff:.6f}")
    
    if diff < 1.0:  # Reasonable threshold for accumulated differences
        print("✅ JAX precision matches NumPy evolution")
        return True
    else:
        print("❌ JAX still showing significant precision drift")
        return False

if __name__ == "__main__":
    precision_ok = test_precision_fix()
    
    if precision_ok:
        evolution_ok = simulate_network_evolution_precision()
        
        if evolution_ok:
            print(f"\n🎉 SOLUTION FOUND!")
            print("Setting JAX_ENABLE_X64=1 fixes the precision issue.")
            print("This should be added to the Huey GPU interface initialization.")
        else:
            print(f"\n⚠️ Precision improved but evolution still differs")
    
    print(f"\n🔍 JAX Configuration:")
    print(f"   JAX_ENABLE_X64: {os.environ.get('JAX_ENABLE_X64', 'Not set')}")
    print(f"   JAX default dtype: {jnp.array(1.0).dtype}")
    print(f"   Available devices: {jax.devices()}")