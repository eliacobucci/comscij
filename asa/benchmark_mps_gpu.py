#!/usr/bin/env python3
"""
Comprehensive MPS GPU benchmark to confirm acceleration at scale.
"""

import torch
import time
import numpy as np

def benchmark_mps_performance():
    """Benchmark MPS performance at various scales."""
    
    print("🚀 MPS GPU Performance Benchmark")
    print("=" * 40)
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available")
        return False
    
    device_cpu = torch.device("cpu")
    device_mps = torch.device("mps")
    
    print(f"Testing various matrix sizes...")
    
    sizes = [100, 500, 1000, 2000, 5000]
    
    for size in sizes:
        print(f"\n📊 Matrix size: {size}x{size}")
        
        # Create test matrices
        A = torch.randn(size, size, dtype=torch.float32)
        B = torch.randn(size, size, dtype=torch.float32)
        
        # CPU benchmark
        start = time.perf_counter()
        C_cpu = torch.mm(A, B)
        cpu_time = time.perf_counter() - start
        
        # MPS benchmark  
        A_mps = A.to(device_mps)
        B_mps = B.to(device_mps)
        
        # Warmup
        _ = torch.mm(A_mps, B_mps)
        torch.mps.synchronize()
        
        start = time.perf_counter()
        C_mps = torch.mm(A_mps, B_mps)
        torch.mps.synchronize()
        mps_time = time.perf_counter() - start
        
        # Calculate speedup
        speedup = cpu_time / mps_time if mps_time > 0 else float('inf')
        
        print(f"   CPU:  {cpu_time:.4f}s")
        print(f"   MPS:  {mps_time:.4f}s")
        print(f"   Speedup: {speedup:.1f}x", end="")
        
        if speedup > 2.0:
            print(" 🚀 EXCELLENT")
        elif speedup > 1.5:
            print(" ✅ GOOD")
        elif speedup > 1.0:
            print(" 🆗 MARGINAL") 
        else:
            print(" ❌ SLOWER")
        
        # Verify correctness
        C_mps_cpu = C_mps.to(device_cpu)
        diff = torch.abs(C_cpu - C_mps_cpu).max().item()
        if diff > 1e-4:
            print(f"   ⚠️  Precision difference: {diff:.2e}")
    
    # Test Hebbian-like operations (similar to Huey)
    print(f"\n🧠 Hebbian Network Simulation")
    print("-" * 30)
    
    n_neurons = 1000
    activations = torch.randn(n_neurons, dtype=torch.float32)
    connections = torch.randn(n_neurons, n_neurons, dtype=torch.float32)
    
    # Make connections symmetric (like Hebbian networks)
    connections = (connections + connections.T) / 2
    
    # CPU Hebbian step
    start = time.perf_counter()
    for _ in range(10):  # Multiple steps
        weighted_sums = torch.mv(connections, activations)
        activations = torch.sigmoid(weighted_sums)
    cpu_hebbian_time = time.perf_counter() - start
    
    # MPS Hebbian step
    activations_mps = torch.randn(n_neurons, dtype=torch.float32, device=device_mps)
    connections_mps = connections.to(device_mps)
    
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        weighted_sums = torch.mv(connections_mps, activations_mps)
        activations_mps = torch.sigmoid(weighted_sums)
    torch.mps.synchronize()
    mps_hebbian_time = time.perf_counter() - start
    
    hebbian_speedup = cpu_hebbian_time / mps_hebbian_time
    
    print(f"Hebbian steps (10 iterations, {n_neurons} neurons):")
    print(f"   CPU:  {cpu_hebbian_time:.4f}s")
    print(f"   MPS:  {mps_hebbian_time:.4f}s") 
    print(f"   Speedup: {hebbian_speedup:.1f}x", end="")
    
    if hebbian_speedup > 2.0:
        print(" 🚀 EXCELLENT")
        gpu_working = True
    elif hebbian_speedup > 1.5:
        print(" ✅ GOOD")
        gpu_working = True
    else:
        print(" 🆗 MARGINAL")
        gpu_working = False
    
    return gpu_working

def test_eigenvalue_performance():
    """Test eigenvalue computation (used in 3D visualization)."""
    
    print(f"\n🔢 Eigenvalue Computation Test")
    print("-" * 30)
    
    if not torch.backends.mps.is_available():
        return False
    
    size = 500
    # Create symmetric matrix (like correlation matrices in Huey)
    A = torch.randn(size, size, dtype=torch.float32)
    A = (A + A.T) / 2
    
    # CPU eigenvalues
    start = time.perf_counter()
    eigenvals_cpu, eigenvecs_cpu = torch.linalg.eigh(A)
    cpu_eigen_time = time.perf_counter() - start
    
    # MPS eigenvalues
    A_mps = A.to(torch.device("mps"))
    torch.mps.synchronize()
    
    start = time.perf_counter()
    eigenvals_mps, eigenvecs_mps = torch.linalg.eigh(A_mps)
    torch.mps.synchronize()
    mps_eigen_time = time.perf_counter() - start
    
    eigen_speedup = cpu_eigen_time / mps_eigen_time
    
    print(f"Eigendecomposition ({size}x{size} matrix):")
    print(f"   CPU:  {cpu_eigen_time:.4f}s")
    print(f"   MPS:  {mps_eigen_time:.4f}s")
    print(f"   Speedup: {eigen_speedup:.1f}x", end="")
    
    if eigen_speedup > 1.5:
        print(" ✅ GOOD")
    else:
        print(" 🆗 MARGINAL")
    
    # Check precision
    eigenvals_mps_cpu = eigenvals_mps.to(torch.device("cpu"))
    diff = torch.abs(eigenvals_cpu - eigenvals_mps_cpu).max().item()
    print(f"   Max eigenvalue diff: {diff:.2e}")
    
    return eigen_speedup > 1.2

if __name__ == "__main__":
    print("🍎 Apple Silicon MPS GPU Acceleration Test")
    print("=" * 50)
    
    matrix_performance = benchmark_mps_performance()
    eigen_performance = test_eigenvalue_performance()
    
    print(f"\n📋 SUMMARY")
    print("=" * 20)
    print(f"Matrix operations: {'✅ WORKING' if matrix_performance else '❌ ISSUES'}")
    print(f"Eigenvalue operations: {'✅ WORKING' if eigen_performance else '❌ ISSUES'}")
    
    if matrix_performance:
        print(f"\n🚀 MPS GPU acceleration is working!")
        print(f"Huey can use PyTorch MPS instead of JAX Metal")
        print(f"Run with: arch -arm64 /usr/bin/python3")
    else:
        print(f"\n⚠️  MPS showing limited speedup")
        print(f"May still be better than CPU-only for large networks")