#!/usr/bin/env python3
"""
Huey PyTorch GPU Interface - Replacement for JAX Metal
This provides GPU acceleration using PyTorch MPS for Apple Silicon Macs.
"""

import torch
import numpy as np
import time
import os
from typing import Optional, Tuple, Dict, Any

class HueyPyTorchGPU:
    """
    GPU acceleration interface using PyTorch MPS for Huey neural networks.
    Provides hybrid CPU/GPU execution for optimal performance.
    """
    
    def __init__(self, use_gpu: bool = True, min_size_for_gpu: int = 1000):
        """
        Initialize PyTorch GPU interface.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            min_size_for_gpu: Minimum matrix size to use GPU (smaller uses CPU)
        """
        self.use_gpu = use_gpu and torch.backends.mps.is_available()
        self.min_size_for_gpu = min_size_for_gpu
        
        if self.use_gpu:
            self.device_gpu = torch.device("mps")
            print("🚀 PyTorch MPS GPU acceleration enabled")
        else:
            print("💻 Using CPU-only mode")
        
        self.device_cpu = torch.device("cpu")
        
        # Performance tracking
        self.kernel_calls = 0
        self.total_gpu_time = 0.0
        self.gpu_operations = []
        
        # Enable MPS fallback for unsupported operations
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    def to_tensor(self, array: np.ndarray, use_gpu: bool = None) -> torch.Tensor:
        """Convert NumPy array to PyTorch tensor on appropriate device."""
        tensor = torch.from_numpy(array.astype(np.float32))
        
        if use_gpu is None:
            # Auto-decide based on size
            use_gpu = self.use_gpu and (tensor.numel() >= self.min_size_for_gpu)
        
        if use_gpu and self.use_gpu:
            return tensor.to(self.device_gpu)
        else:
            return tensor.to(self.device_cpu)
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor back to NumPy array."""
        if tensor.device.type == "mps":
            tensor = tensor.to(self.device_cpu)
        return tensor.detach().numpy()
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated matrix multiplication with automatic device selection.
        """
        start_time = time.perf_counter()
        
        # Determine device based on size
        use_gpu = self.use_gpu and (A.size >= self.min_size_for_gpu)
        
        # Convert to tensors on same device
        A_tensor = self.to_tensor(A, use_gpu=use_gpu)
        B_tensor = self.to_tensor(B, use_gpu=use_gpu)
        
        # Ensure both tensors are on the same device
        if A_tensor.device != B_tensor.device:
            B_tensor = B_tensor.to(A_tensor.device)
        
        # Perform multiplication
        if A_tensor.device.type == "mps":
            result_tensor = torch.mm(A_tensor, B_tensor)
            torch.mps.synchronize()  # Ensure completion
            self.kernel_calls += 1
        else:
            result_tensor = torch.mm(A_tensor, B_tensor)
        
        # Convert back to NumPy
        result = self.to_numpy(result_tensor)
        
        # Track performance
        elapsed = time.perf_counter() - start_time
        if A_tensor.device.type == "mps":
            self.total_gpu_time += elapsed
            self.gpu_operations.append({
                'operation': 'matrix_multiply',
                'shape': f"{A.shape}x{B.shape}",
                'time': elapsed,
                'device': 'mps'
            })
        
        return result.astype(A.dtype)
    
    def matrix_vector_multiply(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        GPU-accelerated matrix-vector multiplication (core Hebbian operation).
        """
        start_time = time.perf_counter()
        
        # Determine device based on matrix size
        use_gpu = self.use_gpu and (matrix.size >= self.min_size_for_gpu)
        
        # Convert to tensors on same device
        matrix_tensor = self.to_tensor(matrix, use_gpu=use_gpu)
        vector_tensor = self.to_tensor(vector, use_gpu=use_gpu)
        
        # Ensure both tensors are on the same device
        if matrix_tensor.device != vector_tensor.device:
            vector_tensor = vector_tensor.to(matrix_tensor.device)
        
        # Perform multiplication
        if matrix_tensor.device.type == "mps":
            result_tensor = torch.mv(matrix_tensor, vector_tensor)
            torch.mps.synchronize()
            self.kernel_calls += 1
        else:
            result_tensor = torch.mv(matrix_tensor, vector_tensor)
        
        result = self.to_numpy(result_tensor)
        
        # Track performance
        elapsed = time.perf_counter() - start_time
        if matrix_tensor.device.type == "mps":
            self.total_gpu_time += elapsed
            self.gpu_operations.append({
                'operation': 'matrix_vector_multiply',
                'shape': f"{matrix.shape}x{vector.shape}",
                'time': elapsed,
                'device': 'mps'
            })
        
        return result.astype(vector.dtype)
    
    def sigmoid_activation(self, x: np.ndarray) -> np.ndarray:
        """GPU-accelerated sigmoid activation function."""
        start_time = time.perf_counter()
        
        x_tensor = self.to_tensor(x)
        
        if x_tensor.device.type == "mps":
            result_tensor = torch.sigmoid(x_tensor)
            torch.mps.synchronize()
            self.kernel_calls += 1
        else:
            result_tensor = torch.sigmoid(x_tensor)
        
        result = self.to_numpy(result_tensor)
        
        elapsed = time.perf_counter() - start_time
        if x_tensor.device.type == "mps":
            self.total_gpu_time += elapsed
        
        return result.astype(x.dtype)
    
    def hebbian_update(self, connections: np.ndarray, activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete Hebbian network update step with GPU acceleration.
        This is the core O(n²) bottleneck that benefits from GPU acceleration.
        """
        start_time = time.perf_counter()
        
        # Step 1: Matrix-vector multiplication (O(n²) operation)
        weighted_sums = self.matrix_vector_multiply(connections, activations)
        
        # Step 2: Sigmoid activation
        new_activations = self.sigmoid_activation(weighted_sums)
        
        elapsed = time.perf_counter() - start_time
        
        # Track this as a high-level operation
        self.gpu_operations.append({
            'operation': 'hebbian_update',
            'neurons': len(activations),
            'connections': connections.size,
            'time': elapsed,
            'device': 'hybrid'
        })
        
        return connections, new_activations
    
    def eigenvalue_decomposition(self, matrix: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eigenvalue decomposition for 3D visualization.
        Uses CPU fallback since MPS doesn't support eigh yet.
        """
        start_time = time.perf_counter()
        
        # Always use CPU for eigenvalue decomposition (MPS fallback)
        matrix_tensor = torch.from_numpy(matrix.astype(np.float32)).to(self.device_cpu)
        
        try:
            eigenvals, eigenvecs = torch.linalg.eigh(matrix_tensor)
            
            # Sort in descending order
            idx = torch.argsort(eigenvals, descending=True)
            eigenvals_sorted = eigenvals[idx]
            eigenvecs_sorted = eigenvecs[:, idx]
            
            # Take top k
            eigenvals_k = self.to_numpy(eigenvals_sorted[:k])
            eigenvecs_k = self.to_numpy(eigenvecs_sorted[:, :k])
            
        except Exception as e:
            print(f"⚠️  Eigendecomposition fallback to NumPy: {e}")
            eigenvals, eigenvecs = np.linalg.eigh(matrix)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals_k = eigenvals[idx][:k]
            eigenvecs_k = eigenvecs[:, idx][:, :k]
        
        elapsed = time.perf_counter() - start_time
        self.gpu_operations.append({
            'operation': 'eigenvalue_decomposition',
            'matrix_size': matrix.shape[0],
            'time': elapsed,
            'device': 'cpu'
        })
        
        return eigenvals_k, eigenvecs_k
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'kernel_calls': self.kernel_calls,
            'total_kernel_time': self.total_gpu_time,
            'average_kernel_time': self.total_gpu_time / max(1, self.kernel_calls),
            'gpu_enabled': self.use_gpu,
            'device_info': f"MPS ({torch.backends.mps.is_available()})" if self.use_gpu else "CPU",
            'total_operations': len(self.gpu_operations),
            'recent_operations': self.gpu_operations[-5:] if self.gpu_operations else []
        }
    
    def benchmark_performance(self, sizes: list = None) -> Dict[str, float]:
        """Run performance benchmark to verify GPU acceleration."""
        if sizes is None:
            sizes = [100, 500, 1000, 2000]
        
        results = {}
        
        print("🔥 Running PyTorch MPS benchmark...")
        
        for size in sizes:
            # Create test data
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            # GPU benchmark
            start = time.perf_counter()
            _ = self.matrix_multiply(A, B)
            gpu_time = time.perf_counter() - start
            
            # CPU benchmark
            start = time.perf_counter()
            _ = np.dot(A, B)
            cpu_time = time.perf_counter() - start
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            results[f"size_{size}"] = speedup
            
            print(f"   {size}x{size}: {speedup:.1f}x speedup")
        
        return results

def test_huey_pytorch_gpu():
    """Test the PyTorch GPU interface."""
    print("🧪 Testing Huey PyTorch GPU Interface")
    print("=" * 40)
    
    # Initialize interface
    gpu_interface = HueyPyTorchGPU()
    
    # Test basic operations
    print("\n1. Matrix operations test...")
    A = np.random.randn(1000, 1000).astype(np.float32)
    B = np.random.randn(1000, 1000).astype(np.float32)
    
    start = time.perf_counter()
    C = gpu_interface.matrix_multiply(A, B)
    elapsed = time.perf_counter() - start
    print(f"   Matrix multiply: {elapsed:.4f}s")
    
    # Test Hebbian update
    print("\n2. Hebbian update test...")
    connections = np.random.randn(500, 500).astype(np.float32)
    connections = (connections + connections.T) / 2  # Symmetric
    activations = np.random.randn(500).astype(np.float32)
    
    start = time.perf_counter()
    new_connections, new_activations = gpu_interface.hebbian_update(connections, activations)
    elapsed = time.perf_counter() - start
    print(f"   Hebbian update: {elapsed:.4f}s")
    
    # Test eigenvalue decomposition
    print("\n3. Eigenvalue decomposition test...")
    matrix = np.random.randn(100, 100).astype(np.float32)
    matrix = (matrix + matrix.T) / 2
    
    start = time.perf_counter()
    eigenvals, eigenvecs = gpu_interface.eigenvalue_decomposition(matrix, k=3)
    elapsed = time.perf_counter() - start
    print(f"   Eigendecomposition: {elapsed:.4f}s")
    print(f"   Top 3 eigenvalues: {eigenvals}")
    
    # Performance summary
    print("\n📊 Performance Summary:")
    stats = gpu_interface.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    return gpu_interface

if __name__ == "__main__":
    interface = test_huey_pytorch_gpu()
    
    # Run benchmark
    print(f"\n🏃 Running performance benchmark...")
    results = interface.benchmark_performance()
    
    max_speedup = max(results.values()) if results.values() else 0
    if max_speedup > 1.5:
        print(f"✅ GPU acceleration working! Max speedup: {max_speedup:.1f}x")
    else:
        print(f"⚠️  Limited GPU speedup. Max: {max_speedup:.1f}x")