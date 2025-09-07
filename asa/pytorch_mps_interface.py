#!/usr/bin/env python3
"""
PyTorch MPS (Metal Performance Shaders) interface for Huey GPU acceleration.
Alternative to JAX for Apple Silicon Macs.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional

# PyTorch with MPS support
try:
    import torch
    TORCH_MPS_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    if TORCH_MPS_AVAILABLE:
        print("🍎 PyTorch MPS (Metal Performance Shaders) available")
    else:
        print("💻 PyTorch available (CPU only)")
except:
    TORCH_MPS_AVAILABLE = False
    print("❌ PyTorch not available")

class HueyPyTorchMPSInterface:
    """
    PyTorch MPS interface for Mac GPU acceleration.
    Uses Apple's Metal Performance Shaders for neural network operations.
    """
    
    def __init__(self, max_neurons: int = 500, use_mps_acceleration: bool = True):
        """Initialize PyTorch MPS interface."""
        
        self.max_neurons = max_neurons
        self.use_mps = use_mps_acceleration and TORCH_MPS_AVAILABLE
        
        # Set device
        if self.use_mps:
            self.device = torch.device("mps")
            print(f"🍎 Using Metal Performance Shaders (MPS)")
        else:
            self.device = torch.device("cpu")
            print(f"💻 Using CPU acceleration")
        
        # Performance tracking
        self.kernel_calls = 0
        self.total_kernel_time = 0.0
        
        print(f"🚀 Huey PyTorch MPS Interface initialized")
        print(f"   Device: {self.device}")
        print(f"   Max neurons: {max_neurons}")
    
    def calculate_activations_mps(self, huey_network) -> Dict[int, float]:
        """Calculate activations using PyTorch MPS acceleration."""
        
        if not self.use_mps:
            return self._calculate_activations_cpu(huey_network)
        
        start_time = time.perf_counter()
        
        try:
            # Get network state
            neurons = list(huey_network.activations.keys())
            n_neurons = len(neurons)
            
            if n_neurons == 0:
                return {}
            
            # Convert to PyTorch tensors on MPS device
            current_activations = torch.zeros(n_neurons, device=self.device)
            connections_matrix = torch.zeros((n_neurons, n_neurons), device=self.device)
            
            # Fill current activations
            for i, neuron_id in enumerate(neurons):
                current_activations[i] = huey_network.activations.get(neuron_id, 0.0)
            
            # Fill connections matrix
            for i, neuron_i in enumerate(neurons):
                for j, neuron_j in enumerate(neurons):
                    if i != j:
                        conn_key = (neuron_i, neuron_j) if neuron_i < neuron_j else (neuron_j, neuron_i)
                        strength = huey_network.connections.get(conn_key, 0.0)
                        connections_matrix[i, j] = strength
            
            # GPU computation: weighted sum + logistic activation
            weighted_sums = torch.matmul(connections_matrix, current_activations)
            new_activations = torch.sigmoid(weighted_sums)  # Logistic function
            
            # Convert back to CPU and create result dict
            new_activations_cpu = new_activations.cpu().numpy()
            result = {neurons[i]: float(new_activations_cpu[i]) for i in range(n_neurons)}
            
            elapsed = time.perf_counter() - start_time
            self.kernel_calls += 1
            self.total_kernel_time += elapsed
            
            return result
            
        except Exception as e:
            print(f"⚠️ MPS calculation failed, falling back to CPU: {e}")
            return self._calculate_activations_cpu(huey_network)
    
    def calculate_eigenvalues_mps(self, association_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues using PyTorch MPS acceleration."""
        
        if not self.use_mps:
            return self._calculate_eigenvalues_cpu(association_matrix)
        
        try:
            # Convert to PyTorch tensor on MPS device
            matrix_tensor = torch.from_numpy(association_matrix).float().to(self.device)
            
            # Eigenvalue decomposition on GPU
            eigenvals, eigenvecs = torch.linalg.eigh(matrix_tensor)
            
            # Sort in descending order
            idx = torch.argsort(eigenvals, descending=True)
            sorted_eigenvals = eigenvals[idx]
            sorted_eigenvecs = eigenvecs[:, idx]
            
            # Convert back to NumPy
            eigenvals_np = sorted_eigenvals.cpu().numpy()
            eigenvecs_np = sorted_eigenvecs.cpu().numpy()
            
            return eigenvals_np, eigenvecs_np
            
        except Exception as e:
            print(f"⚠️ MPS eigenvalue calculation failed, falling back to CPU: {e}")
            return self._calculate_eigenvalues_cpu(association_matrix)
    
    def _calculate_activations_cpu(self, huey_network) -> Dict[int, float]:
        """CPU fallback for activation calculation."""
        neurons = list(huey_network.activations.keys())
        n_neurons = len(neurons)
        
        if n_neurons == 0:
            return {}
        
        # NumPy CPU calculation
        current_activations = np.array([huey_network.activations.get(nid, 0.0) for nid in neurons])
        connections_matrix = np.zeros((n_neurons, n_neurons))
        
        for i, neuron_i in enumerate(neurons):
            for j, neuron_j in enumerate(neurons):
                if i != j:
                    conn_key = (neuron_i, neuron_j) if neuron_i < neuron_j else (neuron_j, neuron_i)
                    connections_matrix[i, j] = huey_network.connections.get(conn_key, 0.0)
        
        weighted_sums = np.dot(connections_matrix, current_activations)
        new_activations = 1.0 / (1.0 + np.exp(-weighted_sums))
        
        return {neurons[i]: float(new_activations[i]) for i in range(n_neurons)}
    
    def _calculate_eigenvalues_cpu(self, association_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for eigenvalue calculation."""
        eigenvals, eigenvecs = np.linalg.eigh(association_matrix)
        idx = np.argsort(eigenvals)[::-1]
        return eigenvals[idx], eigenvecs[:, idx]
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        avg_time = self.total_kernel_time / max(1, self.kernel_calls)
        
        return {
            'kernel_calls': self.kernel_calls,
            'total_kernel_time': self.total_kernel_time,
            'average_kernel_time': avg_time,
            'mps_acceleration_enabled': self.use_mps,
            'device': str(self.device)
        }

# Test function
def test_pytorch_mps():
    """Test PyTorch MPS functionality."""
    print("🧪 Testing PyTorch MPS Interface...")
    
    interface = HueyPyTorchMPSInterface()
    
    # Test matrix operation
    test_matrix = np.random.rand(50, 50).astype(np.float32)
    test_matrix = (test_matrix + test_matrix.T) / 2  # Make symmetric
    
    start_time = time.perf_counter()
    eigenvals, eigenvecs = interface.calculate_eigenvalues_mps(test_matrix)
    elapsed = time.perf_counter() - start_time
    
    print(f"✅ Eigenvalue test completed in {elapsed:.4f}s")
    print(f"   Matrix size: {test_matrix.shape}")
    print(f"   Top 3 eigenvalues: {eigenvals[:3]}")
    print(f"   Device used: {interface.device}")

if __name__ == "__main__":
    test_pytorch_mps()