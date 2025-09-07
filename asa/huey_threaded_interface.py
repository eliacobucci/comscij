#!/usr/bin/env python3
"""
Huey|| Threaded Interface
High-performance computational kernels for Hebbian learning operations.

This module provides threaded implementations for time-critical decay operations
while maintaining vectorized NumPy for Hebbian learning updates.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

class HueyThreadedInterface:
    """
    High-performance interface for Huey++ computational kernels.
    
    Provides vectorized implementations of core Hebbian learning operations
    designed to match Fortran performance characteristics.
    """
    
    def __init__(self, max_neurons: int = 500, use_fortran: bool = False):
        """Initialize the Fortran interface with workspace arrays."""
        
        self.max_neurons = max_neurons
        self.use_fortran = use_fortran
        
        # Pre-allocate workspace arrays for efficiency
        self.connections_workspace = np.zeros((max_neurons * 10, 3), dtype=np.float64)  # (i, j, strength)
        self.masses_workspace = np.zeros((max_neurons * 10, 3), dtype=np.float64)      # (i, j, mass)
        self.activations_workspace = np.zeros(max_neurons, dtype=np.float64)
        self.window_workspace = np.zeros(20, dtype=np.int32)  # Max window size
        
        # Performance tracking
        self.kernel_calls = 0
        self.total_kernel_time = 0.0
        
        print(f"🚀 Huey|| Threaded Interface initialized")
        print(f"   Max neurons: {max_neurons}")
        print(f"   Threading: {'Enabled' if use_fortran else 'Using vectorized NumPy with threading'}")
        
        if not use_fortran:
            print("   📝 Note: Threading enabled for decay operations, vectorized NumPy for Hebbian updates")
    
    def hebbian_update_batch(self, window_neurons: List[int], network) -> Dict[str, int]:
        """
        High-performance Hebbian learning update for a window of neurons.
        
        This is the primary computational kernel - processes all neuron pairs in a window
        and updates both connection strengths and inertial masses using vectorized operations.
        
        Args:
            window_neurons: List of neuron indices in current window
            network: Reference to the Huey network object
            
        Returns:
            Dict with performance statistics
        """
        import time
        start_time = time.perf_counter()
        
        n_window = len(window_neurons)
        updates_applied = 0
        
        if n_window < 2:
            return {'updates_applied': 0, 'computation_time': 0.0}
        
        # Convert to NumPy arrays for vectorized operations
        window_array = np.array(window_neurons, dtype=np.int32)
        
        # Get activations for window neurons
        activations = np.array([network.activations.get(nid, 0.0) for nid in window_neurons])
        
        # Threaded computation for all valid pairs (i < j for temporal causality)
        if n_window <= 5:
            # Small windows: threading overhead not worth it
            updates_applied = self._hebbian_update_sequential(window_neurons, activations, network)
            threading_used = False
        else:
            # Large windows: use threading for pair-wise updates
            updates_applied = self._hebbian_update_threaded(window_neurons, activations, network)
            threading_used = True
        
        elapsed = time.perf_counter() - start_time
        self.kernel_calls += 1
        self.total_kernel_time += elapsed
        
        return {
            'updates_applied': updates_applied,
            'computation_time': elapsed,
            'window_size': n_window,
            'threading_used': threading_used
        }
    
    def _calculate_synaptic_mass_vectorized(self, activity: float, current_mass: float,
                                          M_max: float, A_0: float, k_steepness: float) -> float:
        """
        Vectorized synaptic mass calculation using biologically accurate logistic model.
        
        This function matches the Fortran implementation exactly and can be easily
        replaced with the compiled Fortran kernel.
        """
        
        # Core logistic function for Long Term Potentiation (LTP)
        # Handle extreme values to prevent overflow
        if abs(activity - A_0) > 50.0:
            logistic_mass = M_max if activity > A_0 else 0.0
        else:
            logistic_mass = M_max / (1.0 + np.exp(-k_steepness * (activity - A_0)))
        
        # Homeostatic scaling correction
        homeostatic_correction = -0.1 * max(0.0, activity - 0.8) * current_mass
        
        # Long Term Depression (LTD) correction
        ltd_correction = -0.1 * (0.15 - activity) * current_mass if activity < 0.15 else 0.0
        
        # Calculate mass change toward logistic target
        mass_difference = logistic_mass - current_mass
        mass_change = 0.15 * mass_difference + homeostatic_correction + ltd_correction
        
        # Bounded update
        new_mass = max(0.0, min(current_mass + mass_change, M_max))
        
        return new_mass
    
    def calculate_activations_batch(self, window_neurons: List[int], network) -> Dict[str, float]:
        """
        High-performance activation calculation for all neurons.
        
        Vectorized computation of new activation values using logistic function
        and weighted sums from connections.
        """
        import time
        start_time = time.perf_counter()
        
        n_neurons = network.neuron_count
        window_set = set(window_neurons)
        
        # Pre-allocate arrays
        new_activations = np.zeros(n_neurons)
        
        # Vectorized processing for all neurons
        for neuron_idx in range(n_neurons):
            if neuron_idx in window_set:
                # Window neurons get direct input
                new_activations[neuron_idx] = 1.0
            else:
                # Non-window neurons: calculate weighted sum
                weighted_sum = network.bias
                
                # Vectorized sum over all incoming connections
                for other_idx in range(n_neurons):
                    if other_idx != neuron_idx:
                        conn_key = (other_idx, neuron_idx)
                        strength = network.connections.get(conn_key, 0.0)
                        other_activation = network.activations.get(other_idx, 0.0)
                        weighted_sum += strength * other_activation
                
                # Vectorized logistic activation
                new_activations[neuron_idx] = self._logistic_activation(weighted_sum)
        
        # Update network state
        for neuron_idx in range(n_neurons):
            if neuron_idx in network.neuron_to_word:
                network.activations[neuron_idx] = new_activations[neuron_idx]
        
        elapsed = time.perf_counter() - start_time
        
        return {
            'neurons_updated': n_neurons,
            'window_neurons': len(window_neurons),
            'computation_time': elapsed
        }
    
    def apply_activation_decay_batch(self, network) -> Dict[str, float]:
        """
        High-performance activation decay using threaded operations.
        """
        import time
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        start_time = time.perf_counter()
        
        n_neurons = network.neuron_count
        decay_rate = network.activation_decay_rate
        min_activation = network.minimum_activation
        
        # Get active neuron indices for threading
        active_neurons = [idx for idx in range(n_neurons) if idx in network.neuron_to_word]
        
        if len(active_neurons) < 100:
            # For small networks, threading overhead isn't worth it
            for neuron_idx in active_neurons:
                current = network.activations.get(neuron_idx, 0.0)
                decayed = current * (1.0 - decay_rate)
                network.activations[neuron_idx] = max(decayed, min_activation)
            threading_used = False
        else:
            # Use threading for larger networks
            def decay_chunk(neuron_indices, decay_rate, min_activation):
                """Process a chunk of neurons."""
                chunk_results = {}
                for neuron_idx in neuron_indices:
                    current = network.activations.get(neuron_idx, 0.0)
                    decayed = current * (1.0 - decay_rate)
                    chunk_results[neuron_idx] = max(decayed, min_activation)
                return chunk_results
            
            # Split neurons into chunks for parallel processing
            chunk_size = max(50, len(active_neurons) // 4)  # 4 threads optimal for most CPUs
            chunks = [active_neurons[i:i + chunk_size] for i in range(0, len(active_neurons), chunk_size)]
            
            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_chunk = {executor.submit(decay_chunk, chunk, decay_rate, min_activation): chunk for chunk in chunks}
                
                # Collect results and update network
                for future in as_completed(future_to_chunk):
                    chunk_results = future.result()
                    network.activations.update(chunk_results)
            
            threading_used = True
        
        elapsed = time.perf_counter() - start_time
        
        return {
            'neurons_processed': len(active_neurons),
            'computation_time': elapsed,
            'threading_used': threading_used,
            'chunks_processed': len(chunks) if threading_used else 1
        }
    
    def _logistic_activation(self, weighted_sum: float) -> float:
        """Vectorized logistic activation function with overflow protection."""
        if weighted_sum > 500.0:
            return 1.0  # Handle overflow
        elif weighted_sum < -500.0:
            return 0.0  # Handle underflow
        else:
            return 1.0 / (1.0 + np.exp(-weighted_sum))
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the interface."""
        avg_time = self.total_kernel_time / max(self.kernel_calls, 1)
        
        return {
            'kernel_calls': self.kernel_calls,
            'total_time': self.total_kernel_time,
            'average_time_per_call': avg_time,
            'using_fortran': self.use_fortran
        }
    
    def _hebbian_update_sequential(self, window_neurons, activations, network):
        """Sequential Hebbian update for small windows."""
        updates_applied = 0
        n_window = len(window_neurons)
        
        for pos_i in range(n_window - 1):
            for pos_j in range(pos_i + 1, n_window):
                neuron_i = window_neurons[pos_i]
                neuron_j = window_neurons[pos_j]
                
                if neuron_i not in network.activations or neuron_j not in network.activations:
                    continue
                
                if not network._should_create_connection(neuron_i, neuron_j):
                    continue
                
                ai = activations[pos_i]
                aj = activations[pos_j]
                conn_key = (neuron_i, neuron_j)
                
                current_strength = network.connections.get(conn_key, 0.0)
                current_mass = network.inertial_mass.get(conn_key, 0.0)
                
                inertial_resistance = 1.0 / (1.0 + current_mass * 0.1)
                delta_w = network.hebbian_constant * ai * aj * inertial_resistance
                new_strength = current_strength + delta_w
                
                activity = ai * aj
                new_mass = self._calculate_synaptic_mass_vectorized(
                    activity, current_mass, network.M_max, 
                    network.A_0, network.k_steepness
                )
                
                network._add_sparse_connection(neuron_i, neuron_j, new_strength, new_mass)
                updates_applied += 1
        
        return updates_applied
    
    def _hebbian_update_threaded(self, window_neurons, activations, network):
        """Threaded Hebbian update for large windows."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        
        n_window = len(window_neurons)
        
        # Create all neuron pairs for threading
        pairs = []
        for pos_i in range(n_window - 1):
            for pos_j in range(pos_i + 1, n_window):
                pairs.append((pos_i, pos_j))
        
        # Thread-safe lock for network updates
        update_lock = threading.Lock()
        total_updates = 0
        
        def process_pair_batch(pair_batch):
            """Process a batch of neuron pairs."""
            batch_updates = []
            batch_count = 0
            
            for pos_i, pos_j in pair_batch:
                neuron_i = window_neurons[pos_i]
                neuron_j = window_neurons[pos_j]
                
                if neuron_i not in network.activations or neuron_j not in network.activations:
                    continue
                
                if not network._should_create_connection(neuron_i, neuron_j):
                    continue
                
                ai = activations[pos_i]
                aj = activations[pos_j]
                conn_key = (neuron_i, neuron_j)
                
                current_strength = network.connections.get(conn_key, 0.0)
                current_mass = network.inertial_mass.get(conn_key, 0.0)
                
                inertial_resistance = 1.0 / (1.0 + current_mass * 0.1)
                delta_w = network.hebbian_constant * ai * aj * inertial_resistance
                new_strength = current_strength + delta_w
                
                activity = ai * aj
                new_mass = self._calculate_synaptic_mass_vectorized(
                    activity, current_mass, network.M_max, 
                    network.A_0, network.k_steepness
                )
                
                batch_updates.append((neuron_i, neuron_j, new_strength, new_mass))
                batch_count += 1
            
            # Apply all updates for this batch atomically
            with update_lock:
                for neuron_i, neuron_j, new_strength, new_mass in batch_updates:
                    network._add_sparse_connection(neuron_i, neuron_j, new_strength, new_mass)
            
            return batch_count
        
        # Split pairs into batches for threading
        batch_size = max(5, len(pairs) // 4)  # 4 threads
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_pair_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                total_updates += future.result()
        
        return total_updates

def try_compile_fortran_kernels() -> bool:
    """
    Attempt to compile the Fortran kernels using f2py.
    
    Returns:
        bool: True if compilation successful, False otherwise
    """
    import subprocess
    import os
    
    try:
        # Check if f2py is available
        result = subprocess.run(['f2py', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            print("f2py not available")
            return False
        
        print(f"Found f2py: {result.stdout.strip()}")
        
        # Try to compile the Fortran module
        compile_cmd = [
            'f2py', '-c', '-m', 'huey_fortran_kernels', 
            'huey_fortran_kernels.f90',
            '--quiet'
        ]
        
        print("🔨 Attempting to compile Fortran kernels...")
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Fortran kernels compiled successfully!")
            return True
        else:
            print(f"❌ Fortran compilation failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"❌ Could not compile Fortran kernels: {e}")
        return False

if __name__ == "__main__":
    # Test the interface
    interface = HueyFortranInterface(max_neurons=100)
    
    # Try to compile Fortran kernels
    fortran_available = try_compile_fortran_kernels()
    
    if fortran_available:
        print("🚀 Ready to use Fortran-accelerated kernels!")
    else:
        print("📊 Using optimized NumPy implementation")
    
    print("\n📋 Interface ready for integration into Huey++ network")