#!/usr/bin/env python3
"""
Huey||| Multicore Interface
High-performance computational kernels using multiprocessing for true parallelism.

This module provides multiprocessing implementations that bypass Python's GIL
for CPU-intensive Hebbian learning operations across multiple cores.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time

class HueyMulticoreInterface:
    """
    High-performance interface for Huey||| multicore operations.
    
    Uses multiprocessing to achieve true parallelism for Hebbian learning,
    bypassing Python's Global Interpreter Lock (GIL).
    """
    
    def __init__(self, max_neurons: int = 500, use_multiprocessing: bool = True):
        """Initialize the multicore interface."""
        
        self.max_neurons = max_neurons
        self.use_multiprocessing = use_multiprocessing
        self.cpu_cores = mp.cpu_count()
        
        # Performance tracking
        self.kernel_calls = 0
        self.total_kernel_time = 0.0
        
        print(f"🚀 Huey||| Multicore Interface initialized")
        print(f"   Max neurons: {max_neurons}")
        print(f"   CPU cores: {self.cpu_cores}")
        print(f"   Multiprocessing: {'Enabled' if use_multiprocessing else 'Disabled'}")
    
    def hebbian_update_batch(self, window_neurons: List[int], network) -> Dict[str, int]:
        """
        Multicore Hebbian learning update using process pools.
        
        Splits neuron pair processing across multiple CPU cores to achieve
        true parallelism without GIL limitations.
        """
        import time
        start_time = time.perf_counter()
        
        n_window = len(window_neurons)
        
        if n_window < 2:
            return {'updates_applied': 0, 'computation_time': 0.0}
        
        # Calculate total pairs to determine if multiprocessing is worth it
        total_pairs = (n_window * (n_window - 1)) // 2
        
        if total_pairs < 20 or not self.use_multiprocessing:
            # Small workload: use sequential processing
            updates_applied = self._hebbian_sequential(window_neurons, network)
            multiprocessing_used = False
        else:
            # Large workload: use multiprocessing
            updates_applied = self._hebbian_multiprocess(window_neurons, network)
            multiprocessing_used = True
        
        elapsed = time.perf_counter() - start_time
        self.kernel_calls += 1
        self.total_kernel_time += elapsed
        
        return {
            'updates_applied': updates_applied,
            'computation_time': elapsed,
            'window_size': n_window,
            'total_pairs': total_pairs,
            'multiprocessing_used': multiprocessing_used,
            'cpu_cores_available': self.cpu_cores
        }
    
    def _hebbian_sequential(self, window_neurons, network):
        """Sequential processing for small windows."""
        updates_applied = 0
        n_window = len(window_neurons)
        
        # Get activations once
        activations = [network.activations.get(nid, 0.0) for nid in window_neurons]
        
        for pos_i in range(n_window - 1):
            for pos_j in range(pos_i + 1, n_window):
                neuron_i = window_neurons[pos_i]
                neuron_j = window_neurons[pos_j]
                
                if not network._should_create_connection(neuron_i, neuron_j):
                    continue
                
                # Fast updates
                ai, aj = activations[pos_i], activations[pos_j]
                conn_key = (neuron_i, neuron_j)
                
                current_strength = network.connections.get(conn_key, 0.0)
                current_mass = network.inertial_mass.get(conn_key, 0.0)
                
                # Hebbian update
                inertial_resistance = 1.0 / (1.0 + current_mass * 0.1)
                delta_w = network.hebbian_constant * ai * aj * inertial_resistance
                new_strength = current_strength + delta_w
                
                # Mass update (logistic model)
                activity = ai * aj
                new_mass = self._calculate_synaptic_mass(
                    activity, current_mass, network.M_max, 
                    network.A_0, network.k_steepness
                )
                
                network._add_sparse_connection(neuron_i, neuron_j, new_strength, new_mass)
                updates_applied += 1
        
        return updates_applied
    
    def _hebbian_multiprocess(self, window_neurons, network):
        """Multiprocess Hebbian updates for large windows."""
        n_window = len(window_neurons)
        
        # Create all neuron pairs
        pairs = []
        for pos_i in range(n_window - 1):
            for pos_j in range(pos_i + 1, n_window):
                pairs.append((pos_i, pos_j, window_neurons[pos_i], window_neurons[pos_j]))
        
        # Extract network data for worker processes
        activations_data = {nid: network.activations.get(nid, 0.0) for nid in window_neurons}
        connections_data = dict(network.connections)
        masses_data = dict(network.inertial_mass)
        
        # Network parameters
        network_params = {
            'hebbian_constant': network.hebbian_constant,
            'M_max': network.M_max,
            'A_0': network.A_0,
            'k_steepness': network.k_steepness,
            'max_connections_per_neuron': network.max_connections_per_neuron
        }
        
        # Split pairs into chunks for multiprocessing
        chunk_size = max(5, len(pairs) // min(self.cpu_cores, 8))  # Use up to 8 cores
        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        
        # Process chunks across multiple cores
        all_updates = []
        with ProcessPoolExecutor(max_workers=min(self.cpu_cores, len(chunks))) as executor:
            futures = [
                executor.submit(_process_hebbian_chunk, chunk, activations_data, 
                              connections_data, masses_data, network_params)
                for chunk in chunks
            ]
            
            for future in futures:
                chunk_updates = future.result()
                all_updates.extend(chunk_updates)
        
        # Apply all updates to the network
        for neuron_i, neuron_j, new_strength, new_mass in all_updates:
            network._add_sparse_connection(neuron_i, neuron_j, new_strength, new_mass)
        
        return len(all_updates)
    
    def _calculate_synaptic_mass(self, activity: float, current_mass: float,
                                M_max: float, A_0: float, k_steepness: float) -> float:
        """Synaptic mass calculation (same as vectorized version)."""
        if abs(activity - A_0) > 50.0:
            logistic_mass = M_max if activity > A_0 else 0.0
        else:
            logistic_mass = M_max / (1.0 + np.exp(-k_steepness * (activity - A_0)))
        
        homeostatic_correction = -0.1 * max(0.0, activity - 0.8) * current_mass
        ltd_correction = -0.1 * (0.15 - activity) * current_mass if activity < 0.15 else 0.0
        
        mass_difference = logistic_mass - current_mass
        mass_change = 0.15 * mass_difference + homeostatic_correction + ltd_correction
        
        return max(0.0, min(current_mass + mass_change, M_max))

def _process_hebbian_chunk(pairs_chunk, activations_data, connections_data, masses_data, network_params):
    """
    Process a chunk of neuron pairs in a separate process.
    
    This function runs in its own process, bypassing the GIL completely.
    """
    import numpy as np
    
    updates = []
    
    for pos_i, pos_j, neuron_i, neuron_j in pairs_chunk:
        # Simple connectivity check (can't access full network object)
        if len(connections_data) > network_params['max_connections_per_neuron'] * 2:
            # Skip if network is getting too dense
            continue
        
        # Get values
        ai = activations_data.get(neuron_i, 0.0)
        aj = activations_data.get(neuron_j, 0.0)
        
        conn_key = (neuron_i, neuron_j)
        current_strength = connections_data.get(conn_key, 0.0)
        current_mass = masses_data.get(conn_key, 0.0)
        
        # Hebbian update calculation
        inertial_resistance = 1.0 / (1.0 + current_mass * 0.1)
        delta_w = network_params['hebbian_constant'] * ai * aj * inertial_resistance
        new_strength = current_strength + delta_w
        
        # Mass calculation
        activity = ai * aj
        M_max = network_params['M_max']
        A_0 = network_params['A_0']
        k_steepness = network_params['k_steepness']
        
        # Logistic mass model
        if abs(activity - A_0) > 50.0:
            logistic_mass = M_max if activity > A_0 else 0.0
        else:
            logistic_mass = M_max / (1.0 + np.exp(-k_steepness * (activity - A_0)))
        
        homeostatic_correction = -0.1 * max(0.0, activity - 0.8) * current_mass
        ltd_correction = -0.1 * (0.15 - activity) * current_mass if activity < 0.15 else 0.0
        
        mass_difference = logistic_mass - current_mass
        mass_change = 0.15 * mass_difference + homeostatic_correction + ltd_correction
        new_mass = max(0.0, min(current_mass + mass_change, M_max))
        
        updates.append((neuron_i, neuron_j, new_strength, new_mass))
    
    return updates

if __name__ == "__main__":
    # Test the interface
    interface = HueyMulticoreInterface(max_neurons=500)
    print(f"✅ Huey||| Multicore Interface ready with {interface.cpu_cores} cores")