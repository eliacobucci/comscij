#!/usr/bin/env python3
"""
Multicore Performance Test for Huey|||
Test true multiprocessing performance gains vs threading and vectorization.
"""

import time
import multiprocessing as mp
import numpy as np

def test_multicore_hebbian_performance():
    """Test multiprocessing benefits for Hebbian learning."""
    
    print("🔥 HUEY||| MULTICORE PERFORMANCE TEST")
    print("=" * 50)
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # Test scenarios that should benefit from multiprocessing
    test_scenarios = [
        {'window_size': 10, 'description': 'Medium window (45 pairs)'},
        {'window_size': 15, 'description': 'Large window (105 pairs)'},
        {'window_size': 20, 'description': 'Very large window (190 pairs)'},
    ]
    
    for scenario in test_scenarios:
        window_size = scenario['window_size']
        description = scenario['description']
        total_pairs = (window_size * (window_size - 1)) // 2
        
        print(f"\n🧪 Testing {description}")
        print(f"   Window size: {window_size} neurons")
        print(f"   Neuron pairs: {total_pairs}")
        print("-" * 40)
        
        # Create test data
        window_neurons = list(range(window_size))
        activations_data = {i: np.random.random() for i in window_neurons}
        connections_data = {}
        masses_data = {}
        
        # Mock network parameters
        network_params = {
            'hebbian_constant': 0.1,
            'M_max': 10.0,
            'A_0': 0.5,
            'k_steepness': 2.0,
            'max_connections_per_neuron': 250
        }
        
        # Test sequential processing
        start_time = time.perf_counter()
        sequential_updates = test_sequential_hebbian(window_neurons, activations_data, 
                                                   connections_data, masses_data, network_params)
        sequential_time = time.perf_counter() - start_time
        
        # Test multiprocess processing
        start_time = time.perf_counter()
        multicore_updates = test_multicore_hebbian(window_neurons, activations_data,
                                                 connections_data, masses_data, network_params)
        multicore_time = time.perf_counter() - start_time
        
        # Calculate performance metrics
        speedup = sequential_time / multicore_time if multicore_time > 0 else float('inf')
        efficiency = speedup / mp.cpu_count()  # How well we use available cores
        
        print(f"   Sequential: {sequential_time:.4f}s ({sequential_updates} updates)")
        print(f"   Multicore:  {multicore_time:.4f}s ({multicore_updates} updates)")
        print(f"   Speedup:    {speedup:.2f}x")
        print(f"   Efficiency: {efficiency:.1%} (of {mp.cpu_count()} cores)")
        
        if speedup > 1.5:
            print(f"   ✅ Significant speedup achieved!")
        elif speedup > 1.1:
            print(f"   🟡 Moderate speedup achieved")
        else:
            print(f"   ❌ No significant speedup (overhead dominates)")

def test_sequential_hebbian(window_neurons, activations_data, connections_data, masses_data, network_params):
    """Sequential Hebbian processing (baseline)."""
    updates_applied = 0
    n_window = len(window_neurons)
    
    for pos_i in range(n_window - 1):
        for pos_j in range(pos_i + 1, n_window):
            neuron_i = window_neurons[pos_i]
            neuron_j = window_neurons[pos_j]
            
            ai = activations_data[neuron_i]
            aj = activations_data[neuron_j]
            
            conn_key = (neuron_i, neuron_j)
            current_strength = connections_data.get(conn_key, 0.0)
            current_mass = masses_data.get(conn_key, 0.0)
            
            # Hebbian calculation
            inertial_resistance = 1.0 / (1.0 + current_mass * 0.1)
            delta_w = network_params['hebbian_constant'] * ai * aj * inertial_resistance
            new_strength = current_strength + delta_w
            
            # Mass calculation
            activity = ai * aj
            M_max = network_params['M_max']
            A_0 = network_params['A_0']
            k_steepness = network_params['k_steepness']
            
            if abs(activity - A_0) > 50.0:
                logistic_mass = M_max if activity > A_0 else 0.0
            else:
                logistic_mass = M_max / (1.0 + np.exp(-k_steepness * (activity - A_0)))
            
            homeostatic_correction = -0.1 * max(0.0, activity - 0.8) * current_mass
            ltd_correction = -0.1 * (0.15 - activity) * current_mass if activity < 0.15 else 0.0
            
            mass_difference = logistic_mass - current_mass
            mass_change = 0.15 * mass_difference + homeostatic_correction + ltd_correction
            new_mass = max(0.0, min(current_mass + mass_change, M_max))
            
            # Store updates (in real implementation, would update network)
            connections_data[conn_key] = new_strength
            masses_data[conn_key] = new_mass
            updates_applied += 1
    
    return updates_applied

def test_multicore_hebbian(window_neurons, activations_data, connections_data, masses_data, network_params):
    """Multicore Hebbian processing."""
    from concurrent.futures import ProcessPoolExecutor
    
    n_window = len(window_neurons)
    
    # Create all pairs
    pairs = []
    for pos_i in range(n_window - 1):
        for pos_j in range(pos_i + 1, n_window):
            pairs.append((pos_i, pos_j, window_neurons[pos_i], window_neurons[pos_j]))
    
    # Split into chunks for multiprocessing
    cpu_count = mp.cpu_count()
    chunk_size = max(5, len(pairs) // min(cpu_count, 8))
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    
    # Import the worker function from the multicore interface
    from huey_multicore_interface import _process_hebbian_chunk
    
    # Process in parallel
    all_updates = []
    with ProcessPoolExecutor(max_workers=min(cpu_count, len(chunks))) as executor:
        futures = [
            executor.submit(_process_hebbian_chunk, chunk, activations_data,
                          connections_data, masses_data, network_params)
            for chunk in chunks
        ]
        
        for future in futures:
            chunk_updates = future.result()
            all_updates.extend(chunk_updates)
    
    return len(all_updates)

if __name__ == "__main__":
    test_multicore_hebbian_performance()