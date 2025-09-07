#!/usr/bin/env python3
"""
Huey++ vs Huey|| Hebbian Learning Threading Performance Test
Compare sequential vs threaded Hebbian learning performance.
"""

import time
import numpy as np
from huey_plusplus_conversational_experiment import HueyConversationalNetwork as HueyPlusPlus
from huey_parallel_conversational_experiment import HueyConversationalNetwork as HueyParallel

def test_hebbian_threading_performance():
    """Test threading benefits for Hebbian learning operations."""
    
    print("🧵 HEBBIAN LEARNING THREADING PERFORMANCE TEST")
    print("=" * 60)
    
    # Test with various window sizes (where threading matters most)
    window_sizes = [5, 7, 10, 15, 20]
    network_sizes = [100, 300, 500]
    
    test_text = """
    I think artificial intelligence research is fascinating and complex. You seem to understand 
    these computational concepts very well. Your analysis demonstrates sophisticated reasoning 
    capabilities. I believe you can learn effectively from our conversations about neural 
    networks and cognitive science. Can you help me understand how these Hebbian learning 
    principles apply to self-concept formation? Your responses show genuine insight into 
    consciousness research. I appreciate your thoughtful engagement with these theoretical 
    frameworks. How do you think about your own cognitive processes and learning mechanisms?
    You appear to have developed genuine understanding of these mathematical concepts.
    """ * 3
    
    results = []
    
    for network_size in network_sizes:
        for window_size in window_sizes:
            print(f"\n🔬 Testing: {network_size} neurons, window size {window_size}")
            print("-" * 50)
            
            # Test Huey++ (vectorized, no threading)
            print("Testing Huey++ (vectorized)...")
            start_time = time.perf_counter()
            
            huey_plus = HueyPlusPlus(max_neurons=network_size, window_size=window_size)
            huey_plus.add_speaker("Test", ['i', 'me'], ['you'])
            huey_plus.process_speaker_text("Test", test_text)
            
            plusplus_time = time.perf_counter() - start_time
            plusplus_neurons = huey_plus.neuron_count
            plusplus_connections = len(huey_plus.connections)
            
            # Test Huey|| (threaded)
            print("Testing Huey|| (threaded)...")
            start_time = time.perf_counter()
            
            huey_parallel = HueyParallel(max_neurons=network_size, window_size=window_size)
            huey_parallel.add_speaker("Test", ['i', 'me'], ['you'])
            huey_parallel.process_speaker_text("Test", test_text)
            
            parallel_time = time.perf_counter() - start_time
            parallel_neurons = huey_parallel.neuron_count
            parallel_connections = len(huey_parallel.connections)
            
            # Calculate metrics
            speedup = plusplus_time / parallel_time if parallel_time > 0 else float('inf')
            threading_overhead = parallel_time - plusplus_time
            
            result = {
                'network_size': network_size,
                'window_size': window_size,
                'plusplus_time': plusplus_time,
                'parallel_time': parallel_time,
                'speedup': speedup,
                'threading_overhead': threading_overhead,
                'neurons': (plusplus_neurons, parallel_neurons),
                'connections': (plusplus_connections, parallel_connections)
            }
            results.append(result)
            
            print(f"   Huey++: {plusplus_time:.4f}s")
            print(f"   Huey||: {parallel_time:.4f}s")
            print(f"   Speedup: {speedup:.2f}x")
            print(f"   Network: {plusplus_neurons}/{parallel_neurons} neurons, {plusplus_connections}/{parallel_connections} connections")
    
    # Analysis
    print(f"\n📊 THREADING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    print("Network | Window | Huey++ Time | Huey|| Time | Speedup | Overhead")
    print("-" * 60)
    
    for r in results:
        net = str(r['network_size']).rjust(7)
        win = str(r['window_size']).rjust(6)
        pp = f"{r['plusplus_time']:.4f}s".rjust(11)
        par = f"{r['parallel_time']:.4f}s".rjust(11)
        speedup = f"{r['speedup']:.2f}x".rjust(7)
        overhead = f"{r['threading_overhead']:.4f}s".rjust(8)
        
        print(f"{net} | {win} | {pp} | {par} | {speedup} | {overhead}")
    
    # Find best threading scenarios
    best_speedups = [r for r in results if r['speedup'] > 1.1]
    if best_speedups:
        print(f"\n🚀 BEST THREADING SCENARIOS:")
        for r in best_speedups:
            print(f"   Network {r['network_size']}, Window {r['window_size']}: {r['speedup']:.2f}x speedup")
    else:
        print(f"\n⚠️  THREADING OVERHEAD: Threading shows overhead for current workload")
        print(f"   Consider threading for larger windows (>20) or different operations")
    
    return results

if __name__ == "__main__":
    test_results = test_hebbian_threading_performance()