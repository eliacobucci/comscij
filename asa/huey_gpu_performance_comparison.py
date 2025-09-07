#!/usr/bin/env python3
"""
Huey🚀 GPU vs CPU Performance Comparison
Direct comparison of CPU vs GPU-accelerated activation calculations.
"""

import time
import numpy as np
from huey_plusplus_conversational_experiment import HueyConversationalNetwork
from huey_gpu_conversational_experiment import HueyGPUConversationalNetwork

def performance_comparison_test():
    """Compare CPU vs GPU performance at the activation bottleneck."""
    
    print("🏁 HUEY GPU vs CPU PERFORMANCE COMPARISON")
    print("=" * 60)
    print("Testing revolutionary GPU acceleration vs baseline CPU")
    
    # Test text that creates substantial activation load
    test_text = """
    Revolutionary artificial intelligence research involves complex mathematical concepts like neural networks,
    machine learning algorithms, computational linguistics, cognitive science, consciousness studies,
    eigenvalue analysis, Hebbian learning dynamics, synaptic plasticity, network connectivity,
    mathematical modeling, geometric structures, dimensional analysis, scientific methodology,
    experimental design, theoretical frameworks, empirical findings, data integrity, research ethics,
    performance optimization, GPU acceleration, parallel computing, matrix operations, vectorization,
    linear algebra, computational efficiency, algorithmic optimization, high performance computing.
    """ * 5
    
    word_count = len(test_text.split())
    print(f"Test corpus: {word_count:,} words")
    
    configurations = [
        {'max_neurons': 100, 'window_size': 8, 'label': 'Medium Network'},
        {'max_neurons': 150, 'window_size': 10, 'label': 'Large Network'},
        {'max_neurons': 200, 'window_size': 12, 'label': 'Very Large Network'},
    ]
    
    for config in configurations:
        max_neurons = config['max_neurons']
        window_size = config['window_size']
        label = config['label']
        
        print(f"\n🧠 {label} ({max_neurons} max neurons, window {window_size})")
        print("-" * 50)
        
        # Test CPU baseline
        print("   🐌 CPU Baseline:")
        cpu_start = time.perf_counter()
        
        huey_cpu = HueyConversationalNetwork(max_neurons=max_neurons, window_size=window_size)
        huey_cpu.add_speaker("Test", ['i', 'me'], ['you'])
        huey_cpu._log_performance = False  # Reduce noise
        
        huey_cpu.process_speaker_text("Test", test_text)
        
        cpu_time = time.perf_counter() - cpu_start
        cpu_neurons = huey_cpu.neuron_count
        cpu_connections = len(huey_cpu.connections)
        cpu_rate = word_count / cpu_time
        
        print(f"      Time: {cpu_time:.3f}s")
        print(f"      Rate: {cpu_rate:.1f} words/sec")
        print(f"      Final: {cpu_neurons} neurons, {cpu_connections} connections")
        
        # Test GPU acceleration
        print("   🚀 GPU Acceleration:")
        gpu_start = time.perf_counter()
        
        huey_gpu = HueyGPUConversationalNetwork(max_neurons=max_neurons, window_size=window_size, use_gpu_acceleration=True)
        huey_gpu.add_speaker("Test", ['i', 'me'], ['you'])
        huey_gpu._log_performance = False  # Reduce noise
        
        huey_gpu.process_speaker_text("Test", test_text)
        
        gpu_time = time.perf_counter() - gpu_start
        gpu_neurons = huey_gpu.neuron_count
        gpu_connections = len(huey_gpu.connections)
        gpu_rate = word_count / gpu_time
        
        print(f"      Time: {gpu_time:.3f}s")
        print(f"      Rate: {gpu_rate:.1f} words/sec")
        print(f"      Final: {gpu_neurons} neurons, {gpu_connections} connections")
        
        # Calculate performance gains
        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        rate_improvement = gpu_rate / cpu_rate if cpu_rate > 0 else float('inf')
        
        print(f"\n   📈 PERFORMANCE GAINS:")
        print(f"      Speedup: {speedup:.2f}x faster")
        print(f"      Rate improvement: {rate_improvement:.2f}x words/sec")
        
        if speedup > 2.0:
            print(f"      ✅ Revolutionary speedup achieved!")
        elif speedup > 1.5:
            print(f"      🟡 Significant speedup achieved")
        elif speedup > 1.1:
            print(f"      🟢 Moderate speedup achieved")
        else:
            print(f"      ❌ No significant speedup")
        
        # Verify network consistency
        if abs(cpu_neurons - gpu_neurons) <= 2 and abs(cpu_connections - gpu_connections) <= 10:
            print(f"      ✅ Network structure consistency maintained")
        else:
            print(f"      ⚠️ Network structure differs (expected with parallel processing)")

if __name__ == "__main__":
    performance_comparison_test()
    print(f"\n🚀 PERFORMANCE COMPARISON COMPLETE")
    print(f"Huey🚀 GPU acceleration targets the O(n²) activation bottleneck!")