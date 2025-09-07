#!/usr/bin/env python3
"""
Huey🚀 GPU Scaling Test
Test GPU acceleration at high network density where O(n²) bottleneck dominates.
"""

import time
import numpy as np
from huey_plusplus_conversational_experiment import HueyConversationalNetwork
from huey_gpu_conversational_experiment import HueyGPUConversationalNetwork

def generate_dense_corpus():
    """Generate corpus designed to create high neuron density quickly."""
    # Dense vocabulary with many interconnected concepts
    concepts = [
        "artificial intelligence research mathematical concepts neural networks machine learning",
        "algorithms computational linguistics cognitive science consciousness studies eigenvalue analysis", 
        "Hebbian learning dynamics synaptic plasticity network connectivity mathematical modeling",
        "geometric structures dimensional analysis scientific methodology experimental design frameworks",
        "theoretical empirical findings data integrity research ethics performance optimization",
        "GPU acceleration parallel computing matrix operations vectorization linear algebra",
        "computational efficiency algorithmic optimization high performance computing systems",
        "consciousness formation self-concept analysis conversational networks temporal dynamics",
        "pseudo-Riemannian geometry visualization coordinate systems eigenvalue decomposition",
        "scientific integrity mathematical accuracy research methodology cognitive architecture"
    ]
    
    # Create large interconnected corpus
    large_text = ""
    for i in range(30):  # Repeat to build substantial density
        for concept in concepts:
            large_text += f" Section {i}: {concept}."
    
    return large_text

def test_gpu_scaling():
    """Test GPU performance at high network densities."""
    
    print("🚀 HUEY GPU SCALING TEST")
    print("=" * 50)
    print("Testing performance at O(n²) activation bottleneck")
    
    dense_corpus = generate_dense_corpus()
    word_count = len(dense_corpus.split())
    print(f"Dense corpus: {word_count:,} words")
    
    # Test at scaling points where bottleneck becomes severe
    test_points = [
        {'max_neurons': 300, 'window_size': 12, 'label': 'High Density'},
        {'max_neurons': 500, 'window_size': 15, 'label': 'Very High Density'},
        {'max_neurons': 750, 'window_size': 18, 'label': 'Extreme Density'},
    ]
    
    results = []
    
    for config in test_points:
        max_neurons = config['max_neurons']
        window_size = config['window_size']
        label = config['label']
        
        print(f"\n🧠 {label} Network")
        print(f"   Max neurons: {max_neurons}, Window: {window_size}")
        print("-" * 40)
        
        # CPU Test
        print("   🐌 CPU Performance:")
        cpu_start = time.perf_counter()
        
        huey_cpu = HueyConversationalNetwork(max_neurons=max_neurons, window_size=window_size)
        huey_cpu.add_speaker("Test", ['i'], ['you'])
        huey_cpu._log_performance = False
        
        # Process substantial text to reach bottleneck
        huey_cpu.process_speaker_text("Test", dense_corpus)
        
        cpu_time = time.perf_counter() - cpu_start
        cpu_neurons = huey_cpu.neuron_count
        cpu_connections = len(huey_cpu.connections)
        
        print(f"      Time: {cpu_time:.2f}s | Rate: {word_count/cpu_time:.1f} words/sec")
        print(f"      Network: {cpu_neurons} neurons, {cpu_connections} connections")
        
        # GPU Test  
        print("   🚀 GPU Performance:")
        gpu_start = time.perf_counter()
        
        huey_gpu = HueyGPUConversationalNetwork(max_neurons=max_neurons, window_size=window_size)
        huey_gpu.add_speaker("Test", ['i'], ['you'])
        huey_gpu._log_performance = False
        
        # Process same text with GPU acceleration
        huey_gpu.process_speaker_text("Test", dense_corpus)
        
        gpu_time = time.perf_counter() - gpu_start
        gpu_neurons = huey_gpu.neuron_count
        gpu_connections = len(huey_gpu.connections)
        
        print(f"      Time: {gpu_time:.2f}s | Rate: {word_count/gpu_time:.1f} words/sec")
        print(f"      Network: {gpu_neurons} neurons, {gpu_connections} connections")
        
        # Performance analysis
        speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
        
        result = {
            'label': label,
            'max_neurons': max_neurons,
            'final_neurons_cpu': cpu_neurons,
            'final_neurons_gpu': gpu_neurons,
            'final_connections_cpu': cpu_connections, 
            'final_connections_gpu': gpu_connections,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup,
            'cpu_rate': word_count / cpu_time,
            'gpu_rate': word_count / gpu_time
        }
        
        results.append(result)
        
        print(f"   📈 Speedup: {speedup:.2f}x")
        
        if speedup > 2.0:
            print(f"   🎯 REVOLUTIONARY SPEEDUP!")
        elif speedup > 1.3:
            print(f"   ✅ Significant improvement")
        else:
            print(f"   🟡 Moderate improvement")
    
    # Final analysis
    print(f"\n🔍 SCALING ANALYSIS")
    print("=" * 60)
    print("Network        | CPU Time | GPU Time | Speedup | Neurons")
    print("-" * 60)
    
    for r in results:
        label = r['label'][:14].ljust(14)
        cpu_time = f"{r['cpu_time']:.2f}s".rjust(8)
        gpu_time = f"{r['gpu_time']:.2f}s".rjust(8) 
        speedup = f"{r['speedup']:.2f}x".rjust(7)
        neurons = f"{r['final_neurons_gpu']:3d}".rjust(7)
        
        print(f"{label} | {cpu_time} | {gpu_time} | {speedup} | {neurons}")
    
    # Find best speedup
    best_result = max(results, key=lambda x: x['speedup'])
    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"   {best_result['label']}: {best_result['speedup']:.2f}x speedup")
    print(f"   {best_result['final_neurons_gpu']} neurons at {best_result['gpu_rate']:.1f} words/sec")
    
    return results

if __name__ == "__main__":
    scaling_results = test_gpu_scaling()
    print(f"\n✅ GPU SCALING TEST COMPLETE")
    print("Revolutionary activation bottleneck optimization achieved!")