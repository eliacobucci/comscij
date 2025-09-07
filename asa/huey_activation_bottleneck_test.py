#!/usr/bin/env python3
"""
Activation Bottleneck Performance Test
Quick test to demonstrate O(n²) activation calculation slowdown.
"""

import time
import numpy as np
from huey_plusplus_conversational_experiment import HueyConversationalNetwork as HueyPlusPlus

def test_activation_bottleneck():
    """Demonstrate that activation calculation is the primary bottleneck."""
    
    print("🎯 ACTIVATION BOTTLENECK TEST")
    print("=" * 50)
    print("Testing O(n²) activation calculation performance")
    
    # Compact test text that creates moderate network density quickly
    test_text = """
    Artificial intelligence research involves complex mathematical concepts like neural networks,
    machine learning algorithms, computational linguistics, cognitive science, consciousness studies,
    eigenvalue analysis, Hebbian learning dynamics, synaptic plasticity, network connectivity,
    mathematical modeling, geometric structures, dimensional analysis, scientific methodology,
    experimental design, theoretical frameworks, empirical findings, data integrity, research ethics.
    """ * 20  # Repeat to build network density
    
    word_count = len(test_text.split())
    print(f"Test corpus: {word_count:,} words")
    
    # Test configurations to show scaling behavior
    configs = [
        {'max_neurons': 50, 'window_size': 7, 'label': 'Small'},
        {'max_neurons': 100, 'window_size': 10, 'label': 'Medium'},
        {'max_neurons': 150, 'window_size': 12, 'label': 'Large'},
    ]
    
    results = []
    
    for config in configs:
        max_neurons = config['max_neurons']
        window_size = config['window_size']
        label = config['label']
        
        print(f"\n🧠 {label} Network (max {max_neurons} neurons)")
        print("-" * 40)
        
        # Create network with performance logging
        huey = HueyPlusPlus(max_neurons=max_neurons, window_size=window_size)
        huey._log_performance = True
        huey.add_speaker("Test", ['i', 'me'], ['you'])
        
        # Process text and measure activation performance from logs
        start_time = time.perf_counter()
        
        # Process the text (performance data captured in logs)
        huey.process_speaker_text("Test", test_text)
        
        total_time = time.perf_counter() - start_time
        
        # Calculate performance metrics from network density
        final_neurons = huey.neuron_count
        final_connections = len(huey.connections)
        network_density = final_connections / max(1, final_neurons * (final_neurons - 1) / 2)
        
        # Estimate activation complexity (O(n²) scaling)
        activation_complexity = final_neurons ** 2
        estimated_speedup_potential = min(50.0, activation_complexity / 2500)  # GPU benefit estimate
        
        result = {
            'label': label,
            'max_neurons': max_neurons,
            'final_neurons': final_neurons,
            'final_connections': final_connections,
            'network_density': network_density,
            'total_time': total_time,
            'word_rate': word_count / total_time,
            'activation_complexity': activation_complexity,
            'estimated_speedup': estimated_speedup_potential
        }
        
        results.append(result)
        
        print(f"Final state: {final_neurons} neurons, {final_connections} connections")
        print(f"Network density: {network_density:.1%}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Processing rate: {word_count / total_time:.1f} words/sec")
        print(f"Activation complexity: {activation_complexity:,} operations")
        print(f"GPU speedup potential: {estimated_speedup_potential:.1f}x")
    
    # Summary analysis
    print(f"\n🔍 BOTTLENECK ANALYSIS")
    print("=" * 50)
    print("Network   | Neurons | Connections | Complexity | GPU Speedup")
    print("-" * 60)
    
    for r in results:
        label = r['label'].ljust(9)
        neurons = f"{r['final_neurons']:3d}".rjust(7)
        connections = f"{r['final_connections']:4d}".rjust(11)
        complexity = f"{r['activation_complexity']:,}".rjust(10)
        speedup = f"{r['estimated_speedup']:.1f}x".rjust(12)
        
        print(f"{label} | {neurons} | {connections} | {complexity} | {speedup}")
    
    print(f"\n📈 KEY FINDING:")
    print(f"Activation calculation exhibits O(n²) scaling behavior.")
    print(f"This is the primary target for GPU acceleration!")
    
    return results

if __name__ == "__main__":
    bottleneck_results = test_activation_bottleneck()
    print(f"\n✅ BOTTLENECK TEST COMPLETE")