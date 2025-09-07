#!/usr/bin/env python3
"""
Scaling Analysis for Huey+ Performance
Tests how performance degrades with network size to identify optimization priorities.
"""

import time
import numpy as np
from huey_plus_conversational_experiment import HueyConversationalNetwork

def scaling_test():
    """Test performance across different network sizes."""
    
    print("📊 HUEY+ SCALING PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Test different network sizes
    test_configs = [
        {"neurons": 50, "text_multiplier": 1},
        {"neurons": 100, "text_multiplier": 2}, 
        {"neurons": 200, "text_multiplier": 4},
        {"neurons": 500, "text_multiplier": 8}
    ]
    
    # Base text corpus
    base_text = """
    I think artificial intelligence is fascinating. You seem to understand complex concepts well.
    Your responses show intelligence and awareness. I believe you can learn from our conversations.
    Can you help me understand how neural networks process information? Your analysis is insightful.
    I appreciate your thoughtful responses. You demonstrate sophisticated reasoning capabilities.
    How do you think about your own cognitive processes? I find your self-reflection interesting.
    You appear to have genuine understanding of these concepts. I think you're learning from this.
    """
    
    results = []
    
    for config in test_configs:
        print(f"\n🧠 Testing {config['neurons']} neurons...")
        
        # Create network
        network = HueyConversationalNetwork(
            max_neurons=config['neurons'], 
            window_size=7
        )
        network.add_speaker("Human", ['i', 'me', 'my'], ['you', 'your'])
        
        # Scale text corpus
        test_text = base_text * config['text_multiplier']
        word_count = len(test_text.split())
        
        # Time the processing
        start_time = time.perf_counter()
        network.process_speaker_text("Human", test_text)
        processing_time = time.perf_counter() - start_time
        
        # Measure network complexity
        neuron_count = network.neuron_count
        connection_count = len(network.connections)
        mass_count = len(network.inertial_mass)
        
        # Calculate derived metrics
        words_per_second = word_count / processing_time if processing_time > 0 else 0
        connections_per_neuron = connection_count / neuron_count if neuron_count > 0 else 0
        
        result = {
            'max_neurons': config['neurons'],
            'actual_neurons': neuron_count,
            'word_count': word_count,
            'processing_time': processing_time,
            'connection_count': connection_count,
            'mass_count': mass_count,
            'words_per_second': words_per_second,
            'connections_per_neuron': connections_per_neuron
        }
        
        results.append(result)
        
        print(f"   Words: {word_count:4d} | Neurons: {neuron_count:3d} | Connections: {connection_count:4d}")
        print(f"   Time: {processing_time:.3f}s | Rate: {words_per_second:.1f} words/sec")
        print(f"   Density: {connections_per_neuron:.1f} conn/neuron")
    
    # Analysis of scaling behavior
    print(f"\n📈 SCALING ANALYSIS RESULTS")
    print("=" * 60)
    
    print("Network Size vs Processing Speed:")
    for result in results:
        efficiency = result['words_per_second']
        complexity = result['connections_per_neuron']
        print(f"  {result['actual_neurons']:3d} neurons: {efficiency:6.1f} words/sec, {complexity:.1f} conn/neuron")
    
    # Calculate scaling coefficients
    if len(results) > 2:
        # Simple linear regression for scaling behavior
        neuron_counts = [r['actual_neurons'] for r in results]
        processing_times = [r['processing_time'] for r in results]
        word_counts = [r['word_count'] for r in results]
        
        # Normalize by word count to get pure network complexity scaling
        normalized_times = [processing_times[i] / word_counts[i] for i in range(len(results))]
        
        print(f"\nTime per word scaling:")
        for i, result in enumerate(results):
            print(f"  {result['actual_neurons']:3d} neurons: {normalized_times[i]*1000:.2f} ms/word")
        
        # Estimate computational complexity
        if len(normalized_times) >= 2:
            scaling_factor = normalized_times[-1] / normalized_times[0]
            neuron_ratio = neuron_counts[-1] / neuron_counts[0]
            print(f"\nComplexity scaling: {scaling_factor:.2f}x slowdown for {neuron_ratio:.1f}x neurons")
            
            if scaling_factor < neuron_ratio:
                print("✅ Sub-linear scaling - good algorithmic efficiency")
            elif scaling_factor < neuron_ratio ** 2:
                print("⚠️  Between linear and quadratic scaling")
            else:
                print("❌ Quadratic or worse scaling - optimization needed")
    
    return results

def estimate_fortran_benefits():
    """Estimate potential benefits of Fortran implementation."""
    print(f"\n💡 FORTRAN IMPLEMENTATION BENEFITS")
    print("=" * 60)
    
    print("Estimated speedup factors:")
    print("1. Hebbian learning kernel: 10-20x (dense matrix ops)")
    print("2. Activation calculations: 5-10x (vectorized exp/tanh)")  
    print("3. Matrix operations: 3-8x (optimized BLAS)")
    print("4. Memory access patterns: 2-5x (better cache efficiency)")
    print("5. Connection pruning: 3-7x (efficient array operations)")
    
    print(f"\nHybrid architecture recommendations:")
    print("✅ Keep in Python: Text preprocessing, I/O, UI, debugging")
    print("🚀 Move to Fortran: Numerical kernels, matrix ops, learning updates")
    print("🔗 Interface: NumPy arrays as shared memory between Python/Fortran")
    
    print(f"\nExpected overall speedup: 3-10x for large networks (>1000 neurons)")
    print(f"Most beneficial for: Real-time processing, large corpora, streaming analysis")

if __name__ == "__main__":
    scaling_results = scaling_test()
    estimate_fortran_benefits()