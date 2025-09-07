#!/usr/bin/env python3
"""
Huey+ vs Huey++ Performance Comparison
Compare original Python implementation with vectorized Fortran-style kernels.
"""

import time
import numpy as np
from huey_plus_conversational_experiment import HueyConversationalNetwork as HueyPlus
from huey_plusplus_conversational_experiment import HueyConversationalNetwork as HueyPlusPlus

def run_performance_comparison():
    """Compare Huey+ vs Huey++ performance across different scenarios."""
    
    print("🏁 HUEY+ vs HUEY++ PERFORMANCE COMPARISON")
    print("=" * 70)
    
    # Test scenarios with increasing complexity
    test_scenarios = [
        {
            "name": "Small Network", 
            "max_neurons": 50,
            "text_multiplier": 1,
            "description": "Basic functionality test"
        },
        {
            "name": "Medium Network",
            "max_neurons": 150, 
            "text_multiplier": 3,
            "description": "Typical research scenario"
        },
        {
            "name": "Large Network",
            "max_neurons": 300,
            "text_multiplier": 6,
            "description": "High-volume text analysis"
        }
    ]
    
    # Base test corpus
    base_text = """
    I think artificial intelligence research is fascinating and complex. You seem to understand 
    these computational concepts very well. Your analysis demonstrates sophisticated reasoning 
    capabilities. I believe you can learn effectively from our conversations about neural 
    networks and cognitive science. Can you help me understand how these Hebbian learning 
    principles apply to self-concept formation? Your responses show genuine insight into 
    consciousness research. I appreciate your thoughtful engagement with these theoretical 
    frameworks. How do you think about your own cognitive processes and learning mechanisms?
    You appear to have developed genuine understanding of these mathematical concepts.
    """
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n🧪 {scenario['name']} Test: {scenario['description']}")
        print("-" * 50)
        
        # Prepare test data
        test_text = base_text * scenario['text_multiplier']
        word_count = len(test_text.split())
        max_neurons = scenario['max_neurons']
        
        print(f"Text size: {word_count} words, Max neurons: {max_neurons}")
        
        # Test Huey+ (original)
        print("Testing Huey+ (original Python)...")
        start_time = time.perf_counter()
        
        huey_plus = HueyPlus(max_neurons=max_neurons, window_size=7)
        huey_plus.add_speaker("Human", ['i', 'me', 'my'], ['you', 'your'])
        huey_plus.process_speaker_text("Human", test_text)
        
        huey_plus_time = time.perf_counter() - start_time
        huey_plus_neurons = huey_plus.neuron_count
        huey_plus_connections = len(huey_plus.connections)
        
        # Test Huey++ (with vectorized kernels)
        print("Testing Huey++ (vectorized kernels)...")
        start_time = time.perf_counter()
        
        huey_plusplus = HueyPlusPlus(max_neurons=max_neurons, window_size=7, use_fortran_acceleration=True)
        huey_plusplus.add_speaker("Human", ['i', 'me', 'my'], ['you', 'your'])
        huey_plusplus.process_speaker_text("Human", test_text)
        
        huey_plusplus_time = time.perf_counter() - start_time
        huey_plusplus_neurons = huey_plusplus.neuron_count
        huey_plusplus_connections = len(huey_plusplus.connections)
        
        # Calculate performance metrics
        speedup = huey_plus_time / huey_plusplus_time if huey_plusplus_time > 0 else float('inf')
        plus_rate = word_count / huey_plus_time if huey_plus_time > 0 else 0
        plusplus_rate = word_count / huey_plusplus_time if huey_plusplus_time > 0 else 0
        
        # Get Fortran interface stats
        fortran_stats = huey_plusplus.get_performance_summary()
        
        result = {
            'scenario': scenario['name'],
            'word_count': word_count,
            'max_neurons': max_neurons,
            'huey_plus_time': huey_plus_time,
            'huey_plusplus_time': huey_plusplus_time,
            'speedup': speedup,
            'huey_plus_rate': plus_rate,
            'huey_plusplus_rate': plusplus_rate,
            'neurons_created': (huey_plus_neurons, huey_plusplus_neurons),
            'connections_created': (huey_plus_connections, huey_plusplus_connections),
            'fortran_stats': fortran_stats
        }
        
        results.append(result)
        
        # Print immediate results
        print(f"📊 Results:")
        print(f"   Huey+:  {huey_plus_time:.3f}s ({plus_rate:.1f} words/sec)")
        print(f"   Huey++: {huey_plusplus_time:.3f}s ({plusplus_rate:.1f} words/sec)")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Network: {huey_plus_neurons}/{huey_plusplus_neurons} neurons, {huey_plus_connections}/{huey_plusplus_connections} connections")
    
    # Summary analysis
    print(f"\n📋 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 70)
    
    print("Scenario          | Huey+ Time | Huey++ Time | Speedup | Rate Improvement")
    print("-" * 70)
    
    for result in results:
        name = result['scenario'][:15].ljust(15)
        plus_time = f"{result['huey_plus_time']:.3f}s".rjust(10)
        plusplus_time = f"{result['huey_plusplus_time']:.3f}s".rjust(11)
        speedup = f"{result['speedup']:.2f}x".rjust(7)
        rate_improvement = f"{result['huey_plusplus_rate'] / result['huey_plus_rate']:.2f}x".rjust(15)
        
        print(f"{name} | {plus_time} | {plusplus_time} | {speedup} | {rate_improvement}")
    
    # Analyze trends
    if len(results) > 1:
        print(f"\n📈 Scaling Trends:")
        
        avg_speedup = np.mean([r['speedup'] for r in results])
        max_speedup = max([r['speedup'] for r in results])
        min_speedup = min([r['speedup'] for r in results])
        
        print(f"   Average speedup: {avg_speedup:.2f}x")
        print(f"   Best speedup: {max_speedup:.2f}x")
        print(f"   Worst speedup: {min_speedup:.2f}x")
        
        # Analyze if speedup improves with scale
        large_scenario = results[-1]
        small_scenario = results[0]
        
        if large_scenario['speedup'] > small_scenario['speedup']:
            print(f"   ✅ Speedup improves with scale (vectorization benefits)")
        else:
            print(f"   ⚠️  Speedup decreases with scale (overhead dominates)")
    
    print(f"\n🚀 Huey++ Implementation Status:")
    sample_stats = results[0]['fortran_stats']
    print(f"   Fortran kernels: {'Active' if sample_stats.get('using_fortran_kernels') else 'Vectorized NumPy'}")
    print(f"   Kernel efficiency: {sample_stats.get('fortran_avg_time_per_call', 0)*1000:.2f}ms per call")
    print(f"   Ready for Fortran compilation: ✅")
    
    return results

if __name__ == "__main__":
    comparison_results = run_performance_comparison()