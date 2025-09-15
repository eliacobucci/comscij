#!/usr/bin/env python3
"""
Quick comparison between temporal and windowed learning on small text.
"""

import time
import numpy as np
from huey_temporal_simple import HueyTemporalSimple
from huey_gpu_conversational_experiment_backup import HueyGPUConversationalNetwork

def quick_test():
    """Quick comparison test."""
    
    # Test text - scientific content similar to Feynman
    test_text = """
    The fundamental principles of quantum mechanics describe the behavior of matter and energy at atomic scales.
    Wave-particle duality demonstrates that particles exhibit both wave and particle properties simultaneously.
    The uncertainty principle limits our ability to precisely measure complementary properties of quantum systems.
    Quantum superposition allows particles to exist in multiple states until measurement causes wavefunction collapse.
    Entanglement creates correlations between particles that persist across vast distances instantaneously.
    These quantum phenomena challenge our classical intuition about the nature of physical reality.
    """
    
    print("🧪 QUICK TEMPORAL vs WINDOWED COMPARISON")
    print("="*60)
    print(f"Test text: {len(test_text.split())} words")
    
    # Test windowed method
    print(f"\n🔸 TESTING WINDOWED METHOD:")
    start_time = time.perf_counter()
    
    windowed = HueyGPUConversationalNetwork(max_neurons=100, window_size=8)
    windowed.add_speaker("Test", ['i', 'me'], ['you'])
    windowed.process_speaker_text("Test", test_text)
    
    windowed_time = time.perf_counter() - start_time
    windowed_concepts = len(windowed.concept_neurons)
    windowed_connections = len(windowed.connections)
    
    print(f"   Time: {windowed_time:.3f}s")
    print(f"   Concepts: {windowed_concepts}")  
    print(f"   Connections: {windowed_connections}")
    
    # Test temporal method
    print(f"\n🔸 TESTING TEMPORAL METHOD:")
    start_time = time.perf_counter()
    
    temporal = HueyTemporalSimple(
        max_neurons=100, 
        window_size=8,
        use_temporal_weights=True,
        tau=3.0
    )
    temporal.add_speaker("Test", ['i', 'me'], ['you'])
    temporal.process_speaker_text("Test", test_text)
    
    temporal_time = time.perf_counter() - start_time
    temporal_concepts = len(temporal.concept_neurons)
    temporal_connections = len(temporal.connections)
    
    print(f"   Time: {temporal_time:.3f}s")
    print(f"   Concepts: {temporal_concepts}")
    print(f"   Connections: {temporal_connections}")
    
    # Show temporal connection details
    if temporal_connections > 0:
        print(f"\n🔗 Temporal connections (showing temporal decay):")
        sample_connections = list(temporal.connections.items())[:8]
        for (n1, n2), strength in sample_connections:
            w1 = temporal.neuron_to_word.get(n1, f"neuron_{n1}")
            w2 = temporal.neuron_to_word.get(n2, f"neuron_{n2}")
            print(f"     {w1} <-> {w2}: {strength:.6f}")
    
    # Comparison
    print(f"\n📊 COMPARISON:")
    print(f"   Speed:       Windowed: {windowed_time:.3f}s, Temporal: {temporal_time:.3f}s")
    
    if temporal_time > 0:
        speedup = windowed_time / temporal_time
        print(f"   Speedup:     {speedup:.1f}x {'(Temporal faster)' if speedup > 1 else '(Windowed faster)'}")
    
    print(f"   Concepts:    Windowed: {windowed_concepts}, Temporal: {temporal_concepts}")
    print(f"   Connections: Windowed: {windowed_connections}, Temporal: {temporal_connections}")
    
    # Connection density comparison
    if windowed_concepts > 1:
        windowed_density = windowed_connections / (windowed_concepts * (windowed_concepts - 1) // 2)
        print(f"   Windowed density: {windowed_density:.3f}")
    
    if temporal_concepts > 1:
        temporal_density = temporal_connections / (temporal_concepts * (temporal_concepts - 1) // 2)
        print(f"   Temporal density: {temporal_density:.3f}")
    
    # Analysis
    print(f"\n🎯 ANALYSIS:")
    if temporal_connections > windowed_connections:
        print(f"   ✅ Temporal method created MORE connections ({temporal_connections} vs {windowed_connections})")
    elif temporal_connections == windowed_connections:
        print(f"   ➖ Both methods created same number of connections ({temporal_connections})")
    else:
        print(f"   📉 Temporal method created fewer connections ({temporal_connections} vs {windowed_connections})")
    
    if temporal_concepts > windowed_concepts:
        print(f"   📈 Temporal method processed more concepts ({temporal_concepts} vs {windowed_concepts})")
    elif temporal_concepts == windowed_concepts:
        print(f"   ➖ Both methods processed same concepts ({temporal_concepts})")
    else:
        print(f"   📉 Temporal method processed fewer concepts ({temporal_concepts} vs {windowed_concepts})")
    
    success = temporal_connections > 0 and temporal_concepts > 5
    print(f"\n🎉 RESULT: Temporal learning {'✅ SUCCESS' if success else '❌ NEEDS DEBUGGING'}")
    
    return {
        'windowed': {'time': windowed_time, 'concepts': windowed_concepts, 'connections': windowed_connections},
        'temporal': {'time': temporal_time, 'concepts': temporal_concepts, 'connections': temporal_connections}
    }

if __name__ == "__main__":
    results = quick_test()