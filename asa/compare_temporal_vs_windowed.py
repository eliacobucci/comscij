#!/usr/bin/env python3
"""
Comprehensive comparison between temporal and windowed learning methods.
"""

import time
import numpy as np
from huey_temporal_simple import HueyTemporalSimple
from huey_gpu_conversational_experiment_backup import HueyGPUConversationalNetwork

def test_method(network_class, use_temporal, test_file, label):
    """Test a learning method on a file."""
    print(f"\n{'='*60}")
    print(f"TESTING {label}")
    print(f"{'='*60}")
    
    # Create network
    if network_class == HueyTemporalSimple:
        huey = network_class(
            max_neurons=200, 
            window_size=8,
            use_temporal_weights=use_temporal,
            tau=3.0
        )
    else:
        huey = network_class(max_neurons=200, window_size=8)
    
    huey.add_speaker("Test", ['i', 'me', 'my'], ['you', 'your'])
    
    # Read test file
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📄 Loaded file: {len(text)} characters, {len(text.split())} words")
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        # Use fallback text
        text = """
        The principle of relativity states that the laws of physics are the same in all inertial reference frames.
        This means that there is no absolute reference frame, and all motion is relative.
        Einstein extended this principle to include gravity through his theory of general relativity.
        Space and time are unified into a single continuum called spacetime.
        Massive objects curve spacetime, and this curvature is what we experience as gravity.
        The speed of light is constant in all reference frames, leading to time dilation and length contraction.
        These effects become significant at high velocities approaching the speed of light.
        """
        print(f"📄 Using fallback text: {len(text.split())} words")
    
    # Process text with timing
    start_time = time.perf_counter()
    
    if hasattr(huey, 'process_file_with_mode'):
        # Use file processing for conversation mode
        result = huey.process_file_with_mode(test_file, conversation_mode=False)
        if 'error' not in result:
            print(f"✅ File processed successfully")
        else:
            print(f"⚠️ File processing failed, using direct text processing")
            huey.process_speaker_text("Test", text)
    else:
        # Direct text processing
        huey.process_speaker_text("Test", text)
    
    processing_time = time.perf_counter() - start_time
    
    # Collect metrics
    concepts = len(huey.concept_neurons)
    connections = len(huey.connections)
    activations = len(huey.activations)
    
    # Connection strength analysis
    if connections > 0:
        strengths = list(huey.connections.values())
        avg_strength = np.mean(strengths)
        max_strength = np.max(strengths)
        min_strength = np.min(strengths)
        std_strength = np.std(strengths)
    else:
        avg_strength = max_strength = min_strength = std_strength = 0.0
    
    # Calculate network density
    max_possible_connections = concepts * (concepts - 1) // 2
    density = connections / max_possible_connections if max_possible_connections > 0 else 0.0
    
    results = {
        'method': label,
        'processing_time': processing_time,
        'concepts': concepts,
        'connections': connections,
        'activations': activations,
        'network_density': density,
        'avg_strength': avg_strength,
        'max_strength': max_strength,
        'min_strength': min_strength,
        'std_strength': std_strength,
        'words_per_second': len(text.split()) / processing_time if processing_time > 0 else 0
    }
    
    # Print results
    print(f"\n📊 RESULTS:")
    print(f"   Processing time: {processing_time:.3f}s")
    print(f"   Words per second: {results['words_per_second']:.1f}")
    print(f"   Concepts: {concepts}")
    print(f"   Connections: {connections}")
    print(f"   Network density: {density:.3f}")
    print(f"   Connection strengths:")
    print(f"     Average: {avg_strength:.6f}")
    print(f"     Range: {min_strength:.6f} to {max_strength:.6f}")
    print(f"     Std dev: {std_strength:.6f}")
    
    # Show sample connections with temporal analysis for temporal method
    if connections > 0:
        print(f"\n🔗 Sample connections:")
        sample_connections = list(huey.connections.items())[:5]
        for (n1, n2), strength in sample_connections:
            w1 = huey.neuron_to_word.get(n1, f"neuron_{n1}")
            w2 = huey.neuron_to_word.get(n2, f"neuron_{n2}")
            print(f"     {w1} <-> {w2}: {strength:.6f}")
    
    return results

def compare_methods(test_file="feynman4.pdf"):
    """Compare temporal vs windowed learning methods."""
    print(f"🧪 TEMPORAL vs WINDOWED LEARNING COMPARISON")
    print(f"Using test file: {test_file}")
    
    # Test windowed method (baseline)
    windowed_results = test_method(
        HueyGPUConversationalNetwork, 
        False, 
        test_file, 
        "WINDOWED LEARNING (BASELINE)"
    )
    
    # Test temporal method
    temporal_results = test_method(
        HueyTemporalSimple, 
        True, 
        test_file, 
        "TEMPORAL DECAY LEARNING"
    )
    
    # Comparison analysis
    print(f"\n{'='*60}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\n⏱️  PERFORMANCE:")
    print(f"   Windowed:  {windowed_results['processing_time']:.3f}s ({windowed_results['words_per_second']:.1f} words/sec)")
    print(f"   Temporal:  {temporal_results['processing_time']:.3f}s ({temporal_results['words_per_second']:.1f} words/sec)")
    speedup = windowed_results['processing_time'] / temporal_results['processing_time'] if temporal_results['processing_time'] > 0 else 0
    print(f"   Speedup:   {speedup:.1f}x {'🚀' if speedup > 1 else '🐌'}")
    
    print(f"\n🧠 NETWORK STRUCTURE:")
    print(f"   Concepts:     Windowed: {windowed_results['concepts']}, Temporal: {temporal_results['concepts']}")
    print(f"   Connections:  Windowed: {windowed_results['connections']}, Temporal: {temporal_results['connections']}")
    print(f"   Density:      Windowed: {windowed_results['network_density']:.3f}, Temporal: {temporal_results['network_density']:.3f}")
    
    print(f"\n💪 CONNECTION STRENGTH:")
    print(f"   Average:      Windowed: {windowed_results['avg_strength']:.6f}, Temporal: {temporal_results['avg_strength']:.6f}")
    print(f"   Maximum:      Windowed: {windowed_results['max_strength']:.6f}, Temporal: {temporal_results['max_strength']:.6f}")
    print(f"   Variability:  Windowed: {windowed_results['std_strength']:.6f}, Temporal: {temporal_results['std_strength']:.6f}")
    
    # Qualitative assessment
    print(f"\n🎯 ASSESSMENT:")
    if temporal_results['connections'] > windowed_results['connections']:
        print(f"   ✅ Temporal learning creates MORE connections ({temporal_results['connections']} vs {windowed_results['connections']})")
    elif temporal_results['connections'] < windowed_results['connections']:
        print(f"   📉 Temporal learning creates FEWER connections ({temporal_results['connections']} vs {windowed_results['connections']})")
    else:
        print(f"   ➖ Both methods create same number of connections ({temporal_results['connections']})")
    
    if temporal_results['avg_strength'] > windowed_results['avg_strength']:
        print(f"   💪 Temporal connections are STRONGER on average")
    elif temporal_results['avg_strength'] < windowed_results['avg_strength']:
        print(f"   📉 Temporal connections are WEAKER on average") 
    else:
        print(f"   ➖ Similar connection strengths")
    
    if speedup > 1.2:
        print(f"   🚀 Temporal method is SIGNIFICANTLY FASTER")
    elif speedup > 0.8:
        print(f"   ⚖️  Similar processing speeds")
    else:
        print(f"   🐌 Temporal method is slower")
    
    return windowed_results, temporal_results

if __name__ == "__main__":
    # Run the comparison
    windowed_results, temporal_results = compare_methods("feynman4.pdf")
    
    print(f"\n🎉 COMPARISON COMPLETE!")
    print(f"Temporal learning {'SUCCESS' if temporal_results['connections'] > 0 else 'needs debugging'}")