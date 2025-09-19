#!/usr/bin/env python3
"""
Debug the web interface mapping issue.
"""

from huey_temporal_simple import HueyTemporalSimple
from huey_complete_platform import HueyCompletePlatform

def debug_mappings():
    print("🔍 DEBUGGING WEB INTERFACE MAPPINGS")
    print("=" * 50)
    
    # Create temporal network (what the web interface should be using)
    temporal_network = HueyTemporalSimple(
        max_neurons=500,
        use_temporal_weights=True,
        tau=3.0,
        use_gpu_acceleration=False
    )
    
    # Create platform wrapper (what the web interface actually creates)
    platform = HueyCompletePlatform(
        max_neurons=500,
        window_size=7,
        learning_rate=0.15
    )
    
    # Replace platform's network with temporal network (what web interface does)
    platform.network = temporal_network
    
    print(f"📱 Testing with short Mandarin file...")
    
    # Process the short file
    result = temporal_network.process_file_with_mode("test_short_mandarin.txt", conversation_mode=True)
    
    print(f"\n🧠 TEMPORAL NETWORK MAPPINGS:")
    print(f"   concept_neurons: {len(temporal_network.concept_neurons)}")
    print(f"   word_to_neuron: {len(temporal_network.word_to_neuron)}")
    print(f"   neuron_to_word: {len(temporal_network.neuron_to_word)}")
    
    print(f"\n🎭 PLATFORM WRAPPER MAPPINGS:")
    print(f"   platform.network.concept_neurons: {len(getattr(platform.network, 'concept_neurons', {}))}")
    print(f"   platform.network.word_to_neuron: {len(getattr(platform.network, 'word_to_neuron', {}))}")
    print(f"   platform.network.neuron_to_word: {len(getattr(platform.network, 'neuron_to_word', {}))}")
    
    # Test what the cascade interface would see
    from huey_activation_cascade import create_cascade_interface
    
    print(f"\n🌊 CASCADE INTERFACE TEST:")
    cascade = create_cascade_interface(platform.network)
    
    # The cascade should now see the network properly
    print(f"   Cascade network reference: {type(cascade.huey_network)}")
    print(f"   Cascade sees concept_neurons: {len(getattr(cascade.huey_network, 'concept_neurons', {}))}")
    print(f"   Cascade sees neuron_to_word: {len(getattr(cascade.huey_network, 'neuron_to_word', {}))}")
    
if __name__ == "__main__":
    debug_mappings()