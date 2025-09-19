#!/usr/bin/env python3
"""
Debug what the web interface cascade sees vs what the temporal network has.
"""

from huey_temporal_simple import HueyTemporalSimple
from huey_complete_platform import HueyCompletePlatform
from huey_activation_cascade import create_cascade_interface

def debug_web_interface_cascade():
    print("🔍 DEBUGGING WEB INTERFACE CASCADE ISSUE")
    print("=" * 60)
    
    # Simulate exactly what the web interface does
    print("1️⃣ Creating temporal network...")
    temporal_network = HueyTemporalSimple(
        max_neurons=500,
        use_temporal_weights=True,
        tau=3.0,
        use_gpu_acceleration=False,
        learning_rate=1.0  # High learning rate for strong connections
    )
    
    print("2️⃣ Creating platform wrapper...")
    platform = HueyCompletePlatform(
        max_neurons=500,
        window_size=7,
        learning_rate=0.15
    )
    
    print("3️⃣ Replacing platform network with temporal network...")
    platform.network = temporal_network
    
    print("4️⃣ Processing Mandarin file...")
    result = temporal_network.process_file_with_mode("test_short_mandarin.txt", conversation_mode=True)
    
    print(f"\n📊 AFTER PROCESSING:")
    print(f"   temporal_network.concept_neurons: {len(temporal_network.concept_neurons)}")
    print(f"   temporal_network.neuron_to_word: {len(temporal_network.neuron_to_word)}")
    print(f"   temporal_network.connections: {len(temporal_network.connections)}")
    
    print(f"\n📊 PLATFORM REFERENCES:")
    print(f"   platform.network is temporal_network: {platform.network is temporal_network}")
    print(f"   platform.network.concept_neurons: {len(getattr(platform.network, 'concept_neurons', {}))}")
    print(f"   platform.network.neuron_to_word: {len(getattr(platform.network, 'neuron_to_word', {}))}")
    
    print(f"\n🌊 TESTING CASCADE CREATION (like web interface does)...")
    
    # Test what different cascade calls see
    cascade1 = create_cascade_interface(temporal_network)
    print(f"   create_cascade_interface(temporal_network): sees {len(getattr(temporal_network, 'neuron_to_word', {}))} concepts")
    
    cascade2 = create_cascade_interface(platform.network)
    print(f"   create_cascade_interface(platform.network): sees {len(getattr(platform.network, 'neuron_to_word', {}))} concepts")
    
    # Test direct attribute access
    print(f"\n🔍 DIRECT ATTRIBUTE CHECKS:")
    print(f"   hasattr(temporal_network, 'neuron_to_word'): {hasattr(temporal_network, 'neuron_to_word')}")
    print(f"   hasattr(platform.network, 'neuron_to_word'): {hasattr(platform.network, 'neuron_to_word')}")
    print(f"   temporal_network.neuron_to_word keys: {list(temporal_network.neuron_to_word.keys())[:5]}")
    
    # Check if the web interface might be using a different query
    print(f"\n🧪 SIMULATE WEB INTERFACE VISUALIZATION CALL:")
    try:
        # This is what the web interface does for visualization
        coordinates, eigenvals, labels, eigenvecs = platform.network.get_3d_coordinates()
        print(f"   get_3d_coordinates() returned: {len(labels)} labels")
        print(f"   Labels: {labels[:5] if len(labels) > 0 else 'None'}")
        
        if len(labels) == 0:
            print("   ❌ WEB INTERFACE GETS EMPTY COORDINATES!")
        else:
            print("   ✅ Web interface should see coordinates properly")
            
    except Exception as e:
        print(f"   ❌ Error getting coordinates: {e}")

if __name__ == "__main__":
    debug_web_interface_cascade()