#!/usr/bin/env python3
"""
Debug what's different between working command line and broken web interface
"""

from huey_temporal_simple import HueyTemporalSimple
from huey_complete_platform import HueyCompletePlatform

def test_working_version():
    """Test the version we know works"""
    print("=" * 60)
    print("🟢 TESTING WORKING VERSION (like test_high_learning_rate.py)")
    print("=" * 60)
    
    # Create exactly what worked in test_high_learning_rate.py
    huey = HueyTemporalSimple(
        max_neurons=200,
        use_temporal_weights=True,
        tau=6.0,
        use_gpu_acceleration=False,
        learning_rate=1.0
    )
    
    # Process the same file
    result = huey.process_file_with_mode("test_short_mandarin.txt", conversation_mode=True)
    
    print(f"✅ WORKING VERSION RESULTS:")
    print(f"   Concepts: {len(huey.concept_neurons)}")
    print(f"   Connections: {len(huey.connections)}")
    print(f"   Neuron mapping: {len(huey.neuron_to_word)}")
    
    # Show connection strengths
    if huey.connections:
        avg_strength = sum(huey.connections.values()) / len(huey.connections)
        strong_count = sum(1 for s in huey.connections.values() if s > 0.5)
        print(f"   Average strength: {avg_strength:.3f}")
        print(f"   Strong connections (>0.5): {strong_count}")
    
    return huey

def test_web_interface_version():
    """Test exactly what the web interface creates"""
    print("\n" + "=" * 60)
    print("🔴 TESTING WEB INTERFACE VERSION")
    print("=" * 60)
    
    # Create temporal network exactly like web interface does (lines 1760-1768)
    temporal_network = HueyTemporalSimple(
        max_neurons=500,  # Web interface default
        window_size=7,    # Web interface default  
        learning_rate=1.0,  # What we set in UI
        use_gpu_acceleration=False,
        use_temporal_weights=True,
        tau=6.0,  # What we set in UI
        max_connections_per_neuron=250  # Web interface default
    )
    
    # Wrap in platform like web interface does (lines 1769-1775)
    platform = HueyCompletePlatform(
        session_name="temporal_session_test",
        max_neurons=500,
        window_size=7,
        learning_rate=0.15  # This is different! Web interface platform uses 0.15
    )
    platform.network = temporal_network
    
    print(f"🔍 Web interface network type: {type(platform.network)}")
    print(f"🔍 Has process_file_with_mode: {hasattr(platform.network, 'process_file_with_mode')}")
    
    # Test direct temporal network processing
    print("\n📝 Testing direct temporal network processing...")
    result1 = temporal_network.process_file_with_mode("test_short_mandarin.txt", conversation_mode=True)
    
    print(f"   Direct temporal network:")
    print(f"   Concepts: {len(temporal_network.concept_neurons)}")
    print(f"   Connections: {len(temporal_network.connections)}")
    print(f"   Neuron mapping: {len(temporal_network.neuron_to_word)}")
    
    # Test platform wrapper processing  
    print("\n📝 Testing platform wrapper processing...")
    print(f"   platform.network is temporal_network: {platform.network is temporal_network}")
    
    # Check if platform wrapper breaks anything
    print(f"   platform.network.concept_neurons: {len(getattr(platform.network, 'concept_neurons', {}))}")
    print(f"   platform.network.connections: {len(getattr(platform.network, 'connections', {}))}")
    
    return platform

def main():
    # Test both versions
    working_huey = test_working_version()
    web_huey = test_web_interface_version()
    
    print("\n" + "=" * 60)
    print("🔍 COMPARISON ANALYSIS")
    print("=" * 60)
    
    print(f"Working version concepts: {len(working_huey.concept_neurons)}")
    print(f"Web version concepts: {len(web_huey.network.concept_neurons)}")
    
    print(f"\nWorking version connections: {len(working_huey.connections)}")  
    print(f"Web version connections: {len(web_huey.network.connections)}")
    
    # Check if they're the same objects
    print(f"\nSame concept_neurons object? {working_huey.concept_neurons is web_huey.network.concept_neurons}")
    print(f"Same connections object? {working_huey.connections is web_huey.network.connections}")
    
    # Check attributes
    print(f"\nWorking huey attributes: {[attr for attr in dir(working_huey) if not attr.startswith('_')]}")
    print(f"Web huey.network attributes: {[attr for attr in dir(web_huey.network) if not attr.startswith('_')]}")

if __name__ == "__main__":
    main()