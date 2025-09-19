#!/usr/bin/env python3
"""
Debug why Asian languages fail but Western languages work in web interface
"""

from huey_temporal_simple import HueyTemporalSimple
from huey_complete_platform import HueyCompletePlatform

def test_english_file():
    """Test English file processing"""
    print("=" * 60)
    print("🟢 TESTING ENGLISH FILE (should work)")
    print("=" * 60)
    
    # Create temporal network like web interface
    temporal_network = HueyTemporalSimple(
        max_neurons=500,
        window_size=7,    
        learning_rate=1.0,
        use_gpu_acceleration=False,
        use_temporal_weights=True,
        tau=6.0,
        max_connections_per_neuron=250
    )
    
    # Process English file
    result = temporal_network.process_file_with_mode("test_short_english.txt", conversation_mode=True)
    
    print(f"✅ ENGLISH RESULTS:")
    print(f"   Concepts: {len(temporal_network.concept_neurons)}")
    print(f"   Connections: {len(temporal_network.connections)}")
    print(f"   Neuron mapping: {len(temporal_network.neuron_to_word)}")
    
    if temporal_network.connections:
        avg_strength = sum(temporal_network.connections.values()) / len(temporal_network.connections)
        strong_count = sum(1 for s in temporal_network.connections.values() if s > 0.5)
        print(f"   Average strength: {avg_strength:.3f}")
        print(f"   Strong connections (>0.5): {strong_count}")
    
    # Check what words were created
    print(f"   Words created: {list(temporal_network.neuron_to_word.values())}")
    
    return temporal_network

def test_mandarin_file():
    """Test Mandarin file processing"""
    print("\n" + "=" * 60)
    print("🔴 TESTING MANDARIN FILE (fails in web interface)")
    print("=" * 60)
    
    # Create temporal network like web interface  
    temporal_network = HueyTemporalSimple(
        max_neurons=500,
        window_size=7,    
        learning_rate=1.0,
        use_gpu_acceleration=False,
        use_temporal_weights=True,
        tau=6.0,
        max_connections_per_neuron=250
    )
    
    # Process Mandarin file
    result = temporal_network.process_file_with_mode("test_short_mandarin.txt", conversation_mode=True)
    
    print(f"✅ MANDARIN RESULTS:")
    print(f"   Concepts: {len(temporal_network.concept_neurons)}")
    print(f"   Connections: {len(temporal_network.connections)}")
    print(f"   Neuron mapping: {len(temporal_network.neuron_to_word)}")
    
    if temporal_network.connections:
        avg_strength = sum(temporal_network.connections.values()) / len(temporal_network.connections)
        strong_count = sum(1 for s in temporal_network.connections.values() if s > 0.5)
        print(f"   Average strength: {avg_strength:.3f}")
        print(f"   Strong connections (>0.5): {strong_count}")
    
    # Check what words were created
    print(f"   Words created: {list(temporal_network.neuron_to_word.values())}")
    
    return temporal_network

def test_web_interface_simulation():
    """Test what happens when we simulate the web interface file upload process"""
    print("\n" + "=" * 60)  
    print("🔍 SIMULATING WEB INTERFACE UPLOAD PROCESS")
    print("=" * 60)
    
    # Test the web interface file creation process that might be breaking Asian text
    import tempfile
    
    # Simulate what web interface does - create temp file from conversation_data
    conversation_data = [
        ('Speaker_A', '蓝色'),
        ('Speaker_B', '我记得三个梦。'),
        ('Speaker_A', '梦都是蓝色的。')
    ]
    
    print("🔄 Creating temp file like web interface does...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
        for speaker_id, text in conversation_data:
            tmp_file.write(f"{text}\n")
            tmp_file.write("-1\n")  # Speaker separator
        tmp_file_path = tmp_file.name
    
    print(f"📄 Temp file created: {tmp_file_path}")
    
    # Read back what was written
    with open(tmp_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"📄 Temp file content:\n{repr(content)}")
    
    # Process with temporal network
    temporal_network = HueyTemporalSimple(
        max_neurons=500,
        learning_rate=1.0,
        use_temporal_weights=True,
        tau=6.0,
        use_gpu_acceleration=False
    )
    
    result = temporal_network.process_file_with_mode(tmp_file_path, conversation_mode=True)
    
    print(f"✅ WEB INTERFACE SIMULATION RESULTS:")
    print(f"   Concepts: {len(temporal_network.concept_neurons)}")
    print(f"   Connections: {len(temporal_network.connections)}")
    print(f"   Words: {list(temporal_network.neuron_to_word.values())}")
    
    # Clean up
    import os
    os.unlink(tmp_file_path)
    
    return temporal_network

def main():
    english_net = test_english_file()
    mandarin_net = test_mandarin_file()  
    web_sim_net = test_web_interface_simulation()
    
    print("\n" + "=" * 60)
    print("🔍 COMPARISON")
    print("=" * 60)
    print(f"English concepts: {len(english_net.concept_neurons)}")
    print(f"Mandarin concepts: {len(mandarin_net.concept_neurons)}") 
    print(f"Web sim concepts: {len(web_sim_net.concept_neurons)}")
    
    print(f"\nEnglish connections: {len(english_net.connections)}")
    print(f"Mandarin connections: {len(mandarin_net.connections)}")
    print(f"Web sim connections: {len(web_sim_net.connections)}")

if __name__ == "__main__":
    main()