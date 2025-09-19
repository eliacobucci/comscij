#!/usr/bin/env python3
"""
Test the complete web interface flow including the HueyCompletePlatform wrapper
"""

import tempfile
import os
from datetime import datetime

def test_web_interface_complete_flow():
    # Simulate the EXACT web interface initialization
    from huey_temporal_simple import HueyTemporalSimple
    from huey_complete_platform import HueyCompletePlatform
    
    # Initialize exactly like the web interface does
    max_neurons = 1000
    window_size = 10
    learning_rate = 0.15
    tau = 3.0
    temporal_learning_rate = 0.2
    
    print("🌐 SIMULATING EXACT WEB INTERFACE INITIALIZATION")
    print("="*60)
    
    # Step 1: Create temporal network (like web interface)
    temporal_network = HueyTemporalSimple(
        max_neurons=max_neurons,
        use_temporal_weights=True,  
        tau=tau,
        learning_rate=temporal_learning_rate,
        use_gpu_acceleration=True,
        max_connections_per_neuron=250
    )
    
    # Step 2: Create HueyCompletePlatform wrapper (like web interface)
    huey = HueyCompletePlatform(
        session_name=f"temporal_session_{int(datetime.now().timestamp())}",
        max_neurons=max_neurons,
        window_size=window_size,
        learning_rate=learning_rate
    )
    
    # Step 3: Assign temporal network (like web interface)
    huey.network = temporal_network
    
    print(f"✅ Initialized HueyCompletePlatform with temporal network")
    print(f"   huey type: {type(huey)}")
    print(f"   huey.network type: {type(huey.network)}")
    
    # Step 4: Create test file (like web interface file processing)
    test_content = "蓝色\n\n我记得三个梦。\n\n梦都是蓝色的。"
    enhanced_conversation_data = [('Author', test_content)]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
        for speaker_id, text in enhanced_conversation_data:
            tmp_file.write(f"{text}\n")
        tmp_file_path = tmp_file.name
    
    try:
        print("\n" + "="*60)
        print("🌐 PROCESSING FILE WITH TEMPORAL LEARNING")
        print("="*60)
        
        # Step 5: Process file exactly like web interface
        temporal_result = huey.network.process_file_with_mode(tmp_file_path, conversation_mode=False)
        
        print(f"📋 Processing result: {temporal_result}")
        
        # Step 6: Get results exactly like web interface
        concept_count = len(getattr(huey.network, 'concept_neurons', {}))
        connection_count = len(getattr(huey.network, 'connections', {}))
        
        print(f"\n📊 WEB INTERFACE RESULTS:")
        print(f"   Concept count: {concept_count}")
        print(f"   Connection count: {connection_count}")
        print(f"   Concepts: {getattr(huey.network, 'concept_neurons', {})}")
        
        # Test the web interface condition
        if concept_count > 0:
            print("✅ SUCCESS: 🎉 Temporal learning successfully created semantic network!")
            
            # Test concept display
            if hasattr(huey.network, 'neuron_to_word') and huey.network.neuron_to_word:
                concepts = list(huey.network.neuron_to_word.values())[:20]
                print(f"📋 Learned Concepts: {', '.join(concepts)}")
            
            # Test connection display  
            if hasattr(huey.network, 'connections') and huey.network.connections:
                connections = huey.network.connections
                if connections:
                    avg_strength = sum(connections.values()) / len(connections)
                    max_strength = max(connections.values())
                    print(f"🔗 Connection Strengths: avg={avg_strength:.3f}, max={max_strength:.3f}")
        else:
            print("❌ WARNING: ⚠️ No concepts learned. Check if the text contains meaningful content.")
    
    finally:
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    test_web_interface_complete_flow()