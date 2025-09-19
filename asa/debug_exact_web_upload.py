#!/usr/bin/env python3
"""
Simulate EXACTLY what happens when you upload Mandarin file through web interface
"""

import tempfile
import os
from huey_temporal_simple import HueyTemporalSimple
from huey_complete_platform import HueyCompletePlatform

def simulate_web_upload():
    print("🔍 SIMULATING EXACT WEB INTERFACE UPLOAD PROCESS")
    print("=" * 60)
    
    # Step 1: Read the file like web interface does
    print("📤 Step 1: Reading uploaded file...")
    with open("test_short_mandarin.txt", 'rb') as f:
        file_bytes = f.read()
    
    # Step 2: Decode like web interface does
    print("🔤 Step 2: Decoding file content...")
    content = file_bytes.decode('utf-8')
    print(f"   Decoded content: {repr(content)}")
    
    # Step 3: Create temp file like web interface does
    print("📝 Step 3: Creating temporary file...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    print(f"   Temp file: {tmp_file_path}")
    
    # Step 4: Read temp file back to verify
    print("📖 Step 4: Reading temp file back...")
    with open(tmp_file_path, 'r', encoding='utf-8') as f:
        temp_content = f.read()
    print(f"   Temp content: {repr(temp_content)}")
    
    # Step 5: Plain text mode processing (conversation_mode=False)
    print("🔧 Step 5: Creating plain text result structure...")
    result = {
        'speakers_info': [('Author', 'Author', 'author')],
        'conversation_data': [('Author', temp_content.strip())],
        'detection_confidence': 1.0,
        'detection_strategy': 'plain_text_mode'
    }
    
    print(f"   Speakers: {result['speakers_info']}")
    print(f"   Conversation data: {result['conversation_data']}")
    
    # Step 6: Create temporal network like web interface does
    print("🧠 Step 6: Creating temporal network...")
    temporal_network = HueyTemporalSimple(
        max_neurons=500,
        use_temporal_weights=True,  
        tau=6.0,
        learning_rate=1.0,
        use_gpu_acceleration=False,
        max_connections_per_neuron=250
    )
    
    # Step 7: Create platform wrapper like web interface does
    print("🏗️ Step 7: Creating platform wrapper...")
    platform = HueyCompletePlatform(
        session_name="test_session",
        max_neurons=500,
        window_size=7,
        learning_rate=0.15  # This might be the issue!
    )
    platform.network = temporal_network
    
    # Step 8: Create temp file from conversation data (like our batch fix does)
    print("📄 Step 8: Creating batch processing temp file...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as batch_file:
        for speaker_id, text in result['conversation_data']:
            batch_file.write(f"{text}\n")
        batch_file_path = batch_file.name
    
    print(f"   Batch file: {batch_file_path}")
    
    # Step 9: Read batch file to verify
    print("📖 Step 9: Reading batch file...")
    with open(batch_file_path, 'r', encoding='utf-8') as f:
        batch_content = f.read()
    print(f"   Batch content: {repr(batch_content)}")
    
    # Step 10: Process with temporal learning (the crucial moment)
    print("⚡ Step 10: Processing with temporal learning...")
    try:
        result = temporal_network.process_file_with_mode(batch_file_path, conversation_mode=False)
        
        print("✅ TEMPORAL PROCESSING RESULTS:")
        print(f"   Concepts: {len(temporal_network.concept_neurons)}")
        print(f"   Connections: {len(temporal_network.connections)}")
        print(f"   Neuron mapping: {len(temporal_network.neuron_to_word)}")
        
        if temporal_network.connections:
            avg_strength = sum(temporal_network.connections.values()) / len(temporal_network.connections)
            strong_count = sum(1 for s in temporal_network.connections.values() if s > 0.5)
            print(f"   Average strength: {avg_strength:.3f}")
            print(f"   Strong connections (>0.5): {strong_count}")
        
        print(f"   Words: {list(temporal_network.neuron_to_word.values())}")
        
    except Exception as e:
        print(f"❌ TEMPORAL PROCESSING FAILED: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 11: Check what platform wrapper sees
    print("🔍 Step 11: What does platform wrapper see?")
    print(f"   platform.network.concept_neurons: {len(getattr(platform.network, 'concept_neurons', {}))}")
    print(f"   platform.network.connections: {len(getattr(platform.network, 'connections', {}))}")
    print(f"   platform.network is temporal_network: {platform.network is temporal_network}")
    
    # Cleanup
    os.unlink(tmp_file_path)
    os.unlink(batch_file_path)

if __name__ == "__main__":
    simulate_web_upload()