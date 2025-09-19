#!/usr/bin/env python3
"""
Test script to simulate exactly what the web interface does
"""

import tempfile
import os

def test_web_interface_flow():
    # Step 1: Create a temporary file (like the web interface does)
    test_content = "蓝色\n\n我记得三个梦。\n\n梦都是蓝色的。"
    
    # Simulate the web interface creating enhanced_conversation_data
    enhanced_conversation_data = [('Author', test_content)]
    
    # Create temporary file with enhanced conversation (exactly like web interface)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
        for speaker_id, text in enhanced_conversation_data:
            tmp_file.write(f"{text}\n")
        tmp_file_path = tmp_file.name
    
    print(f"📁 Created temporary file: {tmp_file_path}")
    
    # Read back the temp file to verify content
    with open(tmp_file_path, 'r', encoding='utf-8') as f:
        tmp_content = f.read()
        print(f"📝 Temp file content: {repr(tmp_content)}")
    
    # Step 2: Initialize Huey (like the web interface does)
    from huey_temporal_simple import HueyTemporalSimple
    
    # Initialize in temporal mode 
    huey = HueyTemporalSimple(max_neurons=1000, use_temporal_weights=True)
    
    print(f"🧪 Huey initialized with use_temporal_weights={huey.use_temporal_weights}")
    
    # Step 3: Process file with mode (exactly like the web interface)
    try:
        print("\n" + "="*60)
        print("🌐 SIMULATING WEB INTERFACE PROCESSING")
        print("="*60)
        
        # Process entire file at once with temporal learning (conversation_mode=False for plain text)
        temporal_result = huey.process_file_with_mode(tmp_file_path, conversation_mode=False)
        
        print(f"✅ Processing result: {temporal_result}")
        
        # Get results (like the web interface does)
        concept_count = len(getattr(huey, 'concept_neurons', {}))
        connection_count = len(getattr(huey, 'connections', {}))
        
        print(f"📊 Final results:")
        print(f"   Concepts: {concept_count}")  
        print(f"   Connections: {connection_count}")
        print(f"   Concept neurons: {getattr(huey, 'concept_neurons', {})}")
        
    finally:
        # Clean up
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    test_web_interface_flow()