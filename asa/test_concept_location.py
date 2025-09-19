#!/usr/bin/env python3
"""
Test where concepts are stored after processing
"""

import tempfile
import os
from huey_temporal_simple import HueyTemporalSimple

def test_concept_storage():
    # Create a huey instance like the web interface does
    huey = HueyTemporalSimple(max_neurons=1000, use_temporal_weights=True)
    
    # Create test file
    test_content = "蓝色\n\n我记得三个梦。\n\n梦都是蓝色的。"
    enhanced_conversation_data = [('Author', test_content)]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
        for speaker_id, text in enhanced_conversation_data:
            tmp_file.write(f"{text}\n")
        tmp_file_path = tmp_file.name
    
    try:
        # Process file
        temporal_result = huey.process_file_with_mode(tmp_file_path, conversation_mode=False)
        
        print("🔍 CHECKING CONCEPT STORAGE LOCATIONS:")
        print("="*50)
        
        # Check main huey object
        huey_concepts = getattr(huey, 'concept_neurons', {})
        print(f"huey.concept_neurons: {len(huey_concepts)} - {huey_concepts}")
        
        # Check if huey has a network attribute
        if hasattr(huey, 'network'):
            print(f"huey.network exists: {type(huey.network)}")
            network_concepts = getattr(huey.network, 'concept_neurons', {})
            print(f"huey.network.concept_neurons: {len(network_concepts)} - {network_concepts}")
        else:
            print("huey.network does not exist")
        
        # Check connections
        huey_connections = getattr(huey, 'connections', {})
        print(f"huey.connections: {len(huey_connections)}")
        
        if hasattr(huey, 'network'):
            network_connections = getattr(huey.network, 'connections', {})
            print(f"huey.network.connections: {len(network_connections)}")
        
        print("\n🧪 WEB INTERFACE CALCULATION:")
        # Simulate web interface calculation
        concept_count = len(getattr(huey.network if hasattr(huey, 'network') else huey, 'concept_neurons', {}))
        connection_count = len(getattr(huey.network if hasattr(huey, 'network') else huey, 'connections', {}))
        print(f"concept_count (web interface logic): {concept_count}")
        print(f"connection_count (web interface logic): {connection_count}")
        
        print(f"\n{'SUCCESS' if concept_count > 0 else 'WARNING'}: {'Concepts found!' if concept_count > 0 else 'No concepts learned'}")
        
    finally:
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    test_concept_storage()