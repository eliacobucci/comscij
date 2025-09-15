#!/usr/bin/env python3
"""
Debug the exact web interface upload flow to find where concepts are lost.
"""

import tempfile
import os
from huey_complete_platform import HueyCompletePlatform
from huey_speaker_detector import HueySpeakerDetector
from asian_language_preprocessor import AsianLanguagePreprocessor

def debug_upload_pipeline():
    print("🔍 DEBUGGING WEB INTERFACE UPLOAD PIPELINE")
    print("="*60)
    
    # Read the Japanese file
    with open('test_japanese_simple.txt', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"1. Original content: {repr(content)}")
    print(f"   Length: {len(content)} chars")
    
    # Step 1: Create temp file (simulating upload)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    print(f"2. Temp file created: {tmp_file_path}")
    
    # Step 2: Speaker detection
    print("\n3. SPEAKER DETECTION PHASE:")
    detector = HueySpeakerDetector(conversation_mode=False)
    result = detector.process_conversation_file(tmp_file_path)
    
    print(f"   Detection result keys: {list(result.keys())}")
    if 'error' not in result:
        print(f"   Speakers: {len(result.get('speakers_info', []))}")
        print(f"   Exchanges: {len(result.get('conversation_data', []))}")
        print(f"   First exchange: {result.get('conversation_data', [])[:1]}")
    else:
        print(f"   Error: {result['error']}")
        return
    
    # Step 3: Huey initialization  
    print("\n4. HUEY INITIALIZATION:")
    huey = HueyCompletePlatform(max_neurons=30, window_size=3)
    print("   Huey created")
    
    # Step 4: Register speakers
    print("\n5. SPEAKER REGISTRATION:")
    huey.register_speakers(result['speakers_info'])
    print(f"   Registered speakers: {[s[0] for s in result['speakers_info']]}")
    
    # Step 5: Process each exchange (the critical part)
    print("\n6. EXCHANGE PROCESSING:")
    for i, (speaker_id, text) in enumerate(result['conversation_data']):
        print(f"\n   Exchange {i+1}:")
        print(f"   Speaker: {speaker_id}")
        print(f"   Text: {repr(text)}")
        
        # Asian preprocessing
        preprocessor = AsianLanguagePreprocessor()
        processed_text = preprocessor.preprocess_text(text)
        print(f"   Processed: {repr(processed_text)}")
        
        # The actual Huey processing
        try:
            huey.network.process_speaker_text(speaker_id, processed_text)
            
            # Check network state after processing
            concepts = len(huey.network.neuron_to_word) if hasattr(huey.network, 'neuron_to_word') else 0
            connections = len(huey.network.connections) if hasattr(huey.network, 'connections') else 0
            
            print(f"   Result: {concepts} concepts, {connections} connections")
            
            if concepts > 0:
                sample_concepts = list(huey.network.neuron_to_word.values())[:3]
                print(f"   Sample concepts: {sample_concepts}")
            
        except Exception as e:
            print(f"   Processing error: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 6: Final analysis
    print("\n7. FINAL ANALYSIS:")
    try:
        analysis_results = huey._generate_comprehensive_analysis()
        print(f"   Analysis keys: {list(analysis_results.keys())}")
        
        total_concepts = analysis_results.get('network_statistics', {}).get('total_concepts', 0)
        total_connections = analysis_results.get('network_statistics', {}).get('total_connections', 0)
        
        print(f"   Final concepts: {total_concepts}")
        print(f"   Final connections: {total_connections}")
        
        if total_concepts == 0:
            print("   ❌ ZERO CONCEPTS - This is the problem!")
            
            # Let's inspect the network state directly
            print("\n   NETWORK INSPECTION:")
            if hasattr(huey.network, 'neuron_to_word'):
                print(f"     neuron_to_word entries: {len(huey.network.neuron_to_word)}")
            if hasattr(huey.network, 'word_to_neuron'):
                print(f"     word_to_neuron entries: {len(huey.network.word_to_neuron)}")
            if hasattr(huey.network, 'connections'):
                print(f"     connections: {len(huey.network.connections)}")
                
    except Exception as e:
        print(f"   Analysis error: {e}")
    
    # Cleanup
    os.unlink(tmp_file_path)
    print(f"\n8. Cleanup: {tmp_file_path} removed")

if __name__ == "__main__":
    debug_upload_pipeline()