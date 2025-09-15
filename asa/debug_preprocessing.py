#!/usr/bin/env python3
"""
Debug the Asian preprocessing pipeline to see where it's failing.
"""

from asian_language_preprocessor import AsianLanguagePreprocessor
from huey_temporal_simple import HueyTemporalSimple

def test_full_pipeline():
    """Test the complete pipeline from Japanese text to Huey processing."""
    
    print("🔬 DEBUGGING ASIAN PREPROCESSING PIPELINE")
    print("="*60)
    
    # Test text
    japanese_text = "こんにちは、わたしをりかいできますか？"
    
    print(f"1. Original Japanese text: {japanese_text}")
    
    # Step 1: Preprocessing
    preprocessor = AsianLanguagePreprocessor()
    processed_text = preprocessor.preprocess_text(japanese_text)
    
    print(f"2. Processed text: {processed_text}")
    print(f"3. Processed length: {len(processed_text.split())} tokens")
    
    # Step 2: Try feeding to Huey
    print("\n4. Creating Huey network...")
    huey = HueyTemporalSimple(max_neurons=50, window_size=5)
    huey.add_speaker("Test", ['i', 'me', 'my'], ['you', 'your'])
    
    print("5. Processing through Huey...")
    try:
        huey.process_speaker_text("Test", processed_text)
        print(f"✅ Processing succeeded!")
        print(f"   Concepts found: {len(huey.concept_neurons)}")
        print(f"   Connections: {len(huey.connections)}")
        
        if len(huey.concept_neurons) > 0:
            print(f"   Sample concepts: {list(huey.concept_neurons.keys())[:10]}")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Step 3: Try with simpler text
    print("\n6. Testing with simple ASCII...")
    simple_text = "hello world test"
    try:
        huey2 = HueyTemporalSimple(max_neurons=50, window_size=5)
        huey2.add_speaker("Test", ['i', 'me', 'my'], ['you', 'your'])
        huey2.process_speaker_text("Test", simple_text)
        print(f"✅ ASCII processing: {len(huey2.concept_neurons)} concepts")
    except Exception as e:
        print(f"❌ ASCII processing failed: {e}")

if __name__ == "__main__":
    test_full_pipeline()