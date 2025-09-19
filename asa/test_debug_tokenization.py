#!/usr/bin/env python3
"""
Quick test script to debug the tokenization process directly
"""

from huey_temporal_simple import HueyTemporalSimple

def test_chinese_tokenization():
    # Initialize Huey
    huey = HueyTemporalSimple(max_neurons=1000)
    
    # Test text from test_small.txt
    test_text = "蓝色\n\n我记得三个梦。\n\n梦都是蓝色的。"
    
    print("🧪 TESTING CHINESE TOKENIZATION")
    print(f"Input text: '{test_text}'")
    print("="*60)
    
    # Test the tokenization method directly
    result = huey._tokenize_multilingual(test_text)
    
    print("="*60)
    print(f"✅ Final result: {result}")
    print(f"📊 Token count: {len(result)}")
    
    # Test processing a single speaker text
    print("\n" + "="*60)
    print("🧪 TESTING FULL PROCESSING PIPELINE")
    print("="*60)
    
    huey._process_text_temporal_fallback("TestSpeaker", test_text)

if __name__ == "__main__":
    test_chinese_tokenization()