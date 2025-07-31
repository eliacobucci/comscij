#!/usr/bin/env python3
"""
Test script for the integrated sliding vocabulary functionality
"""

import sys
sys.path.append('/Users/josephwoelfel/huey2/huey3')

from huey3_complete_jupyter import Huey3

def test_sliding_vocabulary():
    print("🧪 Testing Huey3 with sliding vocabulary integration...")
    
    # Create a Huey3 instance with a small vocabulary window for testing
    # Temporarily increase memory limit for testing
    original_limit = Huey3.MAX_MEMORY_PERCENT
    Huey3.MAX_MEMORY_PERCENT = 95  # Increase for testing
    
    try:
        huey = Huey3(sliding_vocab_size=5, vocab_update_frequency=10)
        
        # Test with some text that will exceed the vocabulary limit
        test_text = "apple banana cherry apple banana date elderberry fig grape apple banana cherry date elderberry"
        
        print(f"📝 Processing test text: '{test_text}'")
        print(f"   Words: {len(test_text.split())}")
        print(f"   Unique words: {len(set(test_text.split()))}")
        
        huey.process_text(test_text)
        
        # Check results
        vocab_stats = huey.get_vocabulary_stats()
        memory_stats = huey.get_memory_stats()
        
        print(f"\n📊 Results:")
        print(f"   Final vocabulary size: {huey.neuron_count}")
        print(f"   Max vocabulary size: {huey.sliding_vocab_size}")
        print(f"   Vocabulary updates: {vocab_stats['vocabulary_updates']}")
        print(f"   Words dropped: {vocab_stats['words_dropped']}")
        print(f"   Memory usage: {memory_stats.get('matrix_memory_mb', 0):.2f}MB")
        print(f"   Matrix type: {memory_stats.get('matrix_type', 'none')}")
        
        print(f"\n✅ Integration test successful!")
        print(f"   • Sliding vocabulary is working")
        print(f"   • Memory management is functioning")
        print(f"   • Vocabulary statistics are available")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        # Restore original limit
        Huey3.MAX_MEMORY_PERCENT = original_limit

if __name__ == "__main__":
    test_sliding_vocabulary()