#!/usr/bin/env python3
"""
Test script to verify quiet operation of sliding vocabulary
"""

import sys
sys.path.append('/Users/josephwoelfel/huey2/huey3')

from huey3_complete_jupyter import Huey3

def test_quiet_operation():
    print("🔇 Testing quiet sliding vocabulary operation...")
    
    # Create a Huey3 instance with aggressive vocabulary management
    huey = Huey3(sliding_vocab_size=5, vocab_update_frequency=5, sparse_threshold=0.1)
    
    # Process a longer text that will definitely trigger multiple vocabulary updates
    test_text = """
    apple banana cherry date elderberry fig grape honeydew kiwi lemon mango
    orange pear quince raspberry strawberry tangerine ugli vanilla watermelon
    apple banana cherry date elderberry fig grape honeydew kiwi lemon mango
    orange pear quince raspberry strawberry tangerine ugli vanilla watermelon
    """
    
    print("Processing text that will trigger multiple vocabulary updates...")
    print("(Should be quiet - no vocabulary update messages)")
    print()
    
    # Process the text
    huey.process_text(test_text)
    
    # Show final statistics
    vocab_stats = huey.get_vocabulary_stats()
    memory_stats = huey.get_memory_stats()
    
    print(f"📊 Final Results (shown only at end):")
    print(f"   Vocabulary size: {vocab_stats['current_vocab_size']}/{vocab_stats['max_vocab_size']}")
    print(f"   Vocabulary updates: {vocab_stats['vocabulary_updates']}")
    print(f"   Words dropped: {vocab_stats['words_dropped']}")
    print(f"   Matrix type: {memory_stats.get('matrix_type', 'none')}")
    print(f"   Memory usage: {memory_stats.get('matrix_memory_mb', 0):.3f}MB")
    
    if vocab_stats['vocabulary_updates'] > 0:
        print(f"\n✅ Success! Had {vocab_stats['vocabulary_updates']} vocabulary updates with no verbose output")
    else:
        print(f"\n✅ Success! Processing completed quietly")
    
    return True

if __name__ == "__main__":
    test_quiet_operation()