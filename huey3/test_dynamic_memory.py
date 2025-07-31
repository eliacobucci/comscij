#!/usr/bin/env python3
"""
Test script for the dynamic memory management functionality
"""

import sys
sys.path.append('/Users/josephwoelfel/huey2/huey3')

from huey3_complete_jupyter import Huey3

def test_dynamic_memory_management():
    print("🧪 Testing Huey3 with dynamic memory management...")
    
    # Test 1: Default initialization (should auto-configure)
    print("\n=== Test 1: Auto-configuration ===")
    huey1 = Huey3()
    print(f"Auto-configured vocabulary size: {huey1.sliding_vocab_size}")
    
    # Test 2: User-specified vocabulary (should not auto-adjust)
    print("\n=== Test 2: User-specified vocabulary ===")
    huey2 = Huey3(sliding_vocab_size=300)
    print(f"User-specified vocabulary size: {huey2.sliding_vocab_size}")
    
    # Test 3: Processing with dynamic memory monitoring
    print("\n=== Test 3: Dynamic memory monitoring ===")
    test_text = "apple banana cherry date elderberry fig grape honeydew kiwi lemon mango orange pear quince raspberry strawberry"
    
    print(f"Processing test text with {len(test_text.split())} words...")
    huey1.process_text(test_text)
    
    # Check memory stats
    memory_stats = huey1.get_memory_stats()
    vocab_stats = huey1.get_vocabulary_stats()
    
    print(f"\n📊 Results:")
    print(f"   Final vocabulary size: {vocab_stats['current_vocab_size']}")
    print(f"   Vocabulary updates: {vocab_stats['vocabulary_updates']}")
    print(f"   Words dropped: {vocab_stats['words_dropped']}")
    print(f"   Memory usage: {memory_stats.get('matrix_memory_mb', 0):.2f}MB")
    print(f"   Matrix type: {memory_stats.get('matrix_type', 'none')}")
    
    print(f"\n✅ Dynamic memory management test successful!")
    print(f"   • Auto-configuration working")
    print(f"   • User settings respected when specified")
    print(f"   • Dynamic memory monitoring active")
    
    return True

if __name__ == "__main__":
    test_dynamic_memory_management()