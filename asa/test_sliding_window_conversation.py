#!/usr/bin/env python3
"""
Test Huey: the conversational self-concept analysis system with sliding windows.
"""

from conversational_self_concept_experiment import HueyConversationalNetwork

def test_sliding_window_conversation():
    """Test conversation processing with sliding windows instead of speaker blocks."""
    
    print("🧪 TESTING HUEY: SLIDING WINDOW CONVERSATION PROCESSING")
    print("=" * 60)
    
    # Create Huey network with sliding windows
    net = HueyConversationalNetwork(max_neurons=50, window_size=5)
    
    # Add speakers
    net.add_speaker("alice", ['i', 'me', 'my', 'myself'], ['you', 'your', 'yours'])
    net.add_speaker("bob", ['i', 'me', 'my', 'myself'], ['you', 'your', 'yours'])
    
    # Test conversation
    conversation = [
        ("alice", "Hello, I think this new approach will work better for me."),
        ("bob", "I agree with you. My experience shows that sliding windows help."),
        ("alice", "Yes, I've found that my self-concept develops more naturally this way."),
        ("bob", "That makes sense to me. I feel the same way about my own development.")
    ]
    
    # Process conversation
    for speaker, text in conversation:
        net.process_speaker_text(speaker, text)
    
    # Compare results
    print(f"\n📊 FINAL ANALYSIS:")
    results = net.compare_speaker_self_concepts()
    
    # Verify that self-concepts formed properly
    alice_mass = results['alice']['self_concept_mass'] if 'alice' in results else 0
    bob_mass = results['bob']['self_concept_mass'] if 'bob' in results else 0
    
    print(f"✅ Alice self-concept mass: {alice_mass:.3f}")
    print(f"✅ Bob self-concept mass: {bob_mass:.3f}")
    
    if alice_mass > 0 and bob_mass > 0:
        print(f"🎯 SUCCESS: Huey formed self-concepts with sliding windows!")
        print(f"   Self-concept neurons treated as regular concepts with natural decay")
    else:
        print(f"❌ ISSUE: Self-concepts may not have formed properly")
    
    return net, results

if __name__ == "__main__":
    net, results = test_sliding_window_conversation()