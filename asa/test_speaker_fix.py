#!/usr/bin/env python3
"""
Test the speaker neuron fix for distinct self-concept masses.
"""

from conversational_self_concept_experiment import HueyConversationalNetwork

def test_speaker_fix():
    """Test that speakers now get different self-concept masses."""
    
    print("🧪 TESTING SPEAKER NEURON FIX")
    print("=" * 50)
    
    # Create network
    net = HueyConversationalNetwork(max_neurons=100, window_size=5)
    
    # Add speakers
    net.add_speaker("Alice", ['i', 'me', 'my', 'myself'], ['you', 'your'])
    net.add_speaker("Bob", ['i', 'me', 'my', 'myself'], ['you', 'your'])
    
    # Alice talks about herself a lot
    alice_text = "I think I am very interested in this research. My approach is systematic and I believe my methods are sound. I find myself drawn to these concepts."
    
    # Bob asks questions (less self-reference)
    bob_text = "What do you think about this approach? Can you explain the methodology? How does this work?"
    
    print("\n📝 PROCESSING CONVERSATIONS:")
    print(f"Alice: {alice_text}")
    print(f"Bob: {bob_text}")
    
    # Process conversations
    net.process_speaker_text("Alice", alice_text)
    net.process_speaker_text("Bob", bob_text)
    
    print(f"\n🧠 NETWORK STATE:")
    print(f"Total neurons: {net.neuron_count}")
    print(f"Total connections: {len(net.connections)}")
    
    # Analyze self-concepts
    alice_analysis = net.analyze_speaker_self_concept("Alice")
    bob_analysis = net.analyze_speaker_self_concept("Bob")
    
    print(f"\n📊 SELF-CONCEPT ANALYSIS:")
    print(f"Alice self-concept mass: {alice_analysis['self_concept_mass']:.3f}")
    print(f"Bob self-concept mass: {bob_analysis['self_concept_mass']:.3f}")
    
    # Check if they're different
    mass_diff = abs(alice_analysis['self_concept_mass'] - bob_analysis['self_concept_mass'])
    
    if mass_diff > 0.001:  # Small threshold for floating point comparison
        print(f"✅ SUCCESS: Speakers have different self-concept masses (diff: {mass_diff:.3f})")
        print("🎯 Expected: Alice should have higher mass (more self-reference)")
        
        if alice_analysis['self_concept_mass'] > bob_analysis['self_concept_mass']:
            print("✅ CORRECT: Alice has higher self-concept mass")
        else:
            print("🟡 UNEXPECTED: Bob has higher self-concept mass")
            
    else:
        print(f"❌ FAILED: Speakers still have identical masses")
    
    # Show some details
    print(f"\n🔍 ALICE'S SELF-CONCEPT DETAILS:")
    for pronoun, data in alice_analysis['self_concept_neurons'].items():
        print(f"  {pronoun}: mass={data['mass']:.3f}, connections={len(data['connections'])}")
    
    print(f"\n🔍 BOB'S SELF-CONCEPT DETAILS:")
    for pronoun, data in bob_analysis['self_concept_neurons'].items():
        print(f"  {pronoun}: mass={data['mass']:.3f}, connections={len(data['connections'])}")
    
    # Check for speaker neurons
    alice_speaker_neuron = "speaker_alice"
    bob_speaker_neuron = "speaker_bob"
    
    print(f"\n🎭 SPEAKER NEURONS:")
    print(f"Alice speaker neuron exists: {alice_speaker_neuron in net.word_to_neuron}")
    print(f"Bob speaker neuron exists: {bob_speaker_neuron in net.word_to_neuron}")
    
    return {
        'alice_mass': alice_analysis['self_concept_mass'],
        'bob_mass': bob_analysis['self_concept_mass'],
        'mass_difference': mass_diff,
        'test_passed': mass_diff > 0.001
    }

if __name__ == "__main__":
    result = test_speaker_fix()
    print(f"\n{'='*50}")
    print(f"TEST RESULT: {'PASSED' if result['test_passed'] else 'FAILED'}")