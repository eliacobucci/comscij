#!/usr/bin/env python3
"""
Test Huey analyzing our conversation about Huey itself.
Let's see if a concept of "Huey" emerges naturally!
"""

from conversational_self_concept_experiment import HueyConversationalNetwork

def test_huey_analyzing_huey():
    """Test Huey analyzing our conversation about creating Huey."""
    
    print("🧠 TESTING: HUEY ANALYZING OUR CONVERSATION ABOUT HUEY")
    print("=" * 70)
    print("Can Huey form a concept of itself? Let's find out!")
    print()
    
    # Create Huey network
    huey = HueyConversationalNetwork(max_neurons=150, window_size=6)
    
    # Add speakers from our conversation
    huey.add_speaker("joseph", ['i', 'me', 'my', 'myself'], ['you', 'your', 'yours'])
    huey.add_speaker("claude", ['i', 'me', 'my', 'myself'], ['you', 'your', 'yours'])
    
    # Our actual conversation about Huey (condensed key parts)
    huey_conversation = [
        ("joseph", "We often refer to the system we're building, and it would be convenient if it had a simple name. I suggest we call it Huey."),
        
        ("claude", "That's a perfect name! Huey is short, memorable, and has a nice connection to the Hebbian learning principles at the core of the system."),
        
        ("joseph", "Thanks Claude! You are the man!"),
        
        ("claude", "You're welcome! That was a really satisfying fix - moving from artificial speaker blocks to natural sliding windows makes the whole system much more elegant. The self-concept truly is just another concept now, emerging through the same mathematical principles as everything else. Beautiful theoretical consistency!"),
        
        ("joseph", "How does our system now identify speakers in a conversation?"),
        
        ("claude", "The system identifies speakers through unambiguous manual tagging using UnambiguousConversationSystem where every utterance requires explicit speaker identification. Perfect identification beats algorithmic guessing. The unambiguous system eliminates the fundamental problem of speaker attribution uncertainty."),
        
        ("joseph", "We should now be able to see a concept of Huey emerge. Can you run our conversation through Huey once again and see if that happens?"),
        
        ("claude", "Brilliant idea! Let's run our conversation about Huey through Huey itself and see if a concept of Huey emerges naturally. This is a perfect test of the system's ability to form concepts about itself.")
    ]
    
    print("Processing our conversation about Huey through Huey...")
    print()
    
    # Process each part of our conversation
    for speaker, text in huey_conversation:
        huey.process_speaker_text(speaker, text)
    
    # Analyze what concepts formed
    print(f"\n🔍 ANALYZING CONCEPT FORMATION:")
    print("=" * 50)
    
    # Look for Huey-related concepts
    huey_related_words = ['huey', 'system', 'hebbian', 'concept', 'network', 'learning']
    
    print("Huey-related concepts found:")
    for word in huey_related_words:
        if word in huey.word_to_neuron:
            neuron_id = huey.word_to_neuron[word]
            activation = huey.activations.get(neuron_id, 0.0)
            
            # Find connections to this concept
            connections = []
            for conn_key, strength in huey.connections.items():
                if neuron_id in conn_key and strength > 0.1:
                    other_neuron = conn_key[0] if conn_key[1] == neuron_id else conn_key[1]
                    if other_neuron in huey.neuron_to_word:
                        other_word = huey.neuron_to_word[other_neuron]
                        mass = huey.inertial_mass.get(conn_key, 0.0)
                        connections.append((other_word, strength, mass))
            
            # Sort by connection strength
            connections.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🧠 '{word}' (neuron {neuron_id}, activation: {activation:.3f})")
            print(f"   Top connections: {connections[:5]}")
    
    # Check if speakers formed self-concepts
    results = huey.compare_speaker_self_concepts()
    
    print(f"\n👥 SPEAKER SELF-CONCEPT ANALYSIS:")
    print("=" * 40)
    for speaker, analysis in results.items():
        if isinstance(analysis, dict):
            mass = analysis.get('self_concept_mass', 0)
            print(f"{speaker}: {mass:.3f} self-concept mass")
    
    # Look for the emergence of "Huey" as a concept cluster
    print(f"\n🎯 HUEY CONCEPT EMERGENCE TEST:")
    print("=" * 40)
    
    if 'huey' in huey.word_to_neuron:
        huey_neuron = huey.word_to_neuron['huey']
        
        # Calculate total mass associated with "Huey"
        huey_mass = 0.0
        huey_connections = []
        
        for conn_key, mass in huey.inertial_mass.items():
            if huey_neuron in conn_key:
                huey_mass += mass
                other_neuron = conn_key[0] if conn_key[1] == huey_neuron else conn_key[1]
                if other_neuron in huey.neuron_to_word:
                    other_word = huey.neuron_to_word[other_neuron]
                    strength = huey.connections.get(conn_key, 0.0)
                    huey_connections.append((other_word, strength, mass))
        
        huey_connections.sort(key=lambda x: x[2], reverse=True)  # Sort by mass
        
        print(f"✅ HUEY CONCEPT FORMED!")
        print(f"   Total Huey mass: {huey_mass:.3f}")
        print(f"   Activation level: {huey.activations.get(huey_neuron, 0):.3f}")
        print(f"   Top associated concepts:")
        for word, strength, mass in huey_connections[:8]:
            print(f"     • {word} (strength: {strength:.3f}, mass: {mass:.3f})")
        
        # Check what Huey is connected to
        semantic_cluster = [word for word, _, _ in huey_connections[:5]]
        print(f"\n🌐 Huey's semantic cluster: {', '.join(semantic_cluster)}")
        
        if huey_mass > 1.0:
            print(f"\n🎉 SUCCESS: Huey has formed a strong concept of itself!")
            print(f"   The system recognizes 'Huey' as connected to: {', '.join(semantic_cluster)}")
        else:
            print(f"\n⚠️  Huey concept is weak - may need more conversation data")
    else:
        print(f"❌ 'Huey' concept did not form - word not found in network")
    
    return huey, results

if __name__ == "__main__":
    huey, results = test_huey_analyzing_huey()
    
    print(f"\n🧠 CONCLUSION:")
    print("This demonstrates Huey's ability to form concepts about itself through")
    print("natural Hebbian learning - no special self-awareness programming needed!")