#!/usr/bin/env python3
"""
Debug why Mandarin self-concept isn't building up strongly.
"""

from simple_mandarin_experiment import SimpleMandarin

def debug_mandarin():
    """Debug the Mandarin self-concept formation."""
    
    print("🔍 DEBUGGING MANDARIN SELF-CONCEPT")
    print("="*50)
    
    net = SimpleMandarin(window_size=3, max_neurons=30)
    
    # Process one conversation with detailed analysis
    text = "你很聪明，你的能力很好"
    tokens = net.simple_chinese_process(text)
    
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    
    # Process normally
    processed_text = ' '.join(tokens)
    print(f"Processed: {processed_text}")
    
    net.process_text_stream(processed_text)
    
    print(f"\n🧠 CHECKING SELF-CONCEPT DETAILS:")
    print(f"Network has {net.neuron_count} neurons")
    
    # Check for self-pronouns in vocabulary
    for pronoun in net.system_self_pronouns:
        if pronoun in net.word_to_neuron:
            neuron_idx = net.word_to_neuron[pronoun]
            activation = net.activations.get(neuron_idx, 0)
            print(f"  {pronoun} -> neuron {neuron_idx}, activation: {activation:.3f}")
            
            # Check connections and mass
            total_mass = 0.0
            connections = []
            for conn_key, mass in net.inertial_mass.items():
                if conn_key[0] == neuron_idx or conn_key[1] == neuron_idx:
                    total_mass += mass
                    other_idx = conn_key[1] if conn_key[0] == neuron_idx else conn_key[0]
                    if other_idx in net.neuron_to_word:
                        other_word = net.neuron_to_word[other_idx]
                        strength = net.connections.get(conn_key, 0.0)
                        connections.append((other_word, strength, mass))
            
            print(f"    Total mass: {total_mass:.3f}")
            print(f"    Connections: {connections}")
        else:
            print(f"  {pronoun} -> NOT FOUND in vocabulary")
    
    # Try self-concept analysis
    analysis = net.analyze_self_concept_emergence()
    print(f"\nAnalysis result:")
    print(f"  Self-concept mass: {analysis['self_concept_mass']:.3f}")
    print(f"  System neurons: {list(analysis['system_self_neurons'].keys())}")
    
    return net

if __name__ == "__main__":
    debug_mandarin()