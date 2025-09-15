#!/usr/bin/env python3
"""
Test the punctuation cleaning fix
"""

from experimental_network_complete import ExperimentalNetwork

def test_punctuation_cleaning():
    print("🧪 TESTING PUNCTUATION CLEANING FIX")
    print("="*50)
    
    # Create network
    net = ExperimentalNetwork(window_size=3, max_neurons=20)
    
    # Test text with punctuation variations
    test_text = "The electrodynamics, electrodynamics; and electrodynamics. concepts should merge."
    
    print(f"Input text: {test_text}")
    
    # Process the text
    net.process_text_stream(test_text)
    
    print(f"\nUnique concepts found: {len(net.neuron_to_word)}")
    print("Concepts:", list(net.neuron_to_word.values()))
    
    # Check if electrodynamics variants were unified
    electrodynamics_concepts = [word for word in net.neuron_to_word.values() if 'electrodynamics' in word.lower()]
    
    print(f"\nElectrodynamics variants: {electrodynamics_concepts}")
    
    if len(electrodynamics_concepts) == 1:
        print("✅ SUCCESS: Punctuation variants unified!")
    else:
        print("❌ ISSUE: Still seeing multiple variants")
        
if __name__ == "__main__":
    test_punctuation_cleaning()