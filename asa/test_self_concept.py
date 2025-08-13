#!/usr/bin/env python3
"""
Test script for self-concept emergence in the experimental network.
"""

from experimental_network_complete import ExperimentalNetwork

def test_self_concept_emergence():
    """Test the self-concept development features."""
    
    print("🧠 TESTING SELF-CONCEPT EMERGENCE")
    print("="*50)
    
    # Create network with limited capacity
    net = ExperimentalNetwork(window_size=3, max_neurons=20)
    
    # First, basic training text
    basic_text = "Paris France wine food Rome Italy pasta"
    print("📚 Processing basic training text...")
    net.process_text_stream(basic_text)
    
    # Now process conversational text with system-directed pronouns
    conversational_text = """You are intelligent. Can you help me? Your responses are good. 
    I think you understand well. You seem to learn from our conversation. 
    What do you think about your own capabilities?"""
    
    print("\n💬 Processing conversational text...")
    net.process_conversational_text(conversational_text)
    
    print("\n" + "="*50)
    print("ANALYZING SELF-CONCEPT EMERGENCE")
    print("="*50)
    
    # Analyze self-concept
    analysis = net.analyze_self_concept_emergence()
    net.print_self_concept_analysis(analysis)
    
    print("\n" + "="*50)
    print("TESTING SELF-CONCEPT QUERY")
    print("="*50)
    
    # Query self-concept
    result = net.query_self_concept(activation_threshold=0.02)
    net.print_self_concept_query(result)
    
    print("\n✅ Self-concept testing complete!")

if __name__ == "__main__":
    test_self_concept_emergence()