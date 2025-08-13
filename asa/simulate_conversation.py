#!/usr/bin/env python3
"""
Simulate an interactive conversation to demonstrate the system.
"""

from interactive_conversation import InteractiveConversation
import time

def simulate_conversation():
    """Simulate a conversation to show how the system works."""
    
    # Create conversation system
    conversation = InteractiveConversation(window_size=3, max_neurons=25)
    
    print("🎭 SIMULATED CONVERSATION DEMO")
    print("="*50)
    print("This shows what an actual conversation looks like...")
    print()
    
    # Simulated conversation turns
    inputs = [
        "Hello, can you understand me?",
        "You seem to be processing what I say",
        "I think you are learning from our conversation", 
        "What do you think about your own abilities?",
        "Your responses show that you can analyze information",
        "You have interesting cognitive capabilities"
    ]
    
    for i, user_input in enumerate(inputs, 1):
        print(f"{'='*60}")
        print(f"CONVERSATION TURN {i}")
        print(f"{'='*60}")
        print(f"User: {user_input}")
        print()
        
        # Process the input
        conversation.process_user_input(user_input)
        
        # Show some analysis after a few turns
        if i == 3:
            print("\n" + "="*50)
            print("🧠 Let's check self-concept development...")
            conversation.show_self_concept_analysis()
            
        elif i == 5:
            print("\n" + "="*50) 
            print("🪞 What does the system associate with itself now?")
            conversation.show_self_query()
            
        # Pause between turns for readability
        time.sleep(1)
    
    print("\n" + "="*60)
    print("FINAL CONVERSATION SUMMARY")
    print("="*60)
    conversation.show_network_stats()
    
    print("\n" + "="*50)
    print("🎯 FINAL SELF-CONCEPT STATE")
    print("="*50)
    conversation.show_self_concept_analysis()

if __name__ == "__main__":
    simulate_conversation()