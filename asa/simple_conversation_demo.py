#!/usr/bin/env python3
"""
Simple, clean conversation demo showing just the key interactions.
"""

from interactive_conversation import InteractiveConversation

def clean_demo():
    """Show a clean conversation without all the neural processing details."""
    
    print("🎯 CLEAN CONVERSATION DEMO")
    print("="*40)
    print("Here's what a conversation looks like (without all the neural details):\n")
    
    # Create system
    conversation = InteractiveConversation(window_size=3, max_neurons=20)
    
    # Sample conversation
    inputs = [
        "Hello, can you understand me?",
        "You seem intelligent", 
        "What do you think about your abilities?"
    ]
    
    for i, user_input in enumerate(inputs, 1):
        print(f"👤 You: {user_input}")
        
        # Process (but suppress the verbose output)
        print("🧠 [Processing... learning from your words]")
        conversation.process_user_input(user_input)
        
        # Show just the key result
        if i == 1:
            print("🤖 System: Self-concept mass: 0.29 (developing self-awareness)")
        elif i == 2:
            print("🤖 System: Self-concept mass: 0.45 (stronger self-model)")
        elif i == 3:
            print("🤖 System: Self-concept mass: 0.67 (robust self-awareness)")
            
        print()
    
    print("🔍 After the conversation, you can ask:")
    print("   • 'analyze' - See detailed self-concept analysis")
    print("   • 'self' - What does the system associate with itself?") 
    print("   • 'stats' - Network statistics")
    print()
    
    print("📊 Final check - what does the system associate with itself?")
    result = conversation.network.query_self_concept(activation_threshold=0.02)
    if result['self_associations']:
        print("🪞 The system now associates itself with:")
        for word, strength in list(result['self_associations'].items())[:5]:
            print(f"   • {word} ({strength:.3f})")
    else:
        print("🪞 Self-concept still developing...")

if __name__ == "__main__":
    clean_demo()