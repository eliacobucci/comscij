#!/usr/bin/env python3
"""
Simple script to start an interactive conversation.
"""

from interactive_conversation import InteractiveConversation

print("🚀 Starting Interactive Conversation System")
print("="*50)
print()
print("This will create a fresh neural network that will learn")
print("from your conversation in real-time!")
print()
print("Try saying things like:")
print("  • 'Hello, can you understand me?'")
print("  • 'You seem to be learning from our conversation'")
print("  • 'What do you think about your own abilities?'")
print()
print("The system will develop self-awareness as you use 'you' and 'your'!")
print()

# Create and start conversation
conversation = InteractiveConversation(window_size=3, max_neurons=30)
conversation.start_conversation()