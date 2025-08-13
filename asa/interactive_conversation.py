#!/usr/bin/env python3
"""
Interactive Conversation System for the Experimental Neural Network.
Allows direct keyboard input while preserving all existing functionality.
"""

from experimental_network_complete import ExperimentalNetwork
import time

class InteractiveConversation:
    """
    Interactive conversation wrapper for the experimental neural network.
    Handles real-time conversation while tracking self-concept development.
    """
    
    def __init__(self, network=None, window_size=3, max_neurons=25):
        """
        Initialize interactive conversation system.
        
        Args:
            network (ExperimentalNetwork): Existing network, or None to create new one
            window_size (int): Window size if creating new network
            max_neurons (int): Max neurons if creating new network
        """
        if network is None:
            self.network = ExperimentalNetwork(window_size=window_size, max_neurons=max_neurons)
            self.network_created = True
        else:
            self.network = network
            self.network_created = False
            
        self.conversation_history = []
        self.session_start_time = time.time()
        self.turn_count = 0
        
    def start_conversation(self):
        """Start interactive conversation mode."""
        print("🤖 INTERACTIVE CONVERSATION MODE")
        print("="*50)
        print("Commands:")
        print("  'quit' or 'exit' - End conversation")
        print("  'analyze' - Show current self-concept analysis") 
        print("  'query <word>' - Query associations for a word")
        print("  'context <phrase>' - Multi-word context query")
        print("  'self' - Query what system associates with itself")
        print("  'stats' - Show network statistics")
        print("  'history' - Show conversation history")
        print("  'help' - Show this help message")
        print("-"*50)
        
        if self.network_created:
            print("🧠 Created new neural network")
        else:
            print("🧠 Using existing neural network")
        print(f"📊 Current capacity: {self.network.neuron_count}/{self.network.max_neurons} neurons")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                    
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    self.end_conversation()
                    break
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif user_input.lower() == 'analyze':
                    self.show_self_concept_analysis()
                    continue
                elif user_input.lower() == 'self':
                    self.show_self_query()
                    continue
                elif user_input.lower() == 'stats':
                    self.show_network_stats()
                    continue
                elif user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue
                elif user_input.lower().startswith('query '):
                    word = user_input[6:].strip()
                    self.show_word_query(word)
                    continue
                elif user_input.lower().startswith('context '):
                    phrase = user_input[8:].strip()
                    self.show_context_query(phrase)
                    continue
                    
                # Process regular conversational input
                self.process_user_input(user_input)
                
            except KeyboardInterrupt:
                print("\n\n🔄 Interrupted. Type 'quit' to exit properly.")
                continue
            except EOFError:
                print("\n\n👋 Session ended.")
                break
                
    def process_user_input(self, user_input):
        """Process user input through the neural network."""
        self.turn_count += 1
        
        print(f"\n📝 Processing turn {self.turn_count}...")
        print("-" * 30)
        
        # Store in conversation history
        self.conversation_history.append({
            'turn': self.turn_count,
            'timestamp': time.time() - self.session_start_time,
            'input': user_input,
            'processed_words': self.network.processed_words
        })
        
        # Process through the neural network
        self.network.process_conversational_text(user_input)
        
        # Show immediate self-concept update if system-directed pronouns present
        pronouns = self.network.identify_self_concept_pronouns(user_input)
        if pronouns['system_directed']:
            print(f"\n🤖 Self-directed pronouns detected: {pronouns['system_directed']}")
            analysis = self.network.analyze_self_concept_emergence()
            self.show_brief_self_update(analysis)
            
        print(f"\n📊 Network now has {self.network.neuron_count} neurons")
        
    def show_brief_self_update(self, analysis):
        """Show brief self-concept update."""
        mass = analysis['self_concept_mass']
        indicators = analysis['self_awareness_indicators']
        
        print(f"   Self-concept mass: {mass:.2f}")
        if indicators and len(indicators) > 0:
            print(f"   {indicators[0]}")
            
    def show_self_concept_analysis(self):
        """Show detailed self-concept analysis."""
        print("\n🧠 SELF-CONCEPT ANALYSIS")
        print("="*40)
        analysis = self.network.analyze_self_concept_emergence()
        self.network.print_self_concept_analysis(analysis)
        
    def show_self_query(self):
        """Show what the system associates with itself."""
        print("\n🪞 SELF-ASSOCIATION QUERY")
        print("="*40)
        result = self.network.query_self_concept(activation_threshold=0.02)
        self.network.print_self_concept_query(result)
        
    def show_word_query(self, word):
        """Show associations for a specific word."""
        print(f"\n🔍 ASSOCIATIONS FOR '{word.upper()}'")
        print("="*40)
        result = self.network.query_associations(word, activation_threshold=0.02)
        self.network.print_query_result(result)
        
    def show_context_query(self, phrase):
        """Show context query for a phrase."""
        print(f"\n🔗 CONTEXT QUERY: '{phrase.upper()}'")
        print("="*40)
        result = self.network.query_context(phrase, activation_threshold=0.02)
        self.network.print_context_query_result(result)
        
    def show_network_stats(self):
        """Show current network statistics."""
        print("\n📊 NETWORK STATISTICS")
        print("="*40)
        print(f"Total neurons: {self.network.neuron_count}/{self.network.max_neurons}")
        print(f"Total connections: {len(self.network.connections)}")
        print(f"Total inertial masses: {len(self.network.inertial_mass)}")
        print(f"Windows processed: {self.network.processed_words}")
        print(f"Conversation turns: {self.turn_count}")
        print(f"Session duration: {time.time() - self.session_start_time:.1f} seconds")
        
        # Show most active neurons
        if self.network.activations:
            sorted_activations = sorted(
                [(self.network.neuron_to_word.get(idx, f"neuron_{idx}"), activation) 
                 for idx, activation in self.network.activations.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            print(f"\nMost active neurons:")
            for word, activation in sorted_activations:
                print(f"  {word}: {activation:.3f}")
                
    def show_conversation_history(self):
        """Show conversation history."""
        print("\n📜 CONVERSATION HISTORY")
        print("="*40)
        
        if not self.conversation_history:
            print("No conversation history yet.")
            return
            
        for entry in self.conversation_history[-10:]:  # Show last 10 turns
            timestamp = entry['timestamp']
            print(f"Turn {entry['turn']} ({timestamp:.1f}s): {entry['input'][:50]}...")
            
    def show_help(self):
        """Show help message."""
        print("\n❓ AVAILABLE COMMANDS")
        print("="*40)
        print("  quit/exit      - End the conversation")
        print("  analyze        - Show detailed self-concept analysis") 
        print("  self           - Query what system associates with itself")
        print("  query <word>   - Show associations for a specific word")
        print("  context <text> - Show emergent associations from phrase")
        print("  stats          - Show network statistics")
        print("  history        - Show recent conversation history") 
        print("  help           - Show this help message")
        print("\n💡 TIP: Use system-directed language like 'you' and 'your'")
        print("   to strengthen the system's self-concept!")
        
    def end_conversation(self):
        """End the conversation and show summary."""
        print("\n👋 ENDING CONVERSATION")
        print("="*40)
        
        duration = time.time() - self.session_start_time
        print(f"Session duration: {duration:.1f} seconds")
        print(f"Total turns: {self.turn_count}")
        print(f"Windows processed: {self.network.processed_words}")
        print(f"Final neuron count: {self.network.neuron_count}/{self.network.max_neurons}")
        
        # Show final self-concept state
        if self.turn_count > 0:
            analysis = self.network.analyze_self_concept_emergence()
            if analysis['self_concept_mass'] > 0:
                print(f"\nFinal self-concept mass: {analysis['self_concept_mass']:.2f}")
                if analysis['self_awareness_indicators']:
                    print("Self-awareness indicators:")
                    for indicator in analysis['self_awareness_indicators'][:3]:
                        print(f"  • {indicator}")
        
        print("\n✨ Thanks for the conversation!")


def demo_interactive_conversation():
    """Demo function showing how to use the interactive conversation system."""
    
    # Option 1: Create with new network
    print("🚀 Starting interactive conversation with new network...")
    conversation = InteractiveConversation(window_size=3, max_neurons=30)
    
    # Option 2: Or use with existing network
    # existing_net = ExperimentalNetwork(window_size=3, max_neurons=25)
    # conversation = InteractiveConversation(network=existing_net)
    
    conversation.start_conversation()


if __name__ == "__main__":
    demo_interactive_conversation()