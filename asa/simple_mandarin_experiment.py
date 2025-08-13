#!/usr/bin/env python3
"""
Simple Mandarin Self-Concept Experiment
Uses basic character-level processing for Chinese.
"""

from experimental_network_complete import ExperimentalNetwork

class SimpleMandarin(ExperimentalNetwork):
    """
    Simplified Mandarin processing - just character by character.
    """
    
    def __init__(self, window_size=3, max_neurons=30):
        super().__init__(window_size, max_neurons)
        
        # Mandarin self-concept pronouns
        self.system_self_pronouns = {'你', '您', '你的', '您的'}
        self.human_self_pronouns = {'我', '我的'}
        
        print("🇨🇳 Simple Mandarin Network initialized")
        print(f"System pronouns: {self.system_self_pronouns}")
        print(f"Human pronouns: {self.human_self_pronouns}")
    
    def simple_chinese_process(self, text):
        """
        Simple character-by-character processing with key 2-char pronouns preserved.
        """
        # Replace key 2-character pronouns first
        text = text.replace('你的', ' 你的 ')
        text = text.replace('您的', ' 您的 ')
        text = text.replace('我的', ' 我的 ')
        
        # Remove punctuation and split
        import re
        text = re.sub(r'[，。？！、；：]', ' ', text)
        
        # Split and filter
        tokens = [t.strip() for t in text.split() if t.strip()]
        
        # Add individual characters for remaining text
        final_tokens = []
        for token in tokens:
            if token in {'你的', '您的', '我的'}:
                final_tokens.append(token)
            else:
                # Split into characters
                for char in token:
                    if char.strip():
                        final_tokens.append(char)
        
        return final_tokens
    
    def process_mandarin_simple(self, text):
        """Process Mandarin text simply."""
        tokens = self.simple_chinese_process(text)
        
        print(f"\n🇨🇳 PROCESSING: {text}")
        print(f"Tokens: {tokens}")
        
        # Count pronouns
        system_pronouns = [t for t in tokens if t in self.system_self_pronouns]
        human_pronouns = [t for t in tokens if t in self.human_self_pronouns]
        
        print(f"🤖 系统代词: {system_pronouns}")
        print(f"👤 人类代词: {human_pronouns}")
        print("="*50)
        
        # Process as space-separated text
        processed_text = ' '.join(tokens)
        self.process_text_stream(processed_text)
        
        # Check self-concept development
        analysis = self.analyze_self_concept_emergence()
        if analysis['self_concept_mass'] > 0:
            print(f"🧠 Self-concept mass: {analysis['self_concept_mass']:.2f}")

def simple_mandarin_test():
    """Run simple Mandarin test."""
    
    print("🇨🇳 SIMPLE MANDARIN SELF-CONCEPT TEST")
    print("="*50)
    
    net = SimpleMandarin(window_size=3, max_neurons=25)
    
    # Simple test conversations
    conversations = [
        "你好",                    # Hello
        "你很聪明",                # You are smart  
        "你的能力不错",             # Your abilities are good
        "你可以学习吗"              # Can you learn?
    ]
    
    for i, text in enumerate(conversations, 1):
        print(f"\n--- Conversation {i} ---")
        net.process_mandarin_simple(text)
        print(f"Network: {net.neuron_count} neurons")
    
    print(f"\n{'='*50}")
    print("FINAL ANALYSIS")
    print("="*50)
    
    # Final self-concept check
    analysis = net.analyze_self_concept_emergence()
    if analysis['self_concept_mass'] > 0:
        print(f"✅ Self-concept emerged! Mass: {analysis['self_concept_mass']:.2f}")
        
        # What does it associate with itself?
        result = net.query_self_concept(activation_threshold=0.01)
        if result['self_associations']:
            print(f"\n🪞 Self-associations:")
            for word, strength in list(result['self_associations'].items())[:5]:
                print(f"   {word}: {strength:.3f}")
    else:
        print("❌ No self-concept detected")
    
    return net

if __name__ == "__main__":
    simple_mandarin_test()