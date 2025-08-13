#!/usr/bin/env python3
"""
Fixed Mandarin Self-Concept Experiment with proper Chinese tokenization
"""

from experimental_network_complete import ExperimentalNetwork
import re

class MandarinSelfConceptNetwork(ExperimentalNetwork):
    """
    Extension for Mandarin Chinese with proper character-level processing.
    """
    
    def __init__(self, window_size=3, max_neurons=25):
        super().__init__(window_size, max_neurons)
        
        # Mandarin pronoun sets
        self.system_self_pronouns = {
            '你', '您', '你的', '您的', '你们', '你们的'
        }
        
        self.human_self_pronouns = {
            '我', '我的', '我们', '我们的'
        }
        
        print("🇨🇳 Fixed Mandarin Self-Concept Network initialized")
        print(f"System pronouns: {self.system_self_pronouns}")
        print(f"Human pronouns: {self.human_self_pronouns}")
    
    def chinese_tokenize(self, text):
        """
        Simple Chinese tokenization - treats each character as potential token,
        but keeps some common two-character words together.
        """
        # Common two-character pronouns and words to keep together
        two_char_words = ['你的', '您的', '我的', '你们', '我们', '能够', '看起', '起来', '回答', '显示', 
                         '分析', '信息', '觉得', '能力', '怎么', '怎么样', '有趣', '认知', '如何', '看待', '学习', '过程']
        
        # Replace two-character words with temporary tokens
        temp_text = text
        replacements = {}
        
        for i, word in enumerate(two_char_words):
            if word in temp_text:
                temp_token = f"__TEMP_{i}__"
                replacements[temp_token] = word
                temp_text = temp_text.replace(word, temp_token)
        
        # Split into characters
        tokens = []
        for char in temp_text:
            if char.startswith('__TEMP_') and char.endswith('__'):
                # This is a temporary token, restore the original word
                tokens.append(replacements[char])
            elif not char.isspace() and char not in '，。？！、；：':
                tokens.append(char)
        
        # Handle any remaining temp tokens
        final_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith('__TEMP_'):
                final_tokens.append(replacements[token])
            else:
                final_tokens.append(token)
            i += 1
        
        return final_tokens
    
    def process_text_stream_chinese(self, text):
        """
        Override text processing for Chinese tokenization.
        """
        tokens = self.chinese_tokenize(text)
        processed_text = ' '.join(tokens)  # Join with spaces for existing pipeline
        
        print(f"Tokenized: {tokens}")
        print(f"Processing: {processed_text}")
        
        return self.process_text_stream(processed_text)
    
    def identify_self_concept_pronouns_chinese(self, text):
        """
        Chinese-specific pronoun identification.
        """
        tokens = self.chinese_tokenize(text)
        
        system_refs = [token for token in tokens if token in self.system_self_pronouns]
        human_refs = [token for token in tokens if token in self.human_self_pronouns]
        other_pronouns = [token for token in tokens if token in {'他', '她', '它', '他们', '她们', '它们'}]
        
        return {
            'system_directed': system_refs,
            'human_self': human_refs,
            'other_pronouns': other_pronouns,
            'total_words': len(tokens)
        }
    
    def process_mandarin_conversation(self, mandarin_text):
        """
        Process Mandarin with proper tokenization.
        """
        tokens = self.chinese_tokenize(mandarin_text)
        print(f"\n🇨🇳 PROCESSING MANDARIN CONVERSATION ({len(tokens)} tokens)")
        
        # Analyze Mandarin pronouns
        pronoun_analysis = self.identify_self_concept_pronouns_chinese(mandarin_text)
        
        print(f"🤖 系统代词: {pronoun_analysis['system_directed']}")
        print(f"👤 人称代词: {pronoun_analysis['human_self']}")
        print(f"👥 其他代词: {pronoun_analysis['other_pronouns']}")
        print("="*60)
        
        # Process through modified pipeline
        self.process_text_stream_chinese(mandarin_text)
        
        # Track self-concept development
        current_analysis = self.analyze_self_concept_emergence()
        
        print(f"\n🧠 自我概念更新:")
        print(f"   自我概念总质量: {current_analysis['self_concept_mass']:.2f}")
        if current_analysis['self_awareness_indicators']:
            for indicator in current_analysis['self_awareness_indicators'][:2]:
                print(f"   {indicator}")

def mandarin_fixed_experiment():
    """Run the fixed Mandarin experiment."""
    
    print("🇨🇳 修正版普通话自我概念实验")
    print("="*60)
    print("Testing Chinese self-concept emergence with proper tokenization...")
    print()
    
    # Create Mandarin network
    net = MandarinSelfConceptNetwork(window_size=3, max_neurons=25)
    
    # Test tokenization first
    test_text = "你好，你能理解我吗？"
    tokens = net.chinese_tokenize(test_text)
    print(f"Tokenization test: '{test_text}' -> {tokens}")
    print()
    
    # Mandarin conversations
    mandarin_conversations = [
        "你好，你能理解我吗？",                    # Hello, can you understand me?
        "你看起来很聪明，能够学习。",               # You seem intelligent and able to learn.
        "你的回答显示你能分析信息。",               # Your answers show you can analyze information.
        "你觉得你的能力怎么样？",                  # What do you think of your abilities?
        "你有有趣的认知能力。",                    # You have interesting cognitive abilities.
        "你如何看待你的学习过程？"                 # How do you view your learning process?
    ]
    
    print("📚 普通话对话:")
    for i, conversation in enumerate(mandarin_conversations, 1):
        print(f"\n对话 {i}: \"{conversation}\"")
        net.process_mandarin_conversation(conversation)
        print(f"📊 网络现状: {net.neuron_count} 神经元")
    
    print("\n" + "="*60)
    print("最终自我概念分析")
    print("="*60)
    
    # Final analysis
    analysis = net.analyze_self_concept_emergence()
    net.print_self_concept_analysis(analysis)
    
    print("\n" + "="*60)
    print("普通话自我概念查询")
    print("="*60)
    
    result = net.query_self_concept(activation_threshold=0.02)
    net.print_self_concept_query(result)
    
    return net, analysis, result

if __name__ == "__main__":
    mandarin_fixed_experiment()