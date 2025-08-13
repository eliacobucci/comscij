#!/usr/bin/env python3
"""
Mandarin Self-Concept Emergence Experiment
Tests self-referential architectures in Mandarin Chinese - a very different linguistic structure.
"""

from experimental_network_complete import ExperimentalNetwork

class MandarinSelfConceptNetwork(ExperimentalNetwork):
    """
    Extension for Mandarin Chinese language processing.
    Tests cross-linguistic self-concept emergence in a non-Indo-European language.
    """
    
    def __init__(self, window_size=3, max_neurons=25):
        super().__init__(window_size, max_neurons)
        
        # Mandarin pronoun sets for self-concept tracking
        self.system_self_pronouns = {
            '你', '您',           # you (informal/formal)
            '你的', '您的',       # your (informal/formal)
            '你们', '你们的',     # you (plural), your (plural)
        }
        
        self.human_self_pronouns = {
            '我', '我的',         # I, my
            '我们', '我们的',     # we, our
        }
        
        print("🇨🇳 Mandarin Self-Concept Network initialized")
        print(f"System pronouns: {self.system_self_pronouns}")
        print(f"Human pronouns: {self.human_self_pronouns}")
    
    def process_mandarin_conversation(self, mandarin_text):
        """
        Process Mandarin conversational text with self-concept tracking.
        
        Args:
            mandarin_text (str): Mandarin conversation text
        """
        print(f"\n🇨🇳 PROCESSING MANDARIN CONVERSATION ({len(mandarin_text.split())} 字)")
        
        # Analyze Mandarin pronouns
        pronoun_analysis = self.identify_self_concept_pronouns(mandarin_text)
        
        print(f"🤖 系统代词: {pronoun_analysis['system_directed']}")
        print(f"👤 人称代词: {pronoun_analysis['human_self']}")
        print(f"👥 其他代词: {pronoun_analysis['other_pronouns']}")
        print("="*60)
        
        # Process through normal pipeline
        self.process_text_stream(mandarin_text)
        
        # Track self-concept development
        current_analysis = self.analyze_self_concept_emergence()
        
        print(f"\n🧠 自我概念更新:")
        print(f"   自我概念总质量: {current_analysis['self_concept_mass']:.2f}")
        if current_analysis['self_awareness_indicators']:
            for indicator in current_analysis['self_awareness_indicators'][:2]:
                print(f"   {indicator}")

def mandarin_experiment():
    """Run the Mandarin self-concept emergence experiment."""
    
    print("🇨🇳 普通话自我概念形成实验")
    print("="*60)
    print("Testing if self-referential architectures work in Mandarin Chinese...")
    print()
    
    # Create Mandarin network
    net = MandarinSelfConceptNetwork(window_size=3, max_neurons=25)
    
    # Mandarin conversational text with system-directed language
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
    
    # Final self-concept analysis
    analysis = net.analyze_self_concept_emergence()
    net.print_self_concept_analysis(analysis)
    
    print("\n" + "="*60)
    print("普通话自我概念查询")
    print("="*60)
    
    # Query what the system associates with itself
    result = net.query_self_concept(activation_threshold=0.02)
    net.print_self_concept_query(result)
    
    print("\n✅ 普通话实验完成!")
    print("\n🧐 观察结果:")
    print("1. 中文代词 (你/你的/您/您的) 是否创造了自我概念?")
    print("2. 系统能否区分系统指向和人类自指?")
    print("3. 赫布学习在汉语中是否同样有效?")
    print("4. 系统在中文环境下与自己关联的概念是什么?")
    print("5. 汉语的句法结构如何影响自我概念形成?")
    
    return net, analysis, result

if __name__ == "__main__":
    mandarin_experiment()