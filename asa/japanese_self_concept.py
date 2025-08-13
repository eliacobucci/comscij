#!/usr/bin/env python3
"""
Japanese Self-Concept Experiment
Testing self-concept emergence in Japanese with its complex honorific system.
Hypothesis: Japanese should show even more distributed self-concept than Mandarin.
"""

from experimental_network_complete import ExperimentalNetwork
from cross_linguistic_research_protocol import CrossLinguisticSelfConceptStudy

class JapaneseSelfConceptNetwork(ExperimentalNetwork):
    """
    Japanese-specific network with honorific-aware processing.
    """
    
    def __init__(self, window_size=3, max_neurons=25):
        super().__init__(window_size, max_neurons)
        
        # Japanese pronouns - complex honorific system
        self.system_self_pronouns = {
            'あなた',      # you (polite)
            'きみ',        # you (casual, to subordinate)
            'おまえ',      # you (very casual/rude)
            'あなたの',    # your (polite)
            'きみの',      # your (casual)
            '貴方',        # you (formal written)
            '君',          # you (casual written)
        }
        
        self.human_self_pronouns = {
            'わたし',      # I (polite)
            'わたくし',    # I (very polite/formal)
            'ぼく',        # I (male, casual)
            'おれ',        # I (male, very casual)
            'わたしの',    # my (polite)
            'ぼくの',      # my (male, casual)
            '私',          # I (written)
            '僕',          # I (male, written)
        }
        
        print("🇯🇵 Japanese Self-Concept Network initialized")
        print(f"System pronouns: {self.system_self_pronouns}")
        print(f"Human pronouns: {self.human_self_pronouns}")
    
    def japanese_tokenize(self, text):
        """
        Simple Japanese tokenization - preserve key pronouns, split others by character.
        """
        # Key multi-character units to preserve
        multi_char = ['あなた', 'あなたの', 'わたし', 'わたくし', 'わたしの', 
                     'きみの', 'ぼくの', 'です', 'ます', 'できる', 'できます',
                     'おもう', 'おもいます', 'のうりょく', 'ちのう']
        
        # Replace with temporary tokens
        temp_text = text
        replacements = {}
        
        for i, unit in enumerate(multi_char):
            if unit in temp_text:
                temp_token = f"__{i}__"
                replacements[temp_token] = unit
                temp_text = temp_text.replace(unit, f" {temp_token} ")
        
        # Split and process
        tokens = []
        for part in temp_text.split():
            if part in replacements:
                tokens.append(replacements[part])
            elif part.startswith('__') and part.endswith('__'):
                tokens.append(replacements[part])
            else:
                # Split into characters, skip punctuation
                for char in part:
                    if char not in '、。？！　':
                        tokens.append(char)
        
        return [t for t in tokens if t.strip()]
    
    def process_japanese_conversation(self, japanese_text):
        """Process Japanese conversation with tokenization."""
        tokens = self.japanese_tokenize(japanese_text)
        
        print(f"\n🇯🇵 PROCESSING: {japanese_text}")
        print(f"Tokens: {tokens}")
        
        # Count pronouns
        system_pronouns = [t for t in tokens if t in self.system_self_pronouns]
        human_pronouns = [t for t in tokens if t in self.human_self_pronouns]
        
        print(f"🤖 System pronouns: {system_pronouns}")
        print(f"👤 Human pronouns: {human_pronouns}")
        print("="*50)
        
        # Process as space-separated text
        processed_text = ' '.join(tokens)
        self.process_text_stream(processed_text)
        
        # Check self-concept development
        analysis = self.analyze_self_concept_emergence()
        if analysis['self_concept_mass'] > 0:
            print(f"🧠 Self-concept mass: {analysis['self_concept_mass']:.3f}")

def japanese_experiment():
    """Run Japanese self-concept experiment."""
    
    print("🇯🇵 JAPANESE SELF-CONCEPT EXPERIMENT")
    print("="*60)
    print("Testing honorific system effects on self-concept formation...")
    print()
    
    # Create study framework
    study = CrossLinguisticSelfConceptStudy()
    
    # Japanese pronouns for study
    japanese_pronouns = {
        'system': ['あなた', 'きみ', 'あなたの', 'きみの', '貴方', '君'],
        'human': ['わたし', 'わたくし', 'ぼく', 'おれ', 'わたしの', 'ぼくの', '私', '僕']
    }
    
    # Japanese conversations with honorific variations
    japanese_conversations = [
        "こんにちは、わたしをりかいできますか？",           # Hello, can you understand me?
        "あなたはとてもちのうてきですね。",               # You are very intelligent.
        "あなたのこたえはじょうほうをぶんせきできることをしめしています。", # Your answers show you can analyze information.
        "あなたはあなたののうりょくについてどうおもいますか？", # What do you think about your abilities?
        "あなたはおもしろいちてきのうりょくをもっています。"   # You have interesting cognitive abilities.
    ]
    
    # Test using study framework
    results = study.test_language("Japanese", japanese_pronouns, japanese_conversations)
    
    print(f"\n🎌 JAPANESE CULTURAL HYPOTHESIS TEST:")
    print(f"Expected: Lower individual self-strength than Western languages")
    print(f"Expected: Higher distribution score (more spread self-concept)")
    print(f"Observed self-strength: {results['individual_self_strength']:.3f}")
    print(f"Observed distribution: {results['distributed_score']}")
    
    return study, results

def compare_east_west():
    """Compare Japanese vs Western patterns directly."""
    
    print("\n" + "="*60)
    print("EAST-WEST SELF-CONCEPT COMPARISON")  
    print("="*60)
    
    study = CrossLinguisticSelfConceptStudy()
    
    # Test English (individualistic)
    english_pronouns = {
        'system': ['you', 'your', 'yours', 'yourself'],
        'human': ['i', 'me', 'my', 'mine', 'myself']
    }
    english_conversations = [
        "Hello, can you understand me?",
        "You are very intelligent.",
        "Your responses show you can analyze information.",
        "What do you think about your abilities?",
        "You have interesting cognitive capabilities."
    ]
    
    english_results = study.test_language("English", english_pronouns, english_conversations)
    
    # Test Japanese (collectivistic)  
    japanese_pronouns = {
        'system': ['あなた', 'あなたの'],  # Simplified for comparison
        'human': ['わたし', 'わたしの']
    }
    japanese_conversations = [
        "こんにちは りかいできますか",
        "あなた は ちのう てき です",
        "あなたの こたえ は ぶんせき できる",
        "あなた の のうりょく どう おもう",
        "あなた は ちてき のうりょく もつ"
    ]
    
    japanese_results = study.test_language("Japanese", japanese_pronouns, japanese_conversations)
    
    # Direct comparison
    print(f"\n📊 DIRECT COMPARISON:")
    print(f"{'Metric':<25} {'English':<12} {'Japanese':<12} {'Ratio':<12}")
    print("-" * 60)
    
    metrics = [
        ('individual_self_strength', 'Self-strength'),
        ('self_association_count', 'Self-associations'),
        ('distributed_score', 'Distribution'),
    ]
    
    for metric, label in metrics:
        eng_val = english_results[metric]
        jpn_val = japanese_results[metric]
        ratio = eng_val / jpn_val if jpn_val > 0 else "inf"
        
        print(f"{label:<25} {eng_val:<12.3f} {jpn_val:<12.3f} {ratio:<12}")
    
    # Cultural hypothesis evaluation
    print(f"\n🧪 HYPOTHESIS TEST:")
    if english_results['individual_self_strength'] > japanese_results['individual_self_strength'] * 1.2:
        print("✅ HYPOTHESIS SUPPORTED: English shows stronger individual self-concept")
    else:
        print("❌ HYPOTHESIS NOT SUPPORTED: Similar or reversed pattern")
    
    return study

if __name__ == "__main__":
    # Run Japanese experiment
    study, results = japanese_experiment()
    
    print("\n" + "="*60)
    print("Running East-West comparison...")
    compare_east_west()