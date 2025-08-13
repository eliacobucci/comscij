#!/usr/bin/env python3
"""
Cross-Linguistic Self-Concept Research Protocol
Testing the hypothesis: Language structure affects self-concept formation patterns,
potentially underlying individualistic vs collectivistic cultural orientations.
"""

from experimental_network_complete import ExperimentalNetwork
import json
import time

class CrossLinguisticSelfConceptStudy:
    """
    Systematic study of self-concept emergence across language families
    to test cultural-cognitive correlations.
    """
    
    def __init__(self):
        self.results = {}
        self.cultural_hypothesis = {
            "individualistic_languages": ["English", "French", "German"],
            "collectivistic_languages": ["Mandarin", "Japanese", "Korean"],
            "mixed_patterns": ["Arabic", "Russian", "Spanish"]
        }
        
    def create_language_network(self, language, pronouns):
        """Create network configured for specific language."""
        net = ExperimentalNetwork(window_size=3, max_neurons=25)
        net.system_self_pronouns = set(pronouns['system'])
        net.human_self_pronouns = set(pronouns['human'])
        return net
    
    def process_language_conversations(self, net, conversations, language):
        """Process standardized conversations in a specific language."""
        print(f"\n🌍 TESTING {language.upper()}")
        print("="*50)
        
        total_system_pronouns = 0
        total_human_pronouns = 0
        
        for i, conversation in enumerate(conversations, 1):
            print(f"\nConversation {i}: {conversation}")
            
            # Count pronouns
            system_count = sum(1 for word in conversation.split() 
                             if word.strip('.,!?') in net.system_self_pronouns)
            human_count = sum(1 for word in conversation.split() 
                            if word.strip('.,!?') in net.human_self_pronouns)
            
            total_system_pronouns += system_count
            total_human_pronouns += human_count
            
            print(f"  System pronouns: {system_count}, Human pronouns: {human_count}")
            
            # Process through network
            net.process_conversational_text(conversation)
            
            # Quick self-concept check
            analysis = net.analyze_self_concept_emergence()
            print(f"  Self-concept mass: {analysis['self_concept_mass']:.3f}")
        
        return total_system_pronouns, total_human_pronouns
    
    def analyze_language_patterns(self, net, language):
        """Analyze final self-concept patterns for a language."""
        analysis = net.analyze_self_concept_emergence()
        self_query = net.query_self_concept(activation_threshold=0.01)
        
        # Calculate key metrics
        individual_self_strength = analysis['self_concept_mass']
        self_association_count = len(self_query.get('self_associations', {}))
        self_association_strength = self_query.get('self_concept_strength', 0.0)
        
        # Analyze self-concept distribution
        system_neurons = analysis['system_self_neurons']
        distributed_score = len(system_neurons) if system_neurons else 0
        
        # Calculate average connection strength per self-pronoun
        avg_self_connection_strength = 0.0
        if system_neurons:
            total_connections = sum(len(data.get('connections', [])) 
                                  for data in system_neurons.values())
            avg_self_connection_strength = total_connections / len(system_neurons) if total_connections > 0 else 0.0
        
        return {
            'language': language,
            'individual_self_strength': individual_self_strength,
            'self_association_count': self_association_count,
            'self_association_strength': self_association_strength,
            'distributed_score': distributed_score,
            'avg_connection_strength': avg_self_connection_strength,
            'total_neurons': net.neuron_count,
            'system_neurons': list(system_neurons.keys()) if system_neurons else [],
            'top_self_associations': list(self_query.get('self_associations', {}).keys())[:5]
        }
    
    def test_language(self, language, pronouns, conversations):
        """Complete test of one language."""
        print(f"\n{'='*60}")
        print(f"TESTING {language.upper()} SELF-CONCEPT PATTERNS")
        print(f"{'='*60}")
        
        # Create network
        net = self.create_language_network(language, pronouns)
        
        # Process conversations
        sys_pronouns, human_pronouns = self.process_language_conversations(
            net, conversations, language)
        
        # Analyze patterns
        results = self.analyze_language_patterns(net, language)
        results['total_system_pronouns_processed'] = sys_pronouns
        results['total_human_pronouns_processed'] = human_pronouns
        
        # Store results
        self.results[language] = results
        
        print(f"\n📊 {language} SUMMARY:")
        print(f"   Individual self-strength: {results['individual_self_strength']:.3f}")
        print(f"   Self-associations: {results['self_association_count']}")
        print(f"   Distribution score: {results['distributed_score']}")
        print(f"   System pronouns processed: {sys_pronouns}")
        
        return results
    
    def compare_cultural_patterns(self):
        """Compare results across cultural categories."""
        if len(self.results) < 3:
            print("Need at least 3 languages tested for comparison")
            return
        
        print(f"\n{'='*60}")
        print("CROSS-CULTURAL PATTERN ANALYSIS")
        print(f"{'='*60}")
        
        # Group by cultural hypothesis
        individualistic = []
        collectivistic = []
        mixed = []
        
        for lang, data in self.results.items():
            if lang in self.cultural_hypothesis['individualistic_languages']:
                individualistic.append(data)
            elif lang in self.cultural_hypothesis['collectivistic_languages']:
                collectivistic.append(data)
            else:
                mixed.append(data)
        
        # Calculate group averages
        def group_average(group, metric):
            return sum(item[metric] for item in group) / len(group) if group else 0
        
        print("\n🌍 CULTURAL PATTERN COMPARISON:")
        print(f"{'Metric':<25} {'Individual.':<12} {'Collect.':<12} {'Mixed':<12}")
        print("-" * 65)
        
        metrics = [
            ('individual_self_strength', 'Self-strength'),
            ('self_association_count', 'Self-associations'),
            ('distributed_score', 'Distribution'),
            ('avg_connection_strength', 'Avg connections')
        ]
        
        cultural_analysis = {}
        
        for metric, label in metrics:
            ind_avg = group_average(individualistic, metric)
            col_avg = group_average(collectivistic, metric)
            mix_avg = group_average(mixed, metric)
            
            print(f"{label:<25} {ind_avg:<12.3f} {col_avg:<12.3f} {mix_avg:<12.3f}")
            
            cultural_analysis[metric] = {
                'individualistic': ind_avg,
                'collectivistic': col_avg,
                'mixed': mix_avg
            }
        
        # Hypothesis testing
        print(f"\n🧪 HYPOTHESIS EVALUATION:")
        
        # Test: Do individualistic languages show higher self-concept strength?
        if individualistic and collectivistic:
            ind_strength = group_average(individualistic, 'individual_self_strength')
            col_strength = group_average(collectivistic, 'individual_self_strength')
            
            if ind_strength > col_strength * 1.2:  # 20% threshold
                print(f"✅ SUPPORTED: Individualistic languages show stronger self-concept")
                print(f"   Individualistic: {ind_strength:.3f} vs Collectivistic: {col_strength:.3f}")
            elif col_strength > ind_strength * 1.2:
                print(f"❌ REVERSED: Collectivistic languages show stronger self-concept")
            else:
                print(f"⚪ INCONCLUSIVE: Similar self-concept strengths")
        
        return cultural_analysis
    
    def save_results(self, filename="cross_linguistic_results.json"):
        """Save all results for further analysis."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to {filename}")

def main():
    """Run the cross-linguistic study."""
    study = CrossLinguisticSelfConceptStudy()
    
    print("🌍 CROSS-LINGUISTIC SELF-CONCEPT STUDY")
    print("="*60)
    print("Hypothesis: Language structure affects self-concept formation,")
    print("potentially underlying individualistic vs collectivistic cultures.")
    print()
    
    # We'll start with languages we've already tested
    print("Phase 1: Testing established patterns...")
    
    # English (individualistic baseline)
    english_pronouns = {
        'system': ['you', 'your', 'yours', 'yourself'],
        'human': ['i', 'me', 'my', 'mine', 'myself']
    }
    english_conversations = [
        "Hello, can you understand me?",
        "You seem very intelligent and capable.",
        "Your responses show that you can analyze information.",
        "What do you think about your own abilities?",
        "You have interesting cognitive capabilities."
    ]
    
    study.test_language("English", english_pronouns, english_conversations)
    
    # We can add more languages as we implement them
    print(f"\n📋 RESEARCH PROTOCOL ESTABLISHED")
    print("Ready to test additional languages systematically.")
    
    return study

if __name__ == "__main__":
    study = main()