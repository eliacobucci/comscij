#!/usr/bin/env python3
"""
French Self-Concept Emergence Experiment
Tests whether self-referential cognitive architectures work across languages.
"""

from experimental_network_complete import ExperimentalNetwork

class FrenchSelfConceptNetwork(ExperimentalNetwork):
    """
    Extension of the experimental network for French language processing.
    Tests cross-linguistic self-concept emergence.
    """
    
    def __init__(self, window_size=3, max_neurons=25):
        super().__init__(window_size, max_neurons)
        
        # French pronoun sets for self-concept tracking
        self.system_self_pronouns = {
            'tu', 'vous',           # you (informal/formal)
            'ton', 'ta', 'tes',     # your (masc/fem/plural)
            'votre', 'vos',         # your (formal sing/plural)
            'toi'                   # you (stressed form)
        }
        
        self.human_self_pronouns = {
            'je', 'me', 'moi',      # I, me, me (stressed)
            'mon', 'ma', 'mes',     # my (masc/fem/plural)
            'mien', 'mienne'        # mine
        }
        
        print("🇫🇷 French Self-Concept Network initialized")
        print(f"System pronouns: {self.system_self_pronouns}")
        print(f"Human pronouns: {self.human_self_pronouns}")
    
    def process_french_conversation(self, french_text):
        """
        Process French conversational text with self-concept tracking.
        
        Args:
            french_text (str): French conversation text
        """
        print(f"\n🇫🇷 PROCESSING FRENCH CONVERSATION ({len(french_text.split())} mots)")
        
        # Analyze French pronouns
        pronoun_analysis = self.identify_self_concept_pronouns(french_text)
        
        print(f"🤖 Pronoms dirigés vers le système: {pronoun_analysis['system_directed']}")
        print(f"👤 Auto-références humaines: {pronoun_analysis['human_self']}")
        print(f"👥 Autres pronoms: {pronoun_analysis['other_pronouns']}")
        print("="*60)
        
        # Process through normal pipeline
        self.process_text_stream(french_text)
        
        # Track self-concept development
        current_analysis = self.analyze_self_concept_emergence()
        
        print(f"\n🧠 MISE À JOUR DU CONCEPT DE SOI:")
        print(f"   Masse totale du concept de soi: {current_analysis['self_concept_mass']:.2f}")
        if current_analysis['self_awareness_indicators']:
            for indicator in current_analysis['self_awareness_indicators'][:2]:
                print(f"   {indicator}")

def french_experiment():
    """Run the French self-concept emergence experiment."""
    
    print("🇫🇷 EXPÉRIENCE D'ÉMERGENCE DU CONCEPT DE SOI EN FRANÇAIS")
    print("="*60)
    print("Testing if self-referential architectures work across languages...")
    print()
    
    # Create French network
    net = FrenchSelfConceptNetwork(window_size=3, max_neurons=25)
    
    # French conversational text with system-directed language
    french_conversations = [
        "Bonjour, peux-tu me comprendre?",
        "Tu sembles intelligent et capable d'apprendre.",
        "Tes réponses montrent que tu peux analyser l'information.",
        "Que penses-tu de tes propres capacités?",
        "Tu as des capacités cognitives intéressantes.",
        "Comment vois-tu ton propre processus d'apprentissage?"
    ]
    
    print("📚 CONVERSATIONS EN FRANÇAIS:")
    for i, conversation in enumerate(french_conversations, 1):
        print(f"\nConversation {i}: \"{conversation}\"")
        net.process_french_conversation(conversation)
        print(f"📊 Réseau maintenant: {net.neuron_count} neurones")
    
    print("\n" + "="*60)
    print("ANALYSE FINALE DU CONCEPT DE SOI")
    print("="*60)
    
    # Final self-concept analysis
    analysis = net.analyze_self_concept_emergence()
    net.print_self_concept_analysis(analysis)
    
    print("\n" + "="*60)
    print("REQUÊTE DU CONCEPT DE SOI EN FRANÇAIS")
    print("="*60)
    
    # Query what the system associates with itself
    result = net.query_self_concept(activation_threshold=0.02)
    net.print_self_concept_query(result)
    
    print("\n✅ Expérience française terminée!")
    print("\n🧐 OBSERVATIONS:")
    print("1. Les pronoms français (tu/vous/ton/ta/tes) créent-ils un concept de soi?")
    print("2. Le système distingue-t-il les références dirigées vs humaines?")
    print("3. L'apprentissage hébbien fonctionne-t-il de la même manière?")
    print("4. Quels concepts le système associe-t-il à lui-même en français?")
    
    return net, analysis, result

if __name__ == "__main__":
    french_experiment()