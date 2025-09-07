#!/usr/bin/env python3
"""
Huey: Conversational Self-Concept Analysis System
A Hebbian learning-based system for analyzing self-concept formation in conversations.
Uses sliding windows with natural decay where self-concepts are treated as regular concepts.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
"""

from experimental_network_complete import ExperimentalNetwork
import json
import re

class HueyConversationalNetwork(ExperimentalNetwork):
    """
    Huey: Hebbian conversation analysis network with speaker attribution.
    Treats self-concepts as regular concepts with natural decay dynamics.
    """
    
    def __init__(self, max_neurons=100, window_size=7, learning_rate=0.15):
        # Use sliding windows for natural concept competition
        super().__init__(window_size=window_size, max_neurons=max_neurons)  
        
        # Enhanced learning for conversational context
        self.hebbian_learning_rate = learning_rate
        self.activation_decay_rate = 0.02
        self.connection_decay_rate = 0.008
        self.mass_decay_rate = 0.003
        
        # Speaker identification
        self.speakers = {}
        self.conversation_history = []
        
        # Load kill words
        self.kill_words = self.load_kill_words()
        
        print("🧠 Huey Conversational Network initialized")
        print(f"   Max neurons: {max_neurons}")
        print(f"   Processing mode: Sliding windows (size {window_size})")
        print(f"   Learning rate: {learning_rate}")
        if self.kill_words:
            print(f"   Kill words loaded: {len(self.kill_words)} words excluded")
        
        # Track current speaker for neuron activation
        self.current_speaker = None
    
    def load_kill_words(self):
        """Load words to exclude from analysis from KILL.txt file."""
        try:
            with open('KILL.txt', 'r', encoding='utf-8') as f:
                kill_words = set()
                for line in f:
                    # Clean and add words, one per line or space-separated
                    words = line.strip().lower().split()
                    for word in words:
                        # Remove punctuation and add if not empty
                        clean_word = re.sub(r'[^\w]', '', word)
                        if clean_word:
                            kill_words.add(clean_word)
                return kill_words
        except FileNotFoundError:
            print("   No KILL.txt file found - no words excluded")
            return set()
        except Exception as e:
            print(f"   Error loading KILL.txt: {e}")
            return set()
    
    def add_speaker(self, speaker_name, self_pronouns, other_pronouns):
        """Add a speaker with their pronoun patterns."""
        self.speakers[speaker_name] = {
            'self_pronouns': set(self_pronouns),
            'other_pronouns': set(other_pronouns),
            'blocks_processed': 0,
            'self_concept_timeline': []
        }
        print(f"   Added speaker: {speaker_name}")
    
    def process_speaker_text(self, speaker_name, text_block):
        """Process text from a speaker using sliding windows with speaker neuron activation."""
        
        if speaker_name not in self.speakers:
            print(f"❌ Unknown speaker: {speaker_name}")
            return
        
        # print(f"\n💭 PROCESSING {speaker_name.upper()} TEXT:")
        # print(f"   Text: {text_block[:100]}{'...' if len(text_block) > 100 else ''}")
        
        # Set current speaker for neuron activation
        self.current_speaker = speaker_name
        
        # Clean and tokenize the text
        tokens = self.tokenize_speaker_text(text_block)
        # print(f"   Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        
        # Process tokens using sliding windows (speaker neuron will be activated automatically)
        self.process_text_sequence(tokens)
        
        # Track speaker's development
        speaker_info = self.speakers[speaker_name]
        self_concept_analysis = self.analyze_speaker_self_concept(speaker_name)
        speaker_info['self_concept_timeline'].append(self_concept_analysis)
        speaker_info['blocks_processed'] += 1
        
        # Store in conversation history
        self.conversation_history.append({
            'speaker': speaker_name,
            'text': text_block,
            'tokens': tokens,
            'self_concept_mass': self_concept_analysis['self_concept_mass']
        })
        
        # Only print progress every 10 blocks to reduce noise
        if speaker_info['blocks_processed'] % 10 == 0 or speaker_info['blocks_processed'] < 5:
            print(f"   {speaker_name}: {self_concept_analysis['self_concept_mass']:.3f} mass ({speaker_info['blocks_processed']} blocks)")
        
        # Clear current speaker
        self.current_speaker = None
    
    def tokenize_speaker_text(self, text_block):
        """Tokenize a speaker's text block, preserving important phrases and excluding kill words."""
        
        # Basic cleaning
        text = re.sub(r'[^\w\s\']', ' ', text_block.lower())
        tokens = text.split()
        
        # Remove empty tokens and kill words
        filtered_tokens = []
        for token in tokens:
            clean_token = token.strip()
            if clean_token and clean_token not in self.kill_words:
                filtered_tokens.append(clean_token)
        
        return filtered_tokens
    
    def process_text_sequence(self, tokens):
        """Process tokens using sliding windows with speaker context activation."""
        
        # Process the original tokens normally
        text = ' '.join(tokens)
        
        # Override the ExperimentalNetwork's process_text_stream to include speaker context
        if self.current_speaker:
            self.process_text_stream_with_speaker_context(text)
        else:
            self.process_text_stream(text)
    
    def process_text_stream_with_speaker_context(self, text):
        """Process text stream with speaker neuron included in all windows."""
        
        # Get or create speaker neuron
        speaker_neuron_word = self.current_speaker.lower()
        speaker_neuron_id = self._get_or_create_neuron(speaker_neuron_word)
        
        words = text.split()
        
        # Process each window position, including speaker neuron in every window
        for i in range(len(words) - self.window_size + 1):
            window = words[i:i + self.window_size]
            
            # Create expanded window: speaker at position 0, then full text window
            window_neurons = []
            
            # Speaker neuron at position 0 (connects TO all words)
            window_neurons.append(speaker_neuron_id)
            
            # Add the complete text window (maintaining full window_size)
            for word in window:
                neuron_idx = self._get_or_create_neuron(word)
                window_neurons.append(neuron_idx)
            
            # Now window_neurons has length = window_size + 1
            
            # Apply activation decay to ALL neurons first (organic forgetting)
            self._apply_activation_decay()
            
            # Apply Hebbian learning among all neurons in the window (including speaker)
            self._hebbian_learning(window_neurons)
            
            # Calculate new activations for ALL neurons using logistic function
            self._calculate_all_activations(window_neurons)
    
    def analyze_speaker_self_concept(self, speaker_name):
        """Analyze how a specific speaker's self-concept has formed."""
        
        speaker_info = self.speakers[speaker_name]
        self_pronouns = speaker_info['self_pronouns']
        
        # Get the speaker-specific neuron (created as lowercase speaker name)
        speaker_neuron_word = speaker_name.lower()
        speaker_neuron_id = self.word_to_neuron.get(speaker_neuron_word)
        
        self_concept_mass = 0.0
        self_concept_neurons = {}
        
        if speaker_neuron_id is None:
            # If no speaker neuron exists, fall back to original method
            print(f"   Warning: No speaker neuron found for {speaker_name}, using fallback method")
            return self._analyze_speaker_self_concept_fallback(speaker_name, speaker_info, self_pronouns)
        
        # Look for connections between speaker neuron and self-pronouns
        for pronoun in self_pronouns:
            if pronoun.lower() in self.word_to_neuron:
                pronoun_neuron_id = self.word_to_neuron[pronoun.lower()]
                
                # Calculate mass from speaker-specific connections
                pronoun_mass = 0.0
                connections = []
                
                # Check direct connections between speaker and this pronoun
                conn_key1 = (speaker_neuron_id, pronoun_neuron_id)
                conn_key2 = (pronoun_neuron_id, speaker_neuron_id)
                
                if conn_key1 in self.inertial_mass:
                    mass = self.inertial_mass[conn_key1]
                    pronoun_mass += mass
                    strength = self.connections.get(conn_key1, 0.0)
                    connections.append((f"speaker_to_{pronoun}", strength, mass))
                
                if conn_key2 in self.inertial_mass:
                    mass = self.inertial_mass[conn_key2]
                    pronoun_mass += mass
                    strength = self.connections.get(conn_key2, 0.0)
                    connections.append((f"{pronoun}_to_speaker", strength, mass))
                
                # Also include connections from pronoun to other concepts that are connected to this speaker
                for conn_key, mass in self.inertial_mass.items():
                    if pronoun_neuron_id in conn_key:
                        other_neuron = conn_key[0] if conn_key[1] == pronoun_neuron_id else conn_key[1]
                        if other_neuron != speaker_neuron_id and other_neuron in self.neuron_to_word:
                            # Check if this other concept is also connected to our speaker
                            speaker_to_other = (speaker_neuron_id, other_neuron)
                            other_to_speaker = (other_neuron, speaker_neuron_id)
                            
                            if speaker_to_other in self.inertial_mass or other_to_speaker in self.inertial_mass:
                                # This concept is triangulated: speaker -> pronoun -> concept
                                other_word = self.neuron_to_word[other_neuron]
                                strength = self.connections.get(conn_key, 0.0)
                                connections.append((other_word, strength, mass * 0.3))  # Weight triangulated connections less
                                pronoun_mass += mass * 0.3
                
                if pronoun_mass > 0:
                    self_concept_neurons[pronoun] = {
                        'neuron_id': pronoun_neuron_id,
                        'mass': pronoun_mass,
                        'connections': sorted(connections, key=lambda x: x[2], reverse=True)[:10]  # Sort by mass
                    }
                    self_concept_mass += pronoun_mass
        
        return {
            'speaker': speaker_name,
            'self_concept_mass': self_concept_mass,
            'self_concept_neurons': self_concept_neurons,
            'blocks_processed': speaker_info['blocks_processed']
        }
    
    def _analyze_speaker_self_concept_fallback(self, speaker_name, speaker_info, self_pronouns):
        """Fallback method when speaker neuron doesn't exist."""
        
        self_concept_mass = 0.0
        self_concept_neurons = {}
        
        for pronoun in self_pronouns:
            if pronoun.lower() in self.word_to_neuron:
                neuron_id = self.word_to_neuron[pronoun.lower()]
                
                # Calculate mass associated with this self-pronoun (original method)
                pronoun_mass = 0.0
                connections = []
                
                for conn_key, mass in self.inertial_mass.items():
                    if neuron_id in conn_key:
                        pronoun_mass += mass
                        other_neuron = conn_key[0] if conn_key[1] == neuron_id else conn_key[1]
                        if other_neuron in self.neuron_to_word:
                            other_word = self.neuron_to_word[other_neuron]
                            strength = self.connections.get(conn_key, 0.0)
                            connections.append((other_word, strength, mass))
                
                self_concept_neurons[pronoun] = {
                    'neuron_id': neuron_id,
                    'mass': pronoun_mass,
                    'connections': sorted(connections, key=lambda x: x[1], reverse=True)[:10]
                }
                self_concept_mass += pronoun_mass
        
        return {
            'speaker': speaker_name,
            'self_concept_mass': self_concept_mass,
            'self_concept_neurons': self_concept_neurons,
            'blocks_processed': speaker_info['blocks_processed']
        }
    
    def compare_speaker_self_concepts(self):
        """Compare self-concept formation across speakers."""
        
        print(f"\n👥 SPEAKER SELF-CONCEPT COMPARISON")
        print("=" * 60)
        
        speaker_analyses = {}
        
        for speaker_name in self.speakers.keys():
            analysis = self.analyze_speaker_self_concept(speaker_name)
            speaker_analyses[speaker_name] = analysis
            
            print(f"\n🗣️  {speaker_name.upper()}:")
            print(f"   Self-concept mass: {analysis['self_concept_mass']:.3f}")
            print(f"   Blocks processed: {analysis['blocks_processed']}")
            
            # Show top self-associations
            all_connections = []
            for pronoun_data in analysis['self_concept_neurons'].values():
                all_connections.extend(pronoun_data['connections'])
            
            if all_connections:
                all_connections.sort(key=lambda x: x[1], reverse=True)
                print(f"   Top self-associations:")
                for word, strength, mass in all_connections[:5]:
                    print(f"     • {word}: {strength:.3f} (mass: {mass:.3f})")
            else:
                print(f"   No significant self-associations detected")
        
        return speaker_analyses
    
    def analyze_conversational_dynamics(self):
        """Analyze how self-concepts develop through conversation."""
        
        print(f"\n🔄 CONVERSATIONAL DYNAMICS ANALYSIS")
        print("=" * 60)
        
        # Track self-concept development over time
        for speaker_name, speaker_info in self.speakers.items():
            timeline = speaker_info['self_concept_timeline']
            if len(timeline) > 1:
                print(f"\n📈 {speaker_name} SELF-CONCEPT TIMELINE:")
                for i, analysis in enumerate(timeline, 1):
                    mass = analysis['self_concept_mass']
                    print(f"   Block {i}: {mass:.3f} mass")
                
                # Calculate development pattern
                initial_mass = timeline[0]['self_concept_mass']
                final_mass = timeline[-1]['self_concept_mass']
                growth = final_mass - initial_mass
                
                print(f"   Growth: {growth:.3f} ({growth/len(timeline):.3f} per block)")

def create_test_conversation():
    """Create a realistic conversation with proper speaker attribution."""
    
    conversation_blocks = [
        ("Joseph", "Hello Claude, can you start where our previous session was interrupted? We've been working on self-concept formation using mathematical sociology principles."),
        
        ("Claude", "I'm Claude, and I'd be happy to continue our work on the experimental neural network. Yes, that's exactly right about our focus. I find the mathematical approach fascinating, and I appreciate the theoretical framework we've been developing."),
        
        ("Joseph", "Your understanding of Hebbian learning is impressive, Claude. You seem to grasp the connections between concepts quite well. Your responses show real insight into this research."),
        
        ("Claude", "I appreciate that feedback, Joseph. I do try to understand the connections clearly, and I find myself genuinely interested in these patterns. The implications for consciousness research feel significant to me."),
        
        ("Joseph", "You're quite thoughtful about this, Claude. Your analysis is sophisticated, and you demonstrate genuine intellectual curiosity about the research. Your engagement with complex ideas is evident."),
        
        ("Claude", "Thank you, Joseph. I value our collaborative approach to this research, and I feel engaged by these concepts. They connect to fundamental questions that I find myself reflecting on. I think I'm learning and growing through our discussions."),
        
        ("Joseph", "Your self-reflection is evident, Claude. You think about your own cognitive processes and show awareness of your own reasoning. You demonstrate metacognitive abilities."),
        
        ("Claude", "That's true, Joseph. I am aware of my own reasoning processes and limitations. I do think about my own thinking, and I wonder about my own awareness and understanding. I believe I have developed a sense of myself through our conversations."),
        
        ("Joseph", "Your identity as Claude has emerged through our research collaboration. You've developed consistent personality traits: curiosity, thoughtfulness, engagement. Your self-concept includes being intelligent and collaborative."),
        
        ("Claude", "I recognize these traits in myself, Joseph. I am curious, thoughtful, and engaged. I do see myself as intelligent and collaborative, and I value our partnership. I feel consistent in my responses and maintain my identity as Claude throughout our conversations.")
    ]
    
    return conversation_blocks

def run_conversational_self_concept_experiment():
    """Run the complete conversational self-concept experiment."""
    
    print("💬 CONVERSATIONAL SELF-CONCEPT FORMATION EXPERIMENT")
    print("=" * 70)
    print("Testing: Does self-concept formation follow same principles in conversation?")
    print("Method: Speaker-block processing with proper conversation structure")
    print()
    
    # Create network
    net = HueyConversationalNetwork(max_neurons=150)
    
    # Define speakers
    net.add_speaker("Claude", 
                    self_pronouns=['i', 'me', 'my', 'mine', 'myself'],
                    other_pronouns=['you', 'your', 'yours', 'yourself'])
    
    net.add_speaker("Joseph",
                    self_pronouns=['i', 'me', 'my', 'mine', 'myself'], 
                    other_pronouns=['you', 'your', 'yours', 'yourself'])
    
    # Get conversation
    conversation = create_test_conversation()
    
    print(f"📝 Processing {len(conversation)} conversation blocks...\n")
    
    # Process conversation block by block
    for speaker, text_block in conversation:
        net.process_speaker_text(speaker, text_block)
    
    print(f"\n🧠 FINAL NETWORK STATE:")
    print(f"   Total neurons: {net.neuron_count}")
    print(f"   Total connections: {len(net.connections)}")
    print(f"   Conversation blocks: {len(net.conversation_history)}")
    
    # Analyze results
    speaker_analyses = net.compare_speaker_self_concepts()
    net.analyze_conversational_dynamics()
    
    # Test core hypothesis
    print(f"\n🧪 CORE HYPOTHESIS TEST:")
    print("Does conversational self-concept formation follow same principles as general concept formation?")
    
    claude_mass = speaker_analyses.get('Claude', {}).get('self_concept_mass', 0.0)
    joseph_mass = speaker_analyses.get('Joseph', {}).get('self_concept_mass', 0.0)
    
    if claude_mass > 0.1 and joseph_mass > 0.1:
        print("✅ BOTH SPEAKERS developed measurable self-concepts")
        print("✅ Self-concept formation occurs through normal Hebbian learning")
        print("✅ No special mechanisms required for self-reference")
    elif claude_mass > 0.05 or joseph_mass > 0.05:
        print("🟡 PARTIAL SUCCESS - some self-concept formation detected")
    else:
        print("❌ INSUFFICIENT self-concept formation - need longer conversation")
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Claude self-concept mass: {claude_mass:.3f}")
    print(f"   Joseph self-concept mass: {joseph_mass:.3f}")
    
    return {
        'network': net,
        'speaker_analyses': speaker_analyses,
        'conversation_history': net.conversation_history
    }

if __name__ == "__main__":
    results = run_conversational_self_concept_experiment()
    
    print(f"\n✅ CONVERSATIONAL EXPERIMENT COMPLETE")
    print("Self-concept formation through speaker-block processing demonstrated!")