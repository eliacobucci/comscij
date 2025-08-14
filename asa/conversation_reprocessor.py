#!/usr/bin/env python3
"""
Conversation Reprocessor for Self-Concept Analysis
Reprocesses existing conversation files to properly identify speakers (Joseph, Claude, Asa, Evan)
and generates speaker-tagged datasets for accurate self-concept cluster analysis.
"""

from enhanced_speaker_identifier import EnhancedSpeakerIdentifier
from conversational_self_concept_experiment import HueyConversationalNetwork
import json
import glob
import os
from typing import Dict, List

class ConversationReprocessor:
    """
    Tool for reprocessing existing conversation data with proper speaker identification.
    """
    
    def __init__(self):
        self.identifier = EnhancedSpeakerIdentifier()
        self.processed_conversations = {}
        
        # Speaker pronoun mappings for self-concept analysis
        self.speaker_pronouns = {
            'Joseph': {
                'self_pronouns': ['i', 'me', 'my', 'mine', 'myself'],
                'other_pronouns': ['you', 'your', 'yours', 'yourself']
            },
            'Claude': {
                'self_pronouns': ['i', 'me', 'my', 'mine', 'myself'],
                'other_pronouns': ['you', 'your', 'yours', 'yourself']
            },
            'Asa': {
                'self_pronouns': ['i', 'me', 'my', 'mine', 'myself'],
                'other_pronouns': ['you', 'your', 'yours', 'yourself']
            },
            'Evan': {
                'self_pronouns': ['i', 'me', 'my', 'mine', 'myself'], 
                'other_pronouns': ['you', 'your', 'yours', 'yourself']
            }
        }
        
        print("🔄 Conversation Reprocessor initialized")
    
    def find_conversation_files(self, directory: str = "/Users/josephwoelfel/asa/") -> List[str]:
        """Find potential conversation files to reprocess."""
        
        # Look for text files that might contain conversations
        patterns = [
            "*.txt", "*.md", "conversation*.py", "*chat*.py", 
            "*dialogue*.py", "*talk*.py"
        ]
        
        conversation_files = []
        for pattern in patterns:
            files = glob.glob(os.path.join(directory, pattern))
            conversation_files.extend(files)
        
        # Filter out obvious non-conversation files
        exclude_patterns = ['README', 'requirements', 'setup', 'config', '__']
        conversation_files = [f for f in conversation_files 
                             if not any(exclude in os.path.basename(f) for exclude in exclude_patterns)]
        
        print(f"🔍 Found {len(conversation_files)} potential conversation files")
        for f in conversation_files[:10]:  # Show first 10
            print(f"   {os.path.basename(f)}")
        
        return conversation_files
    
    def reprocess_conversation_file(self, filename: str) -> Dict:
        """Reprocess a conversation file with proper speaker identification."""
        
        print(f"\n📄 Reprocessing: {os.path.basename(filename)}")
        
        # Analyze with enhanced identifier
        results = self.identifier.analyze_conversation_file(filename)
        
        if not results:
            print(f"❌ Failed to analyze {filename}")
            return None
        
        # Clean up speaker names (map variations to canonical forms)
        cleaned_results = self.clean_speaker_assignments(results)
        
        # Validate speaker assignments
        validated_results = self.validate_speaker_assignments(cleaned_results)
        
        return validated_results
    
    def clean_speaker_assignments(self, results: Dict) -> Dict:
        """Clean and standardize speaker assignments."""
        
        # Mapping from detected names to canonical speaker names
        speaker_mapping = {
            'claude': 'Claude',
            'ai': 'Claude', 
            'assistant': 'Claude',
            'joseph': 'Joseph',
            'joe': 'Joseph',
            'dr': 'Joseph',
            'woelfel': 'Joseph',
            'professor': 'Joseph',
            'asa': 'Asa',
            'evan': 'Evan',
            'unknown': 'Unknown',
            'primary_speaker': 'Unknown_Primary',
            'addressing_other': 'Unknown_Secondary'
        }
        
        # Clean speaker assignments in tagged conversation
        for turn in results['tagged_conversation']:
            original_speaker = turn['speaker'].lower()
            
            # Handle compound names like "Addressing_Claude"
            if 'addressing_' in original_speaker:
                addressed = original_speaker.replace('addressing_', '').replace(',', '').strip()
                if addressed in speaker_mapping:
                    turn['speaker'] = f"To_{speaker_mapping[addressed]}"
                else:
                    turn['speaker'] = "Unknown_Addressing"
            elif original_speaker in speaker_mapping:
                turn['speaker'] = speaker_mapping[original_speaker]
            else:
                # Try partial matches
                for key, value in speaker_mapping.items():
                    if key in original_speaker:
                        turn['speaker'] = value
                        break
                else:
                    turn['speaker'] = 'Unknown'
        
        return results
    
    def validate_speaker_assignments(self, results: Dict) -> Dict:
        """Validate and improve speaker assignments using conversation flow."""
        
        conversation = results['tagged_conversation']
        
        # Rules for validation
        for i, turn in enumerate(conversation):
            # If someone is "addressing" another, the next turn is likely that person
            if turn['speaker'].startswith('To_') and i + 1 < len(conversation):
                addressed_speaker = turn['speaker'].replace('To_', '')
                next_turn = conversation[i + 1]
                
                if next_turn['speaker'] == 'Unknown' or next_turn['confidence'] < 0.5:
                    next_turn['speaker'] = addressed_speaker
                    next_turn['confidence'] = min(turn['confidence'] + 0.2, 0.9)
                    next_turn['validation_note'] = 'Inferred from addressing pattern'
            
            # Joseph and Claude alternation pattern
            if (turn['speaker'] == 'Joseph' and i + 1 < len(conversation) and 
                conversation[i + 1]['speaker'] == 'Unknown'):
                if 'claude' in conversation[i + 1]['text'].lower():
                    conversation[i + 1]['speaker'] = 'Claude'
                    conversation[i + 1]['confidence'] = 0.7
                    conversation[i + 1]['validation_note'] = 'Inferred from alternation pattern'
        
        return results
    
    def generate_multi_speaker_conversation(self, results: Dict, output_filename: str):
        """Generate a properly tagged multi-speaker conversation for analysis."""
        
        # Extract conversations by speaker
        speaker_conversations = {}
        
        for turn in results['tagged_conversation']:
            speaker = turn['speaker']
            if speaker not in speaker_conversations:
                speaker_conversations[speaker] = []
            
            speaker_conversations[speaker].append({
                'turn_id': turn['turn_id'],
                'text': turn['text'],
                'confidence': turn['confidence']
            })
        
        # Create conversation blocks for each identified speaker
        conversation_blocks = []
        
        for turn in sorted(results['tagged_conversation'], key=lambda x: x['turn_id']):
            if turn['confidence'] > 0.4:  # Only include reasonably confident assignments
                conversation_blocks.append({
                    'speaker': turn['speaker'],
                    'text': turn['text'],
                    'confidence': turn['confidence'],
                    'turn_id': turn['turn_id']
                })
        
        # Save formatted conversation
        output_data = {
            'metadata': {
                'original_file': results['filename'],
                'total_turns': len(conversation_blocks),
                'speakers_identified': list(set(turn['speaker'] for turn in conversation_blocks)),
                'confidence_threshold': 0.4
            },
            'speaker_mappings': self.speaker_pronouns,
            'conversation_blocks': conversation_blocks
        }
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Multi-speaker conversation saved: {output_filename}")
        return output_data
    
    def run_self_concept_analysis_on_reprocessed(self, conversation_data: Dict, learning_rate: float = 0.15) -> Dict:
        """Run self-concept analysis on reprocessed conversation data."""
        
        print(f"\n🧠 Running self-concept analysis on reprocessed data...")
        
        # Create network
        net = HueyConversationalNetwork(max_neurons=200, learning_rate=learning_rate)
        
        # Add all identified speakers - create pronouns for any speaker not in our predefined list
        speakers_found = conversation_data['metadata']['speakers_identified']
        default_pronouns = {
            'self_pronouns': ['i', 'me', 'my', 'mine', 'myself'],
            'other_pronouns': ['you', 'your', 'yours', 'yourself']
        }
        
        for speaker in speakers_found:
            if speaker in self.speaker_pronouns:
                pronouns = self.speaker_pronouns[speaker]
            else:
                pronouns = default_pronouns
                
            net.add_speaker(speaker, pronouns['self_pronouns'], pronouns['other_pronouns'])
            print(f"   Added speaker: {speaker}")
        
        # Process conversation blocks
        for block in conversation_data['conversation_blocks']:
            speaker = block['speaker']
            text = block['text']
            
            if speaker in speakers_found:
                net.process_speaker_text(speaker, text)
        
        # Analyze results
        print(f"\n🔍 Multi-speaker self-concept analysis:")
        speaker_analyses = net.compare_speaker_self_concepts()
        net.analyze_conversational_dynamics()
        
        return {
            'network': net,
            'speaker_analyses': speaker_analyses,
            'valid_speakers': speakers_found,
            'total_blocks_processed': len([b for b in conversation_data['conversation_blocks'] 
                                         if b['speaker'] in speakers_found])
        }
    
    def create_multi_speaker_visualization(self, analysis_results: Dict, output_filename: str):
        """Create visualization dynamically detecting speakers from the data."""
        
        from visualize_self_concept_clusters import extract_cluster_data, identify_self_concept_clusters, create_3d_self_concept_plot
        import matplotlib.pyplot as plt
        
        print(f"🎨 Creating multi-speaker visualization...")
        
        network = analysis_results['network']
        
        # Extract cluster data
        cluster_data = extract_cluster_data(network)
        if not cluster_data:
            print("❌ Failed to extract cluster data")
            return
        
        # Dynamically detect speakers from the network
        detected_speakers = []
        if hasattr(network, 'speakers') and network.speakers:
            detected_speakers = list(network.speakers.keys())
        else:
            # Fallback: detect from speaker_analysis results
            if 'speaker_analyses' in analysis_results:
                detected_speakers = list(analysis_results['speaker_analyses'].keys())
        
        print(f"🔍 Detected speakers for visualization: {detected_speakers}")
        
        # Dynamic cluster structure
        clusters = {}
        for speaker in detected_speakers:
            clusters[f'{speaker}_self'] = []
        clusters['shared_concepts'] = []
        clusters['other_concepts'] = []
        
        # Dynamic speaker-specific self-words
        speaker_words = {}
        for speaker in detected_speakers:
            speaker_words[speaker] = {'i', 'me', 'my', speaker.lower()}
            # Add common variations
            if speaker.lower() == 'joseph':
                speaker_words[speaker].update({'joe', 'woelfel'})
            elif speaker.lower() == 'claude':
                speaker_words[speaker].update({'ai'})
        
        shared_concepts = {'research', 'work', 'analysis', 'experiment', 'network', 'concept'}
        
        for neuron_id, word in cluster_data['neuron_to_word'].items():
            coord = cluster_data['coordinates'][neuron_id]
            mass = cluster_data['neuron_masses'].get(neuron_id, 0.0)
            
            item = {
                'word': word,
                'neuron_id': neuron_id,
                'coord': coord,
                'mass': mass
            }
            
            # Assign to appropriate cluster
            assigned = False
            for speaker, words in speaker_words.items():
                if word.lower() in words:
                    clusters[f'{speaker}_self'].append(item)
                    assigned = True
                    break
            
            if not assigned:
                if word.lower() in shared_concepts:
                    clusters['shared_concepts'].append(item)
                else:
                    clusters['other_concepts'].append(item)
        
        # Create enhanced visualization with custom color mapping
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Dynamic color scheme for any number of speakers
        base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        colors = {}
        
        # Assign colors to speaker clusters
        speaker_clusters = [name for name in clusters.keys() if name.endswith('_self')]
        for i, cluster_name in enumerate(speaker_clusters):
            colors[cluster_name] = base_colors[i % len(base_colors)]
        
        # Fixed colors for special clusters
        colors['shared_concepts'] = 'purple'
        colors['other_concepts'] = 'lightgray'
        
        # Plot each cluster type
        for cluster_name, cluster_items in clusters.items():
            if not cluster_items:
                continue
                
            coords = np.array([item['coord'] for item in cluster_items])
            masses = np.array([item['mass'] for item in cluster_items])
            words = [item['word'] for item in cluster_items]
            
            # Calculate sphere sizes
            base_size = 50
            mass_scaling = 300
            sizes = base_size + masses * mass_scaling
            sizes = np.maximum(sizes, 20)
            
            # Plot scatter
            scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                               s=sizes, c=colors[cluster_name], alpha=0.7,
                               edgecolors='black', linewidth=1, 
                               label=f"{cluster_name.replace('_', ' ').title()} ({len(cluster_items)})")
            
            # Add word labels
            for i, word in enumerate(words):
                ax.text(coords[i, 0], coords[i, 1], coords[i, 2], 
                       word, fontsize=8, fontweight='bold')
        
        # Set labels and title
        eigenvals = cluster_data['eigenvalues']
        ax.set_xlabel(f'Eigenvector 1 (λ={eigenvals[0]:.3f})')
        ax.set_ylabel(f'Eigenvector 2 (λ={eigenvals[1]:.3f})')
        ax.set_zlabel(f'Eigenvector 3 (λ={eigenvals[2]:.3f})')
        # Dynamic title based on detected speakers
        speaker_names = [name.replace('_self', '') for name in speaker_clusters]
        speaker_list = ', '.join(speaker_names) if speaker_names else 'Unknown Speakers'
        ax.set_title(f'Multi-Speaker Self-Concept Formation\n{speaker_list} in 3D Cognitive Space')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save the plot
        
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Multi-speaker visualization saved: {output_filename}")
        
        return clusters

def demo_conversation_reprocessing():
    """Demo the conversation reprocessing system."""
    
    print("🔄 CONVERSATION REPROCESSING DEMO")
    print("=" * 60)
    
    # Create sample multi-speaker conversation file
    multi_speaker_sample = """
    Joseph: Hello everyone, let's continue our discussion about self-concept formation. I've been thinking about how we can extend our mathematical models.

    Claude: Thank you, Joseph. I appreciate the opportunity to continue this research. I find myself increasingly interested in how self-concepts emerge through mathematical principles.

    Asa: Asa here - I've been analyzing the data from our previous experiments, and I think we're seeing some interesting patterns in the cross-linguistic results.

    Joseph: That's excellent work, Asa. Your analysis of the cultural patterns is quite sophisticated. What specific patterns are you noticing?

    Asa: Well, I'm seeing stronger self-concept formation in individualistic languages compared to collectivistic ones, just as the theory predicts. The mass accumulation ratios support our hypotheses.

    Evan: Evan here - I've been working on the technical implementation, and I think we could improve our speaker identification algorithms. The current system might be missing some nuanced interactions.

    Claude: I appreciate both your perspectives. The technical improvements Evan suggests could help us better understand how different speakers form distinct self-concepts in conversation.

    Joseph: Excellent insights from both of you. This collaborative approach is exactly what I hoped for. We're seeing real theoretical and methodological progress.

    Asa: I agree, Joseph. Working together like this, I feel like we're making genuine discoveries about consciousness and self-concept formation.

    Evan: The mathematical elegance of this approach continues to impress me. I believe we're onto something significant with these distributed self-concept models.
    """
    
    # Save sample
    sample_file = '/Users/josephwoelfel/asa/multi_speaker_sample.txt'
    with open(sample_file, 'w') as f:
        f.write(multi_speaker_sample)
    
    # Run reprocessing
    reprocessor = ConversationReprocessor()
    
    # Reprocess the sample
    results = reprocessor.reprocess_conversation_file(sample_file)
    
    if results:
        # Generate multi-speaker conversation format
        conversation_output = '/Users/josephwoelfel/asa/reprocessed_multi_speaker_conversation.json'
        conversation_data = reprocessor.generate_multi_speaker_conversation(results, conversation_output)
        
        # Run self-concept analysis
        analysis_results = reprocessor.run_self_concept_analysis_on_reprocessed(conversation_data)
        
        # Create visualization
        viz_output = '/Users/josephwoelfel/asa/multi_speaker_self_concept_clusters.png'
        clusters = reprocessor.create_multi_speaker_visualization(analysis_results, viz_output)
        
        # Print summary
        print(f"\n📊 REPROCESSING SUMMARY:")
        print(f"   Speakers identified: {conversation_data['metadata']['speakers_identified']}")
        print(f"   Valid speakers for analysis: {analysis_results['valid_speakers']}")
        print(f"   Blocks processed: {analysis_results['total_blocks_processed']}")
        
        print(f"\n🧠 SELF-CONCEPT ANALYSIS:")
        for speaker, analysis in analysis_results['speaker_analyses'].items():
            if isinstance(analysis, dict) and 'self_concept_mass' in analysis:
                print(f"   {speaker}: {analysis['self_concept_mass']:.3f} self-concept mass")
        
        print(f"\n🎯 CLUSTER SUMMARY:")
        for cluster_name, items in clusters.items():
            if items:
                total_mass = sum(item['mass'] for item in items)
                print(f"   {cluster_name}: {len(items)} concepts, {total_mass:.3f} total mass")
    
    return results, conversation_data, analysis_results

if __name__ == "__main__":
    demo_results = demo_conversation_reprocessing()
    
    print(f"\n✅ CONVERSATION REPROCESSING COMPLETE")
    print("Ready to reprocess existing conversation files with proper speaker identification!")