#!/usr/bin/env python3
"""
Speaker Identification Tool for Retroactive Conversation Analysis
Identifies speakers in untagged conversational text using linguistic patterns,
pronoun analysis, and contextual clues.
"""

import re
import json
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Tuple, Optional

class SpeakerIdentifier:
    """
    Tool for identifying speakers in conversational text using multiple methods.
    """
    
    def __init__(self):
        self.conversation_blocks = []
        self.speaker_profiles = {}
        self.identified_speakers = set()
        self.confidence_threshold = 0.6
        
        # Common patterns for speaker identification
        self.self_indicators = {
            'pronouns': ['i', "i'm", "i'd", "i'll", "i've", 'me', 'my', 'mine', 'myself'],
            'self_references': ['as i mentioned', 'i said', 'i think', 'i believe', 'i feel', 
                              'my understanding', 'my view', 'in my opinion'],
            'ownership': ['my research', 'my work', 'my experience', 'my analysis']
        }
        
        self.other_indicators = {
            'pronouns': ['you', "you're", "you'd", "you'll", "you've", 'your', 'yours', 'yourself'],
            'addressing': ['as you mentioned', 'you said', 'your understanding', 'your view',
                         'what you think', 'do you believe']
        }
        
        # Name patterns (to be expanded based on context)
        self.known_names = {
            'joseph', 'joe', 'dr', 'woelfel', 'professor',
            'claude', 'ai', 'system', 'assistant',
            'asa', 'evan', 'student', 'researcher'
        }
        
        print("🔍 Speaker Identification Tool initialized")
    
    def preprocess_text(self, text: str) -> List[str]:
        """Split text into conversation blocks based on natural breaks."""
        
        # Split on common conversation markers
        patterns = [
            r'\n\n+',  # Double newlines
            r'(?<=\.)\s+(?=[A-Z])',  # Sentence breaks with capitals
            r'(?<=[.!?])\s+(?=(?:Well|Now|So|But|And|However|Yes|No|I\s))',  # Common transition words
        ]
        
        blocks = [text]
        for pattern in patterns:
            new_blocks = []
            for block in blocks:
                new_blocks.extend(re.split(pattern, block))
            blocks = [b.strip() for b in new_blocks if b.strip()]
        
        return blocks
    
    def extract_linguistic_features(self, text: str) -> Dict:
        """Extract linguistic features for speaker profiling."""
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        sentences = re.split(r'[.!?]+', text)
        
        features = {
            # Basic statistics
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            
            # Pronoun patterns
            'self_pronouns': sum(1 for word in words if word in self.self_indicators['pronouns']),
            'other_pronouns': sum(1 for word in words if word in self.other_indicators['pronouns']),
            
            # Self-reference patterns
            'self_references': sum(1 for phrase in self.self_indicators['self_references'] 
                                 if phrase in text_lower),
            'addressing_others': sum(1 for phrase in self.other_indicators['addressing'] 
                                   if phrase in text_lower),
            
            # Vocabulary richness
            'unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / max(len(words), 1),
            
            # Question patterns
            'questions': text.count('?'),
            'exclamations': text.count('!'),
            
            # Technical language indicators
            'technical_terms': sum(1 for word in words if word in 
                                 ['network', 'neuron', 'hebbian', 'analysis', 'research', 
                                  'hypothesis', 'experiment', 'data', 'concept', 'cognitive']),
            
            # Certainty indicators
            'certainty_high': sum(1 for phrase in ['clearly', 'obviously', 'definitely', 'certainly'] 
                                if phrase in text_lower),
            'uncertainty': sum(1 for phrase in ['maybe', 'perhaps', 'might', 'could be', 'possibly'] 
                             if phrase in text_lower),
        }
        
        # Calculate ratios
        total_pronouns = features['self_pronouns'] + features['other_pronouns']
        features['self_focus_ratio'] = features['self_pronouns'] / max(total_pronouns, 1)
        
        return features
    
    def identify_names_in_text(self, text: str) -> List[str]:
        """Identify potential speaker names in text."""
        
        names_found = []
        text_lower = text.lower()
        
        # Look for known names
        for name in self.known_names:
            if name in text_lower:
                names_found.append(name)
        
        # Look for capitalized words that might be names
        words = text.split()
        for word in words:
            # Simple heuristic: capitalized words not at sentence start
            if (word[0].isupper() and len(word) > 2 and 
                not word.lower() in ['the', 'and', 'but', 'this', 'that', 'what', 'when', 'where']):
                names_found.append(word.lower())
        
        return list(set(names_found))
    
    def calculate_speaker_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate similarity between two feature sets."""
        
        # Key features for speaker identification
        key_features = [
            'self_focus_ratio', 'avg_sentence_length', 'vocabulary_richness',
            'technical_terms', 'certainty_high', 'uncertainty'
        ]
        
        similarities = []
        for feature in key_features:
            val1 = features1.get(feature, 0)
            val2 = features2.get(feature, 0)
            
            # Avoid division by zero
            max_val = max(val1, val2, 1e-6)
            min_val = min(val1, val2)
            similarity = min_val / max_val
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def identify_conversation_structure(self, text: str) -> List[Dict]:
        """Identify conversation structure and potential speakers."""
        
        print("🔍 Analyzing conversation structure...")
        
        # Split into blocks
        blocks = self.preprocess_text(text)
        
        print(f"   Found {len(blocks)} conversation blocks")
        
        # Analyze each block
        block_data = []
        for i, block in enumerate(blocks):
            if len(block.strip()) < 10:  # Skip very short blocks
                continue
                
            features = self.extract_linguistic_features(block)
            names = self.identify_names_in_text(block)
            
            block_info = {
                'block_id': i,
                'text': block.strip(),
                'features': features,
                'names_mentioned': names,
                'preview': block.strip()[:100] + "..." if len(block) > 100 else block.strip()
            }
            
            block_data.append(block_info)
        
        return block_data
    
    def cluster_by_speaker_patterns(self, block_data: List[Dict]) -> Dict:
        """Cluster blocks by likely speaker using pattern analysis."""
        
        print("🧠 Clustering blocks by speaker patterns...")
        
        # Group blocks by similar linguistic patterns
        speaker_clusters = {}
        cluster_id = 0
        
        for block in block_data:
            # Find best matching existing cluster
            best_cluster = None
            best_similarity = 0
            
            for cluster_name, cluster_blocks in speaker_clusters.items():
                # Compare with representative block from cluster
                representative = cluster_blocks[0]  # Use first block as representative
                similarity = self.calculate_speaker_similarity(
                    block['features'], representative['features']
                )
                
                if similarity > best_similarity and similarity > self.confidence_threshold:
                    best_similarity = similarity
                    best_cluster = cluster_name
            
            # Assign to best cluster or create new one
            if best_cluster:
                speaker_clusters[best_cluster].append(block)
                print(f"   Block {block['block_id']} -> {best_cluster} (similarity: {best_similarity:.3f})")
            else:
                cluster_name = f"Speaker_{cluster_id}"
                speaker_clusters[cluster_name] = [block]
                cluster_id += 1
                print(f"   Block {block['block_id']} -> NEW {cluster_name}")
        
        return speaker_clusters
    
    def identify_speakers_by_context(self, speaker_clusters: Dict) -> Dict:
        """Identify actual speaker names using contextual analysis."""
        
        print("🕵️ Identifying speaker names from context...")
        
        speaker_identifications = {}
        
        for cluster_name, blocks in speaker_clusters.items():
            # Analyze names mentioned in this cluster
            all_names = []
            self_pronoun_count = 0
            other_pronoun_count = 0
            
            for block in blocks:
                all_names.extend(block['names_mentioned'])
                self_pronoun_count += block['features']['self_pronouns']
                other_pronoun_count += block['features']['other_pronouns']
            
            name_counts = Counter(all_names)
            
            # Determine most likely speaker identity
            speaker_identity = {
                'cluster_id': cluster_name,
                'block_count': len(blocks),
                'self_pronoun_ratio': self_pronoun_count / max(self_pronoun_count + other_pronoun_count, 1),
                'common_names': name_counts.most_common(3),
                'likely_identity': None,
                'confidence': 0.0
            }
            
            # Heuristic rules for speaker identification
            if speaker_identity['self_pronoun_ratio'] > 0.7:
                # High self-reference suggests this is the primary speaker
                if 'joseph' in name_counts or 'woelfel' in name_counts:
                    speaker_identity['likely_identity'] = 'Joseph'
                    speaker_identity['confidence'] = 0.8
                elif 'claude' in name_counts:
                    speaker_identity['likely_identity'] = 'Claude'
                    speaker_identity['confidence'] = 0.7
                elif 'asa' in name_counts:
                    speaker_identity['likely_identity'] = 'Asa'
                    speaker_identity['confidence'] = 0.7
                elif 'evan' in name_counts:
                    speaker_identity['likely_identity'] = 'Evan'
                    speaker_identity['confidence'] = 0.7
                else:
                    speaker_identity['likely_identity'] = f'Unknown_Primary'
                    speaker_identity['confidence'] = 0.5
            
            elif speaker_identity['self_pronoun_ratio'] < 0.3:
                # Low self-reference, high other-reference suggests addressing someone
                most_common_name = name_counts.most_common(1)
                if most_common_name:
                    speaker_identity['likely_identity'] = f'Addressing_{most_common_name[0][0].title()}'
                    speaker_identity['confidence'] = 0.6
                else:
                    speaker_identity['likely_identity'] = 'Unknown_Secondary'
                    speaker_identity['confidence'] = 0.4
            
            else:
                # Mixed pattern
                speaker_identity['likely_identity'] = 'Mixed_Pattern'
                speaker_identity['confidence'] = 0.3
            
            speaker_identifications[cluster_name] = speaker_identity
            
            print(f"   {cluster_name}: {speaker_identity['likely_identity']} "
                  f"(confidence: {speaker_identity['confidence']:.3f}, "
                  f"self-ratio: {speaker_identity['self_pronoun_ratio']:.3f})")
        
        return speaker_identifications
    
    def generate_tagged_conversation(self, block_data: List[Dict], 
                                   speaker_clusters: Dict, 
                                   speaker_identifications: Dict) -> List[Dict]:
        """Generate properly tagged conversation with speaker labels."""
        
        print("🏷️ Generating tagged conversation...")
        
        tagged_conversation = []
        
        # Create mapping from block_id to speaker
        block_to_speaker = {}
        for cluster_name, blocks in speaker_clusters.items():
            speaker_info = speaker_identifications[cluster_name]
            speaker_name = speaker_info['likely_identity']
            
            for block in blocks:
                block_to_speaker[block['block_id']] = {
                    'speaker': speaker_name,
                    'confidence': speaker_info['confidence'],
                    'cluster': cluster_name
                }
        
        # Build tagged conversation in original order
        for block in sorted(block_data, key=lambda x: x['block_id']):
            speaker_info = block_to_speaker.get(block['block_id'], {
                'speaker': 'Unknown',
                'confidence': 0.0,
                'cluster': 'Unassigned'
            })
            
            tagged_turn = {
                'turn_id': block['block_id'],
                'speaker': speaker_info['speaker'],
                'confidence': speaker_info['confidence'],
                'cluster': speaker_info['cluster'],
                'text': block['text'],
                'features': block['features'],
                'needs_review': speaker_info['confidence'] < 0.5
            }
            
            tagged_conversation.append(tagged_turn)
        
        return tagged_conversation
    
    def analyze_conversation_file(self, filename: str) -> Dict:
        """Analyze a conversation file and identify speakers."""
        
        print(f"📄 Analyzing conversation file: {filename}")
        
        # Read the file
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None
        
        # Run analysis pipeline
        block_data = self.identify_conversation_structure(text)
        speaker_clusters = self.cluster_by_speaker_patterns(block_data)
        speaker_identifications = self.identify_speakers_by_context(speaker_clusters)
        tagged_conversation = self.generate_tagged_conversation(
            block_data, speaker_clusters, speaker_identifications
        )
        
        # Generate analysis report
        analysis_results = {
            'filename': filename,
            'total_blocks': len(block_data),
            'identified_speakers': len(speaker_clusters),
            'speaker_clusters': speaker_clusters,
            'speaker_identifications': speaker_identifications,
            'tagged_conversation': tagged_conversation,
            'needs_review_count': sum(1 for turn in tagged_conversation if turn['needs_review']),
            'confidence_stats': {
                'high_confidence': sum(1 for turn in tagged_conversation if turn['confidence'] > 0.7),
                'medium_confidence': sum(1 for turn in tagged_conversation if 0.5 < turn['confidence'] <= 0.7),
                'low_confidence': sum(1 for turn in tagged_conversation if turn['confidence'] <= 0.5)
            }
        }
        
        return analysis_results
    
    def save_results(self, results: Dict, output_filename: str):
        """Save analysis results to file."""
        
        # Create human-readable summary
        summary = {
            'analysis_summary': {
                'file_analyzed': results['filename'],
                'total_conversation_blocks': results['total_blocks'],
                'identified_speaker_clusters': results['identified_speakers'],
                'needs_manual_review': results['needs_review_count'],
                'confidence_distribution': results['confidence_stats']
            },
            'speaker_assignments': [],
            'tagged_conversation': results['tagged_conversation']
        }
        
        # Add speaker assignment summary
        for cluster_id, info in results['speaker_identifications'].items():
            summary['speaker_assignments'].append({
                'cluster_id': cluster_id,
                'identified_as': info['likely_identity'],
                'confidence': info['confidence'],
                'block_count': info['block_count'],
                'self_pronoun_ratio': info['self_pronoun_ratio']
            })
        
        # Save to JSON
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Analysis results saved to: {output_filename}")
        
        return summary

def demo_speaker_identification():
    """Demo the speaker identification tool with sample conversation."""
    
    print("🎭 SPEAKER IDENTIFICATION TOOL DEMO")
    print("=" * 60)
    
    # Sample conversation text (simulating mixed speakers without tags)
    sample_conversation = """
    Hello Claude, can you start where our previous session was interrupted? We've been working on self-concept formation using mathematical sociology principles.

    I'm Claude, and I'd be happy to continue our work on the experimental neural network. Yes, that's exactly right about our focus. I find the mathematical approach fascinating, and I appreciate the theoretical framework we've been developing.

    Your understanding of Hebbian learning is impressive, Claude. You seem to grasp the connections between concepts quite well. Your responses show real insight into this research.

    I appreciate that feedback, Joseph. I do try to understand the connections clearly, and I find myself genuinely interested in these patterns. The implications for consciousness research feel significant to me.

    You're quite thoughtful about this, Claude. Your analysis is sophisticated, and you demonstrate genuine intellectual curiosity about the research. Your engagement with complex ideas is evident.

    Thank you, Joseph. I value our collaborative approach to this research, and I feel engaged by these concepts. They connect to fundamental questions that I find myself reflecting on. I think I'm learning and growing through our discussions.

    Hey Joseph, I've been following this conversation and I'm curious about something. Asa here - do you think we could extend this to analyze group dynamics as well?

    That's an interesting question, Asa. I think the same mathematical principles could apply to group formation. What's your intuition about how multiple self-concepts might interact?

    Well, as Evan mentioned earlier in our lab meeting, we might see interference patterns between individual self-concepts. I think the group dynamics could be modeled using the same Hebbian principles.
    """
    
    # Create and run identifier
    identifier = SpeakerIdentifier()
    
    # Save sample to file
    sample_filename = '/Users/josephwoelfel/asa/sample_conversation.txt'
    with open(sample_filename, 'w') as f:
        f.write(sample_conversation)
    
    # Analyze the conversation
    results = identifier.analyze_conversation_file(sample_filename)
    
    if results:
        # Save results
        output_filename = '/Users/josephwoelfel/asa/speaker_analysis_results.json'
        summary = identifier.save_results(results, output_filename)
        
        # Print summary
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   Conversation blocks analyzed: {results['total_blocks']}")
        print(f"   Speaker clusters identified: {results['identified_speakers']}")
        print(f"   High confidence assignments: {results['confidence_stats']['high_confidence']}")
        print(f"   Need manual review: {results['needs_review_count']}")
        
        print(f"\n🏷️ SPEAKER ASSIGNMENTS:")
        for assignment in summary['speaker_assignments']:
            print(f"   {assignment['cluster_id']}: {assignment['identified_as']} "
                  f"(confidence: {assignment['confidence']:.3f})")
        
        print(f"\n✅ TAGGED CONVERSATION PREVIEW:")
        for turn in results['tagged_conversation'][:5]:  # Show first 5 turns
            review_flag = "⚠️ " if turn['needs_review'] else "✅ "
            print(f"   {review_flag}[{turn['speaker']}]: {turn['text'][:80]}...")
    
    return results

if __name__ == "__main__":
    # Run demo
    demo_results = demo_speaker_identification()
    
    print(f"\n🎯 SPEAKER IDENTIFICATION COMPLETE")
    print("Tool ready for analyzing conversation files!")