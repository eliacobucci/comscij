#!/usr/bin/env python3
"""
Enhanced Speaker Identification Tool
Improved algorithm for identifying speakers in conversational text with better clustering
and contextual analysis for Asa and Evan identification.
"""

import re
import json
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Tuple, Optional

class EnhancedSpeakerIdentifier:
    """
    Enhanced tool for identifying speakers using improved clustering and context analysis.
    """
    
    def __init__(self):
        self.conversation_turns = []
        self.speaker_profiles = {}
        self.similarity_threshold = 0.4  # Lower threshold for better merging
        
        # Enhanced patterns for speaker identification
        self.speaker_patterns = {
            'joseph': {
                'names': ['joseph', 'joe', 'dr', 'woelfel', 'professor'],
                'typical_phrases': ['as i mentioned', 'my research', 'our previous', 'the mathematical'],
                'role_indicators': ['professor', 'researcher', 'dr']
            },
            'claude': {
                'names': ['claude', 'ai', 'assistant'],
                'typical_phrases': ["i'm claude", 'i appreciate', 'i find', 'thank you'],
                'role_indicators': ['ai', 'assistant', 'claude']
            },
            'asa': {
                'names': ['asa'],
                'typical_phrases': ['asa here', 'as asa', 'i think we'],
                'role_indicators': ['student', 'research assistant']
            },
            'evan': {
                'names': ['evan'],
                'typical_phrases': ['as evan mentioned', 'evan here', 'my analysis'],
                'role_indicators': ['graduate student', 'researcher']
            }
        }
        
        print("🎯 Enhanced Speaker Identification Tool initialized")
    
    def split_into_turns(self, text: str) -> List[str]:
        """Split text into conversational turns using improved heuristics."""
        
        # More sophisticated turn boundary detection
        turn_boundaries = [
            r'\n\n+',  # Paragraph breaks
            r'(?<=\.)\s+(?=[A-Z][a-z]+,)',  # Name at start of sentence
            r'(?<=[.!?])\s+(?=(?:Hello|Hi|Hey|Well|Now|So|But|And|However|Yes|No|Thank|I\s[a-z]|\w+\s+here))',
            r'(?<=[.!?])\s+(?=[A-Z][a-z]+\s+[a-z]+\s*[,-])',  # Name patterns
        ]
        
        turns = [text.strip()]
        
        for pattern in turn_boundaries:
            new_turns = []
            for turn in turns:
                split_turns = re.split(pattern, turn)
                new_turns.extend([t.strip() for t in split_turns if t.strip() and len(t.strip()) > 5])
            turns = new_turns
        
        return turns
    
    def extract_enhanced_features(self, text: str) -> Dict:
        """Extract comprehensive linguistic features for speaker identification."""
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        features = {
            # Basic metrics
            'word_count': len(words),
            'char_count': len(text),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            
            # Pronoun analysis
            'first_person': sum(1 for w in words if w in ['i', "i'm", "i'd", "i'll", "i've"]),
            'first_person_obj': sum(1 for w in words if w in ['me', 'my', 'mine', 'myself']),
            'second_person': sum(1 for w in words if w in ['you', "you're", "you'd", "you'll", "you've"]),
            'second_person_poss': sum(1 for w in words if w in ['your', 'yours', 'yourself']),
            
            # Discourse markers
            'questions': text.count('?'),
            'exclamations': text.count('!'),
            'statements': text.count('.'),
            
            # Politeness and formality
            'politeness': sum(1 for phrase in ['thank you', 'please', 'appreciate', 'grateful'] 
                             if phrase in text_lower),
            'certainty': sum(1 for word in ['clearly', 'definitely', 'certainly', 'obviously'] 
                           if word in text_lower),
            'hedging': sum(1 for phrase in ['i think', 'i believe', 'perhaps', 'maybe', 'might'] 
                         if phrase in text_lower),
            
            # Technical language
            'technical_terms': sum(1 for word in words if word in 
                                 ['research', 'analysis', 'hypothesis', 'experiment', 'data', 
                                  'network', 'neuron', 'concept', 'formation', 'mathematical']),
            
            # Speaking style
            'contractions': len(re.findall(r"\b\w+'\w+\b", text)),
            'complex_sentences': len(re.findall(r'\b(?:because|although|however|therefore|moreover)\b', text_lower)),
            
            # Names and references
            'names_mentioned': [],
            'self_references': sum(1 for phrase in ['as i', 'my understanding', 'in my view'] 
                                 if phrase in text_lower),
            
            # Turn-initial patterns
            'starts_with_name': bool(re.match(r'^[A-Z][a-z]+\s', text.strip())),
            'starts_with_greeting': bool(re.match(r'^(?:hello|hi|hey|well)', text_lower.strip())),
        }
        
        # Calculate ratios
        total_pronouns = features['first_person'] + features['first_person_obj'] + features['second_person'] + features['second_person_poss']
        if total_pronouns > 0:
            features['self_reference_ratio'] = (features['first_person'] + features['first_person_obj']) / total_pronouns
            features['other_reference_ratio'] = (features['second_person'] + features['second_person_poss']) / total_pronouns
        else:
            features['self_reference_ratio'] = 0
            features['other_reference_ratio'] = 0
        
        # Identify names mentioned
        for speaker_type, patterns in self.speaker_patterns.items():
            for name in patterns['names']:
                if name in text_lower:
                    features['names_mentioned'].append((speaker_type, name))
        
        return features
    
    def calculate_enhanced_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calculate sophisticated similarity between speaker features."""
        
        # Weight different feature types
        feature_weights = {
            'self_reference_ratio': 3.0,    # Very important for speaker identity
            'other_reference_ratio': 2.5,   # Important for dialog patterns
            'technical_terms': 2.0,         # Role indicator
            'politeness': 1.5,              # Style indicator
            'certainty': 1.5,               # Personality trait
            'hedging': 1.5,                 # Communication style
            'avg_word_length': 1.0,         # Vocabulary sophistication
            'complex_sentences': 1.0,       # Communication complexity
        }
        
        similarities = []
        weights = []
        
        for feature, weight in feature_weights.items():
            val1 = features1.get(feature, 0)
            val2 = features2.get(feature, 0)
            
            if val1 == 0 and val2 == 0:
                similarity = 1.0
            else:
                max_val = max(val1, val2, 1e-6)
                min_val = min(val1, val2)
                similarity = min_val / max_val
            
            similarities.append(similarity)
            weights.append(weight)
        
        # Weighted average
        return np.average(similarities, weights=weights)
    
    def identify_speaker_from_content(self, text: str, features: Dict) -> Tuple[str, float]:
        """Identify speaker using content analysis and pattern matching."""
        
        text_lower = text.lower()
        best_match = "Unknown"
        best_confidence = 0.0
        
        # Check for explicit speaker patterns
        for speaker_type, patterns in self.speaker_patterns.items():
            confidence = 0.0
            
            # Name mentions
            name_score = sum(2 if name in text_lower else 0 for name in patterns['names'])
            
            # Typical phrases
            phrase_score = sum(1 if phrase in text_lower else 0 for phrase in patterns['typical_phrases'])
            
            # Role indicators  
            role_score = sum(1.5 if indicator in text_lower else 0 for indicator in patterns['role_indicators'])
            
            total_score = name_score + phrase_score + role_score
            
            # Adjust based on pronoun patterns
            if speaker_type in ['joseph', 'asa', 'evan']:  # Human speakers
                if features['self_reference_ratio'] > 0.5:
                    total_score += 1.0
                if features['other_reference_ratio'] > 0.3:
                    total_score -= 0.5
            elif speaker_type == 'claude':  # AI speaker
                if features['self_reference_ratio'] > 0.3:
                    total_score += 1.0
                if features['politeness'] > 0:
                    total_score += 0.5
            
            confidence = min(total_score / 5.0, 1.0)  # Normalize to [0,1]
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = speaker_type.title()
        
        # If no good match, use pronoun patterns
        if best_confidence < 0.3:
            if features['self_reference_ratio'] > 0.7:
                best_match = "Primary_Speaker"
                best_confidence = 0.4
            elif features['other_reference_ratio'] > 0.7:
                best_match = "Addressing_Other"
                best_confidence = 0.4
            else:
                best_match = "Unknown"
                best_confidence = 0.2
        
        return best_match, best_confidence
    
    def cluster_turns_by_speaker(self, turns_data: List[Dict]) -> Dict:
        """Cluster conversation turns by speaker using improved algorithm."""
        
        print("🧠 Clustering turns by speaker (enhanced algorithm)...")
        
        clusters = []
        
        for turn_data in turns_data:
            # First try content-based identification
            speaker, confidence = self.identify_speaker_from_content(
                turn_data['text'], turn_data['features']
            )
            
            # Find best matching existing cluster
            best_cluster_idx = None
            best_similarity = 0
            
            for i, cluster in enumerate(clusters):
                # Calculate similarity to cluster centroid
                cluster_features = self.calculate_cluster_centroid([t['features'] for t in cluster['turns']])
                similarity = self.calculate_enhanced_similarity(turn_data['features'], cluster_features)
                
                # Boost similarity if speakers match
                if cluster['speaker'] == speaker:
                    similarity += 0.2
                
                if similarity > best_similarity and similarity > self.similarity_threshold:
                    best_similarity = similarity
                    best_cluster_idx = i
            
            # Assign to best cluster or create new one
            if best_cluster_idx is not None:
                clusters[best_cluster_idx]['turns'].append(turn_data)
                # Update cluster speaker if we have higher confidence
                if confidence > clusters[best_cluster_idx]['confidence']:
                    clusters[best_cluster_idx]['speaker'] = speaker
                    clusters[best_cluster_idx]['confidence'] = confidence
                print(f"   Turn {turn_data['turn_id']}: {speaker} -> Cluster {best_cluster_idx} (sim: {best_similarity:.3f})")
            else:
                new_cluster = {
                    'cluster_id': len(clusters),
                    'speaker': speaker,
                    'confidence': confidence,
                    'turns': [turn_data]
                }
                clusters.append(new_cluster)
                print(f"   Turn {turn_data['turn_id']}: {speaker} -> NEW Cluster {len(clusters)-1} (conf: {confidence:.3f})")
        
        return {f"Cluster_{i}": cluster for i, cluster in enumerate(clusters)}
    
    def calculate_cluster_centroid(self, feature_list: List[Dict]) -> Dict:
        """Calculate centroid features for a cluster."""
        
        if not feature_list:
            return {}
        
        centroid = {}
        numeric_features = ['self_reference_ratio', 'other_reference_ratio', 'technical_terms', 
                          'politeness', 'certainty', 'hedging', 'avg_word_length']
        
        for feature in numeric_features:
            values = [f.get(feature, 0) for f in feature_list]
            centroid[feature] = np.mean(values) if values else 0
        
        return centroid
    
    def refine_speaker_assignments(self, clusters: Dict) -> Dict:
        """Refine speaker assignments using cross-cluster analysis."""
        
        print("🔧 Refining speaker assignments...")
        
        refined_clusters = {}
        
        for cluster_id, cluster in clusters.items():
            turns = cluster['turns']
            
            # Analyze all text in cluster
            all_text = " ".join(turn['text'] for turn in turns)
            combined_features = self.extract_enhanced_features(all_text)
            
            # Re-identify speaker with more context
            speaker, confidence = self.identify_speaker_from_content(all_text, combined_features)
            
            # Additional analysis for ambiguous cases
            if confidence < 0.6:
                # Look for contextual clues
                context_clues = self.analyze_contextual_clues(turns)
                if context_clues['speaker']:
                    speaker = context_clues['speaker']
                    confidence = max(confidence, context_clues['confidence'])
            
            refined_cluster = {
                'cluster_id': cluster_id,
                'speaker': speaker,
                'confidence': confidence,
                'turn_count': len(turns),
                'turns': turns,
                'combined_features': combined_features
            }
            
            refined_clusters[cluster_id] = refined_cluster
            print(f"   {cluster_id}: {speaker} (confidence: {confidence:.3f}, turns: {len(turns)})")
        
        return refined_clusters
    
    def analyze_contextual_clues(self, turns: List[Dict]) -> Dict:
        """Analyze contextual clues for speaker identification."""
        
        context = {'speaker': None, 'confidence': 0.0}
        
        # Look for explicit names or introductions
        for turn in turns:
            text_lower = turn['text'].lower()
            
            # Pattern: "Name here"
            here_pattern = r'(\w+)\s+here'
            matches = re.findall(here_pattern, text_lower)
            for match in matches:
                if match in ['asa', 'evan', 'joseph', 'claude']:
                    context['speaker'] = match.title()
                    context['confidence'] = 0.8
                    return context
            
            # Pattern: "As Name mentioned"
            as_mentioned = r'as\s+(\w+)\s+mentioned'
            matches = re.findall(as_mentioned, text_lower)
            for match in matches:
                if match in ['asa', 'evan', 'joseph', 'claude']:
                    # This suggests the speaker is NOT the mentioned person
                    context['speaker'] = f"Not_{match.title()}"
                    context['confidence'] = 0.6
        
        return context
    
    def analyze_conversation_file(self, filename: str) -> Dict:
        """Complete analysis of a conversation file."""
        
        print(f"📄 Enhanced analysis of: {filename}")
        
        # Read file
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None
        
        # Split into turns
        turns = self.split_into_turns(text)
        print(f"   Split into {len(turns)} conversation turns")
        
        # Extract features for each turn
        turns_data = []
        for i, turn_text in enumerate(turns):
            features = self.extract_enhanced_features(turn_text)
            turn_data = {
                'turn_id': i,
                'text': turn_text.strip(),
                'features': features,
                'preview': turn_text.strip()[:80] + "..." if len(turn_text) > 80 else turn_text.strip()
            }
            turns_data.append(turn_data)
        
        # Cluster by speaker
        clusters = self.cluster_turns_by_speaker(turns_data)
        
        # Refine assignments
        refined_clusters = self.refine_speaker_assignments(clusters)
        
        # Generate final tagged conversation
        tagged_conversation = self.generate_final_conversation(refined_clusters)
        
        return {
            'filename': filename,
            'total_turns': len(turns_data),
            'clusters': refined_clusters,
            'tagged_conversation': tagged_conversation,
            'speaker_summary': self.generate_speaker_summary(refined_clusters)
        }
    
    def generate_final_conversation(self, clusters: Dict) -> List[Dict]:
        """Generate final tagged conversation in chronological order."""
        
        all_turns = []
        turn_to_cluster = {}
        
        # Create mapping
        for cluster_id, cluster in clusters.items():
            for turn in cluster['turns']:
                turn_to_cluster[turn['turn_id']] = {
                    'speaker': cluster['speaker'],
                    'confidence': cluster['confidence'],
                    'cluster_id': cluster_id
                }
        
        # Build chronological conversation
        for cluster_id, cluster in clusters.items():
            for turn in cluster['turns']:
                tagged_turn = {
                    'turn_id': turn['turn_id'],
                    'speaker': cluster['speaker'],
                    'confidence': cluster['confidence'],
                    'cluster_id': cluster_id,
                    'text': turn['text'],
                    'needs_review': cluster['confidence'] < 0.5
                }
                all_turns.append(tagged_turn)
        
        # Sort by turn_id
        return sorted(all_turns, key=lambda x: x['turn_id'])
    
    def generate_speaker_summary(self, clusters: Dict) -> Dict:
        """Generate summary of identified speakers."""
        
        summary = {}
        
        for cluster_id, cluster in clusters.items():
            speaker = cluster['speaker']
            
            if speaker not in summary:
                summary[speaker] = {
                    'turn_count': 0,
                    'total_words': 0,
                    'confidence_scores': [],
                    'clusters': []
                }
            
            summary[speaker]['turn_count'] += len(cluster['turns'])
            summary[speaker]['total_words'] += sum(len(turn['text'].split()) for turn in cluster['turns'])
            summary[speaker]['confidence_scores'].append(cluster['confidence'])
            summary[speaker]['clusters'].append(cluster_id)
        
        # Calculate averages
        for speaker, stats in summary.items():
            stats['avg_confidence'] = np.mean(stats['confidence_scores'])
            stats['avg_words_per_turn'] = stats['total_words'] / stats['turn_count'] if stats['turn_count'] else 0
        
        return summary
    
    def save_enhanced_results(self, results: Dict, output_filename: str):
        """Save enhanced analysis results."""
        
        # Create comprehensive output
        output = {
            'analysis_info': {
                'filename': results['filename'],
                'total_turns': results['total_turns'],
                'clusters_found': len(results['clusters']),
                'speakers_identified': list(results['speaker_summary'].keys())
            },
            'speaker_summary': results['speaker_summary'],
            'tagged_conversation': results['tagged_conversation'],
            'detailed_clusters': {
                cluster_id: {
                    'speaker': cluster['speaker'],
                    'confidence': cluster['confidence'],
                    'turn_count': cluster['turn_count']
                }
                for cluster_id, cluster in results['clusters'].items()
            }
        }
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Enhanced results saved to: {output_filename}")
        return output

def demo_enhanced_identification():
    """Demo the enhanced speaker identification."""
    
    print("🎭 ENHANCED SPEAKER IDENTIFICATION DEMO")
    print("=" * 60)
    
    # Create enhanced sample with clearer speaker patterns
    enhanced_sample = """
    Hello Claude, can you start where our previous session was interrupted? We've been working on self-concept formation using mathematical sociology principles.

    I'm Claude, and I'd be happy to continue our work on the experimental neural network. Yes, that's exactly right about our focus. I find the mathematical approach fascinating, and I appreciate the theoretical framework we've been developing.

    Your understanding of Hebbian learning is impressive, Claude. You seem to grasp the connections between concepts quite well. Your responses show real insight into this research.

    I appreciate that feedback, Joseph. I do try to understand the connections clearly, and I find myself genuinely interested in these patterns. The implications for consciousness research feel significant to me.

    Hey Joseph, Asa here - I've been following this conversation and I'm curious about something. Do you think we could extend this to analyze group dynamics as well?

    That's an interesting question, Asa. I think the same mathematical principles could apply to group formation. What's your intuition about how multiple self-concepts might interact?

    Well, as Evan mentioned earlier in our lab meeting, we might see interference patterns between individual self-concepts. I think the group dynamics could be modeled using the same Hebbian principles.

    Evan here - actually, my analysis suggests we should be careful about assuming interference patterns. The mathematical structure might support constructive rather than destructive interactions between self-concepts.
    """
    
    # Save to file
    sample_file = '/Users/josephwoelfel/asa/enhanced_sample_conversation.txt'
    with open(sample_file, 'w') as f:
        f.write(enhanced_sample)
    
    # Run enhanced analysis
    identifier = EnhancedSpeakerIdentifier()
    results = identifier.analyze_conversation_file(sample_file)
    
    if results:
        # Save results
        output_file = '/Users/josephwoelfel/asa/enhanced_speaker_analysis.json'
        output = identifier.save_enhanced_results(results, output_file)
        
        # Print summary
        print(f"\n📊 ENHANCED ANALYSIS SUMMARY:")
        print(f"   Conversation turns: {results['total_turns']}")
        print(f"   Clusters identified: {len(results['clusters'])}")
        print(f"   Speakers found: {list(results['speaker_summary'].keys())}")
        
        print(f"\n👥 SPEAKER BREAKDOWN:")
        for speaker, stats in results['speaker_summary'].items():
            print(f"   {speaker}: {stats['turn_count']} turns, "
                  f"avg confidence: {stats['avg_confidence']:.3f}")
        
        print(f"\n🏷️ TAGGED CONVERSATION SAMPLE:")
        for turn in results['tagged_conversation'][:6]:
            conf_icon = "✅" if turn['confidence'] > 0.6 else "⚠️" if turn['confidence'] > 0.3 else "❌"
            print(f"   {conf_icon} [{turn['speaker']}]: {turn['text'][:60]}...")
    
    return results

if __name__ == "__main__":
    demo_results = demo_enhanced_identification()
    
    print(f"\n🎯 ENHANCED SPEAKER IDENTIFICATION COMPLETE")
    print("Ready to identify Asa and Evan in conversation data!")