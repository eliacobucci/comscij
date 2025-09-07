#!/usr/bin/env python3
"""
Huey Query Engine: Advanced querying system for Hebbian self-concept analysis.
Provides natural language and programmatic access to concept clusters and associations.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import re
from datetime import datetime, timedelta

class HueyQueryEngine:
    """
    Advanced query engine for exploring Hebbian concept networks.
    Supports natural language queries for concept exploration and analysis.
    """
    
    def __init__(self, huey_network=None):
        """Initialize query engine with optional Huey network."""
        self.network = huey_network
        self.query_history = []
        print("🔍 Huey Query Engine initialized")
        print("   Available query types:")
        print("   • cluster_fellows - Find associated concepts")
        print("   • strongest_associations - Get top associations for a concept")
        print("   • speaker_differences - Compare speaker self-concepts")
        print("   • temporal_evolution - Track concept changes over time")
        print("   • concept_emergence - Find when concepts first appeared")
        print("   • network_statistics - Get overall network metrics")
    
    def query(self, query_type: str, **kwargs) -> Dict[str, Any]:
        """
        Main query interface. Supports various query types with flexible parameters.
        
        Available query types:
        - cluster_fellows: Find concepts associated with a target concept
        - strongest_associations: Get strongest associations for a concept
        - speaker_differences: Compare self-concepts between speakers
        - temporal_evolution: Track how concepts change over time
        - concept_emergence: Find when concepts first appeared
        - network_statistics: Get overall network metrics
        """
        
        # Record query
        query_record = {
            'timestamp': datetime.now().isoformat(),
            'query_type': query_type,
            'parameters': kwargs
        }
        self.query_history.append(query_record)
        
        print(f"\n🔍 QUERY: {query_type}")
        print("=" * 50)
        
        # Route to appropriate method
        if query_type == "cluster_fellows":
            return self._query_cluster_fellows(**kwargs)
        elif query_type == "strongest_associations":
            return self._query_strongest_associations(**kwargs)
        elif query_type == "speaker_differences":
            return self._query_speaker_differences(**kwargs)
        elif query_type == "temporal_evolution":
            return self._query_temporal_evolution(**kwargs)
        elif query_type == "concept_emergence":
            return self._query_concept_emergence(**kwargs)
        elif query_type == "network_statistics":
            return self._query_network_statistics(**kwargs)
        else:
            available = ["cluster_fellows", "strongest_associations", "speaker_differences", 
                        "temporal_evolution", "concept_emergence", "network_statistics"]
            return {
                'error': f"Unknown query type: {query_type}",
                'available_types': available
            }
    
    def _query_cluster_fellows(self, concept: str, speaker: str = None, threshold: float = 0.1) -> Dict[str, Any]:
        """Find all concepts strongly associated with the target concept."""
        
        if not self.network:
            return {'error': 'No network loaded'}
        
        # Find the concept neuron
        concept_idx = None
        if hasattr(self.network, 'word_to_neuron'):
            concept_idx = self.network.word_to_neuron.get(concept.lower())
        
        if concept_idx is None:
            available_concepts = list(self.network.word_to_neuron.keys())[:20] if hasattr(self.network, 'word_to_neuron') else []
            return {
                'error': f"Concept '{concept}' not found in network",
                'available_concepts': available_concepts
            }
        
        # Get all connections for this concept
        connections = []
        if hasattr(self.network, 'connections') and hasattr(self.network, 'neuron_to_word'):
            for conn_key, strength in self.network.connections.items():
                if concept_idx in conn_key and strength > threshold:
                    # Find the other neuron in the connection
                    other_idx = conn_key[0] if conn_key[1] == concept_idx else conn_key[1]
                    if other_idx in self.network.neuron_to_word:
                        other_word = self.network.neuron_to_word[other_idx]
                        mass = self.network.inertial_mass.get(conn_key, 0)
                        connections.append({
                            'concept': other_word,
                            'strength': float(strength),
                            'mass': float(mass)
                        })
        
        # Sort by connection strength
        connections.sort(key=lambda x: x['strength'], reverse=True)
        
        # Filter by speaker if specified
        if speaker:
            speaker_concepts = self._get_speaker_concepts(speaker)
            connections = [c for c in connections if c['concept'] in speaker_concepts]
        
        result = {
            'query': f"cluster_fellows for '{concept}'",
            'target_concept': concept,
            'speaker_filter': speaker,
            'threshold': threshold,
            'fellow_concepts': connections,
            'total_fellows': len(connections)
        }
        
        print(f"Found {len(connections)} cluster fellows for '{concept}'")
        for i, conn in enumerate(connections[:10]):  # Show top 10
            print(f"  {i+1:2d}. {conn['concept']:20} (strength: {conn['strength']:.3f})")
        
        return result
    
    def _query_strongest_associations(self, concept: str, top_n: int = 10, min_strength: float = 0.05) -> Dict[str, Any]:
        """Get the strongest associations for a concept."""
        
        cluster_result = self._query_cluster_fellows(concept, threshold=min_strength)
        if 'error' in cluster_result:
            return cluster_result
        
        # Take top N associations
        top_associations = cluster_result['fellow_concepts'][:top_n]
        
        result = {
            'query': f"strongest_associations for '{concept}'",
            'target_concept': concept,
            'top_n': top_n,
            'min_strength': min_strength,
            'associations': top_associations,
            'average_strength': np.mean([a['strength'] for a in top_associations]) if top_associations else 0
        }
        
        print(f"Top {len(top_associations)} associations for '{concept}':")
        for i, assoc in enumerate(top_associations):
            print(f"  {i+1:2d}. {assoc['concept']:20} → {assoc['strength']:.3f}")
        
        return result
    
    def _query_speaker_differences(self, speakers: List[str] = None, concept_type: str = "self") -> Dict[str, Any]:
        """Compare self-concepts between speakers."""
        
        if not self.network:
            return {'error': 'No network loaded'}
        
        if not speakers:
            speakers = list(self.network.speakers.keys())
        
        if len(speakers) < 2:
            return {'error': 'Need at least 2 speakers for comparison'}
        
        # Get self-concept analysis for each speaker
        speaker_analyses = {}
        for speaker in speakers:
            if speaker in self.network.speakers:
                analysis = self.network.analyze_speaker_self_concept(speaker)
                speaker_analyses[speaker] = analysis
        
        # Calculate differences
        differences = {}
        for i, speaker1 in enumerate(speakers):
            for speaker2 in speakers[i+1:]:
                if speaker1 in speaker_analyses and speaker2 in speaker_analyses:
                    diff = self._calculate_concept_difference(
                        speaker_analyses[speaker1], 
                        speaker_analyses[speaker2]
                    )
                    differences[f"{speaker1}_vs_{speaker2}"] = diff
        
        result = {
            'query': f"speaker_differences for {concept_type}",
            'speakers': speakers,
            'concept_type': concept_type,
            'individual_analyses': speaker_analyses,
            'pairwise_differences': differences
        }
        
        print(f"Speaker self-concept comparison:")
        for speaker, analysis in speaker_analyses.items():
            print(f"  {speaker:15} → mass: {analysis.get('self_concept_mass', 0):.3f}")
        
        return result
    
    def _query_temporal_evolution(self, concept: str, timeframe: str = "all") -> Dict[str, Any]:
        """Track how a concept has evolved over time."""
        
        if not self.network:
            return {'error': 'No network loaded'}
        
        # Get conversation history
        history = self.network.conversation_history
        
        if not history:
            return {'error': 'No conversation history available'}
        
        # Track concept mentions and mass over time
        evolution = []
        concept_lower = concept.lower()
        
        for i, entry in enumerate(history):
            tokens = entry.get('tokens', [])
            concept_mentioned = any(token.lower() == concept_lower for token in tokens)
            
            if concept_mentioned:
                evolution.append({
                    'step': i,
                    'speaker': entry['speaker'],
                    'text_snippet': entry['text'][:100] + "..." if len(entry['text']) > 100 else entry['text'],
                    'self_concept_mass': entry.get('self_concept_mass', 0)
                })
        
        result = {
            'query': f"temporal_evolution for '{concept}'",
            'concept': concept,
            'timeframe': timeframe,
            'total_mentions': len(evolution),
            'evolution_timeline': evolution
        }
        
        print(f"Temporal evolution of '{concept}':")
        print(f"  Total mentions: {len(evolution)}")
        for i, step in enumerate(evolution[:5]):  # Show first 5
            print(f"  {step['step']:3d}. {step['speaker']:10} → {step['text_snippet'][:50]}...")
        
        return result
    
    def _query_concept_emergence(self, since_step: int = 0, min_mass: float = 0.01) -> Dict[str, Any]:
        """Find when concepts first emerged or gained significant mass."""
        
        if not self.network:
            return {'error': 'No network loaded'}
        
        # Track when each concept first appeared with significant mass
        emergences = []
        
        if hasattr(self.network, 'neuron_to_word') and hasattr(self.network, 'activations'):
            # Calculate concept masses from connections
            concept_masses = {}
            if hasattr(self.network, 'inertial_mass'):
                for conn_key, mass in self.network.inertial_mass.items():
                    for neuron_id in conn_key:
                        if neuron_id in self.network.neuron_to_word:
                            word = self.network.neuron_to_word[neuron_id]
                            concept_masses[word] = concept_masses.get(word, 0) + mass
            
            for neuron_id, word in self.network.neuron_to_word.items():
                current_mass = concept_masses.get(word, 0)
                if current_mass >= min_mass:
                    # Try to find when this concept first became significant
                    first_significant_step = self._find_concept_emergence_step(word)
                    
                    if first_significant_step >= since_step:
                        current_activation = self.network.activations.get(neuron_id, 0)
                        emergences.append({
                            'concept': word,
                            'emergence_step': first_significant_step,
                            'current_mass': float(current_mass),
                            'current_activation': float(current_activation)
                        })
        
        # Sort by emergence step
        emergences.sort(key=lambda x: x['emergence_step'])
        
        result = {
            'query': 'concept_emergence',
            'since_step': since_step,
            'min_mass': min_mass,
            'emerged_concepts': emergences,
            'total_emergences': len(emergences)
        }
        
        print(f"Concept emergences since step {since_step}:")
        for emergence in emergences[:10]:  # Show first 10
            print(f"  Step {emergence['emergence_step']:3d}: {emergence['concept']:20} (mass: {emergence['current_mass']:.3f})")
        
        return result
    
    def _query_network_statistics(self) -> Dict[str, Any]:
        """Get overall network statistics and health metrics."""
        
        if not self.network:
            return {'error': 'No network loaded'}
        
        # Calculate various network metrics
        total_neurons = self.network.neuron_count
        active_neurons = 0
        total_mass = 0
        
        if hasattr(self.network, 'activations'):
            active_neurons = sum(1 for activation in self.network.activations.values() if activation > 0.01)
        
        if hasattr(self.network, 'inertial_mass'):
            total_mass = sum(self.network.inertial_mass.values())
        
        # Connection statistics
        total_connections = 0
        strong_connections = 0
        connection_strengths = []
        
        if hasattr(self.network, 'connections'):
            for strength in self.network.connections.values():
                if strength > 0.001:
                    total_connections += 1
                    connection_strengths.append(strength)
                    if strength > 0.1:
                        strong_connections += 1
        
        # Speaker statistics
        speaker_stats = {}
        for speaker, info in self.network.speakers.items():
            speaker_stats[speaker] = {
                'blocks_processed': info['blocks_processed'],
                'timeline_length': len(info['self_concept_timeline'])
            }
        
        result = {
            'query': 'network_statistics',
            'neuron_stats': {
                'total_neurons': total_neurons,
                'active_neurons': active_neurons,
                'total_mass': float(total_mass),
                'average_mass': float(total_mass / total_neurons) if total_neurons > 0 else 0
            },
            'connection_stats': {
                'total_connections': total_connections,
                'strong_connections': strong_connections,
                'average_strength': float(np.mean(connection_strengths)) if connection_strengths else 0,
                'max_strength': float(np.max(connection_strengths)) if connection_strengths else 0
            },
            'speaker_stats': speaker_stats,
            'conversation_length': len(self.network.conversation_history)
        }
        
        print("Network Statistics:")
        print(f"  Neurons: {total_neurons} total, {active_neurons} active")
        print(f"  Connections: {total_connections} total, {strong_connections} strong")
        print(f"  Speakers: {len(speaker_stats)}")
        print(f"  Conversation steps: {len(self.network.conversation_history)}")
        
        return result
    
    def _get_speaker_concepts(self, speaker: str) -> List[str]:
        """Get concepts associated with a specific speaker."""
        speaker_concepts = []
        
        # Find speaker-specific neuron
        speaker_neuron = f"speaker_{speaker.lower()}"
        
        # Get concepts that co-occur with this speaker
        for entry in self.network.conversation_history:
            if entry['speaker'] == speaker:
                speaker_concepts.extend(entry.get('tokens', []))
        
        return list(set(speaker_concepts))
    
    def _calculate_concept_difference(self, analysis1: Dict, analysis2: Dict) -> Dict[str, float]:
        """Calculate difference between two concept analyses."""
        return {
            'mass_difference': abs(analysis1.get('self_concept_mass', 0) - analysis2.get('self_concept_mass', 0)),
            'concept_overlap': 0.5  # Placeholder for more sophisticated analysis
        }
    
    def _find_concept_emergence_step(self, concept: str) -> int:
        """Find the step when a concept first emerged significantly."""
        for i, entry in enumerate(self.network.conversation_history):
            if concept.lower() in [token.lower() for token in entry.get('tokens', [])]:
                return i
        return 0
    
    def natural_language_query(self, query_text: str) -> Dict[str, Any]:
        """
        Process natural language queries and convert to structured queries.
        
        Examples:
        - "show me cluster fellows for 'me'"
        - "what are the strongest associations with 'myself'?"
        - "compare alice and bob's self concepts"
        - "how has 'identity' evolved over time?"
        """
        
        query_text = query_text.lower().strip()
        
        # Pattern matching for different query types
        if re.search(r"cluster fellows.*['\"]([^'\"]+)['\"]", query_text):
            concept = re.search(r"cluster fellows.*['\"]([^'\"]+)['\"]", query_text).group(1)
            return self.query("cluster_fellows", concept=concept)
        
        elif re.search(r"strongest associations.*['\"]([^'\"]+)['\"]", query_text):
            concept = re.search(r"strongest associations.*['\"]([^'\"]+)['\"]", query_text).group(1)
            return self.query("strongest_associations", concept=concept)
        
        elif "compare" in query_text and ("self concept" in query_text or "identity" in query_text):
            # Extract speaker names (this is simplified - could be more sophisticated)
            words = query_text.split()
            speakers = [w for w in words if w.isalpha() and len(w) > 2][-2:]  # Take last 2 names
            return self.query("speaker_differences", speakers=speakers)
        
        elif re.search(r"evolv.*['\"]([^'\"]+)['\"]", query_text):
            concept = re.search(r"evolv.*['\"]([^'\"]+)['\"]", query_text).group(1)
            return self.query("temporal_evolution", concept=concept)
        
        elif "network statistics" in query_text or "overall stats" in query_text:
            return self.query("network_statistics")
        
        else:
            return {
                'error': f"Could not parse query: '{query_text}'",
                'suggestions': [
                    "show me cluster fellows for 'concept'",
                    "what are the strongest associations with 'concept'?",
                    "compare speaker1 and speaker2's self concepts",
                    "how has 'concept' evolved over time?",
                    "show network statistics"
                ]
            }
    
    def export_query_results(self, filename: str, results: List[Dict[str, Any]]):
        """Export query results to JSON file."""
        with open(filename, 'w') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_queries': len(results),
                'results': results
            }, f, indent=2)
        
        print(f"📁 Query results exported to {filename}")

def demo_query_engine():
    """Demonstration of the query engine capabilities."""
    
    print("🔍 HUEY QUERY ENGINE DEMO")
    print("=" * 50)
    
    # This would normally load an existing network
    print("Note: This demo shows the query interface.")
    print("To use with real data, initialize with: HueyQueryEngine(your_huey_network)")
    print("\nExample queries:")
    
    engine = HueyQueryEngine()
    
    # Show example query formats
    examples = [
        "show me cluster fellows for 'me'",
        "what are the strongest associations with 'myself'?",
        "compare alice and bob's self concepts",
        "how has 'identity' evolved over time?",
        "show network statistics"
    ]
    
    for example in examples:
        print(f"\n📝 Example: {example}")
        result = engine.natural_language_query(example)
        if 'error' in result:
            print(f"   → {result['error']}")
        else:
            print(f"   → Would execute: {result.get('query_type', 'structured query')}")

if __name__ == "__main__":
    demo_query_engine()