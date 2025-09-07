#!/usr/bin/env python3
"""
Huey++ Complete Platform: Integrated Hebbian Self-Concept Analysis System with Fortran Acceleration
Combines conversation analysis, querying, visualization, and temporal tracking.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
"""

from huey_plusplus_conversational_experiment import HueyConversationalNetwork
from huey_query_engine import HueyQueryEngine
from huey_interactive_dashboard import HueyInteractiveDashboard
from complete_conversation_analysis_system import HueyCompleteSystem
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np

class HueyCompletePlatform:
    """
    Complete Huey platform integrating all analysis capabilities.
    Provides unified interface for conversation analysis, querying, and visualization.
    """
    
    def __init__(self, session_name: str = None, max_neurons: int = 500, 
                 window_size: int = 7, learning_rate: float = 0.15):
        """Initialize the complete Huey platform."""
        
        # Core components
        self.session_name = session_name or f"huey_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.network = HueyConversationalNetwork(
            max_neurons=max_neurons,
            window_size=window_size, 
            learning_rate=learning_rate
        )
        self.query_engine = HueyQueryEngine(self.network)
        self.dashboard = None  # Initialized on demand
        
        # Session tracking
        self.session_data = {
            'session_name': self.session_name,
            'created_at': datetime.now().isoformat(),
            'parameters': {
                'max_neurons': max_neurons,
                'window_size': window_size,
                'learning_rate': learning_rate
            },
            'conversation_sessions': [],
            'query_history': [],
            'analysis_snapshots': []
        }
        
        # Temporal tracking
        self.temporal_snapshots = []
        self.comparison_baselines = {}
        
        # Enable HueyTime temporal learning in Huey+
        self.network.enable_temporal_learning(method="lagged", max_lag=8, tau=3.0)
        
        print("🧠 HUEY+ COMPLETE PLATFORM INITIALIZED")
        print("=" * 60)
        print(f"   Session: {self.session_name}")
        print(f"   Network: {max_neurons} neurons, window size {window_size}")
        print(f"   Learning rate: {learning_rate}")
        print("   Components:")
        print("   ✅ Conversational Network")
        print("   ✅ Query Engine") 
        print("   ✅ Temporal Tracking")
        print("   ✅ HueyTime Temporal Learning")
        print("   🔄 Interactive Dashboard (on demand)")
        print()
    
    def register_speakers(self, speakers_info: List[tuple]) -> List[str]:
        """
        Register speakers for the session.
        
        Args:
            speakers_info: List of tuples (speaker_id, full_name, role?)
            
        Returns:
            List of registered speaker IDs
        """
        print("👥 REGISTERING SPEAKERS:")
        
        registered_speakers = []
        for speaker_info in speakers_info:
            speaker_id = speaker_info[0]
            
            # Add speaker to network with default pronouns
            self.network.add_speaker(
                speaker_id, 
                ['i', 'me', 'my', 'myself'], 
                ['you', 'your', 'yours']
            )
            
            registered_speakers.append(speaker_id)
            print(f"   ✅ {speaker_id}")
        
        # Update session data
        self.session_data['speakers'] = speakers_info
        return registered_speakers
    
    def process_conversation(self, conversation_data: List[tuple]) -> Dict[str, Any]:
        """
        Process a complete conversation and create analysis snapshot.
        
        Args:
            conversation_data: List of (speaker_id, text) tuples
            
        Returns:
            Analysis results dictionary
        """
        print(f"\n🎙️  PROCESSING CONVERSATION ({len(conversation_data)} exchanges)")
        print("-" * 50)
        
        # Create temporal snapshot before processing
        pre_snapshot = self._create_temporal_snapshot("pre_conversation")
        
        # Process each exchange
        for i, (speaker_id, text) in enumerate(conversation_data):
            print(f"   {i+1:3d}. {speaker_id:10} → {text[:50]}{'...' if len(text) > 50 else ''}")
            self.network.process_speaker_text(speaker_id, text)
        
        # Create post-processing snapshot
        post_snapshot = self._create_temporal_snapshot("post_conversation")
        
        # Generate comprehensive analysis
        analysis = self._generate_comprehensive_analysis()
        
        # Store conversation session
        conversation_session = {
            'timestamp': datetime.now().isoformat(),
            'conversation_data': conversation_data,
            'pre_snapshot': pre_snapshot,
            'post_snapshot': post_snapshot,
            'analysis': analysis
        }
        
        self.session_data['conversation_sessions'].append(conversation_session)
        
        print(f"\n📊 CONVERSATION ANALYSIS COMPLETE")
        print(f"   Total neurons: {self.network.neuron_count}")
        print(f"   Active connections: {self._count_active_connections()}")
        
        return analysis
    
    def query_concepts(self, query_type: str, **kwargs) -> Dict[str, Any]:
        """
        Execute concept queries with automatic temporal tracking.
        
        Args:
            query_type: Type of query to execute
            **kwargs: Query parameters
            
        Returns:
            Query results with temporal context
        """
        print(f"\n🔍 EXECUTING QUERY: {query_type}")
        
        # Execute query
        result = self.query_engine.query(query_type, **kwargs)
        
        # Add temporal context
        result['temporal_context'] = {
            'query_timestamp': datetime.now().isoformat(),
            'session_name': self.session_name,
            'conversation_count': len(self.session_data['conversation_sessions']),
            'total_network_mass': sum(self.network.inertial_mass.values()) if hasattr(self.network, 'inertial_mass') else 0
        }
        
        # Store in query history
        self.session_data['query_history'].append(result)
        
        return result
    
    def natural_language_query(self, query_text: str) -> Dict[str, Any]:
        """Execute natural language query with temporal tracking."""
        print(f"\n💬 NATURAL LANGUAGE QUERY: {query_text}")
        
        result = self.query_engine.natural_language_query(query_text)
        
        # Add temporal context
        if 'error' not in result:
            result['temporal_context'] = {
                'query_timestamp': datetime.now().isoformat(),
                'session_name': self.session_name,
                'query_text': query_text
            }
        
        return result
    
    def compare_temporal_snapshots(self, snapshot1_id: str, snapshot2_id: str) -> Dict[str, Any]:
        """
        Compare two temporal snapshots to analyze concept evolution.
        
        Args:
            snapshot1_id: ID of first snapshot
            snapshot2_id: ID of second snapshot
            
        Returns:
            Comparison analysis
        """
        print(f"\n⏳ COMPARING SNAPSHOTS: {snapshot1_id} vs {snapshot2_id}")
        
        snap1 = next((s for s in self.temporal_snapshots if s['snapshot_id'] == snapshot1_id), None)
        snap2 = next((s for s in self.temporal_snapshots if s['snapshot_id'] == snapshot2_id), None)
        
        if not snap1 or not snap2:
            return {'error': 'One or both snapshots not found'}
        
        # Calculate differences
        comparison = {
            'snapshot1': snap1,
            'snapshot2': snap2,
            'neuron_changes': self._compare_neuron_states(snap1['neuron_states'], snap2['neuron_states']),
            'connection_changes': self._compare_connection_matrices(snap1['connection_matrix'], snap2['connection_matrix']),
            'speaker_evolution': self._compare_speaker_analyses(snap1['speaker_analyses'], snap2['speaker_analyses']),
            'temporal_metrics': {
                'time_elapsed': snap2['timestamp'],
                'mass_change': snap2['total_mass'] - snap1['total_mass'],
                'neuron_count_change': snap2['neuron_count'] - snap1['neuron_count']
            }
        }
        
        print(f"   Mass change: {comparison['temporal_metrics']['mass_change']:+.3f}")
        print(f"   Neuron change: {comparison['temporal_metrics']['neuron_count_change']:+d}")
        
        return comparison
    
    def create_analysis_report(self, include_visualizations: bool = True) -> str:
        """
        Generate comprehensive analysis report for the session.
        
        Args:
            include_visualizations: Whether to include visualization links
            
        Returns:
            Markdown report content
        """
        print("\n📋 GENERATING COMPREHENSIVE ANALYSIS REPORT")
        
        report_lines = [
            f"# Huey Analysis Report: {self.session_name}",
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## Session Overview",
            f"- **Created**: {self.session_data['created_at']}",
            f"- **Speakers**: {len(self.network.speakers)}",
            f"- **Conversations**: {len(self.session_data['conversation_sessions'])}",
            f"- **Total Neurons**: {self.network.neuron_count}",
            f"- **Network Parameters**: {self.session_data['parameters']}",
            "",
            "## Network Statistics"
        ]
        
        # Get current network statistics
        stats = self.query_engine.query('network_statistics')
        if 'error' not in stats:
            neuron_stats = stats['neuron_stats']
            connection_stats = stats['connection_stats']
            
            report_lines.extend([
                f"- **Total Mass**: {neuron_stats['total_mass']:.3f}",
                f"- **Active Neurons**: {neuron_stats['active_neurons']}/{neuron_stats['total_neurons']}",
                f"- **Connections**: {connection_stats['total_connections']} total, {connection_stats['strong_connections']} strong",
                f"- **Average Connection Strength**: {connection_stats['average_strength']:.3f}",
                ""
            ])
        
        # Speaker analyses
        report_lines.extend([
            "## Speaker Self-Concept Analysis",
            ""
        ])
        
        for speaker in self.network.speakers:
            analysis = self.network.analyze_speaker_self_concept(speaker)
            report_lines.extend([
                f"### {speaker}",
                f"- **Self-concept Mass**: {analysis['self_concept_mass']:.3f}",
                f"- **Blocks Processed**: {analysis['blocks_processed']}",
                ""
            ])
        
        # Query history
        if self.session_data['query_history']:
            report_lines.extend([
                "## Query History",
                ""
            ])
            
            for i, query in enumerate(self.session_data['query_history'][-10:]):  # Last 10 queries
                if 'query' in query:
                    report_lines.append(f"{i+1}. {query['query']}")
            
            report_lines.append("")
        
        # Temporal evolution
        if len(self.temporal_snapshots) > 1:
            report_lines.extend([
                "## Temporal Evolution",
                "",
                "### Mass Evolution Over Time",
                ""
            ])
            
            for snapshot in self.temporal_snapshots:
                report_lines.append(f"- **{snapshot['snapshot_id']}**: {snapshot['total_mass']:.3f} total mass")
        
        # Conclusions
        report_lines.extend([
            "",
            "## Key Findings",
            "",
            "This Huey analysis demonstrates the emergence of self-concept through Hebbian learning principles.",
            "Each speaker's identity emerges naturally through associative patterns without artificial privileging.",
            "",
            "---",
            "*Generated by Huey: Hebbian Self-Concept Analysis Platform*"
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_filename = f"{self.session_name}_analysis_report.md"
        with open(report_filename, 'w') as f:
            f.write(report_content)
        
        print(f"   📁 Report saved to: {report_filename}")
        
        return report_content
    
    def launch_dashboard(self, port: int = 8050, debug: bool = False):
        """Launch interactive web dashboard."""
        print(f"\n🌐 LAUNCHING INTERACTIVE DASHBOARD")
        
        if not self.dashboard:
            self.dashboard = HueyInteractiveDashboard(self.network, self.query_engine)
        
        self.dashboard.run(port=port, debug=debug)
    
    def export_session_data(self, filename: str = None) -> str:
        """
        Export complete session data to JSON file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if not filename:
            filename = f"{self.session_name}_complete_data.json"
        
        # Prepare exportable data
        export_data = {
            'session_metadata': self.session_data,
            'network_state': self._serialize_network_state(),
            'temporal_snapshots': self.temporal_snapshots,
            'query_results': self.session_data['query_history']
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📁 Session data exported to: {filename}")
        return filename
    
    def _create_temporal_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Create a temporal snapshot of the current network state."""
        
        # Calculate total mass from inertial_mass dictionary
        total_mass = sum(self.network.inertial_mass.values()) if hasattr(self.network, 'inertial_mass') else 0
        
        # Create neuron states from word_to_neuron mapping
        neuron_states = []
        if hasattr(self.network, 'neuron_to_word') and hasattr(self.network, 'activations'):
            for neuron_id, word in self.network.neuron_to_word.items():
                activation = self.network.activations.get(neuron_id, 0)
                neuron_states.append((word, 0, float(activation)))  # No individual neuron mass in this structure
        
        # Convert connections to serializable format
        connections_serializable = {}
        if hasattr(self.network, 'connections'):
            for key, value in self.network.connections.items():
                connections_serializable[str(key)] = value
        
        snapshot = {
            'snapshot_id': snapshot_id,
            'timestamp': datetime.now().isoformat(),
            'neuron_count': self.network.neuron_count,
            'total_mass': total_mass,
            'neuron_states': neuron_states,
            'connections': connections_serializable,
            'speaker_analyses': {speaker: self.network.analyze_speaker_self_concept(speaker)
                               for speaker in self.network.speakers}
        }
        
        self.temporal_snapshots.append(snapshot)
        return snapshot
    
    def _generate_comprehensive_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of current network state."""
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'network_statistics': self.query_engine.query('network_statistics'),
            'speaker_analyses': {},
            'concept_emergence': self.query_engine.query('concept_emergence'),
            'top_concepts': []
        }
        
        # Individual speaker analyses
        for speaker in self.network.speakers:
            speaker_analysis = self.network.analyze_speaker_self_concept(speaker)
            analysis['speaker_analyses'][speaker] = speaker_analysis
        
        # Find top concepts by mass (using connection masses as proxy)
        concept_masses = []
        if hasattr(self.network, 'neuron_to_word') and hasattr(self.network, 'inertial_mass'):
            # Calculate concept strength from associated connection masses
            concept_strength = {}
            for conn_key, mass in self.network.inertial_mass.items():
                for neuron_id in conn_key:
                    if neuron_id in self.network.neuron_to_word:
                        word = self.network.neuron_to_word[neuron_id]
                        concept_strength[word] = concept_strength.get(word, 0) + mass
            
            concept_masses = list(concept_strength.items())
            concept_masses.sort(key=lambda x: x[1], reverse=True)
        
        analysis['top_concepts'] = concept_masses[:20]
        
        return analysis
    
    def _count_active_connections(self) -> int:
        """Count active connections in the network."""
        if hasattr(self.network, 'connections'):
            return len([conn for conn, strength in self.network.connections.items() if strength > 0.01])
        return 0
    
    def _compare_neuron_states(self, state1: List[tuple], state2: List[tuple]) -> Dict[str, Any]:
        """Compare neuron states between snapshots."""
        # This would implement detailed neuron comparison logic
        return {
            'added_neurons': len(state2) - len(state1),
            'mass_changes': 'detailed_comparison_would_go_here'
        }
    
    def _compare_connection_matrices(self, matrix1: List[List], matrix2: List[List]) -> Dict[str, Any]:
        """Compare connection matrices between snapshots."""
        # This would implement detailed connection comparison logic
        return {
            'connection_changes': 'detailed_comparison_would_go_here'
        }
    
    def _compare_speaker_analyses(self, analyses1: Dict, analyses2: Dict) -> Dict[str, Any]:
        """Compare speaker analyses between snapshots."""
        comparisons = {}
        
        for speaker in analyses1.keys():
            if speaker in analyses2:
                mass_change = analyses2[speaker]['self_concept_mass'] - analyses1[speaker]['self_concept_mass']
                comparisons[speaker] = {
                    'mass_change': mass_change,
                    'blocks_change': analyses2[speaker]['blocks_processed'] - analyses1[speaker]['blocks_processed']
                }
        
        return comparisons
    
    def _serialize_network_state(self) -> Dict[str, Any]:
        """Serialize current network state for export."""
        neuron_data = []
        if hasattr(self.network, 'neuron_to_word') and hasattr(self.network, 'activations'):
            for neuron_id, word in self.network.neuron_to_word.items():
                activation = self.network.activations.get(neuron_id, 0)
                neuron_data.append((word, 0, float(activation)))  # No individual neuron mass
        
        # Convert tuple keys to strings for JSON serialization
        connections_serializable = {}
        if hasattr(self.network, 'connections'):
            for key, value in self.network.connections.items():
                connections_serializable[str(key)] = value
        
        inertial_mass_serializable = {}
        if hasattr(self.network, 'inertial_mass'):
            for key, value in self.network.inertial_mass.items():
                inertial_mass_serializable[str(key)] = value
        
        return {
            'neurons': neuron_data,
            'connections': connections_serializable,
            'inertial_mass': inertial_mass_serializable,
            'speakers': dict(self.network.speakers),
            'conversation_history': self.network.conversation_history
        }

def demo_complete_platform():
    """Demonstration of the complete Huey platform."""
    
    print("🧠 HUEY COMPLETE PLATFORM DEMO")
    print("=" * 60)
    
    # Initialize platform
    platform = HueyCompletePlatform(
        session_name="demo_session",
        max_neurons=100,
        window_size=7,
        learning_rate=0.15
    )
    
    # Register speakers
    speakers = [
        ("alice", "Alice Smith", "researcher"),
        ("bob", "Bob Johnson", "participant")
    ]
    platform.register_speakers(speakers)
    
    # Process sample conversation
    conversation = [
        ("alice", "I think this approach to self-concept analysis is fascinating."),
        ("bob", "I agree. My understanding of identity has always been that it emerges from interactions."),
        ("alice", "Exactly! My research shows that Hebbian learning explains how I develop self-awareness."),
        ("bob", "That makes sense to me. I can see how my own sense of self forms through repeated patterns.")
    ]
    
    analysis = platform.process_conversation(conversation)
    
    # Execute various queries
    queries = [
        ("cluster_fellows", {"concept": "me"}),
        ("strongest_associations", {"concept": "i", "top_n": 5}),
        ("speaker_differences", {"speakers": ["alice", "bob"]}),
        ("network_statistics", {})
    ]
    
    print("\n🔍 EXECUTING DEMO QUERIES:")
    for query_type, kwargs in queries:
        result = platform.query_concepts(query_type, **kwargs)
        print(f"   ✅ {query_type}: {len(str(result))} characters of results")
    
    # Test natural language query
    nl_result = platform.natural_language_query("show me cluster fellows for 'myself'")
    print(f"   ✅ Natural language query: {'success' if 'error' not in nl_result else 'error'}")
    
    # Generate report
    report = platform.create_analysis_report()
    print(f"   📋 Analysis report: {len(report)} characters")
    
    # Export session data
    export_file = platform.export_session_data()
    print(f"   📁 Data exported to: {export_file}")
    
    print("\n🎯 DEMO COMPLETE - Platform ready for real analysis!")
    
    return platform

if __name__ == "__main__":
    # Run demonstration
    demo_platform = demo_complete_platform()
    
    # Optionally launch dashboard
    # demo_platform.launch_dashboard()