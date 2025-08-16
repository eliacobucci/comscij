#!/usr/bin/env python3
"""
Run Huey: Simple command-line interface for Huey platform
This is the easiest way to get started with Huey analysis.
"""

import sys
import argparse
from datetime import datetime
from huey_complete_platform import HueyCompletePlatform

def main():
    """Main command-line interface for Huey."""
    
    parser = argparse.ArgumentParser(
        description="Huey: Hebbian Self-Concept Analysis Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_huey.py demo                    # Run demo with sample data
  python run_huey.py notebook               # Instructions for Jupyter notebook
  python run_huey.py dashboard              # Launch web dashboard
  python run_huey.py --help                 # Show this help
        """
    )
    
    parser.add_argument(
        'mode',
        choices=['demo', 'notebook', 'dashboard', 'interactive'],
        help='Mode to run Huey in'
    )
    
    parser.add_argument(
        '--session-name',
        type=str,
        help='Name for this analysis session'
    )
    
    parser.add_argument(
        '--max-neurons',
        type=int,
        default=500,
        help='Maximum number of neurons (concepts) in network'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=7,
        help='Sliding window size for Hebbian learning'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.15,
        help='Learning rate for concept association formation'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port for web dashboard (dashboard mode only)'
    )
    
    args = parser.parse_args()
    
    print("🧠 HUEY: HEBBIAN SELF-CONCEPT ANALYSIS PLATFORM")
    print("=" * 60)
    
    if args.mode == 'demo':
        run_demo(args)
    elif args.mode == 'notebook':
        show_notebook_instructions()
    elif args.mode == 'dashboard':
        run_dashboard(args)
    elif args.mode == 'interactive':
        run_interactive(args)

def run_demo(args):
    """Run the Huey demo with sample data."""
    
    print("🎯 RUNNING HUEY DEMO")
    print("-" * 30)
    
    # Initialize platform
    session_name = args.session_name or f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    huey = HueyCompletePlatform(
        session_name=session_name,
        max_neurons=args.max_neurons,
        window_size=args.window_size,
        learning_rate=args.learning_rate
    )
    
    # Register demo speakers
    speakers = [
        ("alice", "Alice Smith", "researcher"),
        ("bob", "Bob Johnson", "participant"),
        ("charlie", "Charlie Brown", "observer")
    ]
    huey.register_speakers(speakers)
    
    # Process demo conversation
    conversation = [
        ("alice", "I think this approach to self-concept analysis is fascinating."),
        ("bob", "I agree. My understanding of identity has always been that it emerges from interactions."),
        ("alice", "Exactly! My research shows that Hebbian learning explains how I develop self-awareness."),
        ("bob", "That makes sense to me. I can see how my own sense of self forms through repeated patterns."),
        ("charlie", "This is interesting. I wonder how my own identity compares to yours."),
        ("alice", "What I find most intriguing is how my sense of self changes as I interact with different people."),
        ("bob", "Yes, I notice that too. My identity shifts depending on who I'm talking with."),
        ("charlie", "I think my personality is pretty consistent, but maybe my self-concept does evolve."),
        ("alice", "The beauty of Hebbian learning is that it shows identity as emergent, not fixed."),
        ("bob", "So my sense of who I am is really just patterns of association?"),
        ("alice", "Exactly! Your self-concept is sophisticated pattern matching, nothing mystical."),
        ("charlie", "That's both fascinating and a little unsettling. My identity is just patterns?"),
        ("bob", "But those patterns make me who I am. They're real even if they're emergent."),
        ("alice", "Precisely! Huey demonstrates that emergence doesn't diminish authenticity.")
    ]
    
    print(f"\n🎙️  Processing {len(conversation)} conversation exchanges...")
    analysis = huey.process_conversation(conversation)
    
    # Run demo queries
    demo_queries = [
        ("cluster_fellows", {"concept": "me", "threshold": 0.05}),
        ("strongest_associations", {"concept": "i", "top_n": 8}),
        ("speaker_differences", {"speakers": ["alice", "bob", "charlie"]}),
        ("concept_emergence", {"min_mass": 0.01}),
        ("network_statistics", {})
    ]
    
    print("\n🔍 EXECUTING DEMO QUERIES:")
    print("-" * 30)
    
    for query_type, kwargs in demo_queries:
        print(f"\n📊 {query_type.upper()}:")
        result = huey.query_concepts(query_type, **kwargs)
        
        if 'error' in result:
            print(f"   ❌ {result['error']}")
        else:
            # Show relevant results based on query type
            if query_type == "cluster_fellows":
                if 'fellow_concepts' in result:
                    fellows = result['fellow_concepts'][:5]  # Top 5
                    for i, fellow in enumerate(fellows):
                        print(f"   {i+1}. {fellow['concept']:15} → {fellow['strength']:.3f}")
                        
            elif query_type == "strongest_associations":
                if 'associations' in result:
                    for i, assoc in enumerate(result['associations'][:5]):
                        print(f"   {i+1}. {assoc['concept']:15} → {assoc['strength']:.3f}")
                        
            elif query_type == "speaker_differences":
                if 'individual_analyses' in result:
                    for speaker, analysis in result['individual_analyses'].items():
                        mass = analysis.get('self_concept_mass', 0)
                        print(f"   {speaker:10} → self-concept mass: {mass:.3f}")
                        
            elif query_type == "concept_emergence":
                if 'emerged_concepts' in result:
                    emerged = result['emerged_concepts'][:5]  # First 5
                    for concept in emerged:
                        step = concept['emergence_step']
                        name = concept['concept']
                        mass = concept['current_mass']
                        print(f"   Step {step:2d}: {name:15} (mass: {mass:.3f})")
                        
            elif query_type == "network_statistics":
                if 'neuron_stats' in result:
                    neuron_stats = result['neuron_stats']
                    connection_stats = result['connection_stats']
                    print(f"   Neurons: {neuron_stats['total_neurons']} total, {neuron_stats['active_neurons']} active")
                    print(f"   Connections: {connection_stats['total_connections']} total")
                    print(f"   Total mass: {neuron_stats['total_mass']:.3f}")
    
    # Test natural language queries
    print("\n💬 NATURAL LANGUAGE QUERIES:")
    print("-" * 30)
    
    nl_queries = [
        "show me cluster fellows for 'myself'",
        "what are the strongest associations with 'identity'?",
        "compare alice and bob's self concepts"
    ]
    
    for query_text in nl_queries:
        print(f"\n❓ {query_text}")
        result = huey.natural_language_query(query_text)
        if 'error' in result:
            print(f"   ❌ {result['error']}")
        else:
            print(f"   ✅ Query executed successfully")
    
    # Generate report
    print("\n📋 GENERATING ANALYSIS REPORT:")
    print("-" * 30)
    report = huey.create_analysis_report()
    
    # Export data
    print("\n💾 EXPORTING SESSION DATA:")
    print("-" * 30)
    export_file = huey.export_session_data()
    
    print(f"\n🎯 DEMO COMPLETE!")
    print("=" * 60)
    print("Files created:")
    print(f"  📋 Analysis report: {session_name}_analysis_report.md")
    print(f"  💾 Session data: {export_file}")
    print("\nNext steps:")
    print("  1. Review the analysis report")
    print("  2. Try the Jupyter notebook: python run_huey.py notebook")
    print("  3. Launch the dashboard: python run_huey.py dashboard")

def show_notebook_instructions():
    """Show instructions for using the Jupyter notebook."""
    
    print("📓 JUPYTER NOTEBOOK INTERFACE")
    print("-" * 30)
    print("\nTo use Huey with Jupyter notebook:")
    print("\n1. Make sure you have Jupyter installed:")
    print("   pip install jupyter matplotlib")
    print("\n2. Launch Jupyter notebook:")
    print("   jupyter notebook")
    print("\n3. Open the Huey notebook:")
    print("   huey_notebook_interface.ipynb")
    print("\n4. Run the cells step by step to:")
    print("   • Initialize Huey with your parameters")
    print("   • Register your speakers")
    print("   • Process your conversation data")
    print("   • Execute queries and generate reports")
    print("\nThe notebook provides an interactive, step-by-step")
    print("interface perfect for research and exploration.")
    print("\n📁 Notebook file: huey_notebook_interface.ipynb")

def run_dashboard(args):
    """Launch the interactive web dashboard."""
    
    print("🌐 LAUNCHING INTERACTIVE DASHBOARD")
    print("-" * 30)
    
    session_name = args.session_name or f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Initialize platform with sample data
        huey = HueyCompletePlatform(
            session_name=session_name,
            max_neurons=args.max_neurons,
            window_size=args.window_size,
            learning_rate=args.learning_rate
        )
        
        # Add some sample data for demonstration
        speakers = [("alice", "Alice", "user"), ("bob", "Bob", "user")]
        huey.register_speakers(speakers)
        
        sample_conversation = [
            ("alice", "I think this analysis approach is really interesting."),
            ("bob", "I agree, my understanding of self-concept has evolved through this.")
        ]
        huey.process_conversation(sample_conversation)
        
        print(f"Dashboard will be available at: http://localhost:{args.port}")
        print("Use Ctrl+C to stop the dashboard server")
        print("\nFeatures available in the dashboard:")
        print("  • Interactive concept network visualization")
        print("  • Real-time query interface")
        print("  • Natural language query processing")
        print("  • 3D concept space exploration")
        print("  • Temporal evolution tracking")
        
        # Launch dashboard
        huey.launch_dashboard(port=args.port, debug=False)
        
    except ImportError as e:
        print(f"❌ Missing dependencies for dashboard: {e}")
        print("\nTo install dashboard dependencies:")
        print("pip install dash plotly networkx pandas")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def run_interactive(args):
    """Run interactive command-line interface."""
    
    print("💻 INTERACTIVE HUEY SESSION")
    print("-" * 30)
    
    session_name = args.session_name or f"interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    huey = HueyCompletePlatform(
        session_name=session_name,
        max_neurons=args.max_neurons,
        window_size=args.window_size,
        learning_rate=args.learning_rate
    )
    
    print(f"\nWelcome to Huey interactive session: {session_name}")
    print("Type 'help' for commands, 'quit' to exit")
    
    while True:
        try:
            command = input("\nhuey> ").strip()
            
            if command in ['quit', 'exit', 'q']:
                print("Goodbye! 🧠")
                break
            elif command == 'help':
                show_interactive_help()
            elif command.startswith('add_speaker'):
                handle_add_speaker(huey, command)
            elif command.startswith('process'):
                handle_process_conversation(huey, command)
            elif command.startswith('query'):
                handle_query(huey, command)
            elif command == 'stats':
                show_quick_stats(huey)
            elif command == 'report':
                huey.create_analysis_report()
            elif command == 'export':
                huey.export_session_data()
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\nUse 'quit' to exit")
        except Exception as e:
            print(f"Error: {e}")

def show_interactive_help():
    """Show help for interactive mode."""
    print("\nHuey Interactive Commands:")
    print("  add_speaker <id> <name>     - Add a speaker")
    print("  process <speaker> <text>    - Process speaker text")
    print("  query <type> <concept>      - Execute query")
    print("  stats                       - Show network statistics")
    print("  report                      - Generate analysis report")
    print("  export                      - Export session data")
    print("  help                        - Show this help")
    print("  quit                        - Exit Huey")

def handle_add_speaker(huey, command):
    """Handle add_speaker command."""
    parts = command.split(' ', 2)
    if len(parts) >= 3:
        speaker_id = parts[1]
        speaker_name = parts[2]
        huey.register_speakers([(speaker_id, speaker_name, "user")])
        print(f"Added speaker: {speaker_id}")
    else:
        print("Usage: add_speaker <id> <name>")

def handle_process_conversation(huey, command):
    """Handle process command."""
    parts = command.split(' ', 2)
    if len(parts) >= 3:
        speaker_id = parts[1]
        text = parts[2]
        huey.network.process_speaker_text(speaker_id, text)
        print(f"Processed text for {speaker_id}")
    else:
        print("Usage: process <speaker> <text>")

def handle_query(huey, command):
    """Handle query command."""
    parts = command.split(' ', 2)
    if len(parts) >= 3:
        query_type = parts[1]
        concept = parts[2]
        result = huey.query_concepts(query_type, concept=concept)
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print("Query executed successfully")
    else:
        print("Usage: query <type> <concept>")
        print("Types: cluster_fellows, strongest_associations, network_statistics")

def show_quick_stats(huey):
    """Show quick network statistics."""
    stats = huey.query_concepts("network_statistics")
    if 'neuron_stats' in stats:
        neuron_stats = stats['neuron_stats']
        print(f"Neurons: {neuron_stats['total_neurons']}")
        print(f"Total mass: {neuron_stats['total_mass']:.3f}")
        print(f"Speakers: {len(huey.network.speakers)}")

if __name__ == "__main__":
    main()