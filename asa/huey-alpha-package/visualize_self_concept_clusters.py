#!/usr/bin/env python3
"""
3D Visualization of Self-Concept Clusters from Conversational Experiments
Creates 3D plots showing how self-concepts form and cluster in our neural networks.
"""

from conversational_self_concept_experiment import HueyConversationalNetwork, create_test_conversation
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse
import sys

def extract_cluster_data(network):
    """Extract 3D coordinates and cluster information from the network."""
    
    if network.neuron_count == 0:
        print("❌ No neurons in network")
        return None
    
    # Build connection matrix
    full_matrix = np.zeros((network.neuron_count, network.neuron_count))
    for (i, j), strength in network.connections.items():
        if i < network.neuron_count and j < network.neuron_count:
            full_matrix[i][j] = strength
            full_matrix[j][i] = strength  # Make symmetric
    
    # Apply Torgerson transformation for pseudo-Euclidean space
    try:
        gram_matrix = network._torgerson_transform_proper(full_matrix)
    except AttributeError:
        # Simple fallback if method doesn't exist
        gram_matrix = -0.5 * (full_matrix - np.mean(full_matrix, axis=0) - np.mean(full_matrix, axis=1).reshape(-1, 1) + np.mean(full_matrix))
    
    # Get eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)
    
    # Sort by eigenvalue magnitude
    idx = np.argsort(np.abs(eigenvals))[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    # Get 3D coordinates using first 3 eigenvectors
    coords_3d = np.zeros((network.neuron_count, 3))
    for i in range(min(3, len(eigenvals))):
        eigenval = eigenvals[i]
        eigenvec = eigenvecs[:, i]
        
        if eigenval > 1e-8:
            coords_3d[:, i] = eigenvec * np.sqrt(eigenval)
        elif eigenval < -1e-8:
            coords_3d[:, i] = eigenvec * np.sqrt(-eigenval)
        else:
            coords_3d[:, i] = eigenvec * 1e-3
    
    # Calculate inertial masses for each neuron
    neuron_masses = {}
    for i in range(network.neuron_count):
        if i in network.neuron_to_word:
            total_mass = 0.0
            for conn_key, mass in network.inertial_mass.items():
                if i in conn_key:
                    total_mass += mass
            neuron_masses[i] = total_mass
    
    return {
        'coordinates': coords_3d,
        'eigenvalues': eigenvals,
        'neuron_masses': neuron_masses,
        'word_to_neuron': network.word_to_neuron,
        'neuron_to_word': network.neuron_to_word,
        'connections': network.connections,
        'speakers': network.speakers
    }

def identify_self_concept_clusters(cluster_data, speakers=None):
    """Identify self-concept related clusters in the network dynamically."""
    
    # If no speakers provided, detect from conversation data
    if speakers is None:
        speakers = ['claude', 'joseph']  # Default fallback
    
    # Create dynamic cluster structure
    clusters = {}
    for speaker in speakers:
        clusters[f'{speaker.lower()}_self'] = []
    clusters['shared_concepts'] = []
    clusters['other_concepts'] = []
    
    # Dynamic self-pronoun mapping
    self_pronouns = {'i', 'me', 'my', 'mine', 'myself'}
    shared_concepts = {'research', 'work', 'understanding', 'conversation', 'analysis', 'thinking'}
    
    for neuron_id, word in cluster_data['neuron_to_word'].items():
        coord = cluster_data['coordinates'][neuron_id]
        mass = cluster_data['neuron_masses'].get(neuron_id, 0.0)
        
        word_lower = word.lower()
        
        # Check if word is self-pronoun
        if word_lower in self_pronouns:
            # Assign to first speaker's self-concept (since pronouns are ambiguous)
            speaker_key = f'{speakers[0].lower()}_self'
            clusters[speaker_key].append({
                'word': word,
                'neuron_id': neuron_id,
                'coord': coord,
                'mass': mass
            })
        # Check if word matches any speaker name
        elif any(word_lower == speaker.lower() for speaker in speakers):
            # Find which speaker this matches
            matching_speaker = next(s for s in speakers if s.lower() == word_lower)
            speaker_key = f'{matching_speaker.lower()}_self'
            clusters[speaker_key].append({
                'word': word,
                'neuron_id': neuron_id,
                'coord': coord,
                'mass': mass
            })
        elif word.lower() in shared_concepts:
            clusters['shared_concepts'].append({
                'word': word,
                'neuron_id': neuron_id,
                'coord': coord,
                'mass': mass
            })
        else:
            clusters['other_concepts'].append({
                'word': word,
                'neuron_id': neuron_id,
                'coord': coord,
                'mass': mass
            })
    
    return clusters

def create_3d_self_concept_plot(cluster_data, clusters):
    """Create 3D visualization of self-concept clusters."""
    
    print("🎨 Creating 3D self-concept cluster visualization...")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dynamic color scheme for any number of speakers
    base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = {}
    
    # Generate colors for speaker clusters
    speaker_clusters = [name for name in clusters.keys() if name.endswith('_self')]
    for i, cluster_name in enumerate(speaker_clusters):
        colors[cluster_name] = base_colors[i % len(base_colors)]
    
    # Fixed colors for special clusters
    colors['shared_concepts'] = 'green'
    colors['other_concepts'] = 'lightgray'
    
    # Plot each cluster type
    for cluster_name, cluster_items in clusters.items():
        if not cluster_items:
            continue
            
        coords = np.array([item['coord'] for item in cluster_items])
        masses = np.array([item['mass'] for item in cluster_items])
        words = [item['word'] for item in cluster_items]
        
        # Calculate sphere sizes based on inertial mass
        # Use linear scaling for better visual differentiation
        base_size = 50
        mass_scaling = 300
        sizes = base_size + masses * mass_scaling
        
        # Ensure minimum size visibility
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
    
    # Add connection lines for strong self-concept connections
    connection_threshold = 0.1
    
    for (i, j), strength in cluster_data['connections'].items():
        if strength > connection_threshold:
            if i in cluster_data['neuron_to_word'] and j in cluster_data['neuron_to_word']:
                word1 = cluster_data['neuron_to_word'][i]
                word2 = cluster_data['neuron_to_word'][j]
                
                # Only show connections involving self-pronouns
                if any(word.lower() in {'i', 'me', 'my', 'you', 'your', 'claude', 'joseph'} 
                       for word in [word1.lower(), word2.lower()]):
                    
                    coord1 = cluster_data['coordinates'][i]
                    coord2 = cluster_data['coordinates'][j]
                    
                    ax.plot([coord1[0], coord2[0]], 
                           [coord1[1], coord2[1]], 
                           [coord1[2], coord2[2]], 
                           'k-', alpha=0.3, linewidth=strength*3)
    
    # Set labels and title
    eigenvals = cluster_data['eigenvalues']
    ax.set_xlabel(f'Eigenvector 1 (λ={eigenvals[0]:.3f})')
    ax.set_ylabel(f'Eigenvector 2 (λ={eigenvals[1]:.3f})')
    ax.set_zlabel(f'Eigenvector 3 (λ={eigenvals[2]:.3f})')
    
    pos_dims = np.sum(eigenvals > 1e-8)
    neg_dims = np.sum(eigenvals < -1e-8)
    ax.set_title(f'Self-Concept Clusters in 3D Cognitive Space\n'
                f'Signature: (+{pos_dims}, -{neg_dims}) | Sphere size ∝ Inertial Mass')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    return fig, ax

def analyze_cluster_properties(clusters):
    """Analyze properties of the self-concept clusters."""
    
    print("\n🔍 CLUSTER ANALYSIS")
    print("=" * 50)
    
    for cluster_name, cluster_items in clusters.items():
        if not cluster_items:
            continue
            
        total_mass = sum(item['mass'] for item in cluster_items)
        avg_mass = total_mass / len(cluster_items) if cluster_items else 0
        
        print(f"\n📊 {cluster_name.replace('_', ' ').upper()} CLUSTER:")
        print(f"   Concepts: {len(cluster_items)}")
        print(f"   Total mass: {total_mass:.3f}")
        print(f"   Average mass: {avg_mass:.3f}")
        print(f"   Words: {[item['word'] for item in cluster_items]}")
        
        if cluster_items:
            # Calculate cluster centroid
            coords = np.array([item['coord'] for item in cluster_items])
            centroid = np.mean(coords, axis=0)
            
            # Calculate cluster spread (average distance from centroid)
            distances = [np.linalg.norm(coord - centroid) for coord in coords]
            avg_spread = np.mean(distances)
            
            print(f"   Centroid: ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f})")
            print(f"   Spread: {avg_spread:.3f}")
    
    # Cross-cluster analysis for speaker self-concepts
    print(f"\n🔄 CROSS-CLUSTER COMPARISON:")
    speaker_clusters = [(name, items) for name, items in clusters.items() 
                       if name.endswith('_self') and items]
    
    if len(speaker_clusters) >= 2:
        # Compare all pairs of speaker clusters
        for i, (name1, items1) in enumerate(speaker_clusters):
            for name2, items2 in speaker_clusters[i+1:]:
                mass1 = sum(item['mass'] for item in items1)
                mass2 = sum(item['mass'] for item in items2)
                
                speaker1 = name1.replace('_self', '').title()
                speaker2 = name2.replace('_self', '').title()
                
                print(f"   {speaker1} self-concept mass: {mass1:.3f}")
                print(f"   {speaker2} self-concept mass: {mass2:.3f}")
                print(f"   Mass ratio ({speaker1}/{speaker2}): {mass1/mass2:.3f}" if mass2 > 0 else "   Mass ratio: undefined")
                
                # Calculate distance between clusters
                coords1 = np.array([item['coord'] for item in items1])
                coords2 = np.array([item['coord'] for item in items2])
                
                centroid1 = np.mean(coords1, axis=0)
                centroid2 = np.mean(coords2, axis=0)
                
                inter_cluster_distance = np.linalg.norm(centroid1 - centroid2)
                print(f"   Inter-cluster distance ({speaker1}-{speaker2}): {inter_cluster_distance:.3f}")
                print()
    else:
        print("   Not enough speaker clusters for comparison")

def run_self_concept_cluster_visualization(custom_conversation=None, max_neurons=150, output_filename=None):
    """
    Run the complete self-concept cluster visualization.
    
    Args:
        custom_conversation: List of (speaker, text) tuples. If None, uses default test conversation.
        max_neurons: Maximum number of neurons in the network
        output_filename: Custom filename for the PNG output. If None, uses default.
    """
    
    print("🧠 SELF-CONCEPT CLUSTER VISUALIZATION")
    print("=" * 60)
    print("Creating 3D visualization of conversational self-concept formation...")
    print()
    
    # Create and run the conversational network
    net = HueyConversationalNetwork(max_neurons=max_neurons)
    
    # Get conversation - use custom or default
    if custom_conversation is not None:
        conversation = custom_conversation
        print("📥 Using custom conversation data")
    else:
        conversation = create_test_conversation()
        print("📥 Using default test conversation")
    
    # Extract unique speakers from conversation
    detected_speakers = list(set(speaker for speaker, _ in conversation))
    print(f"🔍 Detected speakers: {detected_speakers}")
    
    # Add all detected speakers dynamically
    for speaker in detected_speakers:
        net.add_speaker(speaker, 
                        self_pronouns=['i', 'me', 'my', 'mine', 'myself'],
                        other_pronouns=['you', 'your', 'yours', 'yourself'])
    
    print("📝 Processing conversation for visualization...")
    for speaker, text_block in conversation:
        net.process_speaker_text(speaker, text_block)
    
    print(f"\n🧠 Network state: {net.neuron_count} neurons, {len(net.connections)} connections")
    
    # Extract cluster data
    cluster_data = extract_cluster_data(net)
    if cluster_data is None:
        print("❌ Failed to extract cluster data")
        return
    
    # Identify self-concept clusters dynamically
    clusters = identify_self_concept_clusters(cluster_data, detected_speakers)
    
    # Analyze cluster properties
    analyze_cluster_properties(clusters)
    
    # Create 3D visualization
    fig, ax = create_3d_self_concept_plot(cluster_data, clusters)
    
    # Save the plot
    if output_filename is None:
        filename = '/Users/josephwoelfel/asa/self_concept_clusters_3d.png'
    else:
        filename = output_filename
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ 3D visualization saved: {filename}")
    
    # Save cluster data for further analysis
    cluster_stats = {
        'total_neurons': net.neuron_count,
        'total_connections': len(net.connections),
        'cluster_counts': {name: len(items) for name, items in clusters.items()},
        'cluster_masses': {name: sum(item['mass'] for item in items) 
                          for name, items in clusters.items()},
        'eigenvalue_signature': {
            'positive_dims': int(np.sum(cluster_data['eigenvalues'] > 1e-8)),
            'negative_dims': int(np.sum(cluster_data['eigenvalues'] < -1e-8)),
            'top_eigenvalues': cluster_data['eigenvalues'][:5].tolist()
        }
    }
    
    # Optionally save cluster statistics (disabled by default)
    # with open('/Users/josephwoelfel/asa/cluster_analysis.json', 'w') as f:
    #     json.dump(cluster_stats, f, indent=2)
    # print(f"📊 Cluster statistics saved: cluster_analysis.json")
    
    return {
        'network': net,
        'cluster_data': cluster_data,
        'clusters': clusters,
        'stats': cluster_stats
    }

def load_conversation_from_file(filename):
    """
    Load conversation from a text file.
    Expected format:
    Speaker1: Text here
    Speaker2: More text
    
    Or JSON format:
    [["Speaker1", "Text here"], ["Speaker2", "More text"]]
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # Try JSON format first
        if content.startswith('['):
            try:
                conversation = json.loads(content)
                if isinstance(conversation, list) and len(conversation) > 0:
                    if isinstance(conversation[0], list) and len(conversation[0]) == 2:
                        return conversation
            except json.JSONDecodeError:
                pass
        
        # Parse text format
        conversation = []
        for line in content.split('\n'):
            line = line.strip()
            if ':' in line and line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    text = parts[1].strip()
                    if speaker and text:
                        conversation.append((speaker, text))
        
        if conversation:
            return conversation
        else:
            print(f"❌ No valid conversation found in {filename}")
            return None
            
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return None
    except Exception as e:
        print(f"❌ Error loading file {filename}: {e}")
        return None

def create_custom_conversation():
    """Create a custom conversation interactively."""
    print("📝 INTERACTIVE CONVERSATION CREATOR")
    print("=" * 50)
    print("Enter conversation turns. Type 'done' when finished.")
    print("Format: Speaker: What they said")
    print("Example: Alice: Hello there!")
    print()
    
    conversation = []
    while True:
        try:
            line = input("Enter turn (or 'done'): ").strip()
            if line.lower() == 'done':
                break
            
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    text = parts[1].strip()
                    if speaker and text:
                        conversation.append((speaker, text))
                        print(f"  ✅ Added: {speaker}")
                    else:
                        print("  ❌ Please include both speaker name and text")
                else:
                    print("  ❌ Please use format: Speaker: Text")
            else:
                print("  ❌ Please use format: Speaker: Text")
                
        except KeyboardInterrupt:
            print("\n\n🔄 Cancelled.")
            return None
    
    if conversation:
        print(f"\n✅ Created conversation with {len(conversation)} turns")
        return conversation
    else:
        print("❌ No conversation created")
        return None

def main():
    """Main function with CLI argument support."""
    parser = argparse.ArgumentParser(
        description="Create 3D visualization of self-concept formation in conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_self_concept_clusters.py
  python visualize_self_concept_clusters.py --file conversation.txt
  python visualize_self_concept_clusters.py --interactive
  python visualize_self_concept_clusters.py --file data.json --output my_viz.png --neurons 200
        """
    )
    
    parser.add_argument('--file', '-f', type=str, 
                       help='Load conversation from file (JSON or text format)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Create conversation interactively')
    parser.add_argument('--output', '-o', type=str,
                       help='Output filename for PNG visualization')
    parser.add_argument('--neurons', '-n', type=int, default=150,
                       help='Maximum number of neurons (default: 150)')
    
    args = parser.parse_args()
    
    # Determine conversation source
    custom_conversation = None
    
    if args.file:
        custom_conversation = load_conversation_from_file(args.file)
        if custom_conversation is None:
            sys.exit(1)
    elif args.interactive:
        custom_conversation = create_custom_conversation()
        if custom_conversation is None:
            sys.exit(1)
    
    # Run visualization
    try:
        results = run_self_concept_cluster_visualization(
            custom_conversation=custom_conversation,
            max_neurons=args.neurons,
            output_filename=args.output
        )
        
        print(f"\n🎯 VISUALIZATION COMPLETE")
        print("Self-concept clusters mapped in 3D cognitive space!")
        
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()