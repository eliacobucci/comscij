#!/usr/bin/env python3
"""
3D Visualization of Self-Concept Clusters from Conversational Experiments
Creates 3D plots showing how self-concepts form and cluster in our neural networks.
"""

from conversational_self_concept_experiment import ConversationalSelfConceptNetwork, create_test_conversation
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

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

def identify_self_concept_clusters(cluster_data):
    """Identify self-concept related clusters in the network."""
    
    clusters = {
        'claude_self': [],
        'joseph_self': [],
        'shared_concepts': [],
        'other_concepts': []
    }
    
    # Claude self-pronouns and references
    claude_self_words = {'i', 'me', 'my', 'mine', 'myself', 'claude'}
    joseph_self_words = {'you', 'your', 'yours', 'yourself', 'joseph'}
    shared_concepts = {'research', 'work', 'understanding', 'conversation', 'analysis', 'thinking'}
    
    for neuron_id, word in cluster_data['neuron_to_word'].items():
        coord = cluster_data['coordinates'][neuron_id]
        mass = cluster_data['neuron_masses'].get(neuron_id, 0.0)
        
        if word.lower() in claude_self_words:
            clusters['claude_self'].append({
                'word': word,
                'neuron_id': neuron_id,
                'coord': coord,
                'mass': mass
            })
        elif word.lower() in joseph_self_words:
            clusters['joseph_self'].append({
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
    
    # Color scheme for different cluster types
    colors = {
        'claude_self': 'red',
        'joseph_self': 'blue', 
        'shared_concepts': 'green',
        'other_concepts': 'lightgray'
    }
    
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
    
    # Cross-cluster analysis
    print(f"\n🔄 CROSS-CLUSTER COMPARISON:")
    if clusters['claude_self'] and clusters['joseph_self']:
        claude_mass = sum(item['mass'] for item in clusters['claude_self'])
        joseph_mass = sum(item['mass'] for item in clusters['joseph_self'])
        
        print(f"   Claude self-concept mass: {claude_mass:.3f}")
        print(f"   Joseph self-concept mass: {joseph_mass:.3f}")
        print(f"   Mass ratio (Claude/Joseph): {claude_mass/joseph_mass:.3f}" if joseph_mass > 0 else "   Mass ratio: undefined")
        
        # Calculate distance between self-concept clusters
        claude_coords = np.array([item['coord'] for item in clusters['claude_self']])
        joseph_coords = np.array([item['coord'] for item in clusters['joseph_self']])
        
        claude_centroid = np.mean(claude_coords, axis=0)
        joseph_centroid = np.mean(joseph_coords, axis=0)
        
        inter_cluster_distance = np.linalg.norm(claude_centroid - joseph_centroid)
        print(f"   Inter-cluster distance: {inter_cluster_distance:.3f}")

def run_self_concept_cluster_visualization():
    """Run the complete self-concept cluster visualization."""
    
    print("🧠 SELF-CONCEPT CLUSTER VISUALIZATION")
    print("=" * 60)
    print("Creating 3D visualization of conversational self-concept formation...")
    print()
    
    # Create and run the conversational network
    net = ConversationalSelfConceptNetwork(max_neurons=150)
    
    # Define speakers
    net.add_speaker("Claude", 
                    self_pronouns=['i', 'me', 'my', 'mine', 'myself'],
                    other_pronouns=['you', 'your', 'yours', 'yourself'])
    
    net.add_speaker("Joseph",
                    self_pronouns=['i', 'me', 'my', 'mine', 'myself'], 
                    other_pronouns=['you', 'your', 'yours', 'yourself'])
    
    # Get conversation and process it
    conversation = create_test_conversation()
    
    print("📝 Processing conversation for visualization...")
    for speaker, text_block in conversation:
        net.process_speaker_block(speaker, text_block)
    
    print(f"\n🧠 Network state: {net.neuron_count} neurons, {len(net.connections)} connections")
    
    # Extract cluster data
    cluster_data = extract_cluster_data(net)
    if cluster_data is None:
        print("❌ Failed to extract cluster data")
        return
    
    # Identify self-concept clusters
    clusters = identify_self_concept_clusters(cluster_data)
    
    # Analyze cluster properties
    analyze_cluster_properties(clusters)
    
    # Create 3D visualization
    fig, ax = create_3d_self_concept_plot(cluster_data, clusters)
    
    # Save the plot
    filename = '/Users/josephwoelfel/asa/self_concept_clusters_3d.png'
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
    
    with open('/Users/josephwoelfel/asa/cluster_analysis.json', 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    
    print(f"📊 Cluster statistics saved: cluster_analysis.json")
    
    return {
        'network': net,
        'cluster_data': cluster_data,
        'clusters': clusters,
        'stats': cluster_stats
    }

if __name__ == "__main__":
    results = run_self_concept_cluster_visualization()
    
    print(f"\n🎯 VISUALIZATION COMPLETE")
    print("Self-concept clusters mapped in 3D cognitive space!")