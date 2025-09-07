#!/usr/bin/env python3
"""
Debug the forward transformation step by step to understand exactly what happens.
This will help us create the perfect inverse.
"""

import numpy as np
import json
from experimental_network import ExperimentalNetwork

def debug_forward_transformation():
    """
    Step through the forward transformation and save all intermediate results.
    """
    print("🔍 DEBUGGING FORWARD TRANSFORMATION")
    print("=" * 50)
    
    # Load our test data to get the original connections
    with open('reverse_test_data.json', 'r') as f:
        data = json.load(f)
    
    # Reconstruct the connections dict from the network that generated our test data
    connections_dict = {}
    original_weights = np.array(data['synaptic_weights'])
    n = len(original_weights)
    
    # Convert matrix back to dict format (sparse)
    for i in range(n):
        for j in range(n):
            if original_weights[i, j] > 0:
                connections_dict[(i, j)] = original_weights[i, j]
    
    print(f"📊 Starting with {len(connections_dict)} connections")
    
    # Now manually perform each step of the forward transformation
    print(f"\n🔄 Step 1: Connections dict → Full matrix")
    full_matrix = np.zeros((n, n))
    for (i, j), strength in connections_dict.items():
        if i < n and j < n:
            full_matrix[i][j] = strength
    
    print(f"   Full matrix shape: {full_matrix.shape}")
    print(f"   Non-zero entries: {np.count_nonzero(full_matrix)}")
    print(f"   Matrix range: [{full_matrix.min():.6f}, {full_matrix.max():.6f}]")
    
    print(f"\n🔄 Step 2: Full matrix → Similarity matrix (bidirectional average)")
    similarity = (full_matrix + full_matrix.T) / 2.0
    print(f"   Similarity matrix range: [{similarity.min():.6f}, {similarity.max():.6f}]")
    print(f"   Similarity non-zeros: {np.count_nonzero(similarity)}")
    
    print(f"\n🔄 Step 3: Similarity → Pseudo-distances")
    max_sim = np.max(np.abs(similarity))
    print(f"   max_sim = {max_sim:.6f}")
    
    if max_sim > 1e-10:
        pseudo_distances = np.sign(similarity) * (max_sim - np.abs(similarity)) / max_sim
    else:
        pseudo_distances = np.zeros_like(similarity)
    
    print(f"   Pseudo-distances range: [{pseudo_distances.min():.6f}, {pseudo_distances.max():.6f}]")
    
    print(f"\n🔄 Step 4: Pseudo-distances → Distance squared")
    distances_squared = np.sign(pseudo_distances) * (pseudo_distances ** 2)
    print(f"   Distance squared range: [{distances_squared.min():.6f}, {distances_squared.max():.6f}]")
    
    print(f"\n🔄 Step 5: Distance squared → Gram matrix (Torgerson)")
    ones = np.ones((n, n))
    centering_matrix = np.eye(n) - (1.0 / n) * ones
    gram_matrix = -0.5 * centering_matrix @ distances_squared @ centering_matrix
    
    print(f"   Gram matrix range: [{gram_matrix.min():.6f}, {gram_matrix.max():.6f}]")
    
    print(f"\n🔄 Step 6: Gram matrix → Eigendecomposition")
    eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)
    
    # Sort by absolute value (as in original code)
    idx = np.argsort(np.abs(eigenvals))[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    print(f"   Eigenvalues (first 5): {eigenvals[:5]}")
    print(f"   Eigenvalue range: [{eigenvals.min():.6f}, {eigenvals.max():.6f}]")
    
    # Determine metric signature
    positive_count = np.sum(eigenvals > 1e-8)
    negative_count = np.sum(eigenvals < -1e-8)
    zero_count = n - positive_count - negative_count
    metric_signature = (positive_count, negative_count, zero_count)
    print(f"   Metric signature: {metric_signature}")
    
    print(f"\n🔄 Step 7: Eigendecomposition → Coordinates")
    n_components = 2
    coords = np.zeros((n, n_components))
    
    for i in range(min(n_components, len(eigenvals))):
        eigenval = eigenvals[i]
        eigenvec = eigenvecs[:, i]
        
        print(f"   Component {i}: eigenval={eigenval:.6f}", end="")
        
        if eigenval > 1e-8:
            coords[:, i] = eigenvec * np.sqrt(eigenval)
            print(f" (positive, scaling by sqrt({eigenval:.6f}) = {np.sqrt(eigenval):.6f})")
        elif eigenval < -1e-8:
            coords[:, i] = eigenvec * np.sqrt(-eigenval)  
            print(f" (negative, scaling by sqrt({-eigenval:.6f}) = {np.sqrt(-eigenval):.6f})")
        else:
            coords[:, i] = eigenvec * 1e-3
            print(f" (zero, scaling by 1e-3)")
    
    print(f"   Final coordinates shape: {coords.shape}")
    print(f"   Coordinate range: [{coords.min():.6f}, {coords.max():.6f}]")
    
    # Save all intermediate results
    debug_data = {
        'original_connections': connections_dict,
        'full_matrix': full_matrix.tolist(),
        'similarity': similarity.tolist(),
        'max_sim': float(max_sim),
        'pseudo_distances': pseudo_distances.tolist(),
        'distances_squared': distances_squared.tolist(),
        'gram_matrix': gram_matrix.tolist(),
        'eigenvalues': eigenvals.tolist(),
        'eigenvectors': eigenvecs.tolist(),
        'metric_signature': metric_signature,
        'coordinates': coords.tolist(),
        'n': n
    }
    
    with open('debug_forward_steps.json', 'w') as f:
        json.dump(debug_data, f, indent=2)
    
    print(f"\n✅ All forward transformation steps saved to debug_forward_steps.json")
    
    # Verify our result matches the original
    original_coords = np.array(data['coordinates'])
    coord_diff = np.max(np.abs(coords - original_coords))
    print(f"\n🔍 Verification: coordinate difference with original = {coord_diff:.2e}")
    
    if coord_diff < 1e-10:
        print("✅ Perfect match - our forward transformation is correct")
    else:
        print("❌ Mismatch - there's an error in our forward transformation")
    
    return debug_data

if __name__ == "__main__":
    debug_data = debug_forward_transformation()