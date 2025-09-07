#!/usr/bin/env python3
"""
Test the invertibility of pseudo-Riemannian embedding by regenerating synaptic weights from coordinates.
This verifies the mathematical consistency of our cognitive space transformation.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def calculate_pseudo_riemannian_distance(coord1, coord2, metric_signature):
    """
    Calculate pseudo-Riemannian distance between two points using the metric signature.
    
    In pseudo-Riemannian geometry:
    - ds² = g_μν dx^μ dx^ν  
    - For our case: ds² = +dx₁² - dx₂² (if signature is (+1, -1))
    - Distance = √|ds²| with appropriate sign handling
    """
    diff = np.array(coord2) - np.array(coord1)
    
    # Apply metric signature: positive for spacelike, negative for timelike dimensions
    pos_dims, neg_dims, zero_dims = metric_signature
    
    # For 2D embedding, typically first dim is spacelike (+), second is timelike (-)
    if len(diff) == 2:
        if pos_dims > 0 and neg_dims > 0:
            # Standard pseudo-Riemannian: +dx² - dy²
            ds_squared = diff[0]**2 - diff[1]**2
        else:
            # Fallback to Euclidean if no mixed signature
            ds_squared = np.sum(diff**2)
    else:
        # General case: apply signature to each dimension
        ds_squared = 0
        total_dims = pos_dims + neg_dims + zero_dims
        for i in range(min(len(diff), total_dims)):
            if i < pos_dims:
                ds_squared += diff[i]**2  # spacelike (positive)
            elif i < pos_dims + neg_dims:
                ds_squared -= diff[i]**2  # timelike (negative)
            # zero_dims contribute nothing
    
    # Return the proper distance
    # In pseudo-Riemannian space, we need to handle the sign carefully
    if ds_squared >= 0:
        return np.sqrt(ds_squared)  # spacelike separation
    else:
        return np.sqrt(-ds_squared)  # timelike separation (imaginary distance -> real positive)

def weights_from_coordinates(coordinates, metric_signature, max_distance_threshold=2.0):
    """
    Regenerate synaptic weights from coordinates using pseudo-Riemannian distance.
    
    The assumption is that synaptic weight ∝ 1/(1 + distance)
    This creates a natural falloff with distance while avoiding singularities.
    """
    n_neurons = len(coordinates)
    regenerated_weights = np.zeros((n_neurons, n_neurons))
    
    for i in range(n_neurons):
        for j in range(n_neurons):
            if i != j:  # No self-connections
                distance = calculate_pseudo_riemannian_distance(
                    coordinates[i], coordinates[j], metric_signature
                )
                
                # Convert distance to weight using inverse relationship
                # Weight = 1/(1 + k*distance) where k is a scaling factor
                # This ensures weights are in [0,1] range and closer points have higher weights
                k = 5.0  # scaling factor to control distance sensitivity
                weight = 1.0 / (1.0 + k * distance)
                
                # Apply threshold to eliminate very weak connections
                if distance < max_distance_threshold:
                    regenerated_weights[i][j] = weight
    
    return regenerated_weights

def analyze_transformation_accuracy(original_weights, regenerated_weights, neuron_to_word):
    """
    Analyze how well the regenerated weights match the original weights.
    """
    # Convert matrices to comparable format
    orig = np.array(original_weights)
    regen = np.array(regenerated_weights)
    
    # Only compare non-zero entries from original matrix
    orig_nonzero_mask = orig > 0
    orig_values = orig[orig_nonzero_mask]
    regen_values = regen[orig_nonzero_mask]
    
    # Calculate correlation
    correlation = np.corrcoef(orig_values, regen_values)[0, 1] if len(orig_values) > 1 else 0
    
    # Calculate mean squared error
    mse = np.mean((orig_values - regen_values)**2)
    
    # Calculate mean absolute error  
    mae = np.mean(np.abs(orig_values - regen_values))
    
    # Calculate percentage of connections preserved (within tolerance)
    tolerance = 0.1
    preserved = np.sum(np.abs(orig_values - regen_values) < tolerance) / len(orig_values)
    
    print(f"🔍 TRANSFORMATION ACCURACY ANALYSIS")
    print(f"=" * 50)
    print(f"Original non-zero connections: {len(orig_values)}")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"Mean squared error: {mse:.6f}")
    print(f"Mean absolute error: {mae:.6f}")
    print(f"Connections preserved (±{tolerance}): {preserved:.1%}")
    
    # Show some example comparisons
    print(f"\n📊 SAMPLE WEIGHT COMPARISONS")
    print(f"{'Connection':<20} {'Original':<10} {'Regenerated':<12} {'Error':<8}")
    print("-" * 52)
    
    # Find indices of non-zero connections
    nonzero_indices = list(zip(*np.where(orig_nonzero_mask)))
    for idx, (i, j) in enumerate(nonzero_indices[:10]):  # Show first 10
        word_i = neuron_to_word.get(str(i), f"neuron{i}")
        word_j = neuron_to_word.get(str(j), f"neuron{j}")
        connection = f"{word_i}→{word_j}"[:19]
        
        orig_weight = orig[i, j]
        regen_weight = regen[i, j]
        error = abs(orig_weight - regen_weight)
        
        print(f"{connection:<20} {orig_weight:<10.4f} {regen_weight:<12.4f} {error:<8.4f}")
    
    return {
        'correlation': correlation,
        'mse': mse,
        'mae': mae,
        'preserved_ratio': preserved,
        'original_connections': len(orig_values),
        'total_comparisons': len(orig_values)
    }

def visualize_weight_comparison(original_weights, regenerated_weights, save_path=None):
    """
    Create visualization comparing original and regenerated weight matrices.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    orig = np.array(original_weights)
    regen = np.array(regenerated_weights)
    diff = orig - regen
    
    # Use same color scale for both matrices
    vmax = max(orig.max(), regen.max())
    vmin = 0
    
    # Original weights
    im1 = axes[0].imshow(orig, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Synaptic Weights")
    axes[0].set_xlabel("Neuron J")
    axes[0].set_ylabel("Neuron I")
    plt.colorbar(im1, ax=axes[0])
    
    # Regenerated weights
    im2 = axes[1].imshow(regen, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("Regenerated Weights (from Coordinates)")
    axes[1].set_xlabel("Neuron J")
    axes[1].set_ylabel("Neuron I")
    plt.colorbar(im2, ax=axes[1])
    
    # Difference matrix
    diff_max = max(abs(diff.min()), diff.max())
    norm = TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)
    im3 = axes[2].imshow(diff, cmap='RdBu_r', norm=norm)
    axes[2].set_title("Difference (Original - Regenerated)")
    axes[2].set_xlabel("Neuron J") 
    axes[2].set_ylabel("Neuron I")
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Weight comparison saved to {save_path}")
    
    plt.show()

def main():
    """
    Main function to test the reverse transformation.
    """
    print("🚀 PSEUDO-RIEMANNIAN REVERSE TRANSFORMATION TEST")
    print("=" * 60)
    
    # Load test data
    try:
        with open('reverse_test_data.json', 'r') as f:
            data = json.load(f)
        print("✅ Test data loaded successfully")
    except FileNotFoundError:
        print("❌ reverse_test_data.json not found. Run the data generation first.")
        return
    
    # Extract data
    original_weights = np.array(data['synaptic_weights'])
    coordinates = np.array(data['coordinates'])  
    metric_signature = tuple(data['metric_signature'])
    neuron_to_word = data['neuron_to_word']
    eigenvalues = np.array(data['eigenvalues'])
    
    print(f"📊 Dataset Information:")
    print(f"   Neurons: {len(coordinates)}")
    print(f"   Coordinate dimensions: {coordinates.shape[1]}")
    print(f"   Metric signature: {metric_signature}")
    print(f"   Original connections: {np.count_nonzero(original_weights)}")
    print(f"   Eigenvalues range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
    
    # Regenerate weights from coordinates
    print(f"\n🔄 Regenerating synaptic weights from coordinates...")
    regenerated_weights = weights_from_coordinates(coordinates, metric_signature)
    
    print(f"✅ Regeneration complete")
    print(f"   Regenerated connections: {np.count_nonzero(regenerated_weights)}")
    
    # Analyze accuracy
    print(f"\n🎯 Analyzing transformation accuracy...")
    accuracy_stats = analyze_transformation_accuracy(
        original_weights, regenerated_weights, neuron_to_word
    )
    
    # Create visualization
    print(f"\n🎨 Creating weight comparison visualization...")
    visualize_weight_comparison(
        original_weights, 
        regenerated_weights,
        save_path="weight_regeneration_comparison.png"
    )
    
    # Summary
    print(f"\n✨ REVERSE TRANSFORMATION COMPLETE!")
    print(f"=" * 60)
    print(f"🎯 Key Findings:")
    print(f"   • Correlation: {accuracy_stats['correlation']:.4f}")
    print(f"   • Mean Absolute Error: {accuracy_stats['mae']:.6f}") 
    print(f"   • Connections Preserved: {accuracy_stats['preserved_ratio']:.1%}")
    
    if accuracy_stats['correlation'] > 0.7:
        print(f"✅ EXCELLENT: Strong correlation indicates successful reverse transformation")
    elif accuracy_stats['correlation'] > 0.4:
        print(f"⚠️  MODERATE: Partial success, some information preserved")
    else:
        print(f"❌ POOR: Low correlation suggests information loss in transformation")
        
    return accuracy_stats

if __name__ == "__main__":
    main()