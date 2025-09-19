#!/usr/bin/env python3
"""
Test high learning rate to get strong connections for good clustering.
"""

from huey_temporal_simple import HueyTemporalSimple

def test_high_learning_rate():
    print("🔬 TESTING HIGH LEARNING RATE FOR GOOD CLUSTERING")
    print("=" * 60)
    
    # Create temporal network with much higher learning rate
    huey = HueyTemporalSimple(
        max_neurons=200,
        use_temporal_weights=True,
        tau=6.0,  # Less temporal decay
        use_gpu_acceleration=False,
        learning_rate=1.0  # Much higher learning rate
    )
    
    # Process Mandarin file
    result = huey.process_file_with_mode("test_short_mandarin.txt", conversation_mode=True)
    
    # Analyze connection strengths
    strong_connections = sum(1 for s in huey.connections.values() if s > 1.0)
    medium_connections = sum(1 for s in huey.connections.values() if s > 0.5)
    total_connections = len(huey.connections)
    avg_strength = sum(huey.connections.values()) / total_connections if total_connections > 0 else 0
    
    print(f"\n📊 RESULTS WITH HIGH LEARNING RATE (1.0):")
    print(f"   Concepts: {len(huey.concept_neurons)}")
    print(f"   Connections: {total_connections}")
    print(f"   Strong (>1.0): {strong_connections}")
    print(f"   Medium (>0.5): {medium_connections}")  
    print(f"   Average strength: {avg_strength:.3f}")
    
    # Show top connections
    sorted_connections = sorted(huey.connections.items(), key=lambda x: x[1], reverse=True)
    print(f"\n🔗 TOP CONNECTIONS:")
    for i, (conn_key, strength) in enumerate(sorted_connections[:8]):
        neuron_i, neuron_j = conn_key
        word_i = huey.neuron_to_word.get(neuron_i, f"neuron_{neuron_i}")
        word_j = huey.neuron_to_word.get(neuron_j, f"neuron_{neuron_j}")
        print(f"   {i+1}. '{word_i}' ↔ '{word_j}': {strength:.3f}")
    
    # Test 3D visualization
    print(f"\n🌍 TESTING 3D CLUSTERING QUALITY:")
    try:
        coordinates, eigenvals, labels, eigenvecs = huey.get_3d_coordinates()
        print(f"   Coordinates shape: {coordinates.shape}")
        print(f"   Top eigenvalues: {eigenvals[:3] if len(eigenvals) >= 3 else eigenvals}")
        
        if len(coordinates) > 0:
            import numpy as np
            # Check if coordinates are well-separated (good clustering)
            distances_from_origin = np.sqrt(np.sum(coordinates**2, axis=1))
            max_distance = distances_from_origin.max()
            mean_distance = distances_from_origin.mean()
            
            print(f"   Max distance from origin: {max_distance:.3f}")
            print(f"   Mean distance from origin: {mean_distance:.3f}")
            
            if max_distance > 0.5:
                print("   ✅ EXCELLENT: Good spatial separation for clustering!")
            elif max_distance > 0.2:
                print("   👍 GOOD: Decent spatial separation")
            else:
                print("   ⚠️  POOR: Still clustered too close to origin")
        
    except Exception as e:
        print(f"   ❌ 3D coordinates error: {e}")

if __name__ == "__main__":
    test_high_learning_rate()