#!/usr/bin/env python3
"""
Debug Mandarin clustering quality by examining the actual network connections.
"""

from huey_temporal_simple import HueyTemporalSimple
import numpy as np

def debug_clustering():
    print("🔍 DEBUGGING MANDARIN CLUSTERING QUALITY")
    print("=" * 50)
    
    # Create temporal network
    huey = HueyTemporalSimple(
        max_neurons=200,
        use_temporal_weights=True,
        tau=3.0,
        use_gpu_acceleration=False
    )
    
    # Process the file
    result = huey.process_file_with_mode("test_short_mandarin.txt", conversation_mode=True)
    
    print(f"\n📊 NETWORK SUMMARY:")
    print(f"   Concepts: {len(huey.concept_neurons)}")
    print(f"   Connections: {len(huey.connections)}")
    print(f"   Speakers: {len(huey.speakers)}")
    
    print(f"\n🔤 CONCEPTS CREATED:")
    for i, (word, neuron_id) in enumerate(list(huey.concept_neurons.items())[:15]):
        print(f"   {i+1:2d}. '{word}' → neuron_{neuron_id}")
    
    print(f"\n🔗 CONNECTION QUALITY CHECK:")
    strong_connections = 0
    total_connections = 0
    
    for conn_key, strength in huey.connections.items():
        total_connections += 1
        if strength > 0.5:
            strong_connections += 1
            # Show some strong connections
            if strong_connections <= 5:
                neuron_i, neuron_j = conn_key
                word_i = huey.neuron_to_word.get(neuron_i, f"neuron_{neuron_i}")
                word_j = huey.neuron_to_word.get(neuron_j, f"neuron_{neuron_j}")
                print(f"   Strong: '{word_i}' ↔ '{word_j}' (strength: {strength:.3f})")
    
    print(f"\n📈 CONNECTION STATISTICS:")
    print(f"   Total connections: {total_connections}")
    print(f"   Strong connections (>0.5): {strong_connections}")
    print(f"   Connection density: {strong_connections/total_connections*100:.1f}%")
    
    # Check association matrix
    print(f"\n🎯 ASSOCIATION MATRIX TEST:")
    try:
        association_matrix = huey.calculate_association_matrix()
        print(f"   Matrix size: {association_matrix.shape}")
        print(f"   Matrix values range: {association_matrix.min():.3f} to {association_matrix.max():.3f}")
        print(f"   Non-zero entries: {np.count_nonzero(association_matrix)}")
        
        # Show some sample associations
        concept_list = list(huey.concept_neurons.keys())
        print(f"\n📝 SAMPLE ASSOCIATIONS:")
        for i in range(min(5, len(concept_list))):
            for j in range(i+1, min(i+4, len(concept_list))):
                if j < len(concept_list):
                    association = association_matrix[i, j]
                    print(f"   '{concept_list[i]}' ↔ '{concept_list[j]}': {association:.3f}")
                    
    except Exception as e:
        print(f"   ❌ Association matrix error: {e}")
    
    # Test 3D coordinates
    print(f"\n🌍 3D COORDINATES TEST:")
    try:
        coordinates, eigenvals, labels, eigenvecs = huey.get_3d_coordinates()
        print(f"   Coordinates shape: {coordinates.shape}")
        print(f"   Eigenvalues: {eigenvals[:5] if len(eigenvals) >= 5 else eigenvals}")
        print(f"   Labels count: {len(labels)}")
        
        if len(coordinates) > 0:
            print(f"   Coordinate ranges:")
            print(f"     X: {coordinates[:, 0].min():.3f} to {coordinates[:, 0].max():.3f}")
            print(f"     Y: {coordinates[:, 1].min():.3f} to {coordinates[:, 1].max():.3f}")
            print(f"     Z: {coordinates[:, 2].min():.3f} to {coordinates[:, 2].max():.3f}")
            
            # Check if coordinates are all clustered at origin (bad clustering)
            distances_from_origin = np.sqrt(np.sum(coordinates**2, axis=1))
            max_distance = distances_from_origin.max()
            print(f"   Max distance from origin: {max_distance:.3f}")
            
            if max_distance < 0.01:
                print("   ⚠️  WARNING: All concepts clustered at origin - poor separation!")
            else:
                print("   ✅ Good spatial separation detected")
        
    except Exception as e:
        print(f"   ❌ 3D coordinates error: {e}")

if __name__ == "__main__":
    debug_clustering()