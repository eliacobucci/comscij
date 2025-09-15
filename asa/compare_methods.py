#!/usr/bin/env python3
"""
Compare windowed vs temporal learning on real conversation data.
"""

from huey_temporal_experiment import HueyTemporalExperiment
import numpy as np

def compare_methods_on_file(filename: str):
    """Compare windowed and temporal methods on the same file."""
    print(f"🔍 COMPARING METHODS ON: {filename}")
    print("="*60)
    
    # Method 1: Windowed learning
    print("\n📊 METHOD 1: WINDOWED LEARNING")
    print("-"*40)
    huey_windowed = HueyTemporalExperiment(
        max_neurons=200,
        use_temporal_learning=False,
        window_size=8,
        learning_rate=0.15
    )
    
    try:
        result_windowed = huey_windowed.process_file_with_mode(filename, conversation_mode=True)
        coords_w, eigenvals_w, labels_w, eigenvecs_w = huey_windowed.get_3d_coordinates()
        debug_w = huey_windowed.get_debug_summary()
        
        print(f"✅ Windowed processing successful")
        print(f"   Speakers: {result_windowed.get('speakers_registered', 0)}")
        print(f"   Exchanges: {result_windowed.get('exchanges_processed', 0)}")
        print(f"   Concepts: {debug_w['network_stats']['concepts']}")
        print(f"   Connections: {debug_w['updates']['total_connections']}")
        print(f"   Avg strength: {debug_w['network_stats']['avg_connection_strength']:.6f}")
        print(f"   Max strength: {debug_w['network_stats']['max_connection_strength']:.6f}")
        print(f"   Top 3 eigenvalues: {eigenvals_w[:3] if len(eigenvals_w) >= 3 else eigenvals_w}")
        
    except Exception as e:
        print(f"❌ Windowed method failed: {e}")
        return
    
    # Method 2: Temporal learning
    print("\n🕒 METHOD 2: TEMPORAL LEARNING")
    print("-"*40)
    huey_temporal = HueyTemporalExperiment(
        max_neurons=200,
        use_temporal_learning=True,
        window_size=8,
        tau=3.0,           # Good decay constant
        eta_fwd=0.01,      # Moderate learning rate
        eta_fb=0.002,      # Smaller feedback rate
        boundary_penalty=0.5  # Reduce cross-sentence learning
    )
    
    try:
        result_temporal = huey_temporal.process_file_with_mode(filename, conversation_mode=True)
        coords_t, eigenvals_t, labels_t, eigenvecs_t = huey_temporal.get_3d_coordinates()
        debug_t = huey_temporal.get_debug_summary()
        
        print(f"✅ Temporal processing successful")
        print(f"   Speakers: {result_temporal.get('speakers_registered', 0)}")
        print(f"   Exchanges: {result_temporal.get('exchanges_processed', 0)}")
        print(f"   Concepts: {debug_t['network_stats']['concepts']}")
        print(f"   Connections: {debug_t['updates']['total_connections']}")
        print(f"   Avg strength: {debug_t['network_stats']['avg_connection_strength']:.6f}")
        print(f"   Max strength: {debug_t['network_stats']['max_connection_strength']:.6f}")
        print(f"   Top 3 eigenvalues: {eigenvals_t[:3] if len(eigenvals_t) >= 3 else eigenvals_t}")
        print(f"   Nonzero updates: {debug_t['updates']['nonzero_updates']}")
        print(f"   Zero updates: {debug_t['updates']['zero_updates']}")
        
    except Exception as e:
        print(f"❌ Temporal method failed: {e}")
        return
    
    # Comparison analysis
    print(f"\n📈 COMPARISON ANALYSIS")
    print("="*60)
    
    concepts_w = debug_w['network_stats']['concepts']
    concepts_t = debug_t['network_stats']['concepts']
    connections_w = debug_w['updates']['total_connections']
    connections_t = debug_t['updates']['total_connections']
    strength_w = debug_w['network_stats']['avg_connection_strength']
    strength_t = debug_t['network_stats']['avg_connection_strength']
    
    print(f"Concepts:     Windowed={concepts_w:3d}, Temporal={concepts_t:3d}, Δ={concepts_t-concepts_w:+d}")
    print(f"Connections:  Windowed={connections_w:3d}, Temporal={connections_t:3d}, Δ={connections_t-connections_w:+d}")
    print(f"Avg Strength: Windowed={strength_w:.6f}, Temporal={strength_t:.6f}, Ratio={strength_t/strength_w:.3f}x")
    
    # Eigenvalue comparison
    if len(eigenvals_w) >= 3 and len(eigenvals_t) >= 3:
        print(f"\nEigenvalue Analysis:")
        for i in range(min(3, len(eigenvals_w), len(eigenvals_t))):
            ratio = eigenvals_t[i] / eigenvals_w[i] if eigenvals_w[i] != 0 else float('inf')
            print(f"  λ{i+1}: Windowed={eigenvals_w[i]:.6f}, Temporal={eigenvals_t[i]:.6f}, Ratio={ratio:.3f}x")
    
    # Concept overlap
    labels_w_set = set(labels_w) if labels_w else set()
    labels_t_set = set(labels_t) if labels_t else set()
    common_concepts = labels_w_set.intersection(labels_t_set)
    
    print(f"\nConcept Overlap:")
    print(f"  Common concepts: {len(common_concepts)}")
    print(f"  Windowed-only: {len(labels_w_set - labels_t_set)}")
    print(f"  Temporal-only: {len(labels_t_set - labels_w_set)}")
    
    # Show some temporal learning details
    if debug_t.get('sample_updates'):
        print(f"\nTemporal Learning Sample:")
        for i, update in enumerate(debug_t['sample_updates'][:3]):
            print(f"  {update['tokens']}: lag={update['lag']}, weight={update['final_weight']:.6f}")

if __name__ == "__main__":
    print("🧪 WINDOWED vs TEMPORAL LEARNING COMPARISON")
    
    # Test on the simple temporal test file
    test_file = "/Users/josephwoelfel/asa/test_temporal.txt"
    compare_methods_on_file(test_file)
    
    print(f"\n🔬 Method Comparison Complete!")