#!/usr/bin/env python3
"""
Test stronger learning parameters to fix weak connections.
"""

from huey_temporal_simple import HueyTemporalSimple

def test_learning_parameters(tau, learning_rate, description):
    print(f"\n🧪 TESTING: {description}")
    print(f"   tau={tau}, learning_rate={learning_rate}")
    print("-" * 50)
    
    # Create temporal network with adjusted parameters
    huey = HueyTemporalSimple(
        max_neurons=200,
        use_temporal_weights=True,
        tau=tau,
        use_gpu_acceleration=False,
        learning_rate=learning_rate
    )
    
    # Process Mandarin file
    result = huey.process_file_with_mode("test_short_mandarin.txt", conversation_mode=True)
    
    # Analyze connection strengths
    strong_connections = sum(1 for s in huey.connections.values() if s > 1.0)
    medium_connections = sum(1 for s in huey.connections.values() if s > 0.5)
    total_connections = len(huey.connections)
    avg_strength = sum(huey.connections.values()) / total_connections if total_connections > 0 else 0
    
    print(f"📊 RESULTS:")
    print(f"   Concepts: {len(huey.concept_neurons)}")
    print(f"   Connections: {total_connections}")
    print(f"   Strong (>1.0): {strong_connections}")
    print(f"   Medium (>0.5): {medium_connections}")  
    print(f"   Average strength: {avg_strength:.3f}")
    
    # Show top connections
    sorted_connections = sorted(huey.connections.items(), key=lambda x: x[1], reverse=True)
    print(f"🔗 TOP CONNECTIONS:")
    for i, (conn_key, strength) in enumerate(sorted_connections[:3]):
        neuron_i, neuron_j = conn_key
        word_i = huey.neuron_to_word.get(neuron_i, f"neuron_{neuron_i}")
        word_j = huey.neuron_to_word.get(neuron_j, f"neuron_{neuron_j}")
        print(f"   {i+1}. '{word_i}' ↔ '{word_j}': {strength:.3f}")
    
    return avg_strength

def main():
    print("🔬 TESTING DIFFERENT LEARNING PARAMETERS")
    print("=" * 60)
    
    # Test different parameter combinations
    test_cases = [
        (3.0, 0.15, "Original parameters (weak)"),
        (6.0, 0.15, "Doubled tau (less temporal decay)"),
        (3.0, 0.3, "Doubled learning rate"),
        (6.0, 0.3, "Doubled both (stronger learning)"),
        (10.0, 0.5, "Much stronger learning")
    ]
    
    results = []
    for tau, lr, desc in test_cases:
        avg_strength = test_learning_parameters(tau, lr, desc)
        results.append((desc, avg_strength))
    
    print(f"\n📈 SUMMARY:")
    print(f"{'Parameters':<35} {'Avg Strength':<12}")
    print("-" * 50)
    for desc, strength in results:
        print(f"{desc:<35} {strength:<12.3f}")

if __name__ == "__main__":
    main()