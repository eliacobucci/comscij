#!/usr/bin/env python3
"""
Compare English vs Mandarin processing to find the connection strength issue.
"""

from huey_temporal_simple import HueyTemporalSimple

def test_language_processing(filename, language):
    print(f"\n🔍 TESTING {language.upper()} PROCESSING: {filename}")
    print("=" * 60)
    
    # Create temporal network
    huey = HueyTemporalSimple(
        max_neurons=200,
        use_temporal_weights=True,
        tau=3.0,
        use_gpu_acceleration=False
    )
    
    # Process the file
    result = huey.process_file_with_mode(filename, conversation_mode=True)
    
    print(f"📊 RESULTS:")
    print(f"   Concepts: {len(huey.concept_neurons)}")
    print(f"   Connections: {len(huey.connections)}")
    
    # Analyze connection strengths
    strong_connections = 0
    medium_connections = 0
    weak_connections = 0
    total_strength = 0
    
    for conn_key, strength in huey.connections.items():
        total_strength += strength
        if strength > 1.0:
            strong_connections += 1
        elif strength > 0.5:
            medium_connections += 1
        else:
            weak_connections += 1
    
    total_connections = len(huey.connections)
    avg_strength = total_strength / total_connections if total_connections > 0 else 0
    
    print(f"🔗 CONNECTION ANALYSIS:")
    print(f"   Strong (>1.0): {strong_connections}")
    print(f"   Medium (0.5-1.0): {medium_connections}")
    print(f"   Weak (<0.5): {weak_connections}")
    print(f"   Average strength: {avg_strength:.3f}")
    
    # Show strongest connections
    sorted_connections = sorted(huey.connections.items(), key=lambda x: x[1], reverse=True)
    print(f"💪 STRONGEST CONNECTIONS:")
    for i, (conn_key, strength) in enumerate(sorted_connections[:5]):
        neuron_i, neuron_j = conn_key
        word_i = huey.neuron_to_word.get(neuron_i, f"neuron_{neuron_i}")
        word_j = huey.neuron_to_word.get(neuron_j, f"neuron_{neuron_j}")
        print(f"   {i+1}. '{word_i}' ↔ '{word_j}': {strength:.3f}")
    
    return huey

def main():
    print("🔬 COMPARING ENGLISH VS MANDARIN PROCESSING")
    print("=" * 70)
    
    # Test English
    english_huey = test_language_processing("test_short_english.txt", "English")
    
    # Test Mandarin  
    mandarin_huey = test_language_processing("test_short_mandarin.txt", "Mandarin")
    
    print(f"\n📈 COMPARISON SUMMARY:")
    print(f"{'Language':<10} {'Concepts':<10} {'Connections':<12} {'Avg Strength':<12}")
    print("-" * 50)
    
    eng_avg = sum(english_huey.connections.values()) / len(english_huey.connections)
    man_avg = sum(mandarin_huey.connections.values()) / len(mandarin_huey.connections)
    
    print(f"{'English':<10} {len(english_huey.concept_neurons):<10} {len(english_huey.connections):<12} {eng_avg:<12.3f}")
    print(f"{'Mandarin':<10} {len(mandarin_huey.concept_neurons):<10} {len(mandarin_huey.connections):<12} {man_avg:<12.3f}")

if __name__ == "__main__":
    main()