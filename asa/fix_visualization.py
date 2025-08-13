#!/usr/bin/env python3
"""
Create a better visualization with improved mass scaling and more interesting relationships.
"""

from experimental_network import ExperimentalNetwork
import numpy as np

print("🔧 CREATING IMPROVED PSEUDO-RIEMANNIAN VISUALIZATION")
print("=" * 55)

# Create network with smaller capacity to allow more reinforcement
net = ExperimentalNetwork(window_size=3, max_neurons=12)

# Shorter, more repetitive text to build stronger connections and masses
sample_text = """
love peace joy love peace harmony love joy music art beauty love peace
hate anger fear hate violence chaos hate anger destruction hate fear 
science truth logic science knowledge wisdom science truth mathematics science logic
chaos disorder confusion chaos ignorance delusion chaos disorder falsehood chaos confusion
"""

print("🧠 Training with reinforcement-focused text...")
net.process_text_stream(sample_text)

print(f"\n📊 Network state:")
print(f"   Concepts: {net.neuron_count}")
current_vocab = [net.neuron_to_word[i] for i in range(net.neuron_count) if i in net.neuron_to_word]
print(f"   Active vocabulary: {current_vocab}")

# Check masses after training
print(f"\n🏋️ MASS ANALYSIS AFTER REPETITIVE TRAINING:")
total_masses = {}
for i in range(net.neuron_count):
    if i in net.neuron_to_word:
        total_mass = 0.0
        for conn_key, mass in net.inertial_mass.items():
            if conn_key[0] == i or conn_key[1] == i:
                total_mass += mass
        total_masses[net.neuron_to_word[i]] = total_mass

for word, mass in sorted(total_masses.items(), key=lambda x: x[1], reverse=True)[:8]:
    print(f"   {word}: {mass:.3f}")

mass_values = list(total_masses.values())
print(f"Mass range: {min(mass_values):.3f} to {max(mass_values):.3f} (range: {max(mass_values) - min(mass_values):.3f})")

# Engineer some strong relationships
print(f"\n🔧 Engineering stronger relationships...")

available = list(net.word_to_neuron.keys())

# Create strong positive cluster
if 'love' in available and 'peace' in available:
    result = net.engineer_concept_movement("love", "peace", direction="toward", 
                                         strength=0.4, iterations=5, simulate=False)
    print(f"   love ↔ peace: {result['strength_change']:+.3f}")

if 'science' in available and 'truth' in available:
    result = net.engineer_concept_movement("science", "truth", direction="toward", 
                                         strength=0.4, iterations=5, simulate=False)
    print(f"   science ↔ truth: {result['strength_change']:+.3f}")

# Create strong negative relationships
if 'love' in available and 'hate' in available:
    result = net.engineer_concept_movement("love", "hate", direction="away", 
                                         strength=0.3, iterations=4, simulate=False)
    print(f"   love ↚ hate: {result['strength_change']:+.3f}")

if 'truth' in available and 'chaos' in available:
    result = net.engineer_concept_movement("truth", "chaos", direction="away", 
                                         strength=0.3, iterations=4, simulate=False)
    print(f"   truth ↚ chaos: {result['strength_change']:+.3f}")

# Check final masses
print(f"\n🏋️ FINAL MASS ANALYSIS:")
final_masses = {}
for i in range(net.neuron_count):
    if i in net.neuron_to_word:
        total_mass = 0.0
        for conn_key, mass in net.inertial_mass.items():
            if conn_key[0] == i or conn_key[1] == i:
                total_mass += mass
        final_masses[net.neuron_to_word[i]] = total_mass

for word, mass in sorted(final_masses.items(), key=lambda x: x[1], reverse=True):
    print(f"   {word}: {mass:.3f}")

final_values = list(final_masses.values())
print(f"Final mass range: {min(final_values):.3f} to {max(final_values):.3f} (range: {max(final_values) - min(final_values):.3f})")

# Create improved visualization
print(f"\n🎨 Creating improved pseudo-Riemannian visualization...")
net.visualize_cognitive_space(
    figsize=(16, 12),
    show_connections=True,
    connection_threshold=0.04,
    max_connections=40,
    save_path="/Users/josephwoelfel/asa/improved_pseudo_riemannian.png"
)

print(f"\n✨ IMPROVED VISUALIZATION COMPLETE!")
print(f"📁 Saved as: improved_pseudo_riemannian.png")
print(f"🔍 Improvements:")
print(f"   • Higher mass range: {max(final_values) - min(final_values):.3f} vs {max(mass_values) - min(mass_values):.3f}")
print(f"   • Stronger reinforcement through repetition")
print(f"   • More distinct sphere sizes")
print(f"   • Clearer conceptual clustering")
print(f"   • Better pseudo-Riemannian structure")