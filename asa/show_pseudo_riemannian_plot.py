#!/usr/bin/env python3
"""
Generate and display a pseudo-Riemannian cognitive space visualization.
"""

from experimental_network import ExperimentalNetwork

print("🎨 CREATING PSEUDO-RIEMANNIAN COGNITIVE SPACE PLOT")
print("=" * 55)

# Create network with interesting conceptual relationships
net = ExperimentalNetwork(window_size=3, max_neurons=18)

# Text with diverse positive and negative associations
sample_text = """
love joy peace harmony beauty art music creativity inspiration wonder
hate anger violence chaos destruction fear anxiety darkness depression pain
science truth knowledge wisdom discovery learning mathematics logic reason
confusion ignorance superstition delusion error falsehood chaos disorder
"""

print("🧠 Training network with diverse conceptual associations...")
net.process_text_stream(sample_text)

print(f"\n📊 Network state:")
print(f"   Concepts: {net.neuron_count}")
print(f"   Vocabulary: {list(net.neuron_to_word.values())}")

# Engineer some relationships to create interesting geometry
print("\n🔧 Engineering conceptual relationships...")

vocab = list(net.word_to_neuron.keys())

# Create positive concept cluster
if 'love' in vocab and 'beauty' in vocab:
    result = net.engineer_concept_movement("love", "beauty", direction="toward", 
                                         strength=0.3, iterations=3, simulate=False)
    print(f"   love → beauty: {result['strength_change']:+.3f}")

if 'science' in vocab and 'truth' in vocab:
    result = net.engineer_concept_movement("science", "truth", direction="toward", 
                                         strength=0.25, iterations=2, simulate=False)  
    print(f"   science → truth: {result['strength_change']:+.3f}")

# Create repulsive relationships
if 'love' in vocab and 'hate' in vocab:
    result = net.engineer_concept_movement("love", "hate", direction="away", 
                                         strength=0.2, iterations=3, simulate=False)
    print(f"   love ← hate: {result['strength_change']:+.3f}")

if 'truth' in vocab and 'delusion' in vocab:
    result = net.engineer_concept_movement("truth", "delusion", direction="away", 
                                         strength=0.2, iterations=2, simulate=False)
    print(f"   truth ← delusion: {result['strength_change']:+.3f}")

# Generate the pseudo-Riemannian visualization
print("\n🌌 Creating pseudo-Riemannian cognitive space visualization...")
net.visualize_cognitive_space(
    figsize=(14, 11),
    show_connections=True,
    connection_threshold=0.06,
    max_connections=35,
    save_path="/Users/josephwoelfel/asa/demo_pseudo_riemannian.png"
)

print("\n✨ VISUALIZATION COMPLETE!")
print(f"📁 Plot saved as: demo_pseudo_riemannian.png")
print("\n🔍 What to look for in the plot:")
print("   • Sphere sizes show concept entrenchment (inertial mass)")  
print("   • Colors show activation levels (warm = recently active)")
print("   • Lines show connection strengths (thickness ∝ strength)")
print("   • Positions preserve pseudo-Riemannian geometry") 
print("   • Title shows metric signature (spacelike vs timelike dimensions)")
print("   • Eigenvalues λ₁, λ₂ reveal the spectral structure")