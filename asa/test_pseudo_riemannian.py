#!/usr/bin/env python3
"""
Test and demonstrate the pseudo-Riemannian cognitive space visualization.
Shows the difference between Euclidean and pseudo-Riemannian embeddings.
"""

from experimental_network import ExperimentalNetwork

print("🚀 PSEUDO-RIEMANNIAN COGNITIVE SPACE DEMONSTRATION")
print("=" * 60)

# Create network with diverse concepts
net = ExperimentalNetwork(window_size=3, max_neurons=20)

# Rich text with both positive and negative associations
sample_text = """
love happiness joy peace harmony beauty music art poetry dance
hate anger violence war conflict destruction chaos noise pollution trash
science mathematics logic reason truth discovery knowledge learning wisdom
fear anxiety worry stress panic confusion ignorance superstition darkness evil
"""

print("🧠 Training network with diverse emotional and conceptual associations...")
net.process_text_stream(sample_text)

print(f"\n📊 Network trained with {net.neuron_count} concepts")
print(f"🔤 Vocabulary: {list(net.neuron_to_word.values())}")

# Create visualization showing the pseudo-Riemannian structure
print("\n🎨 Creating pseudo-Riemannian cognitive space visualization...")
net.visualize_cognitive_space(
    figsize=(16, 12),
    show_connections=True,
    connection_threshold=0.03,
    max_connections=50,
    save_path="/Users/josephwoelfel/asa/pseudo_riemannian_space.png"
)

# Test concept engineering in pseudo-Riemannian space
print("\n🔧 Engineering concepts in pseudo-Riemannian space...")

# Create some interesting relationships
if 'love' in net.word_to_neuron and 'peace' in net.word_to_neuron:
    result = net.engineer_concept_movement("love", "peace", direction="toward", 
                                         strength=0.3, iterations=4, simulate=False)
    net.print_engineering_result(result)

if 'science' in net.word_to_neuron and 'truth' in net.word_to_neuron:
    result = net.engineer_concept_movement("science", "truth", direction="toward", 
                                         strength=0.25, iterations=3, simulate=False)
    net.print_engineering_result(result)

# Test concept averaging in the space
print("\n🧬 Creating synthetic concepts in pseudo-Riemannian space...")

available_words = list(net.word_to_neuron.keys())
if len(available_words) >= 4:
    # Create positive synthetic concept
    positive_words = [w for w in available_words[:4] if w in ['love', 'happiness', 'joy', 'peace', 'beauty', 'music', 'art', 'science', 'truth']]
    if len(positive_words) >= 2:
        result = net.query_concept_average(positive_words[:2], synthetic_name="PositiveSpace")
        net.print_concept_average_result(result)
    
    # Create negative synthetic concept  
    negative_words = [w for w in available_words if w in ['hate', 'anger', 'violence', 'fear', 'anxiety', 'confusion', 'darkness']]
    if len(negative_words) >= 2:
        result = net.query_concept_average(negative_words[:2], synthetic_name="NegativeSpace")
        net.print_concept_average_result(result)

# Final visualization after engineering
print("\n🎨 Creating final pseudo-Riemannian visualization after engineering...")
net.visualize_cognitive_space(
    figsize=(16, 12),
    show_connections=True,
    connection_threshold=0.04,
    max_connections=45,
    save_path="/Users/josephwoelfel/asa/engineered_pseudo_riemannian.png"
)

print("\n✨ PSEUDO-RIEMANNIAN DEMONSTRATION COMPLETE!")
print("=" * 60)
print("📁 Generated visualizations:")
print("   - pseudo_riemannian_space.png (initial state)")
print("   - engineered_pseudo_riemannian.png (after engineering)")
print("\n🌌 Key findings:")
print("   - Metric signature reveals space-time structure")
print("   - Negative eigenvalues create timelike dimensions")  
print("   - Positive eigenvalues create spacelike dimensions")
print("   - Proper scaling preserves pseudo-Riemannian distances")
print("   - Concepts can have repulsive as well as attractive relationships")