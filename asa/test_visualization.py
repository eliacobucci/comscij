#!/usr/bin/env python3
"""
Simple test script to demonstrate the cognitive space visualization capabilities.
"""

from experimental_network import ExperimentalNetwork

# Create a network with moderate capacity
net = ExperimentalNetwork(window_size=3, max_neurons=20)

# Process a rich text stream with diverse concepts
sample_text = """
Tokyo Japan sushi culture busy trains technology modern Asia
London England tea rain history royal tradition Europe
Paris France wine art romance fashion cuisine elegant  
New York America coffee skyscrapers business finance energy
Rome Italy pasta history ancient architecture Mediterranean
"""

print("🧠 Training the cognitive network...")
net.process_text_stream(sample_text)
print(f"✅ Processed {net.processed_words} windows, created {net.neuron_count} neurons")

# Test concept engineering before visualization
print("\n🔧 Engineering concept relationships...")

# Move Japan closer to technology (simulate cultural association)
result = net.engineer_concept_movement("Japan", "technology", direction="toward", 
                                     strength=0.3, iterations=3, simulate=False)
net.print_engineering_result(result)

# Move Paris closer to art (reinforce artistic association)
result = net.engineer_concept_movement("Paris", "art", direction="toward", 
                                     strength=0.25, iterations=2, simulate=False)  
net.print_engineering_result(result)

# Create visualization showing the engineered cognitive space
print("\n🎨 Creating cognitive space visualization...")
net.visualize_cognitive_space(
    figsize=(16, 12),
    show_connections=True,
    connection_threshold=0.08,
    max_connections=40,
    save_path="/Users/josephwoelfel/asa/engineered_cognitive_space.png"
)

# Test concept averaging
print("\n🧬 Testing concept synthesis...")
result = net.query_concept_average(["Japan", "technology"], synthetic_name="TechJapan")
net.print_concept_average_result(result)

result = net.query_concept_average(["Paris", "art"], synthetic_name="ArtParis") 
net.print_concept_average_result(result)

print("\n✨ Visualization test complete!")
print("📁 Check the generated PNG files:")
print("   - cognitive_space_viz.png (original network)")  
print("   - engineered_cognitive_space.png (after concept engineering)")