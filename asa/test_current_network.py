#!/usr/bin/env python3
"""
Test the visualization with the current network state.
"""

from experimental_network import ExperimentalNetwork

# Create network and use the main demo
net = ExperimentalNetwork(window_size=3, max_neurons=15)

# Use the same text from the main demo for consistency
sample_text = "Paris France wine food Rome Italy pasta pizza London England tea rain New York America coffee busy Tokyo Japan sushi trains"

print("🧠 Processing training text...")
net.process_text_stream(sample_text)

print(f"\n📊 Current network state:")
print(f"   Neurons: {net.neuron_count}")
print(f"   Vocabulary: {list(net.neuron_to_word.values())}")

# Test engineering with current vocabulary
print("\n🔧 Engineering current concepts...")
result = net.engineer_concept_movement("coffee", "America", direction="toward", 
                                     strength=0.2, iterations=3, simulate=False)
net.print_engineering_result(result)

result = net.engineer_concept_movement("sushi", "trains", direction="toward", 
                                     strength=0.15, iterations=2, simulate=False)
net.print_engineering_result(result)

# Test concept averaging with current vocabulary
print("\n🧬 Creating synthetic concepts...")
result = net.query_concept_average(["coffee", "America"], synthetic_name="AmericanCoffee")
net.print_concept_average_result(result)

result = net.query_concept_average(["sushi", "trains"], synthetic_name="JapaneseTransport")  
net.print_concept_average_result(result)

# Create visualization
print("\n🎨 Creating cognitive space visualization...")
net.visualize_cognitive_space(
    figsize=(16, 12),
    show_connections=True,
    connection_threshold=0.05,
    max_connections=40,
    save_path="/Users/josephwoelfel/asa/final_cognitive_space.png"
)

print("\n✨ Test complete!")
print("📁 Final visualization: final_cognitive_space.png")
print("🧠 This shows the engineered cognitive space with:")
print("   - Mass-proportional sphere sizes")
print("   - Activation-based coloring") 
print("   - Connection strength lines")
print("   - MDS positioning based on associations")