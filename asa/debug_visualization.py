#!/usr/bin/env python3
"""
Debug the visualization issues: peripheral distribution and uniform sizes.
"""

from experimental_network import ExperimentalNetwork
import numpy as np

# Create same network as before
net = ExperimentalNetwork(window_size=3, max_neurons=18)
sample_text = """
love joy peace harmony beauty art music creativity inspiration wonder
hate anger violence chaos destruction fear anxiety darkness depression pain
science truth knowledge wisdom discovery learning mathematics logic reason
confusion ignorance superstition delusion error falsehood chaos disorder
"""

print("🔍 DEBUGGING VISUALIZATION ISSUES")
print("=" * 50)

net.process_text_stream(sample_text)

print(f"📊 Network state:")
print(f"   Concepts: {net.neuron_count}")
print(f"   Active concepts: {len([i for i in range(net.neuron_count) if i in net.neuron_to_word])}")

# Check inertial masses
print(f"\n🏋️ INERTIAL MASS ANALYSIS:")
total_masses = {}
for i in range(net.neuron_count):
    if i in net.neuron_to_word:
        total_mass = 0.0
        for conn_key, mass in net.inertial_mass.items():
            if conn_key[0] == i or conn_key[1] == i:
                total_mass += mass
        total_masses[net.neuron_to_word[i]] = total_mass
        
print("Individual masses:")
for word, mass in sorted(total_masses.items(), key=lambda x: x[1], reverse=True):
    print(f"   {word}: {mass:.3f}")

mass_values = list(total_masses.values())
print(f"\nMass statistics:")
print(f"   Min: {min(mass_values):.3f}")
print(f"   Max: {max(mass_values):.3f}")  
print(f"   Range: {max(mass_values) - min(mass_values):.3f}")
print(f"   Std dev: {np.std(mass_values):.3f}")

# Check the embedding eigenvalues
print(f"\n🌌 EIGENVALUE ANALYSIS:")
coords_2d, eigenvalues, metric_signature = net._pseudo_riemannian_embedding(net.connections, 2)

print(f"Eigenvalues: {eigenvalues[:5]}")
print(f"Metric signature: {metric_signature}")
print(f"Coordinate range:")
print(f"   X: {coords_2d[:, 0].min():.3f} to {coords_2d[:, 0].max():.3f}")
print(f"   Y: {coords_2d[:, 1].min():.3f} to {coords_2d[:, 1].max():.3f}")

# Check coordinate distribution
center_x, center_y = np.mean(coords_2d, axis=0)
distances_from_center = np.sqrt((coords_2d[:, 0] - center_x)**2 + (coords_2d[:, 1] - center_y)**2)

print(f"\nCoordinate distribution:")
print(f"   Center: ({center_x:.3f}, {center_y:.3f})")
print(f"   Distances from center: {distances_from_center.min():.3f} to {distances_from_center.max():.3f}")
print(f"   Mean distance: {distances_from_center.mean():.3f}")
print(f"   Std distance: {distances_from_center.std():.3f}")

# Check if coordinates are all on a circle/ring
print(f"\n📍 COORDINATE PATTERN ANALYSIS:")
radii = distances_from_center
radius_std = np.std(radii)
radius_mean = np.mean(radii)
print(f"   Radius coefficient of variation: {radius_std/radius_mean:.3f}")
if radius_std/radius_mean < 0.3:
    print("   ⚠️  RING PATTERN DETECTED - concepts arranged in a circle!")
else:
    print("   ✅ Coordinates distributed throughout space")