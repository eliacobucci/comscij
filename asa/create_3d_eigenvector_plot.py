#!/usr/bin/env python3
"""
Create a 3D plot using the first three real eigenvectors directly.
Debug the sphere sizing issue.
"""

from experimental_network import ExperimentalNetwork
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("🎯 CREATING 3D EIGENVECTOR PLOT")
print("=" * 40)

# Use the same network from before
net = ExperimentalNetwork(window_size=3, max_neurons=12)
sample_text = """
love peace joy love peace harmony love joy music art beauty love peace
hate anger fear hate violence chaos hate anger destruction hate fear 
science truth logic science knowledge wisdom science truth mathematics science logic
chaos disorder confusion chaos ignorance delusion chaos disorder falsehood chaos confusion
"""

print("🧠 Training network...")
net.process_text_stream(sample_text)

# Engineer relationships
available = list(net.word_to_neuron.keys())
if 'science' in available and 'truth' in available:
    net.engineer_concept_movement("science", "truth", direction="toward", 
                                strength=0.4, iterations=5, simulate=False)

print(f"\n🔍 DEBUGGING SPHERE SIZE CALCULATION")

# Debug the sphere size calculation
total_masses = {}
for i in range(net.neuron_count):
    if i in net.neuron_to_word:
        total_mass = 0.0
        for conn_key, mass in net.inertial_mass.items():
            if conn_key[0] == i or conn_key[1] == i:
                total_mass += mass
        total_masses[net.neuron_to_word[i]] = total_mass

print("Raw masses:")
for word, mass in sorted(total_masses.items(), key=lambda x: x[1], reverse=True):
    print(f"   {word}: {mass:.3f}")

# Calculate sphere sizes using the SAME formula as in visualization
sphere_sizes = []
sphere_words = []
for i in range(net.neuron_count):
    if i in net.neuron_to_word:
        total_mass = total_masses[net.neuron_to_word[i]]
        # THIS IS THE EXACT FORMULA FROM THE VISUALIZATION CODE
        size = 100 + np.sqrt(max(total_mass, 1.0)) * 200  # Base size + mass component
        sphere_sizes.append(size)
        sphere_words.append(net.neuron_to_word[i])
        print(f"   {net.neuron_to_word[i]}: mass={total_mass:.3f} → size={size:.1f}")

print(f"\nSphere size statistics:")
print(f"   Min size: {min(sphere_sizes):.1f}")
print(f"   Max size: {max(sphere_sizes):.1f}")  
print(f"   Range: {max(sphere_sizes) - min(sphere_sizes):.1f}")
print(f"   Ratio (max/min): {max(sphere_sizes)/min(sphere_sizes):.2f}")

# Get the connection matrix and do proper eigenanalysis
print(f"\n🧮 EIGENANALYSIS")
full_matrix = np.zeros((net.neuron_count, net.neuron_count))
for (i, j), strength in net.connections.items():
    if i < net.neuron_count and j < net.neuron_count:
        full_matrix[i][j] = strength

# Apply Torgerson transformation
gram_matrix = net._torgerson_transform_proper(full_matrix)

# Get eigenvalues and eigenvectors
eigenvals, eigenvecs = np.linalg.eigh(gram_matrix)

# Sort by eigenvalue magnitude
idx = np.argsort(np.abs(eigenvals))[::-1]
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

print(f"Top 5 eigenvalues: {eigenvals[:5]}")

# Get coordinates using first 3 eigenvectors
coords_3d = np.zeros((net.neuron_count, 3))
for i in range(min(3, len(eigenvals))):
    eigenval = eigenvals[i]
    eigenvec = eigenvecs[:, i]
    
    if eigenval > 1e-8:
        coords_3d[:, i] = eigenvec * np.sqrt(eigenval)
    elif eigenval < -1e-8:
        coords_3d[:, i] = eigenvec * np.sqrt(-eigenval)
    else:
        coords_3d[:, i] = eigenvec * 1e-3

# Filter to active concepts only
active_indices = [i for i in range(net.neuron_count) if i in net.neuron_to_word]
active_coords = coords_3d[active_indices]
active_words = [net.neuron_to_word[i] for i in active_indices]
active_masses = [total_masses[word] for word in active_words]
active_sizes = [100 + np.sqrt(max(mass, 1.0)) * 200 for mass in active_masses]

print(f"\nActive concepts: {len(active_indices)}")
print(f"Coordinate ranges:")
print(f"   X: {active_coords[:, 0].min():.3f} to {active_coords[:, 0].max():.3f}")
print(f"   Y: {active_coords[:, 1].min():.3f} to {active_coords[:, 1].max():.3f}")
print(f"   Z: {active_coords[:, 2].min():.3f} to {active_coords[:, 2].max():.3f}")

# Create 3D plot
print(f"\n🎨 Creating 3D eigenvector plot...")
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Use activation levels for coloring
activations = [net.activations.get(i, 0.0) for i in active_indices]

scatter = ax.scatter(active_coords[:, 0], active_coords[:, 1], active_coords[:, 2],
                    s=active_sizes, c=activations, cmap='viridis',
                    alpha=0.8, edgecolors='black', linewidth=1)

# Add word labels
for i, word in enumerate(active_words):
    ax.text(active_coords[i, 0], active_coords[i, 1], active_coords[i, 2], 
           word, fontsize=9, fontweight='bold')

# Set labels
ax.set_xlabel(f'Eigenvector 1 (λ={eigenvals[0]:.3f})')
ax.set_ylabel(f'Eigenvector 2 (λ={eigenvals[1]:.3f})')
ax.set_zlabel(f'Eigenvector 3 (λ={eigenvals[2]:.3f})')

pos_dims = np.sum(eigenvals > 1e-8)
neg_dims = np.sum(eigenvals < -1e-8)
ax.set_title(f'3D Eigenvector Plot\nSignature: (+{pos_dims}, -{neg_dims}) | Sphere size ∝ Inertial Mass')

plt.colorbar(scatter, shrink=0.5, aspect=20, label='Activation Level')
plt.tight_layout()
plt.savefig('/Users/josephwoelfel/asa/3d_eigenvector_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 3D plot saved: 3d_eigenvector_plot.png")

# SPHERE SIZE ANALYSIS
print(f"\n🔍 SPHERE SIZE PROBLEM ANALYSIS:")
print(f"The issue is the sqrt(max(mass, 1.0)) formula!")
print(f"For masses near 0:")
print(f"   mass=0.000 → sqrt(max(0,1)) = sqrt(1) = 1.0 → size = 100+200*1.0 = 300")
print(f"   mass=2.000 → sqrt(max(2,1)) = sqrt(2) = 1.414 → size = 100+200*1.414 = 383") 
print(f"   Ratio: 383/300 = 1.28 (only 28% bigger!)")
print(f"\nThe sqrt() and max(mass,1.0) are killing the size differences!")
print(f"Should use linear scaling: size = 100 + mass * scaling_factor")