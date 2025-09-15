#!/usr/bin/env python3
"""
Simple test to generate working plots for Huey results
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
import json
from datetime import datetime

def create_simple_plot(concepts, n_concepts=20):
    """Create a simple 2D matplotlib plot instead of 3D Plotly"""
    
    # Generate some dummy data for testing
    np.random.seed(42)
    n = min(n_concepts, len(concepts) if concepts else 20)
    
    # Create random 2D positions for concepts
    X = np.random.randn(n, 2)
    
    # Create masses (concept strengths)
    masses = np.random.exponential(2, n) + 1
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=masses, s=masses*50, 
                         cmap='viridis', alpha=0.7, edgecolors='black')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Concept Mass', rotation=270, labelpad=15)
    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'Tamil Cognitive Architecture ({n} concepts)')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tamil_cognitive_plot_{timestamp}.png"
    plt.savefig(f"/Users/josephwoelfel/Downloads/{filename}", 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved working plot: {filename}")
    return filename

if __name__ == "__main__":
    # Test with dummy data
    dummy_concepts = ["concept_" + str(i) for i in range(50)]
    
    # Create plots at different zoom levels
    for n in [20, 50, 100]:
        filename = create_simple_plot(dummy_concepts, n)
        print(f"Created {filename}")