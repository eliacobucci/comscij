#!/usr/bin/env python3
"""
HueyGPU Performance Crossover Benchmark
Finds the optimal exchange count where JAX GPU acceleration becomes faster than NumPy.
"""

import time
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import json
from datetime import datetime
import os

# Try both acceleration methods
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
    print("✅ JAX available for benchmarking")
except ImportError:
    JAX_AVAILABLE = False
    print("❌ JAX not available - install with: pip install jax[metal]")

class HueyPerformanceBenchmark:
    """Benchmark Huey performance across different exchange counts and acceleration methods."""
    
    def __init__(self):
        self.results = {
            'numpy': {},
            'jax': {} if JAX_AVAILABLE else None,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'jax_available': JAX_AVAILABLE
            }
        }
    
    def create_synthetic_network_data(self, num_neurons: int, connectivity: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic network data for benchmarking."""
        np.random.seed(42)  # Reproducible results
        
        # Create activation vector
        activations = np.random.rand(num_neurons).astype(np.float64)
        
        # Create sparse connectivity matrix
        connections = np.random.rand(num_neurons, num_neurons)
        mask = connections < connectivity
        connections = connections * mask  # Apply sparsity
        connections = (connections + connections.T) / 2  # Make symmetric
        
        return activations, connections
    
    def numpy_activation_kernel(self, activations: np.ndarray, connections: np.ndarray, learning_rate: float = 0.15) -> np.ndarray:
        """NumPy-based activation calculation."""
        # Hebbian learning update
        outer_product = np.outer(activations, activations)
        connection_updates = learning_rate * outer_product
        
        # Update connections
        new_connections = connections + connection_updates
        
        # Calculate new activations
        new_activations = np.tanh(np.dot(new_connections, activations))
        
        return new_activations
    
    def jax_activation_kernel_setup(self):
        """Setup JAX-compiled activation kernel."""
        if not JAX_AVAILABLE:
            return None
            
        @jit
        def jax_kernel(activations, connections, learning_rate=0.15):
            # Hebbian learning update
            outer_product = jnp.outer(activations, activations)
            connection_updates = learning_rate * outer_product
            
            # Update connections
            new_connections = connections + connection_updates
            
            # Calculate new activations
            new_activations = jnp.tanh(jnp.dot(new_connections, activations))
            
            return new_activations
        
        return jax_kernel
    
    def benchmark_exchange_sequence(self, exchange_counts: List[int], max_neurons: int = 500) -> Dict:
        """Benchmark performance across different exchange counts."""
        
        print(f"\n🧪 HUEY PERFORMANCE CROSSOVER BENCHMARK")
        print("=" * 60)
        print(f"Testing exchange counts: {exchange_counts}")
        print(f"Max neurons: {max_neurons}")
        print(f"JAX available: {JAX_AVAILABLE}")
        
        # Setup JAX kernel if available
        jax_kernel = self.jax_activation_kernel_setup() if JAX_AVAILABLE else None
        
        for exchange_count in exchange_counts:
            print(f"\n🔬 Benchmarking {exchange_count} exchanges...")
            
            # Estimate neuron count based on exchange count (realistic scaling)
            # Typical: ~1.5 neurons per exchange, capped at max_neurons
            estimated_neurons = min(int(exchange_count * 1.5), max_neurons)
            
            # Create synthetic network data
            activations, connections = self.create_synthetic_network_data(estimated_neurons)
            
            print(f"   Network size: {estimated_neurons} neurons")
            
            # Benchmark NumPy
            numpy_times = []
            for trial in range(3):  # Multiple trials for accuracy
                start_time = time.perf_counter()
                
                current_activations = activations.copy()
                for _ in range(exchange_count):
                    current_activations = self.numpy_activation_kernel(current_activations, connections)
                    
                numpy_time = time.perf_counter() - start_time
                numpy_times.append(numpy_time)
            
            avg_numpy_time = np.mean(numpy_times)
            self.results['numpy'][exchange_count] = {
                'total_time': avg_numpy_time,
                'time_per_exchange': avg_numpy_time / exchange_count,
                'neurons': estimated_neurons,
                'exchanges_per_sec': exchange_count / avg_numpy_time
            }
            
            print(f"   ✅ NumPy: {avg_numpy_time:.3f}s ({avg_numpy_time/exchange_count*1000:.2f}ms/exchange)")
            
            # Benchmark JAX if available
            if JAX_AVAILABLE and jax_kernel:
                # Convert to JAX arrays
                jax_activations = jnp.array(activations)
                jax_connections = jnp.array(connections)
                
                # Warmup JAX compilation
                _ = jax_kernel(jax_activations, jax_connections)
                
                jax_times = []
                for trial in range(3):
                    start_time = time.perf_counter()
                    
                    current_activations = jax_activations
                    for _ in range(exchange_count):
                        current_activations = jax_kernel(current_activations, jax_connections)
                        current_activations.block_until_ready()  # Ensure computation completes
                        
                    jax_time = time.perf_counter() - start_time
                    jax_times.append(jax_time)
                
                avg_jax_time = np.mean(jax_times)
                speedup = avg_numpy_time / avg_jax_time
                
                self.results['jax'][exchange_count] = {
                    'total_time': avg_jax_time,
                    'time_per_exchange': avg_jax_time / exchange_count,
                    'neurons': estimated_neurons,
                    'exchanges_per_sec': exchange_count / avg_jax_time,
                    'speedup_vs_numpy': speedup
                }
                
                print(f"   🚀 JAX: {avg_jax_time:.3f}s ({avg_jax_time/exchange_count*1000:.2f}ms/exchange) - {speedup:.2f}x speedup")
                
                # Determine recommendation
                if speedup > 1.2:  # JAX is 20% faster
                    recommendation = "JAX (GPU)"
                elif speedup > 0.8:   # Within 20% performance
                    recommendation = "Either"
                else:
                    recommendation = "NumPy (CPU)"
                    
                print(f"   💡 Recommendation: {recommendation}")
            
        return self.results
    
    def find_crossover_point(self) -> Dict:
        """Find the exchange count where JAX becomes faster than NumPy."""
        
        if not JAX_AVAILABLE or not self.results['jax']:
            return {
                'crossover_point': None,
                'recommendation': "Use NumPy (JAX not available)"
            }
        
        crossover_point = None
        jax_faster_points = []
        
        for exchange_count in sorted(self.results['jax'].keys()):
            if exchange_count in self.results['numpy']:
                speedup = self.results['jax'][exchange_count]['speedup_vs_numpy']
                if speedup > 1.2:  # JAX is 20% faster
                    jax_faster_points.append(exchange_count)
        
        if jax_faster_points:
            crossover_point = min(jax_faster_points)
        
        return {
            'crossover_point': crossover_point,
            'jax_faster_points': jax_faster_points,
            'recommendation': f"Use JAX for {crossover_point}+ exchanges" if crossover_point else "Use NumPy (JAX not beneficial)"
        }
    
    def plot_performance_comparison(self, save_path: str = "huey_performance_crossover.png"):
        """Create performance comparison plot."""
        
        if not self.results['numpy']:
            print("❌ No benchmark data to plot")
            return
        
        exchange_counts = sorted(self.results['numpy'].keys())
        numpy_times = [self.results['numpy'][count]['time_per_exchange'] * 1000 for count in exchange_counts]  # Convert to ms
        
        plt.figure(figsize=(12, 8))
        plt.plot(exchange_counts, numpy_times, 'b-o', label='NumPy (CPU)', linewidth=2, markersize=6)
        
        if JAX_AVAILABLE and self.results['jax']:
            jax_times = [self.results['jax'][count]['time_per_exchange'] * 1000 for count in exchange_counts]
            speedups = [self.results['jax'][count]['speedup_vs_numpy'] for count in exchange_counts]
            
            plt.plot(exchange_counts, jax_times, 'r-s', label='JAX (Metal GPU)', linewidth=2, markersize=6)
            
            # Mark crossover point
            crossover = self.find_crossover_point()
            if crossover['crossover_point']:
                plt.axvline(x=crossover['crossover_point'], color='green', linestyle='--', alpha=0.7, 
                           label=f'Crossover: {crossover["crossover_point"]} exchanges')
        
        plt.xlabel('Number of Exchanges', fontsize=12)
        plt.ylabel('Processing Time per Exchange (ms)', fontsize=12)
        plt.title('Huey Performance: NumPy vs JAX Metal GPU\nProcessing Time per Exchange', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        # Add performance annotations
        if JAX_AVAILABLE and self.results['jax']:
            for i, (count, speedup) in enumerate(zip(exchange_counts[::2], speedups[::2])):  # Every other point
                plt.annotate(f'{speedup:.1f}x', 
                           xy=(count, jax_times[exchange_counts.index(count)]), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance plot saved to: {save_path}")
        
        return save_path
    
    def save_results(self, filename: str = "huey_performance_benchmark.json"):
        """Save benchmark results to JSON file."""
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Benchmark results saved to: {filename}")
    
    def generate_recommendation_config(self) -> Dict:
        """Generate configuration for intelligent acceleration selection."""
        
        crossover = self.find_crossover_point()
        
        config = {
            'intelligent_acceleration': True,
            'crossover_point': crossover.get('crossover_point', 1000),  # Default to 1000 if not found
            'acceleration_thresholds': {
                'small_file': {
                    'max_exchanges': crossover.get('crossover_point', 1000) - 1 if crossover.get('crossover_point') else 999,
                    'method': 'numpy',
                    'reason': 'NumPy faster for small files'
                },
                'large_file': {
                    'min_exchanges': crossover.get('crossover_point', 1000) if crossover.get('crossover_point') else 1000,
                    'method': 'jax',
                    'reason': 'JAX GPU acceleration beneficial for large files'
                }
            },
            'benchmark_metadata': {
                'timestamp': self.results['metadata']['timestamp'],
                'jax_available': JAX_AVAILABLE
            }
        }
        
        return config

def main():
    """Run the comprehensive performance benchmark."""
    
    print("🚀 HUEY PERFORMANCE CROSSOVER BENCHMARK")
    print("=" * 60)
    
    # Test range: from small to large file sizes
    exchange_counts = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]
    
    benchmark = HueyPerformanceBenchmark()
    
    # Run benchmark
    results = benchmark.benchmark_exchange_sequence(exchange_counts)
    
    # Analyze crossover point
    crossover = benchmark.find_crossover_point()
    
    print(f"\n📊 CROSSOVER ANALYSIS:")
    print("=" * 40)
    print(f"Crossover point: {crossover.get('crossover_point', 'Not found')}")
    print(f"Recommendation: {crossover.get('recommendation', 'Use NumPy')}")
    
    if crossover.get('jax_faster_points'):
        print(f"JAX faster at: {crossover['jax_faster_points']} exchanges")
    
    # Generate configuration
    config = benchmark.generate_recommendation_config()
    
    print(f"\n⚙️  INTELLIGENT ACCELERATION CONFIG:")
    print("=" * 40)
    print(f"Small files (<{config['crossover_point']} exchanges): NumPy")
    print(f"Large files (≥{config['crossover_point']} exchanges): JAX GPU")
    
    # Save results
    benchmark.save_results()
    benchmark.plot_performance_comparison()
    
    # Save config for HueyGPU
    with open('huey_acceleration_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Benchmark complete!")
    print(f"📁 Files created:")
    print(f"   - huey_performance_benchmark.json (raw data)")
    print(f"   - huey_acceleration_config.json (intelligent selection config)")
    print(f"   - huey_performance_crossover.png (performance plot)")

if __name__ == "__main__":
    main()