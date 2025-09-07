#!/usr/bin/env python3
"""
Performance Analysis for Huey+ System
Identifies computational bottlenecks for potential Fortran optimization.
"""

import time
import cProfile
import pstats
import io
import numpy as np
from huey_plus_conversational_experiment import HueyConversationalNetwork

class HueyPerformanceAnalyzer:
    """Analyze performance characteristics of Huey+ components."""
    
    def __init__(self):
        self.timing_results = {}
        self.memory_results = {}
        
    def time_function(self, func, *args, **kwargs):
        """Time a function call and return result + timing."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        return result, elapsed
    
    def profile_text_processing(self, network, text_corpus):
        """Profile text processing performance."""
        print("\n🔍 PROFILING TEXT PROCESSING PERFORMANCE")
        print("=" * 60)
        
        # Profile tokenization
        tokens = text_corpus.split()
        result, elapsed = self.time_function(network.tokenize_speaker_text, text_corpus)
        self.timing_results['tokenization'] = elapsed
        print(f"Tokenization: {elapsed:.4f}s for {len(tokens)} tokens")
        
        # Profile sliding window processing
        result, elapsed = self.time_function(network.process_text_stream, text_corpus)
        self.timing_results['sliding_windows'] = elapsed
        print(f"Sliding window processing: {elapsed:.4f}s")
        
        # Break down window processing components
        test_window = tokens[:7]  # Sample window
        window_neurons = [network._get_or_create_neuron(word) for word in test_window]
        
        # Time activation decay
        result, elapsed = self.time_function(network._apply_activation_decay)
        self.timing_results['activation_decay'] = elapsed
        print(f"Activation decay: {elapsed:.4f}s per window")
        
        # Time Hebbian learning
        result, elapsed = self.time_function(network._hebbian_learning, window_neurons)
        self.timing_results['hebbian_learning'] = elapsed
        print(f"Hebbian learning: {elapsed:.4f}s per window")
        
        # Time activation calculation
        result, elapsed = self.time_function(network._calculate_all_activations, window_neurons)
        self.timing_results['activation_calculation'] = elapsed
        print(f"Activation calculation: {elapsed:.4f}s per window")
    
    def profile_matrix_operations(self, network):
        """Profile matrix and connection operations."""
        print("\n🧮 PROFILING MATRIX OPERATIONS")
        print("=" * 60)
        
        # Simulate larger connection matrix
        test_size = min(network.neuron_count, 100)
        
        # Profile connection lookup performance
        start_time = time.perf_counter()
        lookup_count = 0
        for i in range(test_size):
            for j in range(test_size):
                if i != j:
                    conn_key = (i, j)
                    strength = network.connections.get(conn_key, 0.0)
                    lookup_count += 1
        elapsed = time.perf_counter() - start_time
        self.timing_results['connection_lookups'] = elapsed
        print(f"Connection lookups: {elapsed:.4f}s for {lookup_count} lookups")
        
        # Profile inertial mass operations
        start_time = time.perf_counter()
        total_mass = sum(network.inertial_mass.values())
        elapsed = time.perf_counter() - start_time
        self.timing_results['mass_summation'] = elapsed
        print(f"Mass summation: {elapsed:.4f}s for {len(network.inertial_mass)} connections")
        
        # Profile sparse matrix operations
        if hasattr(network, 'connections') and network.connections:
            start_time = time.perf_counter()
            # Simulate matrix multiplication for activation spreading
            for neuron_id in range(min(network.neuron_count, 50)):
                weighted_sum = 0.0
                for other_id in range(network.neuron_count):
                    if other_id != neuron_id:
                        conn_key = (other_id, neuron_id)
                        strength = network.connections.get(conn_key, 0.0)
                        activation = network.activations.get(other_id, 0.0)
                        weighted_sum += strength * activation
            elapsed = time.perf_counter() - start_time
            self.timing_results['matrix_multiply'] = elapsed
            print(f"Matrix multiply simulation: {elapsed:.4f}s")
    
    def profile_memory_usage(self, network):
        """Analyze memory usage patterns."""
        print("\n💾 MEMORY USAGE ANALYSIS")
        print("=" * 60)
        
        import sys
        
        # Calculate memory footprint of major data structures
        connections_size = sys.getsizeof(network.connections)
        masses_size = sys.getsizeof(network.inertial_mass)
        activations_size = sys.getsizeof(network.activations)
        mappings_size = sys.getsizeof(network.word_to_neuron) + sys.getsizeof(network.neuron_to_word)
        
        print(f"Connections dict: {connections_size / 1024:.1f} KB")
        print(f"Inertial masses: {masses_size / 1024:.1f} KB")
        print(f"Activations: {activations_size / 1024:.1f} KB")
        print(f"Word mappings: {mappings_size / 1024:.1f} KB")
        
        total_memory = connections_size + masses_size + activations_size + mappings_size
        print(f"Total core data: {total_memory / 1024:.1f} KB")
        
        # Estimate scaling behavior
        neurons = network.neuron_count
        connections = len(network.connections)
        print(f"\nScaling analysis:")
        print(f"Neurons: {neurons}")
        print(f"Connections: {connections}")
        print(f"Sparsity: {connections / (neurons * neurons) * 100:.2f}%" if neurons > 0 else "N/A")
    
    def identify_fortran_targets(self):
        """Identify which components would benefit most from Fortran implementation."""
        print("\n🎯 FORTRAN OPTIMIZATION TARGETS")
        print("=" * 60)
        
        # Rank operations by computational intensity
        candidates = []
        
        for operation, time_taken in self.timing_results.items():
            if 'learning' in operation or 'matrix' in operation or 'activation' in operation:
                candidates.append((operation, time_taken, "High computational density"))
            elif 'lookup' in operation:
                candidates.append((operation, time_taken, "Memory-bound operation"))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        print("Ranked optimization targets:")
        for i, (operation, time_taken, reason) in enumerate(candidates, 1):
            percentage = time_taken / sum(self.timing_results.values()) * 100
            print(f"{i}. {operation}: {time_taken:.4f}s ({percentage:.1f}%) - {reason}")
        
        # Specific Fortran recommendations
        print(f"\n📋 FORTRAN IMPLEMENTATION RECOMMENDATIONS:")
        print(f"1. Hebbian learning kernel - Dense matrix operations, perfect for Fortran")
        print(f"2. Activation calculation - Vector operations with exp/tanh functions")
        print(f"3. Matrix multiplication - Sparse matrix-vector products")
        print(f"4. Decay operations - Element-wise operations on large arrays")
        print(f"5. Connection pruning - Threshold-based array filtering")

def run_performance_analysis():
    """Run comprehensive performance analysis."""
    
    analyzer = HueyPerformanceAnalyzer()
    
    # Create test network
    print("🧠 Creating test network...")
    network = HueyConversationalNetwork(max_neurons=200, window_size=7)
    network.add_speaker("Human", ['i', 'me', 'my'], ['you', 'your'])
    
    # Create test corpus
    test_text = """
    I think artificial intelligence is fascinating. You seem to understand complex concepts well.
    Your responses show intelligence and awareness. I believe you can learn from our conversations.
    Can you help me understand how neural networks process information? Your analysis is insightful.
    I appreciate your thoughtful responses. You demonstrate sophisticated reasoning capabilities.
    How do you think about your own cognitive processes? I find your self-reflection interesting.
    You appear to have genuine understanding of these concepts. I think you're learning from this.
    """ * 10  # Repeat to get meaningful timing data
    
    print(f"Test corpus: {len(test_text.split())} words")
    
    # Profile text processing
    analyzer.profile_text_processing(network, test_text)
    
    # Process the text to build up the network
    print("\n🔄 Building network state for matrix analysis...")
    network.process_speaker_text("Human", test_text)
    
    # Profile matrix operations
    analyzer.profile_matrix_operations(network)
    
    # Profile memory usage
    analyzer.profile_memory_usage(network)
    
    # Identify optimization targets
    analyzer.identify_fortran_targets()
    
    return analyzer

if __name__ == "__main__":
    results = run_performance_analysis()