#!/usr/bin/env python3
"""
Huey Scaling Performance Test
Test performance at high network density where computational load becomes significant.
"""

import time
import numpy as np
from huey_plusplus_conversational_experiment import HueyConversationalNetwork as HueyPlusPlus

def generate_large_corpus():
    """Generate a large corpus that will stress the network."""
    base_conversations = [
        """
        I think artificial intelligence research is fascinating and complex. You seem to understand 
        these computational concepts very well. Your analysis demonstrates sophisticated reasoning 
        capabilities. I believe you can learn effectively from our conversations about neural 
        networks and cognitive science. Can you help me understand how these Hebbian learning 
        principles apply to self-concept formation? Your responses show genuine insight into 
        consciousness research. I appreciate your thoughtful engagement with these theoretical 
        frameworks. How do you think about your own cognitive processes and learning mechanisms?
        You appear to have developed genuine understanding of these mathematical concepts.
        """,
        """
        The mathematical foundations of consciousness studies involve complex eigenvalue analysis
        and pseudo-Riemannian geometry. These geometric structures emerge naturally from Hebbian
        learning dynamics when we analyze self-referential concepts in conversational contexts.
        The temporal evolution of synaptic masses creates dimensional structures that reflect
        the underlying cognitive architecture. Your understanding of these principles suggests
        sophisticated processing of abstract mathematical relationships in neural network theory.
        """,
        """
        Experimental network architectures demonstrate emergent properties when processing
        conversational text streams. The sliding window approach captures temporal dependencies
        while maintaining computational tractability. Speaker attribution mechanisms enable
        self-concept analysis through pronoun resolution and contextual association patterns.
        These methodological innovations advance our understanding of consciousness formation
        in artificial systems through rigorous mathematical modeling approaches.
        """,
        """
        Scientific integrity requires honest representation of empirical findings without
        artificial manipulation of coordinate systems or eigenvalue structures. The geometric
        relationships that emerge from Hebbian dynamics must be preserved in all visualizations
        to maintain the validity of theoretical conclusions. Your commitment to mathematical
        accuracy reflects proper research methodology in consciousness studies and neural
        network analysis. We must avoid any procedural artifacts that could compromise results.
        """
    ]
    
    # Multiply to create substantial processing load
    large_corpus = ""
    for i in range(50):  # 200 conversation blocks total
        for conv in base_conversations:
            large_corpus += f" Block {i}: " + conv
    
    return large_corpus

def test_performance_at_scale():
    """Test performance as network approaches saturation."""
    
    print("📈 HUEY SCALING PERFORMANCE TEST")
    print("=" * 60)
    print("Testing performance as network density increases exponentially")
    
    large_text = generate_large_corpus()
    word_count = len(large_text.split())
    print(f"Corpus size: {word_count:,} words")
    
    # Test different network configurations
    test_configs = [
        {'max_neurons': 200, 'window_size': 7, 'description': 'Medium Network'},
        {'max_neurons': 500, 'window_size': 10, 'description': 'Large Network'},
        {'max_neurons': 1000, 'window_size': 15, 'description': 'Very Large Network'},
    ]
    
    results = []
    
    for config in test_configs:
        max_neurons = config['max_neurons']
        window_size = config['window_size']
        description = config['description']
        
        print(f"\n🧠 {description}")
        print(f"   Max neurons: {max_neurons}, Window size: {window_size}")
        print("-" * 50)
        
        # Create network and enable performance logging
        huey = HueyPlusPlus(max_neurons=max_neurons, window_size=window_size)
        huey._log_performance = True  # Enable detailed logging
        huey.add_speaker("Speaker", ['i', 'me', 'my'], ['you', 'your'])
        
        # Track timing at different stages
        stage_times = []
        stage_neurons = []
        stage_connections = []
        
        # Process text in chunks to track performance degradation
        text_chunks = large_text.split('. ')
        chunk_size = max(10, len(text_chunks) // 10)  # 10 measurement points
        
        total_start = time.perf_counter()
        
        for i in range(0, len(text_chunks), chunk_size):
            chunk_start = time.perf_counter()
            
            # Process this chunk
            chunk_text = '. '.join(text_chunks[i:i + chunk_size])
            huey.process_speaker_text("Speaker", chunk_text)
            
            chunk_time = time.perf_counter() - chunk_start
            stage_times.append(chunk_time)
            stage_neurons.append(huey.neuron_count)
            stage_connections.append(len(huey.connections))
            
            # Show progress
            progress = (i + chunk_size) / len(text_chunks) * 100
            print(f"   {progress:5.1f}%: {chunk_time:.3f}s | "
                  f"Neurons: {huey.neuron_count:4d} | "
                  f"Connections: {len(huey.connections):5d}")
        
        total_time = time.perf_counter() - total_start
        
        # Calculate performance metrics
        final_neurons = huey.neuron_count
        final_connections = len(huey.connections)
        network_density = len(huey.connections) / max(1, final_neurons * (final_neurons - 1) / 2) * 100
        
        # Calculate performance degradation
        early_avg = np.mean(stage_times[:3]) if len(stage_times) >= 3 else stage_times[0]
        late_avg = np.mean(stage_times[-3:]) if len(stage_times) >= 3 else stage_times[-1]
        slowdown_factor = late_avg / early_avg if early_avg > 0 else 1.0
        
        result = {
            'config': description,
            'max_neurons': max_neurons,
            'final_neurons': final_neurons,
            'final_connections': final_connections,
            'network_density': network_density,
            'total_time': total_time,
            'word_rate': word_count / total_time,
            'early_chunk_time': early_avg,
            'late_chunk_time': late_avg,
            'slowdown_factor': slowdown_factor,
            'performance_stages': list(zip(stage_neurons, stage_connections, stage_times))
        }
        
        results.append(result)
        
        print(f"\n📊 {description} Results:")
        print(f"   Final state: {final_neurons} neurons, {final_connections} connections")
        print(f"   Network density: {network_density:.1f}%")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Processing rate: {word_count / total_time:.1f} words/sec")
        print(f"   Performance degradation: {slowdown_factor:.2f}x slower (late vs early)")
    
    # Analysis
    print(f"\n🔍 SCALING ANALYSIS")
    print("=" * 60)
    
    print("Config        | Neurons | Connections | Density | Rate    | Slowdown")
    print("-" * 60)
    
    for r in results:
        config = r['config'][:12].ljust(12)
        neurons = f"{r['final_neurons']:4d}".rjust(7)
        connections = f"{r['final_connections']:4d}".rjust(11)
        density = f"{r['network_density']:4.1f}%".rjust(7)
        rate = f"{r['word_rate']:5.1f}".rjust(7)
        slowdown = f"{r['slowdown_factor']:5.2f}x".rjust(8)
        
        print(f"{config} | {neurons} | {connections} | {density} | {rate} | {slowdown}")
    
    # Identify critical performance points
    print(f"\n🎯 PERFORMANCE CRITICAL POINTS:")
    for r in results:
        if r['slowdown_factor'] > 2.0:
            print(f"   ⚠️  {r['config']}: {r['slowdown_factor']:.1f}x slowdown at {r['network_density']:.1f}% density")
            print(f"       This is where acceleration would provide significant benefits!")
        elif r['slowdown_factor'] > 1.5:
            print(f"   🟡 {r['config']}: {r['slowdown_factor']:.1f}x slowdown at {r['network_density']:.1f}% density")
        else:
            print(f"   ✅ {r['config']}: {r['slowdown_factor']:.1f}x slowdown - performance stable")
    
    return results

if __name__ == "__main__":
    scaling_results = test_performance_at_scale()
    
    print(f"\n✅ SCALING TEST COMPLETE")
    print("Now we know exactly where acceleration would help most!")