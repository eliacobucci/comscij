#!/usr/bin/env python3
"""
Quick GPU test with Feynman file to prove acceleration is working.
"""

import time
import sys
import os

sys.path.append('.')

from huey_gpu_conversational_experiment import HueyGPUConversationalNetwork

def test_gpu_speed():
    print("🚀 Quick GPU Speed Test")
    print("=" * 30)
    
    # Initialize with GPU
    network = HueyGPUConversationalNetwork(
        max_neurons=1000,
        use_gpu_acceleration=True
    )
    
    # Add speaker
    network.add_speaker("Feynman", ['i', 'me', 'my'], ['you', 'your'])
    
    # Read Feynman file
    feynman_files = ['FeynmanYoung.txt', 'Feynman young.txt']
    text = None
    
    for filename in feynman_files:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                text = f.read()
            print(f"📄 Loaded: {filename}")
            break
    
    if not text:
        text = """
        Richard Feynman was one of the most influential physicists of the 20th century. 
        He made fundamental contributions to quantum mechanics, quantum electrodynamics, 
        and particle physics. His approach to physics was characterized by intuitive 
        understanding rather than formal mathematical proofs. He believed that if you 
        can't explain something simply, you don't understand it well enough. This 
        philosophy shaped his teaching and research methods throughout his career.
        
        Feynman's work on quantum electrodynamics earned him the Nobel Prize in Physics 
        in 1965, which he shared with Julian Schwinger and Sin-Itiro Tomonaga.
        """
        print("📝 Using test text")
    
    words = len(text.split())
    print(f"📊 Processing {words} words")
    
    # Process with timing
    start = time.perf_counter()
    network.process_speaker_text("Feynman", text)
    end = time.perf_counter()
    
    elapsed = end - start
    rate = words / elapsed
    
    print(f"⏱️  Time: {elapsed:.2f}s")
    print(f"🚀 Rate: {rate:.1f} words/second")
    print(f"🧠 Neurons: {network.neuron_count}")
    print(f"🔗 Connections: {len(network.connections)}")
    
    # GPU stats
    gpu_stats = network.gpu_interface.get_performance_stats()
    print(f"⚡ GPU calls: {gpu_stats['kernel_calls']}")
    print(f"💻 GPU enabled: {gpu_stats['gpu_enabled']}")
    
    if rate > 1000:
        print("✅ FAST! GPU acceleration working")
    elif rate > 100:
        print("🆗 Okay speed")
    else:
        print("❌ SLOW! Something wrong")

if __name__ == "__main__":
    test_gpu_speed()