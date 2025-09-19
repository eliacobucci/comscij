#!/usr/bin/env python3
"""
Test GPU restoration for Huey with realistic network sizes and scenarios.
This simulates the actual workloads that showed 20x slowdown without GPU.
"""

import numpy as np
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from huey_pytorch_gpu_interface import HueyPyTorchGPU

def simulate_huey_network_evolution(gpu_interface, n_neurons, n_iterations=100):
    """
    Simulate Huey network evolution similar to the Feynman file processing.
    This represents the actual workload that was running fast with GPU.
    """
    
    print(f"🧠 Simulating {n_neurons}-neuron network evolution...")
    print(f"   Iterations: {n_iterations}")
    print(f"   Total operations: {n_iterations * n_neurons * n_neurons} (O(n² × iterations))")
    
    # Initialize network similar to Huey
    np.random.seed(42)  # Reproducible results
    
    # Create initial activations (concepts/words)
    activations = np.random.rand(n_neurons).astype(np.float32)
    
    # Create connection matrix (symmetric, like Hebbian networks)
    connections = np.random.rand(n_neurons, n_neurons).astype(np.float32)
    connections = (connections + connections.T) / 2
    
    print(f"   Network density: {np.count_nonzero(connections) / connections.size * 100:.1f}%")
    
    # Evolution loop (this is what was slow without GPU)
    start_time = time.perf_counter()
    
    for iteration in range(n_iterations):
        # This is the O(n²) bottleneck that needs GPU acceleration
        connections, activations = gpu_interface.hebbian_update(connections, activations)
        
        # Add small learning perturbation (simulate learning)
        if iteration % 10 == 0:
            learning_rate = 0.01
            activations += np.random.normal(0, learning_rate, n_neurons).astype(np.float32)
            activations = np.clip(activations, 0, 1)  # Keep in valid range
        
        # Progress update
        if iteration % 20 == 0:
            print(f"   Iteration {iteration}/{n_iterations} - Activation sum: {np.sum(activations):.2f}")
    
    total_time = time.perf_counter() - start_time
    
    # Calculate throughput
    operations_per_second = (n_iterations * n_neurons * n_neurons) / total_time
    
    print(f"   ✅ Completed {n_iterations} iterations in {total_time:.2f}s")
    print(f"   ⚡ Operations/second: {operations_per_second:.0f}")
    print(f"   📊 Final activation sum: {np.sum(activations):.4f}")
    
    return total_time, operations_per_second

def test_different_network_sizes():
    """Test performance across different network sizes like Huey encounters."""
    
    print("🚀 Testing Huey GPU Restoration Across Network Sizes")
    print("=" * 60)
    
    # Initialize GPU interface with optimized thresholds
    gpu_interface = HueyPyTorchGPU(use_gpu=True, min_size_for_gpu=500)
    
    # Test sizes that represent real Huey networks
    # Small: Early conversation (few concepts)
    # Medium: Active conversation (many concepts)  
    # Large: Dense conversation analysis (Feynman file size)
    test_cases = [
        {"neurons": 100, "iterations": 50, "description": "Small network (early conversation)"},
        {"neurons": 300, "iterations": 75, "description": "Medium network (active conversation)"},
        {"neurons": 500, "iterations": 100, "description": "Large network (dense analysis)"},
        {"neurons": 1000, "iterations": 50, "description": "Very large network (Feynman-size)"}
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases):
        print(f"\n🧪 Test {i+1}/4: {test_case['description']}")
        print("-" * 50)
        
        try:
            exec_time, ops_per_sec = simulate_huey_network_evolution(
                gpu_interface, 
                test_case["neurons"], 
                test_case["iterations"]
            )
            
            results.append({
                "neurons": test_case["neurons"],
                "iterations": test_case["iterations"], 
                "time": exec_time,
                "ops_per_sec": ops_per_sec,
                "description": test_case["description"]
            })
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            results.append(None)
    
    # Performance analysis
    print(f"\n📈 PERFORMANCE ANALYSIS")
    print("=" * 40)
    
    gpu_stats = gpu_interface.get_performance_stats()
    print(f"GPU Operations: {gpu_stats['kernel_calls']}")
    print(f"GPU Time: {gpu_stats['total_kernel_time']:.3f}s")
    print(f"GPU Enabled: {gpu_stats['gpu_enabled']}")
    
    print(f"\n📊 Results Summary:")
    print("Size".ljust(12), "Time".ljust(10), "Ops/sec".ljust(15), "Performance")
    print("-" * 55)
    
    for result in results:
        if result:
            perf_rating = "🚀 EXCELLENT" if result["ops_per_sec"] > 1e6 else \
                         "✅ GOOD" if result["ops_per_sec"] > 5e5 else \
                         "🆗 OKAY" if result["ops_per_sec"] > 1e5 else \
                         "❌ SLOW"
            
            print(f"{result['neurons']} neurons".ljust(12), 
                  f"{result['time']:.2f}s".ljust(10),
                  f"{result['ops_per_sec']:.0f}".ljust(15),
                  perf_rating)
    
    # Compare to original Feynman performance
    print(f"\n🎯 RESTORATION ASSESSMENT")
    print("-" * 30)
    
    feynman_result = next((r for r in results if r and r["neurons"] == 1000), None)
    
    if feynman_result:
        # Original Feynman processing was ~20 seconds with GPU
        # Without GPU it became much slower (the "surgery" broke it)
        target_time = 20.0  # Original GPU performance
        actual_time = feynman_result["time"]
        
        if actual_time <= target_time * 1.5:  # Within 50% of original
            print("✅ GPU acceleration RESTORED!")
            print(f"   Target: <{target_time}s, Actual: {actual_time:.2f}s")
            print("   Performance matches pre-surgery levels")
        elif actual_time <= target_time * 3:
            print("⚠️  Partial restoration")
            print(f"   Target: <{target_time}s, Actual: {actual_time:.2f}s")
            print("   Some acceleration present but not optimal")
        else:
            print("❌ GPU acceleration NOT restored")
            print(f"   Target: <{target_time}s, Actual: {actual_time:.2f}s")
            print("   Still running primarily on CPU")
    
    return results

def benchmark_against_cpu_only():
    """Benchmark GPU vs CPU-only to measure speedup."""
    
    print(f"\n⚖️  GPU vs CPU Comparison")
    print("-" * 30)
    
    n_neurons = 500
    n_iterations = 50
    
    # GPU test
    print("Testing with GPU acceleration...")
    gpu_interface = HueyPyTorchGPU(use_gpu=True, min_size_for_gpu=100)
    gpu_time, _ = simulate_huey_network_evolution(gpu_interface, n_neurons, n_iterations)
    
    print(f"\nTesting CPU-only...")
    cpu_interface = HueyPyTorchGPU(use_gpu=False)
    cpu_time, _ = simulate_huey_network_evolution(cpu_interface, n_neurons, n_iterations)
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    
    print(f"\n📊 SPEEDUP ANALYSIS")
    print(f"   CPU Time: {cpu_time:.2f}s")
    print(f"   GPU Time: {gpu_time:.2f}s")
    print(f"   Speedup: {speedup:.1f}x")
    
    if speedup >= 2.0:
        print("   🚀 EXCELLENT GPU acceleration!")
    elif speedup >= 1.5:
        print("   ✅ GOOD GPU acceleration")
    elif speedup >= 1.1:
        print("   🆗 MARGINAL GPU benefit")
    else:
        print("   ❌ No significant GPU speedup")
    
    return speedup

if __name__ == "__main__":
    # Run comprehensive GPU restoration test
    results = test_different_network_sizes()
    
    # Direct comparison
    speedup = benchmark_against_cpu_only()
    
    print(f"\n🏁 FINAL VERDICT")
    print("=" * 30)
    
    if any(r and r["ops_per_sec"] > 5e5 for r in results if r) and speedup > 1.2:
        print("✅ Huey GPU acceleration has been RESTORED!")
        print("   PyTorch MPS successfully replaced JAX Metal")
        print("   Ready for Feynman file processing")
    else:
        print("⚠️  GPU acceleration partially working")
        print("   May need further optimization for full restoration")
        
    print(f"\n💡 NEXT STEPS")
    print("1. Integrate PyTorch GPU interface into Huey codebase")
    print("2. Test with actual Feynman conversation file")
    print("3. Verify 20-second processing time is achievable")