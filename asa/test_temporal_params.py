#!/usr/bin/env python3
"""
Parameter sensitivity test for temporal learning.
Test different tau and learning rate combinations to understand 
what causes zero-matrix problems.
"""

from huey_temporal_experiment import HueyTemporalExperiment

def test_parameter_combination(tau, eta_fwd, eta_fb, test_name):
    """Test a specific parameter combination."""
    print(f"\n{'='*50}")
    print(f"TEST: {test_name}")
    print(f"tau={tau}, eta_fwd={eta_fwd}, eta_fb={eta_fb}")
    print(f"{'='*50}")
    
    huey = HueyTemporalExperiment(
        max_neurons=50,
        use_temporal_learning=True,
        tau=tau,
        eta_fwd=eta_fwd,
        eta_fb=eta_fb,
        boundary_penalty=0.5
    )
    
    huey.add_speaker("Test", ['i', 'me'], ['you'])
    
    # Short test text
    test_text = "cats like to sleep in warm sunny places during afternoon"
    huey.process_speaker_text("Test", test_text)
    
    debug = huey.get_debug_summary()
    
    # Calculate theoretical weights for lag=1,2,3
    import math
    weights = [math.exp(-lag/tau) * eta_fwd for lag in [1,2,3]]
    
    print(f"Results:")
    print(f"  Connections: {debug['updates']['total_connections']}")
    print(f"  Nonzero: {debug['updates']['nonzero_connections']}")
    print(f"  Avg strength: {debug['network_stats']['avg_connection_strength']:.8f}")
    print(f"  Max strength: {debug['network_stats']['max_connection_strength']:.8f}")
    print(f"  Theoretical lag-1 weight: {weights[0]:.8f}")
    print(f"  Theoretical lag-2 weight: {weights[1]:.8f}")
    print(f"  Theoretical lag-3 weight: {weights[2]:.8f}")
    
    if debug['network_stats']['avg_connection_strength'] < 1e-12:
        print(f"  ❌ ZERO MATRIX PROBLEM!")
        return False
    else:
        print(f"  ✅ Non-zero connections")
        return True

if __name__ == "__main__":
    print("🧪 TEMPORAL LEARNING PARAMETER SENSITIVITY TEST")
    
    # Test cases that might cause zero matrices
    test_cases = [
        # Current working parameters
        (3.0, 0.01, 0.002, "CURRENT WORKING"),
        
        # Very small learning rates (might cause zeros)
        (3.0, 1e-6, 1e-7, "TINY LEARNING RATES"),
        (3.0, 1e-8, 1e-9, "MICROSCOPIC RATES"),
        
        # Very large tau (might cause zeros due to minimal decay)
        (100.0, 0.01, 0.002, "LARGE TAU"),
        (1000.0, 0.01, 0.002, "HUGE TAU"),
        
        # Very small tau (might cause zeros due to rapid decay)  
        (0.1, 0.01, 0.002, "TINY TAU"),
        (0.01, 0.01, 0.002, "MICROSCOPIC TAU"),
        
        # Combination that definitely should work
        (2.0, 0.1, 0.02, "LARGE LEARNING RATES"),
        
        # Yesterday's suspected problem parameters
        (3.0, 1e-4, 1e-5, "SUSPECTED PROBLEM PARAMS"),
    ]
    
    working_count = 0
    total_count = len(test_cases)
    
    for tau, eta_fwd, eta_fb, name in test_cases:
        success = test_parameter_combination(tau, eta_fwd, eta_fb, name)
        if success:
            working_count += 1
    
    print(f"\n{'='*60}")
    print(f"PARAMETER SENSITIVITY SUMMARY")
    print(f"{'='*60}")
    print(f"Working parameter sets: {working_count}/{total_count}")
    print(f"Failed parameter sets: {total_count - working_count}")
    
    if working_count == total_count:
        print("✅ All parameter combinations worked - no zero matrix problem!")
    else:
        print(f"❌ {total_count - working_count} combinations failed - parameter sensitivity confirmed")
    
    print(f"\n🧪 Parameter Sensitivity Test Complete!")