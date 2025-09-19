#!/usr/bin/env python3
"""
Test Mandarin processing directly in the temporal engine to isolate the issue.
"""

from huey_temporal_simple import HueyTemporalSimple

def test_mandarin_processing():
    print("🧪 Testing Mandarin processing directly...")
    
    # Create temporal network
    huey = HueyTemporalSimple(
        max_neurons=200,
        use_temporal_weights=True,
        tau=3.0,
        use_gpu_acceleration=False  # Use CPU for debugging
    )
    
    # Test the file processing
    result = huey.process_file_with_mode("blue_mandarin.txt", conversation_mode=True)
    
    print(f"\n📊 PROCESSING RESULTS:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Speakers registered: {result.get('speakers_registered', 0)}")
    print(f"   Exchanges processed: {result.get('exchanges_processed', 0)}")
    print(f"   Total concepts: {len(huey.concept_neurons)}")
    print(f"   Total connections: {len(huey.connections)}")
    
    if len(huey.concept_neurons) > 0:
        print(f"\n🎯 SAMPLE CONCEPTS:")
        for i, (word, neuron_id) in enumerate(list(huey.concept_neurons.items())[:10]):
            print(f"   {i+1:2d}. {word} → neuron_{neuron_id}")
    else:
        print("\n⚠️  NO CONCEPTS CREATED!")
        print("   This indicates the file processing failed")
    
    return huey

if __name__ == "__main__":
    huey = test_mandarin_processing()