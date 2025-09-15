#!/usr/bin/env python3
"""
Test the simple temporal learning approach.
"""

from huey_temporal_simple import HueyTemporalSimple

# Test with temporal weights
print("🧪 Testing Simple Temporal Approach")
print("="*50)

# Create temporal version
huey_temporal = HueyTemporalSimple(
    max_neurons=100,
    window_size=8,
    use_temporal_weights=True,
    tau=3.0
)

huey_temporal.add_speaker("Test", ['i', 'me'], ['you'])

# Test simple text
test_text = "cats like to sleep in warm sunny places"
print(f"\nProcessing: '{test_text}'")

huey_temporal.process_speaker_text("Test", test_text)

print(f"\nResults:")
print(f"  Concepts: {len(huey_temporal.concept_neurons)}")
print(f"  Connections: {len(huey_temporal.connections)}")

# Test weight calculation
print(f"\nWeight calculations:")
for lag in range(1, 6):
    weight = huey_temporal._get_connection_weight(0, lag, 1.0)
    print(f"  lag={lag}: weight={weight:.6f}")

print(f"\n✅ Simple temporal test complete")