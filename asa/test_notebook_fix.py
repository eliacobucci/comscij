#!/usr/bin/env python3
"""
Test the fixed system with the TrumpInterview.txt file to ensure
speakers now have different self-concept masses.
"""

from huey_complete_platform import HueyCompletePlatform
from huey_speaker_detector import HueySpeakerDetector

def test_trump_interview_fix():
    """Test the fix with the actual TrumpInterview.txt file."""
    
    print("🧪 TESTING TRUMP INTERVIEW WITH FIXES")
    print("=" * 60)
    
    # Initialize platform
    huey = HueyCompletePlatform(
        session_name="test_trump_fix",
        max_neurons=200,
        window_size=7,
        learning_rate=0.15
    )
    
    # Process the Trump interview file
    print("\n📄 PROCESSING TRUMP INTERVIEW FILE")
    detector = HueySpeakerDetector()
    file_result = detector.process_conversation_file("TrumpInterview.txt")
    
    if 'error' in file_result:
        print(f"❌ Error: {file_result['error']}")
        return
    
    print(f"✅ File processed: {len(file_result['speakers_info'])} speakers, {len(file_result['conversation_data'])} exchanges")
    
    # Register speakers and process conversation
    huey.register_speakers(file_result['speakers_info'])
    analysis_results = huey.process_conversation(file_result['conversation_data'])
    
    print(f"\n🧠 NETWORK STATE:")
    print(f"Total neurons: {huey.network.neuron_count}")
    print(f"Total connections: {len(huey.network.connections)}")
    
    # Analyze individual speakers
    print(f"\n👥 INDIVIDUAL SPEAKER ANALYSIS:")
    speaker_masses = {}
    
    for speaker in huey.network.speakers:
        analysis = huey.network.analyze_speaker_self_concept(speaker)
        speaker_masses[speaker] = analysis['self_concept_mass']
        print(f"  {speaker:15} → Self-concept mass: {analysis['self_concept_mass']:.3f}")
        print(f"                   → Blocks processed: {analysis['blocks_processed']}")
    
    # Check if masses are different
    masses = list(speaker_masses.values())
    all_same = all(abs(m - masses[0]) < 0.001 for m in masses)
    
    if all_same:
        print(f"\n❌ FAILED: All speakers still have identical masses")
        return False
    else:
        print(f"\n✅ SUCCESS: Speakers have different self-concept masses!")
        
        # Find highest and lowest
        max_speaker = max(speaker_masses.keys(), key=lambda s: speaker_masses[s])
        min_speaker = min(speaker_masses.keys(), key=lambda s: speaker_masses[s])
        
        print(f"   Highest: {max_speaker} ({speaker_masses[max_speaker]:.3f})")
        print(f"   Lowest:  {min_speaker} ({speaker_masses[min_speaker]:.3f})")
        print(f"   Range:   {max(masses) - min(masses):.3f}")
    
    # Test speaker comparison query
    print(f"\n🔍 TESTING SPEAKER COMPARISON QUERY:")
    speaker_list = list(huey.network.speakers.keys())
    comparison = huey.query_concepts("speaker_differences", speakers=speaker_list)
    
    if 'individual_analyses' in comparison and comparison['individual_analyses']:
        print("✅ Speaker comparison query working!")
        for speaker, analysis in comparison['individual_analyses'].items():
            mass = analysis.get('self_concept_mass', 0)
            print(f"   {speaker}: {mass:.3f}")
    else:
        print("❌ Speaker comparison query still returning blank results")
        print(f"   Result keys: {list(comparison.keys())}")
    
    return True

if __name__ == "__main__":
    success = test_trump_interview_fix()
    print(f"\n{'='*60}")
    print(f"OVERALL TEST: {'PASSED' if success else 'FAILED'}")