#!/usr/bin/env python3
"""
Quick test with just a few lines from TrumpInterview to verify the fix.
"""

from huey_complete_platform import HueyCompletePlatform

def test_quick_fix():
    """Test with just a few sample exchanges."""
    
    print("🧪 QUICK TEST OF SPEAKER FIX")
    print("=" * 40)
    
    # Initialize platform
    huey = HueyCompletePlatform(
        session_name="quick_test",
        max_neurons=100,
        window_size=7,
        learning_rate=0.15
    )
    
    # Sample conversation data (first few exchanges from TrumpInterview)
    conversation_data = [
        ("Times", "Obviously, a lot has happened. We want to have a conversation about what your first 100 days have looked like in office."),
        ("Trump", "There's a lot, right? A lot of things happening. I think I'm using it as it was meant to be used. I feel that we've had a very successful presidency."),
        ("Times", "You know better than anyone that the President of the United States is the most powerful person in the world."),
        ("Trump", "Well, I don't feel I'm expanding it. I think I'm using it as it was meant to be used. My approach has been very effective."),
        ("Times", "What we're driving at is that you've taken congressional authority on trade and appropriations."),
        ("Trump", "No, I think that what I'm doing is exactly what I've campaigned on. If you look at what I campaigned on, I said I was going to do these things.")
    ]
    
    # Register speakers
    speakers_info = [
        ("Times", "Times Reporter", "journalist"),
        ("Trump", "Donald Trump", "president")
    ]
    huey.register_speakers(speakers_info)
    
    print(f"\n📝 PROCESSING {len(conversation_data)} EXCHANGES...")
    
    # Process conversation
    huey.process_conversation(conversation_data)
    
    print(f"\n🧠 NETWORK STATE:")
    print(f"Total neurons: {huey.network.neuron_count}")
    
    # Analyze speakers
    print(f"\n👥 SPEAKER ANALYSIS:")
    for speaker in huey.network.speakers:
        analysis = huey.network.analyze_speaker_self_concept(speaker)
        print(f"  {speaker:8} → Self-concept mass: {analysis['self_concept_mass']:.3f}")
    
    # Test comparison query
    print(f"\n🔍 COMPARISON QUERY:")
    comparison = huey.query_concepts("speaker_differences", speakers=["Times", "Trump"])
    
    if 'individual_analyses' in comparison:
        print("✅ Query working!")
        for speaker, data in comparison['individual_analyses'].items():
            print(f"   {speaker}: {data.get('self_concept_mass', 0):.3f}")
    else:
        print("❌ Query still blank")
    
    return True

if __name__ == "__main__":
    test_quick_fix()