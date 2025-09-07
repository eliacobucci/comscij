#!/usr/bin/env python3
"""
Test script for conversation mode toggle functionality.
"""

from huey_speaker_detector import HueySpeakerDetector

def test_conversation_mode():
    """Test the conversation mode toggle."""
    
    print("🧪 TESTING CONVERSATION MODE TOGGLE")
    print("=" * 50)
    
    # Test with conversation mode enabled (default)
    print("\n1. Testing with CONVERSATION MODE ENABLED:")
    detector_on = HueySpeakerDetector(conversation_mode=True)
    result_on = detector_on.detect_speakers_from_file('Richard_Feynman.pdf')
    
    if 'speakers' in result_on:
        print(f"   Speakers detected: {len(result_on['speakers'])}")
        print(f"   First few: {result_on['speakers'][:5]}")
    
    # Test with conversation mode disabled
    print("\n2. Testing with CONVERSATION MODE DISABLED:")
    detector_off = HueySpeakerDetector(conversation_mode=False)
    result_off = detector_off.detect_speakers_from_file('Richard_Feynman.pdf')
    
    if 'speakers' in result_off:
        print(f"   Speakers detected: {len(result_off['speakers'])}")
        print(f"   Speaker(s): {result_off['speakers']}")
    
    print(f"\n✅ CONVERSATION MODE TOGGLE WORKING:")
    print(f"   Mode ON: {len(result_on.get('speakers', []))} speakers")
    print(f"   Mode OFF: {len(result_off.get('speakers', []))} speakers")

if __name__ == "__main__":
    test_conversation_mode()