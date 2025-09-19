#!/usr/bin/env python3
"""
Test if the segmentation fix works for Asian text
"""

import re

def _is_valid_segment_old(segment: str) -> bool:
    """Old broken version"""
    has_letters = bool(re.search(r'[a-zA-Z]', segment))
    reasonable_length = 10 <= len(segment) <= 1000
    return has_letters and reasonable_length

def _is_valid_segment_new(segment: str) -> bool:
    """New fixed version"""
    has_letters = bool(re.search(r'[a-zA-Z\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', segment))
    min_length = 3 if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', segment) else 10
    reasonable_length = min_length <= len(segment) <= 1000
    return has_letters and reasonable_length

def test_segment_validation():
    test_segments = [
        "蓝色",  # Chinese - short
        "我记得三个梦。",  # Chinese - medium  
        "梦都是蓝色的。",  # Chinese - medium
        "蓝色\n\n我记得三个梦。\n\n梦都是蓝色的。",  # Full Chinese text
        "Hello world this is English text",  # English
        "Hi",  # English - too short
        "こんにちは",  # Japanese
        "안녕하세요",  # Korean
        "123",  # Numbers only
        "!!!",  # Punctuation only
    ]
    
    print("🧪 TESTING SEGMENT VALIDATION")
    print("="*60)
    print(f"{'Segment':<40} {'Old':<8} {'New':<8}")
    print("-"*60)
    
    for segment in test_segments:
        old_result = _is_valid_segment_old(segment)
        new_result = _is_valid_segment_new(segment)
        print(f"{segment[:35]:<35} {old_result!s:<8} {new_result!s:<8}")
    
    # Test the specific Chinese text from our test file
    chinese_text = "蓝色\n\n我记得三个梦。\n\n梦都是蓝色的。"
    print(f"\n🧪 SPECIFIC TEST: Chinese text validation")
    print(f"Text: {repr(chinese_text)}")
    print(f"Old validation: {_is_valid_segment_old(chinese_text)}")
    print(f"New validation: {_is_valid_segment_new(chinese_text)}")

if __name__ == "__main__":
    test_segment_validation()