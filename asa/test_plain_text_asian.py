#!/usr/bin/env python3
"""
Test Asian preprocessing in Plain Text Mode specifically
"""

from huey_temporal_web_multilingual import segment_text_linguistically
from asian_language_preprocessor import AsianLanguagePreprocessor

def test_plain_text_asian():
    print("🧪 TESTING PLAIN TEXT MODE ASIAN PREPROCESSING")
    print("="*60)
    
    # Japanese content
    with open('test_japanese_simple.txt', 'r', encoding='utf-8') as f:
        japanese_content = f.read()
    
    print(f"1. Original Japanese: {repr(japanese_content)}")
    
    # Apply Asian preprocessing (the fix we just added)
    preprocessor = AsianLanguagePreprocessor()
    preprocessed_content = preprocessor.preprocess_text(japanese_content)
    
    print(f"2. Preprocessed: {repr(preprocessed_content)}")
    
    # Apply the web interface logic
    if preprocessed_content != japanese_content:
        # Asian language detected - use entire preprocessed content as single segment
        sentences = [preprocessed_content.strip()]
        print(f"3. Asian language detected - using unified segment: {len(sentences)} segments")
    else:
        # European language - use normal segmentation
        sentences = segment_text_linguistically(preprocessed_content)
        print(f"3. European language - segmented into: {len(sentences)} segments")
    
    for i, sentence in enumerate(sentences):
        print(f"   {i+1}: {repr(sentence[:100])}...")
    
    # Test if segments contain Japanese characters (should be space-separated now)
    has_space_separated = any(' ' in sentence and len(sentence.split()) > 3 for sentence in sentences)
    print(f"4. Contains space-separated tokens: {has_space_separated}")
    
    if has_space_separated:
        print("✅ Plain Text Mode Asian preprocessing is working!")
    else:
        print("❌ Plain Text Mode still has issues")

if __name__ == "__main__":
    test_plain_text_asian()