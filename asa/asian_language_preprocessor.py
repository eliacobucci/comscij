#!/usr/bin/env python3
"""
Asian Language Preprocessor for Huey
Safe tokenization preprocessing that converts Asian text to space-separated format
for compatibility with existing Huey temporal learning pipeline.

Based on archaeological evidence from Springfield MO sessions.
"""

import re

class AsianLanguagePreprocessor:
    """Preprocesses Asian languages for Huey compatibility."""
    
    def __init__(self):
        # Japanese multi-character units to preserve (from archaeological code)
        self.japanese_multi_char = [
            'あなた', 'あなたの', 'わたし', 'わたくし', 'わたしの', 
            'きみの', 'ぼくの', 'です', 'ます', 'できる', 'できます',
            'おもう', 'おもいます', 'のうりょく', 'ちのう'
        ]
        
        # Korean multi-character units (common particles and endings)
        self.korean_multi_char = [
            'hamnida', 'sseumnida', 'imnida', 'seumnida', 'haseyo', 'eul', 'neun', 'ga', 'wa'
        ]
        
    def detect_language(self, text):
        """Simple language detection based on character ranges."""
        # Japanese (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        
        # Korean (Hangul)
        if re.search(r'[\uAC00-\uD7AF]', text):
            return 'korean'
        
        # Chinese (CJK Unified Ideographs)
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        
        # Default to European (no preprocessing needed)
        return 'european'
    
    def japanese_tokenize(self, text):
        """
        Japanese tokenization - preserve key multi-character units, split others by character.
        Based on archaeological japanese_self_concept.py
        """
        # Replace multi-character units with temporary tokens
        temp_text = text
        replacements = {}
        
        for i, unit in enumerate(self.japanese_multi_char):
            if unit in temp_text:
                temp_token = f"__{i}__"
                replacements[temp_token] = unit
                temp_text = temp_text.replace(unit, f" {temp_token} ")
        
        # Split and process
        tokens = []
        for part in temp_text.split():
            if part in replacements:
                tokens.append(replacements[part])
            elif part.startswith('__') and part.endswith('__'):
                tokens.append(replacements[part])
            else:
                # Split into characters, skip punctuation
                for char in part:
                    if char not in '、。？！　':
                        tokens.append(char)
        
        return [t for t in tokens if t.strip()]
    
    def chinese_tokenize(self, text):
        """Simple Chinese tokenization - character level with space preservation."""
        tokens = []
        for char in text:
            if char not in '，。？！　\n\r\t':  # Skip punctuation and whitespace
                tokens.append(char)
        return tokens
    
    def korean_tokenize(self, text):
        """Korean tokenization for romanized text."""
        # For romanized Korean, split by spaces and preserve common particles
        tokens = []
        for word in text.split():
            # Check if it's a multi-character particle/ending to preserve
            if word.lower() in [unit.lower() for unit in self.korean_multi_char]:
                tokens.append(word)
            else:
                # For other words, split by syllables (simplified approach)
                tokens.append(word)
        return tokens
    
    def preprocess_text(self, text):
        """
        Main preprocessing function that converts Asian text to space-separated format.
        Returns processed text ready for standard Huey pipeline.
        """
        language = self.detect_language(text)
        
        if language == 'japanese':
            tokens = self.japanese_tokenize(text)
        elif language == 'chinese':
            tokens = self.chinese_tokenize(text)
        elif language == 'korean':
            tokens = self.korean_tokenize(text)
        else:
            # European languages - no preprocessing needed
            return text
        
        # Convert to space-separated format for Huey
        processed_text = ' '.join(tokens)
        
        print(f"🌏 Language detected: {language}")
        print(f"📝 Original: {text}")
        print(f"🔤 Tokenized: {processed_text[:100]}...")
        
        return processed_text

# Test function
if __name__ == "__main__":
    preprocessor = AsianLanguagePreprocessor()
    
    # Test Japanese
    japanese_text = "こんにちは、わたしをりかいできますか？"
    print("🇯🇵 Testing Japanese:")
    result = preprocessor.preprocess_text(japanese_text)
    print(f"Result: {result}\n")
    
    # Test Chinese
    chinese_text = "你好，你能理解我吗？"
    print("🇨🇳 Testing Chinese:")
    result = preprocessor.preprocess_text(chinese_text)
    print(f"Result: {result}\n")
    
    # Test European (should pass through unchanged)
    english_text = "Hello, can you understand me?"
    print("🇺🇸 Testing English:")
    result = preprocessor.preprocess_text(english_text)
    print(f"Result: {result}")