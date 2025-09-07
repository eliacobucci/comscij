#!/usr/bin/env python3
"""
Huey Speaker Detector: Automatic speaker identification for conversation files.
Detects speaker patterns in text files and converts to Huey format.

Copyright (c) 2025 Emary Iacobucci and Joseph Woelfel. All rights reserved.
"""

import re
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

class HueySpeakerDetector:
    """
    Automatic speaker detection for conversation text files.
    Supports multiple common conversation formats.
    """
    
    def __init__(self):
        """Initialize the speaker detector."""
        self.detected_speakers = set()
        self.conversation_data = []
        
        # Common speaker identification patterns
        self.patterns = [
            # "Speaker: text" or "Speaker - text"
            r'^([A-Za-z][A-Za-z0-9_\s\-]*?):\s*(.+)$',
            r'^([A-Za-z][A-Za-z0-9_\s\-]*?)\s*-\s*(.+)$',
            
            # "[Speaker] text" or "(Speaker) text"
            r'^\[([A-Za-z][A-Za-z0-9_\s\-]*?)\]\s*(.+)$',
            r'^\(([A-Za-z][A-Za-z0-9_\s\-]*?)\)\s*(.+)$',
            
            # "Speaker> text" or "> Speaker: text"
            r'^([A-Za-z][A-Za-z0-9_\s\-]*?)>\s*(.+)$',
            r'^>\s*([A-Za-z][A-Za-z0-9_\s\-]*?):\s*(.+)$',
            
            # Chat-style formats like "12:34 Speaker: text"
            r'^\d{1,2}:\d{2}\s+([A-Za-z][A-Za-z0-9_\s\-]*?):\s*(.+)$',
            
            # Numbered speakers "1. text" with context
            r'^(\d+)\.\s*(.+)$',
            
            # Script format "SPEAKER\n text"
            r'^([A-Z][A-Z\s]*?)$'  # Separate pattern for all-caps names
        ]
        
        print("🔍 Huey Speaker Detector initialized")
        print("   Supported formats:")
        print("   • Speaker: text")
        print("   • Speaker - text")
        print("   • [Speaker] text")
        print("   • (Speaker) text")
        print("   • Speaker> text")
        print("   • Numbered exchanges")
        print("   • Script format")
    
    def detect_speakers_from_file(self, filename: str) -> Dict[str, any]:
        """
        Detect speakers and extract conversation from a text file.
        
        Args:
            filename: Path to conversation text file
            
        Returns:
            Dictionary with detected speakers and conversation data
        """
        print(f"\n📄 ANALYZING FILE: {filename}")
        print("-" * 50)
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            return {'error': f"File not found: {filename}"}
        except Exception as e:
            return {'error': f"Error reading file: {str(e)}"}
        
        # Handle line breaks between speaker and text - more aggressive approach
        # First, handle the specific "Speaker: ActualName" format
        content = re.sub(r'^Speaker:\s*([A-Za-z][A-Za-z0-9_\s\-]*?)\s*\n+\s*([^\n]+)', r'\1: \2', content, flags=re.MULTILINE)
        
        # Convert "Speaker:\n text" to "Speaker: text"
        content = re.sub(r'([A-Za-z][A-Za-z0-9_\s]{1,20}?):\s*\n+\s*([^\n]+)', r'\1: \2', content, flags=re.MULTILINE)
        content = re.sub(r'([A-Za-z][A-Za-z0-9_\s]{1,20}?)\s*-\s*\n+\s*([^\n]+)', r'\1 - \2', content, flags=re.MULTILINE)
        content = re.sub(r'\[([A-Za-z][A-Za-z0-9_\s]{1,20}?)\]\s*\n+\s*([^\n]+)', r'[\1] \2', content, flags=re.MULTILINE)
        content = re.sub(r'\(([A-Za-z][A-Za-z0-9_\s]{1,20}?)\)\s*\n+\s*([^\n]+)', r'(\1) \2', content, flags=re.MULTILINE)
        
        # Handle all-caps speaker names on separate lines
        content = re.sub(r'^([A-Z][A-Z\s]{1,20}?)\s*\n+\s*([a-zA-Z][^\n]+)', r'\1: \2', content, flags=re.MULTILINE)
        
        # Handle speaker names followed by multiple newlines
        content = re.sub(r'([A-Za-z][A-Za-z0-9_\s]{1,20}?):\s*\n{2,}\s*([^\n]+)', r'\1: \2', content, flags=re.MULTILINE)
        
        # Split back into lines and clean
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # Try different detection strategies
        strategies = [
            self._detect_labeled_speakers,
            self._detect_alternating_speakers,
            self._detect_paragraph_speakers,
            self._detect_numbered_speakers
        ]
        
        best_result = None
        best_confidence = 0
        
        for strategy in strategies:
            result = strategy(lines)
            if result and result['confidence'] > best_confidence:
                best_result = result
                best_confidence = result['confidence']
        
        if best_result:
            print(f"✅ DETECTION SUCCESS")
            print(f"   Strategy: {best_result['strategy']}")
            print(f"   Confidence: {best_result['confidence']:.2f}")
            print(f"   Speakers detected: {len(best_result['speakers'])}")
            for speaker in best_result['speakers']:
                print(f"   • {speaker}")
            print(f"   Exchanges: {len(best_result['conversation'])}")
            
            return best_result
        else:
            print("❌ DETECTION FAILED")
            print("   Could not identify speaker pattern")
            return {
                'error': 'Could not detect speakers automatically',
                'suggestions': self._suggest_manual_format(lines[:10])
            }
    
    def _detect_labeled_speakers(self, lines: List[str]) -> Optional[Dict]:
        """Detect speakers with explicit labels (Speaker: text)."""
        
        conversation = []
        speakers = set()
        matches = 0
        
        for line in lines:
            # Try each pattern
            for pattern in self.patterns[:7]:  # Skip numbered and script patterns
                match = re.match(pattern, line)
                if match:
                    speaker = match.group(1).strip()
                    text = match.group(2).strip()
                    
                    # Clean speaker name
                    speaker = self._clean_speaker_name(speaker)
                    
                    if text and len(speaker) <= 20:  # Reasonable speaker name length
                        conversation.append((speaker, text))
                        speakers.add(speaker)
                        matches += 1
                        break
        
        if matches > 0 and len(speakers) >= 2:
            confidence = min(matches / len(lines), 1.0)
            return {
                'strategy': 'labeled_speakers',
                'confidence': confidence,
                'speakers': list(speakers),
                'conversation': conversation
            }
        
        return None
    
    def _detect_alternating_speakers(self, lines: List[str]) -> Optional[Dict]:
        """Detect alternating speakers without explicit labels."""
        
        if len(lines) < 4:  # Need at least 4 lines to detect alternating
            return None
        
        # Assume two speakers alternating
        speaker_a = "Speaker_A"
        speaker_b = "Speaker_B"
        
        conversation = []
        for i, line in enumerate(lines):
            if line:  # Skip empty lines
                speaker = speaker_a if i % 2 == 0 else speaker_b
                conversation.append((speaker, line))
        
        # Only return if we have substantial conversation
        if len(conversation) >= 4:
            return {
                'strategy': 'alternating_speakers',
                'confidence': 0.6,  # Medium confidence
                'speakers': [speaker_a, speaker_b],
                'conversation': conversation
            }
        
        return None
    
    def _detect_paragraph_speakers(self, lines: List[str]) -> Optional[Dict]:
        """Detect speakers by paragraph breaks."""
        
        paragraphs = []
        current_paragraph = []
        
        for line in lines:
            if line.strip():
                current_paragraph.append(line)
            else:
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
        
        # Add final paragraph
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
        
        if len(paragraphs) >= 3:  # Need at least 3 paragraphs
            conversation = []
            for i, paragraph in enumerate(paragraphs):
                speaker = f"Speaker_{chr(65 + i % 3)}"  # A, B, C rotation
                conversation.append((speaker, paragraph))
            
            return {
                'strategy': 'paragraph_speakers',
                'confidence': 0.5,  # Lower confidence
                'speakers': [f"Speaker_{chr(65 + i)}" for i in range(min(3, len(paragraphs)))],
                'conversation': conversation
            }
        
        return None
    
    def _detect_numbered_speakers(self, lines: List[str]) -> Optional[Dict]:
        """Detect numbered exchanges (1. text, 2. text)."""
        
        conversation = []
        speakers = {}
        current_number = 1
        
        for line in lines:
            match = re.match(r'^(\d+)\.\s*(.+)$', line)
            if match:
                number = int(match.group(1))
                text = match.group(2).strip()
                
                # Map numbers to speakers (assuming alternating or cyclic)
                if number not in speakers:
                    speaker_name = f"Speaker_{chr(65 + (number - 1) % 3)}"  # A, B, C
                    speakers[number] = speaker_name
                
                conversation.append((speakers[number], text))
        
        if len(conversation) >= 3:
            unique_speakers = list(set(speakers.values()))
            return {
                'strategy': 'numbered_speakers',
                'confidence': 0.7,
                'speakers': unique_speakers,
                'conversation': conversation
            }
        
        return None
    
    def _clean_speaker_name(self, speaker: str) -> str:
        """Clean and normalize speaker names."""
        # Remove common prefixes/suffixes
        speaker = re.sub(r'^(Mr|Ms|Mrs|Dr|Professor|Prof)\.?\s*', '', speaker, flags=re.IGNORECASE)
        
        # Convert to title case and remove extra spaces
        speaker = re.sub(r'\s+', ' ', speaker.strip())
        
        # Make it a valid identifier (replace spaces with underscores for consistency)
        if ' ' in speaker:
            speaker = speaker.replace(' ', '_')
        
        return speaker
    
    def _suggest_manual_format(self, sample_lines: List[str]) -> List[str]:
        """Suggest manual formatting based on the file content."""
        suggestions = [
            "Try formatting your file with one of these patterns:",
            "",
            "Format 1 - Speaker labels:",
            "Alice: I think this is interesting.",
            "Bob: I agree completely.",
            "",
            "Format 2 - Bracketed speakers:",
            "[Alice] I think this is interesting.",
            "[Bob] I agree completely.",
            "",
            "Format 3 - Numbered exchanges:",
            "1. I think this is interesting.",
            "2. I agree completely.",
            "",
            "Your file sample:",
        ]
        
        for i, line in enumerate(sample_lines[:5]):
            suggestions.append(f"Line {i+1}: {line}")
        
        return suggestions
    
    def process_conversation_file(self, filename: str, force_speakers: List[str] = None) -> Dict:
        """
        Complete processing of a conversation file for Huey.
        
        Args:
            filename: Path to conversation file
            force_speakers: Optional list to force specific speaker names
            
        Returns:
            Processed data ready for Huey
        """
        print(f"\n🔄 PROCESSING CONVERSATION FILE: {filename}")
        print("=" * 60)
        
        # Detect speakers automatically
        detection_result = self.detect_speakers_from_file(filename)
        
        if 'error' in detection_result:
            return detection_result
        
        # Override speakers if provided
        if force_speakers:
            print(f"\n🔧 OVERRIDING DETECTED SPEAKERS:")
            print(f"   Original: {detection_result['speakers']}")
            print(f"   Override: {force_speakers}")
            
            # Remap conversation to use forced speaker names
            speaker_mapping = {}
            detected_speakers = detection_result['speakers']
            
            for i, forced_speaker in enumerate(force_speakers):
                if i < len(detected_speakers):
                    speaker_mapping[detected_speakers[i]] = forced_speaker
            
            # Update conversation with new speaker names
            updated_conversation = []
            for speaker, text in detection_result['conversation']:
                new_speaker = speaker_mapping.get(speaker, speaker)
                updated_conversation.append((new_speaker, text))
            
            detection_result['conversation'] = updated_conversation
            detection_result['speakers'] = force_speakers[:len(detected_speakers)]
        
        # Prepare for Huey
        huey_ready = {
            'speakers_info': [(speaker, speaker.replace('_', ' '), 'participant') 
                             for speaker in detection_result['speakers']],
            'conversation_data': detection_result['conversation'],
            'detection_info': {
                'strategy': detection_result['strategy'],
                'confidence': detection_result['confidence'],
                'total_exchanges': len(detection_result['conversation'])
            }
        }
        
        print(f"\n✅ READY FOR HUEY ANALYSIS:")
        print(f"   Speakers: {len(huey_ready['speakers_info'])}")
        print(f"   Exchanges: {len(huey_ready['conversation_data'])}")
        print(f"   Detection confidence: {detection_result['confidence']:.2f}")
        
        return huey_ready

def demo_speaker_detection():
    """Demonstrate speaker detection with sample text."""
    
    print("🔍 HUEY SPEAKER DETECTOR DEMO")
    print("=" * 50)
    
    # Create sample conversation file
    sample_content = """Alice: I think this approach to analysis is really fascinating.
Bob: I completely agree with you. The methodology seems sound.
Alice: What I find most interesting is how patterns emerge naturally.
Bob: Yes, and the implications for understanding identity formation are significant.
Charlie: This is a great discussion. I wonder if we could extend this to other domains?
Alice: That's an excellent point. The principles might be universal.
Bob: We should definitely explore that possibility further."""
    
    # Write sample file
    with open('sample_conversation.txt', 'w') as f:
        f.write(sample_content)
    
    # Test detection
    detector = HueySpeakerDetector()
    result = detector.process_conversation_file('sample_conversation.txt')
    
    if 'error' not in result:
        print("\n📊 SAMPLE CONVERSATION DETECTED:")
        for speaker, text in result['conversation_data'][:3]:
            print(f"   {speaker}: {text[:50]}...")
    
    return detector

if __name__ == "__main__":
    demo_speaker_detection()