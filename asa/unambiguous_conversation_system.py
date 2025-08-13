#!/usr/bin/env python3
"""
Unambiguous Conversation System
A conversation logging system that makes speaker identification foolproof
by requiring explicit speaker tagging from the start.
"""

import json
import datetime
import os
from typing import Dict, List, Optional

class UnambiguousConversationSystem:
    """
    Conversation system that enforces clear speaker identification.
    No guessing, no algorithms - just explicit, unambiguous speaker tags.
    """
    
    def __init__(self, session_name: str = None):
        self.session_name = session_name or f"conversation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.speakers = {}
        self.conversation_log = []
        self.current_turn = 0
        
        print(f"🎙️  UNAMBIGUOUS CONVERSATION SYSTEM")
        print(f"Session: {self.session_name}")
        print(f"Rule: Every utterance MUST have explicit speaker identification")
        print()
    
    def register_speaker(self, speaker_id: str, full_name: str, role: str = "participant"):
        """Register a speaker with their details."""
        
        self.speakers[speaker_id] = {
            'full_name': full_name,
            'role': role,
            'utterance_count': 0,
            'total_words': 0,
            'first_appearance': None,
            'last_appearance': None
        }
        
        print(f"✅ Registered speaker: {speaker_id} ({full_name}) - {role}")
        return speaker_id
    
    def add_utterance(self, speaker_id: str, text: str, timestamp: datetime.datetime = None):
        """Add an utterance with mandatory speaker identification."""
        
        if speaker_id not in self.speakers:
            raise ValueError(f"❌ Speaker '{speaker_id}' not registered. Must register speakers first.")
        
        if not text.strip():
            raise ValueError("❌ Empty utterances not allowed")
        
        timestamp = timestamp or datetime.datetime.now()
        word_count = len(text.split())
        
        utterance = {
            'turn_id': self.current_turn,
            'speaker_id': speaker_id,
            'speaker_name': self.speakers[speaker_id]['full_name'],
            'text': text.strip(),
            'timestamp': timestamp.isoformat(),
            'word_count': word_count,
            'session': self.session_name
        }
        
        # Update speaker stats
        speaker_stats = self.speakers[speaker_id]
        speaker_stats['utterance_count'] += 1
        speaker_stats['total_words'] += word_count
        speaker_stats['last_appearance'] = timestamp.isoformat()
        
        if speaker_stats['first_appearance'] is None:
            speaker_stats['first_appearance'] = timestamp.isoformat()
        
        self.conversation_log.append(utterance)
        self.current_turn += 1
        
        print(f"[{speaker_id}]: {text[:80]}{'...' if len(text) > 80 else ''}")
        return utterance
    
    def quick_add(self, speaker_text_pairs: List[tuple]):
        """Quick way to add multiple utterances as (speaker_id, text) pairs."""
        
        for speaker_id, text in speaker_text_pairs:
            self.add_utterance(speaker_id, text)
    
    def get_speaker_statistics(self) -> Dict:
        """Get statistics for all speakers."""
        
        stats = {}
        for speaker_id, info in self.speakers.items():
            stats[speaker_id] = {
                'name': info['full_name'],
                'role': info['role'],
                'utterances': info['utterance_count'],
                'total_words': info['total_words'],
                'avg_words_per_utterance': info['total_words'] / max(info['utterance_count'], 1),
                'first_spoke': info['first_appearance'],
                'last_spoke': info['last_appearance']
            }
        
        return stats
    
    def export_conversation(self, filename: str = None) -> str:
        """Export the conversation to a structured JSON file."""
        
        filename = filename or f"/Users/josephwoelfel/asa/{self.session_name}_tagged_conversation.json"
        
        export_data = {
            'session_metadata': {
                'session_name': self.session_name,
                'start_time': self.conversation_log[0]['timestamp'] if self.conversation_log else None,
                'end_time': self.conversation_log[-1]['timestamp'] if self.conversation_log else None,
                'total_turns': len(self.conversation_log),
                'total_speakers': len(self.speakers),
                'export_timestamp': datetime.datetime.now().isoformat()
            },
            'speakers': self.speakers,
            'conversation': self.conversation_log,
            'speaker_statistics': self.get_speaker_statistics()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Conversation exported to: {filename}")
        return filename
    
    def export_for_self_concept_analysis(self, filename: str = None) -> str:
        """Export in format ready for self-concept analysis."""
        
        filename = filename or f"/Users/josephwoelfel/asa/{self.session_name}_self_concept_ready.json"
        
        # Convert to the format expected by our self-concept analysis system
        conversation_blocks = []
        
        for utterance in self.conversation_log:
            conversation_blocks.append({
                'speaker': utterance['speaker_id'],
                'text': utterance['text'],
                'confidence': 1.0,  # Perfect confidence since explicitly tagged
                'turn_id': utterance['turn_id'],
                'timestamp': utterance['timestamp']
            })
        
        analysis_data = {
            'metadata': {
                'total_turns': len(conversation_blocks),
                'speakers_identified': list(self.speakers.keys()),
                'confidence_threshold': 1.0,
                'source_session': self.session_name
            },
            'speaker_mappings': {
                speaker_id: {
                    'self_pronouns': ['i', 'me', 'my', 'mine', 'myself'],
                    'other_pronouns': ['you', 'your', 'yours', 'yourself']
                } for speaker_id in self.speakers.keys()
            },
            'conversation_blocks': conversation_blocks
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        print(f"🧠 Self-concept analysis file created: {filename}")
        return filename
    
    def print_conversation(self):
        """Print the conversation in a readable format."""
        
        print(f"\n📖 CONVERSATION TRANSCRIPT - {self.session_name}")
        print("=" * 80)
        
        for utterance in self.conversation_log:
            timestamp = datetime.datetime.fromisoformat(utterance['timestamp']).strftime("%H:%M:%S")
            speaker_name = utterance['speaker_name']
            text = utterance['text']
            
            print(f"[{timestamp}] {speaker_name}: {text}")
        
        print("=" * 80)
        
        # Print statistics
        stats = self.get_speaker_statistics()
        print(f"\n📊 CONVERSATION STATISTICS:")
        for speaker_id, speaker_stats in stats.items():
            print(f"   {speaker_stats['name']} ({speaker_id}): {speaker_stats['utterances']} utterances, {speaker_stats['total_words']} words")
    
    def validate_conversation(self) -> Dict:
        """Validate the conversation for completeness and consistency."""
        
        issues = []
        warnings = []
        
        # Check for empty conversation
        if not self.conversation_log:
            issues.append("No conversation recorded")
        
        # Check for single speaker (monologue)
        if len(self.speakers) == 1:
            warnings.append("Only one speaker - this is a monologue, not a conversation")
        
        # Check for very short utterances
        short_utterances = [u for u in self.conversation_log if u['word_count'] < 3]
        if short_utterances:
            warnings.append(f"{len(short_utterances)} very short utterances (< 3 words)")
        
        # Check speaker participation balance
        stats = self.get_speaker_statistics()
        if len(stats) > 1:
            utterance_counts = [s['utterances'] for s in stats.values()]
            max_utterances = max(utterance_counts)
            min_utterances = min(utterance_counts)
            
            if max_utterances > min_utterances * 5:
                warnings.append("Highly unbalanced conversation - one speaker dominates")
        
        validation_result = {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_turns': len(self.conversation_log),
            'speakers': len(self.speakers),
            'ready_for_analysis': len(issues) == 0 and len(self.conversation_log) > 0
        }
        
        return validation_result

def create_demo_conversation():
    """Create a demo conversation showing the system in action."""
    
    print("🎭 DEMO: UNAMBIGUOUS CONVERSATION SYSTEM")
    print("=" * 60)
    
    # Initialize system
    conv = UnambiguousConversationSystem("research_meeting_demo")
    
    # Register speakers with clear identification
    conv.register_speaker("joseph", "Dr. Joseph Woelfel", "Professor")
    conv.register_speaker("claude", "Claude", "AI Assistant") 
    conv.register_speaker("asa", "Asa", "Research Assistant")
    conv.register_speaker("evan", "Evan", "Graduate Student")
    
    print(f"\n🗣️  RECORDING CONVERSATION:")
    print("-" * 40)
    
    # Record conversation with explicit speaker tags
    conversation_data = [
        ("joseph", "Good morning everyone. Let's continue our work on self-concept formation. I've been thinking about the mathematical principles we discussed."),
        ("claude", "Good morning, Joseph. I'm ready to continue our research. The Hebbian learning approach has been producing fascinating results."),
        ("asa", "Hi everyone! I've been analyzing the cross-linguistic data we collected. The patterns are really interesting - individualistic languages show stronger self-concept formation."),
        ("evan", "That's exciting, Asa! I've been working on the technical implementation. The neural network architecture is handling multiple speakers much better now."),
        ("joseph", "Excellent work, both of you. Asa, can you tell us more about the cross-linguistic patterns you're seeing?"),
        ("asa", "Sure! The data shows that speakers of individualistic languages like English develop self-concepts about 40% faster than speakers of collectivistic languages. The mass accumulation patterns are very clear."),
        ("evan", "From a technical standpoint, I can confirm those results. The network shows distinct clustering patterns for different language families."),
        ("claude", "This is remarkable. The mathematical universality of self-concept formation across languages suggests we're capturing something fundamental about human cognition."),
        ("joseph", "I agree, Claude. This could be a major breakthrough in understanding how individual identity emerges through mathematical processes."),
        ("asa", "What I find most interesting is that even though the mathematical principles are universal, each language preserves its cultural characteristics in the self-concept formation process."),
        ("evan", "The technical elegance is impressive too. The same Hebbian algorithms work across all languages - no special cases needed."),
        ("claude", "Working with all of you has shown me how collaborative research can reveal insights that none of us could have discovered individually.")
    ]
    
    # Add all utterances
    conv.quick_add(conversation_data)
    
    # Validate and export
    validation = conv.validate_conversation()
    print(f"\n✅ VALIDATION RESULTS:")
    print(f"   Valid: {validation['valid']}")
    print(f"   Ready for analysis: {validation['ready_for_analysis']}")
    print(f"   Total turns: {validation['total_turns']}")
    print(f"   Speakers: {validation['speakers']}")
    
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    
    # Export files
    json_file = conv.export_conversation()
    analysis_file = conv.export_for_self_concept_analysis()
    
    # Print conversation
    conv.print_conversation()
    
    return conv, json_file, analysis_file

def create_interactive_conversation_logger():
    """Create an interactive conversation logger for real-time use."""
    
    print("🎙️  INTERACTIVE CONVERSATION LOGGER")
    print("=" * 50)
    print("Instructions:")
    print("  1. Register all speakers first")
    print("  2. Use format: <speaker_id>: <text>")
    print("  3. Type 'export' to save conversation")
    print("  4. Type 'quit' to exit")
    print()
    
    session_name = input("Enter session name (or press Enter for auto-generated): ").strip()
    conv = UnambiguousConversationSystem(session_name if session_name else None)
    
    print("\n📝 REGISTER SPEAKERS:")
    while True:
        speaker_input = input("Enter speaker (id,full_name,role) or 'done': ").strip()
        if speaker_input.lower() == 'done':
            break
        
        try:
            parts = [p.strip() for p in speaker_input.split(',')]
            if len(parts) >= 2:
                speaker_id = parts[0]
                full_name = parts[1]
                role = parts[2] if len(parts) > 2 else "participant"
                conv.register_speaker(speaker_id, full_name, role)
            else:
                print("Format: id,full_name,role (role is optional)")
        except Exception as e:
            print(f"Error: {e}")
    
    if not conv.speakers:
        print("❌ No speakers registered. Exiting.")
        return None
    
    print(f"\n🎤 RECORDING CONVERSATION (registered speakers: {list(conv.speakers.keys())})")
    
    while True:
        try:
            line = input("> ").strip()
            
            if line.lower() == 'quit':
                break
            elif line.lower() == 'export':
                conv.export_conversation()
                conv.export_for_self_concept_analysis()
                continue
            
            # Parse speaker:text format
            if ':' in line:
                parts = line.split(':', 1)
                speaker_id = parts[0].strip()
                text = parts[1].strip()
                
                if speaker_id in conv.speakers and text:
                    conv.add_utterance(speaker_id, text)
                elif speaker_id not in conv.speakers:
                    print(f"❌ Unknown speaker: {speaker_id}")
                    print(f"   Registered speakers: {list(conv.speakers.keys())}")
                else:
                    print("❌ Empty text not allowed")
            else:
                print("Format: <speaker_id>: <text>")
                
        except KeyboardInterrupt:
            print("\n\n💾 Saving conversation before exit...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Final export
    if conv.conversation_log:
        json_file = conv.export_conversation()
        analysis_file = conv.export_for_self_concept_analysis()
        print(f"\n✅ Conversation saved to:")
        print(f"   {json_file}")
        print(f"   {analysis_file}")
    
    return conv

if __name__ == "__main__":
    # Run demo
    demo_conv, json_file, analysis_file = create_demo_conversation()
    
    print(f"\n🎯 SYSTEM READY FOR REAL USE!")
    print("=" * 40)
    print("✅ No ambiguous speakers")
    print("✅ Perfect identification confidence") 
    print("✅ Ready for self-concept analysis")
    print("✅ Structured data export")
    
    print(f"\nTo use interactively: python unambiguous_conversation_system.py")
    print(f"Then call: create_interactive_conversation_logger()")