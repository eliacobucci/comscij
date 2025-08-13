#!/usr/bin/env python3
"""
Find real conversations containing Asa and Evan references
Search through all files more carefully for actual multi-speaker conversations.
"""

import os
import re
import glob
from typing import List, Dict

def search_for_speakers_in_file(filename: str) -> Dict:
    """Search for speaker references in a file."""
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return {}
    
    # Look for explicit speaker patterns
    patterns = {
        'asa': [
            r'\basa\s+here\b',
            r'\basa:\b',
            r'\basa\s+said\b',
            r'\basa\s+mentioned\b',
            r'\bas\s+asa\b',
            r'\basa\s+thinks?\b',
            r'\basa\b.*?\bi\s+',
            r'\bi\s+.*?\basa\b'
        ],
        'evan': [
            r'\bevan\s+here\b',
            r'\bevan:\b', 
            r'\bevan\s+said\b',
            r'\bevan\s+mentioned\b',
            r'\bas\s+evan\b',
            r'\bevan\s+thinks?\b',
            r'\bevan\b.*?\bi\s+',
            r'\bi\s+.*?\bevan\b'
        ],
        'joseph': [
            r'\bjoseph\s+here\b',
            r'\bjoseph:\b',
            r'\bdr\.?\s+woelfel\b',
            r'\bprofessor\b',
            r'\bjoseph\s+said\b'
        ],
        'claude': [
            r'\bclaude\s+here\b',
            r'\bclaude:\b',
            r'\bi\'m\s+claude\b',
            r'\bclaude\s+said\b'
        ]
    }
    
    results = {
        'filename': filename,
        'speakers_found': [],
        'matches': {},
        'potential_conversation': False
    }
    
    content_lower = content.lower()
    
    for speaker, speaker_patterns in patterns.items():
        matches = []
        for pattern in speaker_patterns:
            found = re.findall(pattern, content_lower, re.IGNORECASE)
            if found:
                matches.extend(found)
        
        if matches:
            results['speakers_found'].append(speaker)
            results['matches'][speaker] = matches
    
    # Check if this looks like an actual conversation
    conversation_indicators = [
        r'\b(?:hello|hi|hey)\b.*?(?:claude|joseph|asa|evan)',
        r'(?:claude|joseph|asa|evan).*?(?:you|your)',
        r'(?:i|we).*?(?:think|believe|feel).*?(?:you|your)',
        r'(?:thank|thanks).*?(?:you|joseph|claude|asa|evan)',
        r'(?:can|could|would)\s+you',
        r'(?:what|how|why|when)\s+do\s+you'
    ]
    
    for pattern in conversation_indicators:
        if re.search(pattern, content_lower):
            results['potential_conversation'] = True
            break
    
    return results

def find_all_speaker_references():
    """Find all files with speaker references."""
    
    print("🔍 SEARCHING FOR ASA AND EVAN CONVERSATIONS")
    print("=" * 60)
    
    # Search patterns
    file_patterns = [
        "/Users/josephwoelfel/asa/*.md",
        "/Users/josephwoelfel/asa/*.txt", 
        "/Users/josephwoelfel/asa/*.py"
    ]
    
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))
    
    # Exclude generated files
    exclude_keywords = [
        'sample_conversation', 'enhanced_sample', 'multi_speaker_sample',
        'reprocessed', 'speaker_analysis', 'cluster_analysis',
        'comprehensive_'
    ]
    
    filtered_files = []
    for f in all_files:
        basename = os.path.basename(f).lower()
        if not any(keyword in basename for keyword in exclude_keywords):
            filtered_files.append(f)
    
    print(f"Searching {len(filtered_files)} files...")
    
    # Search each file
    results = []
    for filename in filtered_files:
        result = search_for_speakers_in_file(filename)
        if result['speakers_found'] or result['potential_conversation']:
            results.append(result)
    
    return results

def analyze_speaker_patterns(results: List[Dict]):
    """Analyze patterns in speaker findings."""
    
    print(f"\n📊 SPEAKER REFERENCE ANALYSIS")
    print("=" * 40)
    
    # Count by speaker
    speaker_counts = {'asa': 0, 'evan': 0, 'joseph': 0, 'claude': 0}
    multi_speaker_files = []
    conversation_files = []
    
    for result in results:
        speakers = result['speakers_found']
        if len(speakers) > 1:
            multi_speaker_files.append(result)
        
        if result['potential_conversation']:
            conversation_files.append(result)
        
        for speaker in speakers:
            speaker_counts[speaker] += 1
    
    print(f"Speaker appearances:")
    for speaker, count in speaker_counts.items():
        print(f"   {speaker.title()}: {count} files")
    
    print(f"\nMulti-speaker files: {len(multi_speaker_files)}")
    print(f"Potential conversations: {len(conversation_files)}")
    
    # Show detailed results
    print(f"\n📄 DETAILED FINDINGS:")
    for result in results:
        if result['speakers_found']:
            filename = os.path.basename(result['filename'])
            speakers = ', '.join(result['speakers_found'])
            conv_flag = "💬" if result['potential_conversation'] else "📝"
            print(f"   {conv_flag} {filename}: {speakers}")
            
            # Show sample matches
            for speaker, matches in result['matches'].items():
                if matches and speaker in ['asa', 'evan']:  # Focus on Asa and Evan
                    print(f"      {speaker}: {matches[:3]}")  # Show first 3 matches
    
    return multi_speaker_files, conversation_files

def extract_asa_evan_content(results: List[Dict]):
    """Extract content specifically mentioning Asa and Evan for analysis."""
    
    print(f"\n🎯 EXTRACTING ASA/EVAN CONTENT")
    print("=" * 40)
    
    asa_evan_content = []
    
    for result in results:
        if 'asa' in result['speakers_found'] or 'evan' in result['speakers_found']:
            filename = result['filename']
            print(f"\n📄 Processing: {os.path.basename(filename)}")
            
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract sentences/paragraphs mentioning Asa or Evan
                lines = content.split('\n')
                relevant_lines = []
                
                for line in lines:
                    line = line.strip()
                    if (('asa' in line.lower() or 'evan' in line.lower()) and 
                        len(line) > 20 and
                        not line.startswith('#')):  # Skip markdown headers
                        relevant_lines.append(line)
                
                if relevant_lines:
                    asa_evan_content.append({
                        'filename': filename,
                        'lines': relevant_lines[:10]  # First 10 relevant lines
                    })
                    
                    print(f"   Found {len(relevant_lines)} relevant lines:")
                    for line in relevant_lines[:3]:  # Show first 3
                        print(f"      {line[:100]}...")
            
            except Exception as e:
                print(f"   ❌ Error reading file: {e}")
    
    return asa_evan_content

def create_asa_evan_conversation_file(content_data: List[Dict]):
    """Create a conversation file focused on Asa and Evan."""
    
    if not content_data:
        print("⚠️  No Asa/Evan content found to create conversation file")
        return None
    
    print(f"\n✍️  CREATING ASA/EVAN CONVERSATION FILE")
    print("=" * 40)
    
    conversation_lines = []
    conversation_lines.append("# Asa and Evan Conversation Reconstruction")
    conversation_lines.append("# Extracted from various source files")
    conversation_lines.append("")
    
    for item in content_data:
        filename = os.path.basename(item['filename'])
        conversation_lines.append(f"## From: {filename}")
        conversation_lines.append("")
        
        for line in item['lines']:
            # Clean up the line
            cleaned = line.replace('**', '').replace('*', '').replace('- ', '')
            if cleaned and len(cleaned) > 10:
                conversation_lines.append(cleaned)
        
        conversation_lines.append("")
    
    # Save to file
    output_filename = '/Users/josephwoelfel/asa/asa_evan_extracted_conversations.txt'
    conversation_text = '\n'.join(conversation_lines)
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(conversation_text)
    
    print(f"✅ Asa/Evan conversation file created: {output_filename}")
    print(f"   Lines extracted: {len([l for l in conversation_lines if l and not l.startswith('#')])}")
    
    return output_filename

def main():
    """Run the Asa/Evan conversation finder."""
    
    # Find all speaker references
    results = find_all_speaker_references()
    
    if not results:
        print("❌ No speaker references found in any files")
        return
    
    # Analyze patterns
    multi_speaker, conversations = analyze_speaker_patterns(results)
    
    # Extract Asa/Evan specific content
    asa_evan_content = extract_asa_evan_content(results)
    
    # Create conversation file if we found content
    conversation_file = create_asa_evan_conversation_file(asa_evan_content)
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Files with speaker references: {len(results)}")
    print(f"   Multi-speaker files: {len(multi_speaker)}")
    print(f"   Potential conversation files: {len(conversations)}")
    print(f"   Asa/Evan content sources: {len(asa_evan_content)}")
    
    if conversation_file:
        print(f"   ✅ Created: {os.path.basename(conversation_file)}")
    
    return results, asa_evan_content, conversation_file

if __name__ == "__main__":
    results, content, conv_file = main()