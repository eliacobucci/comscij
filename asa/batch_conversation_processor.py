#!/usr/bin/env python3
"""
Batch Conversation Processor
Process all conversation files in the directory and identify speakers across multiple sessions.
"""

from conversation_reprocessor import ConversationReprocessor
import glob
import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

class BatchConversationProcessor:
    """
    Process multiple conversation files and analyze speaker patterns across sessions.
    """
    
    def __init__(self):
        self.reprocessor = ConversationReprocessor()
        self.all_results = {}
        self.speaker_statistics = defaultdict(lambda: {
            'total_turns': 0,
            'total_words': 0,
            'files_appeared': [],
            'confidence_scores': [],
            'self_concept_masses': []
        })
        
        print("🔄 Batch Conversation Processor initialized")
    
    def find_conversation_files(self) -> list:
        """Find all potential conversation files."""
        
        patterns = [
            "/Users/josephwoelfel/asa/*.md",
            "/Users/josephwoelfel/asa/*.txt",
            "/Users/josephwoelfel/asa/*conversation*.py",
            "/Users/josephwoelfel/asa/*chat*.py"
        ]
        
        files = []
        for pattern in patterns:
            files.extend(glob.glob(pattern))
        
        # Filter out generated/output files
        exclude_keywords = [
            'reprocessed', 'enhanced_sample', 'multi_speaker_sample', 
            'sample_conversation', 'speaker_analysis', 'cluster_analysis'
        ]
        
        filtered_files = []
        for f in files:
            basename = os.path.basename(f).lower()
            if not any(keyword in basename for keyword in exclude_keywords):
                filtered_files.append(f)
        
        print(f"🔍 Found {len(filtered_files)} conversation files to process:")
        for f in filtered_files:
            print(f"   {os.path.basename(f)}")
        
        return filtered_files
    
    def extract_conversation_from_markdown(self, filename: str) -> str:
        """Extract conversational content from markdown files."""
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return ""
        
        # Look for conversational patterns in markdown
        conversation_lines = []
        
        for line in content.split('\n'):
            line = line.strip()
            
            # Skip empty lines and markdown headers
            if not line or line.startswith('#') or line.startswith('```'):
                continue
            
            # Look for conversational indicators
            conversational_indicators = [
                'user asked:', 'user:', 'joseph:', 'claude:', 'asa:', 'evan:',
                'hello', 'hi', 'thank you', 'i think', 'you mentioned',
                'what do you', 'can you', 'let me', 'i\'ve been', 'we should'
            ]
            
            if any(indicator in line.lower() for indicator in conversational_indicators):
                # Clean up markdown formatting
                cleaned_line = line.replace('**', '').replace('*', '').replace('- ', '')
                if len(cleaned_line) > 20:  # Skip very short lines
                    conversation_lines.append(cleaned_line)
        
        return '\n'.join(conversation_lines)
    
    def process_all_files(self) -> dict:
        """Process all conversation files."""
        
        files = self.find_conversation_files()
        
        print(f"\n🎯 PROCESSING {len(files)} FILES")
        print("=" * 60)
        
        for i, filename in enumerate(files, 1):
            print(f"\n📄 [{i}/{len(files)}] Processing: {os.path.basename(filename)}")
            
            # Extract conversation content
            if filename.endswith('.md'):
                conversation_text = self.extract_conversation_from_markdown(filename)
            else:
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        conversation_text = f.read()
                except:
                    print(f"❌ Could not read {filename}")
                    continue
            
            if len(conversation_text.strip()) < 100:
                print(f"⚠️  Skipping - insufficient conversational content")
                continue
            
            # Save extracted content for processing
            temp_filename = f"/tmp/temp_conversation_{i}.txt"
            with open(temp_filename, 'w') as f:
                f.write(conversation_text)
            
            # Process with our enhanced identifier
            try:
                results = self.reprocessor.reprocess_conversation_file(temp_filename)
                if results:
                    self.all_results[filename] = results
                    self.update_speaker_statistics(filename, results)
                    print(f"✅ Processed successfully")
                else:
                    print(f"⚠️  No results from processing")
            except Exception as e:
                print(f"❌ Error processing: {e}")
            
            # Clean up temp file
            try:
                os.remove(temp_filename)
            except:
                pass
        
        return self.all_results
    
    def update_speaker_statistics(self, filename: str, results: dict):
        """Update cumulative speaker statistics."""
        
        for turn in results['tagged_conversation']:
            speaker = turn['speaker']
            
            # Skip unknown or low-confidence speakers
            if speaker.startswith('Unknown') or turn['confidence'] < 0.4:
                continue
            
            stats = self.speaker_statistics[speaker]
            stats['total_turns'] += 1
            stats['total_words'] += len(turn['text'].split())
            stats['confidence_scores'].append(turn['confidence'])
            
            if filename not in stats['files_appeared']:
                stats['files_appeared'].append(filename)
    
    def run_comprehensive_self_concept_analysis(self) -> dict:
        """Run self-concept analysis across all processed conversations."""
        
        print(f"\n🧠 COMPREHENSIVE SELF-CONCEPT ANALYSIS")
        print("=" * 60)
        
        # Combine all conversation data
        all_conversation_blocks = []
        
        for filename, results in self.all_results.items():
            for turn in results['tagged_conversation']:
                if turn['confidence'] > 0.4:  # Only high-confidence assignments
                    all_conversation_blocks.append({
                        'speaker': turn['speaker'],
                        'text': turn['text'],
                        'confidence': turn['confidence'],
                        'source_file': os.path.basename(filename),
                        'turn_id': len(all_conversation_blocks)
                    })
        
        print(f"   Combined {len(all_conversation_blocks)} conversation blocks from {len(self.all_results)} files")
        
        # Create comprehensive conversation data
        comprehensive_data = {
            'metadata': {
                'total_turns': len(all_conversation_blocks),
                'speakers_identified': list(set(block['speaker'] for block in all_conversation_blocks)),
                'source_files': list(self.all_results.keys()),
                'confidence_threshold': 0.4
            },
            'speaker_mappings': self.reprocessor.speaker_pronouns,
            'conversation_blocks': all_conversation_blocks
        }
        
        # Save comprehensive dataset
        output_file = '/Users/josephwoelfel/asa/comprehensive_conversation_dataset.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Comprehensive dataset saved: {output_file}")
        
        # Run self-concept analysis
        analysis_results = self.reprocessor.run_self_concept_analysis_on_reprocessed(comprehensive_data)
        
        return {
            'comprehensive_data': comprehensive_data,
            'analysis_results': analysis_results,
            'speaker_statistics': dict(self.speaker_statistics)
        }
    
    def create_comprehensive_visualization(self, comprehensive_results: dict):
        """Create comprehensive multi-speaker visualization."""
        
        print(f"\n🎨 Creating comprehensive multi-speaker visualization...")
        
        # Create the main 3D cluster plot
        viz_filename = '/Users/josephwoelfel/asa/comprehensive_multi_speaker_clusters.png'
        clusters = self.reprocessor.create_multi_speaker_visualization(
            comprehensive_results['analysis_results'], 
            viz_filename
        )
        
        # Create speaker statistics summary plot
        self.create_speaker_statistics_plot(comprehensive_results['speaker_statistics'])
        
        return clusters
    
    def create_speaker_statistics_plot(self, speaker_stats: dict):
        """Create visualization of speaker statistics across all files."""
        
        # Filter to main speakers only
        main_speakers = {k: v for k, v in speaker_stats.items() 
                        if k in ['Joseph', 'Claude', 'Asa', 'Evan']}
        
        if not main_speakers:
            print("⚠️  No main speakers found for statistics plot")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        speakers = list(main_speakers.keys())
        colors = {'Joseph': 'blue', 'Claude': 'red', 'Asa': 'green', 'Evan': 'orange'}
        
        # Plot 1: Total turns per speaker
        turns = [main_speakers[s]['total_turns'] for s in speakers]
        bars1 = ax1.bar(speakers, turns, color=[colors.get(s, 'gray') for s in speakers])
        ax1.set_title('Total Conversation Turns by Speaker')
        ax1.set_ylabel('Number of Turns')
        for i, bar in enumerate(bars1):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(turns[i]), ha='center', va='bottom')
        
        # Plot 2: Total words per speaker
        words = [main_speakers[s]['total_words'] for s in speakers]
        bars2 = ax2.bar(speakers, words, color=[colors.get(s, 'gray') for s in speakers])
        ax2.set_title('Total Words by Speaker')
        ax2.set_ylabel('Number of Words')
        for i, bar in enumerate(bars2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                    str(words[i]), ha='center', va='bottom')
        
        # Plot 3: Average confidence scores
        avg_confidence = [np.mean(main_speakers[s]['confidence_scores']) 
                         if main_speakers[s]['confidence_scores'] else 0 
                         for s in speakers]
        bars3 = ax3.bar(speakers, avg_confidence, color=[colors.get(s, 'gray') for s in speakers])
        ax3.set_title('Average Speaker Identification Confidence')
        ax3.set_ylabel('Confidence Score')
        ax3.set_ylim(0, 1)
        for i, bar in enumerate(bars3):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{avg_confidence[i]:.3f}', ha='center', va='bottom')
        
        # Plot 4: Files appeared in
        files_count = [len(main_speakers[s]['files_appeared']) for s in speakers]
        bars4 = ax4.bar(speakers, files_count, color=[colors.get(s, 'gray') for s in speakers])
        ax4.set_title('Number of Files Each Speaker Appeared In')
        ax4.set_ylabel('Number of Files')
        for i, bar in enumerate(bars4):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(files_count[i]), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('/Users/josephwoelfel/asa/speaker_statistics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Speaker statistics plot saved: speaker_statistics_summary.png")
    
    def generate_comprehensive_report(self, comprehensive_results: dict):
        """Generate a comprehensive analysis report."""
        
        report = []
        report.append("# COMPREHENSIVE MULTI-SPEAKER SELF-CONCEPT ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Overview
        data = comprehensive_results['comprehensive_data']
        stats = comprehensive_results['speaker_statistics']
        analysis = comprehensive_results['analysis_results']
        
        report.append(f"## OVERVIEW")
        report.append(f"- **Total conversation blocks processed**: {data['metadata']['total_turns']}")
        report.append(f"- **Source files analyzed**: {len(data['metadata']['source_files'])}")
        report.append(f"- **Speakers identified**: {data['metadata']['speakers_identified']}")
        report.append(f"- **Analysis confidence threshold**: {data['metadata']['confidence_threshold']}")
        report.append("")
        
        # Speaker breakdown
        report.append(f"## SPEAKER ANALYSIS")
        for speaker, speaker_stats in stats.items():
            if speaker_stats['total_turns'] > 0:
                avg_conf = np.mean(speaker_stats['confidence_scores']) if speaker_stats['confidence_scores'] else 0
                report.append(f"### {speaker}")
                report.append(f"- Total turns: {speaker_stats['total_turns']}")
                report.append(f"- Total words: {speaker_stats['total_words']}")
                report.append(f"- Average confidence: {avg_conf:.3f}")
                report.append(f"- Files appeared in: {len(speaker_stats['files_appeared'])}")
                report.append("")
        
        # Self-concept analysis
        report.append(f"## SELF-CONCEPT FORMATION ANALYSIS")
        if 'speaker_analyses' in analysis:
            for speaker, speaker_analysis in analysis['speaker_analyses'].items():
                if isinstance(speaker_analysis, dict) and 'self_concept_mass' in speaker_analysis:
                    report.append(f"### {speaker}")
                    report.append(f"- Self-concept mass: {speaker_analysis['self_concept_mass']:.3f}")
                    report.append(f"- Blocks processed: {speaker_analysis.get('blocks_processed', 'N/A')}")
                    report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open('/Users/josephwoelfel/asa/comprehensive_analysis_report.md', 'w') as f:
            f.write(report_text)
        
        print(f"📄 Comprehensive report saved: comprehensive_analysis_report.md")
        
        return report_text

def main():
    """Run comprehensive batch processing."""
    
    print("🚀 COMPREHENSIVE MULTI-SPEAKER CONVERSATION ANALYSIS")
    print("=" * 70)
    print("Processing all conversation files to identify Asa, Evan, Joseph, and Claude...")
    print()
    
    # Initialize processor
    processor = BatchConversationProcessor()
    
    # Process all files
    all_results = processor.process_all_files()
    
    if not all_results:
        print("❌ No conversation files could be processed")
        return
    
    # Run comprehensive analysis
    comprehensive_results = processor.run_comprehensive_self_concept_analysis()
    
    # Create visualizations
    clusters = processor.create_comprehensive_visualization(comprehensive_results)
    
    # Generate report
    report = processor.generate_comprehensive_report(comprehensive_results)
    
    # Print summary
    print(f"\n🎯 BATCH PROCESSING COMPLETE")
    print("=" * 40)
    print(f"✅ Files processed: {len(all_results)}")
    print(f"✅ Total speakers identified: {len(comprehensive_results['speaker_statistics'])}")
    print(f"✅ Conversation blocks analyzed: {comprehensive_results['comprehensive_data']['metadata']['total_turns']}")
    
    main_speakers = [s for s in comprehensive_results['speaker_statistics'] 
                    if s in ['Joseph', 'Claude', 'Asa', 'Evan']]
    print(f"✅ Main speakers found: {main_speakers}")
    
    print(f"\n📊 KEY OUTPUTS:")
    print(f"   • comprehensive_conversation_dataset.json - Complete tagged dataset")
    print(f"   • comprehensive_multi_speaker_clusters.png - 3D cluster visualization")  
    print(f"   • speaker_statistics_summary.png - Speaker statistics plots")
    print(f"   • comprehensive_analysis_report.md - Detailed analysis report")
    
    return comprehensive_results

if __name__ == "__main__":
    results = main()