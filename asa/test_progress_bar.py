#!/usr/bin/env python3
"""
Test the restored progress bar functionality
"""

import tempfile
import os
from datetime import datetime
from huey_temporal_simple import HueyTemporalSimple

def test_progress_callback():
    """Test the progress callback mechanism"""
    print("🧪 TESTING PROGRESS BAR RESTORATION")
    print("="*60)
    
    # Initialize Huey temporal network
    huey = HueyTemporalSimple(max_neurons=1000, use_temporal_weights=True)
    
    # Create test file with multiple segments
    test_content = """蓝色

我记得三个梦。第一个梦是关于飞行的。

第二个梦是关于水的。我在水中游泳。

梦都是蓝色的。蓝色的天空，蓝色的海洋。"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(test_content)
        tmp_file_path = tmp_file.name
    
    try:
        # Test progress callback
        progress_updates = []
        
        def test_progress_callback(current, total, exchange_details):
            progress_info = {
                'current': current,
                'total': total,
                'progress_pct': int((current / total) * 100),
                'speaker': exchange_details.get('speaker'),
                'text_preview': exchange_details.get('text_preview', ''),
                'text_length': exchange_details.get('text_length', 0)
            }
            progress_updates.append(progress_info)
            print(f"📊 Progress: {current}/{total} ({progress_info['progress_pct']}%) | "
                  f"Speaker: {progress_info['speaker']} | "
                  f"Text: {progress_info['text_length']} chars | "
                  f"Preview: {progress_info['text_preview']}")
        
        print(f"\n🚀 Processing with progress callback...")
        result = huey.process_file_with_mode(
            tmp_file_path, 
            conversation_mode=False,
            progress_callback=test_progress_callback
        )
        
        print(f"\n✅ PROGRESS BAR TEST RESULTS:")
        print(f"   Processing result: {result.get('success', False)}")
        print(f"   Exchanges processed: {result.get('exchanges_processed', 0)}")
        print(f"   Progress updates received: {len(progress_updates)}")
        
        if progress_updates:
            print(f"   First update: {progress_updates[0]}")
            print(f"   Last update: {progress_updates[-1]}")
            print(f"   Progress tracking: ✅ WORKING!")
        else:
            print(f"   ❌ No progress updates received")
        
        # Check final results
        concept_count = len(getattr(huey, 'concept_neurons', {}))
        connection_count = len(getattr(huey, 'connections', {}))
        print(f"   Final network: {concept_count} concepts, {connection_count} connections")
        
    finally:
        os.unlink(tmp_file_path)

if __name__ == "__main__":
    test_progress_callback()