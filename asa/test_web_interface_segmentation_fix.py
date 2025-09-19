#!/usr/bin/env python3
"""
Test the web interface with the segmentation fix
"""

import tempfile
import os
from datetime import datetime

def test_web_interface_with_segmentation_fix():
    print("🧪 TESTING WEB INTERFACE WITH SEGMENTATION FIX")
    print("="*60)
    
    # Test the segmentation directly with our Chinese text
    from huey_gpu_web_interface_complete import _is_valid_segment
    
    test_text = "蓝色\n\n我记得三个梦。\n\n梦都是蓝色的。"
    is_valid = _is_valid_segment(test_text)
    print(f"📋 Text validation: '{test_text}' -> {is_valid}")
    
    if not is_valid:
        print("❌ Text still fails validation - segmentation will still filter it out")
        return
    
    print("✅ Text passes validation - proceeding with full test")
    
    # Now test the complete flow
    from huey_temporal_simple import HueyTemporalSimple
    from huey_complete_platform import HueyCompletePlatform
    from huey_gpu_web_interface_complete import process_uploaded_file
    import io
    
    # Create a fake uploaded file
    class FakeUploadedFile:
        def __init__(self, content, name):
            self.content = content
            self.name = name
        
        def getvalue(self):
            return self.content.encode('utf-8')
    
    fake_file = FakeUploadedFile(test_text, "test_chinese.txt")
    
    # Initialize Huey like the web interface
    temporal_network = HueyTemporalSimple(
        max_neurons=1000,
        use_temporal_weights=True,  
        tau=3.0,
        learning_rate=0.2,
        use_gpu_acceleration=True,
        max_connections_per_neuron=250
    )
    
    huey = HueyCompletePlatform(
        session_name=f"test_session_{int(datetime.now().timestamp())}",
        max_neurons=1000,
        window_size=10,
        learning_rate=0.15
    )
    huey.network = temporal_network
    
    print(f"🚀 Initialized Huey platform")
    
    # Test file processing
    print(f"\n🔄 Processing file with conversation_mode=False")
    result = process_uploaded_file(fake_file, huey, 2.0, 10000, conversation_mode=False)
    
    print(f"📋 Processing result: {result}")
    
    # Check final results
    concept_count = len(getattr(huey.network, 'concept_neurons', {}))
    connection_count = len(getattr(huey.network, 'connections', {}))
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Concept count: {concept_count}")
    print(f"   Connection count: {connection_count}")
    
    if concept_count > 0:
        print("🎉 SUCCESS: Concepts learned successfully!")
        print(f"   Concepts: {getattr(huey.network, 'concept_neurons', {})}")
    else:
        print("❌ FAILURE: Still no concepts learned")

if __name__ == "__main__":
    test_web_interface_with_segmentation_fix()