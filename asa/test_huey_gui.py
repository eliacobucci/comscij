#!/usr/bin/env python3
"""
Simple test script to debug Huey GUI GPU issues.
"""

import sys
import traceback

def test_imports():
    """Test all required imports."""
    print("Testing imports...")
    
    try:
        import tkinter as tk
        from tkinter import ttk
        print("✅ tkinter imports OK")
    except Exception as e:
        print(f"❌ tkinter import error: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy import OK")
    except Exception as e:
        print(f"❌ numpy import error: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        print("✅ matplotlib imports OK")
    except Exception as e:
        print(f"❌ matplotlib import error: {e}")
        return False
    
    try:
        from huey_gpu_conversational_experiment import HueyGPUConversationalNetwork
        print("✅ HueyGPUConversationalNetwork import OK")
    except Exception as e:
        print(f"❌ HueyGPUConversationalNetwork import error: {e}")
        return False
    
    try:
        from huey_gui_branding import HueyBrandingManager
        print("✅ HueyBrandingManager import OK")
    except Exception as e:
        print(f"❌ HueyBrandingManager import error: {e}")
        return False
    
    return True

def test_gui_creation():
    """Test basic GUI creation."""
    print("\nTesting GUI creation...")
    
    try:
        from huey_gui_gpu import HueyGUIGPU
        print("✅ HueyGUIGPU import OK")
        
        print("Creating HueyGUIGPU instance...")
        app = HueyGUIGPU()
        print("✅ HueyGUIGPU instance created")
        
        print("Testing configuration screen...")
        # Don't run the main loop, just test creation
        app.root.update()  # Process pending events
        print("✅ Configuration screen seems to work")
        
        app.root.destroy()
        print("✅ GUI cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI creation error: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 HUEY GUI GPU DEBUG TEST")
    print("=" * 50)
    
    if not test_imports():
        print("\n❌ Import test failed!")
        return 1
    
    if not test_gui_creation():
        print("\n❌ GUI creation test failed!")
        return 1
    
    print("\n✅ All tests passed! The GUI should work.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)