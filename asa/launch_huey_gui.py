#!/usr/bin/env python3
"""
🚀 Huey GUI GPU Launcher
=======================

Launch script for the Huey GPU Tkinter interface.
Professional launcher with error handling and setup verification.
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import tkinter
        from tkinter import ttk
    except ImportError:
        missing_deps.append("tkinter (usually comes with Python)")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import matplotlib
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import pandas  
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        from PIL import Image, ImageTk
    except ImportError:
        missing_deps.append("Pillow (PIL)")
    
    return missing_deps

def main():
    """Main launcher function."""
    
    print("🚀 HUEY GUI GPU LAUNCHER")
    print("=" * 50)
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   • {dep}")
        print("\nInstall with:")
        print("   pip install numpy matplotlib pandas pillow")
        return 1
    
    # Check Huey components
    try:
        from huey_gpu_conversational_experiment import HueyGPUConversationalNetwork
        from huey_speaker_detector import HueySpeakerDetector
        print("✅ Huey GPU components loaded")
    except ImportError as e:
        print(f"❌ Could not load Huey components: {e}")
        print("   Make sure you're in the correct directory with Huey files")
        return 1
    
    # Import and launch GUI
    try:
        print("🎨 Initializing Galileo-branded interface...")
        from huey_gui_gpu import HueyGUIGPU
        
        print("✅ All components loaded successfully!")
        print("📱 Launching Huey GUI GPU...")
        print("\n" + "=" * 50)
        print("🌟 Welcome to Huey GPU!")
        print("   Professional Hebbian Self-Concept Analysis")
        print("   with GPU Acceleration and Galileo Branding")
        print("=" * 50)
        
        # Create and run the application
        app = HueyGUIGPU()
        app.run()
        
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Error launching GUI: {e}")
        print("\nFor help, check:")
        print("   • All dependencies are installed")
        print("   • You're in the correct directory")
        print("   • Python version is 3.7+")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)