#!/usr/bin/env python3
"""
🧠 Huey Alpha - Automated Installer
==================================

Automated installation script for Huey Alpha testing package.
Creates virtual environment, installs dependencies, and launches Huey.

Usage: python install.py
"""

import os
import sys
import subprocess
import platform
import webbrowser
import time
from pathlib import Path

def print_banner():
    """Print installation banner"""
    print("=" * 60)
    print("🧠 Huey Alpha - Automated Installer")
    print("   Galileo Company - Internal Testing")
    print("=" * 60)
    print()

def check_python():
    """Check Python version compatibility"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   Huey requires Python 3.8 or higher")
        print("   Please install Python 3.8+ from https://python.org")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_required_files():
    """Check if required files are present"""
    print("🔍 Checking required files...")
    
    required_files = [
        "requirements.txt",
        "huey_complete_platform.py",
        "huey_speaker_detector.py", 
        "huey_web_interface.py",
        "conversational_self_concept_experiment.py",
        "experimental_network_complete.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"   ✅ {file}")
    
    if missing_files:
        print(f"❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("   Please ensure all files are extracted from the package.")
        return False
    
    print("✅ All required files present")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    print("\n🔨 Creating virtual environment...")
    
    venv_name = "huey_env"
    
    try:
        if os.path.exists(venv_name):
            print(f"   Virtual environment '{venv_name}' already exists")
            return venv_name
            
        subprocess.run([sys.executable, "-m", "venv", venv_name], 
                      check=True, capture_output=True)
        print(f"✅ Virtual environment '{venv_name}' created")
        return venv_name
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return None

def get_activation_command(venv_name):
    """Get activation command for current platform"""
    system = platform.system().lower()
    
    if system == "windows":
        return os.path.join(venv_name, "Scripts", "python.exe")
    else:
        return os.path.join(venv_name, "bin", "python")

def install_dependencies(python_path):
    """Install Python dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        print("   Upgrading pip...")
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements
        print("   Installing Huey dependencies...")
        result = subprocess.run([python_path, "-m", "pip", "install", "-r", "requirements.txt"], 
                               check=True, capture_output=True, text=True)
        
        print("✅ Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False

def test_installation(python_path):
    """Test that key modules can be imported"""
    print("\n🧪 Testing installation...")
    
    test_imports = [
        "streamlit",
        "pandas", 
        "numpy",
        "plotly",
        "matplotlib",
        "sklearn",
        "nltk",
        "scipy"
    ]
    
    for module in test_imports:
        try:
            result = subprocess.run([python_path, "-c", f"import {module}; print('OK')"], 
                                   check=True, capture_output=True, text=True)
            print(f"   ✅ {module}")
        except subprocess.CalledProcessError:
            print(f"   ❌ {module} - Import failed")
            return False
    
    print("✅ All modules imported successfully")
    return True

def download_nltk_data(python_path):
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    
    try:
        nltk_script = '''
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
print("NLTK data downloaded")
'''
        
        result = subprocess.run([python_path, "-c", nltk_script], 
                               check=True, capture_output=True, text=True)
        print("✅ NLTK data downloaded")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  NLTK data download failed (may work anyway): {e}")
        return True  # Continue anyway

def create_launch_script(python_path):
    """Create convenient launch script"""
    print("\n📝 Creating launch script...")
    
    system = platform.system().lower()
    
    if system == "windows":
        script_name = "launch_huey.bat"
        script_content = f'''@echo off
echo Starting Huey...
"{python_path}" -m streamlit run huey_web_interface.py --server.port=8501
pause
'''
    else:
        script_name = "launch_huey.sh"
        script_content = f'''#!/bin/bash
echo "Starting Huey..."
"{python_path}" -m streamlit run huey_web_interface.py --server.port=8501
'''
    
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    if system != "windows":
        os.chmod(script_name, 0o755)
    
    print(f"✅ Launch script created: {script_name}")
    return script_name

def launch_huey(python_path):
    """Launch Huey web interface"""
    print("\n🚀 Launching Huey...")
    
    try:
        # Start Streamlit in background
        process = subprocess.Popen([
            python_path, "-m", "streamlit", "run", 
            "huey_web_interface.py", 
            "--server.port=8501",
            "--server.address=localhost"
        ])
        
        print("   Huey is starting...")
        print("   Waiting for web server...")
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        url = "http://localhost:8501"
        print(f"   Opening browser to: {url}")
        webbrowser.open(url)
        
        print("\n✅ Huey is running!")
        print("=" * 60)
        print("🎉 INSTALLATION COMPLETE!")
        print()
        print("Huey is now running in your browser.")
        print("If browser didn't open, go to: http://localhost:8501")
        print()
        print("To stop Huey: Press Ctrl+C in this terminal")
        print("To restart later: Run launch_huey script")
        print("=" * 60)
        
        # Keep the process running
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Shutting down Huey...")
            process.terminate()
            print("✅ Huey stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to launch Huey: {e}")
        return False

def main():
    """Main installation process"""
    print_banner()
    
    # Step 1: Check Python version
    if not check_python():
        input("Press Enter to exit...")
        return False
    
    # Step 2: Check required files
    if not check_required_files():
        input("Press Enter to exit...")
        return False
    
    # Step 3: Create virtual environment
    venv_name = create_virtual_environment()
    if not venv_name:
        input("Press Enter to exit...")
        return False
    
    # Step 4: Get Python path for virtual environment
    python_path = get_activation_command(venv_name)
    
    # Step 5: Install dependencies
    if not install_dependencies(python_path):
        input("Press Enter to exit...")
        return False
    
    # Step 6: Test installation
    if not test_installation(python_path):
        input("Press Enter to exit...")
        return False
    
    # Step 7: Download NLTK data
    download_nltk_data(python_path)
    
    # Step 8: Create launch script
    create_launch_script(python_path)
    
    # Step 9: Launch Huey
    print("\n" + "=" * 60)
    print("🎯 Installation successful!")
    print("=" * 60)
    
    launch_now = input("\nLaunch Huey now? (y/n): ").lower().strip()
    if launch_now in ['y', 'yes', '']:
        launch_huey(python_path)
    else:
        print("\n✅ Installation complete!")
        print("To launch Huey later, run the launch script or:")
        print(f"   {python_path} -m streamlit run huey_web_interface.py")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please contact support with this error message.")
    finally:
        input("\nPress Enter to exit...")