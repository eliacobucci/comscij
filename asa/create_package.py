#!/usr/bin/env python3
"""
Create Huey Alpha Testing Package for Remote Distribution
========================================================

This script bundles all necessary files for remote Huey Alpha testing.
"""

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

def create_huey_package():
    """Create complete Huey Alpha testing package"""
    
    print("🧠 Creating Huey Alpha Testing Package")
    print("=" * 50)
    
    # Package name
    package_name = "huey-alpha-package"
    
    # Clean up existing package directory
    if os.path.exists(package_name):
        print(f"Removing existing {package_name}...")
        shutil.rmtree(package_name)
    
    # Create package directory structure
    print("Creating directory structure...")
    os.makedirs(package_name, exist_ok=True)
    os.makedirs(f"{package_name}/test_data", exist_ok=True)
    os.makedirs(f"{package_name}/troubleshooting", exist_ok=True)
    
    # Files to include in package
    core_files = [
        # Core application files
        "huey_complete_platform.py",
        "huey_speaker_detector.py",
        "huey_web_interface.py", 
        "huey_query_engine.py",
        "huey_interactive_dashboard.py",
        "conversational_self_concept_experiment.py",
        "experimental_network_complete.py",
        "complete_conversation_analysis_system.py",
        "unambiguous_conversation_system.py",
        "conversation_reprocessor.py",
        "enhanced_speaker_identifier.py",
        "visualize_self_concept_clusters.py",
        
        # Installation files
        "install.py",
        "requirements.txt",
        "KILL.txt",
        "check_huey_deps.py",
        "START_HUEY.command",
        "WINDOWS_START_HUEY.bat",
        "SIMPLE_INSTALLATION_GUIDE.md",
        
        # Documentation
        "HUEY_ALPHA_INSTALLATION_GUIDE.md",
        "HUEY_ALPHA_TESTER_GUIDE.md",
        "alpha_testing_checklist.md",
        "HUEY_ALPHA_PACKAGE_README.md",
        
        # New features and documentation
        "quey.html",
        "QUEY_README.md", 
        "QUEY_QUICK_START.txt",
        "DIRECTIONAL_SEMANTIC_CONNECTIONS_THEORY.md",
        "DIRECTIONAL_SEMANTICS_DOCUMENTATION.md",
        "ASA_PRESENTATION_METHODS_DIRECTIONAL_SEMANTICS.md",
        "GALILEO_COMPANY_DIRECTIONAL_FRAMEWORK.md"
    ]
    
    # Test data files
    test_files = [
        "test_data/chat_transcript_fixed.txt",
        "test_data/README_test_data.md"
    ]
    
    # Troubleshooting files  
    trouble_files = [
        "troubleshooting/common_issues.md",
        "troubleshooting/system_requirements.md"
    ]
    
    # Copy core files
    print("Copying core application files...")
    for file in core_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{package_name}/{file}")
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ Missing: {file}")
    
    # Copy test files
    print("Copying test data...")
    for file in test_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{package_name}/{file}")
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ Missing: {file}")
    
    # Copy troubleshooting files
    print("Copying troubleshooting documentation...")
    for file in trouble_files:
        if os.path.exists(file):
            shutil.copy2(file, f"{package_name}/{file}")
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ Missing: {file}")
    
    # Create package info file
    print("Creating package manifest...")
    manifest_content = f"""# Huey Alpha Package Manifest
Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Version: Alpha 0.1
Target: Galileo Company Internal Testing

## Contents:
Core Files: {len(core_files)} files
Test Data: {len(test_files)} files  
Documentation: 6 files
Troubleshooting: {len(trouble_files)} files

## Installation:
1. Extract package
2. Run: python install.py
3. Follow installation guide

## Testing:
- Reference HUEY_ALPHA_TESTER_GUIDE.md
- Use alpha_testing_checklist.md
- Report weekly to development team

Confidential - Galileo Company Only
"""
    
    with open(f"{package_name}/PACKAGE_MANIFEST.txt", 'w') as f:
        f.write(manifest_content)
    
    # Get package size info
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(package_name):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size
            file_count += 1
    
    print(f"\nPackage created successfully!")
    print(f"📁 Directory: {package_name}")
    print(f"📊 Total files: {file_count}")
    print(f"💾 Total size: {total_size / 1024 / 1024:.1f} MB")
    
    # Create ZIP archive
    zip_name = f"{package_name}.zip"
    print(f"\nCreating ZIP archive: {zip_name}")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_name):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, '.')
                zipf.write(file_path, arc_path)
                print(f"  📦 {arc_path}")
    
    zip_size = os.path.getsize(zip_name)
    print(f"\n✅ ZIP package created: {zip_name}")
    print(f"💾 ZIP size: {zip_size / 1024 / 1024:.1f} MB")
    
    # Final summary
    print("\n" + "=" * 50)
    print("🎉 HUEY ALPHA PACKAGE READY FOR DISTRIBUTION")
    print("=" * 50)
    print(f"📦 Package: {zip_name}")
    print(f"📁 Directory: {package_name}")
    print(f"🎯 Target: Galileo Company team members")
    print(f"⚠️  Confidential: Internal testing only")
    print("\nNext steps:")
    print("1. Distribute ZIP file to remote team members")
    print("2. Ensure they have installation prerequisites")
    print("3. Monitor weekly testing reports")
    print("4. Collect feedback for beta release")
    
    return package_name, zip_name

if __name__ == "__main__":
    try:
        package_dir, zip_file = create_huey_package()
        print(f"\n🚀 Ready to send {zip_file} to Galileo team!")
    except Exception as e:
        print(f"❌ Error creating package: {e}")
        import traceback
        traceback.print_exc()