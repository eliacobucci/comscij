#!/usr/bin/env python3
"""
Check all Huey dependencies before starting
"""

def check_all_imports():
    """Test all imports that Huey needs"""
    
    print("🔍 Checking Huey Dependencies...")
    print("=" * 40)
    
    # Basic Python libraries
    basic_deps = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'), 
        ('numpy', 'numpy'),
        ('plotly.express', 'plotly'),
        ('dash', 'dash'),
        ('dash_bootstrap_components', 'dash-bootstrap-components'),
        ('networkx', 'networkx'),
        ('matplotlib.pyplot', 'matplotlib'),
        ('sklearn', 'scikit-learn'),
        ('nltk', 'nltk'),
        ('scipy', 'scipy'),
        ('seaborn', 'seaborn'),
        ('requests', 'requests')
    ]
    
    print("📦 Basic Dependencies:")
    basic_ok = True
    for module, package in basic_deps:
        try:
            __import__(module)
            print(f"   ✅ {package}")
        except ImportError as e:
            print(f"   ❌ {package}: {e}")
            basic_ok = False
    
    # Huey components
    huey_deps = [
        'huey_complete_platform',
        'huey_speaker_detector', 
        'huey_web_interface',
        'huey_query_engine',
        'huey_interactive_dashboard',
        'conversational_self_concept_experiment',
        'experimental_network_complete'
    ]
    
    print("\n🧠 Huey Components:")
    huey_ok = True
    for module in huey_deps:
        try:
            __import__(module)
            print(f"   ✅ {module}.py")
        except ImportError as e:
            print(f"   ❌ {module}.py: {e}")
            huey_ok = False
    
    # Specific class imports
    print("\n🔧 Class Imports:")
    class_ok = True
    try:
        from huey_complete_platform import HueyCompletePlatform
        print("   ✅ HueyCompletePlatform")
    except ImportError as e:
        print(f"   ❌ HueyCompletePlatform: {e}")
        class_ok = False
        
    try:
        from huey_speaker_detector import HueySpeakerDetector  
        print("   ✅ HueySpeakerDetector")
    except ImportError as e:
        print(f"   ❌ HueySpeakerDetector: {e}")
        class_ok = False
    
    # Overall status
    print("\n" + "=" * 40)
    if basic_ok and huey_ok and class_ok:
        print("✅ ALL DEPENDENCIES OK - HUEY CAN START")
        return True
    else:
        print("❌ MISSING DEPENDENCIES - HUEY WILL FAIL")
        print("\nTo fix:")
        if not basic_ok:
            print("  pip install streamlit pandas numpy plotly matplotlib scikit-learn nltk scipy")
        if not huey_ok or not class_ok:
            print("  Check that all Huey .py files are in the same directory")
        return False

if __name__ == "__main__":
    check_all_imports()