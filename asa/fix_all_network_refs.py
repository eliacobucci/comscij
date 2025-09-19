#!/usr/bin/env python3
"""
Comprehensive fix for all huey.network references in the web interface
"""

import re

def fix_all_network_references(content):
    """Fix all variations of huey.network access patterns"""
    
    # First, add network variable at the start of functions that use huey.network
    function_pattern = r'(def \w+\([^)]*huey[^)]*\):(?:\s*"""[^"]*""")?\s*)'
    
    def add_network_var(match):
        func_start = match.group(1)
        return func_start + "\n    network = get_network(huey)\n"
    
    # Don't add to every function, just the key ones
    
    # Fix specific patterns
    fixes = [
        # Basic huey.network.attribute access
        (r'huey\.network\.(\w+)', r'getattr(network, \'\1\', {})'),
        
        # hasattr patterns
        (r'hasattr\(huey\.network, [\'"](\w+)[\'"]\)', r'hasattr(network, \'\1\')'),
        
        # len() patterns
        (r'len\(huey\.network\.(\w+)\)', r'len(getattr(network, \'\1\', {}))'),
        
        # Direct attribute access in conditions
        (r'huey\.network\.(\w+) and', r'getattr(network, \'\1\', {}) and'),
        (r'if huey\.network\.(\w+):', r'if getattr(network, \'\1\', {}):'),
        
        # Iteration patterns
        (r'for (\w+) in huey\.network\.(\w+):', r'for \1 in getattr(network, \'\2\', {}):'),
        
        # Method calls
        (r'huey\.network\.(\w+)\(', r'getattr(network, \'\1\', lambda: None)('),
    ]
    
    # Apply fixes
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)
    
    # Add network variable to functions that need it
    functions_needing_network = [
        'create_3d_concept_plot',
        'process_uploaded_file',
        'main'
    ]
    
    for func_name in functions_needing_network:
        # Add network variable after function definition
        func_pattern = f'(def {func_name}\\([^)]*\\):(?:\\s*"""[^"]*""")?)\\s*'
        replacement = f'\\1\n    network = get_network(huey) if "huey" in locals() else None\n'
        content = re.sub(func_pattern, replacement, content)
    
    return content

# Read the file
with open('huey_web_interface.py', 'r') as f:
    content = f.read()

# Apply comprehensive fixes
fixed_content = fix_all_network_references(content)

# Write back
with open('huey_web_interface.py', 'w') as f:
    f.write(fixed_content)

print("Applied comprehensive network reference fixes!")