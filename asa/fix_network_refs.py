#!/usr/bin/env python3
"""
Fix all huey.network references to use get_network(huey) compatibility function
"""

import re

def fix_network_references(content):
    """Fix all huey.network.xyz references to use get_network pattern"""
    
    # Pattern to match huey.network.attribute usage
    pattern = r'huey\.network\.(\w+)'
    
    def replace_func(match):
        attr = match.group(1)
        return f'getattr(get_network(huey), \'{attr}\', {{}})'
    
    # Replace the pattern
    fixed_content = re.sub(pattern, replace_func, content)
    
    # Fix specific patterns that need special handling
    # Fix hasattr calls
    fixed_content = re.sub(
        r'hasattr\(getattr\(get_network\(huey\), \'(\w+)\', \{\}\), \'(\w+)\'\)',
        r'hasattr(get_network(huey), \'\1\')',
        fixed_content
    )
    
    # Fix len() calls on attributes
    fixed_content = re.sub(
        r'len\(getattr\(get_network\(huey\), \'(\w+)\', \{\}\)\)',
        r'len(getattr(get_network(huey), \'\1\', {}))',
        fixed_content
    )
    
    return fixed_content

# Read the file
with open('huey_web_interface.py', 'r') as f:
    content = f.read()

# Fix the references
fixed_content = fix_network_references(content)

# Write back
with open('huey_web_interface.py', 'w') as f:
    f.write(fixed_content)

print("Fixed all huey.network references!")