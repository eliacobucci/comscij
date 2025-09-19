#!/usr/bin/env python3
"""
Fix f-string syntax errors caused by backslashes in expressions
"""

import re

def fix_fstring_backslashes(content):
    """Fix f-strings that have backslashes in expressions"""
    
    # Find f-strings with backslashes in expressions
    pattern = r'f"([^"]*\{[^}]*\\[^}]*\}[^"]*)"'
    
    def replace_func(match):
        fstring = match.group(1)
        # Replace \' with just '
        fixed = fstring.replace("\\'", "'")
        return f'f"{fixed}"'
    
    return re.sub(pattern, replace_func, content)

# Read the file
with open('huey_web_interface.py', 'r') as f:
    content = f.read()

# Fix the f-strings
fixed_content = fix_fstring_backslashes(content)

# Write back
with open('huey_web_interface.py', 'w') as f:
    f.write(fixed_content)

print("Fixed f-string syntax errors!")