#!/usr/bin/env python3
"""
Remove unicode icons from mldp_shell.py
Replaces icons with empty string while preserving rest of line
"""

import sys

# Icons to remove (each followed by optional space)
ICONS = [
    '✅ ', '✅',
    '❌ ', '❌',
    '⚠️  ', '⚠️ ', '⚠️',
    '🔄 ', '🔄',
    '💾 ', '💾',
    '📊 ', '📊',
    '📜 ', '📜',
    '📋 ', '📋',
    '⏹️  ', '⏹️ ', '⏹️',
    '➕ ', '➕',
    '➖ ', '➖',
    '⚙️  ', '⚙️ ', '⚙️',
    '🔍 ', '🔍',
]

def remove_icons(text):
    """Remove all unicode icons from text"""
    for icon in ICONS:
        text = text.replace(icon, '')
    return text

def main():
    filename = '/Users/kjensen/Documents/GitHub/mldp/mldp_cli/src/mldp_shell.py'

    # Read file
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove icons
    new_content = remove_icons(content)

    # Count changes
    if content == new_content:
        print("No changes needed")
        return

    changes = len([i for i, (a, b) in enumerate(zip(content, new_content)) if a != b])
    print(f"Will remove {changes} characters")

    # Show first 5 changes as examples
    print("\nExample changes:")
    count = 0
    for i, (old_line, new_line) in enumerate(zip(content.split('\n'), new_content.split('\n')), 1):
        if old_line != new_line:
            print(f"Line {i}:")
            print(f"  OLD: {old_line[:100]}")
            print(f"  NEW: {new_line[:100]}")
            count += 1
            if count >= 5:
                break

    # Ask for confirmation
    response = input("\nProceed with changes? (yes/no): ")
    if response.lower() == 'yes':
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"SUCCESS: Updated {filename}")
    else:
        print("Cancelled")

if __name__ == '__main__':
    main()
