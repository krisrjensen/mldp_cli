#!/usr/bin/env python3
"""
Filename: mldp
Author(s): Kristophor Jensen
Date Created: 20250901_240000
Date Revised: 20251116_000000
File version: 2.1.0.17
Description: Main launcher for MLDP CLI - supports both command and shell modes

Changelog:
v2.1.0.17 - Added classifier command group with generate-verification-features
  - Integrated mpcctl_verification_feature_matrix.py with FeatureFunctionLoader
  - Command: mldp classifier generate-verification-features
  - Database-driven feature extraction for classifier verification
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def main():
    """Main entry point that routes to appropriate interface"""
    
    # If no arguments provided, launch interactive shell
    if len(sys.argv) == 1:
        try:
            # Try advanced shell with prompt_toolkit first
            from mldp_shell import main as shell_main
            shell_main()
        except ImportError:
            # Fall back to basic shell if prompt_toolkit not available
            print("Note: Install prompt-toolkit for enhanced shell experience")
            print("  pip install prompt-toolkit")
            print()
            from interactive_cli import main as basic_shell_main
            basic_shell_main()
    
    # Check for shell mode explicitly
    elif len(sys.argv) == 2 and sys.argv[1] in ['shell', 'interactive', 'repl']:
        try:
            from mldp_shell import main as shell_main
            shell_main()
        except ImportError:
            from interactive_cli import main as basic_shell_main
            basic_shell_main()
    
    # Check for basic shell mode
    elif len(sys.argv) == 2 and sys.argv[1] == 'basic-shell':
        from interactive_cli import main as basic_shell_main
        basic_shell_main()
    
    # Otherwise use Click CLI for command mode
    else:
        from cli import cli
        cli()


if __name__ == '__main__':
    main()