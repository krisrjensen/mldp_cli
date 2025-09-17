#!/usr/bin/env python3
"""
Filename: test_experiment_info_fixed.py
Author: Kristophor Jensen
Date Created: 20250916_130000
Date Revised: 20250916_130000
File version: 1.0.0.0
Description: Test that experiment-info shows file labels without needing connect first
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mldp_shell import MLDPShell

def main():
    """Test experiment info without connecting first"""
    print("\n" + "="*70)
    print("TESTING EXPERIMENT-INFO WITHOUT CONNECT")
    print("="*70)
    
    # Create shell instance
    shell = MLDPShell()
    
    # Note: NOT calling connect first
    print("\n🔍 Testing experiment-info 41 (without connecting first)...\n")
    
    # Call experiment-info
    shell.cmd_experiment_info(['41'])
    
    print("\n" + "="*70)
    print("✅ Test complete - file labels should appear above")
    print("="*70)

if __name__ == "__main__":
    main()