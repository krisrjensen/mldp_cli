#!/usr/bin/env python3
"""
Filename: test_exp18_segment_labels.py
Author: Kristophor Jensen
Date Created: 20250919_120000
Date Revised: 20250919_120000
File version: 1.0.0.0
Description: Test segment label display on experiment 18 which has segments
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mldp_shell import MLDPShell

def main():
    """Test segment label display for experiment 18"""
    print("\n" + "="*70)
    print("TESTING SEGMENT LABEL DISPLAY - EXPERIMENT 18")
    print("="*70)
    
    # Create shell instance
    shell = MLDPShell()
    
    print("\n📊 Displaying experiment 18 info (without connecting)...")
    shell.cmd_experiment_info(['18'])
    
    print("\n" + "="*70)
    print("✅ Test complete - checking if segment labels appear")
    print("="*70)

if __name__ == "__main__":
    main()