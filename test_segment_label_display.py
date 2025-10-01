#!/usr/bin/env python3
"""
Filename: test_segment_label_display.py
Author: Kristophor Jensen
Date Created: 20250919_120000
Date Revised: 20250919_120000
File version: 1.0.0.0
Description: Test that experiment-info shows segment label breakdown after segment selection
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from mldp_shell import MLDPShell

def main():
    """Test segment label display in experiment-info"""
    print("\n" + "="*70)
    print("TESTING SEGMENT LABEL DISPLAY IN EXPERIMENT-INFO")
    print("="*70)
    
    # Create shell instance
    shell = MLDPShell()
    
    # Connect to database
    print("\n🔗 Connecting to database...")
    shell.cmd_connect([])
    
    # Set experiment 41
    print("\n🔄 Setting experiment 41...")
    shell.cmd_set(['experiment', '41'])
    
    # Display experiment info to see segment labels
    print("\n📊 Displaying experiment info with segment labels...")
    shell.cmd_experiment_info(['41'])
    
    print("\n" + "="*70)
    print("✅ Test complete - segment label breakdown should appear above")
    print("="*70)

if __name__ == "__main__":
    main()