#!/usr/bin/env python3
"""
Filename: test_experiment_info.py
Author: Kristophor Jensen
Date Created: 20250916_120000
Date Revised: 20250916_120000
File version: 1.0.0.0
Description: Test experiment-info command shows file labels
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Simulate running the shell command
os.system("echo 'experiment-info 41' | ./mldp")