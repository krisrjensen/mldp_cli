#!/usr/bin/env python3
"""
Test script for feature subset dimensionality reduction
Processes only FIRST 10 combinations on FIRST file to validate implementation
"""

import sys
import subprocess
from pathlib import Path

# Modify the main script to process only first 10 combinations
test_script = Path(__file__).parent / "run_feature_subset_dimreduction.py"

# Read the original script
with open(test_script, 'r') as f:
    script_content = f.read()

# Create a test version that only processes first 10 combinations
test_content = script_content.replace(
    "for file_idx, matrix_file in enumerate(matrix_files, 1):",
    "for file_idx, matrix_file in enumerate(matrix_files[:1], 1):  # TEST: Only first file"
)

test_content = test_content.replace(
    "feature_combinations = list(combinations(scalar_features, NUM_FEATURES_PER_SUBSET))",
    "feature_combinations = list(combinations(scalar_features, NUM_FEATURES_PER_SUBSET))[:10]  # TEST: Only first 10 combinations"
)

test_content = test_content.replace(
    "NUM_WORKERS = 8",
    "NUM_WORKERS = 4  # TEST: Fewer workers for testing"
)

# Write test version
test_file = Path("/tmp/test_feature_subset_dimreduction.py")
with open(test_file, 'w') as f:
    f.write(test_content)

print("=" * 80)
print("TESTING FEATURE SUBSET DIMENSIONALITY REDUCTION")
print("=" * 80)
print("Processing:")
print("  - First 10 feature combinations only")
print("  - First verification feature file only")
print("  - 4 workers")
print()
print("This will generate approximately 120 plots (10 combos × 4 methods × 3 outputs)")
print()

# Run the test
subprocess.run(["python3", str(test_file)])
