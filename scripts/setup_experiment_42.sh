#!/bin/bash
# Filename: setup_experiment_42.sh
# Author(s): Kristophor Jensen
# Date Created: 20251104_000000
# Date Revised: 20251104_000000
# File version: 1.0.0.1
# Description: Complete setup script for experiment 42
#
# This script configures experiment 42 in the MLDP CLI:
# - Sets current experiment to 42
# - Configures segment selection settings
# - Selects 150 training files (50 per label)
# - Selects segments with position balancing
# - Adds 4 data types (adc6, adc8, adc10, adc12)
# - Adds 6 decimations (0, 7, 15, 31, 64, 128)
# - Adds 2 distance metrics (L1, Cosine)
#
# Usage (in MLDP CLI):
#   source mldp_cli/scripts/setup_experiment_42.sh
#
# Prerequisites:
# - Experiment 42 must exist in database
# - 100 feature sets must be linked to experiment 42
# - Z-score amplitude method must be configured

echo "============================================================================"
echo "Experiment 42 Setup Script"
echo "============================================================================"
echo ""

# Step 1: Set current experiment
echo "# Step 1: Setting current experiment to 42"
set experiment 42
echo ""

# Step 2: Configure experiment settings
echo "# Step 2: Configuring experiment settings"
update-selection-config --seed 42
update-selection-config --balanced true
update-selection-config --strategy position_balanced_per_file
update-selection-config --max-files 50
echo ""

# Step 3: Select training files (150 files: 50 per label)
echo "# Step 3: Selecting training files (150 total: 50 per label)"
select-files arc --count 50 --strategy random --seed 42
select-files negative_transient --count 50 --strategy random --seed 42
select-files parallel_motor_arc --count 50 --strategy random --seed 42
echo ""

# Step 4: Select segments with position balancing
echo "# Step 4: Selecting segments (position-balanced)"
select-segments --balanced --position-balance at_least_one
echo ""

# Step 5: Add data types
echo "# Step 5: Adding data types"
add-data-type 6
add-data-type 2
add-data-type 3
add-data-type 4
echo ""

# Step 6: Add decimations
echo "# Step 6: Adding decimations"
add-decimation 0
add-decimation 7
add-decimation 15
add-decimation 31
add-decimation 64
add-decimation 128
echo ""

# Step 7: Add distance metrics
echo "# Step 7: Adding distance metrics"
add-distance-metric --metric manhattan
add-distance-metric --metric cosine
echo ""

echo "============================================================================"
echo "Experiment 42 Configuration Complete"
echo "============================================================================"
echo ""
echo "Next steps:"
echo "  1. generate-segment-fileset      (~10-15 min, ~1.6 GB)"
echo "  2. generate-feature-fileset      (~30-45 min, ~31 MB)"
echo "  3. generate-segment-pairs        (~1 min, ~499,500 pairs)"
echo "  4. mpcctl-distance-function      (~8-12 hours, 2.4B calculations)"
echo "  5. mpcctl-distance-insert        (~2-4 hours, 2.4B rows)"
echo "  6. heatmap                       (~1-2 hours, 4,800 images)"
echo ""
echo "Total estimated time: 12-18 hours"
echo "Storage required: ~120-150 GB"
echo ""
