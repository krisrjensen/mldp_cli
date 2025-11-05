#!/bin/bash
# Filename: run_experiment_42_pipeline.sh
# Author(s): Kristophor Jensen
# Date Created: 20251104_000000
# Date Revised: 20251104_000000
# File version: 1.0.0.1
# Description: Complete execution pipeline for experiment 42
#
# This script runs the full experiment 42 pipeline:
# 1. Generate segment files (~10-15 min, ~1.6 GB)
# 2. Generate feature files (~30-45 min, ~31 MB)
# 3. Generate segment pairs (~1 min, ~499,500 pairs)
# 4. Compute distances (~8-12 hours, 2.4B calculations)
# 5. Insert distances (~2-4 hours, 2.4B rows)
# 6. Generate heatmaps (~1-2 hours, 4,800 images)
#
# Usage (in MLDP CLI):
#   source mldp_cli/scripts/run_experiment_42_pipeline.sh
#
# Configuration:
#   Workers: 20 (for parallel processing)
#   Expected Runtime: 12-18 hours
#   Storage Required: ~120-150 GB
#
# Prerequisites:
#   - Experiment 42 must be configured (run setup_experiment_42.sh first)
#   - Sufficient disk space available
#   - Sufficient memory for 20 workers

echo "============================================================================"
echo "Experiment 42 Pipeline Execution"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Experiment ID: 42"
echo "  Workers: 20"
echo "  Expected Runtime: 12-18 hours"
echo "  Storage Required: ~120-150 GB"
echo ""
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "============================================================================"
echo ""

# Set experiment to 42
echo "# Setting experiment to 42"
set experiment 42
echo ""

# Step 1: Generate segment files
echo "============================================================================"
echo "STEP 1/6: Generate Segment Files"
echo "============================================================================"
echo "Expected time: 10-15 minutes"
echo "Expected size: ~1.6 GB"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

generate-segment-fileset

if [ $? -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Segment fileset generation completed"
    echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo ""
    echo "[ERROR] Segment fileset generation failed"
    echo "Stopping pipeline execution"
    exit 1
fi
echo ""

# Step 2: Generate feature files
echo "============================================================================"
echo "STEP 2/6: Generate Feature Files"
echo "============================================================================"
echo "Expected time: 30-45 minutes"
echo "Expected size: ~31 MB"
echo "Scaling method: zscore"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

generate-feature-fileset --scaling zscore

if [ $? -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Feature fileset generation completed"
    echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo ""
    echo "[ERROR] Feature fileset generation failed"
    echo "Stopping pipeline execution"
    exit 1
fi
echo ""

# Step 3: Generate segment pairs
echo "============================================================================"
echo "STEP 3/6: Generate Segment Pairs"
echo "============================================================================"
echo "Expected time: ~1 minute"
echo "Expected pairs: ~499,500"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

generate-segment-pairs

if [ $? -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Segment pairs generation completed"
    echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo ""
    echo "[ERROR] Segment pairs generation failed"
    echo "Stopping pipeline execution"
    exit 1
fi
echo ""

# Step 4: Compute distances
echo "============================================================================"
echo "STEP 4/6: Compute Distance Functions"
echo "============================================================================"
echo "Expected time: 8-12 hours"
echo "Expected calculations: ~2.4 billion"
echo "Workers: 20"
echo "Work files: 4,800 (100 feature sets × 24 data combos × 2 metrics)"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

mpcctl-distance-function --workers 20 --log --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Distance function computation completed"
    echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo ""
    echo "[ERROR] Distance function computation failed"
    echo "Stopping pipeline execution"
    exit 1
fi
echo ""

# Step 5: Insert distances to database
echo "============================================================================"
echo "STEP 5/6: Insert Distances to Database"
echo "============================================================================"
echo "Expected time: 2-4 hours"
echo "Expected rows: ~2.4 billion"
echo "Workers: 20"
echo "Tables: experiment_042_distance_l1, experiment_042_distance_cosine"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

mpcctl-distance-insert --workers 20 --log --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Distance insertion completed"
    echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo ""
    echo "[ERROR] Distance insertion failed"
    echo "Stopping pipeline execution"
    exit 1
fi
echo ""

# Step 6: Generate heatmaps
echo "============================================================================"
echo "STEP 6/6: Generate Heatmaps"
echo "============================================================================"
echo "Expected time: 1-2 hours"
echo "Expected images: ~4,800"
echo "Output directory: plots/experiment_42"
echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

heatmap --output-dir plots/experiment_42

if [ $? -eq 0 ]; then
    echo ""
    echo "[SUCCESS] Heatmap generation completed"
    echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
else
    echo ""
    echo "[ERROR] Heatmap generation failed"
    echo "Note: Pipeline execution completed except for heatmaps"
    exit 1
fi
echo ""

# Final summary
echo "============================================================================"
echo "Experiment 42 Pipeline Execution Complete"
echo "============================================================================"
echo ""
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "All steps completed successfully:"
echo "  [1/6] Segment files generated"
echo "  [2/6] Feature files generated"
echo "  [3/6] Segment pairs generated"
echo "  [4/6] Distance functions computed"
echo "  [5/6] Distances inserted to database"
echo "  [6/6] Heatmaps generated"
echo ""
echo "Next steps:"
echo "  1. Review heatmaps in plots/experiment_42/"
echo "  2. Analyze feature separability results"
echo "  3. Select best-performing feature sets"
echo "  4. Plan classifier training with selected features"
echo ""
echo "============================================================================"
echo ""
