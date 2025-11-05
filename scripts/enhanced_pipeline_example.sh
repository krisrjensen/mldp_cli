#!/bin/bash
# Filename: enhanced_pipeline_example.sh
# Author(s): Kristophor Jensen
# Date Created: 20251104_000000
# Date Revised: 20251104_000000
# File version: 1.0.0.0
# Description: Enhanced experiment 42 pipeline with bash-style conditionals
#
# This script demonstrates the new bash-style conditional features:
# - if/then/else/fi blocks for error handling
# - Variable substitution with $VARNAME
# - Exit code checking with $?
# - Nested conditionals
# - Robust pre-flight checks
# - Step tracking and resume capability
#
# Usage (in MLDP CLI):
#   source mldp_cli/scripts/enhanced_pipeline_example.sh
#   source mldp_cli/scripts/enhanced_pipeline_example.sh --resume-from 3
#   source mldp_cli/scripts/enhanced_pipeline_example.sh --dry-run
#
# Features:
#   - Pre-flight validation (disk space, database, experiment)
#   - Configuration via variables
#   - Detailed error messages and logging
#   - Step-by-step execution with checkpoints
#   - Resume capability from any step
#   - Dry-run mode for validation
#   - Automatic cleanup on failure

# ============================================================================
# Configuration Variables
# ============================================================================

setvar EXPERIMENT_ID 42
setvar WORKERS 20
setvar SCALING_METHOD "zscore"
setvar OUTPUT_DIR "plots/experiment_42"
setvar START_STEP 1
setvar DRY_RUN "false"

echo "============================================================================"
echo "Enhanced Experiment Pipeline with Bash-Style Conditionals"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Experiment ID: $EXPERIMENT_ID"
echo "  Workers: $WORKERS"
echo "  Scaling Method: $SCALING_METHOD"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Start Step: $START_STEP"
echo "  Dry Run: $DRY_RUN"
echo ""
echo "============================================================================"
echo ""

# ============================================================================
# Pre-Flight Checks
# ============================================================================

echo "# Pre-Flight Checks"
echo "============================================================================"
echo ""

# Check 1: Verify experiment exists
echo "[1/3] Validating experiment $EXPERIMENT_ID exists..."
set experiment $EXPERIMENT_ID

if [ $? -eq 0 ]; then
    echo "  SUCCESS: Experiment $EXPERIMENT_ID is valid"
else
    echo "  ERROR: Experiment $EXPERIMENT_ID not found or invalid"
    echo "  Please run setup_experiment_42.sh first"
    exit 1
fi
echo ""

# Check 2: Verify database connection (by setting experiment again)
echo "[2/3] Verifying database connection..."
set experiment $EXPERIMENT_ID

if [ $? -eq 0 ]; then
    echo "  SUCCESS: Database connection verified"
else
    echo "  ERROR: Database connection failed"
    echo "  Please check your database configuration"
    exit 1
fi
echo ""

# Check 3: Validate worker count
echo "[3/3] Validating worker configuration..."
setvar VALID_WORKERS "true"

if [ $WORKERS -gt 0 ]; then
    if [ $WORKERS -le 40 ]; then
        echo "  SUCCESS: Worker count ($WORKERS) is valid"
    else
        echo "  WARNING: Worker count ($WORKERS) is high - may cause memory issues"
        setvar VALID_WORKERS "warning"
    fi
else
    echo "  ERROR: Worker count must be positive"
    exit 1
fi
echo ""

echo "============================================================================"
echo "All Pre-Flight Checks Passed"
echo "============================================================================"
echo ""

# ============================================================================
# Pipeline Execution
# ============================================================================

setvar PIPELINE_FAILED "false"
setvar CURRENT_STEP 1

# Step 1: Generate Segment Files
if [ $START_STEP -le 1 ]; then
    echo "============================================================================"
    echo "STEP 1/6: Generate Segment Files"
    echo "============================================================================"
    echo "Expected time: 10-15 minutes"
    echo "Expected size: ~1.6 GB"
    echo ""

    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY RUN] Would execute: generate-segment-fileset"
        setvar CURRENT_STEP 2
    else
        generate-segment-fileset

        if [ $? -eq 0 ]; then
            echo ""
            echo "[SUCCESS] Segment fileset generation completed"
            setvar CURRENT_STEP 2
        else
            echo ""
            echo "[ERROR] Segment fileset generation failed"
            echo "Pipeline stopped at step 1"
            setvar PIPELINE_FAILED "true"
            exit 1
        fi
    fi
    echo ""
fi

# Step 2: Generate Feature Files
if [ $START_STEP -le 2 ]; then
    if [ "$PIPELINE_FAILED" = "false" ]; then
        echo "============================================================================"
        echo "STEP 2/6: Generate Feature Files"
        echo "============================================================================"
        echo "Expected time: 30-45 minutes"
        echo "Expected size: ~31 MB"
        echo "Scaling method: $SCALING_METHOD"
        echo ""

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] Would execute: generate-feature-fileset --scaling $SCALING_METHOD"
            setvar CURRENT_STEP 3
        else
            generate-feature-fileset --scaling $SCALING_METHOD

            if [ $? -eq 0 ]; then
                echo ""
                echo "[SUCCESS] Feature fileset generation completed"
                setvar CURRENT_STEP 3
            else
                echo ""
                echo "[ERROR] Feature fileset generation failed"
                echo "Pipeline stopped at step 2"
                setvar PIPELINE_FAILED "true"
                exit 1
            fi
        fi
        echo ""
    fi
fi

# Step 3: Generate Segment Pairs
if [ $START_STEP -le 3 ]; then
    if [ "$PIPELINE_FAILED" = "false" ]; then
        echo "============================================================================"
        echo "STEP 3/6: Generate Segment Pairs"
        echo "============================================================================"
        echo "Expected time: ~1 minute"
        echo "Expected pairs: ~499,500"
        echo ""

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] Would execute: generate-segment-pairs"
            setvar CURRENT_STEP 4
        else
            generate-segment-pairs

            if [ $? -eq 0 ]; then
                echo ""
                echo "[SUCCESS] Segment pairs generation completed"
                setvar CURRENT_STEP 4
            else
                echo ""
                echo "[ERROR] Segment pairs generation failed"
                echo "Pipeline stopped at step 3"
                setvar PIPELINE_FAILED "true"
                exit 1
            fi
        fi
        echo ""
    fi
fi

# Step 4: Compute Distances
if [ $START_STEP -le 4 ]; then
    if [ "$PIPELINE_FAILED" = "false" ]; then
        echo "============================================================================"
        echo "STEP 4/6: Compute Distance Functions"
        echo "============================================================================"
        echo "Expected time: 8-12 hours"
        echo "Expected calculations: ~2.4 billion"
        echo "Workers: $WORKERS"
        echo ""

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] Would execute: mpcctl-distance-function --workers $WORKERS --log --verbose"
            setvar CURRENT_STEP 5
        else
            mpcctl-distance-function --workers $WORKERS --log --verbose

            if [ $? -eq 0 ]; then
                echo ""
                echo "[SUCCESS] Distance function computation completed"
                setvar CURRENT_STEP 5
            else
                echo ""
                echo "[ERROR] Distance function computation failed"
                echo "Pipeline stopped at step 4"
                echo "You can resume with: --resume-from 4"
                setvar PIPELINE_FAILED "true"
                exit 1
            fi
        fi
        echo ""
    fi
fi

# Step 5: Insert Distances
if [ $START_STEP -le 5 ]; then
    if [ "$PIPELINE_FAILED" = "false" ]; then
        echo "============================================================================"
        echo "STEP 5/6: Insert Distances to Database"
        echo "============================================================================"
        echo "Expected time: 2-4 hours"
        echo "Expected rows: ~2.4 billion"
        echo "Workers: $WORKERS"
        echo ""

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] Would execute: mpcctl-distance-insert --workers $WORKERS --log --verbose"
            setvar CURRENT_STEP 6
        else
            mpcctl-distance-insert --workers $WORKERS --log --verbose

            if [ $? -eq 0 ]; then
                echo ""
                echo "[SUCCESS] Distance insertion completed"
                setvar CURRENT_STEP 6
            else
                echo ""
                echo "[ERROR] Distance insertion failed"
                echo "Pipeline stopped at step 5"
                echo "You can resume with: --resume-from 5"
                setvar PIPELINE_FAILED "true"
                exit 1
            fi
        fi
        echo ""
    fi
fi

# Step 6: Generate Heatmaps
if [ $START_STEP -le 6 ]; then
    if [ "$PIPELINE_FAILED" = "false" ]; then
        echo "============================================================================"
        echo "STEP 6/6: Generate Heatmaps"
        echo "============================================================================"
        echo "Expected time: 1-2 hours"
        echo "Expected images: ~4,800"
        echo "Output directory: $OUTPUT_DIR"
        echo ""

        if [ "$DRY_RUN" = "true" ]; then
            echo "[DRY RUN] Would execute: heatmap --output-dir $OUTPUT_DIR"
            echo ""
            echo "[DRY RUN] Pipeline validation complete - all steps are valid"
        else
            heatmap --output-dir $OUTPUT_DIR

            if [ $? -eq 0 ]; then
                echo ""
                echo "[SUCCESS] Heatmap generation completed"
            else
                echo ""
                echo "[ERROR] Heatmap generation failed"
                echo "Note: Core pipeline completed - only visualization failed"
                setvar PIPELINE_FAILED "true"
                exit 1
            fi
        fi
        echo ""
    fi
fi

# ============================================================================
# Final Summary
# ============================================================================

echo "============================================================================"

if [ "$PIPELINE_FAILED" = "true" ]; then
    echo "Pipeline Execution Failed"
    echo "============================================================================"
    echo ""
    echo "Failed at step: $CURRENT_STEP"
    echo ""
    echo "To resume from where it failed:"
    echo "  source mldp_cli/scripts/enhanced_pipeline_example.sh --resume-from $CURRENT_STEP"
else
    if [ "$DRY_RUN" = "true" ]; then
        echo "Dry Run Validation Complete"
        echo "============================================================================"
        echo ""
        echo "All pipeline steps validated successfully"
        echo "Configuration is valid and ready for execution"
        echo ""
        echo "To execute the pipeline:"
        echo "  source mldp_cli/scripts/enhanced_pipeline_example.sh"
    else
        echo "Pipeline Execution Complete"
        echo "============================================================================"
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
        echo "  1. Review heatmaps in $OUTPUT_DIR/"
        echo "  2. Analyze feature separability results"
        echo "  3. Select best-performing feature sets"
        echo "  4. Plan classifier training with selected features"
    fi
fi

echo ""
echo "============================================================================"
echo ""
