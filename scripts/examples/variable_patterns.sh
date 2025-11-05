#!/bin/bash
# Filename: variable_patterns.sh
# Author(s): Kristophor Jensen
# Date Created: 20251104_000000
# Date Revised: 20251104_000000
# File version: 1.0.0.0
# Description: Examples of variable usage patterns in MLDP shell scripts
#
# This script demonstrates various variable usage patterns with the new
# bash-style variable substitution support.
#
# Usage (in MLDP CLI):
#   source mldp_cli/scripts/examples/variable_patterns.sh

echo "============================================================================"
echo "Variable Usage Patterns in MLDP Shell Scripts"
echo "============================================================================"
echo ""

# ============================================================================
# Pattern 1: Configuration Variables
# ============================================================================

echo "# Pattern 1: Configuration Variables"
echo "----------------------------------------------------------------------------"
echo ""

setvar EXPERIMENT_ID 42
setvar DISTANCE_METRIC "l2"
setvar WORKERS 20
setvar SCALING_METHOD "zscore"

echo "Configuration:"
echo "  Experiment ID: $EXPERIMENT_ID"
echo "  Distance Metric: $DISTANCE_METRIC"
echo "  Workers: $WORKERS"
echo "  Scaling Method: $SCALING_METHOD"
echo ""

# ============================================================================
# Pattern 2: Using Variables in Commands
# ============================================================================

echo "# Pattern 2: Using Variables in Commands"
echo "----------------------------------------------------------------------------"
echo ""

setvar EXP_ID 42
set experiment $EXP_ID

if [ $? -eq 0 ]; then
    echo "  SUCCESS: Set experiment to $EXP_ID"
else
    echo "  ERROR: Failed to set experiment $EXP_ID"
fi
echo ""

# ============================================================================
# Pattern 3: String Variables and Concatenation
# ============================================================================

echo "# Pattern 3: String Variables and Concatenation"
echo "----------------------------------------------------------------------------"
echo ""

setvar PROJECT_NAME "MLDP"
setvar VERSION "2.0"
setvar FULL_NAME "$PROJECT_NAME v$VERSION"

echo "  Project: $PROJECT_NAME"
echo "  Version: $VERSION"
echo "  Full Name: $FULL_NAME"
echo ""

# ============================================================================
# Pattern 4: Path Variables
# ============================================================================

echo "# Pattern 4: Path Variables"
echo "----------------------------------------------------------------------------"
echo ""

setvar BASE_DIR "plots"
setvar EXPERIMENT_ID 42
setvar OUTPUT_DIR "$BASE_DIR/experiment_$EXPERIMENT_ID"

echo "  Base Directory: $BASE_DIR"
echo "  Experiment ID: $EXPERIMENT_ID"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# ============================================================================
# Pattern 5: Boolean Flags
# ============================================================================

echo "# Pattern 5: Boolean Flags"
echo "----------------------------------------------------------------------------"
echo ""

setvar DRY_RUN "false"
setvar VERBOSE "true"
setvar FORCE_MODE "false"

echo "  Dry Run: $DRY_RUN"
echo "  Verbose: $VERBOSE"
echo "  Force Mode: $FORCE_MODE"
echo ""

if [ "$DRY_RUN" = "true" ]; then
    echo "  [DRY RUN] Would execute commands"
else
    if [ "$VERBOSE" = "true" ]; then
        echo "  [VERBOSE] Executing commands with detailed output"
    else
        echo "  [NORMAL] Executing commands"
    fi
fi
echo ""

# ============================================================================
# Pattern 6: Counter Variables
# ============================================================================

echo "# Pattern 6: Counter Variables"
echo "----------------------------------------------------------------------------"
echo ""

setvar TOTAL_STEPS 6
setvar CURRENT_STEP 1
setvar COMPLETED_STEPS 0
setvar FAILED_STEPS 0

echo "  Total Steps: $TOTAL_STEPS"
echo "  Current Step: $CURRENT_STEP"
echo "  Completed: $COMPLETED_STEPS"
echo "  Failed: $FAILED_STEPS"
echo ""

# Simulate step execution
set experiment 42
if [ $? -eq 0 ]; then
    setvar COMPLETED_STEPS 1
    echo "  Step $CURRENT_STEP completed ($COMPLETED_STEPS/$TOTAL_STEPS)"
else
    setvar FAILED_STEPS 1
    echo "  Step $CURRENT_STEP failed"
fi
echo ""

# ============================================================================
# Pattern 7: Status Tracking
# ============================================================================

echo "# Pattern 7: Status Tracking"
echo "----------------------------------------------------------------------------"
echo ""

setvar PIPELINE_STATUS "not_started"
setvar ERROR_MESSAGE "none"

echo "  Initial Status: $PIPELINE_STATUS"

# Start pipeline
setvar PIPELINE_STATUS "running"
echo "  Status: $PIPELINE_STATUS"

# Execute command
set experiment 42
if [ $? -eq 0 ]; then
    setvar PIPELINE_STATUS "completed"
    echo "  Status: $PIPELINE_STATUS"
else
    setvar PIPELINE_STATUS "failed"
    setvar ERROR_MESSAGE "Experiment configuration failed"
    echo "  Status: $PIPELINE_STATUS"
    echo "  Error: $ERROR_MESSAGE"
fi
echo ""

# ============================================================================
# Pattern 8: Environment Configuration
# ============================================================================

echo "# Pattern 8: Environment Configuration"
echo "----------------------------------------------------------------------------"
echo ""

setvar ENVIRONMENT "development"
setvar DEBUG_MODE "true"
setvar LOG_LEVEL "verbose"

echo "  Environment: $ENVIRONMENT"
echo "  Debug Mode: $DEBUG_MODE"
echo "  Log Level: $LOG_LEVEL"
echo ""

if [ "$ENVIRONMENT" = "production" ]; then
    if [ "$DEBUG_MODE" = "true" ]; then
        echo "  WARNING: Debug mode enabled in production"
    fi
else
    echo "  INFO: Running in $ENVIRONMENT environment"
fi
echo ""

# ============================================================================
# Pattern 9: User Input and Validation
# ============================================================================

echo "# Pattern 9: User Input and Validation"
echo "----------------------------------------------------------------------------"
echo ""

# For demonstration, use setvar instead of input
# In interactive use: input USERNAME "Enter your username"
setvar USERNAME "demo_user"
setvar EMAIL "user@example.com"

echo "  Username: $USERNAME"
echo "  Email: $EMAIL"
echo ""

# Validate username
if [ "$USERNAME" = "" ]; then
    echo "  ERROR: Username cannot be empty"
else
    echo "  SUCCESS: Username validated: $USERNAME"
fi
echo ""

# ============================================================================
# Pattern 10: Multi-Level Configuration
# ============================================================================

echo "# Pattern 10: Multi-Level Configuration"
echo "----------------------------------------------------------------------------"
echo ""

# Level 1: Project configuration
setvar PROJECT_ROOT "/Users/kjensen/Documents/GitHub/mldp"
setvar PROJECT_NAME "MLDP"

# Level 2: Experiment configuration
setvar EXPERIMENT_ID 42
setvar EXPERIMENT_NAME "exp42"

# Level 3: Run configuration
setvar RUN_MODE "full"
setvar WORKER_COUNT 20

# Level 4: Output configuration
setvar OUTPUT_BASE "$PROJECT_ROOT/output"
setvar OUTPUT_EXPERIMENT "$OUTPUT_BASE/$EXPERIMENT_NAME"
setvar OUTPUT_PLOTS "$OUTPUT_EXPERIMENT/plots"

echo "Project Configuration:"
echo "  Root: $PROJECT_ROOT"
echo "  Name: $PROJECT_NAME"
echo ""
echo "Experiment Configuration:"
echo "  ID: $EXPERIMENT_ID"
echo "  Name: $EXPERIMENT_NAME"
echo ""
echo "Run Configuration:"
echo "  Mode: $RUN_MODE"
echo "  Workers: $WORKER_COUNT"
echo ""
echo "Output Configuration:"
echo "  Base: $OUTPUT_BASE"
echo "  Experiment: $OUTPUT_EXPERIMENT"
echo "  Plots: $OUTPUT_PLOTS"
echo ""

# ============================================================================
# Pattern 11: Exit Code Tracking
# ============================================================================

echo "# Pattern 11: Exit Code Tracking"
echo "----------------------------------------------------------------------------"
echo ""

setvar LAST_COMMAND "none"
setvar LAST_EXIT_CODE 0

# Command 1
setvar LAST_COMMAND "set experiment 42"
set experiment 42
setvar LAST_EXIT_CODE $?

echo "  Command: $LAST_COMMAND"
echo "  Exit Code: $LAST_EXIT_CODE"

if [ $LAST_EXIT_CODE -eq 0 ]; then
    echo "  Result: SUCCESS"
else
    echo "  Result: FAILED"
fi
echo ""

# ============================================================================
# Pattern 12: Conditional Variable Assignment
# ============================================================================

echo "# Pattern 12: Conditional Variable Assignment"
echo "----------------------------------------------------------------------------"
echo ""

setvar USE_PRODUCTION "false"

if [ "$USE_PRODUCTION" = "true" ]; then
    setvar DATABASE "production_db"
    setvar LOG_LEVEL "error"
    setvar WORKER_COUNT 40
else
    setvar DATABASE "development_db"
    setvar LOG_LEVEL "debug"
    setvar WORKER_COUNT 10
fi

echo "  Production Mode: $USE_PRODUCTION"
echo "  Database: $DATABASE"
echo "  Log Level: $LOG_LEVEL"
echo "  Workers: $WORKER_COUNT"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================================"
echo "Variable Usage Patterns Complete"
echo "============================================================================"
echo ""
echo "Key Takeaways:"
echo "  1. Use setvar to define variables"
echo "  2. Reference variables with $VARNAME"
echo "  3. Variables persist across commands"
echo "  4. Use meaningful variable names"
echo "  5. Quote string values with spaces"
echo "  6. Variables enable configuration management"
echo "  7. Use variables for paths and constants"
echo "  8. Boolean flags control script behavior"
echo "  9. Counters track progress and status"
echo "  10. Multi-level configuration organizes complex setups"
echo "  11. Track exit codes for debugging"
echo "  12. Conditional assignment adapts to context"
echo ""
echo "============================================================================"
echo ""
