#!/bin/bash
# Filename: error_handling_patterns.sh
# Author(s): Kristophor Jensen
# Date Created: 20251104_000000
# Date Revised: 20251104_000000
# File version: 1.0.0.0
# Description: Examples of error handling patterns using bash-style conditionals
#
# This script demonstrates various error handling patterns that can be used
# in MLDP shell scripts with the new bash-style conditional support.
#
# Usage (in MLDP CLI):
#   source mldp_cli/scripts/examples/error_handling_patterns.sh

echo "============================================================================"
echo "Error Handling Patterns with Bash-Style Conditionals"
echo "============================================================================"
echo ""

# ============================================================================
# Pattern 1: Simple Success/Failure Check
# ============================================================================

echo "# Pattern 1: Simple Success/Failure Check"
echo "----------------------------------------------------------------------------"
echo ""

set experiment 42
if [ $? -eq 0 ]; then
    echo "  SUCCESS: Command succeeded"
else
    echo "  ERROR: Command failed"
    exit 1
fi
echo ""

# ============================================================================
# Pattern 2: Multiple Commands with Exit on First Failure
# ============================================================================

echo "# Pattern 2: Multiple Commands with Exit on First Failure"
echo "----------------------------------------------------------------------------"
echo ""

set experiment 42
if [ $? -ne 0 ]; then
    echo "  ERROR: Failed to set experiment"
    exit 1
fi

set distance l2
if [ $? -ne 0 ]; then
    echo "  ERROR: Failed to set distance metric"
    exit 1
fi

echo "  SUCCESS: All commands succeeded"
echo ""

# ============================================================================
# Pattern 3: Try Alternative Command on Failure
# ============================================================================

echo "# Pattern 3: Try Alternative Command on Failure"
echo "----------------------------------------------------------------------------"
echo ""

set distance manhattan
if [ $? -ne 0 ]; then
    echo "  WARNING: manhattan metric not available, trying l1..."
    set distance l1
    if [ $? -ne 0 ]; then
        echo "  ERROR: No valid distance metric available"
        exit 1
    fi
fi
echo "  SUCCESS: Distance metric configured"
echo ""

# ============================================================================
# Pattern 4: Continue on Non-Critical Failure
# ============================================================================

echo "# Pattern 4: Continue on Non-Critical Failure"
echo "----------------------------------------------------------------------------"
echo ""

setvar CLEANUP_FAILED "false"

# Attempt cleanup (non-critical)
set experiment 42
if [ $? -ne 0 ]; then
    echo "  WARNING: Cleanup command failed (non-critical)"
    setvar CLEANUP_FAILED "true"
else
    echo "  SUCCESS: Cleanup completed"
fi

# Continue with critical operation
set experiment 42
if [ $? -ne 0 ]; then
    echo "  ERROR: Critical operation failed"
    exit 1
fi

if [ "$CLEANUP_FAILED" = "true" ]; then
    echo "  INFO: Pipeline succeeded with warnings"
else
    echo "  SUCCESS: Pipeline succeeded without warnings"
fi
echo ""

# ============================================================================
# Pattern 5: Nested Conditionals for Complex Logic
# ============================================================================

echo "# Pattern 5: Nested Conditionals for Complex Logic"
echo "----------------------------------------------------------------------------"
echo ""

setvar EXPERIMENT_ID 42
set experiment $EXPERIMENT_ID

if [ $? -eq 0 ]; then
    echo "  Experiment $EXPERIMENT_ID configured successfully"

    # Check if we should use l1 or l2
    setvar USE_L1 "true"

    if [ "$USE_L1" = "true" ]; then
        set distance l1
        if [ $? -eq 0 ]; then
            echo "  Distance metric set to l1"
        else
            echo "  WARNING: l1 failed, falling back to l2"
            set distance l2
        fi
    else
        set distance l2
        if [ $? -eq 0 ]; then
            echo "  Distance metric set to l2"
        else
            echo "  ERROR: Failed to set distance metric"
            exit 1
        fi
    fi
else
    echo "  ERROR: Failed to configure experiment"
    exit 1
fi
echo ""

# ============================================================================
# Pattern 6: Validation Before Execution
# ============================================================================

echo "# Pattern 6: Validation Before Execution"
echo "----------------------------------------------------------------------------"
echo ""

setvar EXPERIMENT_ID 42
setvar VALIDATION_PASSED "true"

# Validate experiment
set experiment $EXPERIMENT_ID
if [ $? -ne 0 ]; then
    echo "  VALIDATION FAILED: Invalid experiment ID"
    setvar VALIDATION_PASSED "false"
fi

# Only proceed if validation passed
if [ "$VALIDATION_PASSED" = "true" ]; then
    echo "  VALIDATION PASSED: Starting main operation"
    set distance l2
    if [ $? -eq 0 ]; then
        echo "  SUCCESS: Main operation completed"
    else
        echo "  ERROR: Main operation failed"
        exit 1
    fi
else
    echo "  ERROR: Cannot proceed - validation failed"
    exit 1
fi
echo ""

# ============================================================================
# Pattern 7: Counting Failures
# ============================================================================

echo "# Pattern 7: Counting Failures"
echo "----------------------------------------------------------------------------"
echo ""

setvar FAILURE_COUNT 0
setvar MAX_FAILURES 2

# Operation 1
set experiment 42
if [ $? -ne 0 ]; then
    setvar FAILURE_COUNT 1
    echo "  WARNING: Operation 1 failed (failures: $FAILURE_COUNT)"
fi

# Operation 2
set distance l2
if [ $? -ne 0 ]; then
    setvar FAILURE_COUNT 2
    echo "  WARNING: Operation 2 failed (failures: $FAILURE_COUNT)"
fi

# Check if too many failures
if [ $FAILURE_COUNT -ge $MAX_FAILURES ]; then
    echo "  ERROR: Too many failures ($FAILURE_COUNT >= $MAX_FAILURES)"
    echo "  Aborting pipeline"
    exit 1
else
    echo "  SUCCESS: Operations completed with $FAILURE_COUNT failures (acceptable)"
fi
echo ""

# ============================================================================
# Pattern 8: Resource Availability Check
# ============================================================================

echo "# Pattern 8: Resource Availability Check"
echo "----------------------------------------------------------------------------"
echo ""

# Check if experiment is available
set experiment 42
if [ $? -eq 0 ]; then
    echo "  Resource check: Experiment available"

    # Verify we can set distance metric
    set distance l2
    if [ $? -eq 0 ]; then
        echo "  Resource check: Distance metric configurable"
        echo "  SUCCESS: All resources available"
    else
        echo "  ERROR: Distance metric not available"
        exit 1
    fi
else
    echo "  ERROR: Experiment not available"
    exit 1
fi
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================================"
echo "Error Handling Patterns Complete"
echo "============================================================================"
echo ""
echo "Key Takeaways:"
echo "  1. Always check $? after critical commands"
echo "  2. Use if/then/else for branching logic"
echo "  3. exit 1 stops script execution on critical failures"
echo "  4. Variables can track state across commands"
echo "  5. Nested conditionals enable complex decision trees"
echo "  6. Validate before executing expensive operations"
echo "  7. Track and limit acceptable failure counts"
echo "  8. Check resource availability early"
echo ""
echo "============================================================================"
echo ""
