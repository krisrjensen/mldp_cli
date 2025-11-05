# MLDP Shell Bash-Style Scripting Guide

**Version:** 2.0.13.4
**Date:** 2025-11-04
**Author:** Kristophor Jensen

## Table of Contents

1. [Overview](#overview)
2. [Core Features](#core-features)
3. [Conditional Statements](#conditional-statements)
4. [Variables](#variables)
5. [Exit Codes](#exit-codes)
6. [Test Expressions](#test-expressions)
7. [Best Practices](#best-practices)
8. [Complete Examples](#complete-examples)
9. [Troubleshooting](#troubleshooting)

---

## Overview

MLDP Shell v2.0.13+ supports bash-style scripting features that enable robust, production-ready automation scripts. These features allow you to write scripts with proper error handling, conditional logic, and state management.

### What's New

- **if/then/else/fi** conditional blocks
- **$?** exit code checking
- **$VARNAME** variable substitution
- **Nested conditionals** (any depth)
- **Bash-style test expressions**
- **Session-wide variables**
- **Proper exit code propagation**

### Compatibility

All features work with the existing `source` command:

```bash
source script.sh           # Execute script
source script.sh --echo    # Echo commands as they execute
source script.sh --continue # Continue on errors
```

---

## Core Features

### 1. Conditional Statements

#### Basic if/then/fi

```bash
set experiment 42
if [ $? -eq 0 ]; then
    echo "SUCCESS: Experiment configured"
fi
```

#### if/then/else/fi

```bash
set experiment 42
if [ $? -eq 0 ]; then
    echo "SUCCESS: Experiment configured"
else
    echo "ERROR: Experiment configuration failed"
    exit 1
fi
```

#### Nested Conditionals

```bash
set experiment 42
if [ $? -eq 0 ]; then
    echo "Experiment configured"

    set distance l2
    if [ $? -eq 0 ]; then
        echo "Distance metric configured"
    else
        echo "Failed to set distance metric"
        exit 1
    fi
else
    echo "Failed to configure experiment"
    exit 1
fi
```

### 2. Variables

#### Setting Variables

```bash
# Using setvar command
setvar EXPERIMENT_ID 42
setvar PROJECT_NAME "MLDP"
setvar OUTPUT_DIR "plots/experiment_42"
```

#### Using Variables

```bash
# Variable substitution with $VARNAME
echo "Experiment: $EXPERIMENT_ID"
set experiment $EXPERIMENT_ID
echo "Output directory: $OUTPUT_DIR"
```

#### Variable Scope

Variables are **session-wide** and persist across all commands:

```bash
setvar MY_VAR "value"
# ... many commands later ...
echo "Still available: $MY_VAR"
```

### 3. Exit Codes

#### Checking Exit Codes

```bash
# $? contains the exit code of the last command
set experiment 42
echo "Exit code: $?"  # Prints 0 for success, non-zero for failure
```

#### Exit Code Values

- **0** - Success
- **1** - General failure
- **127** - Command not found

#### Using exit Command

```bash
# Exit script with specific code
if [ $? -ne 0 ]; then
    echo "Critical error occurred"
    exit 1
fi
```

---

## Conditional Statements

### Syntax

```bash
if [ condition ]; then
    # commands executed if condition is true
else
    # commands executed if condition is false
fi
```

### Multi-Line Format

```bash
if [ condition ]
then
    # commands
else
    # commands
fi
```

### Without Else

```bash
if [ condition ]; then
    # commands
fi
```

### Nested If Statements

```bash
if [ condition1 ]; then
    if [ condition2 ]; then
        # Both conditions true
    else
        # condition1 true, condition2 false
    fi
else
    # condition1 false
fi
```

---

## Variables

### Variable Commands

#### setvar

Set a variable explicitly:

```bash
setvar VARNAME value
setvar COUNT 42
setvar NAME "John Doe"
setvar PATH "/home/user/data"
```

#### input

Prompt user for input (interactive):

```bash
input USERNAME "Enter your username"
echo "Hello, $USERNAME"
```

### Variable Substitution

Variables are substituted **before** command execution:

```bash
setvar EXP_ID 42
set experiment $EXP_ID  # Becomes: set experiment 42
```

### Variable Naming Rules

- Must start with letter or underscore
- Can contain letters, numbers, underscores
- Case-sensitive
- Cannot start with a number

Valid:
```bash
setvar MY_VAR "value"
setvar _private "value"
setvar count2 "value"
```

Invalid:
```bash
setvar 2count "value"    # Starts with number
setvar my-var "value"    # Contains hyphen
```

---

## Exit Codes

### How Exit Codes Work

Every command returns an exit code:
- **0** = success
- **Non-zero** = failure

The exit code is stored in `$?` and can be checked immediately:

```bash
set experiment 42
if [ $? -eq 0 ]; then
    echo "Command succeeded"
fi
```

### Commands That Return Exit Codes

All MLDP commands return proper exit codes in v2.0.13+:

- `set` - Returns 0 if valid, 1 if invalid
- `setvar` - Returns 0 on success, 1 on error
- `input` - Returns 0 if input received, 1 if cancelled
- `exit` - Exits with specified code

### Exit Code Patterns

#### Pattern 1: Stop on First Failure

```bash
set experiment 42
if [ $? -ne 0 ]; then
    exit 1
fi

set distance l2
if [ $? -ne 0 ]; then
    exit 1
fi
```

#### Pattern 2: Try Alternatives

```bash
set distance manhattan
if [ $? -ne 0 ]; then
    set distance l1
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi
```

#### Pattern 3: Track Failures

```bash
setvar FAILURES 0

set experiment 42
if [ $? -ne 0 ]; then
    setvar FAILURES 1
fi

if [ $FAILURES -gt 0 ]; then
    echo "Pipeline had $FAILURES failures"
    exit 1
fi
```

---

## Test Expressions

### Syntax

Test expressions are enclosed in `[ ]` brackets:

```bash
if [ expression ]; then
    # commands
fi
```

### Numeric Comparisons

| Operator | Meaning |
|----------|---------|
| `-eq` | Equal to |
| `-ne` | Not equal to |
| `-gt` | Greater than |
| `-lt` | Less than |
| `-ge` | Greater than or equal |
| `-le` | Less than or equal |

Examples:

```bash
if [ $? -eq 0 ]; then
    echo "Success"
fi

if [ $COUNT -gt 10 ]; then
    echo "Count exceeds 10"
fi

if [ $WORKERS -le 40 ]; then
    echo "Worker count acceptable"
fi
```

### String Comparisons

| Operator | Meaning |
|----------|---------|
| `=` | Equal to |
| `!=` | Not equal to |
| `-z` | String is empty |
| `-n` | String is not empty |

Examples:

```bash
if [ "$NAME" = "MLDP" ]; then
    echo "Name matches"
fi

if [ "$STATUS" != "failed" ]; then
    echo "Status is not failed"
fi

if [ -z "$USERNAME" ]; then
    echo "Username is empty"
fi

if [ -n "$EMAIL" ]; then
    echo "Email is provided"
fi
```

**Important:** Always quote string variables:

```bash
# GOOD
if [ "$VAR" = "value" ]; then

# BAD (can break if VAR contains spaces)
if [ $VAR = "value" ]; then
```

### File Tests

| Operator | Meaning |
|----------|---------|
| `-f` | File exists and is a regular file |
| `-d` | Directory exists |
| `-e` | Path exists (file or directory) |

Examples:

```bash
if [ -f "/path/to/file.txt" ]; then
    echo "File exists"
fi

if [ -d "/path/to/directory" ]; then
    echo "Directory exists"
fi

if [ -e "/path/to/something" ]; then
    echo "Path exists"
fi
```

### Logical Operators

#### AND (&&)

Both conditions must be true:

```bash
if [ $COUNT -gt 0 ] && [ $COUNT -lt 100 ]; then
    echo "Count is between 0 and 100"
fi
```

#### OR (||)

Either condition can be true:

```bash
if [ "$MODE" = "dev" ] || [ "$MODE" = "test" ]; then
    echo "Running in development mode"
fi
```

#### NOT (!)

Negates the condition:

```bash
if ! [ $? -eq 0 ]; then
    echo "Command failed"
fi
```

---

## Best Practices

### 1. Always Check Critical Commands

```bash
set experiment 42
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to set experiment"
    exit 1
fi
```

### 2. Use Meaningful Variable Names

```bash
# GOOD
setvar EXPERIMENT_ID 42
setvar WORKER_COUNT 20
setvar OUTPUT_DIRECTORY "plots/exp42"

# BAD
setvar X 42
setvar N 20
setvar DIR "plots/exp42"
```

### 3. Validate Before Expensive Operations

```bash
# Validate configuration first
set experiment 42
if [ $? -ne 0 ]; then
    exit 1
fi

set distance l2
if [ $? -ne 0 ]; then
    exit 1
fi

# Now run expensive operation
generate-segment-fileset
```

### 4. Provide Helpful Error Messages

```bash
set experiment 42
if [ $? -ne 0 ]; then
    echo "ERROR: Invalid experiment ID: 42"
    echo "Use 'experiment-list' to see valid experiments"
    exit 1
fi
```

### 5. Track Progress in Long Scripts

```bash
setvar TOTAL_STEPS 6
setvar CURRENT_STEP 1

echo "Step $CURRENT_STEP/$TOTAL_STEPS: Generate segments"
generate-segment-fileset
if [ $? -eq 0 ]; then
    setvar CURRENT_STEP 2
else
    echo "Failed at step $CURRENT_STEP"
    exit 1
fi
```

### 6. Use Configuration Variables

```bash
# Define configuration at top of script
setvar EXPERIMENT_ID 42
setvar WORKERS 20
setvar SCALING "zscore"
setvar OUTPUT_DIR "plots/exp42"

# Use throughout script
set experiment $EXPERIMENT_ID
generate-feature-fileset --scaling $SCALING
heatmap --output-dir $OUTPUT_DIR
```

### 7. Enable Resume Capability

```bash
setvar START_STEP 1

if [ $START_STEP -le 1 ]; then
    echo "Step 1: Generate segments"
    # ... commands ...
fi

if [ $START_STEP -le 2 ]; then
    echo "Step 2: Generate features"
    # ... commands ...
fi
```

### 8. Use Dry-Run Mode

```bash
setvar DRY_RUN "false"

if [ "$DRY_RUN" = "true" ]; then
    echo "[DRY RUN] Would execute: generate-segment-fileset"
else
    generate-segment-fileset
fi
```

---

## Complete Examples

### Example 1: Simple Pipeline with Error Handling

```bash
#!/bin/bash
# Simple pipeline with error handling

echo "Starting pipeline..."

# Step 1
set experiment 42
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to set experiment"
    exit 1
fi

# Step 2
set distance l2
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to set distance metric"
    exit 1
fi

# Step 3
generate-segment-fileset
if [ $? -eq 0 ]; then
    echo "SUCCESS: Segments generated"
else
    echo "ERROR: Segment generation failed"
    exit 1
fi

echo "Pipeline completed successfully"
```

### Example 2: Configuration-Driven Script

```bash
#!/bin/bash
# Configuration-driven pipeline

# Configuration
setvar EXPERIMENT_ID 42
setvar DISTANCE_METRIC "l2"
setvar WORKERS 20

echo "Configuration:"
echo "  Experiment: $EXPERIMENT_ID"
echo "  Distance: $DISTANCE_METRIC"
echo "  Workers: $WORKERS"

# Validate configuration
set experiment $EXPERIMENT_ID
if [ $? -ne 0 ]; then
    echo "ERROR: Invalid experiment: $EXPERIMENT_ID"
    exit 1
fi

set distance $DISTANCE_METRIC
if [ $? -ne 0 ]; then
    echo "ERROR: Invalid distance metric: $DISTANCE_METRIC"
    exit 1
fi

echo "Configuration validated"

# Execute pipeline
generate-segment-fileset
if [ $? -eq 0 ]; then
    echo "SUCCESS: Segments generated"
else
    exit 1
fi
```

### Example 3: Script with Resume Capability

```bash
#!/bin/bash
# Pipeline with resume capability

setvar START_STEP 1
setvar TOTAL_STEPS 3

echo "Pipeline execution starting at step $START_STEP"

# Step 1
if [ $START_STEP -le 1 ]; then
    echo "[$START_STEP/$TOTAL_STEPS] Generate segments"
    generate-segment-fileset
    if [ $? -ne 0 ]; then
        echo "Failed at step 1"
        echo "To resume: setvar START_STEP 1"
        exit 1
    fi
fi

# Step 2
if [ $START_STEP -le 2 ]; then
    echo "[2/$TOTAL_STEPS] Generate features"
    generate-feature-fileset
    if [ $? -ne 0 ]; then
        echo "Failed at step 2"
        echo "To resume: setvar START_STEP 2"
        exit 1
    fi
fi

# Step 3
if [ $START_STEP -le 3 ]; then
    echo "[3/$TOTAL_STEPS] Generate pairs"
    generate-segment-pairs
    if [ $? -ne 0 ]; then
        echo "Failed at step 3"
        echo "To resume: setvar START_STEP 3"
        exit 1
    fi
fi

echo "All steps completed successfully"
```

### Example 4: Pre-Flight Checks

```bash
#!/bin/bash
# Pipeline with pre-flight validation

echo "Pre-flight checks..."

# Check 1: Experiment exists
set experiment 42
if [ $? -ne 0 ]; then
    echo "FAILED: Experiment not configured"
    exit 1
fi
echo "  Experiment: OK"

# Check 2: Distance metric valid
set distance l2
if [ $? -ne 0 ]; then
    echo "FAILED: Invalid distance metric"
    exit 1
fi
echo "  Distance metric: OK"

# Check 3: Worker count reasonable
setvar WORKERS 20
if [ $WORKERS -gt 40 ]; then
    echo "WARNING: High worker count may cause issues"
fi
echo "  Workers: OK"

echo "All pre-flight checks passed"

# Execute pipeline
echo "Starting pipeline..."
generate-segment-fileset
```

---

## Troubleshooting

### Problem: Variable Not Substituting

**Symptom:**
```bash
setvar NAME "Alice"
echo "Hello $NAME"
# Outputs: Hello $NAME (literal)
```

**Solution:**
Ensure you're using MLDP Shell v2.0.13+. Variable substitution happens before command execution in this version.

### Problem: Exit Code Always Zero

**Symptom:**
```bash
set experiment 99999  # Invalid experiment
if [ $? -eq 0 ]; then
    echo "This shouldn't print"
fi
# But it does print!
```

**Solution:**
Ensure you're using MLDP Shell v2.0.13.2+ where `cmd_set` returns proper exit codes. Earlier versions always returned 0.

### Problem: Script Stops After Failed Command

**Symptom:**
```bash
set experiment 99999  # This fails
if [ $? -eq 0 ]; then
    # This never executes - script already stopped
fi
```

**Solution:**
This was fixed in v2.0.13.4. The script executor now continues to the next block if it's an `if` statement, allowing it to check the exit code.

### Problem: Nested If Not Working

**Symptom:**
```bash
if [ condition1 ]; then
    if [ condition2 ]; then
        echo "Nested"
    fi
fi
# ERROR: Unknown command: if
```

**Solution:**
Ensure you're using MLDP Shell v2.0.13.1+. Nested conditionals were fixed in this version with recursive block parsing.

### Problem: String Comparison Not Working

**Symptom:**
```bash
setvar MODE "production"
if [ $MODE = "production" ]; then
    # This might not work correctly
fi
```

**Solution:**
Always quote string variables:
```bash
if [ "$MODE" = "production" ]; then
    # This works correctly
fi
```

### Problem: Variable Has Spaces

**Symptom:**
```bash
setvar NAME John Doe
echo "$NAME"
# Only outputs: John
```

**Solution:**
Quote values with spaces:
```bash
setvar NAME "John Doe"
echo "$NAME"
# Outputs: John Doe
```

---

## Version History

### v2.0.13.4 (2025-11-04)
- Fixed script executor to continue when if block follows failed command
- All 7 test cases passing

### v2.0.13.3 (2025-11-04)
- Initial fix for script executor continuation (had logic error)

### v2.0.13.2 (2025-11-04)
- Added exit code validation to `cmd_set`
- Experiment ID validated against database
- Distance type validated against allowed values

### v2.0.13.1 (2025-11-04)
- Fixed nested if statement parsing
- Added recursive block parsing

### v2.0.13.0 (2025-11-04)
- Implemented if/then/else/fi conditional blocks
- Added script_parser.py with block-based parsing
- Implemented test expression evaluation

### v2.0.12.1 (2025-11-04)
- Fixed variable substitution in main REPL loop

### v2.0.12.0 (2025-11-04)
- Initial implementation of variables ($VARNAME, $?)
- Added `input` and `setvar` commands
- Added `_substitute_variables()` method

---

## References

- Test script: `mldp_cli/scripts/test_conditionals.sh`
- Example scripts: `mldp_cli/scripts/examples/`
- Enhanced pipeline: `mldp_cli/scripts/enhanced_pipeline_example.sh`

---

**End of Bash-Style Scripting Guide**
