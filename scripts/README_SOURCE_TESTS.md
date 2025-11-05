# Source Command Test Scripts

**Author:** Kristophor Jensen
**Date Created:** 20251104_000000
**Version:** 1.0.0.1

## Overview

This directory contains test scripts for verifying the `source` command functionality in MLDP CLI.

## Available Test Scripts

### 1. test_source_command.sh
Basic functionality test for the source command.

**Purpose:** Verify basic source command operation with simple read-only commands.

**Tests:**
- Comment handling (lines starting with #)
- Blank line skipping
- Command execution without arguments
- Command execution with arguments
- Help system

**Usage:**
```bash
# In MLDP CLI:
source test_source_command.sh
source test_source_command.sh --echo
source test_source_command.sh --continue --echo
```

**Expected Result:**
- All commands execute successfully
- Script displays current context, feature sets, and feature details
- No errors should occur

---

### 2. test_source_advanced.sh
Advanced functionality test with error handling.

**Purpose:** Test source command with more complex scenarios and error handling.

**Tests:**
- Multiple consecutive blank lines
- Multiple comment styles
- Commands with multiple arguments
- Commands with quoted arguments
- Error handling (commented-out intentional errors)

**Usage:**
```bash
# In MLDP CLI:
source test_source_advanced.sh --echo
source test_source_advanced.sh --continue --echo

# To test error handling, uncomment error test commands in Section 5
```

**Expected Result:**
- All non-commented commands execute successfully
- With --echo flag, see each command before execution
- With --continue flag, execution continues past errors

---

## Source Command Options

### --echo
Displays each command before executing it.
```
[line_number] command arguments
```

### --continue
Continues script execution even if a command fails. Without this flag, script stops on first error.

---

## File Format

Script files must follow these rules:

1. **One command per line**
   ```bash
   show
   list-feature-sets
   ```

2. **Comments start with #**
   ```bash
   # This is a comment
   show  # Inline comments are NOT supported
   ```

3. **Blank lines are ignored**
   ```bash
   show

   list-feature-sets
   ```

4. **Arguments with spaces should be quoted**
   ```bash
   create-feature-set "My Feature Set" --description "This has spaces"
   ```

---

## Search Path

The `source` command searches for script files in:

1. Current working directory
2. `~/.mldp/` directory
3. User home directory

---

## Examples

### Run basic test:
```bash
mldp> source test_source_command.sh
📜 Executing script: /path/to/test_source_command.sh
   Continue on error: False
   Echo commands: False

# Output from commands follows...
```

### Run with echo enabled:
```bash
mldp> source test_source_command.sh --echo
📜 Executing script: /path/to/test_source_command.sh
   Continue on error: False
   Echo commands: True

[15] show
Current Experiment ID: 41
Current Classifier ID: 1
...
```

### Run with error continuation:
```bash
mldp> source test_source_advanced.sh --continue --echo
# Will continue past errors and show summary at end
```

---

## Creating Custom Test Scripts

To create your own test scripts:

1. Create a new .sh file in this directory
2. Add header with author, date, version
3. Use # for comments
4. One command per line
5. Test with --echo first to verify commands
6. Use --continue to test error handling

**Template:**
```bash
# My Custom Test Script
# Author: Your Name
# Date Created: YYYYMMDD_HHMMSS
# Description: What this script tests

# Show context
show

# Your commands here
list-feature-sets

# End of script
```

---

## Verification Checklist

After running test scripts, verify:

- ✅ All commands executed without errors (test_source_command.sh)
- ✅ Echo flag displays each command before execution
- ✅ Continue flag allows script to proceed past errors
- ✅ Comments and blank lines are properly skipped
- ✅ Commands with arguments execute correctly
- ✅ Script summary shows correct counts (total, executed, failed, skipped)

---

## Troubleshooting

**Problem:** Script file not found
- Check file exists in search path
- Use absolute path: `source /full/path/to/script.sh`

**Problem:** Command fails immediately
- Check command syntax with `help <command>`
- Try command manually first
- Use --echo to see exactly what's being executed

**Problem:** Script stops on error
- Use --continue flag to proceed past errors
- Check which command failed in output
- Fix failing command or comment it out

---

## Related Documentation

- MLDP CLI Help: `help source`
- Command Reference: `help <command_name>`
- Feature Management: `help list-features`
