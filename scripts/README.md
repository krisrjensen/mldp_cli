# MLDP Script Files

## Overview

This directory contains example MLDP script files that can be executed using the `source` command in the MLDP CLI.

## Using the Source Command

### Syntax

```
source <filename> [--continue] [--echo]
```

### Options

- `--continue`: Continue execution even if a command fails (default: stop on first error)
- `--echo`: Echo each command before executing it (useful for debugging)

### File Search Path

The `source` command searches for script files in the following locations:

1. Current working directory
2. `~/.mldp/` directory
3. User home directory

### Script File Format

- **One command per line**: Each line contains a single MLDP command
- **Comments**: Lines starting with `#` are treated as comments and ignored
- **Blank lines**: Empty lines are ignored
- **Quoted arguments**: Arguments with spaces should be enclosed in quotes

### Example Usage

```bash
# Execute a script from the scripts directory
mldp> source example_new_experiment.mldp

# Execute with echo mode (see each command as it runs)
mldp> source example_full_pipeline.mldp --echo

# Execute with continue mode (don't stop on errors)
mldp> source example_select_data.mldp --continue

# Execute with both options
mldp> source example_full_pipeline.mldp --echo --continue
```

## Example Scripts

### example_new_experiment.mldp

Creates a new experiment and links feature sets. This is a minimal example showing:
- Database connection
- Experiment creation
- Feature set linking
- Configuration display

**Prerequisites**: None

### example_select_data.mldp

Demonstrates file and segment selection workflow. Shows:
- File selection with limits and seeds
- Segment selection with balancing
- Segment pair generation
- Physical file generation

**Prerequisites**: Run `example_new_experiment.mldp` first

### example_full_pipeline.mldp

Complete machine learning pipeline from start to finish. Includes:
- Experiment creation and setup
- Data selection
- Feature extraction
- Classifier creation and configuration
- Model training
- Results visualization

**Prerequisites**: None (self-contained)

## Creating Your Own Scripts

### Template

```bash
# Script Description
# Filename: my_script.mldp
# Author: Your Name
# Date Created: YYYYMMDD
# Description: What this script does

# Connect to database
connect

# Set active experiment
set experiment <id>

# Your commands here...

# Display results
experiment-summary <id>
```

### Best Practices

1. **Add comments**: Explain what each section does
2. **Use error handling**: Use `--continue` for non-critical failures
3. **Echo for debugging**: Use `--echo` during development
4. **Check prerequisites**: Document what needs to exist before running
5. **Use variables**: While MLDP doesn't support variables yet, use clear ID values
6. **Test incrementally**: Test each section before combining

### Common Command Sequences

#### Create Experiment
```bash
connect
experiment-create --name "my_exp" --description "Description"
set experiment <id>
bulk-link-feature-sets <id> --sets 21,22,23
```

#### Select Data
```bash
select-files --max-files 50 --seed 42
select-segments --balance-positions --min-per-file 3
generate-segment-pairs --strategy all_pairs
generate-segment-fileset --workers 16
```

#### Extract Features
```bash
generate-feature-fileset --feature-sets 21,22,23 --parallel 8
```

#### Train Classifier
```bash
classifier-new --name "my_classifier" --type svm
set classifier <id>
classifier-config-create --name "config" --active
classifier-config-set-feature-builder --config-id <id> --include-original
classifier-create-splits-table
classifier-assign-splits --strategy stratified
classifier-build-features --amplitude-method 2
classifier-train-svm --amplitude-method 2 --workers 12
```

## Troubleshooting

### Script Not Found

If you see "Script file not found", ensure:
- The file exists in one of the search paths
- The filename is spelled correctly
- You have read permissions on the file

### Command Failures

If commands fail:
- Use `--echo` to see which command failed
- Use `--continue` to skip past failing commands
- Check that experiment IDs and other values are correct
- Verify database connection is active

### Syntax Errors

Common syntax issues:
- Missing quotes around arguments with spaces
- Invalid command names (use `help` to see valid commands)
- Incorrect argument format (use `help <command>` for syntax)

## Version History

- **2.0.10.36** (2025-11-04): Initial implementation of source command
  - Added script execution feature
  - Created example scripts
  - Added documentation

## See Also

- MLDP CLI Help: Type `help` in the CLI
- Command Reference: Type `help <command>` for specific command help
- Feature Sets Documentation: See `/Users/kjensen/Documents/GitHub/mldp/documentation/wip/notebook_features_implementation_summary.md`
