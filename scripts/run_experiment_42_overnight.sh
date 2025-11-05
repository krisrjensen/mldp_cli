#!/bin/bash
# Filename: run_experiment_42_overnight.sh
# Author(s): Kristophor Jensen
# Date Created: 20251104_000000
# Date Revised: 20251104_000000
# File version: 1.0.0.1
# Description: Wrapper script to run experiment 42 pipeline overnight
#
# This script launches the MLDP shell and executes the full pipeline
# with all output logged to a timestamped file.
#
# Usage (from mldp_cli directory):
#   bash scripts/run_experiment_42_overnight.sh
#
# Or run in background with nohup:
#   nohup bash scripts/run_experiment_42_overnight.sh > /dev/null 2>&1 &
#
# The script will:
#   - Create timestamped log file in logs/
#   - Run entire experiment 42 pipeline
#   - Log all output (stdout and stderr)
#   - Exit when complete
#
# Monitor progress:
#   tail -f logs/experiment_42_pipeline_YYYYMMDD_HHMMSS.log
#
# Expected runtime: 12-18 hours
# Storage required: ~120-150 GB

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MLDP_CLI_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Create logs directory if it doesn't exist
LOGS_DIR="$MLDP_CLI_DIR/logs"
mkdir -p "$LOGS_DIR"

# Generate timestamp for log file
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOGS_DIR/experiment_42_pipeline_${TIMESTAMP}.log"

echo "============================================================================"
echo "Experiment 42 Overnight Pipeline Execution"
echo "============================================================================"
echo ""
echo "Configuration:"
echo "  Working Directory: $MLDP_CLI_DIR"
echo "  Log File: $LOG_FILE"
echo "  Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Expected Runtime: 12-18 hours"
echo "Storage Required: ~120-150 GB"
echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_FILE"
echo ""
echo "============================================================================"
echo ""

# Change to mldp_cli directory
cd "$MLDP_CLI_DIR" || exit 1

# Create temporary script to feed to MLDP shell
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" <<'EOF'
source mldp_cli/scripts/run_experiment_42_pipeline.sh
exit
EOF

# Log environment info
{
    echo "============================================================================"
    echo "Environment Information"
    echo "============================================================================"
    echo "Hostname: $(hostname)"
    echo "User: $(whoami)"
    echo "Working Directory: $(pwd)"
    echo "Python Version: $(python3 --version)"
    echo "Date: $(date)"
    echo "PID: $$"
    echo "============================================================================"
    echo ""
} >> "$LOG_FILE" 2>&1

# Run MLDP shell with pipeline script
{
    echo "Starting MLDP shell and executing pipeline..."
    echo ""
    python3 src/mldp_shell.py < "$TEMP_SCRIPT"
    EXIT_CODE=$?
    echo ""
    echo "============================================================================"
    echo "Pipeline Execution Finished"
    echo "============================================================================"
    echo "Exit Code: $EXIT_CODE"
    echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================================"
    echo ""
    exit $EXIT_CODE
} >> "$LOG_FILE" 2>&1

# Clean up temp script
rm -f "$TEMP_SCRIPT"

# Print completion message
echo "Pipeline execution finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Check log file for details: $LOG_FILE"
echo ""

# Print summary of log file
echo "Last 50 lines of log:"
echo "============================================================================"
tail -50 "$LOG_FILE"
echo "============================================================================"
