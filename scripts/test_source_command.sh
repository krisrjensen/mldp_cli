# Test Script for MLDP CLI Source Command
# Filename: test_source_command.sh
# Author(s): Kristophor Jensen
# Date Created: 20251104_000000
# Date Revised: 20251104_000000
# File version: 1.0.0.1
# Description: Test script to verify the source command functionality
#
# Usage: source test_source_command.sh [--echo] [--continue]
#
# This script tests various aspects of the source command:
# - Comment handling (lines starting with #)
# - Blank line handling
# - Command execution
# - Command arguments
# - Read-only operations (no data modification)

# ============================================================================
# Test 1: Show current context
# ============================================================================

# Display current experiment and classifier context
show

# ============================================================================
# Test 2: List feature sets for current experiment
# ============================================================================

# This tests a command with no arguments
list-feature-sets

# ============================================================================
# Test 3: Show all feature sets (global)
# ============================================================================

# This tests another read-only command
show-all-feature-sets

# ============================================================================
# Test 4: List features with filter
# ============================================================================

# List features in a specific range (new features)
# This tests a command with arguments
list-features --range 95-138

# ============================================================================
# Test 5: Show specific feature details
# ============================================================================

# Show details for feature ID 95 (first new PSD feature)
show-feature 95

# ============================================================================
# Test 6: Help command
# ============================================================================

# Get help on a specific command
help list-feature-sets

# ============================================================================
# End of test script
# ============================================================================
