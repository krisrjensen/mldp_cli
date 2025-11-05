# Advanced Test Script for MLDP CLI Source Command
# Filename: test_source_advanced.sh
# Author(s): Kristophor Jensen
# Date Created: 20251104_000000
# Date Revised: 20251104_000000
# File version: 1.0.0.1
# Description: Advanced test script to verify source command error handling
#
# Usage:
#   source test_source_advanced.sh --echo          # Show each command
#   source test_source_advanced.sh --continue      # Continue on errors
#   source test_source_advanced.sh --echo --continue  # Both options
#
# This script tests:
# - Multiple blank lines
# - Various comment styles
# - Commands with multiple arguments
# - Commands with quoted arguments
# - Error handling (intentional errors to test --continue flag)


# ============================================================================
# Section 1: Context and Configuration
# ============================================================================

# Multiple blank lines should be skipped



# Show current experiment
show

# ============================================================================
# Section 2: Feature Set Queries
# ============================================================================

# Check if new feature sets exist (21-24)
show-all-feature-sets

# List features from the new PSD feature set
# Feature set 21 should have 32 PSD features
list-feature-sets

# ============================================================================
# Section 3: Feature Details
# ============================================================================

# Show first feature from each new category
show-feature 95   # v_ultra_high_snr (PSD)
show-feature 127  # v_volatility_mean (Volatility)
show-feature 135  # v_kurtosis (Time-domain)

# ============================================================================
# Section 4: Help System
# ============================================================================

# Test help for various commands
help show-feature
help list-features

# ============================================================================
# Section 5: Error Handling Tests
# ============================================================================

# NOTE: The following commands should fail
# Use --continue flag to see script continue past these errors

# This should fail - invalid feature ID
# show-feature 99999

# This should fail - unknown command
# invalid-command-that-does-not-exist

# This should fail - missing required argument
# show-feature

# ============================================================================
# Section 6: Multi-line Test
# ============================================================================

# Show that we're still executing after any errors above
show

# ============================================================================
# End of advanced test script
# ============================================================================

# If you see this message, script execution completed
# (Note: Comments aren't printed, so this is for documentation only)
