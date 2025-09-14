#!/bin/bash
# Script to clean up unwanted feature sets from experiment 41
# Keeps only the custom variance feature sets (IDs 7-10)

echo "========================================="
echo "Cleaning up Experiment 41 Feature Sets"
echo "========================================="

cd /Users/kjensen/Documents/GitHub/mldp_cli/src

echo -e "\n1. Current feature sets:"
echo "list-feature-sets" | python mldp_shell.py 2>&1 | grep -A 30 "Feature Sets"

echo -e "\n2. Removing unwanted default feature sets (IDs 1-6):"
echo "Removing basic_stats (ID 1)..."
echo -e "set experiment 41\nremove-feature-set 1" | python mldp_shell.py 2>&1 | grep "removed"

echo "Removing signal_characteristics (ID 2)..."
echo -e "set experiment 41\nremove-feature-set 2" | python mldp_shell.py 2>&1 | grep "removed"

echo "Removing electrical_parameters (ID 3)..."
echo -e "set experiment 41\nremove-feature-set 3" | python mldp_shell.py 2>&1 | grep "removed"

echo "Removing fft_features (ID 4)..."
echo -e "set experiment 41\nremove-feature-set 4" | python mldp_shell.py 2>&1 | grep "removed"

echo "Removing comprehensive (ID 5)..."
echo -e "set experiment 41\nremove-feature-set 5" | python mldp_shell.py 2>&1 | grep "removed"

echo "Removing test_voltage (ID 6)..."
echo -e "set experiment 41\nremove-feature-set 6" | python mldp_shell.py 2>&1 | grep "removed"

echo -e "\n3. Final feature sets (should only show custom variance sets):"
echo "list-feature-sets" | python mldp_shell.py 2>&1 | grep -A 15 "Feature Sets"

echo -e "\n========================================="
echo "Cleanup complete!"
echo "Remaining feature sets should be:"
echo "  - voltage_variance (ID 7)"
echo "  - current_variance (ID 8)"
echo "  - impedance_variance (ID 9)"
echo "  - power_variance (ID 10)"
echo "========================================="