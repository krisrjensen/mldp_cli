#!/bin/bash
# Test script for configuring experiment 41 via MLDP CLI
# Author: Kristophor Jensen
# Date: 20250913_182000

echo "========================================="
echo "Testing Experiment 41 Configuration"
echo "========================================="

cd /Users/kjensen/Documents/GitHub/mldp_cli/src

echo -e "\n1. Setting current experiment to 41:"
echo "set experiment 41" | python mldp_shell.py 2>&1 | grep -A 1 "Current experiment"

echo -e "\n2. Showing current settings:"
echo "show" | python mldp_shell.py 2>&1 | grep -A 5 "Current Settings"

echo -e "\n3. Updating decimations to [0, 7, 15]:"
echo -e "set experiment 41\nupdate-decimations 0 7 15" | python mldp_shell.py 2>&1 | grep "Decimations"

echo -e "\n4. Updating segment sizes to [128, 1024, 8192]:"
echo -e "set experiment 41\nupdate-segment-sizes 128 1024 8192" | python mldp_shell.py 2>&1 | grep "Segment sizes"

echo -e "\n5. Updating amplitude methods to [minmax, zscore]:"
echo -e "set experiment 41\nupdate-amplitude-methods minmax zscore" | python mldp_shell.py 2>&1 | grep "Amplitude methods"

echo -e "\n6. Creating voltage_variance feature set:"
echo -e 'set experiment 41\ncreate-feature-set --name voltage_variance --features voltage,variance(voltage) --n-value 128' | python mldp_shell.py 2>&1 | grep -A 4 "Feature set"

echo -e "\n7. Creating current_variance feature set:"
echo -e 'set experiment 41\ncreate-feature-set --name current_variance --features current,variance(current) --n-value 128' | python mldp_shell.py 2>&1 | grep -A 4 "Feature set"

echo -e "\n8. Creating impedance_variance feature set:"
echo -e 'set experiment 41\ncreate-feature-set --name impedance_variance --features impedance,variance(impedance) --n-value 128' | python mldp_shell.py 2>&1 | grep -A 4 "Feature set"

echo -e "\n9. Creating power_variance feature set:"
echo -e 'set experiment 41\ncreate-feature-set --name power_variance --features power,variance(power) --n-value 128' | python mldp_shell.py 2>&1 | grep -A 4 "Feature set"

echo -e "\n10. Viewing experiment configuration:"
echo -e "set experiment 41\nexperiment-config 41" | python mldp_shell.py 2>&1 | grep -A 20 "Configuration for Experiment"

echo -e "\n========================================="
echo "Test complete! Review output above."
echo "========================================="

echo -e "\nTo run segment selection for experiment 41:"
echo "  echo 'experiment-select 41' | python mldp_shell.py"