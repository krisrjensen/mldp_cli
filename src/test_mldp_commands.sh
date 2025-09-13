#!/bin/bash
# Test script for MLDP CLI experiment commands

echo "Testing MLDP CLI Experiment Commands"
echo "====================================="

cd /Users/kjensen/Documents/GitHub/mldp/mldp_cli/src

echo -e "\n1. Testing experiment-list command:"
echo "experiment-list" | python mldp_shell.py 2>&1 | grep -A 10 "Available Experiments"

echo -e "\n2. Testing experiment-info command:"
echo "experiment-info 18" | python mldp_shell.py 2>&1 | grep -A 20 "EXPERIMENT 18"

echo -e "\n3. Testing experiment-config command:"
echo "experiment-config 18" | python mldp_shell.py 2>&1 | grep -A 10 "Configuration for Experiment"

echo -e "\n4. Testing experiment-summary command:"
echo "experiment-summary 18" | python mldp_shell.py 2>&1 | grep -A 5 "EXPERIMENT 18"

echo -e "\n5. Testing help command for experiments:"
echo "help" | python mldp_shell.py 2>&1 | grep -A 5 "EXPERIMENT COMMANDS"

echo -e "\nAll tests completed!"