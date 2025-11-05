#!/bin/bash
# Test script for bash-style conditionals in MLDP shell
#
# Tests:
# - if/then/else/fi blocks
# - Exit code checking with $?
# - Variable substitution
# - Nested conditions

echo "============================================"
echo "Testing Bash-Style Conditionals"
echo "============================================"
echo ""

# Test 1: Simple if/then/fi
echo "Test 1: Simple if/then/fi"
set experiment 42
if [ $? -eq 0 ]; then
    echo "  SUCCESS: Experiment set successfully"
fi
echo ""

# Test 2: if/then/else/fi
echo "Test 2: if/then/else/fi with failure"
set experiment 99999
if [ $? -eq 0 ]; then
    echo "  ERROR: Should not reach here"
else
    echo "  SUCCESS: Correctly handled failure"
fi
echo ""

# Test 3: String comparison
echo "Test 3: String comparison"
setvar TEST_VAR "hello"
if [ "$TEST_VAR" = "hello" ]; then
    echo "  SUCCESS: String equality works"
else
    echo "  ERROR: String equality failed"
fi
echo ""

# Test 4: Numeric comparison
echo "Test 4: Numeric comparison"
setvar COUNT 42
if [ $COUNT -gt 40 ]; then
    echo "  SUCCESS: Numeric comparison works"
else
    echo "  ERROR: Numeric comparison failed"
fi
echo ""

# Test 5: Nested if statements
echo "Test 5: Nested if statements"
setvar LEVEL 2
if [ $LEVEL -gt 0 ]; then
    echo "  Level is positive"
    if [ $LEVEL -eq 2 ]; then
        echo "  SUCCESS: Level is exactly 2 (nested condition works)"
    fi
fi
echo ""

# Test 6: Variable in echo after if block
echo "Test 6: Variable substitution in commands"
setvar NAME "MLDP"
echo "  Hello from $NAME"
echo ""

# Test 7: Exit code persistence
echo "Test 7: Exit code persistence"
setvar DUMMY "value"
echo "  Last exit code after setvar: $?"
echo ""

echo "============================================"
echo "All Tests Complete"
echo "============================================"
