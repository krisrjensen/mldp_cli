#!/usr/bin/env python3
"""
Filename: script_parser.py
Author(s): Kristophor Jensen
Date Created: 20251104_000000
Date Revised: 20251104_000000
File version: 1.0.0.0
Description: Bash-style script parser for MLDP shell

Supports:
- if/then/else/fi conditional blocks
- Bash-style test expressions [ condition ]
- Nested conditionals
- Block-based execution structures

Test Expression Syntax:
- String: [ "$VAR" = "value" ], [ "$VAR" != "value" ]
- Numeric: [ $VAR -eq 0 ], [ $VAR -ne 0 ], [ $VAR -gt 0 ], [ $VAR -lt 0 ]
- File: [ -f path ], [ -d path ], [ -e path ]
- Logical: [ cond ] && [ cond ], [ cond ] || [ cond ], ! [ cond ]
- String tests: [ -z "$VAR" ], [ -n "$VAR" ]
"""

import re
import os
from typing import List, Dict, Optional, Tuple


class ExecutionBlock:
    """
    Represents a block of commands with optional conditional execution.

    Blocks can be:
    - sequential: Simple list of commands to execute
    - if: Commands executed if condition is true
    - else: Commands executed if if-condition was false
    """

    def __init__(self, block_type: str, lines: List[str], condition: str = None):
        """
        Initialize an execution block.

        Args:
            block_type: Type of block ('sequential', 'if', 'else')
            lines: List of command lines in this block
            condition: Condition string for if blocks (e.g., "[ $? -eq 0 ]")
        """
        self.type = block_type
        self.lines = lines
        self.condition = condition
        self.else_block = None  # ExecutionBlock for else clause
        self.nested_blocks = []  # List of nested ExecutionBlock objects

    def __repr__(self):
        return f"ExecutionBlock(type={self.type}, lines={len(self.lines)}, condition={self.condition})"


class ScriptParser:
    """
    Parser for bash-style scripts with if/then/else/fi support.

    Parses script lines into executable blocks with conditional logic.
    """

    def __init__(self, shell_instance):
        """
        Initialize the script parser.

        Args:
            shell_instance: Reference to MLDPShell instance for variable access
        """
        self.shell = shell_instance
        self.lines = []
        self.current_line = 0

    def parse_script(self, lines: List[str]) -> List[ExecutionBlock]:
        """
        Parse script lines into execution blocks.

        Args:
            lines: List of script lines (including comments and blanks)

        Returns:
            List of ExecutionBlock objects ready for execution
        """
        self.lines = lines
        self.current_line = 0
        blocks = []

        while self.current_line < len(self.lines):
            block = self._parse_next_block()
            if block:
                blocks.append(block)

        return blocks

    def _parse_next_block(self) -> Optional[ExecutionBlock]:
        """
        Parse the next block starting from current_line.

        Returns:
            ExecutionBlock or None if end of script
        """
        # Skip blank lines and comments
        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].strip()
            if line and not line.startswith('#'):
                break
            self.current_line += 1

        if self.current_line >= len(self.lines):
            return None

        line = self.lines[self.current_line].strip()

        # Check if this is an if statement
        if line.startswith('if '):
            return self._parse_if_block()
        else:
            # Sequential block - collect non-control-flow lines
            return self._parse_sequential_block()

    def _parse_if_block(self) -> ExecutionBlock:
        """
        Parse an if/then/else/fi block.

        Returns:
            ExecutionBlock with type='if' and optional else_block
        """
        if_line_num = self.current_line
        if_line = self.lines[self.current_line].strip()

        # Extract condition from "if [ condition ]; then" or "if [ condition ]"
        condition = self._extract_condition(if_line)

        # Skip the "if" line and "then" line (if separate)
        self.current_line += 1
        if self.current_line < len(self.lines):
            next_line = self.lines[self.current_line].strip()
            if next_line == 'then':
                self.current_line += 1

        # Find matching fi
        fi_line_num = self._find_matching_fi(if_line_num)
        if fi_line_num == -1:
            raise SyntaxError(f"No matching 'fi' found for 'if' at line {if_line_num + 1}")

        # Find else clause (if any)
        else_line_num = self._find_else(if_line_num, fi_line_num)

        # Extract if-block lines
        if else_line_num != -1:
            if_block_lines = self.lines[self.current_line:else_line_num]
        else:
            if_block_lines = self.lines[self.current_line:fi_line_num]

        # Create if block
        if_block = ExecutionBlock('if', if_block_lines, condition)

        # Parse else block if present
        if else_line_num != -1:
            # Skip 'else' line
            else_start = else_line_num + 1
            else_block_lines = self.lines[else_start:fi_line_num]
            if_block.else_block = ExecutionBlock('else', else_block_lines)

        # Move past fi
        self.current_line = fi_line_num + 1

        return if_block

    def _parse_sequential_block(self) -> ExecutionBlock:
        """
        Parse a sequential block of commands (no control flow).

        Returns:
            ExecutionBlock with type='sequential'
        """
        lines = []

        while self.current_line < len(self.lines):
            line = self.lines[self.current_line].strip()

            # Stop at control flow keywords
            if line.startswith('if ') or line in ('else', 'fi', 'then'):
                break

            # Skip blank lines and comments
            if not line or line.startswith('#'):
                self.current_line += 1
                continue

            lines.append(self.lines[self.current_line])
            self.current_line += 1

        return ExecutionBlock('sequential', lines) if lines else None

    def _extract_condition(self, if_line: str) -> str:
        """
        Extract condition from if statement.

        Args:
            if_line: Line like "if [ $? -eq 0 ]; then" or "if [ condition ]"

        Returns:
            Condition string like "[ $? -eq 0 ]"
        """
        # Remove 'if ' prefix
        if_line = if_line[3:].strip()

        # Remove '; then' suffix if present
        if '; then' in if_line:
            if_line = if_line.split('; then')[0].strip()

        # Remove ' then' suffix if present
        if if_line.endswith(' then'):
            if_line = if_line[:-5].strip()

        return if_line

    def _find_matching_fi(self, if_line_num: int) -> int:
        """
        Find the matching 'fi' for an 'if' statement.

        Handles nested if/fi blocks correctly.

        Args:
            if_line_num: Line number of the 'if' statement

        Returns:
            Line number of matching 'fi', or -1 if not found
        """
        depth = 1
        line_num = if_line_num + 1

        while line_num < len(self.lines):
            line = self.lines[line_num].strip()

            if line.startswith('if '):
                depth += 1
            elif line == 'fi':
                depth -= 1
                if depth == 0:
                    return line_num

            line_num += 1

        return -1

    def _find_else(self, if_line_num: int, fi_line_num: int) -> int:
        """
        Find 'else' clause between if and fi.

        Args:
            if_line_num: Line number of 'if'
            fi_line_num: Line number of matching 'fi'

        Returns:
            Line number of 'else', or -1 if no else clause
        """
        depth = 1
        line_num = if_line_num + 1

        while line_num < fi_line_num:
            line = self.lines[line_num].strip()

            if line.startswith('if '):
                depth += 1
            elif line == 'fi':
                depth -= 1
            elif line == 'else' and depth == 1:
                return line_num

            line_num += 1

        return -1

    def evaluate_condition(self, condition: str) -> bool:
        """
        Evaluate a bash-style test condition.

        Supports:
        - [ "$VAR" = "value" ] - string equality
        - [ "$VAR" != "value" ] - string inequality
        - [ $VAR -eq N ] - numeric equality
        - [ $VAR -ne N ] - numeric inequality
        - [ $VAR -gt N ] - greater than
        - [ $VAR -lt N ] - less than
        - [ $VAR -ge N ] - greater or equal
        - [ $VAR -le N ] - less or equal
        - [ -z "$VAR" ] - string is empty
        - [ -n "$VAR" ] - string is not empty
        - [ -f path ] - file exists
        - [ -d path ] - directory exists
        - [ -e path ] - path exists

        Args:
            condition: Condition string like "[ $? -eq 0 ]"

        Returns:
            True if condition is true, False otherwise
        """
        condition = condition.strip()

        # Remove outer [ ] brackets
        if condition.startswith('[') and condition.endswith(']'):
            condition = condition[1:-1].strip()

        # Handle negation: ! [ condition ]
        if condition.startswith('!'):
            inner_condition = condition[1:].strip()
            return not self.evaluate_condition(f"[ {inner_condition} ]")

        # Handle logical AND: [ cond1 ] && [ cond2 ]
        if ' && ' in condition:
            parts = condition.split(' && ')
            return all(self.evaluate_condition(f"[ {p.strip()} ]") for p in parts)

        # Handle logical OR: [ cond1 ] || [ cond2 ]
        if ' || ' in condition:
            parts = condition.split(' || ')
            return any(self.evaluate_condition(f"[ {p.strip()} ]") for p in parts)

        # String tests
        if condition.startswith('-z '):
            # String is empty
            var = condition[3:].strip().strip('"\'')
            return len(var) == 0

        if condition.startswith('-n '):
            # String is not empty
            var = condition[3:].strip().strip('"\'')
            return len(var) > 0

        # File tests
        if condition.startswith('-f '):
            # File exists
            path = condition[3:].strip().strip('"\'')
            return os.path.isfile(path)

        if condition.startswith('-d '):
            # Directory exists
            path = condition[3:].strip().strip('"\'')
            return os.path.isdir(path)

        if condition.startswith('-e '):
            # Path exists
            path = condition[3:].strip().strip('"\'')
            return os.path.exists(path)

        # String equality/inequality: "$VAR" = "value" or "$VAR" != "value"
        for op in [' = ', ' != ']:
            if op in condition:
                left, right = condition.split(op, 1)
                left = left.strip().strip('"\'')
                right = right.strip().strip('"\'')

                if op == ' = ':
                    return left == right
                else:  # !=
                    return left != right

        # Numeric comparisons: $VAR -eq N, etc.
        numeric_ops = {
            ' -eq ': lambda a, b: a == b,
            ' -ne ': lambda a, b: a != b,
            ' -gt ': lambda a, b: a > b,
            ' -lt ': lambda a, b: a < b,
            ' -ge ': lambda a, b: a >= b,
            ' -le ': lambda a, b: a <= b,
        }

        for op, func in numeric_ops.items():
            if op in condition:
                left, right = condition.split(op, 1)
                try:
                    left_val = int(left.strip())
                    right_val = int(right.strip())
                    return func(left_val, right_val)
                except ValueError:
                    # Not valid integers
                    return False

        # Default: if non-empty string, return True
        return len(condition.strip()) > 0


# Example usage and testing
if __name__ == '__main__':
    # Mock shell instance for testing
    class MockShell:
        def __init__(self):
            self.script_variables = {'TEST': 'value', 'COUNT': '42'}
            self.last_exit_code = 0

    parser = ScriptParser(MockShell())

    # Test condition evaluation
    print("Testing condition evaluation:")
    print(f"  [ 0 -eq 0 ] = {parser.evaluate_condition('[ 0 -eq 0 ]')}")  # True
    print(f"  [ 1 -eq 0 ] = {parser.evaluate_condition('[ 1 -eq 0 ]')}")  # False
    cond1 = '[ "a" = "a" ]'
    print(f'  [ "a" = "a" ] = {parser.evaluate_condition(cond1)}')  # True
    cond2 = '[ "a" != "b" ]'
    print(f'  [ "a" != "b" ] = {parser.evaluate_condition(cond2)}')  # True
    print(f"  [ 5 -gt 3 ] = {parser.evaluate_condition('[ 5 -gt 3 ]')}")  # True
    cond3 = '[ -z "" ]'
    print(f'  [ -z "" ] = {parser.evaluate_condition(cond3)}')  # True
    cond4 = '[ -n "text" ]'
    print(f'  [ -n "text" ] = {parser.evaluate_condition(cond4)}')  # True

    # Test script parsing
    print("\nTesting script parsing:")
    script_lines = [
        'echo "Start"',
        'if [ $? -eq 0 ]; then',
        '    echo "Success"',
        'else',
        '    echo "Failed"',
        'fi',
        'echo "End"',
    ]

    blocks = parser.parse_script(script_lines)
    for i, block in enumerate(blocks):
        print(f"  Block {i}: {block}")
