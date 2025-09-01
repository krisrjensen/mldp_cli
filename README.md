# MLDP CLI

Master command-line interface for orchestrating all MLDP (Machine Learning Data Processing) tools.

## Purpose

MLDP CLI provides a unified interface to all existing MLDP tools, including:
- Distance calculations using mpcctl protocol
- Database operations
- Visualization tools
- Experiment management
- Data processing pipelines

## Installation

```bash
pip install -e .
```

This will install the `mldp` command globally.

## Usage

### List Available Tools
```bash
mldp list-tools
```

### Verify Tool Installation
```bash
mldp verify
```

### Distance Calculations

Calculate distances using the existing mpcctl_distance_calculator:
```bash
# Calculate Euclidean distance
mldp distance calculate --distance-type euclidean --segment-size 8192

# Calculate all distance types
mldp distance calculate --distance-type all

# Insert results into database
mldp distance insert --distance-type euclidean
```

### Visualization

Visualize segments:
```bash
mldp visualize segment --segment-id 12345 --file-id 678
```

Launch database browser:
```bash
mldp database browser
```

### Experiment Management

Generate experiments:
```bash
mldp experiment generate --experiment-id 18
```

## Architecture

MLDP CLI acts as an orchestrator that calls existing MLDP tools:

```
mldp (CLI)
    ├── mpcctl_distance_calculator.py (existing)
    ├── mpcctl_distance_db_insert.py (existing)
    ├── segment_visualizer.py (existing)
    ├── database_browser.py (existing)
    ├── experiment_generator (existing)
    └── other MLDP tools...
```

## Key Features

1. **No Reimplementation**: Uses existing MLDP tools directly
2. **Unified Interface**: Single CLI for all operations
3. **Tool Discovery**: Automatically finds and verifies tool availability
4. **Process Orchestration**: Manages tool execution and pipelines

## Commands

### Distance Operations
- `mldp distance calculate` - Calculate distances using mpcctl
- `mldp distance insert` - Insert distances into PostgreSQL

### Visualization
- `mldp visualize segment` - Visualize segment data
- `mldp database browser` - Launch database browser

### Experiment Management
- `mldp experiment generate` - Generate experiments

### Utility
- `mldp list-tools` - List all available tools
- `mldp verify` - Verify tool installation

## Requirements

- Python 3.9+
- MLDP project must be in the same parent directory
- PostgreSQL database (for distance operations)
- Required MLDP tools installed

## Directory Structure

```
mldp_cli/
├── src/
│   ├── cli.py              # Main CLI interface
│   ├── commands/           # Command implementations
│   ├── orchestrators/      # Tool orchestration logic
│   └── utils/              # Utility functions
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── setup.py               # Package setup
└── README.md              # This file
```

## Contributing

This CLI is designed to orchestrate existing MLDP tools. When adding new functionality:
1. Use existing tools where possible
2. Only add orchestration logic, not reimplementations
3. Maintain compatibility with existing workflows

## License

Part of the MLDP project.