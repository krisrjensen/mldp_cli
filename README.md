# MLDP CLI

Master command-line interface for orchestrating all MLDP (Machine Learning Data Processing) tools with both traditional CLI and interactive shell modes.

## Purpose

MLDP CLI provides a unified interface to all existing MLDP tools, including:
- Distance calculations using mpcctl protocol
- Database operations
- Visualization tools
- Experiment management
- Data processing pipelines

## Features

✨ **NEW: Interactive Shell Mode**
- Tab completion for commands and arguments
- Command history with arrow keys
- Persistent session state
- Dynamic prompt showing current settings
- Built-in SQL query interface
- Export results to CSV/JSON

## Installation

```bash
pip install -e .
```

This will install the `mldp` command globally.

## Usage

### Interactive Shell Mode (Recommended)

Launch the interactive MLDP shell for a full CLI experience:

```bash
# Launch interactive shell
mldp

# Or explicitly request shell mode
mldp shell
```

The shell provides a persistent session with:
- **Tab Completion**: Press Tab to complete commands
- **Command History**: Use ↑/↓ arrows for previous commands
- **Session State**: Maintains database connection and settings
- **Dynamic Prompt**: Shows `mldp[exp18:l2]>` with current config

#### Shell Commands
```bash
# Database
mldp> connect                      # Connect to PostgreSQL
mldp> query SELECT * FROM segments LIMIT 10
mldp> tables experiment*           # List matching tables
mldp> stats                        # Show distance statistics

# Experiment Configuration
mldp> set experiment 41            # Set current experiment
mldp> update-decimations 0 7 15    # Update decimation factors
mldp> update-segment-sizes 128 1024 8192  # Update segment sizes
mldp> update-amplitude-methods minmax zscore  # Update amplitude methods
mldp> create-feature-set --name voltage_variance --features voltage,variance(voltage) --n-value 128
mldp> add-feature-set 3            # Add existing feature set
mldp> add-feature-set 1,2,3,4 --n 8192 --channel source_current # Add multiple with options
mldp> list-feature-sets            # List linked feature sets
mldp> show-all-feature-sets        # Show all available feature sets
mldp> remove-feature-set 5         # Remove feature set from experiment
mldp> clear-feature-sets           # Remove all feature sets (with confirmation)
mldp> update-selection-config --max-files 50 --seed 42  # Update selection parameters
mldp> experiment-info 41           # Show detailed experiment information
mldp> experiment-config 41         # Show experiment configuration
mldp> experiment-select 41         # Run segment selection

# Distance Operations
mldp> calculate --segment-size 8192 --distance-type euclidean
mldp> insert_distances --distance-type l2
mldp> closest 20                   # Find 20 closest pairs

# Visualization
mldp> heatmap --version 7
mldp> histogram --bins 100
mldp> visualize --segment-id 12345
mldp> browser                      # Launch database browser

# Settings
mldp> set experiment 18            # Change experiment ID
mldp> set distance l2              # Change distance type
mldp> show                         # Display current settings

# Utilities
mldp> verify                       # Check tool availability
mldp> export results.csv           # Export last query
mldp> clear                        # Clear screen
mldp> help                         # Show all commands
mldp> exit                         # Leave shell
```

### Command-Line Mode

For scripting and automation, use traditional command-line syntax:

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