#!/usr/bin/env python3
"""
Filename: cli.py
Author(s): Kristophor Jensen
Date Created: 20250901_230000
Date Revised: 20250901_230000
File version: 0.0.0.1
Description: Main CLI for orchestrating MLDP tools
"""

import click
import subprocess
import sys
import os
from pathlib import Path
import json

# Path to MLDP main project
MLDP_ROOT = Path(__file__).parent.parent.parent / "mldp"


@click.group()
@click.pass_context
def cli(ctx):
    """MLDP Master CLI - Orchestrates all MLDP tools"""
    ctx.ensure_object(dict)
    ctx.obj['mldp_root'] = MLDP_ROOT
    
    # Verify MLDP project exists
    if not MLDP_ROOT.exists():
        click.echo(f"Error: MLDP project not found at {MLDP_ROOT}")
        click.echo("Please ensure mldp_cli is in the same parent directory as mldp")
        sys.exit(1)


@cli.group()
def distance():
    """Distance calculation commands using existing tools"""
    pass


@distance.command()
@click.option('--input-folder', default='/Volumes/ArcData/V3_database/experiment18/segment_files',
              help='Input directory containing segment files')
@click.option('--segment-size', type=int, help='Segment size to process (e.g., 8192)')
@click.option('--distance-type', 
              type=click.Choice(['euclidean', 'cityblock', 'cosine', 'correlation', 
                                'braycurtis', 'canberra', 'chebyshev', 'wasserstein',
                                'kullback_leibler', 'all']),
              default='euclidean',
              help='Distance metric to calculate')
@click.option('--workers', type=int, default=16, help='Number of worker processes')
@click.pass_context
def calculate(ctx, input_folder, segment_size, distance_type, workers):
    """Calculate distances using mpcctl_distance_calculator"""
    
    # Path to existing distance calculator
    calculator_path = ctx.obj['mldp_root'] / "mldp_exp18_distance" / "mpcctl_distance_calculator.py"
    
    if not calculator_path.exists():
        click.echo(f"Error: mpcctl_distance_calculator.py not found at {calculator_path}")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, str(calculator_path)]
    cmd.extend(['--input-folder', input_folder])
    
    if segment_size:
        cmd.extend(['--segment-size', str(segment_size)])
    
    # Add distance type flags
    if distance_type == 'all':
        # Add all distance types
        for dist in ['euclidean', 'cityblock', 'cosine', 'correlation', 
                    'braycurtis', 'canberra', 'chebyshev']:
            cmd.append(f'--{dist}')
    else:
        cmd.append(f'--{distance_type}')
    
    cmd.extend(['--workers', str(workers)])
    
    # Execute the existing tool
    click.echo(f"Running mpcctl_distance_calculator with {distance_type} distance...")
    click.echo(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        click.echo("Error: Distance calculation failed")
        sys.exit(1)
    
    click.echo("Distance calculation complete!")


@distance.command()
@click.option('--input-folder', help='Folder containing .processed directories')
@click.option('--distance-type', help='Distance type to insert')
@click.option('--batch-size', type=int, default=5000, help='Batch size for insertion')
@click.pass_context
def insert(ctx, input_folder, distance_type, batch_size):
    """Insert calculated distances into database using mpcctl_distance_db_insert"""
    
    # Path to existing database insert tool
    insert_path = ctx.obj['mldp_root'] / "mldp_distance_db_insert" / "mpcctl_distance_db_insert.py"
    
    if not insert_path.exists():
        click.echo(f"Error: mpcctl_distance_db_insert.py not found at {insert_path}")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, str(insert_path)]
    
    if input_folder:
        cmd.extend(['--input-folder', input_folder])
    if distance_type:
        cmd.extend(['--distance-type', distance_type])
    if batch_size:
        cmd.extend(['--batch-size', str(batch_size)])
    
    # Execute the existing tool
    click.echo("Running mpcctl_distance_db_insert...")
    click.echo(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        click.echo("Error: Database insertion failed")
        sys.exit(1)
    
    click.echo("Database insertion complete!")


@cli.group()
def visualize():
    """Visualization commands using existing tools"""
    pass


@visualize.command()
@click.option('--segment-id', type=int, help='Segment ID to visualize')
@click.option('--file-id', type=int, help='File ID to visualize')
@click.pass_context
def segment(ctx, segment_id, file_id):
    """Visualize segments using segment_visualizer"""
    
    # Path to existing visualizer
    visualizer_path = ctx.obj['mldp_root'] / "segment_visualizer" / "segment_visualizer.py"
    
    if not visualizer_path.exists():
        click.echo(f"Error: segment_visualizer.py not found at {visualizer_path}")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, str(visualizer_path)]
    
    if segment_id:
        cmd.extend(['--segment-id', str(segment_id)])
    if file_id:
        cmd.extend(['--file-id', str(file_id)])
    
    # Execute the existing tool
    click.echo("Running segment_visualizer...")
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        click.echo("Error: Visualization failed")
        sys.exit(1)


@cli.group()
def database():
    """Database operations using existing tools"""
    pass


@database.command()
@click.pass_context
def browser(ctx):
    """Launch database browser"""
    
    # Path to existing database browser
    browser_path = ctx.obj['mldp_root'] / "database_browser" / "database_browser.py"
    
    if not browser_path.exists():
        click.echo(f"Error: database_browser.py not found at {browser_path}")
        sys.exit(1)
    
    # Execute the existing tool
    click.echo("Launching database browser...")
    subprocess.run([sys.executable, str(browser_path)])


@cli.group()
def experiment():
    """Experiment generation and management"""
    pass


@experiment.command()
@click.option('--experiment-id', type=int, default=18, help='Experiment ID')
@click.option('--config-file', help='Configuration file for experiment')
@click.pass_context
def generate(ctx, experiment_id, config_file):
    """Generate experiment using experiment_generator"""
    
    # Path to experiment generator
    generator_path = ctx.obj['mldp_root'] / "experiment_generator"
    
    if not generator_path.exists():
        click.echo(f"Error: experiment_generator not found at {generator_path}")
        sys.exit(1)
    
    # Find the main script
    possible_scripts = list(generator_path.glob("*generate*.py"))
    if not possible_scripts:
        click.echo("Error: No experiment generator script found")
        sys.exit(1)
    
    generator_script = possible_scripts[0]
    
    # Build command
    cmd = [sys.executable, str(generator_script)]
    cmd.extend(['--experiment-id', str(experiment_id)])
    
    if config_file:
        cmd.extend(['--config', config_file])
    
    # Execute
    click.echo(f"Running experiment generator for experiment {experiment_id}...")
    subprocess.run(cmd)


@cli.command()
@click.pass_context
def list_tools(ctx):
    """List all available MLDP tools"""
    
    click.echo("Available MLDP Tools:")
    click.echo("=" * 60)
    
    tools = {
        "Distance Calculation": [
            "mpcctl_distance_calculator.py - Calculate distances with MPCCTL protocol",
            "mpcctl_distance_db_insert.py - Insert distances into PostgreSQL"
        ],
        "Visualization": [
            "segment_visualizer - Visualize segment data",
            "distance_visualizer - Visualize distance matrices",
            "database_browser - Browse database tables"
        ],
        "Data Processing": [
            "segment_verifier - Verify segment integrity",
            "data_cleaning_tool - Clean and preprocess data",
            "real_time_sync_hub - Real-time data synchronization"
        ],
        "Experiment Management": [
            "experiment_generator - Generate experiments",
            "ml_code - Machine learning pipelines"
        ]
    }
    
    for category, tool_list in tools.items():
        click.echo(f"\n{category}:")
        for tool in tool_list:
            click.echo(f"  • {tool}")
    
    click.echo("\n" + "=" * 60)
    click.echo("Use 'mldp <category> <command> --help' for more information")


@cli.command()
def verify():
    """Verify all MLDP tools are accessible"""
    
    click.echo("Verifying MLDP tools...")
    
    tools_to_check = [
        ("mldp_exp18_distance/mpcctl_distance_calculator.py", "Distance Calculator"),
        ("mldp_distance_db_insert/mpcctl_distance_db_insert.py", "Distance DB Insert"),
        ("segment_visualizer/segment_visualizer.py", "Segment Visualizer"),
        ("database_browser/database_browser.py", "Database Browser"),
        ("experiment_generator", "Experiment Generator"),
        ("segment_verifier", "Segment Verifier"),
        ("data_cleaning_tool", "Data Cleaning Tool")
    ]
    
    all_found = True
    for tool_path, tool_name in tools_to_check:
        full_path = MLDP_ROOT / tool_path
        if full_path.exists():
            click.echo(f"✅ {tool_name}: {full_path}")
        else:
            click.echo(f"❌ {tool_name}: Not found at {full_path}")
            all_found = False
    
    if all_found:
        click.echo("\n✅ All tools verified successfully!")
    else:
        click.echo("\n⚠️  Some tools are missing. Please check your MLDP installation.")


def main():
    """Main entry point"""
    cli(obj={})


if __name__ == '__main__':
    main()