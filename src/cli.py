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

# Path to MLDP main project (mldp_cli is now a submodule inside mldp)
MLDP_ROOT = Path(__file__).parent.parent.parent


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


@visualize.command()
@click.option('--experiment-id', type=int, default=18, help='Experiment ID')
@click.option('--distance-type', type=click.Choice(['l1', 'l2', 'cosine', 'pearson', 'all']), 
              default='l2', help='Distance type for heatmap')
@click.option('--output-dir', help='Output directory for heatmaps')
@click.option('--segment-size', type=int, help='Segment size to visualize')
@click.option('--version', type=int, default=7, help='Heatmap generator version (1-7)')
@click.pass_context
def heatmap(ctx, experiment_id, distance_type, output_dir, segment_size, version):
    """Generate distance heatmaps using existing tools"""
    
    # Path to heatmap generator
    heatmap_path = ctx.obj['mldp_root'] / "experiment_generator" / "src" / "heatmaps" / f"generate_exp18_heatmaps_v{version}.py"
    
    if not heatmap_path.exists():
        # Try the general heatmap generator
        heatmap_path = ctx.obj['mldp_root'] / "experiment_generator" / "src" / "heatmaps" / "generate_exp18_heatmaps.py"
    
    if not heatmap_path.exists():
        click.echo(f"Error: Heatmap generator not found")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, str(heatmap_path)]
    
    if distance_type != 'all':
        cmd.extend(['--distance-type', distance_type])
    
    if output_dir:
        cmd.extend(['--output-dir', output_dir])
    
    if segment_size:
        cmd.extend(['--segment-size', str(segment_size)])
    
    # Execute
    click.echo(f"Generating {distance_type} heatmap using v{version} generator...")
    subprocess.run(cmd, capture_output=False)


@visualize.command()
@click.option('--experiment-id', type=int, default=18, help='Experiment ID')
@click.option('--distance-type', type=click.Choice(['l1', 'l2', 'cosine', 'pearson']), 
              default='l2', help='Distance type')
@click.option('--output-dir', help='Output directory for histograms')
@click.option('--bins', type=int, default=50, help='Number of histogram bins')
@click.option('--version', type=str, default='1_3', help='Histogram version (1_0, 1_1, 1_2, 1_3)')
@click.pass_context
def histogram(ctx, experiment_id, distance_type, output_dir, bins, version):
    """Generate distance histograms using existing tools"""
    
    # Path to histogram generator
    histogram_path = ctx.obj['mldp_root'] / "experiment_generator" / "src" / "heatmaps" / f"histogram_plot_generator_v{version}.py"
    
    if not histogram_path.exists():
        # Try simple histogram generator
        histogram_path = ctx.obj['mldp_root'] / "experiment_generator" / "src" / "heatmaps" / "simple_histogram_generator.py"
    
    if not histogram_path.exists():
        click.echo(f"Error: Histogram generator not found")
        sys.exit(1)
    
    # Build command
    cmd = [sys.executable, str(histogram_path)]
    cmd.extend(['--distance-type', distance_type])
    
    if output_dir:
        cmd.extend(['--output-dir', output_dir])
    
    cmd.extend(['--bins', str(bins)])
    
    # Execute
    click.echo(f"Generating {distance_type} histogram...")
    subprocess.run(cmd, capture_output=False)


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


@database.command()
@click.option('--table', help='Table name to query')
@click.option('--limit', type=int, default=10, help='Number of rows to return')
@click.option('--where', help='WHERE clause for filtering')
@click.option('--format', type=click.Choice(['table', 'json', 'csv']), default='table')
def query(table, limit, where, format):
    """Query database tables directly"""
    import psycopg2
    from tabulate import tabulate
    
    # Connect to database
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )
    
    cursor = conn.cursor()
    
    # Build query
    query = f"SELECT * FROM {table}"
    if where:
        query += f" WHERE {where}"
    query += f" LIMIT {limit}"
    
    try:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        if format == 'table':
            click.echo(tabulate(rows, headers=columns, tablefmt='grid'))
        elif format == 'json':
            import json
            data = [dict(zip(columns, row)) for row in rows]
            click.echo(json.dumps(data, indent=2, default=str))
        elif format == 'csv':
            import csv
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(columns)
            writer.writerows(rows)
            click.echo(output.getvalue())
            
    except Exception as e:
        click.echo(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()


@database.command()
@click.option('--experiment-id', type=int, default=18, help='Experiment ID')
@click.option('--distance-type', default='l2', help='Distance type')
def stats(experiment_id, distance_type):
    """Show database statistics for distances"""
    import psycopg2
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )
    
    cursor = conn.cursor()
    
    # Get table name
    table_name = f"experiment_{experiment_id:03d}_distance_{distance_type.lower()}"
    
    try:
        # Count records
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        
        # Get min/max/avg distances
        cursor.execute(f"""
            SELECT 
                MIN(distance_s) as min_dist,
                MAX(distance_s) as max_dist,
                AVG(distance_s) as avg_dist,
                STDDEV(distance_s) as std_dist
            FROM {table_name}
        """)
        
        stats = cursor.fetchone()
        
        click.echo(f"\nStatistics for {table_name}:")
        click.echo(f"  Total records: {count:,}")
        click.echo(f"  Min distance: {stats[0]:.6f}")
        click.echo(f"  Max distance: {stats[1]:.6f}")
        click.echo(f"  Avg distance: {stats[2]:.6f}")
        click.echo(f"  Std deviation: {stats[3]:.6f}")
        
    except Exception as e:
        click.echo(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()


@cli.group()
def analyze():
    """Analysis commands for distance data"""
    pass


@analyze.command()
@click.option('--experiment-id', type=int, default=18, help='Experiment ID')
@click.option('--distance-type', default='l2', help='Distance type')
@click.option('--top-n', type=int, default=10, help='Number of closest pairs to show')
def closest_pairs(experiment_id, distance_type, top_n):
    """Find closest segment pairs by distance"""
    import psycopg2
    from tabulate import tabulate
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )
    
    cursor = conn.cursor()
    table_name = f"experiment_{experiment_id:03d}_distance_{distance_type.lower()}"
    
    try:
        # Find closest pairs
        cursor.execute(f"""
            SELECT 
                segment_id_1,
                segment_id_2,
                distance_s,
                file_id_1,
                file_id_2
            FROM {table_name}
            ORDER BY distance_s ASC
            LIMIT {top_n}
        """)
        
        rows = cursor.fetchall()
        headers = ['Segment 1', 'Segment 2', 'Distance', 'File 1', 'File 2']
        
        click.echo(f"\nTop {top_n} closest segment pairs ({distance_type} distance):")
        click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
        
    except Exception as e:
        click.echo(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()


@analyze.command()
@click.option('--experiment-id', type=int, default=18, help='Experiment ID')
@click.option('--distance-type', default='l2', help='Distance type')
@click.option('--file-id', type=int, help='File ID to analyze')
def file_distances(experiment_id, distance_type, file_id):
    """Analyze distances for a specific file"""
    import psycopg2
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )
    
    cursor = conn.cursor()
    table_name = f"experiment_{experiment_id:03d}_distance_{distance_type.lower()}"
    
    try:
        # Get statistics for file
        cursor.execute(f"""
            SELECT 
                COUNT(*) as count,
                MIN(distance_s) as min_dist,
                MAX(distance_s) as max_dist,
                AVG(distance_s) as avg_dist,
                STDDEV(distance_s) as std_dist
            FROM {table_name}
            WHERE file_id_1 = {file_id} OR file_id_2 = {file_id}
        """)
        
        stats = cursor.fetchone()
        
        click.echo(f"\nDistance statistics for file {file_id}:")
        click.echo(f"  Total pairs: {stats[0]:,}")
        click.echo(f"  Min distance: {stats[1]:.6f}")
        click.echo(f"  Max distance: {stats[2]:.6f}")
        click.echo(f"  Avg distance: {stats[3]:.6f}")
        click.echo(f"  Std deviation: {stats[4]:.6f}")
        
    except Exception as e:
        click.echo(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()


@analyze.command()
@click.option('--segment-size', type=int, help='Segment size to analyze')
@click.option('--output-format', type=click.Choice(['table', 'json']), default='table')
def segment_distribution(segment_size, output_format):
    """Analyze segment distribution across files"""
    import psycopg2
    from tabulate import tabulate
    import json
    
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )
    
    cursor = conn.cursor()
    
    try:
        # Get segment distribution
        query = """
            SELECT 
                file_id,
                COUNT(*) as segment_count,
                MIN(segment_id) as min_segment,
                MAX(segment_id) as max_segment
            FROM segments
        """
        
        if segment_size:
            query += f" WHERE segment_size = {segment_size}"
        
        query += " GROUP BY file_id ORDER BY file_id"
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if output_format == 'table':
            headers = ['File ID', 'Segments', 'Min ID', 'Max ID']
            click.echo("\nSegment distribution by file:")
            click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
        else:
            data = [{'file_id': r[0], 'segment_count': r[1], 
                    'min_segment': r[2], 'max_segment': r[3]} for r in rows]
            click.echo(json.dumps(data, indent=2))
            
    except Exception as e:
        click.echo(f"Error: {e}")
    finally:
        cursor.close()
        conn.close()


@cli.group()
def servers():
    """Server management (start, stop, status, restart)"""
    pass


@servers.command()
@click.pass_context
def start(ctx):
    """Start all MLDP servers"""
    scripts_path = ctx.obj['mldp_root'] / "scripts" / "start_services.sh"
    
    if not scripts_path.exists():
        click.echo(f"Error: start_services.sh not found at {scripts_path}")
        sys.exit(1)
    
    click.echo("Starting all MLDP servers...")
    click.echo("=" * 60)
    
    try:
        result = subprocess.run(
            ["bash", str(scripts_path)],
            capture_output=False,
            text=True,
            cwd=str(ctx.obj['mldp_root'])
        )
        if result.returncode == 0:
            click.echo("\n✅ All servers started successfully!")
        else:
            click.echo("\n⚠️  Some servers may have failed to start")
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error starting servers: {e}")
        sys.exit(1)


@servers.command()
@click.pass_context
def stop(ctx):
    """Stop all MLDP servers"""
    scripts_path = ctx.obj['mldp_root'] / "scripts" / "stop_services.sh"
    
    if not scripts_path.exists():
        click.echo(f"Error: stop_services.sh not found at {scripts_path}")
        sys.exit(1)
    
    click.echo("Stopping all MLDP servers...")
    
    try:
        result = subprocess.run(
            ["bash", str(scripts_path)],
            capture_output=True,
            text=True,
            cwd=str(ctx.obj['mldp_root'])
        )
        click.echo(result.stdout)
        if result.returncode == 0:
            click.echo("✅ All servers stopped successfully!")
        else:
            click.echo("⚠️  Some servers may still be running")
    except Exception as e:
        click.echo(f"Error stopping servers: {e}")
        sys.exit(1)


@servers.command()
@click.pass_context
def status(ctx):
    """Check status of all MLDP servers"""
    operation_pid_path = ctx.obj['mldp_root'] / "operation" / "pid"
    
    services = [
        ("real_time_sync_hub", 5035, "Real-Time Sync Hub"),
        ("database_browser", 5020, "Database Browser"),
        ("data_cleaning_tool", 5030, "Data Cleaning Tool"),
        ("transient_viewer", 5031, "Transient Viewer"),
        ("segment_visualizer", 5032, "Segment Visualizer"),
        ("distance_visualizer", 5037, "Distance Visualizer"),
        ("experiment_generator", 5040, "ML Experiment Generator"),
        ("jupyter_integration", 5041, "Jupyter Integration"),
        ("segment_verifier", 5034, "Segment Verifier"),
    ]
    
    click.echo("\n📊 MLDP Server Status")
    click.echo("=" * 60)
    click.echo(f"{'Service':<30} {'Port':<8} {'PID':<10} {'Status'}")
    click.echo("-" * 60)
    
    running_count = 0
    total_count = len(services)
    
    for service_name, port, display_name in services:
        pid_file = operation_pid_path / f"{service_name}.pid"
        
        status = "❓ Unknown"
        pid = "-"
        
        if pid_file.exists():
            try:
                with open(pid_file, 'r') as f:
                    pid = f.read().strip()
                
                # Check if process is running
                import psutil
                try:
                    process = psutil.Process(int(pid))
                    if process.is_running():
                        status = "✅ Running"
                        running_count += 1
                    else:
                        status = "❌ Not Running"
                except (psutil.NoSuchProcess, ValueError):
                    status = "❌ Stale PID"
            except Exception:
                status = "❌ Error"
        else:
            status = "⏹️  Stopped"
        
        click.echo(f"{display_name:<30} {port:<8} {pid:<10} {status}")
    
    click.echo("-" * 60)
    click.echo(f"Summary: {running_count}/{total_count} services running")
    
    if running_count == total_count:
        click.echo("\n🎉 All services are running!")
    elif running_count == 0:
        click.echo("\n⚠️  No services are running. Use 'mldp servers start' to start them.")
    else:
        click.echo(f"\n⚠️  Only {running_count}/{total_count} services are running.")


@servers.command()
@click.pass_context
def restart(ctx):
    """Restart all MLDP servers"""
    click.echo("Restarting all MLDP servers...")
    click.echo("=" * 60)
    
    # Stop servers
    ctx.invoke(stop)
    
    # Wait a moment
    import time
    click.echo("\n⏳ Waiting for services to shut down...")
    time.sleep(3)
    
    # Start servers
    ctx.invoke(start)


@servers.command()
@click.option('--service', help='Specific service to show logs for')
@click.option('--lines', default=50, help='Number of lines to show')
@click.pass_context
def logs(ctx, service, lines):
    """View server logs"""
    logs_path = ctx.obj['mldp_root'] / "operation" / "logs"
    
    if service:
        log_file = logs_path / f"{service}.log"
        if log_file.exists():
            click.echo(f"\n📋 Last {lines} lines of {service}.log:")
            click.echo("=" * 60)
            result = subprocess.run(
                ["tail", f"-{lines}", str(log_file)],
                capture_output=True,
                text=True
            )
            click.echo(result.stdout)
        else:
            click.echo(f"Log file not found: {log_file}")
    else:
        # Show available log files
        click.echo("\n📁 Available log files:")
        click.echo("=" * 60)
        if logs_path.exists():
            for log_file in sorted(logs_path.glob("*.log")):
                size = log_file.stat().st_size
                size_str = f"{size / 1024:.1f}K" if size < 1024*1024 else f"{size / (1024*1024):.1f}M"
                click.echo(f"  {log_file.name:<40} {size_str:>10}")
            click.echo("\nUse 'mldp servers logs --service <name>' to view specific logs")
        else:
            click.echo("No logs directory found")


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
    """List all available MLDP tools and CLI commands"""
    
    click.echo("MLDP CLI - Available Commands and Tools:")
    click.echo("=" * 60)
    
    # CLI commands
    cli_commands = {
        "Distance Operations (mldp distance)": [
            "calculate - Calculate distances using mpcctl_distance_calculator",
            "insert - Insert calculated distances into PostgreSQL database"
        ],
        "Visualization (mldp visualize)": [
            "segment - Visualize segment data with segment_visualizer",
            "heatmap - Generate distance heatmaps (multiple versions)",
            "histogram - Generate distance histograms with various styles"
        ],
        "Database Operations (mldp database)": [
            "browser - Launch interactive database browser",
            "query - Query database tables directly with SQL",
            "stats - Show statistics for distance tables"
        ],
        "Analysis (mldp analyze)": [
            "closest-pairs - Find closest segment pairs by distance",
            "file-distances - Analyze distances for specific files",
            "segment-distribution - Analyze segment distribution across files"
        ],
        "Experiment Management (mldp experiment)": [
            "generate - Generate experiments using experiment_generator"
        ],
        "Utility Commands": [
            "verify - Verify all MLDP tools are accessible",
            "list-tools - Show this help message"
        ]
    }
    
    # Integrated tools
    integrated_tools = {
        "Distance Calculation Tools": [
            "mpcctl_distance_calculator.py - Parallel distance calculator",
            "mpcctl_distance_db_insert.py - Batch database insertion"
        ],
        "Visualization Tools": [
            "segment_visualizer.py - Interactive segment plots",
            "database_browser.py - Database table browser",
            "generate_exp18_heatmaps_v[1-7].py - Distance heatmaps",
            "histogram_plot_generator_v1_[0-3].py - Histograms",
            "simple_histogram_generator.py - Basic histograms"
        ],
        "Data Processing Tools": [
            "segment_verifier - Segment integrity verification",
            "data_cleaning_tool - Data preprocessing",
            "real_time_sync_hub - Real-time synchronization"
        ],
        "Experiment Tools": [
            "experiment_generator - Experiment configuration",
            "ml_code - Machine learning pipelines"
        ]
    }
    
    # Display CLI commands
    click.echo("\n=== CLI Commands ===")
    for category, commands in cli_commands.items():
        click.echo(f"\n{category}:")
        for cmd in commands:
            click.echo(f"  • {cmd}")
    
    # Display integrated tools
    click.echo("\n\n=== Integrated MLDP Tools ===")
    for category, tool_list in integrated_tools.items():
        click.echo(f"\n{category}:")
        for tool in tool_list:
            click.echo(f"  • {tool}")
    
    click.echo("\n" + "=" * 60)
    click.echo("Use 'mldp <group> <command> --help' for detailed help")
    click.echo("Example: mldp distance calculate --help")


@cli.group(name='segment-fileset')
def segment_fileset():
    """Generate and manage segment filesets with decimation and data types"""
    pass


@segment_fileset.command()
@click.option('--experiment-id', type=int, default=18, help='Experiment ID')
@click.option('--segment-size', type=int, default=262144, help='Segment size (must be 2^N)')
@click.option('--decimation', type=int, multiple=True, default=[0], 
              help='Decimation factors (0=none, or use 2^n-1 values like 1,3,7,15,31,63,127,255...)')
@click.option('--data-type', type=click.Choice(['RAW', 'ADC8', 'ADC6', 'ADC4', 'INT8', 'INT16']), 
              multiple=True, default=['RAW'], help='Data types to generate')
@click.option('--file-range', help='File ID range from experiment_018_file_training_data (e.g., 200-500)')
@click.option('--workers', type=int, default=16, help='Number of parallel workers')
@click.option('--output-path', default='/Volumes/ArcData/V3_database/experiment18/segment_files',
              help='Output directory for segment files (experiment-specific)')
@click.pass_context
def generate(ctx, experiment_id, segment_size, decimation, data_type, file_range, workers, output_path):
    """Generate segment fileset from experiment files with 2^N preservation"""
    import math
    from pathlib import Path
    import psycopg2
    
    # Validate segment size is 2^N
    if segment_size & (segment_size - 1) != 0:
        click.echo(f"Error: Segment size {segment_size} is not a power of 2")
        sys.exit(1)
    
    # Validate decimation factors preserve 2^N segments
    for dec_factor in decimation:
        if dec_factor > 0:
            # For 2^n-1 decimation (e.g., keep every nth sample)
            decimated_size = segment_size // (dec_factor + 1)
            if decimated_size * (dec_factor + 1) != segment_size:
                click.echo(f"Warning: Decimation factor {dec_factor} doesn't evenly divide {segment_size}")
                click.echo(f"  Would produce {decimated_size} samples, losing {segment_size - decimated_size * (dec_factor + 1)} samples")
                # Check if result is still power of 2
                if decimated_size & (decimated_size - 1) != 0:
                    click.echo(f"  ERROR: Result {decimated_size} is not a power of 2!")
                    sys.exit(1)
    
    click.echo(f"Generating segment fileset for experiment {experiment_id}")
    click.echo(f"  Segment size: {segment_size} (2^{int(math.log2(segment_size))})")
    click.echo(f"  Decimation factors: {list(decimation)}")
    click.echo(f"  Data types: {list(data_type)}")
    click.echo(f"  Output path: {output_path}")
    
    # Get files from experiment_018_file_training_data
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )
    cursor = conn.cursor()
    
    try:
        # Build query for files
        if file_range:
            parts = file_range.split('-')
            if len(parts) == 2:
                min_id, max_id = int(parts[0]), int(parts[1])
                cursor.execute("""
                    SELECT DISTINCT file_id, assigned_label 
                    FROM experiment_018_file_training_data 
                    WHERE experiment_id = %s AND file_id BETWEEN %s AND %s
                    ORDER BY file_id
                """, (experiment_id, min_id, max_id))
            else:
                click.echo(f"Invalid file range format: {file_range}")
                sys.exit(1)
        else:
            cursor.execute("""
                SELECT DISTINCT file_id, assigned_label 
                FROM experiment_018_file_training_data 
                WHERE experiment_id = %s
                ORDER BY file_id
            """, (experiment_id,))
        
        files = cursor.fetchall()
        click.echo(f"\nFound {len(files)} files in experiment {experiment_id}")
        
        if len(files) > 0:
            click.echo("\nSample files:")
            for file_id, label in files[:5]:
                click.echo(f"  File {file_id}: {label}")
            
            # Check segments for these files
            cursor.execute("""
                SELECT COUNT(DISTINCT segment_id) 
                FROM experiment_018_segment_training_data 
                WHERE experiment_id = %s
            """, (experiment_id,))
            segment_count = cursor.fetchone()[0]
            click.echo(f"\nTotal segments in experiment: {segment_count}")
            
    except Exception as e:
        click.echo(f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()
    
    click.echo("\n⚠️  Full segment generation implementation pending")
    click.echo("Next steps:")
    click.echo("  1. Load raw files from /Volumes/ArcData/V3_database/fileset/")
    click.echo("  2. Create segments matching experiment_018_segment_training_data")
    click.echo("  3. Apply decimation preserving 2^N samples")
    click.echo("  4. Apply data type conversions")
    click.echo("  5. Save to segment_fileset with proper naming")


@segment_fileset.command()
@click.option('--experiment-id', type=int, default=18, help='Experiment ID')
@click.pass_context
def status(ctx, experiment_id):
    """Check segment fileset generation status"""
    import psycopg2
    from pathlib import Path
    
    output_path = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}/segment_files')
    
    # Check database status
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT COUNT(DISTINCT file_id) as file_count,
                   COUNT(DISTINCT segment_id) as segment_count,
                   COUNT(*) as total_training_segments
            FROM experiment_018_segment_training_data
            WHERE experiment_id = %s
        """, (experiment_id,))
        
        result = cursor.fetchone()
        if result:
            click.echo(f"Experiment {experiment_id} Database Status:")
            click.echo(f"  Files in training: {result[0]}")
            click.echo(f"  Unique segments: {result[1]}")
            click.echo(f"  Total training entries: {result[2]}")
            
    except Exception as e:
        click.echo(f"Database error: {e}")
    finally:
        cursor.close()
        conn.close()
    
    # Check filesystem status
    if not output_path.exists():
        click.echo(f"\nSegment fileset directory does not exist: {output_path}")
        return
    
    segment_files = list(output_path.glob("*.npy"))
    click.echo(f"\nFilesystem Status:")
    click.echo(f"  Found {len(segment_files)} segment files in {output_path}")
    
    if segment_files:
        # Analyze file patterns
        decimations = set()
        for f in segment_files[:100]:  # Sample first 100
            if '_D' in f.name:
                dec_part = f.name.split('_D')[1].split('.')[0]
                decimations.add(int(dec_part))
        
        if decimations:
            click.echo(f"  Decimation factors found: {sorted(decimations)}")


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