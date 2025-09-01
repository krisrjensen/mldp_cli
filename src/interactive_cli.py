#!/usr/bin/env python3
"""
Filename: interactive_cli.py
Author(s): Kristophor Jensen
Date Created: 20250901_240000
Date Revised: 20250901_240000
File version: 0.0.0.1
Description: Interactive CLI shell for MLDP ecosystem
"""

import cmd
import sys
import os
import subprocess
import json
import psycopg2
from pathlib import Path
from tabulate import tabulate
import readline
import atexit
from datetime import datetime

# Enable command history
histfile = os.path.expanduser('~/.mldp_history')
try:
    readline.read_history_file(histfile)
    readline.set_history_length(1000)
except FileNotFoundError:
    pass
atexit.register(readline.write_history_file, histfile)

# Path to MLDP main project
MLDP_ROOT = Path(__file__).parent.parent.parent / "mldp"


class MLDPShell(cmd.Cmd):
    """Interactive MLDP Command Shell"""
    
    intro = """
╔══════════════════════════════════════════════════════════════════╗
║                    MLDP Interactive Shell v1.0                    ║
║           Machine Learning Data Processing Platform               ║
╠══════════════════════════════════════════════════════════════════╣
║  Type 'help' for commands or 'help <command>' for details        ║
║  Type 'exit' or 'quit' to leave                                  ║
║  Tab completion and command history available                     ║
╚══════════════════════════════════════════════════════════════════╝
"""
    
    prompt = 'mldp> '
    
    def __init__(self):
        super().__init__()
        self.db_conn = None
        self.current_experiment = 18
        self.current_distance_type = 'l2'
        self.last_result = None
        self.verify_mldp_root()
        
    def verify_mldp_root(self):
        """Verify MLDP project exists"""
        if not MLDP_ROOT.exists():
            print(f"⚠️  Warning: MLDP project not found at {MLDP_ROOT}")
            print("Some commands may not work properly")
    
    def preloop(self):
        """Initialize before entering command loop"""
        print(f"Connected to MLDP ecosystem at: {MLDP_ROOT}")
        print(f"Current experiment: {self.current_experiment}")
        print(f"Current distance type: {self.current_distance_type}")
        print()
    
    def postloop(self):
        """Cleanup when exiting"""
        if self.db_conn:
            self.db_conn.close()
        print("\nGoodbye! Thank you for using MLDP.")
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass
    
    def default(self, line):
        """Handle unknown commands"""
        print(f"Unknown command: {line}")
        print("Type 'help' for available commands")
    
    # ========== Database Commands ==========
    
    def do_connect(self, arg):
        """Connect to database: connect [host] [port] [database] [user]
        Default: localhost 5432 arc_detection kjensen"""
        args = arg.split() if arg else []
        host = args[0] if len(args) > 0 else 'localhost'
        port = args[1] if len(args) > 1 else '5432'
        database = args[2] if len(args) > 2 else 'arc_detection'
        user = args[3] if len(args) > 3 else 'kjensen'
        
        try:
            if self.db_conn:
                self.db_conn.close()
            self.db_conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user
            )
            print(f"✅ Connected to {database}@{host}:{port}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    
    def do_query(self, arg):
        """Execute SQL query: query SELECT * FROM segments LIMIT 10"""
        if not self.db_conn:
            print("Not connected to database. Use 'connect' first.")
            return
        
        if not arg:
            print("Usage: query <SQL statement>")
            return
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(arg)
            
            if cursor.description:
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                if rows:
                    print(tabulate(rows[:100], headers=columns, tablefmt='grid'))
                    if len(rows) > 100:
                        print(f"... showing first 100 of {len(rows)} rows")
                    self.last_result = rows
                else:
                    print("No results found")
            else:
                self.db_conn.commit()
                print(f"Query executed: {cursor.rowcount} rows affected")
            
            cursor.close()
        except Exception as e:
            print(f"Query error: {e}")
            self.db_conn.rollback()
    
    def do_tables(self, arg):
        """List database tables: tables [pattern]"""
        if not self.db_conn:
            print("Not connected to database. Use 'connect' first.")
            return
        
        pattern = arg if arg else '%'
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE %s
            ORDER BY table_name
        """
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(query, (pattern,))
            tables = cursor.fetchall()
            
            if tables:
                print("\nAvailable tables:")
                for table in tables:
                    print(f"  • {table[0]}")
            else:
                print("No tables found")
            
            cursor.close()
        except Exception as e:
            print(f"Error: {e}")
    
    # ========== Distance Commands ==========
    
    def do_calculate(self, arg):
        """Calculate distances: calculate [--segment-size N] [--distance-type TYPE] [--workers N]"""
        calculator_path = MLDP_ROOT / "mldp_exp18_distance" / "mpcctl_distance_calculator.py"
        
        if not calculator_path.exists():
            print(f"❌ Distance calculator not found at {calculator_path}")
            return
        
        # Parse arguments
        args = arg.split() if arg else []
        segment_size = None
        distance_type = self.current_distance_type
        workers = 16
        
        i = 0
        while i < len(args):
            if args[i] == '--segment-size' and i + 1 < len(args):
                segment_size = args[i + 1]
                i += 2
            elif args[i] == '--distance-type' and i + 1 < len(args):
                distance_type = args[i + 1]
                i += 2
            elif args[i] == '--workers' and i + 1 < len(args):
                workers = args[i + 1]
                i += 2
            else:
                i += 1
        
        # Build command
        cmd = [sys.executable, str(calculator_path)]
        cmd.extend(['--input-folder', '/Volumes/ArcData/V3_database/experiment18/segment_files'])
        
        if segment_size:
            cmd.extend(['--segment-size', str(segment_size)])
        
        cmd.append(f'--{distance_type}')
        cmd.extend(['--workers', str(workers)])
        
        print(f"Running distance calculation ({distance_type})...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Distance calculation complete!")
        except subprocess.CalledProcessError:
            print("❌ Distance calculation failed")
    
    def do_insert_distances(self, arg):
        """Insert distances to database: insert_distances [--input-folder PATH] [--distance-type TYPE]"""
        insert_path = MLDP_ROOT / "mldp_distance_db_insert" / "mpcctl_distance_db_insert.py"
        
        if not insert_path.exists():
            print(f"❌ Distance insert tool not found at {insert_path}")
            return
        
        # Parse arguments
        args = arg.split() if arg else []
        input_folder = None
        distance_type = self.current_distance_type
        
        i = 0
        while i < len(args):
            if args[i] == '--input-folder' and i + 1 < len(args):
                input_folder = args[i + 1]
                i += 2
            elif args[i] == '--distance-type' and i + 1 < len(args):
                distance_type = args[i + 1]
                i += 2
            else:
                i += 1
        
        cmd = [sys.executable, str(insert_path)]
        if input_folder:
            cmd.extend(['--input-folder', input_folder])
        cmd.extend(['--distance-type', distance_type])
        
        print("Inserting distances to database...")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Database insertion complete!")
        except subprocess.CalledProcessError:
            print("❌ Database insertion failed")
    
    # ========== Visualization Commands ==========
    
    def do_heatmap(self, arg):
        """Generate heatmap: heatmap [--version N] [--output-dir PATH]"""
        args = arg.split() if arg else []
        version = 7
        output_dir = None
        
        i = 0
        while i < len(args):
            if args[i] == '--version' and i + 1 < len(args):
                version = args[i + 1]
                i += 2
            elif args[i] == '--output-dir' and i + 1 < len(args):
                output_dir = args[i + 1]
                i += 2
            else:
                i += 1
        
        heatmap_path = MLDP_ROOT / "experiment_generator" / "src" / "heatmaps" / f"generate_exp18_heatmaps_v{version}.py"
        
        if not heatmap_path.exists():
            heatmap_path = MLDP_ROOT / "experiment_generator" / "src" / "heatmaps" / "generate_exp18_heatmaps.py"
        
        if not heatmap_path.exists():
            print(f"❌ Heatmap generator not found")
            return
        
        cmd = [sys.executable, str(heatmap_path)]
        cmd.extend(['--distance-type', self.current_distance_type])
        
        if output_dir:
            cmd.extend(['--output-dir', output_dir])
        
        print(f"Generating {self.current_distance_type} heatmap (v{version})...")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Heatmap generated!")
        except subprocess.CalledProcessError:
            print("❌ Heatmap generation failed")
    
    def do_histogram(self, arg):
        """Generate histogram: histogram [--version 1_3] [--bins N]"""
        args = arg.split() if arg else []
        version = '1_3'
        bins = 50
        
        i = 0
        while i < len(args):
            if args[i] == '--version' and i + 1 < len(args):
                version = args[i + 1]
                i += 2
            elif args[i] == '--bins' and i + 1 < len(args):
                bins = args[i + 1]
                i += 2
            else:
                i += 1
        
        histogram_path = MLDP_ROOT / "experiment_generator" / "src" / "heatmaps" / f"histogram_plot_generator_v{version}.py"
        
        if not histogram_path.exists():
            histogram_path = MLDP_ROOT / "experiment_generator" / "src" / "heatmaps" / "simple_histogram_generator.py"
        
        if not histogram_path.exists():
            print(f"❌ Histogram generator not found")
            return
        
        cmd = [sys.executable, str(histogram_path)]
        cmd.extend(['--distance-type', self.current_distance_type])
        cmd.extend(['--bins', str(bins)])
        
        print(f"Generating {self.current_distance_type} histogram...")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Histogram generated!")
        except subprocess.CalledProcessError:
            print("❌ Histogram generation failed")
    
    def do_visualize(self, arg):
        """Visualize segment: visualize --segment-id ID [--file-id ID]"""
        visualizer_path = MLDP_ROOT / "segment_visualizer" / "segment_visualizer.py"
        
        if not visualizer_path.exists():
            print(f"❌ Segment visualizer not found")
            return
        
        args = arg.split() if arg else []
        segment_id = None
        file_id = None
        
        i = 0
        while i < len(args):
            if args[i] == '--segment-id' and i + 1 < len(args):
                segment_id = args[i + 1]
                i += 2
            elif args[i] == '--file-id' and i + 1 < len(args):
                file_id = args[i + 1]
                i += 2
            else:
                i += 1
        
        if not segment_id and not file_id:
            print("Usage: visualize --segment-id ID [--file-id ID]")
            return
        
        cmd = [sys.executable, str(visualizer_path)]
        if segment_id:
            cmd.extend(['--segment-id', str(segment_id)])
        if file_id:
            cmd.extend(['--file-id', str(file_id)])
        
        print("Launching segment visualizer...")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("❌ Visualization failed")
    
    def do_browser(self, arg):
        """Launch database browser"""
        browser_path = MLDP_ROOT / "database_browser" / "database_browser.py"
        
        if not browser_path.exists():
            print(f"❌ Database browser not found")
            return
        
        print("Launching database browser...")
        subprocess.Popen([sys.executable, str(browser_path)])
        print("✅ Database browser launched in background")
    
    # ========== Analysis Commands ==========
    
    def do_stats(self, arg):
        """Show distance statistics: stats [distance_type]"""
        if not self.db_conn:
            print("Not connected to database. Use 'connect' first.")
            return
        
        distance_type = arg if arg else self.current_distance_type
        table_name = f"experiment_{self.current_experiment:03d}_distance_{distance_type.lower()}"
        
        try:
            cursor = self.db_conn.cursor()
            
            # Count records
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            # Get statistics
            cursor.execute(f"""
                SELECT 
                    MIN(distance_s) as min_dist,
                    MAX(distance_s) as max_dist,
                    AVG(distance_s) as avg_dist,
                    STDDEV(distance_s) as std_dist
                FROM {table_name}
            """)
            
            stats = cursor.fetchone()
            
            print(f"\n📊 Statistics for {table_name}:")
            print(f"  Total records: {count:,}")
            print(f"  Min distance:  {stats[0]:.6f}")
            print(f"  Max distance:  {stats[1]:.6f}")
            print(f"  Avg distance:  {stats[2]:.6f}")
            print(f"  Std deviation: {stats[3]:.6f}")
            
            cursor.close()
        except Exception as e:
            print(f"Error: {e}")
    
    def do_closest(self, arg):
        """Find closest pairs: closest [N]"""
        if not self.db_conn:
            print("Not connected to database. Use 'connect' first.")
            return
        
        n = int(arg) if arg else 10
        table_name = f"experiment_{self.current_experiment:03d}_distance_{self.current_distance_type.lower()}"
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(f"""
                SELECT 
                    segment_id_1,
                    segment_id_2,
                    distance_s,
                    file_id_1,
                    file_id_2
                FROM {table_name}
                ORDER BY distance_s ASC
                LIMIT {n}
            """)
            
            rows = cursor.fetchall()
            headers = ['Segment 1', 'Segment 2', 'Distance', 'File 1', 'File 2']
            
            print(f"\n🔍 Top {n} closest pairs ({self.current_distance_type} distance):")
            print(tabulate(rows, headers=headers, tablefmt='grid'))
            
            cursor.close()
        except Exception as e:
            print(f"Error: {e}")
    
    # ========== Settings Commands ==========
    
    def do_set(self, arg):
        """Set configuration: set experiment 18 | set distance l2"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: set <parameter> <value>")
            print("Parameters: experiment, distance")
            return
        
        param, value = args
        
        if param == 'experiment':
            self.current_experiment = int(value)
            print(f"✅ Current experiment set to: {self.current_experiment}")
        elif param == 'distance':
            self.current_distance_type = value
            print(f"✅ Current distance type set to: {self.current_distance_type}")
        else:
            print(f"Unknown parameter: {param}")
    
    def do_show(self, arg):
        """Show current settings"""
        print("\n⚙️  Current Settings:")
        print(f"  Experiment ID:  {self.current_experiment}")
        print(f"  Distance Type:  {self.current_distance_type}")
        print(f"  MLDP Root:      {MLDP_ROOT}")
        print(f"  Database:       {'Connected' if self.db_conn else 'Not connected'}")
    
    # ========== Utility Commands ==========
    
    def do_verify(self, arg):
        """Verify MLDP tools are accessible"""
        print("Verifying MLDP tools...\n")
        
        tools = [
            ("mldp_exp18_distance/mpcctl_distance_calculator.py", "Distance Calculator"),
            ("mldp_distance_db_insert/mpcctl_distance_db_insert.py", "Distance DB Insert"),
            ("segment_visualizer/segment_visualizer.py", "Segment Visualizer"),
            ("database_browser/database_browser.py", "Database Browser"),
            ("experiment_generator", "Experiment Generator"),
        ]
        
        all_found = True
        for tool_path, tool_name in tools:
            full_path = MLDP_ROOT / tool_path
            if full_path.exists():
                print(f"✅ {tool_name}: Found")
            else:
                print(f"❌ {tool_name}: Not found")
                all_found = False
        
        if all_found:
            print("\n✅ All tools verified successfully!")
        else:
            print("\n⚠️  Some tools are missing")
    
    def do_clear(self, arg):
        """Clear the screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def do_export(self, arg):
        """Export last query result: export filename.csv"""
        if not self.last_result:
            print("No results to export. Run a query first.")
            return
        
        if not arg:
            print("Usage: export <filename>")
            return
        
        filename = arg
        
        try:
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.last_result)
            print(f"✅ Exported {len(self.last_result)} rows to {filename}")
        except Exception as e:
            print(f"Export failed: {e}")
    
    def do_time(self, arg):
        """Show current time"""
        print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def do_exit(self, arg):
        """Exit the MLDP shell"""
        return True
    
    def do_quit(self, arg):
        """Exit the MLDP shell"""
        return True
    
    def do_EOF(self, arg):
        """Handle Ctrl-D"""
        print()
        return True
    
    # ========== Help System ==========
    
    def do_help(self, arg):
        """Show help for commands"""
        if arg:
            # Show help for specific command
            super().do_help(arg)
        else:
            # Show categorized help
            print("""
╔══════════════════════════════════════════════════════════════════╗
║                        MLDP Commands                              ║
╚══════════════════════════════════════════════════════════════════╝

DATABASE COMMANDS:
  connect           Connect to PostgreSQL database
  query <SQL>       Execute SQL query
  tables [pattern]  List database tables
  browser           Launch database browser GUI

DISTANCE OPERATIONS:
  calculate         Calculate distances using mpcctl
  insert_distances  Insert distances into database
  stats             Show distance statistics
  closest [N]       Find N closest segment pairs

VISUALIZATION:
  heatmap           Generate distance heatmap
  histogram         Generate distance histogram
  visualize         Visualize segment data

ANALYSIS:
  stats             Show distance statistics
  closest           Find closest pairs

SETTINGS:
  set <param> <val> Set configuration (experiment, distance)
  show              Show current settings

UTILITIES:
  verify            Verify MLDP tools
  clear             Clear screen
  export <file>     Export query results
  time              Show current time
  help [command]    Show help
  exit/quit         Exit shell

Tab completion available. Use arrow keys for command history.
""")


def main():
    """Main entry point for interactive shell"""
    shell = MLDPShell()
    
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Use 'exit' or 'quit' to leave.")
        shell.cmdloop()


if __name__ == '__main__':
    main()