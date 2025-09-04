#!/usr/bin/env python3
"""
Filename: mldp_shell.py
Author(s): Kristophor Jensen
Date Created: 20250901_240000
Date Revised: 20250901_240000
File version: 0.0.0.1
Description: Advanced interactive shell for MLDP with prompt_toolkit
"""

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter, Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import clear
import subprocess
import sys
import os
from pathlib import Path
import psycopg2
from tabulate import tabulate
import json
from datetime import datetime
import shlex

# Path to MLDP main project (mldp_cli is now a submodule inside mldp)
MLDP_ROOT = Path(__file__).parent.parent.parent

# Define style for the prompt
style = Style.from_dict({
    'prompt': '#00aa00 bold',
    'experiment': '#0088ff',
    'distance': '#ff8800',
    'separator': '#666666',
})


class MLDPCompleter(Completer):
    """Custom completer for MLDP commands"""
    
    def __init__(self):
        self.commands = {
            # Database commands
            'connect': ['localhost', '5432', 'arc_detection', 'kjensen'],
            'query': ['SELECT', 'FROM', 'WHERE', 'LIMIT', 'ORDER BY', 'GROUP BY'],
            'tables': [],
            'browser': [],
            
            # Distance commands
            'calculate': ['--segment-size', '--distance-type', '--workers', '8192', '16384', '32768', 'euclidean', 'l1', 'l2', 'cosine'],
            'insert_distances': ['--input-folder', '--distance-type', 'l1', 'l2', 'cosine', 'pearson'],
            
            # Visualization
            'heatmap': ['--version', '--output-dir', '1', '2', '3', '4', '5', '6', '7'],
            'histogram': ['--version', '--bins', '1_0', '1_1', '1_2', '1_3', '50', '100'],
            'visualize': ['--segment-id', '--file-id'],
            
            # Analysis
            'stats': ['l1', 'l2', 'cosine', 'pearson'],
            'closest': ['10', '20', '50', '100'],
            
            # Settings
            'set': ['experiment', 'distance', '18', 'l1', 'l2', 'cosine'],
            'show': [],
            
            # Server Management
            'servers': ['start', 'stop', 'restart', 'status', 'logs'],
            'start': [],
            'stop': [],
            'restart': [],
            'status': [],
            'logs': ['real_time_sync_hub', 'database_browser', 'data_cleaning_tool', 
                    'transient_viewer', 'segment_visualizer', 'distance_visualizer',
                    'experiment_generator', 'jupyter_integration', 'segment_verifier'],
            
            # Utilities
            'verify': [],
            'clear': [],
            'export': [],
            'time': [],
            'help': [],  # Will be populated with all command names
            'exit': [],
            'quit': [],
        }
        # Add all command names to help completions
        self.commands['help'] = list(self.commands.keys())
    
    def get_completions(self, document, complete_event):
        """Get completions for current input"""
        text = document.text_before_cursor
        words = text.split()
        
        if not words:
            # Complete commands
            for cmd in self.commands.keys():
                yield Completion(cmd, start_position=0)
        elif len(words) == 1:
            # Still completing the command
            word = words[0]
            for cmd in self.commands.keys():
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word))
        else:
            # Complete command arguments
            cmd = words[0]
            if cmd in self.commands:
                current_word = words[-1] if len(text) > 0 and not text.endswith(' ') else ''
                start_pos = -len(current_word) if current_word else 0
                
                for option in self.commands[cmd]:
                    if not current_word or option.startswith(current_word):
                        yield Completion(option, start_position=start_pos)


class MLDPShell:
    """Advanced MLDP Interactive Shell"""
    
    def __init__(self):
        self.session = PromptSession(
            history=FileHistory(os.path.expanduser('~/.mldp_shell_history')),
            auto_suggest=AutoSuggestFromHistory(),
            completer=MLDPCompleter(),
            style=style,
            message=self.get_prompt,
            vi_mode=False,  # Set to True if you prefer vi mode
        )
        
        self.db_conn = None
        self.current_experiment = 18
        self.current_distance_type = 'l2'
        self.last_result = None
        self.running = True
        
        # Command handlers
        self.commands = {
            'connect': self.cmd_connect,
            'query': self.cmd_query,
            'tables': self.cmd_tables,
            'browser': self.cmd_browser,
            'calculate': self.cmd_calculate,
            'insert_distances': self.cmd_insert_distances,
            'heatmap': self.cmd_heatmap,
            'histogram': self.cmd_histogram,
            'visualize': self.cmd_visualize,
            'stats': self.cmd_stats,
            'closest': self.cmd_closest,
            'set': self.cmd_set,
            'show': self.cmd_show,
            'verify': self.cmd_verify,
            'clear': self.cmd_clear,
            'export': self.cmd_export,
            'time': self.cmd_time,
            'help': self.cmd_help,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            # Server management commands
            'servers': self.cmd_servers,
            'start': self.cmd_servers_start,
            'stop': self.cmd_servers_stop,
            'restart': self.cmd_servers_restart,
            'status': self.cmd_servers_status,
            'logs': self.cmd_servers_logs,
            # Segment generation commands
            'segment-generate': self.cmd_segment_generate,
            'segment-status': self.cmd_segment_status,
            'segment-test': self.cmd_segment_test,
            'segment-validate': self.cmd_segment_validate,
            'segment-plot': self.cmd_segment_plot,
        }
    
    def get_prompt(self):
        """Generate dynamic prompt with current settings"""
        return FormattedText([
            ('class:prompt', 'mldp'),
            ('class:separator', '['),
            ('class:experiment', f'exp{self.current_experiment}'),
            ('class:separator', ':'),
            ('class:distance', self.current_distance_type),
            ('class:separator', ']'),
            ('class:prompt', '> '),
        ])
    
    def print_banner(self):
        """Print welcome banner"""
        clear()
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MLDP Interactive Shell v2.0                          ║
║                  Machine Learning Data Processing Platform                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • Tab completion and auto-suggestions available                             ║
║  • Type 'help' for commands or 'help <command>' for details                  ║
║  • Current settings shown in prompt: mldp[exp18:l2]>                         ║
║  • Type 'exit' or Ctrl-D to leave                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        
        if MLDP_ROOT.exists():
            print(f"✅ Connected to MLDP ecosystem at: {MLDP_ROOT}")
        else:
            print(f"⚠️  Warning: MLDP not found at {MLDP_ROOT}")
        print()
    
    def run(self):
        """Main shell loop"""
        self.print_banner()
        
        while self.running:
            try:
                # Get user input
                text = self.session.prompt()
                
                if not text.strip():
                    continue
                
                # Parse command
                parts = shlex.split(text)
                if not parts:
                    continue
                
                cmd = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                # Execute command
                if cmd in self.commands:
                    self.commands[cmd](args)
                else:
                    print(f"❌ Unknown command: {cmd}")
                    print("Type 'help' for available commands")
                
            except KeyboardInterrupt:
                print("\nUse 'exit' or Ctrl-D to quit")
                continue
            except EOFError:
                self.cmd_exit([])
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # ========== Command Handlers ==========
    
    def cmd_connect(self, args):
        """Connect to database"""
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
    
    def cmd_query(self, args):
        """Execute SQL query"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        if not args:
            print("Usage: query <SQL statement>")
            return
        
        sql = ' '.join(args)
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(sql)
            
            if cursor.description:
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                if rows:
                    # Limit display to 100 rows
                    display_rows = rows[:100]
                    print(tabulate(display_rows, headers=columns, tablefmt='grid'))
                    
                    if len(rows) > 100:
                        print(f"\n... showing first 100 of {len(rows)} rows")
                    
                    self.last_result = rows
                    print(f"\n📊 {len(rows)} rows returned")
                else:
                    print("No results found")
            else:
                self.db_conn.commit()
                print(f"✅ Query executed: {cursor.rowcount} rows affected")
            
            cursor.close()
        except Exception as e:
            print(f"❌ Query error: {e}")
            self.db_conn.rollback()
    
    def cmd_tables(self, args):
        """List database tables"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        pattern = args[0] if args else '%'
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE %s
                ORDER BY table_name
            """, (pattern,))
            
            tables = cursor.fetchall()
            
            if tables:
                print("\n📋 Available tables:")
                for i, (table,) in enumerate(tables, 1):
                    print(f"  {i:3d}. {table}")
                print(f"\nTotal: {len(tables)} tables")
            else:
                print("No tables found")
            
            cursor.close()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_browser(self, args):
        """Launch database browser"""
        browser_path = MLDP_ROOT / "database_browser" / "database_browser.py"
        
        if not browser_path.exists():
            print(f"❌ Database browser not found")
            return
        
        print("🚀 Launching database browser...")
        subprocess.Popen([sys.executable, str(browser_path)])
        print("✅ Database browser launched in background")
    
    def cmd_calculate(self, args):
        """Calculate distances"""
        calculator_path = MLDP_ROOT / "mldp_exp18_distance" / "mpcctl_distance_calculator.py"
        
        if not calculator_path.exists():
            print(f"❌ Distance calculator not found")
            return
        
        # Parse arguments
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
        
        cmd = [sys.executable, str(calculator_path)]
        cmd.extend(['--input-folder', '/Volumes/ArcData/V3_database/experiment18/segment_files'])
        
        if segment_size:
            cmd.extend(['--segment-size', str(segment_size)])
        
        cmd.append(f'--{distance_type}')
        cmd.extend(['--workers', str(workers)])
        
        print(f"🔄 Running distance calculation ({distance_type})...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Distance calculation complete!")
            else:
                print(f"❌ Distance calculation failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_insert_distances(self, args):
        """Insert distances to database"""
        insert_path = MLDP_ROOT / "mldp_distance_db_insert" / "mpcctl_distance_db_insert.py"
        
        if not insert_path.exists():
            print(f"❌ Distance insert tool not found")
            return
        
        # Parse arguments
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
        
        print("🔄 Inserting distances to database...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Database insertion complete!")
            else:
                print(f"❌ Insertion failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_heatmap(self, args):
        """Generate heatmap"""
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
        
        print(f"🎨 Generating {self.current_distance_type} heatmap (v{version})...")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Heatmap generated!")
        except subprocess.CalledProcessError:
            print("❌ Heatmap generation failed")
    
    def cmd_histogram(self, args):
        """Generate histogram"""
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
        
        print(f"📊 Generating {self.current_distance_type} histogram...")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Histogram generated!")
        except subprocess.CalledProcessError:
            print("❌ Histogram generation failed")
    
    def cmd_visualize(self, args):
        """Visualize segment"""
        visualizer_path = MLDP_ROOT / "segment_visualizer" / "segment_visualizer.py"
        
        if not visualizer_path.exists():
            print(f"❌ Segment visualizer not found")
            return
        
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
        
        print("🔍 Launching segment visualizer...")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("❌ Visualization failed")
    
    def cmd_stats(self, args):
        """Show distance statistics"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        distance_type = args[0] if args else self.current_distance_type
        table_name = f"experiment_{self.current_experiment:03d}_distance_{distance_type.lower()}"
        
        try:
            cursor = self.db_conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            if not cursor.fetchone()[0]:
                print(f"❌ Table {table_name} does not exist")
                cursor.close()
                return
            
            # Count records
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            # Get statistics
            cursor.execute(f"""
                SELECT 
                    MIN(distance_s) as min_dist,
                    MAX(distance_s) as max_dist,
                    AVG(distance_s) as avg_dist,
                    STDDEV(distance_s) as std_dist,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY distance_s) as q1,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY distance_s) as median,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY distance_s) as q3
                FROM {table_name}
            """)
            
            stats = cursor.fetchone()
            
            print(f"\n📊 Statistics for {table_name}:")
            print(f"{'─' * 50}")
            print(f"  Total records:  {count:,}")
            print(f"  Min distance:   {stats[0]:.6f}")
            print(f"  Q1 (25%):       {stats[4]:.6f}")
            print(f"  Median (50%):   {stats[5]:.6f}")
            print(f"  Q3 (75%):       {stats[6]:.6f}")
            print(f"  Max distance:   {stats[1]:.6f}")
            print(f"  Mean distance:  {stats[2]:.6f}")
            print(f"  Std deviation:  {stats[3]:.6f}")
            
            cursor.close()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_closest(self, args):
        """Find closest pairs"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        n = int(args[0]) if args else 10
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
            print(f"❌ Error: {e}")
    
    def cmd_set(self, args):
        """Set configuration"""
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
            print(f"❌ Unknown parameter: {param}")
    
    def cmd_show(self, args):
        """Show current settings"""
        print("\n⚙️  Current Settings:")
        print(f"{'─' * 40}")
        print(f"  Experiment ID:  {self.current_experiment}")
        print(f"  Distance Type:  {self.current_distance_type}")
        print(f"  MLDP Root:      {MLDP_ROOT}")
        print(f"  Database:       {'✅ Connected' if self.db_conn else '❌ Not connected'}")
        
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT current_database(), current_user")
                db, user = cursor.fetchone()
                print(f"  DB Name:        {db}")
                print(f"  DB User:        {user}")
                cursor.close()
            except:
                pass
    
    def cmd_verify(self, args):
        """Verify MLDP tools"""
        print("\n🔍 Verifying MLDP tools...")
        print(f"{'─' * 50}")
        
        tools = [
            ("mldp_exp18_distance/mpcctl_distance_calculator.py", "Distance Calculator"),
            ("mldp_distance_db_insert/mpcctl_distance_db_insert.py", "Distance DB Insert"),
            ("segment_visualizer/segment_visualizer.py", "Segment Visualizer"),
            ("database_browser/database_browser.py", "Database Browser"),
            ("experiment_generator", "Experiment Generator"),
            ("segment_verifier", "Segment Verifier"),
            ("data_cleaning_tool", "Data Cleaning Tool"),
        ]
        
        found = 0
        missing = 0
        
        for tool_path, tool_name in tools:
            full_path = MLDP_ROOT / tool_path
            if full_path.exists():
                print(f"  ✅ {tool_name:25s} Found")
                found += 1
            else:
                print(f"  ❌ {tool_name:25s} Not found")
                missing += 1
        
        print(f"{'─' * 50}")
        print(f"Summary: {found} found, {missing} missing")
        
        if missing == 0:
            print("✅ All tools verified successfully!")
        else:
            print("⚠️  Some tools are missing")
    
    def cmd_clear(self, args):
        """Clear screen"""
        clear()
        self.print_banner()
    
    def cmd_export(self, args):
        """Export last query result"""
        if not self.last_result:
            print("❌ No results to export. Run a query first.")
            return
        
        if not args:
            print("Usage: export <filename>")
            return
        
        filename = args[0]
        
        try:
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(self.last_result, f, indent=2, default=str)
            else:
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.last_result)
            
            print(f"✅ Exported {len(self.last_result)} rows to {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def cmd_time(self, args):
        """Show current time"""
        now = datetime.now()
        print(f"🕐 Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Unix timestamp: {int(now.timestamp())}")
    
    def cmd_help(self, args):
        """Show help"""
        if args:
            cmd = args[0]
            if cmd in self.commands:
                print(f"\nHelp for '{cmd}':")
                print(f"  {self.commands[cmd].__doc__}")
            else:
                print(f"❌ Unknown command: {cmd}")
        else:
            print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              MLDP Commands                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 DATABASE COMMANDS:
  connect [host] [port] [db] [user]  Connect to PostgreSQL database
  query <SQL>                         Execute SQL query
  tables [pattern]                    List database tables
  browser                             Launch database browser GUI

📐 DISTANCE OPERATIONS:
  calculate [options]                 Calculate distances using mpcctl
  insert_distances [options]          Insert distances into database
  stats [distance_type]               Show distance statistics
  closest [N]                         Find N closest segment pairs

🎨 VISUALIZATION:
  heatmap [--version N]               Generate distance heatmap
  histogram [--version] [--bins]      Generate distance histogram
  visualize --segment-id ID           Visualize segment data

⚙️  SETTINGS:
  set <param> <value>                 Set configuration (experiment, distance)
  show                                Show current settings

🖥️  SERVER MANAGEMENT:
  start                               Start all MLDP servers
  stop                                Stop all MLDP servers
  restart                             Restart all MLDP servers
  status                              Check status of all servers
  logs [service] [lines]              View server logs
  servers <command>                   Server management (start/stop/status/etc)

🛠️  UTILITIES:
  verify                              Verify MLDP tools
  clear                               Clear screen
  export <filename>                   Export query results (.csv or .json)
  time                                Show current time
  help [command]                      Show help
  exit/quit                           Exit shell

💡 TIPS:
  • Use Tab for command completion
  • Use ↑/↓ arrows for command history
  • Current settings shown in prompt: mldp[exp18:l2]>
  • SQL queries support all PostgreSQL syntax
  • Export supports .csv and .json formats
""")
    
    def cmd_exit(self, args):
        """Exit the shell"""
        if self.db_conn:
            self.db_conn.close()
        print("\n👋 Goodbye! Thank you for using MLDP.")
        self.running = False
    
    # ========== Server Management Commands ==========
    
    def cmd_servers(self, args):
        """Server management - show help for server commands"""
        if args and args[0] in ['start', 'stop', 'restart', 'status', 'logs']:
            # Handle subcommands
            if args[0] == 'start':
                self.cmd_servers_start([])
            elif args[0] == 'stop':
                self.cmd_servers_stop([])
            elif args[0] == 'restart':
                self.cmd_servers_restart([])
            elif args[0] == 'status':
                self.cmd_servers_status([])
            elif args[0] == 'logs':
                self.cmd_servers_logs(args[1:])
        else:
            print("""
🖥️  Server Management Commands:
────────────────────────────────
  servers start    - Start all MLDP servers
  servers stop     - Stop all MLDP servers
  servers restart  - Restart all MLDP servers
  servers status   - Check status of all servers
  servers logs     - View server logs
  
  Shortcuts:
  start            - Start all servers
  stop             - Stop all servers
  restart          - Restart all servers
  status           - Check server status
  logs [service]   - View logs
""")
    
    def cmd_servers_start(self, args):
        """Start all MLDP servers"""
        scripts_path = MLDP_ROOT / "scripts" / "start_services.sh"
        
        if not scripts_path.exists():
            print(f"❌ start_services.sh not found at {scripts_path}")
            return
        
        print("🚀 Starting all MLDP servers...")
        print("This may take a moment...")
        print("─" * 60)
        
        try:
            result = subprocess.run(
                ["bash", str(scripts_path)],
                capture_output=False,
                text=True,
                cwd=str(MLDP_ROOT)
            )
            if result.returncode == 0:
                print("\n✅ All servers started successfully!")
                print("\nUse 'status' to check server status")
            else:
                print("\n⚠️  Some servers may have failed to start")
                print("Use 'status' to check which services are running")
        except Exception as e:
            print(f"❌ Error starting servers: {e}")
    
    def cmd_servers_stop(self, args):
        """Stop all MLDP servers"""
        scripts_path = MLDP_ROOT / "scripts" / "stop_services.sh"
        
        if not scripts_path.exists():
            print(f"❌ stop_services.sh not found at {scripts_path}")
            return
        
        print("🛑 Stopping all MLDP servers...")
        
        try:
            result = subprocess.run(
                ["bash", str(scripts_path)],
                capture_output=True,
                text=True,
                cwd=str(MLDP_ROOT)
            )
            print(result.stdout)
            if result.returncode == 0:
                print("✅ All servers stopped successfully!")
            else:
                print("⚠️  Some servers may still be running")
                print("Use 'status' to check")
        except Exception as e:
            print(f"❌ Error stopping servers: {e}")
    
    def cmd_servers_restart(self, args):
        """Restart all MLDP servers"""
        print("🔄 Restarting all MLDP servers...")
        print("─" * 60)
        
        # Stop servers
        self.cmd_servers_stop([])
        
        # Wait
        import time
        print("\n⏳ Waiting for services to shut down...")
        time.sleep(3)
        
        # Start servers
        self.cmd_servers_start([])
    
    def cmd_servers_status(self, args):
        """Check status of all MLDP servers"""
        operation_pid_path = MLDP_ROOT / "operation" / "pid"
        
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
        
        print("\n📊 MLDP Server Status")
        print("=" * 70)
        print(f"{'Service':<30} {'Port':<8} {'PID':<10} {'Status':<15} {'URL'}")
        print("-" * 70)
        
        running_count = 0
        total_count = len(services)
        
        for service_name, port, display_name in services:
            pid_file = operation_pid_path / f"{service_name}.pid"
            
            status = "❓ Unknown"
            pid_str = "-"
            url = f"http://localhost:{port}"
            
            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        pid_str = f.read().strip()
                    
                    # Check if process is running using ps command
                    result = subprocess.run(
                        ["ps", "-p", pid_str],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        status = "✅ Running"
                        running_count += 1
                    else:
                        status = "❌ Not Running"
                        url = "-"
                except Exception:
                    status = "❌ Error"
                    url = "-"
            else:
                status = "⏹️  Stopped"
                url = "-"
            
            print(f"{display_name:<30} {port:<8} {pid_str:<10} {status:<15} {url}")
        
        print("-" * 70)
        print(f"Summary: {running_count}/{total_count} services running")
        
        if running_count == total_count:
            print("\n🎉 All services are running!")
        elif running_count == 0:
            print("\n⚠️  No services are running. Use 'start' to start them.")
        else:
            print(f"\n⚠️  Only {running_count}/{total_count} services are running.")
            print("Use 'restart' to restart all services.")
    
    def cmd_servers_logs(self, args):
        """View server logs"""
        logs_path = MLDP_ROOT / "operation" / "logs"
        
        if args and len(args) > 0:
            service = args[0]
            lines = int(args[1]) if len(args) > 1 else 50
            
            log_file = logs_path / f"{service}.log"
            if log_file.exists():
                print(f"\n📋 Last {lines} lines of {service}.log:")
                print("=" * 60)
                result = subprocess.run(
                    ["tail", f"-{lines}", str(log_file)],
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
            else:
                print(f"❌ Log file not found: {log_file}")
                print("\nAvailable services:")
                for log_file in sorted(logs_path.glob("*.log")):
                    print(f"  • {log_file.stem}")
        else:
            # Show available log files
            print("\n📁 Available log files:")
            print("=" * 60)
            if logs_path.exists():
                log_files = list(logs_path.glob("*.log"))
                if log_files:
                    for log_file in sorted(log_files):
                        size = log_file.stat().st_size
                        size_str = f"{size / 1024:.1f}K" if size < 1024*1024 else f"{size / (1024*1024):.1f}M"
                        print(f"  {log_file.stem:<30} {size_str:>10}")
                    print("\nUsage: logs <service> [lines]")
                    print("Example: logs real_time_sync_hub 100")
                else:
                    print("No log files found")
            else:
                print("❌ Logs directory not found")
    
    def cmd_segment_generate(self, args):
        """Generate segment fileset for experiment"""
        try:
            from .segment_processor import SegmentFilesetProcessor
        except ImportError:
            # Fallback for when running as script
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from segment_processor import SegmentFilesetProcessor
        
        # Parse arguments
        # args is already a list from the shell parser
        parts = args if isinstance(args, list) else args.split()
        if not parts or parts[0] != 'exp18':
            print("Usage: segment-generate exp18 [options]")
            print("Options:")
            print("  --files <range>   File range (e.g., 200-210)")
            print("  --types <list>    Data types (comma-separated)")
            print("  --decimations <list>  Decimation factors (comma-separated)")
            print("  --sizes <list>    Segment sizes to process (comma-separated)")
            print("                    Available: 8192,32768,65536,262144,524288")
            return
        
        # Check for options
        file_range = None
        data_types = None
        decimations = None
        sizes = None
        
        for i, part in enumerate(parts):
            if part == '--files' and i + 1 < len(parts):
                file_range = parts[i + 1]
            elif part == '--types' and i + 1 < len(parts):
                data_types = parts[i + 1].split(',')
            elif part == '--decimations' and i + 1 < len(parts):
                decimations = [int(d) for d in parts[i + 1].split(',')]
            elif part == '--sizes' and i + 1 < len(parts):
                sizes = [int(s) for s in parts[i + 1].split(',')]
        
        # Use defaults for experiment 18
        if decimations is None:
            decimations = [1, 3, 7, 15, 31, 63, 127, 255, 511]
        if data_types is None:
            data_types = ['ADC14', 'ADC12', 'ADC10', 'ADC8', 'ADC6']
        
        # Note: decimation 0 means no decimation (keep all samples)
        
        print("\n" + "="*70)
        print("Starting Experiment 18 Segment Generation")
        print("="*70)
        print(f"Decimations: {decimations}")
        print(f"Data Types: {data_types}")
        print(f"File Range: {file_range if file_range else 'all files'}")
        print(f"Segment Sizes: {sizes if sizes else 'all available (8192,32768,65536,262144,524288)'}")
        
        # Estimate file count
        if file_range:
            parts = file_range.split('-')
            if len(parts) == 2:
                num_files = int(parts[1]) - int(parts[0]) + 1
            else:
                num_files = 1
        else:
            num_files = 750  # Approximate total files in experiment 18
        
        # Estimate segments per file based on sizes filter
        if sizes:
            # Rough estimate based on typical distribution when filtering by size
            segments_per_file = len(sizes) * 2  # ~2 segments per size per file on average
        else:
            segments_per_file = 13  # Average when processing all sizes
        
        estimated_files = num_files * segments_per_file * len(decimations) * len(data_types)
        print(f"\nEstimated files to generate: ~{estimated_files:,}")
        
        # Confirm
        response = input("\nProceed? (y/n): ")
        if response.lower() != 'y':
            print("Generation cancelled.")
            return
        
        # Create processor and run
        print("\nInitializing processor...")
        processor = SegmentFilesetProcessor(experiment_id=18)
        
        print("Starting generation (this may take several hours)...")
        stats = processor.generate(
            decimations=decimations,
            data_types=data_types,
            file_range=file_range,
            sizes=sizes,
            workers=16
        )
        
        print("\n✅ Generation complete!")
    
    def cmd_segment_status(self, args):
        """Check segment generation status"""
        from pathlib import Path
        import json
        
        base_path = Path('/Volumes/ArcData/V3_database/experiment018/segment_files')
        progress_file = base_path / 'generation_progress.json'
        
        print("\n📊 Segment Generation Status")
        print("="*60)
        
        # Check progress file
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                completed = progress.get('completed', [])
                print(f"Segments processed: {len(completed):,}")
        else:
            print("No generation in progress")
        
        # Count existing files
        total_files = 0
        for pattern in ['S*/T*/D*/*.npy']:
            files = list(base_path.glob(pattern))
            total_files += len(files)
        
        print(f"Total segment files: {total_files:,}")
        
        # Show breakdown by size
        print("\nBreakdown by segment size:")
        for size in [8192, 32768, 65536, 131072, 262144, 524288]:
            size_files = list(base_path.glob(f"S{size:06d}/*/*/*.npy"))
            if size_files:
                print(f"  {size:7d} samples: {len(size_files):,} files")
        
        # Show breakdown by type
        print("\nBreakdown by data type:")
        for data_type in ['TRAW', 'TADC14', 'TADC12', 'TADC10', 'TADC8', 'TADC6']:
            type_files = list(base_path.glob(f"*/T{data_type}/*/*.npy"))
            if type_files:
                print(f"  {data_type}: {len(type_files):,} files")
    
    def cmd_segment_test(self, args):
        """Test segment generation with small dataset"""
        try:
            from .segment_processor import SegmentFilesetProcessor
        except ImportError:
            # Fallback for when running as script
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from segment_processor import SegmentFilesetProcessor
        
        print("\n🧪 Testing Segment Generation")
        print("="*60)
        print("Test parameters:")
        print("  Files: 200-201 (2 files)")
        print("  Decimations: [1, 3]")
        print("  Data Types: [ADC12, ADC8]")
        print("  Expected files: ~104 (2 files × 13 segments × 2 decimations × 2 types)")
        
        response = input("\nRun test? (y/n): ")
        if response.lower() != 'y':
            print("Test cancelled.")
            return
        
        print("\nRunning test...")
        processor = SegmentFilesetProcessor(experiment_id=18)
        
        stats = processor.generate(
            decimations=[1, 3],
            data_types=['ADC12', 'ADC8'],
            file_range='200-201',
            workers=2
        )
        
        print("\n✅ Test complete!")
    
    def cmd_segment_validate(self, args):
        """Validate generated segment files"""
        import numpy as np
        from pathlib import Path
        
        base_path = Path('/Volumes/ArcData/V3_database/experiment018/segment_files')
        
        print("\n🔍 Validating Segment Files")
        print("="*60)
        
        # Sample some files
        sample_files = list(base_path.glob("*/T*/*/*.npy"))[:10]
        
        if not sample_files:
            print("No segment files found to validate")
            return
        
        print(f"Validating {len(sample_files)} sample files...")
        
        for filepath in sample_files:
            try:
                data = np.load(filepath)
                size = data.shape[0]
                is_power_of_2 = (size & (size - 1)) == 0
                
                # Parse filename
                filename = filepath.name
                parts = filename.split('_')
                segment_id = parts[0]
                file_id = parts[1]
                data_type = parts[3]
                
                status = "✅" if is_power_of_2 else "❌"
                print(f"{status} {filename[:40]:<40} Shape: {data.shape}, 2^N: {is_power_of_2}")
                
            except Exception as e:
                print(f"❌ Error validating {filepath.name}: {e}")
    
    def cmd_segment_plot(self, args):
        """Plot segment files with statistical analysis
        
        Usage: segment-plot [options]
        
        Examples:
            segment-plot --original-segment 104075 --decimations 0 --output-folder ~/plots/
            segment-plot --result-segment-size 131072 --types RAW --output-folder ~/plots/
            segment-plot --file-labels 200,201 --num-points 500 --peak-detect --output-folder ~/plots/
        """
        try:
            from segment_file_plotter import plot_segment_files
        except ImportError:
            import segment_file_plotter
            plot_segment_files = segment_file_plotter.plot_segment_files
        
        # Parse arguments
        parts = args if isinstance(args, list) else args.split()
        
        # Initialize parameters
        params = {
            'experiment_id': self.current_experiment,
            'original_segment': None,
            'result_segment_size': None,
            'segment_labels': None,
            'file_labels': None,
            'decimations': None,
            'types': None,
            'num_points': 1000,
            'peak_detect': False,
            'plot_actual': True,
            'plot_minimums': False,
            'plot_maximums': False,
            'plot_average': False,
            'plot_variance': False,
            'plot_stddev': False,
            'minimums_point': False,
            'minimums_line': False,
            'maximums_point': False,
            'maximums_line': False,
            'average_point': False,
            'average_line': True,
            'variance_point': False,
            'variance_line': True,
            'stddev_point': False,
            'stddev_line': True,
            'no_subplots': False,
            'subplots': 'file',
            'max_subplot': (3, 3),
            'dpi': 300,
            'format': 'png',
            'title': None,
            'plot_style': 'cleaning',
            'output_folder': None
        }
        
        # Parse command line arguments
        i = 0
        while i < len(parts):
            if parts[i] == '--original-segment' and i + 1 < len(parts):
                params['original_segment'] = int(parts[i + 1])
                i += 2
            elif parts[i] == '--result-segment-size' and i + 1 < len(parts):
                params['result_segment_size'] = int(parts[i + 1])
                i += 2
            elif parts[i] == '--segment-labels' and i + 1 < len(parts):
                if parts[i + 1].lower() == 'all':
                    params['segment_labels'] = None
                else:
                    params['segment_labels'] = [int(x) for x in parts[i + 1].split(',')]
                i += 2
            elif parts[i] == '--file-labels' and i + 1 < len(parts):
                if parts[i + 1].lower() == 'all':
                    params['file_labels'] = None
                else:
                    params['file_labels'] = [int(x) for x in parts[i + 1].split(',')]
                i += 2
            elif parts[i] == '--decimations' and i + 1 < len(parts):
                if parts[i + 1].lower() == 'all':
                    params['decimations'] = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511]
                else:
                    params['decimations'] = [int(x) for x in parts[i + 1].split(',')]
                i += 2
            elif parts[i] == '--types' and i + 1 < len(parts):
                if parts[i + 1].lower() == 'all':
                    params['types'] = ['RAW', 'ADC14', 'ADC12', 'ADC10', 'ADC8', 'ADC6']
                else:
                    params['types'] = parts[i + 1].split(',')
                i += 2
            elif parts[i] == '--num-points' and i + 1 < len(parts):
                params['num_points'] = int(parts[i + 1])
                i += 2
            elif parts[i] == '--peak-detect':
                params['peak_detect'] = True
                i += 1
            elif parts[i] == '--plot-actual':
                params['plot_actual'] = True
                i += 1
            elif parts[i] == '--plot-minimums':
                params['plot_minimums'] = True
                i += 1
            elif parts[i] == '--plot-minimums-point':
                params['plot_minimums'] = True
                params['minimums_point'] = True
                i += 1
            elif parts[i] == '--plot-minimums-line':
                params['plot_minimums'] = True
                params['minimums_line'] = True
                i += 1
            elif parts[i] == '--plot-maximums':
                params['plot_maximums'] = True
                i += 1
            elif parts[i] == '--plot-maximums-point':
                params['plot_maximums'] = True
                params['maximums_point'] = True
                i += 1
            elif parts[i] == '--plot-maximums-line':
                params['plot_maximums'] = True
                params['maximums_line'] = True
                i += 1
            elif parts[i] == '--plot-average':
                params['plot_average'] = True
                i += 1
            elif parts[i] == '--plot-average-point':
                params['plot_average'] = True
                params['average_point'] = True
                i += 1
            elif parts[i] == '--plot-average-line':
                params['plot_average'] = True
                params['average_line'] = True
                i += 1
            elif parts[i] == '--plot-variance':
                params['plot_variance'] = True
                i += 1
            elif parts[i] == '--plot-variance-point':
                params['plot_variance'] = True
                params['variance_point'] = True
                i += 1
            elif parts[i] == '--plot-variance-line':
                params['plot_variance'] = True
                params['variance_line'] = True
                i += 1
            elif parts[i] == '--plot-stddev':
                params['plot_stddev'] = True
                i += 1
            elif parts[i] == '--plot-stddev-point':
                params['plot_stddev'] = True
                params['stddev_point'] = True
                i += 1
            elif parts[i] == '--plot-stddev-line':
                params['plot_stddev'] = True
                params['stddev_line'] = True
                i += 1
            elif parts[i] == '--no-subplots':
                params['no_subplots'] = True
                i += 1
            elif parts[i] == '--subplots' and i + 1 < len(parts):
                params['subplots'] = parts[i + 1]
                i += 2
            elif parts[i] == '--max-subplot' and i + 1 < len(parts):
                rows, cols = parts[i + 1].split(',')
                params['max_subplot'] = (int(rows), int(cols))
                i += 2
            elif parts[i] == '--dpi' and i + 1 < len(parts):
                params['dpi'] = int(parts[i + 1])
                i += 2
            elif parts[i] == '--format' and i + 1 < len(parts):
                params['format'] = parts[i + 1]
                i += 2
            elif parts[i] == '--title' and i + 1 < len(parts):
                params['title'] = parts[i + 1]
                i += 2
            elif parts[i] == '--plot-style' and i + 1 < len(parts):
                params['plot_style'] = parts[i + 1]
                i += 2
            elif parts[i] == '--output-folder' and i + 1 < len(parts):
                params['output_folder'] = parts[i + 1]
                i += 2
            else:
                i += 1
        
        # Check required parameters
        if not params['output_folder']:
            print("❌ Error: --output-folder is required")
            print("\nUsage: segment-plot --output-folder <path> [options]")
            print("\nExample:")
            print("  segment-plot --original-segment 104075 --decimations 0 --output-folder ~/plots/")
            return
        
        # Set defaults if nothing specified
        if params['decimations'] is None:
            params['decimations'] = [0]
        if params['types'] is None:
            params['types'] = ['RAW']
        
        print(f"\n📊 Starting Segment Plot Generation")
        print(f"Output folder: {params['output_folder']}")
        print(f"Experiment: {params['experiment_id']}")
        print(f"Decimations: {params['decimations']}")
        print(f"Types: {params['types']}")
        print(f"Num points: {params['num_points']}")
        print(f"Peak detect: {params['peak_detect']}")
        
        # Call the plotting function
        try:
            plot_segment_files(**params)
            print("\n✅ Plotting complete!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    shell = MLDPShell()
    shell.run()


if __name__ == '__main__':
    main()