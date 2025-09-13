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

# Path to MLDP main project
MLDP_ROOT = Path(__file__).parent.parent.parent / "mldp"

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
            
            # Experiments
            'experiment-select': ['41', '42', '43'],
            
            # Settings
            'set': ['experiment', 'distance', '18', 'l1', 'l2', 'cosine'],
            'show': [],
            
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
            'experiment-select': self.cmd_experiment_select,
            'help': self.cmd_help,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
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

🔬 EXPERIMENTS:
  experiment-select <id>              Run segment selection for experiment

⚙️  SETTINGS:
  set <param> <value>                 Set configuration (experiment, distance)
  show                                Show current settings

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
    
    def cmd_experiment_select(self, args):
        """Run segment selection for an experiment"""
        if not args:
            print("Usage: experiment-select <experiment_id>")
            print("Example: experiment-select 41")
            return
            
        try:
            experiment_id = int(args[0])
        except ValueError:
            print(f"❌ Invalid experiment ID: {args[0]}")
            return
            
        # Import and run the segment selector
        try:
            from experiment_segment_selector import ExperimentSegmentSelector
            
            # Use current database connection if available
            if self.db_conn:
                db_config = {
                    'host': 'localhost',
                    'database': 'arc_detection',
                    'user': 'kjensen'
                }
            else:
                print("⚠️ No database connection. Using default configuration.")
                db_config = {
                    'host': 'localhost',
                    'database': 'arc_detection',
                    'user': 'kjensen'
                }
            
            print(f"🔄 Starting segment selection for experiment {experiment_id}...")
            print("This may take several minutes...")
            
            selector = ExperimentSegmentSelector(experiment_id, db_config)
            summary = selector.run_selection()
            
            print(f"\n✅ Segment Selection Complete!")
            print(f"  Total labels: {summary['total_labels']}")
            print(f"  Total files: {summary['total_files']}")
            print(f"  Total segments: {summary['total_segments']}")
            print(f"  Total pairs: {summary['total_pairs']:,}")
            print(f"  Table: experiment_{experiment_id:03d}_segment_pairs")
            
        except ImportError as e:
            print(f"❌ Could not import segment selector: {e}")
            print("Make sure experiment_segment_selector.py is in the same directory")
        except Exception as e:
            print(f"❌ Error during segment selection: {e}")
    
    def cmd_exit(self, args):
        """Exit the shell"""
        if self.db_conn:
            self.db_conn.close()
        print("\n👋 Goodbye! Thank you for using MLDP.")
        self.running = False


def main():
    """Main entry point"""
    shell = MLDPShell()
    shell.run()


if __name__ == '__main__':
    main()