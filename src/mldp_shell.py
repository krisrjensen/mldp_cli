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
            
            # Experiment commands
            'experiment-list': [],
            'experiment-info': ['17', '18', '19', '20'],
            'experiment-config': ['17', '18', '19', '20', '--json'],
            'experiment-summary': ['17', '18', '19', '20'],
            'experiment-generate': ['balanced', 'small', 'large', '--dry-run'],
            'experiment-create': ['--name', '--max-files', '--segment-sizes', '--data-types', '--help'],
            
            # Distance commands
            'calculate': ['--segment-size', '--distance-type', '--workers', '8192', '16384', '32768', 'euclidean', 'l1', 'l2', 'cosine'],
            'insert_distances': ['--input-folder', '--distance-type', 'l1', 'l2', 'cosine', 'pearson'],
            
            # Visualization
            'heatmap': ['--version', '--output-dir', '1', '2', '3', '4', '5', '6', '7'],
            'histogram': ['--version', '--bins', '1_0', '1_1', '1_2', '1_3', '50', '100'],
            'visualize': ['--segment-id', '--file-id'],
            'segment-plot': ['--amplitude-method', '--original-segment', '--result-segment-size', '--types',
                           '--decimations', '--output-folder', 'raw', 'minmax', 'zscore', 'amplitude_0', 'amplitude_1',
                           'RAW', 'ADC6', 'ADC8', 'ADC10', 'ADC12', 'ADC14', '0', '7', '15'],
            'feature-plot': ['--file', '--save', '--output-folder'],

            # Distance calculations
            'init-distance-tables': ['--drop-existing', '--help'],
            'show-distance-metrics': [],
            'add-distance-metric': ['--metric', 'L1', 'L2', 'cosine', 'pearson', 'euclidean', 'manhattan', 'wasserstein'],
            'remove-distance-metric': ['--metric', '--all-except', 'L1', 'L2', 'cosine', 'pearson'],
            'clean-distance-tables': ['--dry-run', '--force'],
            'show-distance-functions': ['--active-only'],
            'update-distance-function': ['--pairwise-metric', '--library', '--function-import', '--description', '--active'],
            'mpcctl-distance-function': ['--start', '--pause', '--continue', '--stop', '--status', '--workers', '--log', '--verbose', '--help'],

            # Analysis
            'stats': ['l1', 'l2', 'cosine', 'pearson'],
            'closest': ['10', '20', '50', '100'],
            
            # Experiments
            'select-segments': ['41', '42', '43'],
            'update-decimations': ['0', '1', '3', '7', '15', '31', '63', '127', '255', '511'],
            'update-segment-sizes': ['128', '256', '512', '1024', '2048', '4096', '8192', '16384', '32768', '65536', '131072', '262144'],
            'update-amplitude-methods': ['minmax', 'zscore', 'maxabs', 'robust', 'TRAW', 'TADC14', 'TADC12', 'TADC10', 'TADC8', 'TADC6'],
            'create-feature-set': ['--name', '--features', '--n-value', 'voltage', 'current', 'impedance', 'power'],
            'remove-feature-set': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            'clear-feature-sets': [],
            'list-feature-sets': [],
            'show-all-feature-sets': [],
            # New feature management commands
            'create-feature': ['--name', '--category', '--behavior', '--description', 'electrical', 'statistical', 'spectral', 'temporal', 'compute', 'driver', 'derived', 'aggregate', 'transform'],
            'list-features': ['--category', 'electrical', 'statistical', 'spectral', 'temporal', 'compute'],
            'show-feature': [],
            'update-feature': ['--name', '--category', '--description'],
            'delete-feature': [],
            'create-global-feature-set': ['--name', '--category', '--description', 'electrical', 'statistical', 'custom'],
            'add-features-to-set': ['--features'],
            'remove-features-from-set': ['--features'],
            'clone-feature-set': ['--name'],
            'link-feature-set': ['--n-value', '--channel', '--priority', 'load_voltage', 'source_current'],
            'bulk-link-feature-sets': ['--sets', '--n-values'],
            'update-feature-link': ['--n-value', '--priority', '--active'],
            'show-feature-config': [],
            'update-selection-config': ['--max-files', '--seed', '--strategy', '--balanced', '10', '25', '50', '100'],
            'select-files': ['--max-files', '--label', '--seed', '50', '100'],
            'remove-files': ['--label', '--file-ids'],
            'remove-file-labels': ['trash', 'voltage_only', 'arc_short_gap', 'arc_extinguish', 'other'],
            'remove-segments': ['--label', '--segment-ids'],
            
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
            # Experiment commands
            'experiment-list': self.cmd_experiment_list,
            'experiment-info': self.cmd_experiment_info,
            'experiment-config': self.cmd_experiment_config,
            'experiment-summary': self.cmd_experiment_summary,
            'experiment-generate': self.cmd_experiment_generate,
            'experiment-create': self.cmd_experiment_create,
            # Distance commands
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
            'select-segments': self.cmd_select_segments,
            'update-decimations': self.cmd_update_decimations,
            'update-segment-sizes': self.cmd_update_segment_sizes,
            'update-amplitude-methods': self.cmd_update_amplitude_methods,
            'create-feature-set': self.cmd_create_feature_set,
            'add-feature-set': self.cmd_add_feature_set,
            'remove-feature-set': self.cmd_remove_feature_set,
            'clear-feature-sets': self.cmd_clear_feature_sets,
            'list-feature-sets': self.cmd_list_feature_sets,
            'show-all-feature-sets': self.cmd_show_all_feature_sets,
            # New feature management commands
            'create-feature': self.cmd_create_feature,
            'list-features': self.cmd_list_features,
            'show-feature': self.cmd_show_feature,
            'update-feature': self.cmd_update_feature,
            'delete-feature': self.cmd_delete_feature,
            'create-global-feature-set': self.cmd_create_global_feature_set,
            'add-features-to-set': self.cmd_add_features_to_set,
            'remove-features-from-set': self.cmd_remove_features_from_set,
            'clone-feature-set': self.cmd_clone_feature_set,
            'link-feature-set': self.cmd_link_feature_set,
            'bulk-link-feature-sets': self.cmd_bulk_link_feature_sets,
            'update-feature-link': self.cmd_update_feature_link,
            'show-feature-config': self.cmd_show_feature_config,
            'update-selection-config': self.cmd_update_selection_config,
            'select-files': self.cmd_select_files,
            'remove-files': self.cmd_remove_files,
            'remove-file-labels': self.cmd_remove_file_labels,
            'remove-segments': self.cmd_remove_segments,
            'generate-training-data': self.cmd_generate_training_data,
            'generate-segment-pairs': self.cmd_generate_segment_pairs,
            'generate-feature-fileset': self.cmd_generate_feature_fileset,
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
            'generate-segment-fileset': self.cmd_generate_segment_fileset,
            'show-segment-status': self.cmd_show_segment_status,
            'segment-test': self.cmd_segment_test,
            'validate-segments': self.cmd_validate_segments,
            'segment-plot': self.cmd_segment_plot,
            'feature-plot': self.cmd_feature_plot,
            # Distance calculation commands
            'init-distance-tables': self.cmd_init_distance_tables,
            'show-distance-metrics': self.cmd_show_distance_metrics,
            'add-distance-metric': self.cmd_add_distance_metric,
            'remove-distance-metric': self.cmd_remove_distance_metric,
            'clean-distance-tables': self.cmd_clean_distance_tables,
            # Distance function LUT management
            'show-distance-functions': self.cmd_show_distance_functions,
            'update-distance-function': self.cmd_update_distance_function,
            # MPCCTL distance calculation
            'mpcctl-distance-function': self.cmd_mpcctl_distance_function,
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
    
    # ========== Experiment Commands ==========
    
    def cmd_experiment_list(self, args):
        """List all experiments in the database"""
        try:
            from experiment_query_pg import ExperimentQueryPG
            query = ExperimentQueryPG()
            experiments = query.list_experiments()
            
            if not experiments:
                print("No experiments found")
                return
            
            print(f"\n📋 Available Experiments ({len(experiments)} total):")
            print("-" * 80)
            
            for exp in experiments:
                status_emoji = {
                    'completed': '✅',
                    'in_progress': '🔄',
                    'failed': '❌',
                    'initialized': '🆕'
                }.get(exp.get('status', ''), '❓')
                
                print(f"{status_emoji} Experiment {exp['experiment_id']:3d}: {exp['name'][:50]}")
                if exp.get('description'):
                    print(f"   {exp['description'][:70]}")
            
            query.disconnect()
            
        except Exception as e:
            print(f"❌ Error listing experiments: {e}")
    
    def cmd_experiment_info(self, args):
        """Show detailed information about an experiment"""
        if not args:
            # Use current experiment if no ID provided
            exp_id = self.current_experiment
        else:
            try:
                exp_id = int(args[0])
            except ValueError:
                print(f"❌ Invalid experiment ID: {args[0]}")
                return
        
        try:
            from experiment_query_pg import ExperimentQueryPG
            
            query = ExperimentQueryPG()
            query.print_experiment_summary(exp_id)
            
            # Also check for file training data
            # Create connection if needed
            db_conn = self.db_conn
            if not db_conn:
                try:
                    import psycopg2
                    db_conn = psycopg2.connect(
                        host='localhost',
                        port=5432,
                        database='arc_detection',
                        user='kjensen'
                    )
                    temp_conn = True
                except:
                    db_conn = None
                    temp_conn = False
            else:
                temp_conn = False
            
            if db_conn:
                table_name = f"experiment_{exp_id:03d}_file_training_data"
                cursor = db_conn.cursor()
                try:
                    # Check if training data table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """, (table_name,))
                    
                    if cursor.fetchone()[0]:
                        # Check which column name is used for labels
                        # Note: Experiment 18 uses 'assigned_label' (published data, cannot change)
                        # All other experiments should use 'file_label_name' (standard)
                        cursor.execute("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_name = %s
                            AND column_name IN ('assigned_label', 'file_label_name')
                            LIMIT 1
                        """, (table_name,))

                        label_column_result = cursor.fetchone()
                        if label_column_result:
                            label_column = label_column_result[0]

                            # Get file label statistics using the correct column
                            cursor.execute(f"""
                                SELECT
                                    {label_column} as file_label_name,
                                    COUNT(*) as count
                                FROM {table_name}
                                WHERE experiment_id = %s
                                GROUP BY {label_column}
                                ORDER BY count DESC, {label_column}
                            """, (exp_id,))

                            labels = cursor.fetchall()

                            if labels:
                                print("\n📁 FILE TRAINING DATA:")
                                print("=" * 60)

                                # Get total counts using the correct column
                                cursor.execute(f"""
                                    SELECT
                                        COUNT(DISTINCT file_id) as total_files,
                                        COUNT(DISTINCT {label_column}) as unique_labels
                                    FROM {table_name}
                                    WHERE experiment_id = %s
                                """, (exp_id,))

                                stats = cursor.fetchone()
                                print(f"Total files: {stats[0]}")
                                print(f"Unique labels: {stats[1]}")

                                # Show label distribution
                                print("\nLabel Distribution:")
                                for label_name, count in labels:
                                    bar_length = int(count / max(l[1] for l in labels) * 30)
                                    bar = '█' * bar_length
                                    print(f"  {label_name:30} {count:4} {bar}")

                                # Check for segment training data too
                                seg_table = f"experiment_{exp_id:03d}_segment_training_data"
                                cursor.execute("""
                                    SELECT EXISTS (
                                        SELECT 1 FROM information_schema.tables
                                        WHERE table_name = %s
                                    )
                                """, (seg_table,))

                                if cursor.fetchone()[0]:
                                    cursor.execute(f"""
                                        SELECT COUNT(*) FROM {seg_table}
                                        WHERE experiment_id = %s
                                    """, (exp_id,))
                                    seg_count = cursor.fetchone()[0]
                                    if seg_count > 0:
                                        print(f"\n📊 SEGMENT TRAINING DATA:")
                                        print("=" * 60)
                                        print(f"Total segments: {seg_count}")

                                        # Get segment label distribution by joining with data_segments
                                        cursor.execute(f"""
                                            SELECT
                                                COALESCE(sl.label_name, 'unlabeled') as label_name,
                                                COUNT(*) as count
                                            FROM {seg_table} st
                                            JOIN data_segments ds ON st.segment_id = ds.segment_id
                                            LEFT JOIN segment_labels sl ON ds.segment_label_id = sl.label_id
                                            GROUP BY sl.label_name
                                            ORDER BY count DESC, label_name
                                        """)

                                        seg_labels = cursor.fetchall()
                                        if seg_labels:
                                            print(f"Unique segment labels: {len(seg_labels)}")
                                            print("\nSegment Label Distribution:")
                                            max_seg_count = max(l[1] for l in seg_labels)
                                            for label_name, count in seg_labels:
                                                bar_length = int(count / max_seg_count * 30)
                                                bar = '█' * bar_length
                                                print(f"  {label_name:35} {count:4} {bar}")

                                        # Get position distribution from segment_selection_log if available
                                        cursor.execute("""
                                            SELECT EXISTS (
                                                SELECT 1 FROM information_schema.tables
                                                WHERE table_name = 'segment_selection_log'
                                            )
                                        """)

                                        if cursor.fetchone()[0]:
                                            cursor.execute("""
                                                SELECT
                                                    position_type,
                                                    COUNT(*) as count
                                                FROM segment_selection_log
                                                WHERE experiment_id = %s
                                                GROUP BY position_type
                                                ORDER BY position_type
                                            """, (exp_id,))

                                            positions = cursor.fetchall()
                                            if positions:
                                                print("\nPosition Distribution:")
                                                for pos, count in positions:
                                                    print(f"  {pos:10}: {count} segments")

                                        # Also get segment type distribution
                                        cursor.execute(f"""
                                            SELECT
                                                ds.segment_type,
                                                COUNT(*) as count
                                            FROM {seg_table} st
                                            JOIN data_segments ds ON st.segment_id = ds.segment_id
                                            GROUP BY ds.segment_type
                                            ORDER BY ds.segment_type
                                        """)

                                        seg_types = cursor.fetchall()
                                        if seg_types:
                                            print("\nSegment Type Distribution:")
                                            for seg_type, count in seg_types:
                                                print(f"  {seg_type:10}: {count} segments")

                                # Check for segment pairs too
                                pairs_table = f"experiment_{exp_id:03d}_segment_pairs"
                                cursor.execute("""
                                    SELECT EXISTS (
                                        SELECT 1 FROM information_schema.tables
                                        WHERE table_name = %s
                                    )
                                """, (pairs_table,))

                                if cursor.fetchone()[0]:
                                    cursor.execute(f"""
                                        SELECT COUNT(*) FROM {pairs_table}
                                        WHERE experiment_id = %s
                                    """, (exp_id,))
                                    pairs_count = cursor.fetchone()[0]
                                    if pairs_count > 0:
                                        print(f"🔗 Segment Pairs: {pairs_count} pairs generated")
                            
                except Exception as e:
                    # Silently continue if there's an error (table might not exist)
                    pass
                finally:
                    cursor.close()
                    # Close temporary connection if we created one
                    if temp_conn and db_conn:
                        db_conn.close()
            
            query.disconnect()
            
        except Exception as e:
            print(f"❌ Error getting experiment info: {e}")
    
    def cmd_experiment_config(self, args):
        """Get experiment configuration from database"""
        if not args:
            # Use current experiment if no ID provided
            exp_id = self.current_experiment
        else:
            try:
                exp_id = int(args[0])
            except ValueError:
                print(f"❌ Invalid experiment ID: {args[0]}")
                return
        
        output_json = '--json' in args
        
        try:
            # Try new configurator first for more detailed info
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(exp_id, db_config)
            config = configurator.get_current_config()
            
            if output_json:
                import json
                print(json.dumps(config, indent=2))
            else:
                print(f"\n📊 Configuration for Experiment {exp_id}:")
                print("-" * 60)
                
                # Show decimations
                if config.get('decimations'):
                    print(f"Decimations: {config['decimations']}")
                
                # Show segment sizes
                if config.get('segment_sizes'):
                    print(f"Segment sizes: {config['segment_sizes']}")
                
                # Show amplitude methods
                if config.get('amplitude_methods'):
                    print(f"Amplitude methods: {config['amplitude_methods']}")
                
                # Show feature sets
                if config.get('feature_sets'):
                    print(f"\nFeature Sets:")
                    for fs in config['feature_sets']:
                        print(f"  • {fs['name']}")
                        print(f"    Features: {fs['features']}")
                        if fs['n_values']:
                            print(f"    N values: {fs['n_values']}")
                
                print("-" * 60)
                for key, value in config.items():
                    if isinstance(value, list) and len(value) > 5:
                        print(f"{key:20}: {value[:5]} ... ({len(value)} items)")
                    elif isinstance(value, dict):
                        print(f"{key:20}: {len(value)} entries")
                    else:
                        print(f"{key:20}: {value}")
            
            query.disconnect()
            
        except ValueError:
            print(f"❌ Invalid experiment ID: {args[0]}")
        except Exception as e:
            print(f"❌ Error getting experiment config: {e}")
    
    def cmd_experiment_summary(self, args):
        """Show experiment summary with junction table data"""
        if not args:
            # Show summary of all experiments
            self.cmd_experiment_list([])
        else:
            # Show detailed summary of specific experiment
            self.cmd_experiment_info(args)
    
    def cmd_experiment_create(self, args):
        """Create a new experiment with full CLI specification"""
        try:
            from experiment_cli_builder import ExperimentCLIBuilder
            from experiment_creator import ExperimentCreator
            
            # Check for help
            if not args or '--help' in args:
                print("Usage: experiment-create --name <name> [options]")
                print("\nRequired:")
                print("  --name NAME                    Experiment name")
                print("\nFile Selection:")
                print("  --file-selection {random,all}  File selection strategy (default: random)")
                print("  --max-files N                  Maximum files to select (default: 50)")
                print("  --random-seed N                Random seed (default: 42)")
                print("  --min-examples N               Min examples per class (default: 25)")
                print("  --exclude-labels LABELS        Labels to exclude (default: trash current_only voltage_only other)")
                print("  --target-labels IDS            Specific label IDs (auto-detect if not specified)")
                print("\nSegment Configuration:")
                print("  --segment-sizes SIZES          Segment sizes (default: 8192)")
                print("  --decimations FACTORS          Decimation factors (default: 0)")
                print("  --data-types TYPES             Data types: raw adc6 adc8 adc10 adc12 adc14")
                print("\nProcessing Methods:")
                print("  --amplitude-methods METHODS    Amplitude methods (use 'all' for all available)")
                print("  --distance-functions FUNCS     Distance functions (use 'all' for all available)")
                print("\nSegment Selection:")
                print("  --min-segments-per-position N  Min segments per position (default: 1)")
                print("  --min-segments-per-file N      Min segments per file (default: 3)")
                print("  --position-balance-mode MODE   Balance mode: at_least_one, equal, proportional")
                print("\nOptions:")
                print("  --dry-run                      Validate without creating")
                print("  --force                        Skip confirmation")
                print("\nExample:")
                print("  experiment-create --name random_50files \\")
                print("    --max-files 50 --segment-sizes 128 1024 8192 \\")
                print("    --decimations 0 7 15 --data-types raw adc6 adc8 adc10 adc12 adc14 \\")
                print("    --amplitude-methods all --distance-functions all")
                return
            
            # Build configuration from CLI arguments
            builder = ExperimentCLIBuilder()
            config = builder.create_from_cli(args)
            
            # Validate
            if not config.validate():
                print("❌ Configuration validation failed")
                builder.close()
                return
            
            # Check dry-run
            if config.dry_run:
                print("\n✅ Configuration validated (dry-run mode)")
                builder.close()
                return
            
            # Confirm creation
            force = '--force' in args
            if not force:
                response = input("\nCreate experiment? (y/n): ")
                if response.lower() != 'y':
                    print("❌ Creation cancelled")
                    builder.close()
                    return
            
            # Create experiment
            creator = ExperimentCreator()
            experiment_id = creator.create_experiment(config)
            
            print(f"\n✅ Successfully created experiment {experiment_id}")
            print(f"📊 Experiment: {config.experiment_name}")
            
            # Show what was created
            info = creator.get_experiment_info(experiment_id)
            print(f"\nConfiguration applied:")
            print(f"  • Data Types: {len(info.get('data_types', []))}")
            print(f"  • Amplitude Methods: {len(info.get('amplitude_methods', []))}")
            print(f"  • Decimations: {len(info.get('decimations', []))}")
            print(f"  • Distance Functions: {len(info.get('distance_functions', []))}")
            
            builder.close()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def cmd_experiment_generate(self, args):
        """Generate a new experiment with configurable parameters"""
        try:
            from experiment_generation_config import (
                ExperimentGenerationConfig,
                BALANCED_18CLASS_CONFIG,
                SMALL_TEST_CONFIG,
                LARGE_UNBALANCED_CONFIG
            )
            from experiment_query_pg import ExperimentQueryPG
            import json
            
            # Parse arguments
            if not args:
                print("Usage: experiment-generate <config_name|config_file> [--dry-run]")
                print("\nAvailable configs:")
                print("  balanced    - 18 classes × 750 instances each")
                print("  small       - 3 classes × 100 instances (test)")
                print("  large       - 18 classes × 1000 instances (unbalanced)")
                print("  <file.json> - Load from JSON file")
                print("\nFor dynamic configuration, use: experiment-create --help")
                print("\nOptions:")
                print("  --dry-run   - Validate configuration without creating experiment")
                return
            
            config_name = args[0]
            dry_run = '--dry-run' in args
            
            # Load configuration
            if config_name == 'balanced':
                config = BALANCED_18CLASS_CONFIG
            elif config_name == 'small':
                config = SMALL_TEST_CONFIG
            elif config_name == 'large':
                config = LARGE_UNBALANCED_CONFIG
            elif config_name.endswith('.json'):
                try:
                    with open(config_name, 'r') as f:
                        config_data = json.load(f)
                    config = ExperimentGenerationConfig.from_dict(config_data)
                except FileNotFoundError:
                    print(f"❌ Configuration file not found: {config_name}")
                    return
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON in configuration file: {config_name}")
                    return
            else:
                print(f"❌ Unknown configuration: {config_name}")
                return
            
            # Set dry run mode
            config.dry_run = dry_run
            
            # Validate configuration
            if not config.validate():
                print("❌ Configuration validation failed")
                return
            
            # Display configuration summary
            print("\nExperiment Generation Configuration:")
            print("=" * 60)
            print(config.summary())
            print("=" * 60)
            
            if dry_run:
                print("\n🔍 DRY RUN MODE - No changes will be made")
            
            # Confirm generation
            if not dry_run:
                response = input("\nGenerate experiment? (y/n): ")
                if response.lower() != 'y':
                    print("❌ Generation cancelled")
                    return
            
            # Connect to database
            query_tool = ExperimentQueryPG()
            
            # Check if experiment name already exists
            existing = query_tool.execute_query(
                "SELECT experiment_id FROM ml_experiments WHERE experiment_name = %s",
                (config.experiment_name,)
            )
            
            if existing:
                print(f"❌ Experiment '{config.experiment_name}' already exists (ID: {existing[0][0]})")
                return
            
            print(f"\n✅ Configuration validated")
            print(f"📊 Will create experiment: {config.experiment_name}")
            print(f"📁 Target: {len(config.target_labels)} labels × {config.instances_per_label} instances")
            print(f"🎲 Selection: {config.selection_strategy} (seed={config.random_seed})")
            
            if dry_run:
                print("\n✅ Dry run completed successfully")
            else:
                # Create the experiment
                try:
                    from experiment_creator import ExperimentCreator
                    
                    creator = ExperimentCreator()
                    experiment_id = creator.create_experiment(config)
                    
                    print(f"\n✅ Successfully created experiment {experiment_id}")
                    print(f"📊 Experiment: {config.experiment_name}")
                    
                    # Show what was created
                    info = creator.get_experiment_info(experiment_id)
                    print(f"\nConfiguration applied:")
                    print(f"  • Data Types: {len(info.get('data_types', []))}")
                    print(f"  • Amplitude Methods: {len(info.get('amplitude_methods', []))}")
                    print(f"  • Decimations: {len(info.get('decimations', []))}")
                    print(f"  • Distance Functions: {len(info.get('distance_functions', []))}")
                    
                    print(f"\n📁 Next steps:")
                    print(f"  1. Run segment selection: experiment-select {experiment_id}")
                    print(f"  2. Generate segment files: experiment-generate-files {experiment_id}")
                    print(f"  3. Calculate distances: experiment-calculate-distances {experiment_id}")
                    print(f"  4. View progress: experiment-info {experiment_id}")
                    
                except ImportError as e:
                    print(f"❌ Failed to import experiment creator: {e}")
                except Exception as e:
                    print(f"❌ Failed to create experiment: {e}")
            
        except ImportError as e:
            print(f"❌ Failed to import required modules: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
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

🧪 EXPERIMENT COMMANDS:
  experiment-list                     List all experiments in database
  experiment-info <id>                Show detailed experiment information
  experiment-config <id> [--json]     Get experiment configuration
  experiment-summary [id]             Show experiment summary
  experiment-generate <config>        Generate new experiment (balanced|small|large)
  experiment-create --name <name>     Create experiment with full CLI specification

🔧 EXPERIMENT CONFIGURATION:
  update-decimations <d1> <d2>...     Update decimation factors
  update-segment-sizes <s1> <s2>...   Update segment sizes
  update-amplitude-methods <m1>...    Update amplitude/ADC methods
  update-selection-config [options]   Update segment selection parameters
  create-feature-set --name <n>       Create custom feature set
  add-feature-set <ids> [options]     Add feature sets (--n N --channel source_current|load_voltage)
  list-feature-sets                   List feature sets for current experiment
  show-all-feature-sets                Show ALL feature sets in database
  remove-feature-set <id>              Remove a feature set from experiment
  clear-feature-sets                   Remove ALL feature sets from experiment
  select-files [--max-files N]        Select files for training data
  remove-file-labels <label1>...      Remove files with specified labels from training data
  remove-files <id1> <id2>...         Remove specific files by ID from training data
  remove-segments <id1> <id2>...      Remove specific segments by ID from training data

📐 DISTANCE OPERATIONS:
  calculate [options]                 Calculate distances using mpcctl
  insert_distances [options]          Insert distances into database
  stats [distance_type]               Show distance statistics
  closest [N]                         Find N closest segment pairs

🎨 VISUALIZATION:
  heatmap [--version N]               Generate distance heatmap
  histogram [--version] [--bins]      Generate distance histogram
  visualize --segment-id ID           Visualize segment data

🔬 ML PIPELINE COMMANDS:
  select-files                        Select files for training (DB table)
  select-segments                     Select segments for training (DB table)
  generate-segment-pairs              Generate segment pairs (DB table)
  generate-segment-fileset            Generate physical segment files from raw data
  generate-feature-fileset            Extract features and save to disk

🔍 SEGMENT COMMANDS:
  segment-generate                    Generate segments from raw data
  show-segment-status                 Show segment generation status
  segment-test                        Test segment generation with small dataset
  validate-segments                   Validate generated segments
  segment-plot                        Plot segment data

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
    
    def cmd_select_segments(self, args):
        """Select segments for training with proper segment code balancing"""
        # Parse experiment ID if provided, otherwise use current
        if args and args[0].isdigit():
            experiment_id = int(args[0])
            args = args[1:]  # Remove experiment ID from args
        else:
            experiment_id = self.current_experiment

        if not experiment_id:
            print("❌ No experiment specified. Use: select-segments <experiment_id> [options]")
            print("   Or set current experiment: set experiment <id>")
            return

        # Parse options
        strategy = 'balanced'  # Default
        segments_per_type = 3  # Default for fixed_per_type strategy
        seed = 42

        i = 0
        while i < len(args):
            if args[i] == '--strategy' and i + 1 < len(args):
                strategy = args[i + 1]
                i += 2
            elif args[i] == '--segments-per-type' and i + 1 < len(args):
                segments_per_type = int(args[i + 1])
                i += 2
            elif args[i] == '--seed' and i + 1 < len(args):
                seed = int(args[i + 1])
                i += 2
            elif args[i] == '--help':
                print("\nUsage: select-segments [experiment_id] [options]")
                print("\nOptions:")
                print("  --strategy STRAT           Selection strategy (default: balanced)")
                print("                             balanced: Find min count across segment types,")
                print("                                      select that number from EACH type")
                print("                             fixed_per_type: Select N segments from each type")
                print("                             proportional: Select proportionally from each type")
                print("  --segments-per-type N      For fixed_per_type: segments to select per type (default: 3)")
                print("  --seed N                   Random seed (default: 42)")
                print("\n📊 BALANCED STRATEGY (recommended):")
                print("  Per file: Groups segments by code type (L, R, C, Cm, Cl, Cr, etc.)")
                print("  Example: File has L=45, R=40, C=5, Cm=25, Cl=3, Cr=2 segments")
                print("  → Finds minimum: min(45,40,5,25,3,2) = 2")
                print("  → Selects 2 from EACH type: 2L + 2R + 2C + 2Cm + 2Cl + 2Cr = 12 total")
                print("\nExamples:")
                print("  select-segments 41 --strategy balanced")
                print("  select-segments 41 --strategy fixed_per_type --segments-per-type 5")
                print("  select-segments --strategy balanced  (uses current experiment)")
                return
            else:
                i += 1

        print(f"🔄 Selecting segments for experiment {experiment_id}...")
        print(f"   Strategy: {strategy}")
        if strategy == 'fixed_per_type':
            print(f"   Segments per type: {segments_per_type}")
        elif strategy == 'balanced':
            print(f"   Will select minimum count across all segment types from EACH type")
        print(f"   Random seed: {seed}")

        try:
            # Try to use the improved v2 selector first
            try:
                from experiment_segment_selector_v2 import SegmentSelectorV2
                use_v2 = True
            except ImportError:
                # Fallback to original if v2 not available
                from experiment_segment_selector import ExperimentSegmentSelector
                use_v2 = False
                print("⚠️  Using legacy selector. For better results, ensure experiment_segment_selector_v2.py is available.")

            # Database configuration
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            # Run selection with appropriate selector
            if use_v2:
                selector = SegmentSelectorV2(experiment_id, db_config)
                result = selector.run_selection(
                    strategy=strategy,
                    segments_per_type=segments_per_type
                )
            else:
                # Fallback to old selector
                selector = ExperimentSegmentSelector(experiment_id, db_config)
                result = selector.run_selection()

            # Display results
            if result and 'total_segments' in result:
                print(f"\n✅ Successfully selected {result['total_segments']} segments")
                print(f"   From {result.get('total_files', 0)} files")

                # Show average per file
                if result.get('total_files', 0) > 0:
                    avg_per_file = result['total_segments'] / result['total_files']
                    print(f"   Average per file: {avg_per_file:.1f}")

                # Show segment type distribution from v2 selector
                if 'segments_by_type' in result:
                    print("\n📊 Segment type distribution:")
                    for code_type, count in sorted(result['segments_by_type'].items()):
                        print(f"     {code_type}: {count} segments")

                # Show strategy used
                if 'strategy' in result:
                    print(f"\n📋 Selection strategy: {result['strategy']}")

                print(f"\n💾 Data saved to:")
                print(f"   experiment_{experiment_id:03d}_segment_training_data")
            else:
                print(f"❌ Failed to select segments")
                if isinstance(result, dict) and 'error' in result:
                    print(f"   Error: {result['error']}")

        except ImportError as e:
            print(f"❌ Could not import segment selector: {e}")
            print("Make sure experiment_segment_selector_v2.py is in the same directory")
        except Exception as e:
            print(f"❌ Error during segment selection: {e}")
    
    def cmd_update_decimations(self, args):
        """Update decimation factors for current experiment"""
        if not args:
            print("Usage: update-decimations <decimation1> <decimation2> ...")
            print("Example: update-decimations 0 7 15")
            return
        
        try:
            decimations = [int(arg) for arg in args]
        except ValueError:
            print(f"❌ Invalid decimation values. Must be integers.")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Updating decimations for experiment {self.current_experiment}...")
            if configurator.update_decimations(decimations):
                print(f"✅ Decimations updated: {decimations}")
            else:
                print(f"❌ Failed to update decimations")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error updating decimations: {e}")
    
    def cmd_update_segment_sizes(self, args):
        """Update segment sizes for current experiment"""
        if not args:
            print("Usage: update-segment-sizes <size1> <size2> ...")
            print("Example: update-segment-sizes 128 1024 8192")
            return
        
        try:
            sizes = [int(arg) for arg in args]
        except ValueError:
            print(f"❌ Invalid segment sizes. Must be integers.")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Updating segment sizes for experiment {self.current_experiment}...")
            if configurator.update_segment_sizes(sizes):
                print(f"✅ Segment sizes updated: {sizes}")
            else:
                print(f"❌ Failed to update segment sizes")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error updating segment sizes: {e}")
    
    def cmd_update_amplitude_methods(self, args):
        """Update amplitude methods for current experiment"""
        if not args:
            print("Usage: update-amplitude-methods <method1> <method2> ...")
            print("Example: update-amplitude-methods minmax zscore")
            print("Available: minmax, zscore, maxabs, robust, TRAW, TADC14, TADC12, TADC10, TADC8, TADC6")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Updating amplitude methods for experiment {self.current_experiment}...")
            if configurator.update_amplitude_methods(args):
                print(f"✅ Amplitude methods updated: {args}")
            else:
                print(f"❌ Failed to update amplitude methods")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error updating amplitude methods: {e}")
    
    def cmd_create_feature_set(self, args):
        """Create a custom feature set for current experiment"""
        if not args or '--name' not in args or '--features' not in args:
            print("Usage: create-feature-set --name <name> --features <feature1,feature2,...> [--n-value <n>]")
            print("Example: create-feature-set --name voltage_variance --features voltage,variance(voltage) --n-value 128")
            return
        
        try:
            # Parse arguments
            name = None
            features = None
            n_value = 128  # Default
            
            i = 0
            while i < len(args):
                if args[i] == '--name' and i + 1 < len(args):
                    name = args[i + 1]
                    i += 2
                elif args[i] == '--features' and i + 1 < len(args):
                    features = args[i + 1].split(',')
                    i += 2
                elif args[i] == '--n-value' and i + 1 < len(args):
                    n_value = int(args[i + 1])
                    i += 2
                else:
                    i += 1
            
            if not name or not features:
                print("❌ Both --name and --features are required")
                return
            
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Creating feature set '{name}' for experiment {self.current_experiment}...")
            feature_set_id = configurator.create_feature_set(name, features, n_value)
            
            if feature_set_id:
                print(f"✅ Feature set created (ID: {feature_set_id})")
                print(f"   Name: {name}")
                print(f"   Features: {', '.join(features)}")
                print(f"   N value: {n_value}")
            else:
                print(f"❌ Failed to create feature set")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error creating feature set: {e}")
    
    def cmd_add_feature_set(self, args):
        """Add existing feature set(s) to current experiment"""
        if not args or '--help' in args:
            print("Usage: add-feature-set <feature_set_id> [options]")
            print("   or: add-feature-set <id1,id2,id3,...> [options]")
            print("\nOptions:")
            print("  --n <value>          N value for chunk size")
            print("  --channel <channel>  Data channel: source_current or load_voltage (default: load_voltage)")
            print("\nExamples:")
            print("  add-feature-set 3                              # Add with defaults")
            print("  add-feature-set 3 --n 1024                     # With N=1024")
            print("  add-feature-set 3 --channel source_current     # From source current")
            print("  add-feature-set 1,2,3,4 --channel load_voltage --n 8192")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            config = ExperimentConfigurator(self.current_experiment, db_config)
            
            # Parse arguments
            ids_arg = args[0]
            n_value = None
            data_channel = 'load_voltage'
            
            # Parse optional arguments
            i = 1
            while i < len(args):
                if args[i] == '--n' and i + 1 < len(args):
                    n_value = int(args[i + 1])
                    i += 2
                elif args[i] == '--channel' and i + 1 < len(args):
                    data_channel = args[i + 1]
                    if data_channel not in ['source_current', 'load_voltage']:
                        print(f"❌ Invalid channel: {data_channel}")
                        print("   Must be 'source_current' or 'load_voltage'")
                        return
                    i += 2
                else:
                    # Legacy support for positional N value
                    if i == 1 and args[i].isdigit():
                        n_value = int(args[i])
                    i += 1
            
            # Check if comma-separated list
            if ',' in ids_arg:
                # Multiple feature sets
                feature_set_ids = [int(id.strip()) for id in ids_arg.split(',')]
                
                print(f"🔄 Adding {len(feature_set_ids)} feature sets to experiment {self.current_experiment}...")
                print(f"   Data channel: {data_channel}")
                if n_value:
                    print(f"   Using N value: {n_value}")
                
                results = config.add_multiple_feature_sets(feature_set_ids, n_value, data_channel)
                
                # Report results
                success_count = sum(1 for success in results.values() if success)
                print(f"\n✅ Successfully added {success_count}/{len(feature_set_ids)} feature sets")
                
                for fs_id, success in results.items():
                    if not success:
                        print(f"   ⚠️  Feature set {fs_id} was already linked or doesn't exist")
            else:
                # Single feature set
                feature_set_id = int(ids_arg)
                
                print(f"🔄 Adding feature set {feature_set_id} to experiment {self.current_experiment}...")
                print(f"   Data channel: {data_channel}")
                if n_value:
                    print(f"   Using N value: {n_value}")
                
                if config.add_feature_set(feature_set_id, n_value, data_channel):
                    print(f"✅ Feature set {feature_set_id} added successfully")
                else:
                    print(f"⚠️  Feature set {feature_set_id} is already linked or doesn't exist")
            
            config.disconnect()
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            print("Feature set IDs and N value must be integers")
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error adding feature set: {e}")
    
    def cmd_remove_feature_set(self, args):
        """Remove a feature set from current experiment"""
        if not args:
            print("Usage: remove-feature-set <feature_set_id>")
            print("Use 'list-feature-sets' to see IDs")
            return
        
        try:
            feature_set_id = int(args[0])
        except ValueError:
            print(f"❌ Invalid feature set ID: {args[0]}")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Removing feature set {feature_set_id} from experiment {self.current_experiment}...")
            if configurator.remove_feature_set(feature_set_id):
                print(f"✅ Feature set {feature_set_id} removed")
            else:
                print(f"❌ Failed to remove feature set")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error removing feature set: {e}")
    
    def cmd_clear_feature_sets(self, args):
        """Remove all feature sets from current experiment"""
        response = input(f"⚠️  Remove ALL feature sets from experiment {self.current_experiment}? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Clearing all feature sets from experiment {self.current_experiment}...")
            if configurator.clear_all_feature_sets():
                print(f"✅ All feature sets cleared")
            else:
                print(f"❌ Failed to clear feature sets")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error clearing feature sets: {e}")
    
    def cmd_list_feature_sets(self, args):
        """List feature sets for current experiment"""
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            config = configurator.get_current_config()
            
            feature_sets = config.get('feature_sets', [])
            
            if not feature_sets:
                print(f"No feature sets linked to experiment {self.current_experiment}")
                return
            
            print(f"\n🧬 Feature Sets for Experiment {self.current_experiment}:")
            print("-" * 60)
            
            for fs in feature_sets:
                print(f"• ID {fs.get('id', '?')}: {fs['name']}")
                print(f"  Features: {fs['features']}")
                print(f"  Data channel: {fs.get('data_channel', 'load_voltage')}")
                if fs['n_values']:
                    print(f"  N values: {fs['n_values']}")
            
            print("-" * 60)
            print(f"Total: {len(feature_sets)} feature sets")
            print("\nUse 'remove-feature-set <id>' to remove a specific set")
            print("Use 'clear-feature-sets' to remove all")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error listing feature sets: {e}")
    
    def cmd_show_all_feature_sets(self, args):
        """Show all available feature sets in the database"""
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get all feature sets from the lookup table
            cursor.execute("""
                SELECT 
                    fsl.feature_set_id,
                    fsl.feature_set_name,
                    fsl.num_features,
                    fsl.category,
                    fsl.description,
                    STRING_AGG(fl.feature_name || ' (' || fl.behavior_type || ')', ', ' ORDER BY fsf.feature_order) as features
                FROM ml_feature_sets_lut fsl
                LEFT JOIN ml_feature_set_features fsf ON fsl.feature_set_id = fsf.feature_set_id
                LEFT JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                GROUP BY fsl.feature_set_id, fsl.feature_set_name, fsl.num_features, fsl.category, fsl.description
                ORDER BY fsl.feature_set_id
            """)
            
            results = cursor.fetchall()
            
            if not results:
                print("No feature sets found in database")
                return
            
            print(f"\n📚 ALL AVAILABLE FEATURE SETS IN DATABASE:")
            print("=" * 70)
            
            for fs in results:
                print(f"\n📦 ID {fs['feature_set_id']}: {fs['feature_set_name']}")
                print(f"   Category: {fs['category']}")
                if fs['description']:
                    print(f"   Description: {fs['description']}")
                print(f"   Number of features: {fs['num_features']}")
                
                if fs['features']:
                    features_str = fs['features']
                    if len(features_str) > 150:
                        # Truncate long feature lists
                        feature_list = features_str.split(', ')[:3]
                        print(f"   Features: {', '.join(feature_list)}...")
                        print(f"             (and {len(features_str.split(', ')) - 3} more)")
                    else:
                        print(f"   Features: {features_str}")
                
                # Check which experiments use this feature set
                cursor.execute("""
                    SELECT ARRAY_AGG(DISTINCT experiment_id ORDER BY experiment_id) as experiments
                    FROM ml_experiments_feature_sets
                    WHERE feature_set_id = %s
                """, (fs['feature_set_id'],))
                exp_result = cursor.fetchone()
                if exp_result and exp_result['experiments']:
                    print(f"   Used by experiments: {exp_result['experiments']}")
            
            print("\n" + "=" * 70)
            print(f"Total: {len(results)} feature sets available")
            print("\nTo link a feature set to current experiment, create it with:")
            print("  create-feature-set --name <name> --features <f1,f2,...>")
            
            cursor.close()
            conn.close()
                
        except psycopg2.Error as e:
            print(f"❌ Database error: {e}")
        except Exception as e:
            print(f"❌ Error showing feature sets: {e}")

    def cmd_create_feature(self, args):
        """Create a new feature in ml_features_lut"""
        if not args or '--name' not in args:
            print("Usage: create-feature --name <name> --category <category> --behavior <behavior> [--description <desc>]")
            print("\nCategories: electrical, statistical, spectral, temporal, compute")
            print("Behaviors: driver, derived, aggregate, transform")
            print("\nExample: create-feature --name impedance --category electrical --behavior derived --description 'Electrical impedance Z=V/I'")
            return

        name = None
        category = 'electrical'
        behavior = 'driver'
        description = None

        i = 0
        while i < len(args):
            if args[i] == '--name' and i + 1 < len(args):
                name = args[i + 1]
                i += 2
            elif args[i] == '--category' and i + 1 < len(args):
                category = args[i + 1]
                i += 2
            elif args[i] == '--behavior' and i + 1 < len(args):
                behavior = args[i + 1]
                i += 2
            elif args[i] == '--description' and i + 1 < len(args):
                description = args[i + 1]
                i += 2
            else:
                i += 1

        if not name:
            print("❌ Feature name is required")
            return

        valid_categories = ['electrical', 'statistical', 'spectral', 'temporal', 'compute']
        if category not in valid_categories:
            print(f"❌ Invalid category: {category}")
            print(f"   Must be one of: {', '.join(valid_categories)}")
            return

        valid_behaviors = ['driver', 'derived', 'aggregate', 'transform']
        if behavior not in valid_behaviors:
            print(f"❌ Invalid behavior: {behavior}")
            print(f"   Must be one of: {', '.join(valid_behaviors)}")
            return

        try:
            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("SELECT feature_id FROM ml_features_lut WHERE feature_name = %s", (name,))
            if cursor.fetchone():
                print(f"❌ Feature '{name}' already exists")
                conn.close()
                return

            cursor.execute("SELECT COALESCE(MAX(feature_id), 0) + 1 FROM ml_features_lut")
            feature_id = cursor.fetchone()[0]

            cursor.execute("""
                INSERT INTO ml_features_lut
                (feature_id, feature_name, feature_category, behavior_type, description, is_active, created_at)
                VALUES (%s, %s, %s, %s, %s, true, CURRENT_TIMESTAMP)
            """, (feature_id, name, category, behavior, description or f"{name} feature"))

            conn.commit()
            print(f"✅ Created feature '{name}' (ID: {feature_id})")
            print(f"   Category: {category}")
            print(f"   Behavior: {behavior}")
            if description:
                print(f"   Description: {description}")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error creating feature: {e}")

    def cmd_list_features(self, args):
        """List all available features"""
        category_filter = None
        if args and '--category' in args:
            idx = args.index('--category')
            if idx + 1 < len(args):
                category_filter = args[idx + 1]

        try:
            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            query = """
                SELECT
                    feature_id,
                    feature_name,
                    feature_category,
                    behavior_type,
                    description,
                    is_active
                FROM ml_features_lut
            """
            params = []

            if category_filter:
                query += " WHERE feature_category = %s"
                params.append(category_filter)

            query += " ORDER BY feature_category, feature_id"

            cursor.execute(query, params)
            features = cursor.fetchall()

            if not features:
                if category_filter:
                    print(f"No features found in category '{category_filter}'")
                else:
                    print("No features found in database")
                return

            from collections import defaultdict
            by_category = defaultdict(list)
            for f in features:
                by_category[f['feature_category']].append(f)

            print("\n📊 Available Features:")
            print("=" * 80)

            for category in sorted(by_category.keys()):
                print(f"\n🏷️  {category.upper()} Features:")
                print("-" * 40)

                for f in by_category[category]:
                    status = "✓" if f['is_active'] else "✗"
                    print(f"  {status} ID {f['feature_id']:3d}: {f['feature_name']:20s} ({f['behavior_type']:10s})")
                    if f['description'] and f['description'] != f'{f["feature_name"]} feature':
                        print(f"           {f['description'][:60]}")

            print("\n" + "=" * 80)
            print(f"Total: {len(features)} features")

            if not category_filter:
                print("\nFilter by category: list-features --category <category>")
                print("Categories: electrical, statistical, spectral, temporal, compute")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error listing features: {e}")

    def cmd_show_feature(self, args):
        """Show details of a specific feature"""
        if not args:
            print("Usage: show-feature <feature_id|feature_name>")
            return

        try:
            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            feature_arg = args[0]
            if feature_arg.isdigit():
                cursor.execute("SELECT * FROM ml_features_lut WHERE feature_id = %s", (int(feature_arg),))
            else:
                cursor.execute("SELECT * FROM ml_features_lut WHERE feature_name = %s", (feature_arg,))

            feature = cursor.fetchone()

            if not feature:
                print(f"❌ Feature '{feature_arg}' not found")
                return

            print(f"\n📊 Feature Details:")
            print("=" * 60)
            print(f"ID:           {feature['feature_id']}")
            print(f"Name:         {feature['feature_name']}")
            print(f"Category:     {feature.get('feature_category', 'N/A')}")
            print(f"Behavior:     {feature.get('behavior_type', 'N/A')}")
            print(f"Active:       {'✓' if feature.get('is_active', False) else '✗'}")
            print(f"Description:  {feature.get('description', 'N/A')}")
            print(f"Created:      {feature.get('created_at', 'N/A')}")

            cursor.execute("""
                SELECT
                    fs.feature_set_id,
                    fs.feature_set_name
                FROM ml_feature_set_features fsf
                JOIN ml_feature_sets_lut fs ON fsf.feature_set_id = fs.feature_set_id
                WHERE fsf.feature_id = %s
                ORDER BY fs.feature_set_id
            """, (feature['feature_id'],))

            feature_sets = cursor.fetchall()
            if feature_sets:
                print(f"\nUsed in {len(feature_sets)} feature set(s):")
                for fs in feature_sets:
                    print(f"  • ID {fs['feature_set_id']}: {fs['feature_set_name']}")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error showing feature: {e}")

    def cmd_update_feature(self, args):
        """Update feature properties"""
        if not args or len(args) < 2:
            print("Usage: update-feature <feature_id> [--name <name>] [--category <category>] [--description <desc>]")
            return

        try:
            feature_id = int(args[0])

            updates = {}
            i = 1
            while i < len(args):
                if args[i] == '--name' and i + 1 < len(args):
                    updates['feature_name'] = args[i + 1]
                    i += 2
                elif args[i] == '--category' and i + 1 < len(args):
                    updates['feature_category'] = args[i + 1]
                    i += 2
                elif args[i] == '--description' and i + 1 < len(args):
                    updates['description'] = args[i + 1]
                    i += 2
                else:
                    i += 1

            if not updates:
                print("❌ No updates specified")
                return

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            set_clauses = []
            params = []
            for key, value in updates.items():
                set_clauses.append(f"{key} = %s")
                params.append(value)
            params.append(feature_id)

            query = f"UPDATE ml_features_lut SET {', '.join(set_clauses)} WHERE feature_id = %s"
            cursor.execute(query, params)

            if cursor.rowcount == 0:
                print(f"❌ Feature {feature_id} not found")
            else:
                conn.commit()
                print(f"✅ Updated feature {feature_id}")
                for key, value in updates.items():
                    print(f"   {key}: {value}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid feature ID")
        except Exception as e:
            print(f"❌ Error updating feature: {e}")

    def cmd_delete_feature(self, args):
        """Delete a feature if not in use"""
        if not args:
            print("Usage: delete-feature <feature_id>")
            return

        try:
            feature_id = int(args[0])

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM ml_feature_set_features WHERE feature_id = %s
            """, (feature_id,))

            count = cursor.fetchone()[0]
            if count > 0:
                print(f"❌ Cannot delete feature {feature_id}: used in {count} feature set(s)")
                print("   Remove from feature sets first using 'remove-features-from-set'")
                return

            cursor.execute("DELETE FROM ml_features_lut WHERE feature_id = %s", (feature_id,))

            if cursor.rowcount == 0:
                print(f"❌ Feature {feature_id} not found")
            else:
                conn.commit()
                print(f"✅ Deleted feature {feature_id}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid feature ID")
        except Exception as e:
            print(f"❌ Error deleting feature: {e}")

    def cmd_create_global_feature_set(self, args):
        """Create a feature set without linking to any experiment"""
        if not args or '--name' not in args:
            print("Usage: create-global-feature-set --name <name> [--category <category>] [--description <desc>]")
            print("\nExample: create-global-feature-set --name basic_electrical --category electrical --description 'Basic electrical measurements'")
            return

        name = None
        category = 'custom'
        description = None

        i = 0
        while i < len(args):
            if args[i] == '--name' and i + 1 < len(args):
                name = args[i + 1]
                i += 2
            elif args[i] == '--category' and i + 1 < len(args):
                category = args[i + 1]
                i += 2
            elif args[i] == '--description' and i + 1 < len(args):
                description = args[i + 1]
                i += 2
            else:
                i += 1

        if not name:
            print("❌ Feature set name is required")
            return

        try:
            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("SELECT feature_set_id FROM ml_feature_sets_lut WHERE feature_set_name = %s", (name,))
            if cursor.fetchone():
                print(f"❌ Feature set '{name}' already exists")
                conn.close()
                return

            cursor.execute("SELECT COALESCE(MAX(feature_set_id), 0) + 1 FROM ml_feature_sets_lut")
            feature_set_id = cursor.fetchone()[0]

            cursor.execute("""
                INSERT INTO ml_feature_sets_lut
                (feature_set_id, feature_set_name, category, description, is_active, created_at)
                VALUES (%s, %s, %s, %s, true, CURRENT_TIMESTAMP)
            """, (feature_set_id, name, category, description or f"{name} feature set"))

            conn.commit()
            print(f"✅ Created global feature set '{name}' (ID: {feature_set_id})")
            print(f"   Category: {category}")
            if description:
                print(f"   Description: {description}")
            print(f"\nNext: Add features using: add-features-to-set {feature_set_id} --features <id1,id2,...>")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error creating feature set: {e}")

    def cmd_add_features_to_set(self, args):
        """Add features to an existing feature set with optional per-feature overrides"""
        if not args or len(args) < 2 or '--features' not in args:
            print("Usage: add-features-to-set <feature_set_id> --features <feature_id1,feature_id2,...> [--channels <ch1,ch2,...>] [--n-values <n1,n2,...>]")
            print("\nExample: add-features-to-set 15 --features 1,2,3,4")
            print("         add-features-to-set 15 --features 2,2,2,2 --channels load_voltage,source_current,impedance,power")
            print("         add-features-to-set 15 --features 2,5 --channels impedance,null --n-values 128,null")
            print("\nUse 'list-features' to see available feature IDs")
            print("\nChannels: source_current, load_voltage, impedance, power, null (inherit from set)")
            return

        try:
            feature_set_id = int(args[0])

            features = []
            channels = []
            n_values = []

            # Parse --features
            if '--features' in args:
                idx = args.index('--features')
                if idx + 1 < len(args):
                    features = [int(f.strip()) for f in args[idx + 1].split(',')]

            # Parse --channels
            if '--channels' in args:
                idx = args.index('--channels')
                if idx + 1 < len(args):
                    channels = [ch.strip() if ch.strip().lower() not in ['default', 'null', 'none'] else None
                               for ch in args[idx + 1].split(',')]

            # Parse --n-values
            if '--n-values' in args:
                idx = args.index('--n-values')
                if idx + 1 < len(args):
                    n_values = [int(n.strip()) if n.strip().lower() not in ['default', 'null', 'none'] else None
                               for n in args[idx + 1].split(',')]

            if not features:
                print("❌ No features specified")
                return

            # Validate counts match
            if channels and len(channels) != len(features):
                print(f"❌ Channel count ({len(channels)}) must match feature count ({len(features)})")
                return

            if n_values and len(n_values) != len(features):
                print(f"❌ N-value count ({len(n_values)}) must match feature count ({len(features)})")
                return

            # Validate channels
            valid_channels = ['source_current', 'load_voltage', 'impedance', 'power', 'source_current,load_voltage']
            for ch in channels:
                if ch is not None and ch not in valid_channels:
                    print(f"❌ Invalid channel: {ch}")
                    print(f"   Must be one of: {', '.join(valid_channels)}")
                    return

            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute("SELECT feature_set_name FROM ml_feature_sets_lut WHERE feature_set_id = %s", (feature_set_id,))
            result = cursor.fetchone()
            if not result:
                print(f"❌ Feature set {feature_set_id} does not exist")
                conn.close()
                return

            feature_set_name = result['feature_set_name']

            cursor.execute("SELECT feature_id, feature_name FROM ml_features_lut WHERE feature_id = ANY(%s)", (features,))
            valid_features = {row['feature_id']: row['feature_name'] for row in cursor}

            invalid = [f for f in features if f not in valid_features]
            if invalid:
                print(f"❌ Invalid feature IDs: {invalid}")
                conn.close()
                return

            cursor.execute("""
                SELECT COALESCE(MAX(feature_order), 0) as max_order
                FROM ml_feature_set_features
                WHERE feature_set_id = %s
            """, (feature_set_id,))
            max_order = cursor.fetchone()['max_order']

            added = []
            skipped = []
            for i, feature_id in enumerate(features, 1):
                channel = channels[i-1] if channels and (i-1) < len(channels) else None
                n_value = n_values[i-1] if n_values and (i-1) < len(n_values) else None

                try:
                    cursor.execute("""
                        INSERT INTO ml_feature_set_features
                        (feature_set_id, feature_id, feature_order, data_channel, n_value_override)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (feature_set_id, feature_id, max_order + i, channel, n_value))

                    override_info = []
                    if channel:
                        override_info.append(f"ch={channel}")
                    if n_value:
                        override_info.append(f"n={n_value}")
                    info = f" [{', '.join(override_info)}]" if override_info else ""
                    added.append(f"{valid_features[feature_id]}{info}")
                    conn.commit()
                except psycopg2.IntegrityError:
                    skipped.append(valid_features[feature_id])
                    conn.rollback()

            print(f"✅ Updated feature set '{feature_set_name}' (ID: {feature_set_id})")
            if added:
                print(f"   Added {len(added)} features: {', '.join(added)}")
            if skipped:
                print(f"   Skipped {len(skipped)} (already in set): {', '.join(skipped)}")

            cursor.execute("""
                SELECT fl.feature_name
                FROM ml_feature_set_features fsf
                JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                WHERE fsf.feature_set_id = %s
                ORDER BY fsf.feature_order
            """, (feature_set_id,))

            all_features = [row['feature_name'] for row in cursor]
            print(f"\n   Total features in set: {len(all_features)}")
            print(f"   Features: {', '.join(all_features)}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid feature set ID or feature IDs")
        except Exception as e:
            print(f"❌ Error adding features: {e}")

    def cmd_remove_features_from_set(self, args):
        """Remove features from a feature set"""
        if not args or len(args) < 2 or '--features' not in args:
            print("Usage: remove-features-from-set <feature_set_id> --features <feature_id1,feature_id2,...>")
            return

        try:
            feature_set_id = int(args[0])

            features = []
            idx = args.index('--features')
            if idx + 1 < len(args):
                features = [int(f.strip()) for f in args[idx + 1].split(',')]

            if not features:
                print("❌ No features specified")
                return

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM ml_feature_set_features
                WHERE feature_set_id = %s AND feature_id = ANY(%s)
            """, (feature_set_id, features))

            removed = cursor.rowcount
            if removed > 0:
                conn.commit()
                print(f"✅ Removed {removed} feature(s) from feature set {feature_set_id}")

                cursor.execute("""
                    WITH reordered AS (
                        SELECT feature_set_id, feature_id,
                               ROW_NUMBER() OVER (PARTITION BY feature_set_id ORDER BY feature_order) as new_order
                        FROM ml_feature_set_features
                        WHERE feature_set_id = %s
                    )
                    UPDATE ml_feature_set_features fsf
                    SET feature_order = r.new_order
                    FROM reordered r
                    WHERE fsf.feature_set_id = r.feature_set_id
                      AND fsf.feature_id = r.feature_id
                """, (feature_set_id,))
                conn.commit()
                print("   Reordered remaining features")
            else:
                print(f"❌ No features removed (not found in set)")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid feature set ID or feature IDs")
        except Exception as e:
            print(f"❌ Error removing features: {e}")

    def cmd_update_feature_in_set(self, args):
        """Update feature assignment in a feature set"""
        if not args or len(args) < 2:
            print("Usage: update-feature-in-set <feature_set_id> <feature_id> [--channel <ch>] [--n-value <n>] [--order <order>]")
            print("\nExamples:")
            print("  update-feature-in-set 15 2 --channel impedance")
            print("  update-feature-in-set 15 2 --n-value 256")
            print("  update-feature-in-set 15 2 --channel null  (clear override, inherit from set)")
            print("  update-feature-in-set 15 2 --channel power --n-value 512 --order 3")
            return

        try:
            feature_set_id = int(args[0])
            feature_id = int(args[1])

            updates = {}
            i = 2
            while i < len(args):
                if args[i] == '--channel' and i + 1 < len(args):
                    value = args[i + 1]
                    updates['data_channel'] = None if value.lower() in ['null', 'none', 'default'] else value
                    i += 2
                elif args[i] == '--n-value' and i + 1 < len(args):
                    value = args[i + 1]
                    updates['n_value_override'] = None if value.lower() in ['null', 'none', 'default'] else int(value)
                    i += 2
                elif args[i] == '--order' and i + 1 < len(args):
                    updates['feature_order'] = int(args[i + 1])
                    i += 2
                else:
                    i += 1

            if not updates:
                print("❌ No updates specified")
                return

            # Validate channel if provided
            if 'data_channel' in updates and updates['data_channel'] is not None:
                valid_channels = ['source_current', 'load_voltage', 'impedance', 'power', 'source_current,load_voltage']
                if updates['data_channel'] not in valid_channels:
                    print(f"❌ Invalid channel: {updates['data_channel']}")
                    print(f"   Must be one of: {', '.join(valid_channels)}")
                    return

            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Build UPDATE query
            set_clause = ', '.join([f"{k} = %s" for k in updates.keys()])
            values = list(updates.values())

            cursor.execute(f"""
                UPDATE ml_feature_set_features
                SET {set_clause}
                WHERE feature_set_id = %s AND feature_id = %s
            """, values + [feature_set_id, feature_id])

            if cursor.rowcount == 0:
                print(f"❌ Feature {feature_id} not found in set {feature_set_id}")
                cursor.close()
                conn.close()
                return

            conn.commit()

            # Show updated configuration
            cursor.execute("""
                SELECT
                    fl.feature_name,
                    fsf.feature_order,
                    fsf.data_channel as feature_channel,
                    fsf.n_value_override,
                    efs.data_channel as set_channel,
                    efs.n_value as set_n_value
                FROM ml_feature_set_features fsf
                JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                JOIN ml_experiments_feature_sets efs ON fsf.feature_set_id = efs.feature_set_id
                WHERE fsf.feature_set_id = %s AND fsf.feature_id = %s
                LIMIT 1
            """, (feature_set_id, feature_id))

            row = cursor.fetchone()
            if row:
                effective_channel = row['feature_channel'] or row['set_channel']
                effective_n = row['n_value_override'] or row['set_n_value']

                print(f"✅ Updated {row['feature_name']} in set {feature_set_id} (order {row['feature_order']})")
                print(f"   Channel: {effective_channel} {'(override)' if row['feature_channel'] else '(inherit)'}")
                print(f"   N-value: {effective_n} {'(override)' if row['n_value_override'] else '(inherit)'}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid feature set ID or feature ID")
        except Exception as e:
            print(f"❌ Error updating feature: {e}")

    def cmd_clone_feature_set(self, args):
        """Create a copy of an existing feature set"""
        if not args or len(args) < 2 or '--name' not in args:
            print("Usage: clone-feature-set <source_feature_set_id> --name <new_name>")
            return

        try:
            source_id = int(args[0])

            new_name = None
            idx = args.index('--name')
            if idx + 1 < len(args):
                new_name = args[idx + 1]

            if not new_name:
                print("❌ New name is required")
                return

            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute("""
                SELECT * FROM ml_feature_sets_lut WHERE feature_set_id = %s
            """, (source_id,))
            source = cursor.fetchone()

            if not source:
                print(f"❌ Source feature set {source_id} not found")
                return

            cursor.execute("SELECT 1 FROM ml_feature_sets_lut WHERE feature_set_name = %s", (new_name,))
            if cursor.fetchone():
                print(f"❌ Feature set '{new_name}' already exists")
                return

            cursor.execute("SELECT COALESCE(MAX(feature_set_id), 0) + 1 FROM ml_feature_sets_lut")
            new_id = cursor.fetchone()['next_id']

            cursor.execute("""
                INSERT INTO ml_feature_sets_lut
                (feature_set_id, feature_set_name, category, description, is_active)
                VALUES (%s, %s, %s, %s, %s)
            """, (new_id, new_name, source['category'],
                  f"Clone of {source['feature_set_name']}: {source.get('description', '')}",
                  True))

            cursor.execute("""
                INSERT INTO ml_feature_set_features (feature_set_id, feature_id, feature_order)
                SELECT %s, feature_id, feature_order
                FROM ml_feature_set_features
                WHERE feature_set_id = %s
            """, (new_id, source_id))

            conn.commit()

            print(f"✅ Cloned feature set '{source['feature_set_name']}' (ID: {source_id})")
            print(f"   New set: '{new_name}' (ID: {new_id})")

            cursor.execute("""
                SELECT COUNT(*) as count FROM ml_feature_set_features WHERE feature_set_id = %s
            """, (new_id,))
            count = cursor.fetchone()['count']
            print(f"   Copied {count} features")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid source feature set ID")
        except Exception as e:
            print(f"❌ Error cloning feature set: {e}")

    def cmd_link_feature_set(self, args):
        """Link a feature set to an experiment with configuration"""
        if not args or len(args) < 2:
            print("Usage: link-feature-set <experiment_id> <feature_set_id> [--n-value <n>] [--channel <channel>] [--priority <p>] [--windowing <strategy>]")
            print("\nChannels: load_voltage, source_current, impedance, power")
            print("Windowing: non_overlapping (default), sliding_window")
            print("\nExample: link-feature-set 41 6 --n-value 64 --channel load_voltage --priority 1 --windowing non_overlapping")
            return

        try:
            experiment_id = int(args[0])
            feature_set_id = int(args[1])

            n_value = None
            channel = 'load_voltage'
            priority = None
            windowing_strategy = 'non_overlapping'

            i = 2
            while i < len(args):
                if args[i] == '--n-value' and i + 1 < len(args):
                    n_value = int(args[i + 1])
                    i += 2
                elif args[i] == '--channel' and i + 1 < len(args):
                    channel = args[i + 1]
                    i += 2
                elif args[i] == '--priority' and i + 1 < len(args):
                    priority = int(args[i + 1])
                    i += 2
                elif args[i] == '--windowing' and i + 1 < len(args):
                    windowing_strategy = args[i + 1]
                    if windowing_strategy not in ['non_overlapping', 'sliding_window']:
                        print(f"❌ Invalid windowing strategy: {windowing_strategy}")
                        print("   Must be 'non_overlapping' or 'sliding_window'")
                        return
                    i += 2
                else:
                    i += 1

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COALESCE(MAX(experiment_feature_set_id), 0) + 1 FROM ml_experiments_feature_sets
            """)
            efs_id = cursor.fetchone()[0]

            if priority is None:
                cursor.execute("""
                    SELECT COALESCE(MAX(priority_order), 0) + 1
                    FROM ml_experiments_feature_sets
                    WHERE experiment_id = %s
                """, (experiment_id,))
                priority = cursor.fetchone()[0]

            cursor.execute("""
                INSERT INTO ml_experiments_feature_sets
                (experiment_feature_set_id, experiment_id, feature_set_id, n_value, priority_order, is_active, data_channel, windowing_strategy)
                VALUES (%s, %s, %s, %s, %s, true, %s, %s)
            """, (efs_id, experiment_id, feature_set_id, n_value, priority, channel, windowing_strategy))

            conn.commit()
            print(f"✅ Linked feature set {feature_set_id} to experiment {experiment_id}")
            print(f"   Channel: {channel}")
            if n_value:
                print(f"   N-value: {n_value}")
            print(f"   Priority: {priority}")
            print(f"   Windowing: {windowing_strategy}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid experiment ID or feature set ID")
        except psycopg2.IntegrityError as e:
            print(f"❌ Link already exists or invalid IDs: {e}")
        except Exception as e:
            print(f"❌ Error linking feature set: {e}")

    def cmd_bulk_link_feature_sets(self, args):
        """Link multiple feature sets to an experiment"""
        if not args or len(args) < 2 or '--sets' not in args:
            print("Usage: bulk-link-feature-sets <experiment_id> --sets <id1,id2,id3,...> [--n-values <n1,n2,n3,...>]")
            print("\nExample: bulk-link-feature-sets 41 --sets 1,2,3,4,5 --n-values null,null,null,null,null")
            print("         bulk-link-feature-sets 41 --sets 6,7,8,9 --n-values 64,64,64,64")
            return

        try:
            experiment_id = int(args[0])

            sets = []
            idx = args.index('--sets')
            if idx + 1 < len(args):
                sets = [int(s.strip()) for s in args[idx + 1].split(',')]

            if not sets:
                print("❌ No feature sets specified")
                return

            n_values = [None] * len(sets)
            if '--n-values' in args:
                idx = args.index('--n-values')
                if idx + 1 < len(args):
                    n_val_strs = args[idx + 1].split(',')
                    for i, val in enumerate(n_val_strs[:len(sets)]):
                        if val.strip().lower() != 'null' and val.strip():
                            n_values[i] = int(val.strip())

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("SELECT COALESCE(MAX(experiment_feature_set_id), 0) + 1 FROM ml_experiments_feature_sets")
            next_efs_id = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COALESCE(MAX(priority_order), 0) + 1
                FROM ml_experiments_feature_sets
                WHERE experiment_id = %s
            """, (experiment_id,))
            next_priority = cursor.fetchone()[0]

            success = 0
            failed = 0

            for i, fs_id in enumerate(sets):
                try:
                    cursor.execute("""
                        INSERT INTO ml_experiments_feature_sets
                        (experiment_feature_set_id, experiment_id, feature_set_id, n_value, priority_order, is_active, data_channel)
                        VALUES (%s, %s, %s, %s, %s, true, 'load_voltage')
                    """, (next_efs_id, experiment_id, fs_id, n_values[i], next_priority))

                    next_efs_id += 1
                    next_priority += 1
                    success += 1
                    conn.commit()
                except psycopg2.IntegrityError:
                    failed += 1
                    conn.rollback()

            print(f"✅ Linked {success}/{len(sets)} feature sets to experiment {experiment_id}")
            if failed > 0:
                print(f"   ⚠️  {failed} feature sets were already linked or don't exist")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid experiment ID or feature set IDs")
        except Exception as e:
            print(f"❌ Error linking feature sets: {e}")

    def cmd_update_feature_link(self, args):
        """Update properties of an experiment-feature set link"""
        if not args or len(args) < 2:
            print("Usage: update-feature-link <experiment_id> <feature_set_id> [--n-value <n>] [--priority <p>] [--active <bool>] [--windowing <strategy>]")
            print("\nWindowing: non_overlapping, sliding_window")
            return

        try:
            experiment_id = int(args[0])
            feature_set_id = int(args[1])

            updates = {}
            i = 2
            while i < len(args):
                if args[i] == '--n-value' and i + 1 < len(args):
                    val = args[i + 1]
                    updates['n_value'] = None if val.lower() == 'null' else int(val)
                    i += 2
                elif args[i] == '--priority' and i + 1 < len(args):
                    updates['priority_order'] = int(args[i + 1])
                    i += 2
                elif args[i] == '--active' and i + 1 < len(args):
                    updates['is_active'] = args[i + 1].lower() in ['true', '1', 'yes']
                    i += 2
                elif args[i] == '--windowing' and i + 1 < len(args):
                    windowing_strategy = args[i + 1]
                    if windowing_strategy not in ['non_overlapping', 'sliding_window']:
                        print(f"❌ Invalid windowing strategy: {windowing_strategy}")
                        print("   Must be 'non_overlapping' or 'sliding_window'")
                        return
                    updates['windowing_strategy'] = windowing_strategy
                    i += 2
                else:
                    i += 1

            if not updates:
                print("❌ No updates specified")
                return

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            set_clauses = []
            params = []
            for key, value in updates.items():
                set_clauses.append(f"{key} = %s")
                params.append(value)
            params.extend([experiment_id, feature_set_id])

            query = f"""
                UPDATE ml_experiments_feature_sets
                SET {', '.join(set_clauses)}
                WHERE experiment_id = %s AND feature_set_id = %s
            """
            cursor.execute(query, params)

            if cursor.rowcount == 0:
                print(f"❌ Link between experiment {experiment_id} and feature set {feature_set_id} not found")
            else:
                conn.commit()
                print(f"✅ Updated link between experiment {experiment_id} and feature set {feature_set_id}")
                for key, value in updates.items():
                    print(f"   {key}: {value}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid experiment ID or feature set ID")
        except Exception as e:
            print(f"❌ Error updating feature link: {e}")

    def cmd_show_feature_config(self, args):
        """Show complete feature configuration for an experiment"""
        experiment_id = None
        if args and args[0].isdigit():
            experiment_id = int(args[0])
        else:
            experiment_id = self.current_experiment

        try:
            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            print(f"\n🧬 Feature Configuration for Experiment {experiment_id}:")
            print("=" * 80)

            # First get feature sets
            cursor.execute("""
                SELECT
                    efs.priority_order,
                    fs.feature_set_id,
                    fs.feature_set_name,
                    fs.category,
                    efs.n_value,
                    efs.data_channel,
                    efs.is_active,
                    efs.windowing_strategy,
                    fs.description
                FROM ml_experiments_feature_sets efs
                JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
                WHERE efs.experiment_id = %s
                ORDER BY efs.priority_order
            """, (experiment_id,))

            feature_sets = cursor.fetchall()

            # Get features for each set with overrides
            feature_details = {}
            for fs in feature_sets:
                cursor.execute("""
                    SELECT
                        fl.feature_name,
                        fsf.feature_order,
                        fsf.data_channel as feature_channel,
                        fsf.n_value_override,
                        COALESCE(fsf.data_channel, %s) as effective_channel,
                        COALESCE(fsf.n_value_override, %s) as effective_n_value
                    FROM ml_feature_set_features fsf
                    JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                    WHERE fsf.feature_set_id = %s
                    ORDER BY fsf.feature_order
                """, (fs['data_channel'], fs['n_value'], fs['feature_set_id']))
                feature_details[fs['feature_set_id']] = cursor.fetchall()

            if not feature_sets:
                print("No feature sets configured for this experiment")
                print("\nAdd feature sets using:")
                print("  link-feature-set <exp_id> <fs_id> [options]")
                print("  bulk-link-feature-sets <exp_id> --sets <ids> [options]")
                return

            # Display feature sets with detailed features
            for row in feature_sets:
                status = "✓" if row['is_active'] else "✗"
                print(f"\n[{row['feature_set_id']}] {row['feature_set_name']} (Priority: {row['priority_order']}, Status: {status})")
                print(f"    Default Channel: {row['data_channel'] or 'N/A'}")
                print(f"    Default N-value: {row['n_value'] or 'N/A'}")
                print(f"    Windowing: {row['windowing_strategy']}")

                if row['description'] and row['description'] != f"{row['feature_set_name']} feature set":
                    print(f"    Description: {row['description']}")

                # Show features with overrides
                features = feature_details.get(row['feature_set_id'], [])
                if features:
                    print(f"    Features ({len(features)}):")
                    for feat in features:
                        # Build override indicators
                        overrides = []
                        if feat['feature_channel']:
                            overrides.append(f"ch={feat['feature_channel']}")
                        if feat['n_value_override']:
                            overrides.append(f"n={feat['n_value_override']}")

                        override_str = f" [{', '.join(overrides)}]" if overrides else ""
                        print(f"      {feat['feature_order']}. {feat['feature_name']}({feat['effective_channel']}, n={feat['effective_n_value']}){override_str}")
                else:
                    print(f"    Features: None configured")

            print("\n" + "=" * 80)
            print(f"Total: {len(feature_sets)} feature sets configured")

            active = sum(1 for r in feature_sets if r['is_active'])
            with_n = sum(1 for r in feature_sets if r['n_value'])

            print(f"Status: {active} active, {len(feature_sets) - active} inactive")
            print(f"N-values: {with_n} sets have window sizes configured")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error showing feature configuration: {e}")

    def cmd_update_selection_config(self, args):
        """Update segment selection configuration"""
        if not args or all(not arg.startswith('--') for arg in args):
            print("Usage: update-selection-config [options]")
            print("Options:")
            print("  --max-files <n>    Max files per label (e.g., 50)")
            print("  --seed <n>         Random seed for reproducibility")
            print("  --strategy <s>     Selection strategy (e.g., position_balanced_per_file)")
            print("  --balanced <bool>  Enable balanced segments (true/false)")
            print("\nExample: update-selection-config --max-files 50 --seed 42")
            return
        
        try:
            # Parse arguments
            config_updates = {}
            i = 0
            while i < len(args):
                if args[i] == '--max-files' and i + 1 < len(args):
                    config_updates['max_files_per_label'] = int(args[i + 1])
                    i += 2
                elif args[i] == '--seed' and i + 1 < len(args):
                    config_updates['random_seed'] = int(args[i + 1])
                    i += 2
                elif args[i] == '--strategy' and i + 1 < len(args):
                    config_updates['selection_strategy'] = args[i + 1]
                    i += 2
                elif args[i] == '--balanced' and i + 1 < len(args):
                    config_updates['balanced_segments'] = args[i + 1].lower() == 'true'
                    i += 2
                else:
                    i += 1
            
            if not config_updates:
                print("❌ No valid parameters provided")
                return
            
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Updating segment selection config for experiment {self.current_experiment}...")
            if configurator.update_segment_selection_config(config_updates):
                print(f"✅ Segment selection config updated:")
                for key, value in config_updates.items():
                    print(f"   {key}: {value}")
            else:
                print(f"❌ Failed to update segment selection config")
                
        except ValueError as e:
            print(f"❌ Invalid value: {e}")
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error updating selection config: {e}")
    
    def cmd_select_files(self, args):
        """Select files for experiment training data"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
            
        max_files = 50  # Default
        seed = 42  # Default for experiment 41
        strategy = 'random'  # Default strategy
        min_quality = None
        dry_run = False
        
        # Parse arguments
        i = 0
        while i < len(args):
            if args[i] == '--max-files' and i + 1 < len(args):
                max_files = int(args[i + 1])
                i += 2
            elif args[i] == '--seed' and i + 1 < len(args):
                seed = int(args[i + 1])
                i += 2
            elif args[i] == '--strategy' and i + 1 < len(args):
                strategy = args[i + 1]
                i += 2
            elif args[i] == '--min-quality' and i + 1 < len(args):
                min_quality = float(args[i + 1])
                i += 2
            elif args[i] == '--dry-run':
                dry_run = True
                i += 1
            elif args[i] == '--help':
                print("\nUsage: select-files [options]")
                print("\nOptions:")
                print("  --strategy STRATEGY    Selection strategy: random|balanced|quality_first (default: random)")
                print("  --max-files N         Maximum files per label (default: 50)")
                print("  --seed N              Random seed for reproducibility (default: 42)")
                print("  --min-quality N       Minimum quality score for quality_first strategy")
                print("  --dry-run             Preview selection without saving to database")
                print("\nExample:")
                print("  select-files --strategy random --max-files 50 --seed 42")
                return
            else:
                i += 1
        
        print(f"🔄 Selecting files for experiment {self.current_experiment}...")
        print(f"   Strategy: {strategy}")
        print(f"   Max files per label: {max_files}")
        print(f"   Random seed: {seed}")
        if min_quality:
            print(f"   Minimum quality: {min_quality}")
        
        try:
            from experiment_file_selector import ExperimentFileSelector
            
            selector = ExperimentFileSelector(self.current_experiment, self.db_conn)
            
            if dry_run:
                # Preview available files
                files_by_label = selector.get_available_files()
                print(f"\n📊 Available files by label:")
                total_available = 0
                for label, files in files_by_label.items():
                    print(f"   {label}: {len(files)} files")
                    total_available += len(files)
                print(f"   Total: {total_available} files")
                print("\n💡 Run without --dry-run to save selection")
                return
            
            # Perform selection
            result = selector.select_files(
                strategy=strategy,
                max_files_per_label=max_files,
                seed=seed,
                min_quality=min_quality
            )
            
            if result['success']:
                print(f"\n✅ Successfully selected {result['total_selected']} files")
                
                # Display statistics
                stats = result['statistics']
                if stats and 'label_counts' in stats:
                    print("\n📊 Files selected per label:")
                    for label, count in stats['label_counts'].items():
                        print(f"   {label}: {count} files")
                    print(f"\n   Total unique files: {stats['unique_files']}")
                    print(f"   Total unique labels: {stats['unique_labels']}")
                
                print(f"\n💾 Data saved to: experiment_{self.current_experiment:03d}_file_training_data")
            else:
                print(f"❌ Failed to select files: {result.get('error', 'Unknown error')}")
            
        except ImportError:
            print("❌ ExperimentFileSelector module not found")
        except Exception as e:
            print(f"❌ Error selecting files: {e}")
    
    def cmd_remove_file_labels(self, args):
        """Remove specific file labels from experiment training data"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        if not args:
            print("Usage: remove-file-labels <label1> [label2] [label3] ...")
            print("\nExample:")
            print("  remove-file-labels trash voltage_only arc_short_gap")
            print("\nThis removes all files with the specified labels from the training data.")
            return
        
        # Parse labels from arguments
        labels_to_remove = args
        
        table_name = f"experiment_{self.current_experiment:03d}_file_training_data"
        
        print(f"🗑️  Removing file labels from experiment {self.current_experiment}...")
        print(f"   Labels to remove: {', '.join(labels_to_remove)}")
        
        cursor = self.db_conn.cursor()
        try:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            if not cursor.fetchone()[0]:
                print(f"❌ Table {table_name} does not exist")
                return
            
            # Check which column name is used for labels
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                AND column_name IN ('assigned_label', 'file_label_name')
                LIMIT 1
            """, (table_name,))

            label_column_result = cursor.fetchone()
            if not label_column_result:
                print("❌ No label column found in the table")
                return

            label_column = label_column_result[0]

            # Get counts before deletion using correct column
            cursor.execute(f"""
                SELECT {label_column}, COUNT(*) as count
                FROM {table_name}
                WHERE {label_column} = ANY(%s)
                GROUP BY {label_column}
            """, (labels_to_remove,))
            
            labels_found = {}
            for row in cursor:
                labels_found[row[0]] = row[1]
            
            if not labels_found:
                print("⚠️  No files found with the specified labels")
                return
            
            print("\n📊 Files to be removed:")
            total_to_remove = 0
            for label, count in labels_found.items():
                print(f"   {label}: {count} files")
                total_to_remove += count
            
            # Ask for confirmation
            response = input(f"\n⚠️  Remove {total_to_remove} files? (y/n): ")
            if response.lower() != 'y':
                print("❌ Removal cancelled")
                return
            
            # Delete the files using correct column
            cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {label_column} = ANY(%s)
            """, (labels_to_remove,))
            
            deleted = cursor.rowcount
            self.db_conn.commit()
            
            print(f"\n✅ Successfully removed {deleted} files")
            
            # Show remaining statistics
            cursor.execute(f"""
                SELECT
                    COUNT(DISTINCT file_id) as total_files,
                    COUNT(DISTINCT {label_column}) as unique_labels
                FROM {table_name}
                WHERE experiment_id = %s
            """, (self.current_experiment,))

            stats = cursor.fetchone()
            print(f"\n📊 Remaining in training data:")
            print(f"   Total files: {stats[0]}")
            print(f"   Unique labels: {stats[1]}")

            # Show remaining label distribution
            cursor.execute(f"""
                SELECT {label_column}, COUNT(*) as count
                FROM {table_name}
                WHERE experiment_id = %s
                GROUP BY {label_column}
                ORDER BY count DESC
            """, (self.current_experiment,))
            
            print("\n📊 Remaining label distribution:")
            for row in cursor:
                print(f"   {row[0]}: {row[1]} files")
            
        except Exception as e:
            self.db_conn.rollback()
            print(f"❌ Error removing file labels: {e}")
        finally:
            cursor.close()

    def cmd_remove_files(self, args):
        """Remove specific files from experiment training data"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        if not args:
            print("Usage: remove-files <file_id1> [file_id2] [file_id3] ...")
            print("\nExample:")
            print("  remove-files 1234 5678 9012")
            print("\nThis removes specific files by ID from the training data.")
            return

        # Parse file IDs from arguments
        file_ids = []
        for arg in args:
            try:
                file_ids.append(int(arg))
            except ValueError:
                print(f"⚠️ Skipping invalid file ID: {arg}")

        if not file_ids:
            print("❌ No valid file IDs provided")
            return

        table_name = f"experiment_{self.current_experiment:03d}_file_training_data"

        print(f"🗑️  Removing {len(file_ids)} files from experiment {self.current_experiment}...")

        cursor = self.db_conn.cursor()
        try:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (table_name,))

            if not cursor.fetchone()[0]:
                print(f"❌ Table {table_name} does not exist")
                return

            # Delete the files
            cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE file_id = ANY(%s) AND experiment_id = %s
            """, (file_ids, self.current_experiment))

            deleted = cursor.rowcount
            self.db_conn.commit()

            print(f"✅ Successfully removed {deleted} files")

            # Show remaining statistics
            cursor.execute(f"""
                SELECT COUNT(DISTINCT file_id) FROM {table_name}
                WHERE experiment_id = %s
            """, (self.current_experiment,))

            remaining = cursor.fetchone()[0]
            print(f"📊 Remaining files in training data: {remaining}")

        except Exception as e:
            self.db_conn.rollback()
            print(f"❌ Error removing files: {e}")
        finally:
            cursor.close()

    def cmd_remove_segments(self, args):
        """Remove specific segments from experiment training data"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        if not args:
            print("Usage: remove-segments <segment_id1> [segment_id2] [segment_id3] ...")
            print("\nExample:")
            print("  remove-segments 104075 104076 104077")
            print("\nThis removes specific segments by ID from the training data.")
            return

        # Parse segment IDs from arguments
        segment_ids = []
        for arg in args:
            try:
                segment_ids.append(int(arg))
            except ValueError:
                print(f"⚠️ Skipping invalid segment ID: {arg}")

        if not segment_ids:
            print("❌ No valid segment IDs provided")
            return

        table_name = f"experiment_{self.current_experiment:03d}_segment_training_data"

        print(f"🗑️  Removing {len(segment_ids)} segments from experiment {self.current_experiment}...")

        cursor = self.db_conn.cursor()
        try:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (table_name,))

            if not cursor.fetchone()[0]:
                print(f"❌ Table {table_name} does not exist")
                print("   Run 'select-segments' first to create segment training data")
                return

            # Delete the segments
            cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE segment_id = ANY(%s) AND experiment_id = %s
            """, (segment_ids, self.current_experiment))

            deleted = cursor.rowcount
            self.db_conn.commit()

            print(f"✅ Successfully removed {deleted} segments")

            # Show remaining statistics
            cursor.execute(f"""
                SELECT COUNT(DISTINCT segment_id) FROM {table_name}
                WHERE experiment_id = %s
            """, (self.current_experiment,))

            remaining = cursor.fetchone()[0]
            print(f"📊 Remaining segments in training data: {remaining}")

        except Exception as e:
            self.db_conn.rollback()
            print(f"❌ Error removing segments: {e}")
        finally:
            cursor.close()

    def cmd_generate_training_data(self, args):
        """Deprecated - use select-segments instead"""
        print("⚠️  This command has been replaced by 'select-segments' for clarity.")
        print("\nUse: select-segments [experiment_id] [options]")
        print("\nExample:")
        print("  select-segments 41 --strategy balanced")
        print("  select-segments --help  (for all options)")
        print("\nRedirecting to select-segments...")
        print()

        # Redirect to the proper command
        self.cmd_select_segments(args)
    
    def cmd_generate_segment_pairs(self, args):
        """Generate segment pairs for distance calculations"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        pairing_strategy = 'all_combinations'  # Default
        max_pairs_per_segment = None
        same_label_ratio = 0.5
        seed = 42
        
        # Parse arguments
        i = 0
        while i < len(args):
            if args[i] == '--strategy' and i + 1 < len(args):
                pairing_strategy = args[i + 1]
                i += 2
            elif args[i] == '--max-pairs-per-segment' and i + 1 < len(args):
                max_pairs_per_segment = int(args[i + 1])
                i += 2
            elif args[i] == '--same-label-ratio' and i + 1 < len(args):
                same_label_ratio = float(args[i + 1])
                i += 2
            elif args[i] == '--seed' and i + 1 < len(args):
                seed = int(args[i + 1])
                i += 2
            elif args[i] == '--help':
                print("\nUsage: generate-segment-pairs [options]")
                print("\nOptions:")
                print("  --strategy STRAT            Pairing strategy (default: all_combinations)")
                print("                              Options: all_combinations, balanced, code_type_balanced, random_sample")
                print("  --max-pairs-per-segment N   Maximum pairs per segment")
                print("  --same-label-ratio RATIO    Ratio of same-label pairs for balanced strategy (0.0-1.0)")
                print("  --seed N                    Random seed (default: 42)")
                print("\nStrategies:")
                print("  all_combinations   - Generate all possible pairs (N choose 2)")
                print("  balanced          - Balance same/different label pairs")
                print("  code_type_balanced - Balance pairs by segment code type (L, R, C, etc.)")
                print("  random_sample     - Random sample of possible pairs")
                print("\nExample:")
                print("  generate-segment-pairs --strategy all_combinations")
                print("  generate-segment-pairs --strategy code_type_balanced --max-pairs-per-segment 100")
                print("  generate-segment-pairs --strategy balanced --same-label-ratio 0.3")
                return
            else:
                i += 1
        
        print(f"🔄 Generating segment pairs for experiment {self.current_experiment}...")
        print(f"   Strategy: {pairing_strategy}")
        if max_pairs_per_segment:
            print(f"   Max pairs per segment: {max_pairs_per_segment}")
        print(f"   Same label ratio: {same_label_ratio}")
        print(f"   Random seed: {seed}")
        
        try:
            # Import the v2 segment pair generator module (compatible with v2 selector)
            from experiment_segment_pair_generator_v2 import ExperimentSegmentPairGeneratorV2

            # Create generator instance
            generator = ExperimentSegmentPairGeneratorV2(self.current_experiment, self.db_conn)
            
            # Generate pairs
            result = generator.generate_pairs(
                strategy=pairing_strategy,
                max_pairs_per_segment=max_pairs_per_segment,
                same_label_ratio=same_label_ratio,
                seed=seed
            )
            
            if result['success']:
                print(f"\n✅ Successfully generated segment pairs!")
                print(f"   Total segments: {result['total_segments']}")
                print(f"   Total pairs: {result['total_pairs']}")
                
                # Display statistics
                if 'statistics' in result and result['statistics']:
                    stats = result['statistics']
                    print("\n📊 Pair Statistics:")
                    print(f"   Same segment label pairs: {stats.get('same_segment_label_pairs', 0)}")
                    print(f"   Same file label pairs: {stats.get('same_file_label_pairs', 0)}")
                    print(f"   Same code type pairs: {stats.get('same_code_type_pairs', 0)}")

                    if 'type_distribution' in stats:
                        print("\n   Pair type distribution:")
                        for pair_type, count in sorted(stats['type_distribution'].items()):
                            print(f"     {pair_type}: {count}")

                    if 'top_code_type_pairs' in stats and stats['top_code_type_pairs']:
                        print("\n   Top code type combinations:")
                        for pair in stats['top_code_type_pairs'][:5]:
                            print(f"     {pair}")
            else:
                print(f"\n❌ Failed to generate pairs: {result.get('error', 'Unknown error')}")
                
        except ImportError:
            print("❌ ExperimentSegmentPairGeneratorV2 module not found")
            print("   Make sure experiment_segment_pair_generator_v2.py is in the same directory")
        except Exception as e:
            print(f"❌ Error generating segment pairs: {e}")

    def cmd_init_distance_tables(self, args):
        """Initialize distance result tables for current experiment

        Usage: init-distance-tables [options]

        Options:
            --drop-existing    Drop existing tables before creating (WARNING: destroys data)
            --help             Show this help message

        This command creates all necessary distance result tables for the current experiment
        based on the distance functions configured in ml_distance_functions_lut.

        Examples:
            init-distance-tables
            init-distance-tables --drop-existing
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        drop_existing = False
        if '--help' in args:
            print(self.cmd_init_distance_tables.__doc__)
            return
        if '--drop-existing' in args:
            drop_existing = True

        print(f"\n🔄 Initializing distance tables for experiment {self.current_experiment}...")

        try:
            import psycopg2

            cursor = self.db_conn.cursor()

            # Get distance functions configured for this experiment
            cursor.execute("""
                SELECT df.distance_function_id, df.function_name, df.result_table_prefix, df.display_name
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s AND df.is_active = true
                ORDER BY df.distance_function_id
            """, (self.current_experiment,))

            distance_functions = cursor.fetchall()

            if not distance_functions:
                print(f"❌ No distance functions configured for experiment {self.current_experiment}")
                print("   Check ml_experiments_distance_measurements table")
                return

            print(f"📊 Found {len(distance_functions)} active distance functions")
            print()

            created_count = 0
            skipped_count = 0
            error_count = 0

            for func_id, func_name, table_prefix, display_name in distance_functions:
                table_name = f"experiment_{self.current_experiment:03d}_{table_prefix}"

                try:
                    # Check if table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = %s
                        )
                    """, (table_name,))

                    table_exists = cursor.fetchone()[0]

                    if table_exists:
                        if drop_existing:
                            print(f"🗑️  Dropping existing table: {table_name}")
                            cursor.execute(f"DROP TABLE {table_name} CASCADE")
                            self.db_conn.commit()
                        else:
                            print(f"⏭️  Skipping {table_name} (already exists)")
                            skipped_count += 1
                            continue

                    # Create distance result table
                    create_sql = f"""
                        CREATE TABLE {table_name} (
                            distance_id BIGSERIAL PRIMARY KEY,
                            pair_id INTEGER NOT NULL,
                            amplitude_processing_method_id INTEGER NOT NULL,
                            distance_s DOUBLE PRECISION,
                            distance_i DOUBLE PRECISION,
                            distance_j DOUBLE PRECISION,
                            distance_k DOUBLE PRECISION,
                            created_at TIMESTAMP DEFAULT NOW()
                        )
                    """

                    cursor.execute(create_sql)

                    # Create indexes
                    cursor.execute(f"CREATE INDEX idx_{table_name}_pair ON {table_name}(pair_id)")
                    cursor.execute(f"CREATE INDEX idx_{table_name}_amp ON {table_name}(amplitude_processing_method_id)")
                    cursor.execute(f"CREATE INDEX idx_{table_name}_pair_amp ON {table_name}(pair_id, amplitude_processing_method_id)")

                    self.db_conn.commit()

                    print(f"✅ Created table: {table_name} ({display_name})")
                    created_count += 1

                except Exception as e:
                    print(f"❌ Error creating {table_name}: {e}")
                    self.db_conn.rollback()
                    error_count += 1

            print()
            print(f"📊 Summary:")
            print(f"   Created: {created_count}")
            print(f"   Skipped: {skipped_count}")
            print(f"   Errors: {error_count}")
            print()

            if created_count > 0:
                print(f"✅ Distance tables initialized for experiment {self.current_experiment}")
            elif skipped_count > 0:
                print(f"ℹ️  All tables already exist. Use --drop-existing to recreate them.")

        except Exception as e:
            print(f"❌ Error initializing distance tables: {e}")
            import traceback
            traceback.print_exc()

    def cmd_show_distance_metrics(self, args):
        """Show distance metrics configured for current experiment

        Usage: show-distance-metrics

        Displays all distance metrics configured in ml_experiments_distance_measurements
        for the current experiment.
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        try:
            cursor = self.db_conn.cursor()

            # Get distance functions configured for this experiment
            cursor.execute("""
                SELECT df.distance_function_id, df.function_name, df.display_name, df.result_table_prefix
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s
                ORDER BY df.function_name
            """, (self.current_experiment,))

            metrics = cursor.fetchall()

            if not metrics:
                print(f"\n❌ No distance metrics configured for experiment {self.current_experiment}")
                return

            print(f"\n📊 Distance metrics configured for experiment {self.current_experiment}:")
            print(f"\nID  | Function Name        | Display Name                      | Table Prefix")
            print("-" * 90)
            for metric in metrics:
                print(f"{metric[0]:<4}| {metric[1]:<20} | {metric[2]:<33} | {metric[3]}")

            print(f"\nTotal: {len(metrics)} metrics")

        except Exception as e:
            print(f"❌ Error showing distance metrics: {e}")

    def cmd_add_distance_metric(self, args):
        """Add distance metric to current experiment

        Usage: add-distance-metric --metric <metric_name>

        Options:
            --metric <name>    Metric name (e.g., L1, L2, cosine, pearson, wasserstein)

        Examples:
            add-distance-metric --metric wasserstein
            add-distance-metric --metric euclidean
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        metric_name = None
        if '--metric' in args:
            idx = args.index('--metric')
            if idx + 1 < len(args):
                metric_name = args[idx + 1]

        if not metric_name:
            print("❌ Error: --metric is required")
            print("\nUsage: add-distance-metric --metric <metric_name>")
            print("\nExample: add-distance-metric --metric wasserstein")
            return

        try:
            cursor = self.db_conn.cursor()

            # Find distance function by name
            cursor.execute("""
                SELECT distance_function_id, function_name, display_name
                FROM ml_distance_functions_lut
                WHERE function_name = %s AND is_active = true
            """, (metric_name,))

            function = cursor.fetchone()

            if not function:
                print(f"❌ Distance function '{metric_name}' not found or not active")
                print("\nAvailable metrics:")
                cursor.execute("SELECT function_name FROM ml_distance_functions_lut WHERE is_active = true ORDER BY function_name")
                available = cursor.fetchall()
                for avail in available:
                    print(f"  - {avail[0]}")
                return

            func_id, func_name, display_name = function

            # Check if already configured
            cursor.execute("""
                SELECT COUNT(*) FROM ml_experiments_distance_measurements
                WHERE experiment_id = %s AND distance_function_id = %s
            """, (self.current_experiment, func_id))

            if cursor.fetchone()[0] > 0:
                print(f"⚠️  {func_name} ({display_name}) is already configured for experiment {self.current_experiment}")
                return

            # Add to experiment
            cursor.execute("""
                INSERT INTO ml_experiments_distance_measurements (experiment_id, distance_function_id)
                VALUES (%s, %s)
            """, (self.current_experiment, func_id))

            self.db_conn.commit()

            print(f"✅ Added {func_name} ({display_name}) to experiment {self.current_experiment}")

        except Exception as e:
            print(f"❌ Error adding distance metric: {e}")
            self.db_conn.rollback()

    def cmd_remove_distance_metric(self, args):
        """Remove distance metric from current experiment

        Usage: remove-distance-metric [options]

        Options:
            --metric <name>       Remove specific metric (e.g., wasserstein)
            --all-except <list>   Remove all except specified metrics (comma-separated)

        Examples:
            remove-distance-metric --metric wasserstein
            remove-distance-metric --all-except L1,L2,cosine,pearson
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        metric_name = None
        keep_only = None

        if '--metric' in args:
            idx = args.index('--metric')
            if idx + 1 < len(args):
                metric_name = args[idx + 1]

        if '--all-except' in args:
            idx = args.index('--all-except')
            if idx + 1 < len(args):
                keep_only = [m.strip() for m in args[idx + 1].split(',')]

        if not metric_name and not keep_only:
            print("❌ Error: --metric or --all-except is required")
            print("\nUsage:")
            print("  remove-distance-metric --metric <metric_name>")
            print("  remove-distance-metric --all-except L1,L2,cosine,pearson")
            return

        try:
            cursor = self.db_conn.cursor()

            if keep_only:
                # Remove all except specified metrics
                print(f"\n🔄 Removing all distance metrics except: {', '.join(keep_only)}")

                # Get IDs of metrics to keep
                placeholders = ','.join(['%s'] * len(keep_only))
                cursor.execute(f"""
                    SELECT distance_function_id, function_name
                    FROM ml_distance_functions_lut
                    WHERE function_name IN ({placeholders})
                """, keep_only)

                keep_ids = cursor.fetchall()

                if not keep_ids:
                    print(f"❌ None of the specified metrics found: {', '.join(keep_only)}")
                    return

                print(f"ℹ️  Keeping {len(keep_ids)} metrics:")
                for func_id, func_name in keep_ids:
                    print(f"   - {func_name}")

                # Delete all except these
                keep_id_list = [func_id for func_id, _ in keep_ids]
                placeholders = ','.join(['%s'] * len(keep_id_list))
                cursor.execute(f"""
                    DELETE FROM ml_experiments_distance_measurements
                    WHERE experiment_id = %s
                    AND distance_function_id NOT IN ({placeholders})
                    RETURNING distance_function_id
                """, [self.current_experiment] + keep_id_list)

                deleted = cursor.fetchall()
                self.db_conn.commit()

                print(f"\n✅ Removed {len(deleted)} distance metrics from experiment {self.current_experiment}")

            else:
                # Remove specific metric
                cursor.execute("""
                    SELECT distance_function_id, function_name, display_name
                    FROM ml_distance_functions_lut
                    WHERE function_name = %s
                """, (metric_name,))

                function = cursor.fetchone()

                if not function:
                    print(f"❌ Distance function '{metric_name}' not found")
                    return

                func_id, func_name, display_name = function

                cursor.execute("""
                    DELETE FROM ml_experiments_distance_measurements
                    WHERE experiment_id = %s AND distance_function_id = %s
                    RETURNING experiment_distance_id
                """, (self.current_experiment, func_id))

                deleted = cursor.fetchone()

                if not deleted:
                    print(f"⚠️  {func_name} ({display_name}) was not configured for experiment {self.current_experiment}")
                    return

                self.db_conn.commit()

                print(f"✅ Removed {func_name} ({display_name}) from experiment {self.current_experiment}")

        except Exception as e:
            print(f"❌ Error removing distance metric: {e}")
            self.db_conn.rollback()
            import traceback
            traceback.print_exc()

    def cmd_clean_distance_tables(self, args):
        """Clean unconfigured empty distance tables for current experiment

        Usage: clean-distance-tables [options]

        Options:
            --dry-run    Show what would be deleted without actually deleting
            --force      Skip confirmation prompt

        This command removes distance result tables that are:
        1. NOT configured in ml_experiments_distance_measurements for current experiment
        2. Have 0 rows (empty tables)

        Tables with data are NEVER deleted (safety check).

        Examples:
            clean-distance-tables                # Interactive mode
            clean-distance-tables --dry-run      # Show what would be deleted
            clean-distance-tables --force        # Skip confirmation
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        dry_run = '--dry-run' in args
        force = '--force' in args

        print(f"\n🔄 Scanning distance tables for experiment {self.current_experiment}...")

        try:
            cursor = self.db_conn.cursor()

            # Get configured distance metrics for this experiment
            cursor.execute("""
                SELECT df.result_table_prefix
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s
            """, (self.current_experiment,))

            configured_prefixes = [row[0] for row in cursor.fetchall()]
            configured_tables = [f"experiment_{self.current_experiment:03d}_{prefix}" for prefix in configured_prefixes]
            # PostgreSQL lowercases table names, so normalize for comparison
            configured_tables_lower = [t.lower() for t in configured_tables]

            print(f"📊 Found {len(configured_tables)} configured distance tables")

            # Get all distance tables for this experiment
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE %s
                ORDER BY table_name
            """, (f"experiment_{self.current_experiment:03d}_distance_%",))

            all_tables = [row[0] for row in cursor.fetchall()]

            print(f"📁 Found {len(all_tables)} total distance tables in database")

            # Find unconfigured tables (case-insensitive comparison)
            unconfigured_tables = [t for t in all_tables if t.lower() not in configured_tables_lower]

            if not unconfigured_tables:
                print("\n✅ No unconfigured distance tables found. All tables match configuration.")
                return

            print(f"\n⚠️  Found {len(unconfigured_tables)} unconfigured tables:")

            # Check row counts and categorize
            empty_tables = []
            non_empty_tables = []

            for table_name in unconfigured_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                if row_count == 0:
                    empty_tables.append(table_name)
                    print(f"   🗑️  {table_name}: 0 rows (can be deleted)")
                else:
                    non_empty_tables.append((table_name, row_count))
                    print(f"   ⚠️  {table_name}: {row_count:,} rows (WILL NOT DELETE - has data)")

            if not empty_tables:
                print("\n✅ No empty unconfigured tables to clean.")
                if non_empty_tables:
                    print(f"\nℹ️  {len(non_empty_tables)} tables have data and were not deleted.")
                return

            print(f"\n📋 Summary:")
            print(f"   Empty tables to delete: {len(empty_tables)}")
            print(f"   Tables with data (protected): {len(non_empty_tables)}")

            if dry_run:
                print("\n🔍 DRY RUN - No tables will be deleted")
                print("\nWould delete:")
                for table in empty_tables:
                    print(f"   - {table}")
                return

            # Confirmation prompt
            if not force:
                print(f"\n⚠️  About to delete {len(empty_tables)} empty unconfigured tables")
                response = input("Continue? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("❌ Cancelled")
                    return

            # Delete empty unconfigured tables
            deleted_count = 0
            for table_name in empty_tables:
                try:
                    cursor.execute(f"DROP TABLE {table_name} CASCADE")
                    self.db_conn.commit()
                    print(f"✅ Deleted: {table_name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Error deleting {table_name}: {e}")
                    self.db_conn.rollback()

            print(f"\n✅ Cleaned {deleted_count} empty unconfigured distance tables")

            if non_empty_tables:
                print(f"\nℹ️  {len(non_empty_tables)} tables with data were preserved")

        except Exception as e:
            print(f"❌ Error cleaning distance tables: {e}")
            import traceback
            traceback.print_exc()

    def cmd_show_distance_functions(self, args):
        """Show all distance functions in ml_distance_functions_lut

        Usage: show-distance-functions [--active-only]

        Options:
            --active-only    Only show functions where is_active = true

        Displays all distance functions available in the system.
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        active_only = '--active-only' in args

        try:
            cursor = self.db_conn.cursor()

            # Check if pairwise_metric_name column exists
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'ml_distance_functions_lut'
                AND column_name = 'pairwise_metric_name'
            """)

            has_pairwise_column = cursor.fetchone() is not None

            if has_pairwise_column:
                # Query with pairwise_metric_name
                query = """
                    SELECT distance_function_id, function_name, display_name,
                           library_name, function_import, pairwise_metric_name,
                           result_table_prefix, is_active
                    FROM ml_distance_functions_lut
                """
            else:
                # Query without pairwise_metric_name (backward compatible)
                query = """
                    SELECT distance_function_id, function_name, display_name,
                           library_name, function_import,
                           result_table_prefix, is_active
                    FROM ml_distance_functions_lut
                """

            if active_only:
                query += " WHERE is_active = true"

            query += " ORDER BY distance_function_id"

            cursor.execute(query)
            functions = cursor.fetchall()

            if not functions:
                print("\n❌ No distance functions found in ml_distance_functions_lut")
                return

            # Show warning if pairwise_metric_name column doesn't exist
            if not has_pairwise_column:
                print("\n⚠️  WARNING: pairwise_metric_name column not found!")
                print("   Run this SQL script to add it:")
                print("   psql -h localhost -p 5432 -d arc_detection -f /Users/kjensen/Documents/GitHub/mldp/mldp_cli/sql/update_distance_functions_lut.sql")
                print()

            print(f"\n📊 Distance Functions in ml_distance_functions_lut:")
            if active_only:
                print("(Showing only active functions)")
            print()

            if has_pairwise_column:
                print(f"{'ID':<4} | {'Name':<20} | {'Display Name':<30} | {'Pairwise Metric':<15} | {'Active':<6}")
                print("-" * 95)

                for func in functions:
                    func_id, name, display, library, func_import, pairwise, prefix, active = func
                    pairwise_str = pairwise or 'N/A'
                    active_str = '✅' if active else '❌'
                    print(f"{func_id:<4} | {name:<20} | {display:<30} | {pairwise_str:<15} | {active_str:<6}")
            else:
                # Without pairwise_metric_name column
                print(f"{'ID':<4} | {'Name':<20} | {'Display Name':<30} | {'Library':<30} | {'Active':<6}")
                print("-" * 100)

                for func in functions:
                    func_id, name, display, library, func_import, prefix, active = func
                    library_str = library or 'N/A'
                    active_str = '✅' if active else '❌'
                    print(f"{func_id:<4} | {name:<20} | {display:<30} | {library_str:<30} | {active_str:<6}")

            print(f"\nTotal: {len(functions)} functions")

            # Show additional details if not many
            if len(functions) <= 5:
                print("\nDetailed Information:")
                for func in functions:
                    if has_pairwise_column:
                        func_id, name, display, library, func_import, pairwise, prefix, active = func
                    else:
                        func_id, name, display, library, func_import, prefix, active = func
                        pairwise = None

                    print(f"\n{name} (ID: {func_id}):")
                    print(f"  Display: {display}")
                    print(f"  Library: {library or 'N/A'}")
                    print(f"  Function: {func_import or 'N/A'}")
                    if has_pairwise_column:
                        print(f"  Pairwise Metric: {pairwise or 'N/A'}")
                    print(f"  Table Prefix: {prefix}")
                    print(f"  Active: {'Yes' if active else 'No'}")

        except Exception as e:
            print(f"❌ Error showing distance functions: {e}")
            import traceback
            traceback.print_exc()

    def cmd_update_distance_function(self, args):
        """Update distance function in ml_distance_functions_lut

        Usage: update-distance-function <function_name> [options]

        Options:
            --pairwise-metric <name>    Set pairwise metric name for sklearn.metrics.pairwise_distances
            --library <name>            Set library name
            --function-import <name>    Set function import name
            --description <text>        Set description
            --active <true|false>       Set is_active flag

        Examples:
            update-distance-function pearson --pairwise-metric correlation
            update-distance-function manhattan --pairwise-metric manhattan --library sklearn.metrics.pairwise --function-import pairwise_distances
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        if not args or args[0].startswith('--'):
            print("❌ Error: function_name is required")
            print("\nUsage: update-distance-function <function_name> [options]")
            return

        function_name = args[0]
        updates = {}

        i = 1
        while i < len(args):
            if args[i] == '--pairwise-metric' and i + 1 < len(args):
                updates['pairwise_metric_name'] = args[i + 1]
                i += 2
            elif args[i] == '--library' and i + 1 < len(args):
                updates['library_name'] = args[i + 1]
                i += 2
            elif args[i] == '--function-import' and i + 1 < len(args):
                updates['function_import'] = args[i + 1]
                i += 2
            elif args[i] == '--description' and i + 1 < len(args):
                updates['description'] = args[i + 1]
                i += 2
            elif args[i] == '--active' and i + 1 < len(args):
                updates['is_active'] = args[i + 1].lower() in ['true', '1', 'yes']
                i += 2
            else:
                i += 1

        if not updates:
            print("❌ Error: At least one update option is required")
            print("\nAvailable options: --pairwise-metric, --library, --function-import, --description, --active")
            return

        try:
            cursor = self.db_conn.cursor()

            # Check if function exists
            cursor.execute("""
                SELECT distance_function_id, function_name, display_name
                FROM ml_distance_functions_lut
                WHERE function_name = %s
            """, (function_name,))

            function = cursor.fetchone()

            if not function:
                print(f"❌ Distance function '{function_name}' not found in ml_distance_functions_lut")
                return

            func_id, func_name, display_name = function

            # Build UPDATE query
            set_clauses = []
            values = []

            for key, value in updates.items():
                set_clauses.append(f"{key} = %s")
                values.append(value)

            values.append(func_id)

            update_query = f"""
                UPDATE ml_distance_functions_lut
                SET {', '.join(set_clauses)}, updated_at = NOW()
                WHERE distance_function_id = %s
            """

            cursor.execute(update_query, values)
            self.db_conn.commit()

            print(f"✅ Updated {func_name} ({display_name})")
            print("\nUpdated fields:")
            for key, value in updates.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"❌ Error updating distance function: {e}")
            self.db_conn.rollback()
            import traceback
            traceback.print_exc()

    def cmd_mpcctl_distance_function(self, args):
        """Control MPCCTL distance calculation with background execution."""

        if '--help' in args:
            print("\nUsage: mpcctl-distance-function [options]")
            print("\nBackground distance calculation with pause/resume/stop control.")
            print("\nCommands:")
            print("  --start                  Start distance calculation in background")
            print("  --pause                  Pause running calculation")
            print("  --continue               Resume paused calculation")
            print("  --stop                   Stop calculation")
            print("  --status                 Show progress")
            print("\nOptions for --start:")
            print("  --workers N              Number of worker processes (default: 16)")
            print("  --log                    Create log file (yyyymmdd_hhmmss_mpcctl_distance_calculation.log)")
            print("  --verbose                Show verbose output in CLI")
            print("\nExamples:")
            print("  mpcctl-distance-function --start --workers 20")
            print("  mpcctl-distance-function --start --workers 20 --log --verbose")
            print("  mpcctl-distance-function --status")
            print("  mpcctl-distance-function --pause")
            print("  mpcctl-distance-function --continue")
            print("  mpcctl-distance-function --stop")
            return

        if '--start' in args:
            # Start distance calculation in background
            if not self.current_experiment:
                print("❌ No experiment selected. Use 'set experiment <id>' first.")
                return

            # Parse options
            workers = 16
            log_enabled = '--log' in args
            verbose = '--verbose' in args

            for i, arg in enumerate(args):
                if arg == '--workers' and i + 1 < len(args):
                    try:
                        workers = int(args[i + 1])
                    except ValueError:
                        print(f"❌ Invalid workers value: {args[i + 1]}")
                        return

            # Import required modules
            import multiprocessing as mp
            from pathlib import Path
            from datetime import datetime
            import sys
            sys.path.insert(0, '/Users/kjensen/Documents/GitHub/mldp/mldp_distance')
            from mpcctl_cli_distance_calculator import manager_process

            # Prepare configuration
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            feature_base_path = Path('/Volumes/ArcData/V3_database/experiment041/feature_files')
            mpcctl_base_dir = Path('/Volumes/ArcData/V3_database/experiment041')

            # Create log file if requested
            log_file = None
            if log_enabled:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = Path(f"{timestamp}_mpcctl_distance_calculation.log")

            # Spawn manager process in background (daemon)
            manager = mp.Process(
                target=manager_process,
                args=(self.current_experiment, workers, feature_base_path,
                      db_config, log_file, verbose, mpcctl_base_dir),
                daemon=True
            )
            manager.start()

            print(f"🚀 Distance calculation started in background")
            print(f"   Experiment: {self.current_experiment}")
            print(f"   Workers: {workers}")
            if log_file:
                print(f"   Log file: {log_file}")
            print(f"\n📊 Monitor progress:")
            print(f"   mpcctl-distance-function --status")
            print(f"\n⏸️  Control:")
            print(f"   mpcctl-distance-function --pause")
            print(f"   mpcctl-distance-function --continue")
            print(f"   mpcctl-distance-function --stop")

        elif '--status' in args:
            # Show progress
            from pathlib import Path
            import json

            state_file = Path('/Volumes/ArcData/V3_database/experiment041/.mpcctl_state.json')
            if not state_file.exists():
                print("❌ No active distance calculation found")
                print("   Start with: mpcctl-distance-function --start --workers 20")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                progress = state.get('progress', {})
                status = state.get('status', 'unknown')

                # Progress bar
                bar_width = 50
                percent = progress.get('percent_complete', 0)
                filled = int(bar_width * percent / 100)
                bar = '█' * filled + '░' * (bar_width - filled)

                # Format time
                eta_seconds = progress.get('estimated_time_remaining_seconds', 0)
                eta_minutes = eta_seconds // 60
                eta_seconds_remainder = eta_seconds % 60

                print(f"\n📊 Distance Calculation Progress")
                print(f"   Status: {status}")
                print(f"   [{bar}] {percent:.1f}%")
                print(f"   Completed: {progress.get('completed_pairs', 0):,} / {progress.get('total_pairs', 0):,} pairs")
                print(f"   Rate: {progress.get('pairs_per_second', 0):.0f} pairs/sec")
                print(f"   ETA: {eta_minutes} min {eta_seconds_remainder} sec")
                print(f"   Workers: {state.get('workers_count', 0)}")

                if state.get('log_file'):
                    print(f"   Log: {state['log_file']}")

            except Exception as e:
                print(f"❌ Error reading status: {e}")

        elif '--pause' in args:
            # Send pause command
            from pathlib import Path
            import json
            from datetime import datetime

            state_file = Path('/Volumes/ArcData/V3_database/experiment041/.mpcctl_state.json')
            if not state_file.exists():
                print("❌ No active distance calculation found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                state['control'] = {
                    'command': 'pause',
                    'command_time': datetime.now().strftime('%Y%m%d_%H%M%S')
                }

                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)

                print("⏸️  Pause signal sent")
                print("   Workers will pause after current pair")
                print("   Use 'mpcctl-distance-function --status' to verify")

            except Exception as e:
                print(f"❌ Error sending pause signal: {e}")

        elif '--continue' in args:
            # Send resume command
            from pathlib import Path
            import json
            from datetime import datetime

            state_file = Path('/Volumes/ArcData/V3_database/experiment041/.mpcctl_state.json')
            if not state_file.exists():
                print("❌ No active distance calculation found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                state['control'] = {
                    'command': 'resume',
                    'command_time': datetime.now().strftime('%Y%m%d_%H%M%S')
                }

                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)

                print("▶️  Resume signal sent")
                print("   Workers will continue processing")
                print("   Use 'mpcctl-distance-function --status' to verify")

            except Exception as e:
                print(f"❌ Error sending resume signal: {e}")

        elif '--stop' in args:
            # Send stop command
            from pathlib import Path
            import json
            from datetime import datetime

            state_file = Path('/Volumes/ArcData/V3_database/experiment041/.mpcctl_state.json')
            if not state_file.exists():
                print("❌ No active distance calculation found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                state['control'] = {
                    'command': 'stop',
                    'command_time': datetime.now().strftime('%Y%m%d_%H%M%S')
                }

                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)

                print("⏹️  Stop signal sent")
                print("   Workers will exit gracefully after current pair")
                print("   Use 'mpcctl-distance-function --status' to verify")

            except Exception as e:
                print(f"❌ Error sending stop signal: {e}")

        else:
            print("❌ Unknown option. Use --help for usage information.")

    def cmd_generate_feature_fileset(self, args):
        """Generate feature files from segment data"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Show help if requested or if no experiment set
        if '--help' in args or not self.current_experiment:
            print("\nUsage: generate-feature-fileset [options]")
            print("\nThis command extracts features from ALL segment files (all decimation levels).")
            print("\nBy default, processes ALL segment files and ALL active feature sets.")
            print("\nOptions:")
            print("  --feature-sets <list>    Comma-separated feature set IDs (default: all active)")
            print("  --max-segments N         Maximum segment FILES to process (default: all)")
            print("  --force                  Force re-extraction of existing features")
            print("\nExamples:")
            print("  generate-feature-fileset")
            print("  generate-feature-fileset --feature-sets 1,2,3")
            print("  generate-feature-fileset --max-segments 1000")
            print("  generate-feature-fileset --force")
            print("\n📝 Pipeline Order:")
            print("  1. select-files          - Select files for training (DB)")
            print("  2. select-segments       - Select segments for training (DB)")
            print("  3. generate-training-data - Create training data tables (DB)")
            print("  4. generate-segment-fileset - Create physical segment files (Disk)")
            print("  5. generate-feature-fileset - Extract features from segments (Disk)")
            print("\n📁 Input Structure:")
            print("  experiment{NNN}/segment_files/S{size}/T{type}/D{decimation}/*.npy")
            print("\n📁 Output Structure:")
            print("  experiment{NNN}/feature_files/S{size}/T{type}/D{decimation}/*_FS{id}[_N_{n}].npy")
            print("\n📊 Processing Details:")
            print("  - Processes ALL decimation levels (S000512 to S524288)")
            print("  - Processes ALL ADC types (TRAW, TADC6, TADC8, TADC10, TADC12, TADC14)")
            print("  - Mirrors segment_files/ directory structure exactly")
            print("  - Tracks original_segment_length AND stored_segment_length")
            print("  - Enables decimation/information-loss analysis")
            print("\n⚙️  Database Records:")
            print("  Each extraction creates a record in experiment_{NNN}_feature_fileset with:")
            print("    - segment_id, file_id, feature_set_id, n_value")
            print("    - original_segment_length (from data_segments)")
            print("    - stored_segment_length (from filesystem path)")
            print("    - adc_type, adc_division")
            print("    - feature_file_path, num_chunks, extraction_time")
            if not self.current_experiment:
                print("\n⚠️  No experiment selected. Use 'set experiment <id>' first.")
            return

        feature_set_ids = None  # Default: all configured feature sets
        max_segments = None
        force_reextract = False

        # Parse arguments
        i = 0
        while i < len(args):
            if args[i] == '--feature-sets' and i + 1 < len(args):
                feature_set_ids = [int(x) for x in args[i + 1].split(',')]
                i += 2
            elif args[i] == '--max-segments' and i + 1 < len(args):
                max_segments = int(args[i + 1])
                i += 2
            elif args[i] == '--force':
                force_reextract = True
                i += 1
            else:
                i += 1

        print(f"🔄 Generating feature files for experiment {self.current_experiment}...")
        if feature_set_ids:
            print(f"   Feature sets: {feature_set_ids}")
        else:
            print(f"   Feature sets: All configured sets")
        if max_segments:
            print(f"   Max segment files: {max_segments:,}")
        else:
            print(f"   Max segment files: All")
        if force_reextract:
            print(f"   Mode: Force re-extraction")

        try:
            # Import the feature extractor module
            from experiment_feature_extractor import ExperimentFeatureExtractor

            # Create extractor instance
            extractor = ExperimentFeatureExtractor(self.current_experiment, self.db_conn)

            # Extract features
            result = extractor.extract_features(
                feature_set_ids=feature_set_ids,
                max_segments=max_segments,
                force_reextract=force_reextract
            )
            
            if result['success']:
                print(f"\n✅ Successfully extracted features!")
                print(f"   Total segments: {result['total_segments']}")
                print(f"   Total feature sets: {result['total_feature_sets']}")
                print(f"   Total extracted: {result['total_extracted']}")
                
                if result['failed_count'] > 0:
                    print(f"\n⚠️  Failed extractions: {result['failed_count']}")
                    if result.get('failed_extractions'):
                        print("   First few failures:")
                        for fail in result['failed_extractions'][:5]:
                            print(f"     Segment {fail['segment_id']}, FS {fail['feature_set_id']}: {fail['error']}")
                
                if result.get('average_extraction_time'):
                    print(f"\n⏱️  Performance:")
                    print(f"   Average time per extraction: {result['average_extraction_time']:.2f}s")
                    print(f"   Total extraction time: {result['total_extraction_time']:.2f}s")
            else:
                print(f"\n❌ Failed to extract features: {result.get('error', 'Unknown error')}")
                
        except ImportError:
            print("❌ ExperimentFeatureExtractor module not found")
            print("   Make sure experiment_feature_extractor.py is in the same directory")
        except Exception as e:
            print(f"❌ Error generating feature fileset: {e}")
    
    def _create_segment_selector_module(self):
        """Create the segment selector module if it doesn't exist"""
        print("\n📝 Creating ExperimentSegmentSelector module...")
        # The module has been created separately
        print("   Module should be available at: experiment_segment_selector.py")
    
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

    def cmd_generate_segment_fileset(self, args):
        """Generate physical segment files from raw data on disk

        This command creates the actual segment files on disk by processing
        raw data files. It performs decimation and data type conversions.

        Note: This is different from generate-training-data which only
        creates database tables for tracking which segments to use.
        """
        # Show help if requested
        if '--help' in args:
            print("\nUsage: generate-segment-fileset [options]")
            print("\nThis command generates physical segment files from raw data.")
            print("\nBy default, uses the experiment's configured data types and decimations.")
            print("\nOptions:")
            print("  --data-types <list>      Override data types (RAW,ADC14,ADC12,ADC10,ADC8,ADC6)")
            print("  --decimations <list>     Override decimation factors (0=none, comma-separated)")
            print("  --max-segments N         Maximum segments to process")
            print("\nNote: If no --data-types or --decimations are specified, uses experiment config.")
            print("\nExamples:")
            print("  generate-segment-fileset")
            print("  generate-segment-fileset --data-types RAW")
            print("  generate-segment-fileset --data-types RAW,ADC14 --decimations 0,7,15")
            print("\n📝 Pipeline Order:")
            print("  1. select-files          - Select files for training (DB)")
            print("  2. select-segments       - Select segments for training (DB)")
            print("  3. generate-training-data - Create training data tables (DB)")
            print("  4. generate-segment-fileset - Create physical segment files (Disk)")
            print("  5. generate-feature-fileset - Extract features from segments (Disk)")
            print("\n📁 Output Structure:")
            print("  experiment{NNN}/segment_files/S{size}/T{type}/D{decimation}/*.npy")
            return

        # Determine experiment_id: use current experiment or first arg if it's a number
        experiment_id = None
        arg_offset = 0

        if args and args[0].isdigit():
            # Legacy support: first arg is experiment_id
            experiment_id = int(args[0])
            arg_offset = 1
        elif self.current_experiment:
            # Use current experiment set via 'set experiment'
            experiment_id = self.current_experiment
        else:
            print("❌ No experiment specified. Use 'set experiment <id>' first or provide experiment_id as argument.")
            return

        # Special handling for experiment 18 with legacy code
        if experiment_id == 18 and '--files' in args:
            print(f"🔄 Using legacy generator for experiment 18...")
            self.cmd_segment_generate(args)
            return

        # Use new generator for all experiments
        print(f"🔄 Generating segment fileset for experiment {experiment_id}...")

        # Parse arguments - only use if explicitly provided
        data_types = None  # Will use experiment config if not specified
        decimations = None  # Will use experiment config if not specified
        max_segments = None
        use_experiment_config = True

        i = arg_offset
        while i < len(args):
            if args[i] == '--data-types' and i + 1 < len(args):
                data_types = [dt.upper() for dt in args[i + 1].split(',')]
                use_experiment_config = False
                i += 2
            elif args[i] == '--decimations' and i + 1 < len(args):
                decimations = [int(d) for d in args[i + 1].split(',')]
                use_experiment_config = False
                i += 2
            elif args[i] == '--max-segments' and i + 1 < len(args):
                max_segments = int(args[i + 1])
                i += 2
            else:
                i += 1

        if use_experiment_config:
            print(f"📋 Using experiment {experiment_id} configuration (data types & decimations)")
        else:
            if data_types:
                print(f"📋 Using custom data types: {data_types}")
            if decimations is not None:
                print(f"📋 Using custom decimations: {decimations}")

        try:
            from experiment_segment_fileset_generator_v2 import ExperimentSegmentFilesetGeneratorV2

            # Database configuration
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            # Create generator
            generator = ExperimentSegmentFilesetGeneratorV2(experiment_id, db_config)

            # Generate fileset - pass None to use experiment config
            result = generator.generate_segment_fileset(
                data_types=data_types,  # None = use experiment config
                decimations=decimations,  # None = use experiment config
                max_segments=max_segments,
                parallel_workers=1
            )

            if result.get('files_created', 0) > 0:
                print(f"\n✅ Successfully generated segment files!")
                print(f"   Files created: {result['files_created']}")
                print(f"   Files skipped: {result['files_skipped']}")
                print(f"   Segments processed: {result['segments_processed']}")
                print(f"   Output path: {generator.segment_path}")
            else:
                print(f"\n❌ No segment files generated")
                print(f"   Files failed: {result.get('files_failed', 0)}")

        except ImportError:
            print("❌ ExperimentSegmentFilesetGeneratorV2 module not found")
            print("   Make sure experiment_segment_fileset_generator_v2.py is in the same directory")
        except Exception as e:
            print(f"❌ Error generating segment fileset: {e}")

    def cmd_show_segment_status(self, args):
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
    
    def cmd_validate_segments(self, args):
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

    def cmd_feature_plot(self, args):
        """Plot feature files with statistical visualization

        Usage: feature-plot [options]

        Options:
            --file <path>           Path to feature file (.npy)
            --output-folder <path>  Output directory for plots
            --save <filename>       Save to specific filename (overrides --output-folder)

        Examples:
            feature-plot --file /path/to/feature.npy
            feature-plot --file /path/to/feature.npy --output-folder ~/plots/
            feature-plot --file /path/to/feature.npy --save ~/plots/my_feature.png
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        # Parse arguments
        parts = args if isinstance(args, list) else args.split()

        file_path = None
        output_folder = None
        save_path = None

        i = 0
        while i < len(parts):
            if parts[i] == '--file' and i + 1 < len(parts):
                file_path = parts[i + 1]
                i += 2
            elif parts[i] == '--output-folder' and i + 1 < len(parts):
                output_folder = parts[i + 1]
                i += 2
            elif parts[i] == '--save' and i + 1 < len(parts):
                save_path = parts[i + 1]
                i += 2
            else:
                i += 1

        # Validate required parameters
        if not file_path:
            print("❌ Error: --file is required")
            print("\nUsage: feature-plot --file <path> [--output-folder <path>] [--save <filename>]")
            print("\nExample:")
            print("  feature-plot --file /Volumes/ArcData/V3_database/experiment041/feature_files/S000512/TADC8/D000015/SID00012527_F00000238_D000015_TADC8_S008192_R000512_FS0001_N_00000064.npy")
            return

        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            return

        # Determine save location
        if save_path:
            save_location = Path(save_path)
        elif output_folder:
            output_dir = Path(output_folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_location = output_dir / f"{file_path.stem}_plot.png"
        else:
            save_location = None

        # Load and validate data
        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return

        if data.ndim != 2:
            print(f"❌ Error: Expected 2D array (windows × features), got shape {data.shape}")
            return

        filename = file_path.name
        print(f"\n📊 Feature File: {filename}")
        print(f"   Shape: {data.shape}")
        print(f"   Windows: {data.shape[0]:,}")
        print(f"   Features: {data.shape[1]}")
        print()

        # Create plot
        fig, axes = plt.subplots(data.shape[1], 1, figsize=(14, 2.5 * data.shape[1]), sharex=True)
        if data.shape[1] == 1:
            axes = [axes]

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

        for i in range(data.shape[1]):
            color = colors[i % len(colors)]
            axes[i].plot(data[:, i], linewidth=1, color=color)
            axes[i].set_ylabel(f'Feature {i}', fontsize=10, fontweight='bold')
            axes[i].grid(True, alpha=0.3, linestyle='--')

            # Add statistics
            mean = np.mean(data[:, i])
            std = np.std(data[:, i])
            min_val = np.min(data[:, i])
            max_val = np.max(data[:, i])
            axes[i].text(0.02, 0.95,
                        f'μ={mean:.2f}, σ={std:.2f}, min={min_val:.2f}, max={max_val:.2f}',
                        transform=axes[i].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        fontsize=8)

        axes[-1].set_xlabel('Window', fontsize=12, fontweight='bold')
        plt.suptitle(f'Feature File: {filename}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_location:
            plt.savefig(save_location, dpi=150, bbox_inches='tight')
            print(f"💾 Saved plot to: {save_location}")
            plt.close()
        else:
            plt.show()

        print("\n✅ Feature plotting complete!")

    def cmd_segment_plot(self, args):
        """Plot segment files with statistical analysis

        Usage: segment-plot [options]

        Options:
            --amplitude-method <method>   Select amplitude processing: raw, minmax, zscore (default: raw)
            --original-segment <id>       Original segment ID to plot
            --result-segment-size <size>  Result segment size
            --types <types>               Data types (RAW, ADC6, ADC8, etc.)
            --decimations <list>          Decimation factors (0, 7, 15, etc.)
            --output-folder <path>        Output directory for plots (required)

        Examples:
            segment-plot --original-segment 104075 --decimations 0 --output-folder ~/plots/
            segment-plot --result-segment-size 131072 --types RAW --output-folder ~/plots/
            segment-plot --file-labels 200,201 --num-points 500 --peak-detect --output-folder ~/plots/
            segment-plot --original-segment 104075 --amplitude-method minmax --output-folder ~/plots/
            segment-plot --original-segment 104075 --amplitude-method zscore --output-folder ~/plots/
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
            'output_folder': None,
            'amplitude_method': 'raw'
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
            elif parts[i] == '--amplitude-method' and i + 1 < len(parts):
                params['amplitude_method'] = parts[i + 1]
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
        print(f"Amplitude method: {params['amplitude_method']}")
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