#!/usr/bin/env python3
"""
Dynamic Experiment Configuration Builder for MLDP CLI

Filename: experiment_cli_builder.py
Author(s): Kristophor Jensen  
Date Created: 20250908_123000
Date Revised: 20250908_123000
File version: 0.0.0.1
Description: Build experiment configurations dynamically from CLI arguments
"""

import argparse
from typing import List, Optional, Dict, Any
from experiment_generation_config_extended import ExtendedExperimentConfig, FileSelectionConfig
import psycopg2
import logging

logger = logging.getLogger(__name__)


class ExperimentCLIBuilder:
    """Build experiment configurations from CLI arguments"""
    
    def __init__(self):
        """Initialize the CLI builder"""
        self.conn = None
        
    def connect_db(self):
        """Connect to database"""
        self.conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='arc_detection',
            user='kjensen'
        )
        
    def get_valid_labels(self, min_examples: int = 25, 
                        exclude_labels: List[str] = None) -> List[int]:
        """Get valid label IDs based on criteria"""
        if not self.conn:
            self.connect_db()
            
        cursor = self.conn.cursor()
        
        if exclude_labels is None:
            exclude_labels = ['trash', 'current_only', 'voltage_only', 'other']
            
        cursor.execute('''
            SELECT el.label_id, el.experiment_label, COUNT(DISTINCT fy.file_id) as file_count
            FROM experiment_labels el
            LEFT JOIN files_Y fy ON el.label_id = fy.label_id
            WHERE el.experiment_label NOT IN %s
            AND el.active = true
            GROUP BY el.label_id, el.experiment_label
            HAVING COUNT(DISTINCT fy.file_id) >= %s
            ORDER BY el.label_id
        ''', (tuple(exclude_labels), min_examples))
        
        results = cursor.fetchall()
        label_ids = [r[0] for r in results]
        
        print(f"Found {len(label_ids)} valid label classes:")
        for label_id, label_name, count in results:
            print(f"  • {label_id}: {label_name} ({count} files)")
            
        return label_ids
    
    def get_all_distance_functions(self) -> List[str]:
        """Get all available distance functions"""
        if not self.conn:
            self.connect_db()
            
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT function_name 
            FROM ml_distance_functions_lut 
            ORDER BY function_name
        ''')
        
        return [r[0] for r in cursor.fetchall()]
    
    def get_all_amplitude_methods(self) -> List[str]:
        """Get all available amplitude normalization methods"""
        if not self.conn:
            self.connect_db()
            
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT method_name 
            FROM ml_amplitude_normalization_lut 
            ORDER BY method_id
        ''')
        
        return [r[0] for r in cursor.fetchall()]
    
    def parse_cli_args(self, args_string: str) -> Dict[str, Any]:
        """Parse CLI argument string into configuration parameters"""
        parser = argparse.ArgumentParser(prog='experiment-create')
        
        # Basic experiment info
        parser.add_argument('--name', required=True, help='Experiment name')
        parser.add_argument('--description', default='', help='Experiment description')
        parser.add_argument('--type', default='classification', help='Experiment type')
        parser.add_argument('--version', default='1.0.0', help='Experiment version')
        
        # File selection
        parser.add_argument('--file-selection', default='random', 
                          choices=['random', 'all', 'specific'],
                          help='File selection strategy')
        parser.add_argument('--max-files', type=int, default=50,
                          help='Maximum number of files to select')
        parser.add_argument('--random-seed', type=int, default=42,
                          help='Random seed for reproducibility')
        parser.add_argument('--min-examples', type=int, default=25,
                          help='Minimum examples per class')
        parser.add_argument('--exclude-labels', nargs='+', 
                          default=['trash', 'current_only', 'voltage_only', 'other'],
                          help='Labels to exclude')
        parser.add_argument('--target-labels', nargs='+', type=int,
                          help='Specific label IDs to use (auto-detect if not specified)')
        
        # Segment configuration
        parser.add_argument('--segment-sizes', nargs='+', type=int,
                          default=[8192], help='Segment sizes')
        parser.add_argument('--decimations', nargs='+', type=int,
                          default=[0], help='Decimation factors')
        parser.add_argument('--data-types', nargs='+',
                          default=['raw'], help='Data types (raw, adc6, adc8, adc10, adc12, adc14)')
        
        # Amplitude methods
        parser.add_argument('--amplitude-methods', nargs='+',
                          help='Amplitude normalization methods (use "all" for all methods)')
        
        # Distance functions
        parser.add_argument('--distance-functions', nargs='+',
                          help='Distance functions to use (use "all" for all functions)')
        
        # Segment selection
        parser.add_argument('--selection-strategy', default='position_balanced_per_file',
                          help='Segment selection strategy')
        parser.add_argument('--min-segments-per-position', type=int, default=1,
                          help='Minimum segments per position type')
        parser.add_argument('--min-segments-per-file', type=int, default=3,
                          help='Minimum segments per file')
        parser.add_argument('--position-balance-mode', default='at_least_one',
                          choices=['at_least_one', 'equal', 'proportional'],
                          help='Position balance mode')
        parser.add_argument('--balanced-segments', action='store_true', default=True,
                          help='Use balanced segment selection')
        
        # Execution options
        parser.add_argument('--dry-run', action='store_true',
                          help='Validate without creating experiment')
        parser.add_argument('--force', action='store_true',
                          help='Skip confirmation prompt')
        
        # Parse the arguments
        args_list = args_string.split() if isinstance(args_string, str) else args_string
        parsed = parser.parse_args(args_list)
        
        return vars(parsed)
    
    def build_config(self, params: Dict[str, Any]) -> ExtendedExperimentConfig:
        """Build ExtendedExperimentConfig from parsed parameters"""
        
        # Handle special values for amplitude methods and distance functions
        amplitude_methods = params.get('amplitude_methods')
        if amplitude_methods and 'all' in amplitude_methods:
            amplitude_methods = self.get_all_amplitude_methods()
            print(f"Using all {len(amplitude_methods)} amplitude methods")
        elif not amplitude_methods:
            amplitude_methods = ['none']  # Default to no normalization
            
        distance_functions = params.get('distance_functions')
        if distance_functions and 'all' in distance_functions:
            distance_functions = self.get_all_distance_functions()
            print(f"Using all {len(distance_functions)} distance functions")
        elif not distance_functions:
            distance_functions = ['euclidean']  # Default to euclidean
        
        # Auto-detect target labels if not specified
        target_labels = params.get('target_labels')
        if not target_labels:
            target_labels = self.get_valid_labels(
                min_examples=params['min_examples'],
                exclude_labels=params['exclude_labels']
            )
        
        # Create file selection config
        file_selection = FileSelectionConfig(
            strategy=params['file_selection'],
            max_files=params['max_files'],
            random_seed=params['random_seed'],
            min_examples_per_class=params['min_examples'],
            exclude_labels=params['exclude_labels']
        )
        
        # Build the configuration
        config = ExtendedExperimentConfig(
            experiment_name=params['name'],
            experiment_description=params['description'],
            experiment_type=params['type'],
            experiment_version=params['version'],
            file_selection=file_selection,
            target_labels=target_labels,
            segment_sizes=params['segment_sizes'],
            decimation_factors=params['decimations'],
            data_types=params['data_types'],
            amplitude_methods=amplitude_methods,
            distance_functions=distance_functions,
            selection_strategy=params['selection_strategy'],
            min_segments_per_position=params['min_segments_per_position'],
            min_segments_per_file=params['min_segments_per_file'],
            random_seed=params['random_seed'],
            balanced_segments=params['balanced_segments'],
            position_balance_mode=params['position_balance_mode'],
            dry_run=params.get('dry_run', False)
        )
        
        return config
    
    def create_from_cli(self, args_string: str) -> ExtendedExperimentConfig:
        """Create configuration from CLI arguments string"""
        params = self.parse_cli_args(args_string)
        config = self.build_config(params)
        
        # Display configuration summary
        print("\nExperiment Configuration Built:")
        print("=" * 60)
        print(config.summary())
        print("=" * 60)
        
        return config
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


def main():
    """Example usage"""
    import sys
    
    # Example command line
    example_args = [
        '--name', 'random_50files_all_positions',
        '--description', 'Random selection of up to 50 files with all positions',
        '--file-selection', 'random',
        '--max-files', '50',
        '--random-seed', '42',
        '--min-examples', '25',
        '--exclude-labels', 'trash', 'current_only', 'voltage_only', 'other',
        '--segment-sizes', '128', '1024', '8192',
        '--decimations', '0', '7', '15',
        '--data-types', 'raw', 'adc6', 'adc8', 'adc10', 'adc12', 'adc14',
        '--amplitude-methods', 'all',
        '--distance-functions', 'all',
        '--min-segments-per-position', '1',
        '--min-segments-per-file', '3',
        '--position-balance-mode', 'at_least_one',
        '--dry-run'
    ]
    
    builder = ExperimentCLIBuilder()
    
    try:
        # Use command line args if provided, otherwise use example
        args = sys.argv[1:] if len(sys.argv) > 1 else example_args
        config = builder.create_from_cli(args)
        
        if config.validate():
            print("\n✅ Configuration is valid and ready for use")
        else:
            print("\n❌ Configuration validation failed")
            
    finally:
        builder.close()


if __name__ == "__main__":
    main()