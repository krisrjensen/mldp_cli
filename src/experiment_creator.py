#!/usr/bin/env python3
"""
Experiment Creator for MLDP CLI

Filename: experiment_creator.py
Author(s): Kristophor Jensen  
Date Created: 20250907_191500
Date Revised: 20250907_191500
File version: 0.0.0.1
Description: Creates experiments in ml_experiments table and populates junction tables
"""

import psycopg2
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class ExperimentCreator:
    """Creates experiments and populates junction tables"""
    
    def __init__(self, host='localhost', port=5432, database='arc_detection', user='kjensen'):
        """Initialize with database connection parameters"""
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user
        }
    
    def create_experiment(self, config) -> int:
        """
        Create experiment in ml_experiments table and populate junction tables
        
        Args:
            config: ExperimentGenerationConfig or ExtendedExperimentConfig
            
        Returns:
            experiment_id of created experiment
        """
        conn = psycopg2.connect(**self.connection_params)
        cursor = conn.cursor()
        
        try:
            # Start transaction
            conn.autocommit = False
            
            # 1. Create experiment record
            experiment_id = self._create_experiment_record(cursor, config)
            logger.info(f"Created experiment {experiment_id}: {config.experiment_name}")
            
            # 2. Populate junction tables
            self._populate_data_types(cursor, experiment_id, config)
            self._populate_amplitude_methods(cursor, experiment_id, config)
            self._populate_decimations(cursor, experiment_id, config)
            self._populate_distance_functions(cursor, experiment_id, config)
            
            # 3. Store segment selection config
            if hasattr(config, 'get_segment_selection_config'):
                self._update_segment_selection_config(cursor, experiment_id, config)
            
            # Commit transaction
            conn.commit()
            logger.info(f"Successfully created experiment {experiment_id} with all configurations")
            
            return experiment_id
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create experiment: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def _create_experiment_record(self, cursor, config) -> int:
        """Create the main experiment record"""
        
        # Get next experiment_id (since there's no auto-increment)
        cursor.execute('SELECT COALESCE(MAX(experiment_id), 0) + 1 FROM ml_experiments')
        next_experiment_id = cursor.fetchone()[0]
        
        # Prepare segment selection config
        if hasattr(config, 'get_segment_selection_config'):
            selection_config = json.dumps(config.get_segment_selection_config())
        else:
            selection_config = json.dumps({
                'selection_strategy': config.selection_strategy,
                'random_seed': config.random_seed,
                'balanced_segments': config.balanced_segments,
                'position_balance_mode': getattr(config, 'position_balance_mode', 'equal')
            })
        
        cursor.execute('''
            INSERT INTO ml_experiments (
                experiment_id,
                experiment_name, 
                description, 
                experiment_type,
                experiment_version,
                data_granularity_id,
                algorithm_type,
                segment_selection_config,
                created_at,
                updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, NOW(), NOW())
            RETURNING experiment_id
        ''', (
            next_experiment_id,
            config.experiment_name,
            getattr(config, 'experiment_description', ''),
            config.experiment_type,
            config.experiment_version,
            2,  # data_granularity_id: 2 for segment-level experiments
            'TBD',  # algorithm_type: default value used by all experiments
            selection_config
        ))
        
        experiment_id = cursor.fetchone()[0]
        return experiment_id
    
    def _populate_data_types(self, cursor, experiment_id: int, config):
        """Populate ml_experiments_data_types junction table"""
        
        # Delete existing entries (3-step pattern: DELETE, VALIDATE, INSERT)
        cursor.execute(
            'DELETE FROM ml_experiments_data_types WHERE experiment_id = %s',
            (experiment_id,)
        )
        
        # Get data types from config
        data_types = config.data_types if hasattr(config, 'data_types') else ['raw']
        
        for data_type in data_types:
            # Validate and insert
            cursor.execute('''
                INSERT INTO ml_experiments_data_types (experiment_id, data_type_id)
                SELECT %s, data_type_id 
                FROM ml_data_types_lut 
                WHERE data_type_name = %s
            ''', (experiment_id, data_type))
        
        logger.info(f"Added {len(data_types)} data types to experiment {experiment_id}")
    
    def _populate_amplitude_methods(self, cursor, experiment_id: int, config):
        """Populate ml_experiments_amplitude_methods junction table"""
        
        # Delete existing entries
        cursor.execute(
            'DELETE FROM ml_experiments_amplitude_methods WHERE experiment_id = %s',
            (experiment_id,)
        )
        
        # Get amplitude methods from config
        amplitude_methods = config.amplitude_methods if hasattr(config, 'amplitude_methods') else ['none']
        
        for method in amplitude_methods:
            # Map method names to IDs
            cursor.execute('''
                INSERT INTO ml_experiments_amplitude_methods (experiment_id, amplitude_method_id)
                SELECT %s, method_id 
                FROM ml_amplitude_normalization_lut 
                WHERE method_name = %s
            ''', (experiment_id, method))
        
        logger.info(f"Added {len(amplitude_methods)} amplitude methods to experiment {experiment_id}")
    
    def _populate_decimations(self, cursor, experiment_id: int, config):
        """Populate ml_experiment_decimation_junction table"""
        
        # Delete existing entries
        cursor.execute(
            'DELETE FROM ml_experiment_decimation_junction WHERE experiment_id = %s',
            (experiment_id,)
        )
        
        # Get decimation factors from config
        decimation_factors = config.decimation_factors if hasattr(config, 'decimation_factors') else [0]
        
        for decimation in decimation_factors:
            # Validate and insert
            cursor.execute('''
                INSERT INTO ml_experiment_decimation_junction (experiment_id, decimation_id)
                SELECT %s, decimation_id 
                FROM ml_experiment_decimation_lut 
                WHERE decimation_factor = %s
            ''', (experiment_id, decimation))
        
        logger.info(f"Added {len(decimation_factors)} decimation factors to experiment {experiment_id}")
    
    def _populate_distance_functions(self, cursor, experiment_id: int, config):
        """Populate ml_experiments_distance_measurements junction table"""
        
        # Delete existing entries
        cursor.execute(
            'DELETE FROM ml_experiments_distance_measurements WHERE experiment_id = %s',
            (experiment_id,)
        )
        
        # Get distance functions from config
        distance_functions = config.distance_functions if hasattr(config, 'distance_functions') else []
        
        if distance_functions:
            for function in distance_functions:
                # Validate and insert
                cursor.execute('''
                    INSERT INTO ml_experiments_distance_measurements (
                        experiment_id, distance_function_id
                    )
                    SELECT %s, distance_function_id 
                    FROM ml_distance_functions_lut 
                    WHERE function_name = %s
                ''', (experiment_id, function))
            
            logger.info(f"Added {len(distance_functions)} distance functions to experiment {experiment_id}")
    
    def _update_segment_selection_config(self, cursor, experiment_id: int, config):
        """Update segment selection config if not already set"""
        
        if hasattr(config, 'get_segment_selection_config'):
            selection_config = json.dumps(config.get_segment_selection_config())
        elif hasattr(config, 'min_segments_per_position'):
            # Extended config format
            selection_config = json.dumps({
                'selection_strategy': config.selection_strategy,
                'random_seed': config.random_seed,
                'balanced_segments': config.balanced_segments,
                'position_balance_mode': config.position_balance_mode,
                'min_segments_per_position': config.min_segments_per_position,
                'min_segments_per_file': config.min_segments_per_file,
                'file_selection': {
                    'strategy': config.file_selection.strategy,
                    'max_files': config.file_selection.max_files,
                    'random_seed': config.file_selection.random_seed
                } if hasattr(config, 'file_selection') else None
            })
        else:
            return  # No additional config to update
        
        cursor.execute('''
            UPDATE ml_experiments 
            SET segment_selection_config = %s::jsonb,
                updated_at = NOW()
            WHERE experiment_id = %s
        ''', (selection_config, experiment_id))
    
    def get_experiment_info(self, experiment_id: int) -> Dict[str, Any]:
        """Get experiment information including junction table data"""
        
        conn = psycopg2.connect(**self.connection_params)
        cursor = conn.cursor()
        
        try:
            # Get main experiment info
            cursor.execute('''
                SELECT experiment_name, description, experiment_type, 
                       experiment_version, segment_selection_config, created_at
                FROM ml_experiments
                WHERE experiment_id = %s
            ''', (experiment_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            info = {
                'experiment_id': experiment_id,
                'experiment_name': row[0],
                'description': row[1],
                'experiment_type': row[2],
                'experiment_version': row[3],
                'segment_selection_config': row[4],
                'created_at': row[5]
            }
            
            # Get data types
            cursor.execute('''
                SELECT dt.data_type_name, dt.bit_depth
                FROM ml_experiments_data_types edt
                JOIN ml_data_types_lut dt ON edt.data_type_id = dt.data_type_id
                WHERE edt.experiment_id = %s
                ORDER BY dt.bit_depth
            ''', (experiment_id,))
            info['data_types'] = [{'name': r[0], 'bits': r[1]} for r in cursor.fetchall()]
            
            # Get amplitude methods
            cursor.execute('''
                SELECT am.method_name, am.display_name
                FROM ml_experiments_amplitude_methods eam
                JOIN ml_amplitude_normalization_lut am ON eam.amplitude_method_id = am.method_id
                WHERE eam.experiment_id = %s
                ORDER BY am.method_id
            ''', (experiment_id,))
            info['amplitude_methods'] = [{'name': r[0], 'display': r[1]} for r in cursor.fetchall()]
            
            # Get decimations
            cursor.execute('''
                SELECT d.decimation_factor, d.sampling_rate_hz
                FROM ml_experiment_decimation_junction edj
                JOIN ml_experiment_decimation_lut d ON edj.decimation_id = d.decimation_id
                WHERE edj.experiment_id = %s
                ORDER BY d.decimation_factor
            ''', (experiment_id,))
            info['decimations'] = [{'factor': r[0], 'rate': r[1]} for r in cursor.fetchall()]
            
            # Get distance functions
            cursor.execute('''
                SELECT df.function_name, df.display_name
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s
                ORDER BY df.function_name
            ''', (experiment_id,))
            info['distance_functions'] = [{'name': r[0], 'display': r[1]} for r in cursor.fetchall()]
            
            return info
            
        finally:
            cursor.close()
            conn.close()


def create_experiment_from_config(config_name: str, dry_run: bool = False) -> Optional[int]:
    """
    Create an experiment from a named configuration
    
    Args:
        config_name: Name of configuration (balanced, small, large, random50, or path to JSON)
        dry_run: If True, validate but don't create
        
    Returns:
        experiment_id if created, None if dry run or error
    """
    # Import configurations
    from experiment_generation_config import (
        ExperimentGenerationConfig,
        BALANCED_18CLASS_CONFIG,
        SMALL_TEST_CONFIG,
        LARGE_UNBALANCED_CONFIG
    )
    
    # Try to import extended config
    try:
        from experiment_generation_config_extended import RANDOM_50FILES_CONFIG
        has_extended = True
    except ImportError:
        has_extended = False
    
    # Load configuration
    if config_name == 'balanced':
        config = BALANCED_18CLASS_CONFIG
    elif config_name == 'small':
        config = SMALL_TEST_CONFIG
    elif config_name == 'large':
        config = LARGE_UNBALANCED_CONFIG
    elif config_name == 'random50' and has_extended:
        config = RANDOM_50FILES_CONFIG
    elif config_name.endswith('.json'):
        import json
        with open(config_name, 'r') as f:
            config_data = json.load(f)
        
        # Determine which config class to use
        if 'file_selection' in config_data:
            from experiment_generation_config_extended import ExtendedExperimentConfig, FileSelectionConfig
            file_sel = config_data.get('file_selection', {})
            config = ExtendedExperimentConfig(
                experiment_name=config_data['experiment_name'],
                file_selection=FileSelectionConfig(**file_sel) if file_sel else FileSelectionConfig(),
                **{k: v for k, v in config_data.items() if k not in ['experiment_name', 'file_selection']}
            )
        else:
            config = ExperimentGenerationConfig.from_dict(config_data)
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
    
    # Validate configuration
    if not config.validate():
        raise ValueError("Configuration validation failed")
    
    if dry_run:
        print(f"✅ Configuration '{config_name}' is valid")
        print(f"Would create experiment: {config.experiment_name}")
        return None
    
    # Create experiment
    creator = ExperimentCreator()
    experiment_id = creator.create_experiment(config)
    
    return experiment_id


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python experiment_creator.py <config_name> [--dry-run]")
        print("\nAvailable configs:")
        print("  balanced  - 18 classes × 750 instances")
        print("  small     - 3 classes × 100 instances")
        print("  large     - 18 classes × 1000 instances")
        print("  random50  - 50 random files, all positions")
        sys.exit(1)
    
    config_name = sys.argv[1]
    dry_run = '--dry-run' in sys.argv
    
    try:
        experiment_id = create_experiment_from_config(config_name, dry_run)
        
        if experiment_id:
            print(f"✅ Created experiment {experiment_id}")
            
            # Show experiment info
            creator = ExperimentCreator()
            info = creator.get_experiment_info(experiment_id)
            
            print(f"\nExperiment Details:")
            print(f"  Name: {info['experiment_name']}")
            print(f"  Type: {info['experiment_type']}")
            print(f"  Data Types: {len(info['data_types'])}")
            print(f"  Amplitude Methods: {len(info['amplitude_methods'])}")
            print(f"  Decimations: {len(info['decimations'])}")
            print(f"  Distance Functions: {len(info['distance_functions'])}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)