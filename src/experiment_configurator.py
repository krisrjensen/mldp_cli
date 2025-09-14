"""
Filename: experiment_configurator.py
Author(s): Kristophor Jensen
Date Created: 20250913_182000
Date Revised: 20250913_182000
File version: 1.0.0.0
Description: Configure experiment parameters and feature sets
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
import json
import logging

class ExperimentConfigurator:
    """Configure experiment parameters and feature sets."""
    
    def __init__(self, experiment_id: int, db_config: Dict[str, Any]):
        self.experiment_id = experiment_id
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self):
        """Establish database connection."""
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
        
    def disconnect(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
            
    def get_next_id(self, table_name: str, id_column: str) -> int:
        """Get next ID for tables without auto-increment."""
        self.cursor.execute(
            f"SELECT COALESCE(MAX({id_column}), 0) + 1 AS next_id FROM {table_name}"
        )
        return self.cursor.fetchone()['next_id']
    
    def update_decimations(self, decimation_factors: List[int]) -> bool:
        """
        Update decimation factors for the experiment.
        
        Args:
            decimation_factors: List of decimation factors (e.g., [0, 7, 15])
            
        Returns:
            Success status
        """
        self.connect()
        try:
            # Clear existing decimations
            self.cursor.execute("""
                DELETE FROM ml_experiment_decimation_junction 
                WHERE experiment_id = %s
            """, (self.experiment_id,))
            
            # Get decimation IDs for the factors
            for factor in decimation_factors:
                # Check if decimation exists in LUT
                self.cursor.execute("""
                    SELECT decimation_id FROM ml_experiment_decimation_lut 
                    WHERE decimation_factor = %s
                """, (factor,))
                result = self.cursor.fetchone()
                
                if not result:
                    # Create new decimation entry
                    dec_id = self.get_next_id('ml_experiment_decimation_lut', 'decimation_id')
                    sample_rate = 5000000 // (factor + 1) if factor > 0 else 5000000
                    self.cursor.execute("""
                        INSERT INTO ml_experiment_decimation_lut 
                        (decimation_id, decimation_factor, equivalent_sample_rate_hz, description)
                        VALUES (%s, %s, %s, %s)
                    """, (dec_id, factor, sample_rate, f"Decimation factor {factor}"))
                else:
                    dec_id = result['decimation_id']
                
                # Add to junction table
                junction_id = self.get_next_id('ml_experiment_decimation_junction', 'experiment_decimation_id')
                self.cursor.execute("""
                    INSERT INTO ml_experiment_decimation_junction 
                    (experiment_decimation_id, experiment_id, decimation_id)
                    VALUES (%s, %s, %s)
                """, (junction_id, self.experiment_id, dec_id))
            
            self.conn.commit()
            self.logger.info(f"Updated decimations for experiment {self.experiment_id}: {decimation_factors}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error updating decimations: {e}")
            return False
        finally:
            self.disconnect()
    
    def update_segment_sizes(self, segment_sizes: List[int]) -> bool:
        """
        Update segment sizes for the experiment.
        
        Args:
            segment_sizes: List of segment sizes (e.g., [128, 1024, 8192])
            
        Returns:
            Success status
        """
        self.connect()
        try:
            # Clear existing segment sizes
            self.cursor.execute("""
                DELETE FROM ml_experiments_segment_sizes 
                WHERE experiment_id = %s
            """, (self.experiment_id,))
            
            # Add new segment sizes
            for size in segment_sizes:
                # Check if size exists in LUT
                self.cursor.execute("""
                    SELECT segment_size_id FROM ml_segment_sizes_lut 
                    WHERE segment_size_n = %s
                """, (size,))
                result = self.cursor.fetchone()
                
                if not result:
                    # Create new size entry
                    size_id = self.get_next_id('ml_segment_sizes_lut', 'segment_size_id')
                    self.cursor.execute("""
                        INSERT INTO ml_segment_sizes_lut 
                        (segment_size_id, segment_size_n, description)
                        VALUES (%s, %s, %s)
                    """, (size_id, size, f"{size} samples"))
                else:
                    size_id = result['segment_size_id']
                
                # Add to junction table
                junction_id = self.get_next_id('ml_experiments_segment_sizes', 'experiment_segment_size_id')
                self.cursor.execute("""
                    INSERT INTO ml_experiments_segment_sizes 
                    (experiment_segment_size_id, experiment_id, segment_size_id)
                    VALUES (%s, %s, %s)
                """, (junction_id, self.experiment_id, size_id))
            
            self.conn.commit()
            self.logger.info(f"Updated segment sizes for experiment {self.experiment_id}: {segment_sizes}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error updating segment sizes: {e}")
            return False
        finally:
            self.disconnect()
    
    def update_amplitude_methods(self, method_names: List[str]) -> bool:
        """
        Update amplitude normalization methods for the experiment.
        
        Args:
            method_names: List of method names (e.g., ['minmax', 'zscore'])
            
        Returns:
            Success status
        """
        self.connect()
        try:
            # Clear existing amplitude methods
            self.cursor.execute("""
                DELETE FROM ml_experiments_amplitude_methods 
                WHERE experiment_id = %s
            """, (self.experiment_id,))
            
            # Add new amplitude methods
            for method_name in method_names:
                # Get method ID from normalization LUT
                self.cursor.execute("""
                    SELECT method_id FROM ml_amplitude_normalization_lut 
                    WHERE method_name = %s
                """, (method_name,))
                result = self.cursor.fetchone()
                
                if not result:
                    self.logger.error(f"Amplitude method '{method_name}' not found in LUT")
                    continue
                    
                method_id = result['method_id']
                
                # Add to junction table
                junction_id = self.get_next_id('ml_experiments_amplitude_methods', 'experiment_amplitude_id')
                self.cursor.execute("""
                    INSERT INTO ml_experiments_amplitude_methods 
                    (experiment_amplitude_id, experiment_id, method_id)
                    VALUES (%s, %s, %s)
                """, (junction_id, self.experiment_id, method_id))
            
            self.conn.commit()
            self.logger.info(f"Updated amplitude methods for experiment {self.experiment_id}: {method_names}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error updating amplitude methods: {e}")
            return False
        finally:
            self.disconnect()
    
    def create_feature_set(self, name: str, features: List[str], n_value: int = 128) -> Optional[int]:
        """
        Create a new feature set and assign it to the experiment.
        
        Args:
            name: Feature set name
            features: List of feature names
            n_value: N value for chunking (default 128)
            
        Returns:
            Feature set ID if successful, None otherwise
        """
        self.connect()
        try:
            # Create feature set
            feature_set_id = self.get_next_id('ml_feature_sets_lut', 'feature_set_id')
            self.cursor.execute("""
                INSERT INTO ml_feature_sets_lut 
                (feature_set_id, feature_set_name, num_features, category, description)
                VALUES (%s, %s, %s, %s, %s)
            """, (feature_set_id, name, len(features), 'custom', f"Custom feature set: {name}"))
            
            # Map features to feature IDs and add to junction table
            for order, feature_name in enumerate(features, 1):
                # Handle special features like variance(voltage)
                if '(' in feature_name and ')' in feature_name:
                    # Extract base feature and function
                    func_name = feature_name.split('(')[0]  # e.g., 'variance'
                    base_feature = feature_name.split('(')[1].rstrip(')')  # e.g., 'voltage'
                    lookup_name = f"{func_name}_{base_feature}" if func_name != base_feature else func_name
                else:
                    lookup_name = feature_name
                
                # Check if feature exists in LUT
                self.cursor.execute("""
                    SELECT feature_id FROM ml_features_lut 
                    WHERE feature_name = %s
                """, (lookup_name,))
                result = self.cursor.fetchone()
                
                if not result:
                    # Create new feature
                    feature_id = self.get_next_id('ml_features_lut', 'feature_id')
                    
                    # Determine behavior type based on feature name
                    if 'variance' in lookup_name or 'mean' in lookup_name or 'std' in lookup_name:
                        behavior_type = 'chunk_statistic'
                    elif 'fft' in lookup_name.lower():
                        behavior_type = 'windowed_transform'
                    else:
                        behavior_type = 'sample_wise'
                    
                    self.cursor.execute("""
                        INSERT INTO ml_features_lut 
                        (feature_id, feature_name, feature_category, behavior_type, 
                         computation_function, output_type, description)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (feature_id, lookup_name, 'custom', behavior_type, 
                          f"calc_{lookup_name}", 'real', f"Custom feature: {lookup_name}"))
                else:
                    feature_id = result['feature_id']
                
                # Add to feature set junction
                junction_id = self.get_next_id('ml_feature_set_features', 'feature_set_feature_id')
                self.cursor.execute("""
                    INSERT INTO ml_feature_set_features 
                    (feature_set_feature_id, feature_set_id, feature_id, feature_order)
                    VALUES (%s, %s, %s, %s)
                """, (junction_id, feature_set_id, feature_id, order))
            
            # Link feature set to experiment
            exp_feature_id = self.get_next_id('ml_experiments_feature_sets', 'experiment_feature_set_id')
            self.cursor.execute("""
                INSERT INTO ml_experiments_feature_sets 
                (experiment_feature_set_id, experiment_id, feature_set_id, priority_order)
                VALUES (%s, %s, %s, %s)
            """, (exp_feature_id, self.experiment_id, feature_set_id, feature_set_id))
            
            # Add N value for this feature set
            n_value_id = self.get_next_id('ml_experiments_feature_n_values', 'experiment_feature_n_id')
            self.cursor.execute("""
                INSERT INTO ml_experiments_feature_n_values 
                (experiment_feature_n_id, experiment_id, feature_set_id, n_value)
                VALUES (%s, %s, %s, %s)
            """, (n_value_id, self.experiment_id, feature_set_id, n_value))
            
            self.conn.commit()
            self.logger.info(f"Created feature set '{name}' (ID: {feature_set_id}) with features: {features}")
            return feature_set_id
            
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error creating feature set: {e}")
            return None
        finally:
            self.disconnect()
    
    def remove_feature_set(self, feature_set_id: int) -> bool:
        """
        Remove a feature set from the experiment.
        
        Args:
            feature_set_id: ID of the feature set to remove
            
        Returns:
            Success status
        """
        self.connect()
        try:
            # Remove from junction table
            self.cursor.execute("""
                DELETE FROM ml_experiments_feature_sets 
                WHERE experiment_id = %s AND feature_set_id = %s
            """, (self.experiment_id, feature_set_id))
            
            # Also remove N values for this feature set
            self.cursor.execute("""
                DELETE FROM ml_experiments_feature_n_values 
                WHERE experiment_id = %s AND feature_set_id = %s
            """, (self.experiment_id, feature_set_id))
            
            self.conn.commit()
            self.logger.info(f"Removed feature set {feature_set_id} from experiment {self.experiment_id}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error removing feature set: {e}")
            return False
        finally:
            self.disconnect()
    
    def update_segment_selection_config(self, config_updates: Dict[str, Any]) -> bool:
        """
        Update segment selection configuration (JSONB field).
        
        Args:
            config_updates: Dictionary of configuration updates to merge
            
        Returns:
            Success status
        """
        self.connect()
        try:
            # Get current config
            self.cursor.execute("""
                SELECT segment_selection_config 
                FROM ml_experiments 
                WHERE experiment_id = %s
            """, (self.experiment_id,))
            
            result = self.cursor.fetchone()
            current_config = result['segment_selection_config'] if result else {}
            
            # Merge updates with current config
            if current_config is None:
                current_config = {}
            current_config.update(config_updates)
            
            # Update in database - use Json wrapper for proper JSONB handling
            from psycopg2.extras import Json
            self.cursor.execute("""
                UPDATE ml_experiments 
                SET segment_selection_config = %s
                WHERE experiment_id = %s
            """, (Json(current_config), self.experiment_id))
            
            self.conn.commit()
            self.logger.info(f"Updated segment selection config for experiment {self.experiment_id}: {config_updates}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error updating segment selection config: {e}")
            return False
        finally:
            self.disconnect()
    
    def clear_all_feature_sets(self) -> bool:
        """
        Remove all feature sets from the experiment.
        
        Returns:
            Success status
        """
        self.connect()
        try:
            # Remove all from junction table
            self.cursor.execute("""
                DELETE FROM ml_experiments_feature_sets 
                WHERE experiment_id = %s
            """, (self.experiment_id,))
            
            # Also remove all N values
            self.cursor.execute("""
                DELETE FROM ml_experiments_feature_n_values 
                WHERE experiment_id = %s
            """, (self.experiment_id,))
            
            self.conn.commit()
            self.logger.info(f"Cleared all feature sets from experiment {self.experiment_id}")
            return True
            
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Error clearing feature sets: {e}")
            return False
        finally:
            self.disconnect()
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current experiment configuration."""
        self.connect()
        try:
            config = {}
            
            # Get decimations
            self.cursor.execute("""
                SELECT dl.decimation_factor 
                FROM ml_experiment_decimation_junction edj
                JOIN ml_experiment_decimation_lut dl ON edj.decimation_id = dl.decimation_id
                WHERE edj.experiment_id = %s
                ORDER BY dl.decimation_factor
            """, (self.experiment_id,))
            config['decimations'] = [row['decimation_factor'] for row in self.cursor.fetchall()]
            
            # Get segment sizes
            self.cursor.execute("""
                SELECT ssl.segment_size_n 
                FROM ml_experiments_segment_sizes ess
                JOIN ml_segment_sizes_lut ssl ON ess.segment_size_id = ssl.segment_size_id
                WHERE ess.experiment_id = %s
                ORDER BY ssl.segment_size_n
            """, (self.experiment_id,))
            config['segment_sizes'] = [row['segment_size_n'] for row in self.cursor.fetchall()]
            
            # Get amplitude methods
            self.cursor.execute("""
                SELECT anl.method_name 
                FROM ml_experiments_amplitude_methods eam
                JOIN ml_amplitude_normalization_lut anl ON eam.method_id = anl.method_id
                WHERE eam.experiment_id = %s
                ORDER BY anl.method_id
            """, (self.experiment_id,))
            config['amplitude_methods'] = [row['method_name'] for row in self.cursor.fetchall()]
            
            # Get feature sets with IDs and data channels (if column exists)
            self.cursor.execute("""
                SELECT 
                    fsl.feature_set_id,
                    fsl.feature_set_name,
                    STRING_AGG(fl.feature_name, ', ' ORDER BY fsf.feature_order) as features,
                    ARRAY_AGG(DISTINCT efn.n_value ORDER BY efn.n_value) as n_values,
                    COALESCE(efs.data_channel, 'load_voltage') as data_channel
                FROM ml_experiments_feature_sets efs
                JOIN ml_feature_sets_lut fsl ON efs.feature_set_id = fsl.feature_set_id
                JOIN ml_feature_set_features fsf ON fsl.feature_set_id = fsf.feature_set_id
                JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                LEFT JOIN ml_experiments_feature_n_values efn 
                    ON efs.experiment_id = efn.experiment_id 
                    AND efs.feature_set_id = efn.feature_set_id
                WHERE efs.experiment_id = %s
                GROUP BY fsl.feature_set_id, fsl.feature_set_name, efs.data_channel
                ORDER BY MIN(efs.priority_order)
            """, (self.experiment_id,))
            
            config['feature_sets'] = []
            for row in self.cursor.fetchall():
                config['feature_sets'].append({
                    'id': row['feature_set_id'],
                    'name': row['feature_set_name'],
                    'features': row['features'],
                    'n_values': row['n_values'],
                    'data_channel': row.get('data_channel', 'load_voltage')
                })
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error getting configuration: {e}")
            return {}
        finally:
            self.disconnect()
    
    def add_feature_set(self, feature_set_id: int, n_value: int = None, data_channel: str = 'load_voltage') -> bool:
        """Add an existing feature set to the experiment with specified data channel
        
        Args:
            feature_set_id: ID of the feature set to add
            n_value: Optional N value for chunk size
            data_channel: Data channel ('source_current' or 'load_voltage', default 'load_voltage')
        """
        try:
            self.connect()
            
            # Check if feature set exists
            self.cursor.execute("""
                SELECT feature_set_id, feature_set_name 
                FROM ml_feature_sets_lut 
                WHERE feature_set_id = %s
            """, (feature_set_id,))
            
            result = self.cursor.fetchone()
            if not result:
                self.logger.warning(f"Feature set {feature_set_id} does not exist")
                return False
            
            # Check if already linked
            self.cursor.execute("""
                SELECT experiment_feature_set_id FROM ml_experiments_feature_sets 
                WHERE experiment_id = %s AND feature_set_id = %s
            """, (self.experiment_id, feature_set_id))
            
            if self.cursor.fetchone():
                self.logger.info(f"Feature set {feature_set_id} already linked to experiment {self.experiment_id}")
                return False
            
            # Get next ID for junction table
            self.cursor.execute("""
                SELECT COALESCE(MAX(experiment_feature_set_id), 0) + 1 AS next_id 
                FROM ml_experiments_feature_sets
            """)
            next_id = self.cursor.fetchone()['next_id']
            
            # Add to junction table with data channel
            self.cursor.execute("""
                INSERT INTO ml_experiments_feature_sets (experiment_feature_set_id, experiment_id, feature_set_id, priority_order, data_channel)
                VALUES (%s, %s, %s, %s, %s)
            """, (next_id, self.experiment_id, feature_set_id, next_id, data_channel))
            
            # Add N value if specified
            if n_value:
                self.cursor.execute("""
                    SELECT COALESCE(MAX(experiment_feature_n_id), 0) + 1 AS next_id 
                    FROM ml_experiments_feature_n_values
                """)
                n_value_id = self.cursor.fetchone()['next_id']
                
                self.cursor.execute("""
                    INSERT INTO ml_experiments_feature_n_values 
                    (experiment_feature_n_id, experiment_id, feature_set_id, n_value)
                    VALUES (%s, %s, %s, %s)
                """, (n_value_id, self.experiment_id, feature_set_id, n_value))
            
            self.conn.commit()
            self.logger.info(f"Added feature set {feature_set_id} to experiment {self.experiment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding feature set: {e}")
            if self.conn:
                self.conn.rollback()
            return False
        finally:
            self.disconnect()
    
    def add_multiple_feature_sets(self, feature_set_ids: list, n_value: int = None, data_channel: str = 'load_voltage') -> dict:
        """Add multiple existing feature sets to the experiment with specified data channel
        
        Args:
            feature_set_ids: List of feature set IDs to add
            n_value: Optional N value for chunk size
            data_channel: Data channel ('source_current' or 'load_voltage', default 'load_voltage')
        """
        results = {}
        for fs_id in feature_set_ids:
            results[fs_id] = self.add_feature_set(fs_id, n_value, data_channel)
        return results