#!/usr/bin/env python3
"""
Filename: experiment_query_pg.py
Author(s): Kristophor Jensen
Date Created: 20250907_140000
Date Revised: 20250907_140000
File version: 0.0.0.1
Description: Query ML experiments from PostgreSQL database with normalized junction tables
"""

import os
import sys
import psycopg2
import psycopg2.extras
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentQueryPG:
    """
    Query ML experiments from PostgreSQL database with normalized schema
    Handles junction tables for decimations, segment sizes, amplitude normalizations
    """
    
    def __init__(self, host='localhost', port=5432, database='arc_detection', user='kjensen'):
        """Initialize with PostgreSQL connection"""
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user
        }
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish PostgreSQL connection"""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            logger.info(f"✓ Connected to PostgreSQL: {self.connection_params['database']}@{self.connection_params['host']}")
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from PostgreSQL")
    
    def execute_query(self, query: str, params: tuple = None) -> List[tuple]:
        """Execute a query and return results"""
        self.connect()
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            return results
        finally:
            self.disconnect()
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments from ml_experiments table
        
        Returns:
            List of experiment dictionaries with basic info
        """
        # Ensure connection is clean
        self.conn.rollback()  # Clear any failed transactions
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        try:
            # Query ml_experiments table with actual column names
            cursor.execute("""
                SELECT 
                    experiment_id,
                    experiment_name,
                    description,
                    experiment_type,
                    experiment_version,
                    data_granularity_id,
                    algorithm_type,
                    status,
                    created_by,
                    reference_hash,
                    segment_balance,
                    max_segments,
                    generate_debug_info,
                    wait_for_vscode_debugger
                FROM ml_experiments
                ORDER BY experiment_id
            """)
            
            experiments = []
            for row in cursor:
                exp = {
                    'experiment_id': row['experiment_id'],
                    'name': row['experiment_name'] or f"Experiment {row['experiment_id']}",
                    'description': row['description'] or 'No description',
                    'status': row['status'] or 'unknown',
                    'experiment_type': row['experiment_type'],
                    'experiment_version': row['experiment_version'],
                    'data_granularity_id': row['data_granularity_id'],
                    'algorithm_type': row['algorithm_type'],
                    'created_by': row['created_by'],
                    'segment_balance': row['segment_balance'],
                    'max_segments': row['max_segments']
                }
                
                experiments.append(exp)
            
            return experiments
            
        finally:
            cursor.close()
    
    def get_experiment_details(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get detailed information including junction table relationships
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary with complete experiment details
        """
        # Create a new connection for this query to avoid transaction issues
        conn = psycopg2.connect(**self.connection_params)
        cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        try:
            # Get basic experiment info with all relevant fields
            cursor.execute("""
                SELECT 
                    experiment_id,
                    experiment_name,
                    experiment_type,
                    experiment_version,
                    description,
                    algorithm_type,
                    segment_balance,
                    max_segments,
                    random_seed,
                    segment_selection_config,
                    feature_extraction_method,
                    feature_configuration,
                    status,
                    training_batch_size,
                    max_epochs,
                    cross_validation_folds,
                    data_granularity_id,
                    created_at,
                    updated_at
                FROM ml_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            details = dict(row)
            
            # Get decimation factors from junction table
            try:
                details['decimations'] = self._get_decimations(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting decimations: {e}")
                details['decimations'] = []
            
            # Get segment sizes from junction table
            try:
                details['segment_sizes'] = self._get_segment_sizes(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting segment_sizes: {e}")
                details['segment_sizes'] = []
            
            # Get amplitude normalization methods
            try:
                details['amplitude_methods'] = self._get_amplitude_methods(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting amplitude_methods: {e}")
                details['amplitude_methods'] = []
            
            # Get data types
            try:
                details['data_types'] = self._get_data_types(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting data_types: {e}")
                details['data_types'] = []
            
            # Get segment statistics
            try:
                # Reset connection state if needed
                conn.rollback()
                details['segments'] = self._get_segment_statistics(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting segments: {e}")
                details['segments'] = {'total': 0, 'by_size': [], 'by_label': []}
            
            # Get distance table info
            try:
                details['distances'] = self._get_distance_info(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting distances: {e}")
                details['distances'] = {}
            
            # Get distance functions from junction table
            try:
                details['distance_functions'] = self._get_distance_functions(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting distance_functions: {e}")
                details['distance_functions'] = []
            
            # Get file/segment labels from junction tables
            try:
                details['file_labels'] = self._get_file_labels(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting file_labels: {e}")
                details['file_labels'] = []
                
            try:
                details['segment_labels'] = self._get_segment_labels(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting segment_labels: {e}")
                details['segment_labels'] = []
            
            # Get feature sets from junction table
            try:
                details['feature_sets'] = self._get_feature_sets(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting feature_sets: {e}")
                details['feature_sets'] = []
            
            # Get pipeline associations
            try:
                details['pipelines'] = self._get_pipelines(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting pipelines: {e}")
                details['pipelines'] = []

            # Get segment training data
            try:
                details['segment_training_data'] = self._get_segment_training_data(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting segment training data: {e}")
                details['segment_training_data'] = {'total_segments': 0, 'unique_labels': 0, 'label_distribution': []}

            # Get file-segment label cross-distribution
            try:
                details['file_segment_distribution'] = self._get_file_segment_label_distribution(cursor, experiment_id)
            except Exception as e:
                logger.debug(f"Error getting file-segment distribution: {e}")
                details['file_segment_distribution'] = {'has_data': False, 'distributions': []}

            return details
            
        finally:
            cursor.close()
            conn.close()
    
    def _get_decimations(self, cursor, experiment_id: int) -> List[Dict]:
        """Get decimation factors for experiment from junction table"""
        try:
            # Query junction table with LUT join for decimation factors
            cursor.execute("""
                SELECT 
                    dl.decimation_id,
                    dl.decimation_factor,
                    dl.description,
                    dl.is_active
                FROM ml_experiment_decimation_junction dj
                JOIN ml_experiment_decimation_lut dl ON dj.decimation_id = dl.decimation_id
                WHERE dj.experiment_id = %s
                ORDER BY dl.decimation_factor
            """, (experiment_id,))
            
            decimations = []
            for row in cursor.fetchall():
                decimations.append({
                    'decimation_id': row[0],
                    'decimation_factor': row[1],
                    'description': row[2],
                    'is_active': row[3]
                })
            
            return decimations
        except Exception as e:
            logger.debug(f"Error getting decimations: {e}")
            return []
    
    def _get_segment_sizes(self, cursor, experiment_id: int) -> List[Dict]:
        """Get segment sizes from junction table"""
        try:
            # Query junction table with segment sizes lookup table
            cursor.execute("""
                SELECT 
                    ssl.segment_size_id,
                    ssl.segment_size_n,
                    ssl.description
                FROM ml_experiments_segment_sizes ess
                JOIN ml_segment_sizes_lut ssl ON ess.segment_size_id = ssl.segment_size_id
                WHERE ess.experiment_id = %s
                ORDER BY ssl.segment_size_n
            """, (experiment_id,))
            
            segment_sizes = []
            for row in cursor.fetchall():
                segment_sizes.append({
                    'segment_size_id': row[0],
                    'segment_size': row[1],  # Keep as 'segment_size' for backward compatibility
                    'description': row[2] if len(row) > 2 else None
                })
            
            return segment_sizes
        except Exception as e:
            logger.debug(f"Error getting segment sizes: {e}")
            return []
    
    def _get_amplitude_methods(self, cursor, experiment_id: int) -> List[Dict]:
        """Get amplitude normalization methods from junction table"""
        try:
            # Query junction table with amplitude normalization LUT
            cursor.execute("""
                SELECT 
                    al.method_id,
                    al.method_name,
                    al.description,
                    al.is_active
                FROM ml_experiments_amplitude_methods eam
                JOIN ml_amplitude_normalization_lut al ON eam.method_id = al.method_id
                WHERE eam.experiment_id = %s
                ORDER BY al.method_id
            """, (experiment_id,))
            
            amplitude_methods = []
            for row in cursor.fetchall():
                amplitude_methods.append({
                    'method_id': row[0],
                    'method_name': row[1],
                    'description': row[2],
                    'is_active': row[3]
                })
            
            return amplitude_methods
        except Exception as e:
            logger.debug(f"Error getting amplitude methods: {e}")
            return []
    
    def _get_data_types(self, cursor, experiment_id: int) -> List[Dict]:
        """Get data types from junction table"""  
        try:
            # Query junction table with data types LUT
            cursor.execute("""
                SELECT 
                    dt.data_type_id,
                    dt.data_type_name,
                    dt.bit_depth,
                    dt.is_active
                FROM ml_experiments_data_types edt
                JOIN ml_data_types_lut dt ON edt.data_type_id = dt.data_type_id
                WHERE edt.experiment_id = %s
                ORDER BY dt.data_type_id
            """, (experiment_id,))
            
            data_types = []
            for row in cursor.fetchall():
                data_types.append({
                    'data_type_id': row[0],
                    'data_type_name': row[1],
                    'bit_depth': row[2],
                    'is_active': row[3]
                })
            
            return data_types
        except Exception as e:
            logger.debug(f"Error getting data types: {e}")
            return []
    
    def _get_segment_statistics(self, cursor, experiment_id: int) -> Dict[str, Any]:
        """Get segment statistics for experiment"""
        stats = {
            'total': 0,
            'by_size': [],
            'by_label': []
        }
        
        # Check for experiment-specific segment table
        table_name = f"experiment_{experiment_id:03d}_segments"
        
        # Check if table exists
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            )
        """, (table_name,))
        
        if cursor.fetchone()[0]:
            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            stats['total'] = cursor.fetchone()[0]
            
            # Get distribution by segment size
            cursor.execute(f"""
                SELECT 
                    segment_size,
                    COUNT(*) as count
                FROM {table_name}
                GROUP BY segment_size
                ORDER BY segment_size
            """)
            
            for row in cursor.fetchall():
                stats['by_size'].append({
                    'size': row[0],
                    'count': row[1]
                })
            
            # Get distribution by label if column exists
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = %s AND column_name = 'segment_label_id'
            """, (table_name,))
            
            if cursor.fetchone():
                cursor.execute(f"""
                    SELECT 
                        segment_label_id,
                        COUNT(*) as count
                    FROM {table_name}
                    GROUP BY segment_label_id
                    ORDER BY count DESC
                """)
                
                for row in cursor.fetchall():
                    stats['by_label'].append({
                        'label_id': row[0],
                        'count': row[1]
                    })
        
        return stats
    
    def _get_distance_info(self, cursor, experiment_id: int) -> Dict[str, int]:
        """Get distance calculation info"""
        distances = {}
        
        # Check for distance tables with 4-field primary key
        for metric in ['L1', 'L2', 'cosine']:
            table_name = f"experiment_{experiment_id:03d}_distance_{metric}"
            
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            if cursor.fetchone()[0]:
                # Count records considering the 4-field PK structure
                cursor.execute(f"""
                    SELECT COUNT(*) FROM {table_name}
                """)
                distances[metric] = cursor.fetchone()[0]
        
        return distances
    
    def _get_distance_functions(self, cursor, experiment_id: int) -> List[Dict]:
        """Get distance functions from junction table"""
        try:
            cursor.execute("""
                SELECT 
                    df.distance_function_id,
                    df.function_name,
                    df.description,
                    df.is_active
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s
                ORDER BY df.function_name
            """, (experiment_id,))
            
            distance_functions = []
            for row in cursor.fetchall():
                distance_functions.append({
                    'distance_function_id': row[0],
                    'function_name': row[1],
                    'description': row[2] if len(row) > 2 else None,
                    'is_active': row[3] if len(row) > 3 else True
                })
            
            return distance_functions
        except Exception as e:
            logger.debug(f"Error getting distance functions: {e}")
            return []
    
    def _get_file_labels(self, cursor, experiment_id: int) -> List[Dict]:
        """Get file labels from junction table"""
        try:
            cursor.execute("""
                SELECT 
                    el.label_id,
                    el.experiment_label,
                    el.short_name,
                    el.description
                FROM ml_experiments_file_labels_junction flj
                JOIN experiment_labels el ON flj.experiment_label_id = el.label_id
                WHERE flj.experiment_id = %s
                ORDER BY el.label_id
            """, (experiment_id,))
            
            file_labels = []
            for row in cursor.fetchall():
                file_labels.append({
                    'label_id': row[0],
                    'experiment_label': row[1],
                    'short_name': row[2],
                    'description': row[3]
                })
            
            return file_labels
        except Exception as e:
            logger.debug(f"Error getting file labels: {e}")
            return []
    
    def _get_segment_labels(self, cursor, experiment_id: int) -> List[Dict]:
        """Get segment labels from junction table"""
        try:
            cursor.execute("""
                SELECT 
                    sl.label_id,
                    sl.label_name,
                    sl.display_name,
                    sl.description
                FROM ml_experiments_segment_labels_junction slj
                JOIN segment_labels sl ON slj.segment_label_id = sl.label_id
                WHERE slj.experiment_id = %s
                ORDER BY sl.label_id
            """, (experiment_id,))
            
            segment_labels = []
            for row in cursor.fetchall():
                segment_labels.append({
                    'label_id': row[0],
                    'label_name': row[1],
                    'display_name': row[2],
                    'description': row[3]
                })
            
            return segment_labels
        except Exception as e:
            logger.debug(f"Error getting segment labels: {e}")
            return []
    
    def _get_feature_sets(self, cursor, experiment_id: int) -> List[Dict]:
        """Get feature sets from junction tables"""
        try:
            cursor.execute("""
                SELECT 
                    fsl.feature_set_id,
                    fsl.feature_set_name,
                    COUNT(DISTINCT fsf.feature_id) as num_features,
                    fsl.category,
                    fsl.description,
                    STRING_AGG(fl.feature_name || ' (' || fl.behavior_type || ')', ', ' ORDER BY fsf.feature_order) as features,
                    efs.n_value,
                    COALESCE(efs.data_channel, 'load_voltage') as data_channel
                FROM ml_experiments_feature_sets efs
                JOIN ml_feature_sets_lut fsl ON efs.feature_set_id = fsl.feature_set_id
                LEFT JOIN ml_feature_set_features fsf ON fsl.feature_set_id = fsf.feature_set_id
                LEFT JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                WHERE efs.experiment_id = %s
                GROUP BY fsl.feature_set_id, fsl.feature_set_name, fsl.category, fsl.description, efs.data_channel, efs.n_value
                ORDER BY MIN(efs.priority_order)
            """, (experiment_id,))
            
            feature_sets = []
            for row in cursor.fetchall():
                feature_sets.append({
                    'feature_set_id': row[0],
                    'feature_set_name': row[1],
                    'num_features': row[2],
                    'category': row[3],
                    'description': row[4],
                    'features': row[5],
                    'n_value': row[6],
                    'data_channel': row[7] if len(row) > 7 else 'load_voltage'
                })
            
            return feature_sets
        except Exception as e:
            logger.debug(f"Error getting feature sets: {e}")
            return []
    
    def _get_pipelines(self, cursor, experiment_id: int) -> List[Dict]:
        """Get pipeline associations for experiment"""
        try:
            cursor.execute("""
                SELECT
                    ep.pipeline_id,
                    p.pipeline_name,
                    p.description,
                    p.status,
                    ep.created_at
                FROM ml_experiment_pipelines ep
                JOIN ml_pipelines p ON ep.pipeline_id = p.pipeline_id
                WHERE ep.experiment_id = %s
                ORDER BY ep.created_at DESC
            """, (experiment_id,))

            pipelines = []
            for row in cursor.fetchall():
                pipelines.append({
                    'pipeline_id': row[0],
                    'pipeline_name': row[1],
                    'description': row[2],
                    'status': row[3],
                    'created_at': row[4].isoformat() if row[4] else None
                })

            return pipelines
        except Exception as e:
            logger.debug(f"Error getting pipelines: {e}")
            return []

    def _get_file_segment_label_distribution(self, cursor, experiment_id: int) -> Dict[str, Any]:
        """
        Get distribution of segment labels within each file label.

        Note: Handles column name variations:
        - Experiment 18: Uses 'assigned_label' (published data, cannot change)
        - Other experiments: Use 'file_label_name' (standard)
        """
        try:
            data = {
                'has_data': False,
                'distributions': []
            }

            # Check if both file and segment training data tables exist
            file_table = f"experiment_{experiment_id:03d}_file_training_data"
            seg_table = f"experiment_{experiment_id:03d}_segment_training_data"

            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                ) AND EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (file_table, seg_table))

            if cursor.fetchone()[0]:
                # Detect which column name is used (assigned_label for exp 18, file_label_name for others)
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = %s
                    AND column_name IN ('assigned_label', 'file_label_name')
                    LIMIT 1
                """, (file_table,))

                label_column_result = cursor.fetchone()
                if not label_column_result:
                    return data  # No label column found

                label_column = label_column_result[0]

                # Get the cross-distribution using the correct column
                cursor.execute(f"""
                    SELECT
                        ft.{label_column} as file_label,
                        COALESCE(sl.label_name, 'unlabeled') as segment_label,
                        COUNT(DISTINCT st.segment_id) as segment_count
                    FROM {file_table} ft
                    JOIN data_segments ds ON ft.file_id = ds.experiment_file_id
                    JOIN {seg_table} st ON ds.segment_id = st.segment_id
                    LEFT JOIN segment_labels sl ON ds.segment_label_id = sl.label_id
                    WHERE ft.experiment_id = %s AND st.experiment_id = %s
                    GROUP BY ft.{label_column}, sl.label_name
                    ORDER BY ft.{label_column}, segment_count DESC
                """, (experiment_id, experiment_id))

                results = cursor.fetchall()
                if results:
                    data['has_data'] = True
                    # Organize by file label
                    file_label_dict = {}
                    for file_label, seg_label, count in results:
                        if file_label not in file_label_dict:
                            file_label_dict[file_label] = []
                        file_label_dict[file_label].append({
                            'segment_label': seg_label,
                            'count': count
                        })

                    # Convert to list format
                    for file_label, segments in file_label_dict.items():
                        data['distributions'].append({
                            'file_label': file_label,
                            'segments': segments,
                            'total': sum(s['count'] for s in segments)
                        })

            return data
        except Exception as e:
            logger.debug(f"Error getting file-segment label distribution: {e}")
            return {'has_data': False, 'distributions': []}

    def _get_segment_training_data(self, cursor, experiment_id: int) -> Dict[str, Any]:
        """Get segment training data with label distribution"""
        try:
            data = {
                'total_segments': 0,
                'unique_labels': 0,
                'label_distribution': []
            }

            # Check if segment training data table exists
            seg_table = f"experiment_{experiment_id:03d}_segment_training_data"
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (seg_table,))

            if cursor.fetchone()[0]:
                # Get total count
                cursor.execute(f"""
                    SELECT COUNT(DISTINCT segment_id) as total_segments
                    FROM {seg_table}
                    WHERE experiment_id = %s
                """, (experiment_id,))
                result = cursor.fetchone()
                if result:
                    data['total_segments'] = result[0]

                # Get segment label distribution
                cursor.execute(f"""
                    SELECT
                        COALESCE(sl.label_name, 'unlabeled') as label_name,
                        COUNT(*) as count
                    FROM {seg_table} st
                    JOIN data_segments ds ON st.segment_id = ds.segment_id
                    LEFT JOIN segment_labels sl ON ds.segment_label_id = sl.label_id
                    WHERE st.experiment_id = %s
                    GROUP BY sl.label_name
                    ORDER BY count DESC, label_name
                """, (experiment_id,))

                labels = cursor.fetchall()
                if labels:
                    data['unique_labels'] = len(labels)
                    data['label_distribution'] = [
                        {'label_name': label[0], 'count': label[1]}
                        for label in labels
                    ]
            else:
                # Try to get data from segment_selection_log if training table doesn't exist
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_name = 'segment_selection_log'
                    )
                """)

                if cursor.fetchone()[0]:
                    # Get total segments for this experiment
                    cursor.execute("""
                        SELECT COUNT(DISTINCT segment_id) as total_segments
                        FROM segment_selection_log
                        WHERE experiment_id = %s
                    """, (experiment_id,))
                    result = cursor.fetchone()
                    if result and result[0]:
                        data['total_segments'] = result[0]

                        # Get segment label distribution
                        cursor.execute("""
                            SELECT
                                COALESCE(sl.label_name, 'unlabeled') as label_name,
                                COUNT(DISTINCT ssl.segment_id) as count
                            FROM segment_selection_log ssl
                            JOIN data_segments ds ON ssl.segment_id = ds.segment_id
                            LEFT JOIN segment_labels sl ON ds.segment_label_id = sl.label_id
                            WHERE ssl.experiment_id = %s
                            GROUP BY sl.label_name
                            ORDER BY count DESC, label_name
                        """, (experiment_id,))

                        labels = cursor.fetchall()
                        if labels:
                            data['unique_labels'] = len(labels)
                            data['label_distribution'] = [
                                {'label_name': label[0], 'count': label[1]}
                                for label in labels
                            ]

            return data
        except Exception as e:
            logger.debug(f"Error getting segment training data: {e}")
            return {'total_segments': 0, 'unique_labels': 0, 'label_distribution': []}
    
    def get_experiment_configuration(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get complete configuration including all junction table relationships
        
        Returns configuration suitable for experiment generation
        """
        details = self.get_experiment_details(experiment_id)
        
        config = {
            'experiment_id': experiment_id,
            'name': details.get('experiment_name', f'Experiment {experiment_id}'),
            'description': details.get('description', ''),
            'status': details.get('status', 'unknown'),
            'experiment_type': details.get('experiment_type'),
            'experiment_version': details.get('experiment_version'),
            'data_granularity_id': details.get('data_granularity_id'),
            'algorithm_type': details.get('algorithm_type'),
            'segment_balance': details.get('segment_balance'),
            'max_segments': details.get('max_segments'),
            # Junction table data
            'decimations': [d['decimation_factor'] for d in details.get('decimations', [])],
            'segment_sizes': [s['segment_size'] for s in details.get('segment_sizes', [])],
            'data_types': [dt['data_type_name'] for dt in details.get('data_types', [])],
            'amplitude_methods': [am['method_name'] for am in details.get('amplitude_methods', [])],
            'distance_functions': [df['function_name'] for df in details.get('distance_functions', [])],
            'file_labels': [fl['experiment_label'] for fl in details.get('file_labels', [])],
            'segment_labels': [sl['label_name'] for sl in details.get('segment_labels', [])],
            # Statistics
            'total_segments': details.get('segments', {}).get('total', 0),
            'segment_distribution': details.get('segments', {}).get('by_size', []),
            'distance_calculations': details.get('distances', {})
        }
        
        return config
    
    def print_experiment_summary(self, experiment_id: int):
        """Print formatted experiment summary"""
        details = self.get_experiment_details(experiment_id)
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {experiment_id}: {details.get('experiment_name', f'Experiment {experiment_id}')}")
        print(f"{'='*70}")
        
        # Basic Information
        print(f"\n📋 BASIC INFORMATION")
        print(f"Status: {details.get('status', 'unknown')}")
        print(f"Type: {details.get('experiment_type', 'N/A')}")
        print(f"Version: {details.get('experiment_version', 'N/A')}")
        print(f"Algorithm: {details.get('algorithm_type', 'N/A')}")
        print(f"Description: {details.get('description', 'No description')}")
        print(f"Created: {details.get('created_at', 'N/A')}")
        print(f"Updated: {details.get('updated_at', 'N/A')}")
        
        # Segment Selection Configuration
        print(f"\n📋 SEGMENT SELECTION")
        print(f"Segment Balance: {details.get('segment_balance', 'N/A')}")
        print(f"Max Segments: {details.get('max_segments') or 'Unlimited'}")
        print(f"Random Seed: {details.get('random_seed') or 'Not set'}")
        
        # Segment selection config (JSONB field)
        seg_config = details.get('segment_selection_config')
        if seg_config:
            # Show max_files_per_label prominently if it exists
            if 'max_files_per_label' in seg_config:
                print(f"Max Files Per Label: {seg_config['max_files_per_label']}")
            
            print(f"\nSelection Strategy:")
            for key, value in seg_config.items():
                key_display = key.replace('_', ' ').title()
                print(f"  • {key_display}: {value}")
        
        print(f"\n📊 CONFIGURATION (Junction Tables)")
        
        # Decimations
        decimations = details.get('decimations', [])
        if decimations:
            print(f"Decimation Factors: {[d['decimation_factor'] for d in decimations]}")
        
        # Segment Sizes
        segment_sizes = details.get('segment_sizes', [])
        if segment_sizes:
            print(f"Segment Sizes: {[s['segment_size'] for s in segment_sizes]}")
        
        # Data Types
        data_types = details.get('data_types', [])
        if data_types:
            print(f"Data Types: {[dt['data_type_name'] for dt in data_types]}")
        
        # Amplitude Methods
        amplitude_methods = details.get('amplitude_methods', [])
        if amplitude_methods:
            print(f"Amplitude Methods: {[am['method_name'] for am in amplitude_methods]}")
        
        # Distance Functions
        distance_functions = details.get('distance_functions', [])
        if distance_functions:
            print(f"Distance Functions: {[df['function_name'] for df in distance_functions]}")
        
        # Feature Sets
        feature_sets = details.get('feature_sets', [])
        if feature_sets:
            print(f"\n🧬 FEATURE SETS ({len(feature_sets)} total)")
            for fs in feature_sets:
                print(f"\n  📦 {fs['feature_set_name']} (ID: {fs['feature_set_id']})")
                if fs['description']:
                    print(f"     Description: {fs['description']}")
                print(f"     Category: {fs['category']}")
                print(f"     Data channel: {fs.get('data_channel', 'load_voltage')}")
                print(f"     Number of features: {fs['num_features']}")
                if fs.get('n_value'):
                    print(f"     N value (chunk size): {fs['n_value']}")
                if fs['features']:
                    # Split features for better display
                    features_str = fs['features']
                    if features_str and len(features_str) > 100:
                        # Show first few features if too long
                        feature_list = features_str.split(', ')[:5]
                        print(f"     Features: {', '.join(feature_list)}...")
                        print(f"               (and {len(features_str.split(', ')) - 5} more)")
                    else:
                        print(f"     Features: {features_str}")
        
        # Labels
        file_labels = details.get('file_labels', [])
        if file_labels:
            print(f"\n📁 File Labels:")
            for fl in file_labels:
                print(f"  - {fl['experiment_label']} (ID: {fl['label_id']})")
        
        segment_labels = details.get('segment_labels', [])
        if segment_labels:
            print(f"\n🏷️ Segment Labels:")
            for sl in segment_labels:
                print(f"  - {sl['label_name']} ({sl['display_name']}, ID: {sl['label_id']})")

        # Segment Training Data with bar chart
        segment_training = details.get('segment_training_data', {})
        if segment_training and segment_training.get('total_segments'):
            print(f"\n📊 SEGMENT TRAINING DATA:")
            print("=" * 60)
            print(f"Total segments: {segment_training['total_segments']}")
            print(f"Unique labels: {segment_training['unique_labels']}")

            label_dist = segment_training.get('label_distribution', [])
            if label_dist:
                print("\nSegment Label Distribution:")
                max_count = max(item['count'] for item in label_dist)
                for item in label_dist:
                    bar_length = int(item['count'] / max_count * 30)
                    bar = '█' * bar_length
                    print(f"  {item['label_name']:35} {item['count']:4} {bar}")

        # File-Segment Label Cross-Distribution
        file_seg_dist = details.get('file_segment_distribution', {})
        if file_seg_dist.get('has_data'):
            print(f"\n📂 FILE LABEL → SEGMENT LABEL DISTRIBUTION:")
            print("=" * 60)

            distributions = file_seg_dist.get('distributions', [])
            # Sort by file label for consistent display
            distributions.sort(key=lambda x: x['file_label'])

            for dist in distributions:
                file_label = dist['file_label']
                segments = dist['segments']
                total = dist['total']

                print(f"\n{file_label} ({total} segments):")

                if segments:
                    # Find max count for this file label's segments
                    max_seg_count = max(s['count'] for s in segments)

                    for seg in segments:
                        # Create proportional bar relative to this file label's max
                        bar_length = int(seg['count'] / max_seg_count * 25)
                        bar = '▪' * bar_length
                        # Calculate percentage of segments in this file label
                        percentage = (seg['count'] / total * 100) if total > 0 else 0
                        print(f"    {seg['segment_label']:32} {seg['count']:4} ({percentage:5.1f}%) {bar}")

        # Segments
        segments = details.get('segments', {})
        if segments.get('total'):
            print(f"\n📈 SEGMENTS")
            print(f"Total: {segments['total']:,}")
            if segments.get('by_size'):
                print("\nBy Size:")
                data = [(f"{item['size']:,}", f"{item['count']:,}") 
                       for item in segments['by_size']]
                print(tabulate(data, headers=['Size', 'Count'], tablefmt='simple'))
        
        # Distance calculations
        distances = details.get('distances', {})
        if distances:
            print(f"\n📏 DISTANCE CALCULATIONS")
            for metric, count in distances.items():
                print(f"{metric}: {count:,}")
        
        # Pipelines
        pipelines = details.get('pipelines', [])
        if pipelines:
            print(f"\n🔧 PIPELINES")
            for pipeline in pipelines:
                status_icon = '✅' if pipeline['status'] == 'completed' else '❌' if pipeline['status'] == 'failed' else '🔄'
                print(f"  {status_icon} {pipeline['pipeline_name']} (ID: {pipeline['pipeline_id']}, Status: {pipeline['status']})")
        
        print(f"\n{'='*70}\n")
    
    def list_junction_tables(self) -> Dict[str, List[str]]:
        """List all junction tables in the database"""
        cursor = self.conn.cursor()
        
        try:
            # Find experiment-related junction tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND (
                    table_name LIKE 'experiment_%' 
                    OR table_name LIKE '%_junction%'
                    OR table_name IN (
                        'experiment_decimations',
                        'experiment_segment_sizes',
                        'experiment_amplitude_methods',
                        'experiment_data_types'
                    )
                )
                ORDER BY table_name
            """)
            
            tables = {}
            for row in cursor:
                table_name = row[0]
                if 'decimation' in table_name:
                    category = 'decimations'
                elif 'segment_size' in table_name:
                    category = 'segment_sizes'
                elif 'amplitude' in table_name:
                    category = 'amplitude_methods'
                elif 'data_type' in table_name:
                    category = 'data_types'
                elif 'distance' in table_name:
                    category = 'distances'
                else:
                    category = 'other'
                
                if category not in tables:
                    tables[category] = []
                tables[category].append(table_name)
            
            return tables
            
        finally:
            cursor.close()


def main():
    """Test the PostgreSQL experiment query"""
    try:
        # Connect using MLDP defaults
        query = ExperimentQueryPG(
            host='localhost',
            port=5432,
            database='arc_detection',
            user='kjensen'
        )
        
        # List junction tables
        print("\n📋 Junction Tables:")
        junction_tables = query.list_junction_tables()
        for category, tables in junction_tables.items():
            print(f"\n{category.upper()}:")
            for table in tables:
                print(f"  • {table}")
        
        # List experiments
        print("\n📋 Available Experiments:")
        experiments = query.list_experiments()
        for exp in experiments:
            print(f"  • Experiment {exp['experiment_id']}: {exp['name']}")
        
        # Show details for available experiments
        for exp in experiments[:2]:  # Show first 2
            query.print_experiment_summary(exp['experiment_id'])
        
        query.disconnect()
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()