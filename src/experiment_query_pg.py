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
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments from ml_experiments table
        
        Returns:
            List of experiment dictionaries with basic info
        """
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
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        
        try:
            # Get basic experiment info
            cursor.execute("""
                SELECT * FROM ml_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            details = dict(row)
            
            # Get decimation factors from junction table
            details['decimations'] = self._get_decimations(cursor, experiment_id)
            
            # Get segment sizes from junction table
            details['segment_sizes'] = self._get_segment_sizes(cursor, experiment_id)
            
            # Get amplitude normalization methods
            details['amplitude_methods'] = self._get_amplitude_methods(cursor, experiment_id)
            
            # Get data types
            details['data_types'] = self._get_data_types(cursor, experiment_id)
            
            # Get segment statistics
            details['segments'] = self._get_segment_statistics(cursor, experiment_id)
            
            # Get distance table info
            details['distances'] = self._get_distance_info(cursor, experiment_id)
            
            return details
            
        finally:
            cursor.close()
    
    def _get_decimations(self, cursor, experiment_id: int) -> List[Dict]:
        """Get decimation factors for experiment from junction table"""
        # Check if junction table exists and has data
        cursor.execute("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'ml_experiment_decimation_junction'
            )
        """)
        
        if not cursor.fetchone()[0]:
            return []
        
        # First check what columns exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'ml_experiment_decimation_junction'
            ORDER BY ordinal_position
        """)
        
        columns = [row[0] for row in cursor.fetchall()]
        
        # If table has columns, query it
        if columns:
            # Use * to get all columns since we don't know the exact names
            cursor.execute("""
                SELECT * 
                FROM ml_experiment_decimation_junction
                WHERE experiment_id = %s
            """, (experiment_id,))
        
        decimations = []
        for row in cursor.fetchall():
            decimations.append({
                'decimation_factor': row[0],
                'experiment_id': row[1]
            })
        
        return decimations
    
    def _get_segment_sizes(self, cursor, experiment_id: int) -> List[Dict]:
        """Get segment sizes from junction table"""
        # For now, return empty list as junction tables may not exist
        # TODO: Check actual junction table names once database is properly documented
        return []
    
    def _get_amplitude_methods(self, cursor, experiment_id: int) -> List[Dict]:
        """Get amplitude normalization methods from junction table"""
        # For now, return empty list as junction tables may not exist
        # TODO: Check actual junction table names once database is properly documented
        return []
    
    def _get_data_types(self, cursor, experiment_id: int) -> List[Dict]:
        """Get data types from junction table"""  
        # For now, return empty list as junction tables may not exist
        # TODO: Check actual junction table names once database is properly documented
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
    
    def get_experiment_configuration(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get complete configuration including all junction table relationships
        
        Returns configuration suitable for experiment generation
        """
        details = self.get_experiment_details(experiment_id)
        
        config = {
            'experiment_id': experiment_id,
            'name': details['experiment_name'],
            'decimations': [d['decimation_factor'] for d in details['decimations']],
            'segment_sizes': [s['segment_size'] for s in details['segment_sizes']],
            'data_types': [dt['data_type_name'] for dt in details['data_types']],
            'amplitude_methods': [am['method_name'] for am in details['amplitude_methods']],
            'total_segments': details['segments']['total'],
            'segment_distribution': details['segments']['by_size']
        }
        
        return config
    
    def print_experiment_summary(self, experiment_id: int):
        """Print formatted experiment summary"""
        details = self.get_experiment_details(experiment_id)
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENT {experiment_id}: {details.get('experiment_name', f'Experiment {experiment_id}')}")
        print(f"{'='*70}")
        print(f"Status: {details.get('status', 'unknown')}")
        print(f"Type: {details.get('experiment_type', 'N/A')}")
        print(f"Version: {details.get('experiment_version', 'N/A')}")
        print(f"Description: {details.get('description', 'No description')}")
        
        print(f"\n📊 CONFIGURATION")
        print(f"Decimations: {[d['decimation_factor'] for d in details['decimations']]}")
        print(f"Segment Sizes: {[s['segment_size'] for s in details['segment_sizes']]}")
        print(f"Data Types: {[dt['data_type_name'] for dt in details['data_types']]}")
        print(f"Amplitude Methods: {[am['method_name'] for am in details['amplitude_methods']]}")
        
        print(f"\n📈 SEGMENTS")
        print(f"Total: {details['segments']['total']:,}")
        if details['segments']['by_size']:
            print("\nBy Size:")
            data = [(f"{item['size']:,}", f"{item['count']:,}") 
                   for item in details['segments']['by_size']]
            print(tabulate(data, headers=['Size', 'Count'], tablefmt='simple'))
        
        if details['distances']:
            print(f"\n📏 DISTANCES")
            for metric, count in details['distances'].items():
                print(f"{metric}: {count:,}")
        
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