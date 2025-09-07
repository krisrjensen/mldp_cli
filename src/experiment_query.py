#!/usr/bin/env python3
"""
Filename: experiment_query.py
Author(s): Kristophor Jensen
Date Created: 20250907_130000
Date Revised: 20250907_130000
File version: 0.0.0.1
Description: Query and analyze ML experiments in the MLDP database
"""

import os
import sys
import psycopg2
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from tabulate import tabulate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentQuery:
    """
    Query and analyze ML experiments from MLDP databases
    """
    
    def __init__(self, use_postgresql=False):
        """Initialize query tool with database connections"""
        # PostgreSQL for segment data (optional)
        self.use_postgresql = use_postgresql
        if self.use_postgresql:
            self.pg_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'arc_detection',
                'user': 'postgres'
            }
        
        # SQLite for metadata (primary database)
        self.sqlite_path = '/Volumes/ArcData/V3_database/V3_analysis_database.db'
        
        # Alternative SQLite path if primary not found
        alt_sqlite_path = '/Volumes/ArcData/V3_database/arc_detection.db'
        if not Path(self.sqlite_path).exists() and Path(alt_sqlite_path).exists():
            self.sqlite_path = alt_sqlite_path
            logger.info(f"Using alternative database: {alt_sqlite_path}")
        
        # Verify databases are accessible
        self._verify_connections()
    
    def _verify_connections(self):
        """Verify database connections are accessible"""
        # Check PostgreSQL (optional)
        if self.use_postgresql:
            try:
                conn = psycopg2.connect(**self.pg_config)
                conn.close()
                logger.info("✓ PostgreSQL connection verified")
            except Exception as e:
                logger.warning(f"⚠ PostgreSQL connection failed: {e}")
                logger.info("Continuing with SQLite only")
                self.use_postgresql = False
        
        # Check SQLite (required)
        if not Path(self.sqlite_path).exists():
            logger.error(f"✗ SQLite database not found: {self.sqlite_path}")
            raise FileNotFoundError(f"Database not found: {self.sqlite_path}")
        logger.info(f"✓ SQLite database found: {self.sqlite_path}")
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments with basic information
        
        Returns:
            List of experiment dictionaries
        """
        experiments = []
        
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        try:
            # Get experiments from ml_experiments table
            cursor.execute("""
                SELECT experiment_id, experiment_name, experiment_description,
                       status, created_at, updated_at
                FROM ml_experiments
                ORDER BY experiment_id
            """)
            
            for row in cursor.fetchall():
                exp = {
                    'experiment_id': row[0],
                    'name': row[1] or f'Experiment {row[0]}',
                    'description': row[2] or 'No description',
                    'status': row[3] or 'unknown',
                    'created_at': row[4],
                    'updated_at': row[5]
                }
                
                # Get segment count for this experiment
                segment_count = self._get_experiment_segment_count(cursor, row[0])
                exp['total_segments'] = segment_count
                
                # Get file count
                file_count = self._get_experiment_file_count(cursor, row[0])
                exp['total_files'] = file_count
                
                experiments.append(exp)
            
            return experiments
            
        finally:
            conn.close()
    
    def get_experiment_details(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific experiment
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary with experiment details
        """
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()
        
        try:
            # Get basic experiment info
            cursor.execute("""
                SELECT experiment_id, experiment_name, experiment_description,
                       status, created_at, updated_at
                FROM ml_experiments
                WHERE experiment_id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if not row:
                raise ValueError(f"Experiment {experiment_id} not found")
            
            details = {
                'experiment_id': row[0],
                'name': row[1] or f'Experiment {row[0]}',
                'description': row[2] or 'No description',
                'status': row[3] or 'unknown',
                'created_at': row[4],
                'updated_at': row[5]
            }
            
            # Get segment statistics
            details['segments'] = self._get_segment_statistics(cursor, experiment_id)
            
            # Get file statistics
            details['files'] = self._get_file_statistics(cursor, experiment_id)
            
            # Get segment pair statistics
            details['segment_pairs'] = self._get_segment_pair_statistics(cursor, experiment_id)
            
            # Get distance calculation status
            details['distances'] = self._get_distance_statistics(cursor, experiment_id)
            
            # Check for experiment-specific tables
            details['tables'] = self._get_experiment_tables(cursor, experiment_id)
            
            return details
            
        finally:
            conn.close()
    
    def get_experiment_statistics(self, experiment_id: int) -> Dict[str, Any]:
        """
        Get comprehensive statistics for an experiment
        
        Args:
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary with experiment statistics
        """
        stats = self.get_experiment_details(experiment_id)
        
        # Add additional analysis
        if stats['segments']['by_label']:
            # Calculate class balance
            label_counts = [item['count'] for item in stats['segments']['by_label']]
            if label_counts:
                max_count = max(label_counts)
                min_count = min(label_counts)
                stats['class_balance'] = {
                    'max_instances': max_count,
                    'min_instances': min_count,
                    'balance_ratio': min_count / max_count if max_count > 0 else 0,
                    'total_classes': len(label_counts)
                }
        
        # Add segment size distribution
        if stats['segments']['by_size']:
            stats['segment_sizes'] = [item['size'] for item in stats['segments']['by_size']]
        
        return stats
    
    def _get_experiment_segment_count(self, cursor, experiment_id: int) -> int:
        """Get total segment count for experiment"""
        # Check for experiment-specific segment training data table
        table_name = f"experiment_{experiment_id:03d}_segment_training_data"
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        
        if cursor.fetchone():
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        
        # Fallback: count from data_segments with configuration_id
        cursor.execute("""
            SELECT COUNT(*) FROM data_segments 
            WHERE configuration_id = ?
        """, (experiment_id,))
        return cursor.fetchone()[0]
    
    def _get_experiment_file_count(self, cursor, experiment_id: int) -> int:
        """Get total file count for experiment"""
        # Check for experiment-specific files training data table
        table_name = f"experiment_{experiment_id:03d}_files_training_data"
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,))
        
        if cursor.fetchone():
            cursor.execute(f"SELECT COUNT(DISTINCT experiment_file_id) FROM {table_name}")
            return cursor.fetchone()[0]
        
        return 0
    
    def _get_segment_statistics(self, cursor, experiment_id: int) -> Dict[str, Any]:
        """Get segment statistics for experiment"""
        stats = {
            'total': 0,
            'by_label': [],
            'by_size': [],
            'by_position': []
        }
        
        # Check for experiment-specific segment table
        segment_table = f"experiment_{experiment_id:03d}_segment_training_data"
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (segment_table,))
        
        if cursor.fetchone():
            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM {segment_table}")
            stats['total'] = cursor.fetchone()[0]
            
            # Get distribution by label
            cursor.execute(f"""
                SELECT sl.label_name, COUNT(*) as count
                FROM {segment_table} est
                JOIN data_segments ds ON est.segment_id = ds.segment_id
                JOIN segment_labels sl ON ds.segment_label_id = sl.label_id
                GROUP BY sl.label_name
                ORDER BY count DESC
            """)
            
            for row in cursor.fetchall():
                stats['by_label'].append({
                    'label': row[0],
                    'count': row[1]
                })
            
            # Get distribution by segment size
            cursor.execute(f"""
                SELECT ds.segment_length, COUNT(*) as count
                FROM {segment_table} est
                JOIN data_segments ds ON est.segment_id = ds.segment_id
                GROUP BY ds.segment_length
                ORDER BY ds.segment_length
            """)
            
            for row in cursor.fetchall():
                stats['by_size'].append({
                    'size': row[0],
                    'count': row[1]
                })
            
            # Get distribution by position type (if available)
            cursor.execute(f"""
                SELECT 
                    CASE 
                        WHEN ds.segment_id_code LIKE 'L%' THEN 'LNNN'
                        WHEN ds.segment_id_code LIKE 'R%' THEN 'RNNN'
                        WHEN ds.segment_id_code LIKE 'C%' THEN 'CNNN'
                        ELSE 'OTHER'
                    END as position_type,
                    COUNT(*) as count
                FROM {segment_table} est
                JOIN data_segments ds ON est.segment_id = ds.segment_id
                GROUP BY position_type
                ORDER BY count DESC
            """)
            
            for row in cursor.fetchall():
                stats['by_position'].append({
                    'position': row[0],
                    'count': row[1]
                })
        
        return stats
    
    def _get_file_statistics(self, cursor, experiment_id: int) -> Dict[str, Any]:
        """Get file statistics for experiment"""
        stats = {
            'total': 0,
            'by_label': []
        }
        
        # Check for experiment-specific files table
        files_table = f"experiment_{experiment_id:03d}_files_training_data"
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (files_table,))
        
        if cursor.fetchone():
            # Get total count
            cursor.execute(f"SELECT COUNT(DISTINCT experiment_file_id) FROM {files_table}")
            stats['total'] = cursor.fetchone()[0]
            
            # Get distribution by label
            cursor.execute(f"""
                SELECT el.experiment_label, COUNT(DISTINCT eft.experiment_file_id) as count
                FROM {files_table} eft
                JOIN files f ON eft.experiment_file_id = f.file_id
                JOIN files_Y fy ON f.file_id = fy.file_id
                JOIN experiment_labels el ON fy.label_id = el.label_id
                GROUP BY el.experiment_label
                ORDER BY count DESC
            """)
            
            for row in cursor.fetchall():
                stats['by_label'].append({
                    'label': row[0],
                    'count': row[1]
                })
        
        return stats
    
    def _get_segment_pair_statistics(self, cursor, experiment_id: int) -> Dict[str, Any]:
        """Get segment pair statistics for experiment"""
        stats = {
            'total': 0,
            'same_class': 0,
            'different_class': 0
        }
        
        # Check for experiment-specific segment pairs table
        pairs_table = f"experiment_{experiment_id:03d}_segment_pairs"
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (pairs_table,))
        
        if cursor.fetchone():
            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM {pairs_table}")
            stats['total'] = cursor.fetchone()[0]
            
            # Get same-class vs different-class counts
            cursor.execute(f"""
                SELECT 
                    CASE 
                        WHEN ds_a.segment_label_id = ds_b.segment_label_id THEN 'same_class'
                        ELSE 'different_class'
                    END as pair_type,
                    COUNT(*) as count
                FROM {pairs_table} esp
                JOIN data_segments ds_a ON esp.segment_a_id = ds_a.segment_id
                JOIN data_segments ds_b ON esp.segment_b_id = ds_b.segment_id
                GROUP BY pair_type
            """)
            
            for row in cursor.fetchall():
                if row[0] == 'same_class':
                    stats['same_class'] = row[1]
                else:
                    stats['different_class'] = row[1]
        
        return stats
    
    def _get_distance_statistics(self, cursor, experiment_id: int) -> Dict[str, Any]:
        """Get distance calculation statistics for experiment"""
        stats = {}
        
        # Check for distance tables
        for metric in ['L1', 'L2', 'cosine']:
            table_name = f"experiment_{experiment_id:03d}_distance_{metric}"
            
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name=?
            """, (table_name,))
            
            if cursor.fetchone():
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                stats[metric] = count
        
        return stats
    
    def _get_experiment_tables(self, cursor, experiment_id: int) -> List[str]:
        """Get list of experiment-specific tables"""
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name LIKE ?
            ORDER BY name
        """, (f'experiment_{experiment_id:03d}_%',))
        
        return [row[0] for row in cursor.fetchall()]
    
    def export_to_json(self, experiment_id: int, output_path: str = None) -> str:
        """
        Export experiment details to JSON file
        
        Args:
            experiment_id: ID of the experiment
            output_path: Optional output path (defaults to experiment_{id}.json)
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            output_path = f"experiment_{experiment_id:03d}.json"
        
        details = self.get_experiment_statistics(experiment_id)
        
        with open(output_path, 'w') as f:
            json.dump(details, f, indent=2, default=str)
        
        logger.info(f"Exported experiment {experiment_id} to {output_path}")
        return output_path
    
    def export_to_csv(self, experiment_id: int, output_dir: str = None) -> List[str]:
        """
        Export experiment data to CSV files
        
        Args:
            experiment_id: ID of the experiment
            output_dir: Optional output directory
            
        Returns:
            List of exported file paths
        """
        if output_dir is None:
            output_dir = f"experiment_{experiment_id:03d}_export"
        
        Path(output_dir).mkdir(exist_ok=True)
        exported_files = []
        
        details = self.get_experiment_statistics(experiment_id)
        
        # Export segment distribution
        if details['segments']['by_label']:
            df = pd.DataFrame(details['segments']['by_label'])
            path = Path(output_dir) / 'segment_distribution.csv'
            df.to_csv(path, index=False)
            exported_files.append(str(path))
        
        # Export file distribution
        if details['files']['by_label']:
            df = pd.DataFrame(details['files']['by_label'])
            path = Path(output_dir) / 'file_distribution.csv'
            df.to_csv(path, index=False)
            exported_files.append(str(path))
        
        logger.info(f"Exported {len(exported_files)} CSV files to {output_dir}")
        return exported_files
    
    def print_experiment_summary(self, experiment_id: int):
        """Print formatted summary of experiment"""
        details = self.get_experiment_statistics(experiment_id)
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {experiment_id}: {details['name']}")
        print(f"{'='*60}")
        print(f"Status: {details['status']}")
        print(f"Created: {details['created_at']}")
        print(f"Updated: {details['updated_at']}")
        
        print(f"\n📊 SEGMENTS")
        print(f"Total: {details['segments']['total']:,}")
        if details['segments']['by_label']:
            print("\nBy Label:")
            data = [(item['label'], f"{item['count']:,}") 
                   for item in details['segments']['by_label'][:10]]
            print(tabulate(data, headers=['Label', 'Count'], tablefmt='simple'))
        
        if details['segments']['by_size']:
            print("\nBy Size:")
            data = [(f"{item['size']:,}", f"{item['count']:,}") 
                   for item in details['segments']['by_size']]
            print(tabulate(data, headers=['Size', 'Count'], tablefmt='simple'))
        
        if details['segments']['by_position']:
            print("\nBy Position:")
            data = [(item['position'], f"{item['count']:,}") 
                   for item in details['segments']['by_position']]
            print(tabulate(data, headers=['Position', 'Count'], tablefmt='simple'))
        
        print(f"\n📁 FILES")
        print(f"Total: {details['files']['total']:,}")
        if details['files']['by_label']:
            print("\nBy Label:")
            data = [(item['label'], f"{item['count']:,}") 
                   for item in details['files']['by_label']]
            print(tabulate(data, headers=['Label', 'Count'], tablefmt='simple'))
        
        print(f"\n🔗 SEGMENT PAIRS")
        print(f"Total: {details['segment_pairs']['total']:,}")
        print(f"Same Class: {details['segment_pairs']['same_class']:,}")
        print(f"Different Class: {details['segment_pairs']['different_class']:,}")
        
        if details['distances']:
            print(f"\n📏 DISTANCES")
            for metric, count in details['distances'].items():
                print(f"{metric}: {count:,}")
        
        if 'class_balance' in details:
            print(f"\n⚖️  CLASS BALANCE")
            print(f"Classes: {details['class_balance']['total_classes']}")
            print(f"Max instances: {details['class_balance']['max_instances']:,}")
            print(f"Min instances: {details['class_balance']['min_instances']:,}")
            print(f"Balance ratio: {details['class_balance']['balance_ratio']:.2%}")
        
        print(f"\n{'='*60}\n")


def main():
    """Main entry point for testing"""
    query = ExperimentQuery()
    
    # List all experiments
    print("\n📋 Available Experiments:")
    experiments = query.list_experiments()
    for exp in experiments:
        print(f"  • Experiment {exp['experiment_id']}: {exp['name']}")
        print(f"    Segments: {exp['total_segments']:,}, Files: {exp['total_files']:,}")
    
    # Show details for specific experiments
    for exp_id in [17, 18]:
        try:
            query.print_experiment_summary(exp_id)
        except Exception as e:
            logger.error(f"Error getting details for experiment {exp_id}: {e}")


if __name__ == "__main__":
    main()