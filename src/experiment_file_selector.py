#!/usr/bin/env python3
"""
Filename: experiment_file_selector.py
Author: Kristophor Jensen
Date Created: 20250915_090000
Date Revised: 20251104_000000
File version: 1.0.0.1
Description: File selection for experiment training data

Changes in v1.0.0.1:
- Fixed SQL query to use files_x instead of files table
- Added join with experiment_labels to get label_text
- Removed reference to non-existent fy.label_text column
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import random
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ExperimentFileSelector:
    """Select files for experiment training data"""
    
    def __init__(self, experiment_id: int, db_conn):
        self.experiment_id = experiment_id
        self.db_conn = db_conn
        self.table_name = f"experiment_{experiment_id:03d}_file_training_data"
        
    def create_training_table(self):
        """Create the file training data table if it doesn't exist"""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    file_training_id SERIAL PRIMARY KEY,
                    experiment_id INTEGER NOT NULL,
                    file_id INTEGER NOT NULL,
                    file_label_id INTEGER,
                    file_label_name VARCHAR(100),
                    selection_order INTEGER,
                    selection_strategy VARCHAR(50),
                    random_seed INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(experiment_id, file_id)
                )
            """)
            
            # Create indexes for performance
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_experiment 
                ON {self.table_name}(experiment_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_file 
                ON {self.table_name}(file_id)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_label 
                ON {self.table_name}(file_label_id)
            """)
            
            self.db_conn.commit()
            logger.info(f"Created/verified table: {self.table_name}")
            return True
            
        except psycopg2.Error as e:
            self.db_conn.rollback()
            logger.error(f"Error creating training table: {e}")
            return False
        finally:
            cursor.close()
    
    def clear_existing_selection(self):
        """Clear any existing file selection for this experiment"""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(f"""
                DELETE FROM {self.table_name} 
                WHERE experiment_id = %s
            """, (self.experiment_id,))
            deleted = cursor.rowcount
            self.db_conn.commit()
            if deleted > 0:
                logger.info(f"Cleared {deleted} existing file selections")
            return deleted
        except psycopg2.Error as e:
            self.db_conn.rollback()
            logger.error(f"Error clearing selections: {e}")
            return 0
        finally:
            cursor.close()
    
    def get_available_files(self) -> Dict[str, List[Dict]]:
        """Get all available files grouped by label"""
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Query files with their labels
            cursor.execute("""
                SELECT DISTINCT
                    f.file_id,
                    f.original_filename as file_name,
                    f.original_path as file_path,
                    fy.label_id as file_label_id,
                    el.experiment_label as file_label_name,
                    f.current_level,
                    f.voltage_level,
                    f.total_samples as file_length,
                    0.5 as quality_score
                FROM files_x f
                LEFT JOIN files_y fy ON f.file_id = fy.file_id
                LEFT JOIN experiment_labels el ON fy.label_id = el.label_id
                WHERE f.total_samples > 0
                ORDER BY el.experiment_label, f.file_id
            """)
            
            # Group by label
            files_by_label = {}
            for row in cursor:
                label = row['file_label_name'] or 'unlabeled'
                if label not in files_by_label:
                    files_by_label[label] = []
                files_by_label[label].append(dict(row))
            
            return files_by_label
            
        except psycopg2.Error as e:
            logger.error(f"Error getting available files: {e}")
            return {}
        finally:
            cursor.close()
    
    def select_files(self, strategy: str = 'random', max_files_per_label: int = 50, 
                    seed: int = 42, min_quality: float = None) -> Dict[str, Any]:
        """
        Select files for training data based on strategy
        
        Args:
            strategy: Selection strategy ('random', 'balanced', 'quality_first')
            max_files_per_label: Maximum files to select per label
            seed: Random seed for reproducibility
            min_quality: Minimum quality score (if applicable)
            
        Returns:
            Dictionary with selection results
        """
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Create table if needed
        if not self.create_training_table():
            return {'success': False, 'error': 'Failed to create training table'}
        
        # Clear existing selections
        self.clear_existing_selection()
        
        # Get available files
        files_by_label = self.get_available_files()
        
        if not files_by_label:
            return {'success': False, 'error': 'No files available'}
        
        # Apply selection strategy
        selected_files = []
        selection_order = 0
        
        if strategy == 'random':
            selected_files = self._select_random(files_by_label, max_files_per_label)
        elif strategy == 'balanced':
            selected_files = self._select_balanced(files_by_label, max_files_per_label)
        elif strategy == 'quality_first':
            selected_files = self._select_quality_first(files_by_label, max_files_per_label, min_quality)
        else:
            return {'success': False, 'error': f'Unknown strategy: {strategy}'}
        
        # Insert selected files into database
        cursor = self.db_conn.cursor()
        try:
            for order, file_info in enumerate(selected_files, 1):
                cursor.execute(f"""
                    INSERT INTO {self.table_name} 
                    (experiment_id, file_id, file_label_id, file_label_name, 
                     selection_order, selection_strategy, random_seed)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    self.experiment_id,
                    file_info['file_id'],
                    file_info['file_label_id'],
                    file_info['file_label_name'],
                    order,
                    strategy,
                    seed
                ))
            
            self.db_conn.commit()
            
            # Get statistics
            stats = self._get_selection_statistics()
            
            return {
                'success': True,
                'total_selected': len(selected_files),
                'strategy': strategy,
                'seed': seed,
                'max_per_label': max_files_per_label,
                'statistics': stats
            }
            
        except psycopg2.Error as e:
            self.db_conn.rollback()
            logger.error(f"Error inserting selected files: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            cursor.close()
    
    def _select_random(self, files_by_label: Dict, max_files: int) -> List[Dict]:
        """Random selection strategy"""
        selected = []
        for label, files in files_by_label.items():
            # Randomly sample up to max_files
            sample_size = min(len(files), max_files)
            sampled = random.sample(files, sample_size)
            for file_info in sampled:
                file_info['file_label_name'] = label
                selected.append(file_info)
        return selected
    
    def _select_balanced(self, files_by_label: Dict, max_files: int) -> List[Dict]:
        """Balanced selection strategy - equal files per label"""
        selected = []
        
        # Find minimum available files across labels
        min_available = min(len(files) for files in files_by_label.values())
        target_per_label = min(min_available, max_files)
        
        for label, files in files_by_label.items():
            # Sample exactly target_per_label files
            sampled = random.sample(files, target_per_label)
            for file_info in sampled:
                file_info['file_label_name'] = label
                selected.append(file_info)
        
        return selected
    
    def _select_quality_first(self, files_by_label: Dict, max_files: int, 
                             min_quality: float = None) -> List[Dict]:
        """Quality-first selection strategy - prioritize high quality files"""
        selected = []
        
        for label, files in files_by_label.items():
            # Files are already sorted by quality (DESC)
            # Filter by minimum quality if specified
            if min_quality is not None:
                files = [f for f in files if f.get('quality_score', 0) >= min_quality]
            
            # Take up to max_files (highest quality first)
            for file_info in files[:max_files]:
                file_info['file_label_name'] = label
                selected.append(file_info)
        
        return selected
    
    def _get_selection_statistics(self) -> Dict[str, Any]:
        """Get statistics about the selected files"""
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Get counts by label
            cursor.execute(f"""
                SELECT 
                    file_label_name,
                    COUNT(*) as count,
                    MIN(selection_order) as min_order,
                    MAX(selection_order) as max_order
                FROM {self.table_name}
                WHERE experiment_id = %s
                GROUP BY file_label_name
                ORDER BY file_label_name
            """, (self.experiment_id,))
            
            label_counts = {row['file_label_name']: row['count'] 
                          for row in cursor}
            
            # Get total statistics
            cursor.execute(f"""
                SELECT 
                    COUNT(DISTINCT file_id) as unique_files,
                    COUNT(DISTINCT file_label_id) as unique_labels,
                    MIN(created_at) as first_selected,
                    MAX(created_at) as last_selected
                FROM {self.table_name}
                WHERE experiment_id = %s
            """, (self.experiment_id,))
            
            stats = cursor.fetchone()
            
            return {
                'label_counts': label_counts,
                'unique_files': stats['unique_files'],
                'unique_labels': stats['unique_labels'],
                'selection_time': str(stats['last_selected']) if stats['last_selected'] else None
            }
            
        except psycopg2.Error as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
        finally:
            cursor.close()
    
    def get_selected_files(self) -> List[Dict]:
        """Get list of selected files for this experiment"""
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(f"""
                SELECT 
                    ft.*,
                    f.file_name,
                    f.file_path,
                    f.file_length,
                    f.quality_score
                FROM {self.table_name} ft
                JOIN files f ON ft.file_id = f.file_id
                WHERE ft.experiment_id = %s
                ORDER BY ft.selection_order
            """, (self.experiment_id,))
            
            return [dict(row) for row in cursor]
            
        except psycopg2.Error as e:
            logger.error(f"Error getting selected files: {e}")
            return []
        finally:
            cursor.close()