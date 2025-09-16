#!/usr/bin/env python3
"""
Filename: experiment_segment_pair_generator.py
Author: Kristophor Jensen
Date Created: 20250916_090000
Date Revised: 20250916_090000
File version: 1.0.0.0
Description: Generate segment pairs for distance calculations
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Dict, List, Any, Tuple, Optional
import itertools
import random

logger = logging.getLogger(__name__)

class ExperimentSegmentPairGenerator:
    """Generate segment pairs for distance calculations"""
    
    def __init__(self, experiment_id: int, db_conn):
        self.experiment_id = experiment_id
        self.db_conn = db_conn
        self.segment_table = f"experiment_{experiment_id:03d}_segment_training_data"
        self.pairs_table = f"experiment_{experiment_id:03d}_segment_pairs"
        
    def create_pairs_table(self):
        """Create the segment pairs table if it doesn't exist"""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.pairs_table} (
                    pair_id SERIAL PRIMARY KEY,
                    experiment_id INTEGER NOT NULL,
                    segment_id_1 INTEGER NOT NULL,
                    segment_id_2 INTEGER NOT NULL,
                    file_id_1 INTEGER NOT NULL,
                    file_id_2 INTEGER NOT NULL,
                    label_id_1 INTEGER,
                    label_id_2 INTEGER,
                    label_name_1 VARCHAR(100),
                    label_name_2 VARCHAR(100),
                    is_same_label BOOLEAN,
                    pair_type VARCHAR(50),
                    pair_weight FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unique_pair UNIQUE(experiment_id, segment_id_1, segment_id_2)
                )
            """)
            
            # Create indexes for performance
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pairs_table}_segment1 
                ON {self.pairs_table}(segment_id_1)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pairs_table}_segment2 
                ON {self.pairs_table}(segment_id_2)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pairs_table}_same_label 
                ON {self.pairs_table}(is_same_label)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pairs_table}_type 
                ON {self.pairs_table}(pair_type)
            """)
            
            self.db_conn.commit()
            logger.info(f"Created/verified table: {self.pairs_table}")
            return True
            
        except psycopg2.Error as e:
            self.db_conn.rollback()
            logger.error(f"Error creating pairs table: {e}")
            return False
        finally:
            cursor.close()
    
    def clear_existing_pairs(self):
        """Clear any existing segment pairs for this experiment"""
        cursor = self.db_conn.cursor()
        try:
            # Check if table exists first
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (self.pairs_table,))
            
            if cursor.fetchone()[0]:
                cursor.execute(f"""
                    DELETE FROM {self.pairs_table} 
                    WHERE experiment_id = %s
                """, (self.experiment_id,))
                deleted = cursor.rowcount
                self.db_conn.commit()
                if deleted > 0:
                    logger.info(f"Cleared {deleted} existing segment pairs")
                return deleted
            return 0
        except psycopg2.Error as e:
            self.db_conn.rollback()
            logger.error(f"Error clearing pairs: {e}")
            return 0
        finally:
            cursor.close()
    
    def get_selected_segments(self) -> List[Dict]:
        """Get all selected segments for this experiment"""
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute(f"""
                SELECT 
                    st.segment_id,
                    st.file_id,
                    st.segment_label_id,
                    st.segment_label_name,
                    st.position_label,
                    st.segment_index,
                    st.selection_order
                FROM {self.segment_table} st
                WHERE st.experiment_id = %s
                ORDER BY st.selection_order
            """, (self.experiment_id,))
            
            return [dict(row) for row in cursor]
            
        except psycopg2.Error as e:
            logger.error(f"Error getting segments: {e}")
            return []
        finally:
            cursor.close()
    
    def generate_pairs(self, 
                      strategy: str = 'all_combinations',
                      max_pairs_per_segment: int = None,
                      same_label_ratio: float = 0.5,
                      seed: int = 42) -> Dict[str, Any]:
        """
        Generate segment pairs based on strategy
        
        Args:
            strategy: Pairing strategy ('all_combinations', 'balanced', 'random_sample')
            max_pairs_per_segment: Maximum pairs per segment (for sampling strategies)
            same_label_ratio: Ratio of same-label pairs (for balanced strategy)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with generation results
        """
        random.seed(seed)
        
        # Create table if needed
        if not self.create_pairs_table():
            return {'success': False, 'error': 'Failed to create pairs table'}
        
        # Clear existing pairs
        self.clear_existing_pairs()
        
        # Get selected segments
        segments = self.get_selected_segments()
        
        if len(segments) < 2:
            return {'success': False, 'error': f'Need at least 2 segments, found {len(segments)}'}
        
        # Generate pairs based on strategy
        if strategy == 'all_combinations':
            pairs = self._generate_all_combinations(segments)
        elif strategy == 'balanced':
            pairs = self._generate_balanced_pairs(segments, max_pairs_per_segment, same_label_ratio)
        elif strategy == 'random_sample':
            pairs = self._generate_random_sample(segments, max_pairs_per_segment)
        else:
            return {'success': False, 'error': f'Unknown strategy: {strategy}'}
        
        # Insert pairs into database
        inserted = self._insert_pairs(pairs)
        
        # Get statistics
        stats = self._get_pair_statistics()
        
        return {
            'success': True,
            'total_segments': len(segments),
            'total_pairs': inserted,
            'strategy': strategy,
            'seed': seed,
            'statistics': stats
        }
    
    def _generate_all_combinations(self, segments: List[Dict]) -> List[Tuple]:
        """Generate all possible segment pairs (N choose 2)"""
        pairs = []
        
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                seg1 = segments[i]
                seg2 = segments[j]
                
                is_same_label = seg1['segment_label_id'] == seg2['segment_label_id']
                pair_type = 'within_label' if is_same_label else 'across_label'
                
                pairs.append((
                    seg1['segment_id'], seg2['segment_id'],
                    seg1['file_id'], seg2['file_id'],
                    seg1['segment_label_id'], seg2['segment_label_id'],
                    seg1['segment_label_name'], seg2['segment_label_name'],
                    is_same_label, pair_type
                ))
        
        logger.info(f"Generated {len(pairs)} pairs from {len(segments)} segments")
        return pairs
    
    def _generate_balanced_pairs(self, segments: List[Dict], 
                                 max_pairs_per_segment: int,
                                 same_label_ratio: float) -> List[Tuple]:
        """Generate balanced pairs with controlled same/different label ratio"""
        pairs = []
        segments_by_label = {}
        
        # Group segments by label
        for seg in segments:
            label = seg['segment_label_id']
            if label not in segments_by_label:
                segments_by_label[label] = []
            segments_by_label[label].append(seg)
        
        # For each segment, generate balanced pairs
        for seg1 in segments:
            same_label_segments = [s for s in segments_by_label[seg1['segment_label_id']] 
                                 if s['segment_id'] != seg1['segment_id']]
            diff_label_segments = [s for s in segments 
                                 if s['segment_label_id'] != seg1['segment_label_id']]
            
            # Calculate how many of each type
            if max_pairs_per_segment:
                n_same = min(int(max_pairs_per_segment * same_label_ratio), 
                           len(same_label_segments))
                n_diff = min(max_pairs_per_segment - n_same, 
                           len(diff_label_segments))
            else:
                n_same = len(same_label_segments)
                n_diff = len(diff_label_segments)
            
            # Sample pairs
            if same_label_segments and n_same > 0:
                for seg2 in random.sample(same_label_segments, n_same):
                    if seg1['segment_id'] < seg2['segment_id']:  # Avoid duplicates
                        pairs.append((
                            seg1['segment_id'], seg2['segment_id'],
                            seg1['file_id'], seg2['file_id'],
                            seg1['segment_label_id'], seg2['segment_label_id'],
                            seg1['segment_label_name'], seg2['segment_label_name'],
                            True, 'within_label'
                        ))
            
            if diff_label_segments and n_diff > 0:
                for seg2 in random.sample(diff_label_segments, n_diff):
                    if seg1['segment_id'] < seg2['segment_id']:  # Avoid duplicates
                        pairs.append((
                            seg1['segment_id'], seg2['segment_id'],
                            seg1['file_id'], seg2['file_id'],
                            seg1['segment_label_id'], seg2['segment_label_id'],
                            seg1['segment_label_name'], seg2['segment_label_name'],
                            False, 'across_label'
                        ))
        
        # Remove duplicates (in case any were created)
        pairs = list(set(pairs))
        
        logger.info(f"Generated {len(pairs)} balanced pairs from {len(segments)} segments")
        return pairs
    
    def _generate_random_sample(self, segments: List[Dict], 
                               max_pairs_per_segment: int) -> List[Tuple]:
        """Generate random sample of segment pairs"""
        all_pairs = self._generate_all_combinations(segments)
        
        if max_pairs_per_segment:
            max_total = max_pairs_per_segment * len(segments) // 2
            if len(all_pairs) > max_total:
                pairs = random.sample(all_pairs, max_total)
                logger.info(f"Sampled {len(pairs)} pairs from {len(all_pairs)} possible")
                return pairs
        
        return all_pairs
    
    def _insert_pairs(self, pairs: List[Tuple]) -> int:
        """Insert pairs into database"""
        cursor = self.db_conn.cursor()
        try:
            for pair in pairs:
                cursor.execute(f"""
                    INSERT INTO {self.pairs_table}
                    (experiment_id, segment_id_1, segment_id_2, 
                     file_id_1, file_id_2,
                     label_id_1, label_id_2,
                     label_name_1, label_name_2,
                     is_same_label, pair_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (experiment_id, segment_id_1, segment_id_2) DO NOTHING
                """, (
                    self.experiment_id,
                    pair[0], pair[1],  # segment_ids
                    pair[2], pair[3],  # file_ids
                    pair[4], pair[5],  # label_ids
                    pair[6], pair[7],  # label_names
                    pair[8], pair[9]   # is_same_label, pair_type
                ))
            
            inserted = cursor.rowcount
            self.db_conn.commit()
            logger.info(f"Inserted {inserted} segment pairs")
            return len(pairs)  # Return total attempted
            
        except psycopg2.Error as e:
            self.db_conn.rollback()
            logger.error(f"Error inserting pairs: {e}")
            return 0
        finally:
            cursor.close()
    
    def _get_pair_statistics(self) -> Dict[str, Any]:
        """Get statistics about the generated pairs"""
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Get counts by pair type
            cursor.execute(f"""
                SELECT 
                    pair_type,
                    COUNT(*) as count,
                    COUNT(DISTINCT segment_id_1) + COUNT(DISTINCT segment_id_2) as unique_segments
                FROM {self.pairs_table}
                WHERE experiment_id = %s
                GROUP BY pair_type
            """, (self.experiment_id,))
            
            type_counts = {row['pair_type']: row['count'] for row in cursor}
            
            # Get label statistics
            cursor.execute(f"""
                SELECT 
                    is_same_label,
                    COUNT(*) as count
                FROM {self.pairs_table}
                WHERE experiment_id = %s
                GROUP BY is_same_label
            """, (self.experiment_id,))
            
            label_stats = {row['is_same_label']: row['count'] for row in cursor}
            
            # Get overall statistics
            cursor.execute(f"""
                SELECT 
                    COUNT(*) as total_pairs,
                    COUNT(DISTINCT segment_id_1) + COUNT(DISTINCT segment_id_2) as unique_segments,
                    COUNT(DISTINCT file_id_1) + COUNT(DISTINCT file_id_2) as unique_files
                FROM {self.pairs_table}
                WHERE experiment_id = %s
            """, (self.experiment_id,))
            
            overall = cursor.fetchone()
            
            return {
                'total_pairs': overall['total_pairs'],
                'unique_segments': overall['unique_segments'] // 2,  # Approximate
                'unique_files': overall['unique_files'] // 2,  # Approximate
                'type_distribution': type_counts,
                'same_label_pairs': label_stats.get(True, 0),
                'diff_label_pairs': label_stats.get(False, 0)
            }
            
        except psycopg2.Error as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
        finally:
            cursor.close()
    
    def get_generated_pairs(self, limit: int = None) -> List[Dict]:
        """Get list of generated pairs"""
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            query = f"""
                SELECT *
                FROM {self.pairs_table}
                WHERE experiment_id = %s
                ORDER BY pair_id
            """
            if limit:
                query += f" LIMIT {limit}"
                
            cursor.execute(query, (self.experiment_id,))
            return [dict(row) for row in cursor]
            
        except psycopg2.Error as e:
            logger.error(f"Error getting pairs: {e}")
            return []
        finally:
            cursor.close()