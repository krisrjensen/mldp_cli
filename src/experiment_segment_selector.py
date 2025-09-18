"""
Filename: experiment_segment_selector.py
Author(s): Kristophor Jensen
Date Created: 20250913_174500
Date Revised: 20250913_174500
File version: 1.0.0.0
Description: Segment selection for ML experiments with position balancing
"""

import logging
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import json

class ExperimentSegmentSelector:
    """
    Handles segment selection for ML experiments with:
    - File selection per label (up to 50 files per label)
    - Position-balanced segment selection
    - Segment pair generation with constraints
    """
    
    def __init__(self, experiment_id: int, db_config: Dict[str, Any]):
        """
        Initialize the segment selector.
        
        Args:
            experiment_id: ID of the experiment
            db_config: Database connection parameters
        """
        self.experiment_id = experiment_id
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)
        
        # Selection configuration
        self.max_files_per_label = 50
        self.min_segments_per_file = 3
        self.random_seed = 42
        
        # Position types for arcs
        self.standard_positions = ['L', 'C', 'R']  # Standard arc positions
        self.restriking_positions = ['L', 'C', 'R', 'L2', 'C2', 'R2']  # Restriking arcs
        
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
        """
        Get the next ID for a table without auto-increment.
        
        Args:
            table_name: Name of the table
            id_column: Name of the ID column
            
        Returns:
            Next available ID
        """
        self.cursor.execute(
            f"SELECT COALESCE(MAX({id_column}), 0) + 1 FROM {table_name}"
        )
        return self.cursor.fetchone()['coalesce']
    
    def load_experiment_config(self) -> Dict[str, Any]:
        """
        Load experiment configuration from database.
        
        Returns:
            Experiment configuration dictionary
        """
        self.cursor.execute("""
            SELECT 
                experiment_name,
                segment_selection_config,
                segment_balance
            FROM ml_experiments
            WHERE experiment_id = %s
        """, (self.experiment_id,))
        
        result = self.cursor.fetchone()
        if not result:
            raise ValueError(f"Experiment {self.experiment_id} not found")
            
        config = result['segment_selection_config'] or {}
        config['segment_balance'] = result['segment_balance']
        config['experiment_name'] = result['experiment_name']
        
        # Set defaults if not specified
        config.setdefault('random_seed', self.random_seed)
        config.setdefault('max_files_per_label', self.max_files_per_label)
        config.setdefault('min_segments_per_file', self.min_segments_per_file)
        config.setdefault('balanced_segments', True)
        config.setdefault('selection_strategy', 'position_balanced_per_file')
        config.setdefault('position_balance_mode', 'at_least_one')
        
        return config
    
    def get_valid_labels(self) -> List[Dict[str, Any]]:
        """
        Get valid label classes with sufficient data.
        
        Returns:
            List of label dictionaries with IDs and names
        """
        self.cursor.execute("""
            SELECT 
                el.label_id as label_class_id,
                el.experiment_label as label_name,
                COUNT(DISTINCT f.file_id) as file_count
            FROM experiment_labels el
            JOIN files f ON el.experiment_label = f.selected_label
            WHERE el.active = TRUE
            GROUP BY el.label_id, el.experiment_label
            HAVING COUNT(DISTINCT f.file_id) >= 10
            ORDER BY el.label_id
        """)
        
        return self.cursor.fetchall()
    
    def select_files_per_label(self, label_id: int, label_name: str, 
                              max_files: int, seed: int) -> List[int]:
        """
        Select up to max_files files for a specific label.
        
        Args:
            label_id: Label class ID
            label_name: Label name
            max_files: Maximum files to select
            seed: Random seed for reproducibility
            
        Returns:
            List of selected file IDs
        """
        # Get all available files for this label
        self.cursor.execute("""
            SELECT DISTINCT f.file_id
            FROM files f
            WHERE f.selected_label = %s
            ORDER BY f.file_id
        """, (label_name,))
        
        available_files = [row['file_id'] for row in self.cursor.fetchall()]
        
        # Apply random selection with seed
        random.seed(seed + label_id)  # Unique seed per label
        selected_files = random.sample(
            available_files, 
            min(max_files, len(available_files))
        )
        
        self.logger.info(f"Selected {len(selected_files)} files for label {label_name}")
        return selected_files
    
    def get_file_segments(self, file_id: int) -> List[Dict[str, Any]]:
        """
        Get all segments for a file with position information.
        
        Args:
            file_id: File ID
            
        Returns:
            List of segment dictionaries
        """
        self.cursor.execute("""
            SELECT 
                s.segment_id,
                s.experiment_file_id as file_id,
                s.beginning_index as start_sample_index,
                s.beginning_index + s.segment_length as end_sample_index,
                s.segment_type,
                s.segment_id_code,
                s.transient_relative_position as position_arc
            FROM data_segments s
            WHERE s.experiment_file_id = %s
                AND s.enabled = true
            ORDER BY s.beginning_index
        """, (file_id,))
        
        return self.cursor.fetchall()
    
    def select_balanced_segments(self, segments: List[Dict[str, Any]], 
                                min_segments: int) -> List[Dict[str, Any]]:
        """
        Select segments with position balance using segment_id_code prefixes.
        
        For balanced mode:
        - Group segments by segment_id_code prefix (Cl, Cm, Cr, L, R)
        - Find the minimum count across all groups
        - Randomly select that number of segments from EACH group
        
        Args:
            segments: Available segments
            min_segments: Minimum segments per position (not used in balanced mode)
            
        Returns:
            Selected segments with position balance
        """
        if not segments:
            return []
            
        # Group segments by segment_id_code prefix
        by_prefix = {}
        for seg in segments:
            # Get segment_id_code (it's the 6th element in the tuple from SQL query)
            segment_id_code = seg.get('segment_id_code') or seg[5] if len(seg) > 5 else None
            
            if segment_id_code:
                # Extract prefix (Cl, Cm, Cr, L, R)
                if segment_id_code.startswith('Cm'):
                    prefix = 'Cm'
                elif segment_id_code.startswith('Cl'):
                    prefix = 'Cl'
                elif segment_id_code.startswith('Cr'):
                    prefix = 'Cr'
                elif segment_id_code.startswith('L'):
                    prefix = 'L'
                elif segment_id_code.startswith('R'):
                    prefix = 'R'
                else:
                    prefix = 'Other'
            else:
                # Fallback to old position_arc if segment_id_code not available
                pos = seg.get('position_arc') or seg[6] if len(seg) > 6 else 'unknown'
                prefix = pos[0] if pos and pos != 'unknown' else 'Other'
            
            if prefix not in by_prefix:
                by_prefix[prefix] = []
            by_prefix[prefix].append(seg)
        
        selected = []
        
        # Find minimum count across all groups
        prefix_counts = {prefix: len(segs) for prefix, segs in by_prefix.items()}
        
        if not prefix_counts:
            return []
        
        # Find the minimum count across groups
        min_count = min(prefix_counts.values())
        
        # Select min_count segments from EACH group
        for prefix, prefix_segments in by_prefix.items():
            if prefix_segments:
                # Randomly select min_count segments from this group
                num_to_select = min(min_count, len(prefix_segments))
                selected_from_prefix = random.sample(prefix_segments, num_to_select)
                selected.extend(selected_from_prefix)
        
        logger.info(f"Balanced selection by segment_id_code prefix:")
        for prefix, count in prefix_counts.items():
            logger.info(f"  {prefix}: {count} available, selected {min(min_count, count)}")
        logger.info(f"Total: {min_count} × {len(by_prefix)} groups = {len(selected)} segments")
        
        return selected
    
    def generate_segment_pairs(self, segment_ids: List[int]) -> List[Tuple[int, int]]:
        """
        Generate all unique segment pairs with constraint segment_a < segment_b.
        
        Args:
            segment_ids: List of segment IDs
            
        Returns:
            List of segment pairs (segment_a_id, segment_b_id)
        """
        pairs = []
        for i in range(len(segment_ids)):
            for j in range(i + 1, len(segment_ids)):
                pairs.append((segment_ids[i], segment_ids[j]))
        return pairs
    
    def log_selection(self, file_id: int, segment_id: int, position: str, 
                     label_id: int, order: int):
        """
        Log segment selection to database.
        
        Args:
            file_id: File ID
            segment_id: Segment ID
            position: Position type
            label_id: Label class ID
            order: Selection order
        """
        log_id = self.get_next_id('segment_selection_log', 'log_id')
        
        self.cursor.execute("""
            INSERT INTO segment_selection_log 
            (log_id, experiment_id, segment_id, file_id, position_type, 
             label_id, selection_order, selected_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (log_id, self.experiment_id, segment_id, file_id, 
              position, label_id, order, datetime.now()))
    
    def create_segment_pairs_table(self):
        """Create the segment pairs table for the experiment."""
        table_name = f"experiment_{self.experiment_id:03d}_segment_pairs"
        
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                pair_id BIGINT PRIMARY KEY,
                segment_a_id INTEGER NOT NULL,
                segment_b_id INTEGER NOT NULL,
                file_a_id INTEGER,
                file_b_id INTEGER,
                label_a_id INTEGER,
                label_b_id INTEGER,
                is_same_file BOOLEAN,
                is_same_label BOOLEAN,
                created_at TIMESTAMP DEFAULT NOW(),
                CONSTRAINT unique_pair UNIQUE (segment_a_id, segment_b_id),
                CONSTRAINT ordered_segments CHECK (segment_a_id < segment_b_id)
            )
        """)
        
        # Create indexes
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_segment_a 
            ON {table_name}(segment_a_id)
        """)
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_segment_b 
            ON {table_name}(segment_b_id)
        """)
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_same_file 
            ON {table_name}(is_same_file)
        """)
        
        self.conn.commit()
        self.logger.info(f"Created table {table_name}")
    
    def insert_segment_pairs(self, pairs: List[Dict[str, Any]]):
        """
        Insert segment pairs into the database.
        
        Args:
            pairs: List of pair dictionaries
        """
        table_name = f"experiment_{self.experiment_id:03d}_segment_pairs"
        
        for batch_start in range(0, len(pairs), 10000):
            batch = pairs[batch_start:batch_start + 10000]
            
            values = []
            for pair in batch:
                pair_id = self.get_next_id(table_name, 'pair_id')
                values.append((
                    pair_id,
                    pair['segment_a_id'],
                    pair['segment_b_id'],
                    pair['file_a_id'],
                    pair['file_b_id'],
                    pair['label_a_id'],
                    pair['label_b_id'],
                    pair['is_same_file'],
                    pair['is_same_label']
                ))
            
            self.cursor.executemany(f"""
                INSERT INTO {table_name}
                (pair_id, segment_a_id, segment_b_id, file_a_id, file_b_id,
                 label_a_id, label_b_id, is_same_file, is_same_label)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, values)
            
            self.conn.commit()
            self.logger.info(f"Inserted batch of {len(batch)} pairs")
    
    def run_selection(self) -> Dict[str, Any]:
        """
        Run the complete segment selection process.
        
        Returns:
            Summary statistics of the selection
        """
        self.connect()
        
        try:
            # Load configuration
            config = self.load_experiment_config()
            self.logger.info(f"Running selection for experiment {self.experiment_id}")
            
            # Set random seed
            random.seed(config['random_seed'])
            
            # Get valid labels
            labels = self.get_valid_labels()
            self.logger.info(f"Found {len(labels)} valid labels")
            
            # Create segment pairs table
            self.create_segment_pairs_table()
            
            # Track all segments and statistics
            all_selected_segments = []
            file_segment_map = {}
            label_segment_map = {}
            selection_order = 0
            
            # Process each label
            for label in labels:
                label_id = label['label_class_id']
                label_name = label['label_name']
                
                # Select files for this label
                selected_files = self.select_files_per_label(
                    label_id, 
                    label_name,
                    config['max_files_per_label'],
                    config['random_seed']
                )
                
                label_segments = []
                
                # Process each selected file
                for file_id in selected_files:
                    # Get segments for this file
                    segments = self.get_file_segments(file_id)
                    
                    if not segments:
                        continue
                    
                    # Select balanced segments
                    if config['segment_balance']:
                        selected = self.select_balanced_segments(
                            segments, 
                            config['min_segments_per_file']
                        )
                    else:
                        # Random selection
                        num_to_select = min(
                            config['min_segments_per_file'],
                            len(segments)
                        )
                        selected = random.sample(segments, num_to_select)
                    
                    # Log selections
                    for seg in selected:
                        selection_order += 1
                        
                        # Get segment_id_code for position tracking
                        segment_id_code = seg.get('segment_id_code') or seg[5] if len(seg) > 5 else None
                        
                        # Determine position from segment_id_code prefix
                        if segment_id_code:
                            if segment_id_code.startswith('Cm'):
                                position = 'Cm'
                            elif segment_id_code.startswith('Cl'):
                                position = 'Cl'
                            elif segment_id_code.startswith('Cr'):
                                position = 'Cr'
                            elif segment_id_code.startswith('L'):
                                position = 'L'
                            elif segment_id_code.startswith('R'):
                                position = 'R'
                            else:
                                position = 'Other'
                        else:
                            position = seg.get('position_arc') or seg[6] if len(seg) > 6 else 'unknown'
                        
                        self.log_selection(
                            file_id,
                            seg['segment_id'] if isinstance(seg, dict) else seg[0],
                            position,
                            label_id,
                            selection_order
                        )
                        
                        # Track segment metadata
                        seg_info = {
                            'segment_id': seg['segment_id'] if isinstance(seg, dict) else seg[0],
                            'file_id': file_id,
                            'label_id': label_id,
                            'segment_id_code': segment_id_code,
                            'position': seg['position_arc']
                        }
                        all_selected_segments.append(seg_info)
                        label_segments.append(seg_info)
                        
                        if file_id not in file_segment_map:
                            file_segment_map[file_id] = []
                        file_segment_map[file_id].append(seg_info)
                
                label_segment_map[label_id] = label_segments
                self.logger.info(f"Selected {len(label_segments)} segments for label {label_name}")
            
            # Generate all segment pairs
            self.logger.info("Generating segment pairs...")
            all_pairs = []
            
            segment_list = all_selected_segments
            for i in range(len(segment_list)):
                for j in range(i + 1, len(segment_list)):
                    seg_a = segment_list[i]
                    seg_b = segment_list[j]
                    
                    # Ensure ordering constraint
                    if seg_a['segment_id'] > seg_b['segment_id']:
                        seg_a, seg_b = seg_b, seg_a
                    
                    pair = {
                        'segment_a_id': seg_a['segment_id'],
                        'segment_b_id': seg_b['segment_id'],
                        'file_a_id': seg_a['file_id'],
                        'file_b_id': seg_b['file_id'],
                        'label_a_id': seg_a['label_id'],
                        'label_b_id': seg_b['label_id'],
                        'is_same_file': seg_a['file_id'] == seg_b['file_id'],
                        'is_same_label': seg_a['label_id'] == seg_b['label_id']
                    }
                    all_pairs.append(pair)
            
            self.logger.info(f"Generated {len(all_pairs)} segment pairs")
            
            # Insert pairs into database
            self.insert_segment_pairs(all_pairs)
            
            # Prepare summary
            summary = {
                'experiment_id': self.experiment_id,
                'total_labels': len(labels),
                'total_files': len(file_segment_map),
                'total_segments': len(all_selected_segments),
                'total_pairs': len(all_pairs),
                'segments_per_label': {
                    label_id: len(segments) 
                    for label_id, segments in label_segment_map.items()
                },
                'files_per_label': {
                    label['label_class_id']: len([
                        f for f in file_segment_map 
                        if any(s['label_id'] == label['label_class_id'] 
                              for s in file_segment_map[f])
                    ])
                    for label in labels
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Log summary to database
            self.cursor.execute("""
                UPDATE ml_experiments
                SET segment_selection_config = %s
                WHERE experiment_id = %s
            """, (json.dumps({**config, 'selection_summary': summary}), 
                  self.experiment_id))
            
            self.conn.commit()
            
            return summary
            
        finally:
            self.disconnect()


def main():
    """Main entry point for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Segment selection for ML experiments')
    parser.add_argument('--experiment-id', type=int, required=True,
                       help='Experiment ID')
    parser.add_argument('--db-host', default='localhost',
                       help='Database host')
    parser.add_argument('--db-name', default='arc_detection',
                       help='Database name')
    parser.add_argument('--db-user', default='kjensen',
                       help='Database user')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Database configuration
    db_config = {
        'host': args.db_host,
        'database': args.db_name,
        'user': args.db_user
    }
    
    # Run selection
    selector = ExperimentSegmentSelector(args.experiment_id, db_config)
    summary = selector.run_selection()
    
    # Print summary
    print(f"\nSegment Selection Summary for Experiment {args.experiment_id}:")
    print(f"  Total labels: {summary['total_labels']}")
    print(f"  Total files: {summary['total_files']}")
    print(f"  Total segments: {summary['total_segments']}")
    print(f"  Total pairs: {summary['total_pairs']}")
    print(f"  Completed at: {summary['timestamp']}")


if __name__ == '__main__':
    main()