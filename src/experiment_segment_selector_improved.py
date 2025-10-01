#!/usr/bin/env python3
"""
Filename: experiment_segment_selector_improved.py
Author(s): Kristophor Jensen
Date Created: 20250920_170000
Date Revised: 20250920_170000
File version: 1.0.0.0
Description: Improved segment selection using proper segment_id_code parsing
"""

import logging
import random
import re
from typing import Dict, List, Tuple, Optional, Any, Set
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import json

logger = logging.getLogger(__name__)


class ImprovedSegmentSelector:
    """
    Improved segment selection that properly uses segment_id_code:
    - Parses segment_id_code into type and number
    - Groups segments by type prefix
    - Ensures balanced selection across types
    """

    def __init__(self, experiment_id: int, db_config: Dict[str, Any]):
        """Initialize the segment selector"""
        self.experiment_id = experiment_id
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)

        # Default configuration
        self.max_files_per_label = 50
        self.min_segments_per_file = 3
        self.random_seed = 42

        # Known segment type prefixes
        self.segment_type_prefixes = {
            'L': 'left_arc',
            'R': 'right_arc',
            'C': 'center_arc',
            'Cl': 'center_left',
            'Cm': 'center_middle',
            'Cr': 'center_right',
            'A': 'augmented',
            'N': 'normal',
            'S': 'steady_state'
        }

    def parse_segment_id_code(self, segment_id_code: str) -> Tuple[str, int]:
        """
        Parse segment_id_code into type prefix and number.

        Args:
            segment_id_code: Code like 'L001', 'Cm003', etc.

        Returns:
            Tuple of (type_prefix, number)
        """
        if not segment_id_code:
            return ('Unknown', 0)

        # Match pattern: letters followed by digits
        match = re.match(r'^([A-Za-z]+)(\d+)$', segment_id_code)
        if match:
            type_prefix = match.group(1)
            number = int(match.group(2))
            return (type_prefix, number)

        # Fallback for non-standard codes
        return (segment_id_code, 0)

    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def get_file_segments_detailed(self, file_id: int) -> List[Dict[str, Any]]:
        """
        Get all segments for a file with detailed information.

        Args:
            file_id: File ID

        Returns:
            List of segment dictionaries with parsed code info
        """
        self.cursor.execute("""
            SELECT
                s.segment_id,
                s.experiment_file_id as file_id,
                s.beginning_index as start_sample_index,
                s.beginning_index + s.segment_length as end_sample_index,
                s.segment_length,
                s.segment_type,
                s.segment_id_code,
                s.transient_relative_position as position_arc,
                s.segment_label_id
            FROM data_segments s
            WHERE s.experiment_file_id = %s
                AND s.enabled = true
            ORDER BY s.beginning_index
        """, (file_id,))

        segments = []
        for row in self.cursor.fetchall():
            segment = dict(row)

            # Parse segment_id_code
            type_prefix, code_number = self.parse_segment_id_code(segment['segment_id_code'])
            segment['code_type'] = type_prefix
            segment['code_number'] = code_number

            segments.append(segment)

        return segments

    def select_balanced_segments_improved(self, segments: List[Dict[str, Any]],
                                         min_segments_per_file: int = 3,
                                         balance_strategy: str = 'balanced') -> List[Dict[str, Any]]:
        """
        Select segments with improved segment_id_code handling.

        Strategies:
        - 'balanced': Find minimum count across all type prefixes, select that many from each
        - 'proportional': Select proportionally from each type
        - 'random': Random selection regardless of type

        Args:
            segments: Available segments with parsed code info
            min_segments_per_file: Minimum segments to select
            balance_strategy: Selection strategy

        Returns:
            Selected segments
        """
        if not segments:
            return []

        # Group segments by code type
        segments_by_type = {}
        for seg in segments:
            code_type = seg.get('code_type', 'Unknown')

            if code_type not in segments_by_type:
                segments_by_type[code_type] = []
            segments_by_type[code_type].append(seg)

        selected = []

        if balance_strategy == 'balanced':
            # Find minimum count across all types (excluding very small groups)
            type_counts = {t: len(segs) for t, segs in segments_by_type.items()}

            # Filter out types with very few segments (less than min_segments_per_file)
            valid_types = {t: count for t, count in type_counts.items()
                          if count >= min_segments_per_file}

            if not valid_types:
                # If no type has enough segments, just select from all
                valid_types = type_counts

            if valid_types:
                # Find the minimum count across valid types
                min_count = min(valid_types.values())

                # Ensure we select at least min_segments_per_file from each type
                segments_per_type = max(min_count, min_segments_per_file)

                # Select segments from each type
                for code_type, type_segments in segments_by_type.items():
                    if code_type in valid_types:
                        num_to_select = min(segments_per_type, len(type_segments))

                        # Random selection within type
                        selected_from_type = random.sample(type_segments, num_to_select)
                        selected.extend(selected_from_type)

                        self.logger.info(f"  Type '{code_type}': {len(type_segments)} available, selected {num_to_select}")

        elif balance_strategy == 'proportional':
            # Calculate total segments needed
            total_needed = max(len(segments_by_type) * min_segments_per_file, 20)

            # Select proportionally from each type
            for code_type, type_segments in segments_by_type.items():
                proportion = len(type_segments) / len(segments)
                num_to_select = max(int(total_needed * proportion), 1)
                num_to_select = min(num_to_select, len(type_segments))

                selected_from_type = random.sample(type_segments, num_to_select)
                selected.extend(selected_from_type)

                self.logger.info(f"  Type '{code_type}': {len(type_segments)} available, selected {num_to_select} (proportional)")

        else:  # random
            # Just random selection
            num_to_select = max(min_segments_per_file * len(segments_by_type), 20)
            num_to_select = min(num_to_select, len(segments))
            selected = random.sample(segments, num_to_select)

            self.logger.info(f"  Random selection: {len(segments)} available, selected {num_to_select}")

        self.logger.info(f"Total selected: {len(selected)} segments from {len(segments_by_type)} types")
        return selected

    def create_segment_training_table(self):
        """Create the segment training data table"""
        table_name = f"experiment_{self.experiment_id:03d}_segment_training_data"

        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                selection_id SERIAL PRIMARY KEY,
                experiment_id INTEGER NOT NULL,
                segment_id INTEGER NOT NULL,
                file_id INTEGER NOT NULL,
                segment_index INTEGER,
                segment_code_type VARCHAR(10),
                segment_code_number INTEGER,
                segment_label_id INTEGER,
                file_label_id INTEGER,
                selection_order INTEGER,
                selection_group VARCHAR(50),
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(experiment_id, segment_id)
            )
        """)

        # Create indexes
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_segment
            ON {table_name}(segment_id)
        """)
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_file
            ON {table_name}(file_id)
        """)
        self.cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_code_type
            ON {table_name}(segment_code_type)
        """)

        self.conn.commit()
        self.logger.info(f"Created/verified table {table_name}")

    def insert_selected_segments(self, selected_segments: List[Dict[str, Any]]):
        """Insert selected segments into training data table"""
        table_name = f"experiment_{self.experiment_id:03d}_segment_training_data"

        # Clear existing selections
        self.cursor.execute(f"DELETE FROM {table_name} WHERE experiment_id = %s", (self.experiment_id,))

        # Insert new selections
        selection_order = 0
        for seg in selected_segments:
            selection_order += 1

            self.cursor.execute(f"""
                INSERT INTO {table_name} (
                    experiment_id, segment_id, file_id,
                    segment_index, segment_code_type, segment_code_number,
                    segment_label_id, file_label_id, selection_order
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (experiment_id, segment_id) DO NOTHING
            """, (
                self.experiment_id,
                seg['segment_id'],
                seg['file_id'],
                seg.get('segment_index', 0),
                seg.get('code_type'),
                seg.get('code_number'),
                seg.get('segment_label_id'),
                seg.get('file_label_id'),
                selection_order
            ))

        self.conn.commit()
        self.logger.info(f"Inserted {selection_order} segments into {table_name}")

    def analyze_segment_distribution(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of selected segments"""
        analysis = {
            'total_segments': len(segments),
            'by_type': {},
            'by_file': {},
            'code_number_ranges': {}
        }

        for seg in segments:
            code_type = seg.get('code_type', 'Unknown')

            # Count by type
            if code_type not in analysis['by_type']:
                analysis['by_type'][code_type] = 0
            analysis['by_type'][code_type] += 1

            # Count by file
            file_id = seg['file_id']
            if file_id not in analysis['by_file']:
                analysis['by_file'][file_id] = 0
            analysis['by_file'][file_id] += 1

            # Track code number ranges
            code_number = seg.get('code_number', 0)
            if code_type not in analysis['code_number_ranges']:
                analysis['code_number_ranges'][code_type] = {'min': code_number, 'max': code_number}
            else:
                analysis['code_number_ranges'][code_type]['min'] = min(
                    analysis['code_number_ranges'][code_type]['min'], code_number
                )
                analysis['code_number_ranges'][code_type]['max'] = max(
                    analysis['code_number_ranges'][code_type]['max'], code_number
                )

        return analysis

    def run_selection(self, balance_strategy: str = 'balanced') -> Dict[str, Any]:
        """
        Run the complete segment selection process.

        Args:
            balance_strategy: Strategy for selection ('balanced', 'proportional', 'random')

        Returns:
            Summary statistics of the selection
        """
        self.connect()

        try:
            # Create table if needed
            self.create_segment_training_table()

            # Get files from file training data
            file_table = f"experiment_{self.experiment_id:03d}_file_training_data"

            self.cursor.execute(f"""
                SELECT DISTINCT file_id, file_label_id
                FROM {file_table}
                WHERE experiment_id = %s
            """, (self.experiment_id,))

            files = self.cursor.fetchall()

            if not files:
                self.logger.warning(f"No files in {file_table}. Run select-files first.")
                return {'error': 'No files selected', 'total_segments': 0}

            self.logger.info(f"Processing {len(files)} files")

            # Collect all selected segments
            all_selected = []

            for file_row in files:
                file_id = file_row['file_id']
                file_label_id = file_row['file_label_id']

                # Get segments for file
                segments = self.get_file_segments_detailed(file_id)

                if not segments:
                    continue

                # Add file label to segments
                for seg in segments:
                    seg['file_label_id'] = file_label_id

                # Select balanced segments
                selected = self.select_balanced_segments_improved(
                    segments,
                    min_segments_per_file=self.min_segments_per_file,
                    balance_strategy=balance_strategy
                )

                all_selected.extend(selected)

            # Insert selected segments
            if all_selected:
                self.insert_selected_segments(all_selected)

                # Analyze distribution
                analysis = self.analyze_segment_distribution(all_selected)

                self.logger.info(f"\nSelection complete:")
                self.logger.info(f"  Total segments: {analysis['total_segments']}")
                self.logger.info(f"  Files: {len(analysis['by_file'])}")
                self.logger.info(f"  Segment types: {len(analysis['by_type'])}")

                for code_type, count in sorted(analysis['by_type'].items()):
                    range_info = analysis['code_number_ranges'].get(code_type, {})
                    self.logger.info(f"    {code_type}: {count} segments (numbers {range_info.get('min', 0)}-{range_info.get('max', 0)})")

                return {
                    'success': True,
                    'total_segments': len(all_selected),
                    'total_files': len(analysis['by_file']),
                    'analysis': analysis
                }

            else:
                return {
                    'success': False,
                    'error': 'No segments selected',
                    'total_segments': 0
                }

        except Exception as e:
            self.logger.error(f"Error in segment selection: {e}")
            if self.conn:
                self.conn.rollback()
            raise

        finally:
            self.disconnect()


def main():
    """Test the improved segment selector"""
    import argparse

    parser = argparse.ArgumentParser(description='Improved segment selection')
    parser.add_argument('experiment_id', type=int, help='Experiment ID')
    parser.add_argument('--strategy', default='balanced',
                       choices=['balanced', 'proportional', 'random'],
                       help='Selection strategy')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'arc_detection',
        'user': 'kjensen'
    }

    selector = ImprovedSegmentSelector(args.experiment_id, db_config)
    result = selector.run_selection(balance_strategy=args.strategy)

    print(f"\nResult: {json.dumps(result, indent=2)}")

    return 0 if result.get('success') else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())