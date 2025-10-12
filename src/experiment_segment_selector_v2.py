#!/usr/bin/env python3
"""
Filename: experiment_segment_selector_v2.py
Author(s): Kristophor Jensen
Date Created: 20250920_183000
Date Revised: 20251011_000000
File version: 1.0.0.2
Description: Corrected segment selection with proper per-segment-code-type logic
             Filters segments by experiment-configured segment sizes
             Fixed table names: ml_experiments_segment_sizes, ml_segment_sizes_lut
"""

import logging
import random
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import json

logger = logging.getLogger(__name__)


class SegmentSelectorV2:
    """
    Segment selector with correct per-segment-code-type selection logic.

    Selection strategies:
    1. balanced: Find min count across all segment code types, select that many from EACH type
    2. fixed_per_type: Select N segments from each type (or all if less than N available)
    3. proportional: Select proportionally from each type
    """

    def __init__(self, experiment_id: int, db_config: Dict[str, Any]):
        """Initialize the segment selector"""
        self.experiment_id = experiment_id
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.random_seed = 42
        self.segment_sizes = None  # Will be loaded from experiment config

    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)

    def load_experiment_config(self):
        """Load experiment configuration including segment sizes"""
        self.cursor.execute("""
            SELECT DISTINCT ss.segment_size_n
            FROM ml_experiments_segment_sizes ess
            JOIN ml_segment_sizes_lut ss ON ess.segment_size_id = ss.segment_size_id
            WHERE ess.experiment_id = %s
            ORDER BY ss.segment_size_n
        """, (self.experiment_id,))

        self.segment_sizes = [row['segment_size_n'] for row in self.cursor.fetchall()]

        if not self.segment_sizes:
            self.logger.warning(f"No segment sizes configured for experiment {self.experiment_id}")
            self.logger.warning("Will select segments of ALL sizes")
        else:
            self.logger.info(f"Configured segment sizes: {self.segment_sizes}")

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()

    def parse_segment_code(self, segment_id_code: str) -> Tuple[str, int]:
        """Parse segment_id_code into type prefix and number"""
        if not segment_id_code:
            return ('Unknown', 0)

        match = re.match(r'^([A-Za-z]+)(\d+)$', segment_id_code)
        if match:
            return (match.group(1), int(match.group(2)))
        return (segment_id_code, 0)

    def get_file_segments_grouped(self, file_id: int) -> Dict[str, List[Dict]]:
        """
        Get all segments for a file, grouped by segment code type.
        Filters by configured segment sizes if available.

        Returns:
            Dictionary mapping segment code type to list of segments
        """
        # Build query with optional segment size filter
        if self.segment_sizes:
            # Filter by configured segment sizes
            size_placeholders = ','.join(['%s'] * len(self.segment_sizes))
            query = f"""
                SELECT
                    s.segment_id,
                    s.experiment_file_id as file_id,
                    s.beginning_index as start_index,
                    s.segment_length,
                    s.segment_type,
                    s.segment_id_code,
                    s.segment_label_id
                FROM data_segments s
                WHERE s.experiment_file_id = %s
                    AND s.enabled = true
                    AND s.segment_id_code IS NOT NULL
                    AND s.segment_length IN ({size_placeholders})
                ORDER BY s.beginning_index
            """
            params = (file_id,) + tuple(self.segment_sizes)
        else:
            # No size filter - select all sizes
            query = """
                SELECT
                    s.segment_id,
                    s.experiment_file_id as file_id,
                    s.beginning_index as start_index,
                    s.segment_length,
                    s.segment_type,
                    s.segment_id_code,
                    s.segment_label_id
                FROM data_segments s
                WHERE s.experiment_file_id = %s
                    AND s.enabled = true
                    AND s.segment_id_code IS NOT NULL
                ORDER BY s.beginning_index
            """
            params = (file_id,)

        self.cursor.execute(query, params)

        segments_by_type = {}

        for row in self.cursor.fetchall():
            segment = dict(row)

            # Parse segment code
            code_type, code_number = self.parse_segment_code(segment['segment_id_code'])
            segment['code_type'] = code_type
            segment['code_number'] = code_number

            # Group by type
            if code_type not in segments_by_type:
                segments_by_type[code_type] = []
            segments_by_type[code_type].append(segment)

        return segments_by_type

    def select_segments_from_file(self, segments_by_type: Dict[str, List[Dict]],
                                 strategy: str = 'balanced',
                                 segments_per_type: int = 3) -> List[Dict]:
        """
        Select segments from a single file based on strategy.

        Args:
            segments_by_type: Segments grouped by code type
            strategy: Selection strategy ('balanced', 'fixed_per_type', 'proportional')
            segments_per_type: Target segments per type (for fixed_per_type strategy)

        Returns:
            List of selected segments
        """
        if not segments_by_type:
            return []

        selected = []

        if strategy == 'balanced':
            # Find minimum count across all types
            type_counts = {t: len(segs) for t, segs in segments_by_type.items()}
            min_count = min(type_counts.values())

            self.logger.info(f"  Balanced selection - min count: {min_count}")
            self.logger.info(f"  Available segments by type: {type_counts}")

            # Select min_count from EACH type
            for code_type, type_segments in segments_by_type.items():
                num_to_select = min(min_count, len(type_segments))
                if num_to_select > 0:
                    selected_from_type = random.sample(type_segments, num_to_select)
                    selected.extend(selected_from_type)
                    self.logger.info(f"    {code_type}: selected {num_to_select} of {len(type_segments)}")

        elif strategy == 'fixed_per_type':
            # Select fixed number from each type (or all if fewer available)
            for code_type, type_segments in segments_by_type.items():
                num_to_select = min(segments_per_type, len(type_segments))
                if num_to_select > 0:
                    selected_from_type = random.sample(type_segments, num_to_select)
                    selected.extend(selected_from_type)
                    self.logger.info(f"    {code_type}: selected {num_to_select} of {len(type_segments)} (target: {segments_per_type})")

        elif strategy == 'proportional':
            # Calculate total segments to select
            total_available = sum(len(segs) for segs in segments_by_type.values())
            total_to_select = min(total_available, len(segments_by_type) * segments_per_type)

            for code_type, type_segments in segments_by_type.items():
                proportion = len(type_segments) / total_available
                num_to_select = max(1, int(total_to_select * proportion))
                num_to_select = min(num_to_select, len(type_segments))

                if num_to_select > 0:
                    selected_from_type = random.sample(type_segments, num_to_select)
                    selected.extend(selected_from_type)
                    self.logger.info(f"    {code_type}: selected {num_to_select} of {len(type_segments)} (proportional)")

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
                selection_strategy VARCHAR(50),
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
            CREATE INDEX IF NOT EXISTS idx_{table_name}_code_type
            ON {table_name}(segment_code_type)
        """)

        self.conn.commit()
        self.logger.info(f"Created/verified table {table_name}")

    def run_selection(self, strategy: str = 'balanced',
                     segments_per_type: int = 3) -> Dict[str, Any]:
        """
        Run segment selection with proper per-segment-code-type logic.

        Args:
            strategy: Selection strategy ('balanced', 'fixed_per_type', 'proportional')
            segments_per_type: Target segments per type (used differently based on strategy)

        Returns:
            Summary statistics
        """
        self.connect()

        try:
            # Load experiment configuration (segment sizes)
            self.load_experiment_config()

            # Set random seed
            random.seed(self.random_seed)

            # Create table
            self.create_segment_training_table()

            # Get files from file training data
            file_table = f"experiment_{self.experiment_id:03d}_file_training_data"

            self.cursor.execute(f"""
                SELECT DISTINCT file_id, file_label_id
                FROM {file_table}
                WHERE experiment_id = %s
                ORDER BY file_id
            """, (self.experiment_id,))

            files = self.cursor.fetchall()

            if not files:
                self.logger.warning(f"No files in {file_table}. Run select-files first.")
                return {'error': 'No files selected', 'total_segments': 0}

            self.logger.info(f"\nProcessing {len(files)} files with strategy: {strategy}")
            if strategy == 'balanced':
                self.logger.info("  Will select minimum count across all segment types from EACH type")
            elif strategy == 'fixed_per_type':
                self.logger.info(f"  Will select up to {segments_per_type} segments from EACH type")

            # Clear existing selections
            table_name = f"experiment_{self.experiment_id:03d}_segment_training_data"
            self.cursor.execute(f"DELETE FROM {table_name} WHERE experiment_id = %s", (self.experiment_id,))

            # Process each file
            all_selected = []
            selection_order = 0

            for file_row in files:
                file_id = file_row['file_id']
                file_label_id = file_row['file_label_id']

                self.logger.info(f"\nFile {file_id}:")

                # Get segments grouped by type
                segments_by_type = self.get_file_segments_grouped(file_id)

                if not segments_by_type:
                    self.logger.info("  No segments with segment_id_code")
                    continue

                # Select segments based on strategy
                selected = self.select_segments_from_file(
                    segments_by_type,
                    strategy=strategy,
                    segments_per_type=segments_per_type
                )

                # Add metadata and insert
                for seg in selected:
                    seg['file_label_id'] = file_label_id
                    selection_order += 1

                    self.cursor.execute(f"""
                        INSERT INTO {table_name} (
                            experiment_id, segment_id, file_id,
                            segment_index, segment_code_type, segment_code_number,
                            segment_label_id, file_label_id, selection_order,
                            selection_strategy
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (experiment_id, segment_id) DO NOTHING
                    """, (
                        self.experiment_id, seg['segment_id'], seg['file_id'],
                        0, seg['code_type'], seg['code_number'],
                        seg.get('segment_label_id'), file_label_id, selection_order,
                        strategy
                    ))

                all_selected.extend(selected)
                self.logger.info(f"  Total selected from file: {len(selected)}")

            self.conn.commit()

            # Summary statistics
            segment_types = {}
            for seg in all_selected:
                code_type = seg['code_type']
                if code_type not in segment_types:
                    segment_types[code_type] = 0
                segment_types[code_type] += 1

            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"SELECTION COMPLETE")
            self.logger.info(f"Strategy: {strategy}")
            self.logger.info(f"Total files: {len(files)}")
            self.logger.info(f"Total segments selected: {len(all_selected)}")
            self.logger.info(f"Average per file: {len(all_selected)/len(files):.1f}")
            self.logger.info(f"\nSegment type distribution:")
            for code_type, count in sorted(segment_types.items()):
                self.logger.info(f"  {code_type}: {count}")

            return {
                'success': True,
                'total_segments': len(all_selected),
                'total_files': len(files),
                'segments_by_type': segment_types,
                'strategy': strategy
            }

        except Exception as e:
            self.logger.error(f"Error in segment selection: {e}")
            if self.conn:
                self.conn.rollback()
            raise

        finally:
            self.disconnect()


def main():
    """Test the segment selector"""
    import argparse

    parser = argparse.ArgumentParser(description='Segment selection with proper logic')
    parser.add_argument('experiment_id', type=int, help='Experiment ID')
    parser.add_argument('--strategy', default='balanced',
                       choices=['balanced', 'fixed_per_type', 'proportional'],
                       help='Selection strategy')
    parser.add_argument('--segments-per-type', type=int, default=3,
                       help='Target segments per type (meaning varies by strategy)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'arc_detection',
        'user': 'kjensen'
    }

    selector = SegmentSelectorV2(args.experiment_id, db_config)
    result = selector.run_selection(
        strategy=args.strategy,
        segments_per_type=args.segments_per_type
    )

    return 0 if result.get('success') else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())