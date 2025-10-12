#!/usr/bin/env python3
"""
Filename: experiment_segment_pair_generator_v2.py
Author(s): Kristophor Jensen
Date Created: 20250920_200000
Date Revised: 20251011_000000
File version: 1.0.0.1
Description: Generate segment pairs compatible with v2 segment selector
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from typing import Dict, List, Any, Tuple, Optional
import itertools
import random

logger = logging.getLogger(__name__)

class ExperimentSegmentPairGeneratorV2:
    """Generate segment pairs for distance calculations - compatible with v2 selector"""

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
                    segment_code_type_1 VARCHAR(10),
                    segment_code_type_2 VARCHAR(10),
                    segment_code_number_1 INTEGER,
                    segment_code_number_2 INTEGER,
                    segment_label_id_1 INTEGER,
                    segment_label_id_2 INTEGER,
                    file_label_id_1 INTEGER,
                    file_label_id_2 INTEGER,
                    is_same_segment_label BOOLEAN,
                    is_same_file_label BOOLEAN,
                    is_same_code_type BOOLEAN,
                    pair_type VARCHAR(50),
                    pair_weight FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT unique_pair_{self.experiment_id:03d} UNIQUE(experiment_id, segment_id_1, segment_id_2)
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
                CREATE INDEX IF NOT EXISTS idx_{self.pairs_table}_same_segment_label
                ON {self.pairs_table}(is_same_segment_label)
            """)
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pairs_table}_same_code_type
                ON {self.pairs_table}(is_same_code_type)
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
        """Get all selected segments for this experiment with segment_length"""
        cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Query with segment_length from data_segments for length matching
            cursor.execute(f"""
                SELECT
                    st.segment_id,
                    st.file_id,
                    st.segment_code_type,
                    st.segment_code_number,
                    st.segment_label_id,
                    st.file_label_id,
                    st.segment_index,
                    st.selection_order,
                    ds.segment_length
                FROM {self.segment_table} st
                JOIN data_segments ds ON st.segment_id = ds.segment_id
                WHERE st.experiment_id = %s
                ORDER BY st.selection_order
            """, (self.experiment_id,))

            segments = [dict(row) for row in cursor]

            if not segments:
                logger.warning(f"No segments found in {self.segment_table}")
            else:
                # Log segment length distribution
                lengths = {}
                for seg in segments:
                    length = seg.get('segment_length', 'unknown')
                    lengths[length] = lengths.get(length, 0) + 1
                logger.info(f"Segment length distribution: {', '.join(f'{k}={v}' for k,v in sorted(lengths.items()))}")

            return segments

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
            strategy: Pairing strategy ('all_combinations', 'match_lengths_all_combinations',
                     'balanced', 'random_sample', 'code_type_balanced')
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

        logger.info(f"Generating pairs from {len(segments)} segments using strategy: {strategy}")

        # Generate pairs based on strategy
        if strategy == 'all_combinations':
            pairs = self._generate_all_combinations(segments)
        elif strategy == 'match_lengths_all_combinations':
            pairs = self._generate_match_lengths_all_combinations(segments)
        elif strategy == 'balanced':
            pairs = self._generate_balanced_pairs(segments, max_pairs_per_segment, same_label_ratio)
        elif strategy == 'code_type_balanced':
            pairs = self._generate_code_type_balanced_pairs(segments, max_pairs_per_segment)
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

                is_same_segment_label = seg1.get('segment_label_id') == seg2.get('segment_label_id')
                is_same_file_label = seg1.get('file_label_id') == seg2.get('file_label_id')
                is_same_code_type = seg1.get('segment_code_type') == seg2.get('segment_code_type')

                # Determine pair type
                if is_same_code_type and is_same_segment_label:
                    pair_type = 'same_type_same_label'
                elif is_same_code_type:
                    pair_type = 'same_type_diff_label'
                elif is_same_segment_label:
                    pair_type = 'diff_type_same_label'
                else:
                    pair_type = 'diff_type_diff_label'

                pairs.append((
                    seg1['segment_id'], seg2['segment_id'],
                    seg1['file_id'], seg2['file_id'],
                    seg1.get('segment_code_type'), seg2.get('segment_code_type'),
                    seg1.get('segment_code_number'), seg2.get('segment_code_number'),
                    seg1.get('segment_label_id'), seg2.get('segment_label_id'),
                    seg1.get('file_label_id'), seg2.get('file_label_id'),
                    is_same_segment_label, is_same_file_label, is_same_code_type,
                    pair_type
                ))

        logger.info(f"Generated {len(pairs)} pairs from {len(segments)} segments")
        return pairs

    def _generate_match_lengths_all_combinations(self, segments: List[Dict]) -> List[Tuple]:
        """Generate all possible segment pairs, but only for segments with matching lengths

        This ensures that distance calculations compare segments of the same size,
        which is critical for meaningful analysis. Segments of different lengths
        cannot be meaningfully compared in most distance metrics.
        """
        pairs = []

        # Group segments by length
        segments_by_length = {}
        for seg in segments:
            length = seg.get('segment_length')
            if length is None:
                logger.warning(f"Segment {seg.get('segment_id')} has no length, skipping")
                continue
            if length not in segments_by_length:
                segments_by_length[length] = []
            segments_by_length[length].append(seg)

        logger.info(f"Grouping by length: {', '.join(f'{k}={len(v)}' for k,v in sorted(segments_by_length.items()))}")

        # Generate all combinations within each length group
        total_pairs = 0
        for length, length_segments in segments_by_length.items():
            length_pairs = 0
            for i in range(len(length_segments)):
                for j in range(i + 1, len(length_segments)):
                    seg1 = length_segments[i]
                    seg2 = length_segments[j]

                    is_same_segment_label = seg1.get('segment_label_id') == seg2.get('segment_label_id')
                    is_same_file_label = seg1.get('file_label_id') == seg2.get('file_label_id')
                    is_same_code_type = seg1.get('segment_code_type') == seg2.get('segment_code_type')

                    # Determine pair type
                    if is_same_code_type and is_same_segment_label:
                        pair_type = 'same_type_same_label'
                    elif is_same_code_type:
                        pair_type = 'same_type_diff_label'
                    elif is_same_segment_label:
                        pair_type = 'diff_type_same_label'
                    else:
                        pair_type = 'diff_type_diff_label'

                    pairs.append((
                        seg1['segment_id'], seg2['segment_id'],
                        seg1['file_id'], seg2['file_id'],
                        seg1.get('segment_code_type'), seg2.get('segment_code_type'),
                        seg1.get('segment_code_number'), seg2.get('segment_code_number'),
                        seg1.get('segment_label_id'), seg2.get('segment_label_id'),
                        seg1.get('file_label_id'), seg2.get('file_label_id'),
                        is_same_segment_label, is_same_file_label, is_same_code_type,
                        pair_type
                    ))
                    length_pairs += 1

            logger.info(f"Generated {length_pairs} pairs for length {length} ({len(length_segments)} segments)")
            total_pairs += length_pairs

        logger.info(f"Generated {total_pairs} length-matched pairs from {len(segments)} segments across {len(segments_by_length)} length groups")
        return pairs

    def _generate_balanced_pairs(self, segments: List[Dict],
                                 max_pairs_per_segment: int,
                                 same_label_ratio: float) -> List[Tuple]:
        """Generate balanced pairs with controlled same/different label ratio"""
        pairs = []
        segments_by_label = {}

        # Group segments by segment label
        for seg in segments:
            label = seg.get('segment_label_id', 'unknown')
            if label not in segments_by_label:
                segments_by_label[label] = []
            segments_by_label[label].append(seg)

        # For each segment, generate balanced pairs
        for seg1 in segments:
            seg1_label = seg1.get('segment_label_id', 'unknown')
            same_label_segments = [s for s in segments_by_label[seg1_label]
                                 if s['segment_id'] != seg1['segment_id']]
            diff_label_segments = [s for s in segments
                                 if s.get('segment_label_id') != seg1_label]

            # Calculate how many of each type
            if max_pairs_per_segment:
                n_same = min(int(max_pairs_per_segment * same_label_ratio),
                           len(same_label_segments))
                n_diff = min(max_pairs_per_segment - n_same,
                           len(diff_label_segments))
            else:
                n_same = len(same_label_segments)
                n_diff = len(diff_label_segments)

            # Sample same-label pairs
            if same_label_segments and n_same > 0:
                for seg2 in random.sample(same_label_segments, n_same):
                    if seg1['segment_id'] < seg2['segment_id']:  # Avoid duplicates
                        is_same_code_type = seg1.get('segment_code_type') == seg2.get('segment_code_type')
                        is_same_file_label = seg1.get('file_label_id') == seg2.get('file_label_id')

                        if is_same_code_type:
                            pair_type = 'same_type_same_label'
                        else:
                            pair_type = 'diff_type_same_label'

                        pairs.append((
                            seg1['segment_id'], seg2['segment_id'],
                            seg1['file_id'], seg2['file_id'],
                            seg1.get('segment_code_type'), seg2.get('segment_code_type'),
                            seg1.get('segment_code_number'), seg2.get('segment_code_number'),
                            seg1.get('segment_label_id'), seg2.get('segment_label_id'),
                            seg1.get('file_label_id'), seg2.get('file_label_id'),
                            True, is_same_file_label, is_same_code_type,
                            pair_type
                        ))

            # Sample different-label pairs
            if diff_label_segments and n_diff > 0:
                for seg2 in random.sample(diff_label_segments, n_diff):
                    if seg1['segment_id'] < seg2['segment_id']:  # Avoid duplicates
                        is_same_code_type = seg1.get('segment_code_type') == seg2.get('segment_code_type')
                        is_same_file_label = seg1.get('file_label_id') == seg2.get('file_label_id')

                        if is_same_code_type:
                            pair_type = 'same_type_diff_label'
                        else:
                            pair_type = 'diff_type_diff_label'

                        pairs.append((
                            seg1['segment_id'], seg2['segment_id'],
                            seg1['file_id'], seg2['file_id'],
                            seg1.get('segment_code_type'), seg2.get('segment_code_type'),
                            seg1.get('segment_code_number'), seg2.get('segment_code_number'),
                            seg1.get('segment_label_id'), seg2.get('segment_label_id'),
                            seg1.get('file_label_id'), seg2.get('file_label_id'),
                            False, is_same_file_label, is_same_code_type,
                            pair_type
                        ))

        # Remove duplicates (in case any were created)
        pairs = list(set(pairs))

        logger.info(f"Generated {len(pairs)} balanced pairs from {len(segments)} segments")
        return pairs

    def _generate_code_type_balanced_pairs(self, segments: List[Dict],
                                          max_pairs_per_segment: int) -> List[Tuple]:
        """Generate pairs balanced by segment code type"""
        pairs = []
        segments_by_code_type = {}

        # Group segments by code type
        for seg in segments:
            code_type = seg.get('segment_code_type', 'unknown')
            if code_type not in segments_by_code_type:
                segments_by_code_type[code_type] = []
            segments_by_code_type[code_type].append(seg)

        logger.info(f"Segment distribution by code type: {', '.join(f'{k}={len(v)}' for k,v in segments_by_code_type.items())}")

        # Generate within-type pairs (same code type)
        for code_type, type_segments in segments_by_code_type.items():
            if len(type_segments) > 1:
                for i in range(len(type_segments)):
                    for j in range(i + 1, len(type_segments)):
                        seg1 = type_segments[i]
                        seg2 = type_segments[j]

                        is_same_segment_label = seg1.get('segment_label_id') == seg2.get('segment_label_id')
                        is_same_file_label = seg1.get('file_label_id') == seg2.get('file_label_id')

                        pair_type = 'same_type_same_label' if is_same_segment_label else 'same_type_diff_label'

                        pairs.append((
                            seg1['segment_id'], seg2['segment_id'],
                            seg1['file_id'], seg2['file_id'],
                            seg1.get('segment_code_type'), seg2.get('segment_code_type'),
                            seg1.get('segment_code_number'), seg2.get('segment_code_number'),
                            seg1.get('segment_label_id'), seg2.get('segment_label_id'),
                            seg1.get('file_label_id'), seg2.get('file_label_id'),
                            is_same_segment_label, is_same_file_label, True,
                            pair_type
                        ))

        # Generate across-type pairs (different code types) - sample to keep balanced
        code_types = list(segments_by_code_type.keys())
        for i in range(len(code_types)):
            for j in range(i + 1, len(code_types)):
                type1_segments = segments_by_code_type[code_types[i]]
                type2_segments = segments_by_code_type[code_types[j]]

                # Sample pairs between types
                n_pairs = min(len(type1_segments), len(type2_segments))
                if max_pairs_per_segment:
                    n_pairs = min(n_pairs, max_pairs_per_segment // len(code_types))

                sampled_type1 = random.sample(type1_segments, n_pairs)
                sampled_type2 = random.sample(type2_segments, n_pairs)

                for seg1, seg2 in zip(sampled_type1, sampled_type2):
                    is_same_segment_label = seg1.get('segment_label_id') == seg2.get('segment_label_id')
                    is_same_file_label = seg1.get('file_label_id') == seg2.get('file_label_id')

                    pair_type = 'diff_type_same_label' if is_same_segment_label else 'diff_type_diff_label'

                    pairs.append((
                        seg1['segment_id'], seg2['segment_id'],
                        seg1['file_id'], seg2['file_id'],
                        seg1.get('segment_code_type'), seg2.get('segment_code_type'),
                        seg1.get('segment_code_number'), seg2.get('segment_code_number'),
                        seg1.get('segment_label_id'), seg2.get('segment_label_id'),
                        seg1.get('file_label_id'), seg2.get('file_label_id'),
                        is_same_segment_label, is_same_file_label, False,
                        pair_type
                    ))

        logger.info(f"Generated {len(pairs)} code-type-balanced pairs from {len(segments)} segments")
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
            inserted = 0
            for pair in pairs:
                cursor.execute(f"""
                    INSERT INTO {self.pairs_table}
                    (experiment_id, segment_id_1, segment_id_2,
                     file_id_1, file_id_2,
                     segment_code_type_1, segment_code_type_2,
                     segment_code_number_1, segment_code_number_2,
                     segment_label_id_1, segment_label_id_2,
                     file_label_id_1, file_label_id_2,
                     is_same_segment_label, is_same_file_label, is_same_code_type,
                     pair_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (experiment_id, segment_id_1, segment_id_2) DO NOTHING
                """, (
                    self.experiment_id,
                    pair[0], pair[1],  # segment_ids
                    pair[2], pair[3],  # file_ids
                    pair[4], pair[5],  # segment_code_types
                    pair[6], pair[7],  # segment_code_numbers
                    pair[8], pair[9],  # segment_label_ids
                    pair[10], pair[11],  # file_label_ids
                    pair[12], pair[13], pair[14],  # booleans
                    pair[15]  # pair_type
                ))
                if cursor.rowcount > 0:
                    inserted += 1

            self.db_conn.commit()
            logger.info(f"Inserted {inserted} segment pairs")
            return inserted

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
                    COUNT(*) as count
                FROM {self.pairs_table}
                WHERE experiment_id = %s
                GROUP BY pair_type
            """, (self.experiment_id,))

            type_counts = {row['pair_type']: row['count'] for row in cursor}

            # Get label statistics
            cursor.execute(f"""
                SELECT
                    is_same_segment_label,
                    is_same_file_label,
                    is_same_code_type,
                    COUNT(*) as count
                FROM {self.pairs_table}
                WHERE experiment_id = %s
                GROUP BY is_same_segment_label, is_same_file_label, is_same_code_type
            """, (self.experiment_id,))

            label_stats = {}
            for row in cursor:
                key = f"seg_{'same' if row['is_same_segment_label'] else 'diff'}_file_{'same' if row['is_same_file_label'] else 'diff'}_type_{'same' if row['is_same_code_type'] else 'diff'}"
                label_stats[key] = row['count']

            # Get code type distribution
            cursor.execute(f"""
                SELECT
                    segment_code_type_1,
                    segment_code_type_2,
                    COUNT(*) as count
                FROM {self.pairs_table}
                WHERE experiment_id = %s
                GROUP BY segment_code_type_1, segment_code_type_2
                ORDER BY count DESC
                LIMIT 10
            """, (self.experiment_id,))

            code_type_pairs = [f"{row['segment_code_type_1']}-{row['segment_code_type_2']}: {row['count']}"
                              for row in cursor]

            # Get overall statistics
            cursor.execute(f"""
                SELECT
                    COUNT(*) as total_pairs,
                    COUNT(DISTINCT segment_id_1) + COUNT(DISTINCT segment_id_2) as total_segment_refs,
                    COUNT(DISTINCT file_id_1) + COUNT(DISTINCT file_id_2) as total_file_refs,
                    COUNT(CASE WHEN is_same_segment_label THEN 1 END) as same_segment_label_count,
                    COUNT(CASE WHEN is_same_file_label THEN 1 END) as same_file_label_count,
                    COUNT(CASE WHEN is_same_code_type THEN 1 END) as same_code_type_count
                FROM {self.pairs_table}
                WHERE experiment_id = %s
            """, (self.experiment_id,))

            overall = cursor.fetchone()

            return {
                'total_pairs': overall['total_pairs'],
                'unique_segments': overall['total_segment_refs'] // 2,  # Approximate
                'unique_files': overall['total_file_refs'] // 2,  # Approximate
                'type_distribution': type_counts,
                'label_combinations': label_stats,
                'same_segment_label_pairs': overall['same_segment_label_count'],
                'same_file_label_pairs': overall['same_file_label_count'],
                'same_code_type_pairs': overall['same_code_type_count'],
                'top_code_type_pairs': code_type_pairs
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


def main():
    """Test the segment pair generator"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate segment pairs for ML training')
    parser.add_argument('experiment_id', type=int, help='Experiment ID')
    parser.add_argument('--strategy', default='match_lengths_all_combinations',
                       choices=['all_combinations', 'match_lengths_all_combinations', 'balanced', 'code_type_balanced', 'random_sample'],
                       help='Pairing strategy')
    parser.add_argument('--max-pairs-per-segment', type=int, default=None,
                       help='Maximum pairs per segment (for sampling strategies)')
    parser.add_argument('--same-label-ratio', type=float, default=0.5,
                       help='Ratio of same-label pairs (for balanced strategy)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )

    # Database config
    import psycopg2
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )

    try:
        generator = ExperimentSegmentPairGeneratorV2(args.experiment_id, conn)
        result = generator.generate_pairs(
            strategy=args.strategy,
            max_pairs_per_segment=args.max_pairs_per_segment,
            same_label_ratio=args.same_label_ratio,
            seed=args.seed
        )

        if result['success']:
            logger.info("\nPair generation complete!")
            logger.info(f"Total pairs: {result['total_pairs']}")

            stats = result.get('statistics', {})
            if stats:
                logger.info("\nStatistics:")
                logger.info(f"  Unique segments: {stats.get('unique_segments', 'N/A')}")
                logger.info(f"  Unique files: {stats.get('unique_files', 'N/A')}")
                logger.info(f"  Same segment label: {stats.get('same_segment_label_pairs', 0)}")
                logger.info(f"  Same file label: {stats.get('same_file_label_pairs', 0)}")
                logger.info(f"  Same code type: {stats.get('same_code_type_pairs', 0)}")

                if 'type_distribution' in stats:
                    logger.info("\nPair type distribution:")
                    for ptype, count in stats['type_distribution'].items():
                        logger.info(f"  {ptype}: {count}")

                if 'top_code_type_pairs' in stats:
                    logger.info("\nTop code type pairs:")
                    for pair in stats['top_code_type_pairs']:
                        logger.info(f"  {pair}")
        else:
            logger.error(f"Pair generation failed: {result.get('error', 'Unknown error')}")
            return 1

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())