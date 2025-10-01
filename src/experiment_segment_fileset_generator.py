#!/usr/bin/env python3
"""
Filename: experiment_segment_fileset_generator.py
Author(s): Kristophor Jensen
Date Created: 20250920_150000
Date Revised: 20250920_150000
File version: 0.0.0.1
Description: Generate physical segment files from raw data for any experiment
"""

import os
import sys
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentSegmentFilesetGenerator:
    """
    Generate segment filesets with proper directory structure.

    Directory structure:
    experiment{NNN}/segment_files/
    ├── S{size}/                    # Segment size (e.g., S262144)
    │   ├── T{type}/                # Data type (e.g., TRAW, TADC14)
    │   │   ├── D{decimation}/      # Decimation factor (e.g., D000000)
    │   │   │   ├── {segment_files}.npy  # Actual segment files

    Parallel structure for features:
    experiment{NNN}/feature_files/
    └── [same structure as segment_files]
    """

    def __init__(self, experiment_id: int, db_config: Dict[str, Any]):
        """Initialize the segment fileset generator"""
        self.experiment_id = experiment_id
        self.db_config = db_config

        # Base paths
        self.base_path = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}')
        self.segment_path = self.base_path / 'segment_files'
        self.feature_path = self.base_path / 'feature_files'

        # Source data paths
        self.fileset_path = Path('/Volumes/ArcData/V3_database/fileset')
        self.adc_path = Path('/Volumes/ArcData/V3_database/adc_data')

        # Create output directories
        self.segment_path.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats = {
            'files_created': 0,
            'files_skipped': 0,
            'files_failed': 0,
            'segments_processed': 0,
            'start_time': None,
            'end_time': None
        }

        # Progress tracking
        self.progress_file = self.segment_path / 'generation_progress.json'
        self.completed = self.load_progress()

        logger.info(f"Initialized generator for experiment {experiment_id}")
        logger.info(f"Output path: {self.segment_path}")

    def connect_db(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_config)

    def load_progress(self) -> set:
        """Load progress from checkpoint file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('completed', []))
            except Exception as e:
                logger.warning(f"Could not load progress file: {e}")
        return set()

    def save_progress(self):
        """Save progress to checkpoint file"""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump({
                    'completed': list(self.completed),
                    'stats': self.stats,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")

    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration from database"""
        conn = self.connect_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Get ML experiment configuration
            cursor.execute("""
                SELECT * FROM ml_experiments
                WHERE experiment_id = %s
            """, (self.experiment_id,))

            config = cursor.fetchone()
            if not config:
                raise ValueError(f"No configuration found for experiment {self.experiment_id}")

            return dict(config)

        finally:
            cursor.close()
            conn.close()

    def get_segments_to_generate(self) -> List[Dict]:
        """Get segments that need to be generated from training data"""
        conn = self.connect_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            table_name = f"experiment_{self.experiment_id:03d}_segment_training_data"

            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (table_name,))

            if not cursor.fetchone()['exists']:
                logger.warning(f"Table {table_name} does not exist. Run select-segments first.")
                return []

            # Get segments with all necessary information
            query = f"""
                SELECT
                    st.segment_id,
                    st.file_id,
                    st.segment_index,
                    s.segment_start,
                    s.segment_end,
                    s.segment_length,
                    s.segment_label_id,
                    sl.label_name as segment_label,
                    f.file_path
                FROM {table_name} st
                JOIN segments s ON st.segment_id = s.segment_id
                JOIN segment_labels sl ON s.segment_label_id = sl.label_id
                JOIN files f ON st.file_id = f.file_id
                WHERE st.experiment_id = %s
                ORDER BY st.file_id, st.segment_index
            """

            cursor.execute(query, (self.experiment_id,))
            segments = [dict(row) for row in cursor]

            logger.info(f"Found {len(segments)} segments to process")
            return segments

        finally:
            cursor.close()
            conn.close()

    def get_feature_sets(self) -> List[Dict]:
        """Get configured feature sets for the experiment"""
        conn = self.connect_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            cursor.execute("""
                SELECT
                    efs.*,
                    fs.feature_set_name,
                    fs.category
                FROM ml_experiments_feature_sets efs
                JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
                WHERE efs.experiment_id = %s
                ORDER BY efs.priority_order
            """, (self.experiment_id,))

            return [dict(row) for row in cursor]

        finally:
            cursor.close()
            conn.close()

    def load_source_data(self, file_id: int, data_type: str = 'RAW') -> Optional[np.ndarray]:
        """Load source data file"""
        try:
            if data_type == 'RAW':
                file_path = self.fileset_path / f"{file_id:08d}.npy"
            else:
                file_path = self.adc_path / f"{file_id:08d}.npy"

            if not file_path.exists():
                logger.error(f"Source file not found: {file_path}")
                return None

            data = np.load(file_path)
            return data

        except Exception as e:
            logger.error(f"Failed to load file {file_id}: {e}")
            return None

    def extract_segment(self, data: np.ndarray, start: int, end: int) -> np.ndarray:
        """Extract segment from source data"""
        segment = data[start:end]

        # Ensure 2D array with 2 channels
        if segment.ndim == 1:
            # If 1D, duplicate to create 2 channels
            segment = np.column_stack([segment, segment])
        elif segment.shape[1] < 2:
            # If less than 2 channels, duplicate first channel
            segment = np.column_stack([segment[:, 0], segment[:, 0]])
        else:
            # Use first 2 channels (voltage, current)
            segment = segment[:, :2]

        return segment

    def apply_decimation(self, data: np.ndarray, decimation: int) -> np.ndarray:
        """Apply decimation (downsampling)"""
        if decimation <= 0:
            return data

        # Decimation = N means keep every (N+1)th sample
        step = decimation + 1
        return data[::step]

    def apply_data_type_conversion(self, data: np.ndarray, source_type: str, target_type: str) -> np.ndarray:
        """Convert data between different ADC bit depths"""
        if target_type == 'RAW' or source_type == target_type:
            return data

        # Map data types to bit depths
        bit_depths = {
            'ADC14': 14, 'ADC12': 12, 'ADC10': 10,
            'ADC8': 8, 'ADC6': 6, 'ADC4': 4
        }

        if source_type not in bit_depths or target_type not in bit_depths:
            logger.warning(f"Unknown data type conversion: {source_type} -> {target_type}")
            return data

        source_bits = bit_depths[source_type]
        target_bits = bit_depths[target_type]

        if source_bits > target_bits:
            # Downsample: right-shift
            shift = source_bits - target_bits
            converted = (data >> shift).astype(np.uint16 if target_bits > 8 else np.uint8)
        else:
            # Upsample: left-shift
            shift = target_bits - source_bits
            converted = np.clip(data << shift, 0, (1 << target_bits) - 1)
            converted = converted.astype(np.uint16 if target_bits > 8 else np.uint8)

        return converted

    def apply_amplitude_normalization(self, data: np.ndarray, methods: List[str]) -> np.ndarray:
        """Apply amplitude normalization methods"""
        # Clean data
        data = np.nan_to_num(data, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)

        # Start with original data
        result_columns = [data]

        # Apply each normalization method
        for method in methods:
            try:
                if method.upper() in ['STANDARDIZE', 'STANDARD']:
                    scaler = StandardScaler()
                    normalized = scaler.fit_transform(data)
                elif method.upper() in ['MINMAX', 'MIN_MAX']:
                    scaler = MinMaxScaler()
                    normalized = scaler.fit_transform(data)
                elif method.upper() in ['ROBUST']:
                    scaler = RobustScaler()
                    normalized = scaler.fit_transform(data)
                else:
                    logger.warning(f"Unknown normalization method: {method}")
                    normalized = data

                result_columns.append(normalized)

            except Exception as e:
                logger.warning(f"Failed to apply {method}: {e}")
                result_columns.append(data)

        # Combine all columns
        return np.column_stack(result_columns)

    def get_segment_file_path(self, segment_id: int, file_id: int, size: int,
                             data_type: str, decimation: int) -> Path:
        """Generate the output file path following the directory structure"""
        # Build directory structure
        size_dir = f"S{size:06d}"
        type_dir = f"T{data_type}"
        dec_dir = f"D{decimation:06d}"

        # Create full directory path
        full_dir = self.segment_path / size_dir / type_dir / dec_dir
        full_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"{segment_id:08d}_{file_id:08d}_D{decimation:06d}_T{data_type}_S{size:06d}.npy"

        return full_dir / filename

    def process_segment(self, segment_info: Dict, data_types: List[str],
                       decimations: List[int], amplitude_methods: List[str] = None) -> Dict[str, int]:
        """Process a single segment with all variations"""
        results = {'created': 0, 'skipped': 0, 'failed': 0}

        file_id = segment_info['file_id']
        segment_id = segment_info['segment_id']
        start = segment_info['segment_start']
        end = segment_info['segment_end']

        # Load source data (RAW format)
        source_data = self.load_source_data(file_id, 'RAW')
        if source_data is None:
            results['failed'] += 1
            return results

        # Extract segment
        segment_data = self.extract_segment(source_data, start, end)

        # Process each data type
        for data_type in data_types:
            # Convert data type if needed
            if data_type != 'RAW':
                type_data = self.apply_data_type_conversion(segment_data, 'RAW', data_type)
            else:
                type_data = segment_data

            # Process each decimation
            for decimation in decimations:
                # Create progress key
                progress_key = f"{segment_id}_{data_type}_{decimation}"

                # Skip if already processed
                if progress_key in self.completed:
                    results['skipped'] += 1
                    continue

                # Apply decimation
                decimated_data = self.apply_decimation(type_data, decimation)

                # Apply amplitude normalization if requested
                if amplitude_methods:
                    processed_data = self.apply_amplitude_normalization(decimated_data, amplitude_methods)
                else:
                    processed_data = decimated_data

                # Get output path
                output_path = self.get_segment_file_path(
                    segment_id, file_id,
                    processed_data.shape[0],
                    data_type, decimation
                )

                # Check if file exists
                if output_path.exists():
                    results['skipped'] += 1
                    self.completed.add(progress_key)
                    continue

                try:
                    # Save segment file
                    np.save(output_path, processed_data)
                    results['created'] += 1
                    self.completed.add(progress_key)

                    # Log progress
                    if results['created'] % 100 == 0:
                        logger.info(f"Created {results['created']} files...")

                except Exception as e:
                    logger.error(f"Failed to save {output_path}: {e}")
                    results['failed'] += 1

        return results

    def generate_segment_fileset(self,
                                data_types: List[str] = None,
                                decimations: List[int] = None,
                                amplitude_methods: List[str] = None,
                                max_segments: int = None,
                                parallel_workers: int = 1) -> Dict[str, Any]:
        """
        Generate segment fileset for the experiment

        Args:
            data_types: List of data types to generate (RAW, ADC14, ADC12, etc.)
            decimations: List of decimation factors (0 = no decimation)
            amplitude_methods: List of normalization methods to apply
            max_segments: Maximum number of segments to process
            parallel_workers: Number of parallel workers

        Returns:
            Dictionary with generation statistics
        """
        self.stats['start_time'] = datetime.now()

        # Default values
        if data_types is None:
            data_types = ['RAW']
        if decimations is None:
            decimations = [0]  # No decimation by default

        logger.info(f"Starting segment fileset generation")
        logger.info(f"Data types: {data_types}")
        logger.info(f"Decimations: {decimations}")
        logger.info(f"Amplitude methods: {amplitude_methods}")

        # Get segments to process
        segments = self.get_segments_to_generate()

        if not segments:
            logger.warning("No segments to process")
            return self.stats

        # Limit segments if requested
        if max_segments:
            segments = segments[:max_segments]

        logger.info(f"Processing {len(segments)} segments")

        # Process segments
        with tqdm(total=len(segments) * len(data_types) * len(decimations)) as pbar:
            for segment in segments:
                results = self.process_segment(segment, data_types, decimations, amplitude_methods)

                self.stats['files_created'] += results['created']
                self.stats['files_skipped'] += results['skipped']
                self.stats['files_failed'] += results['failed']
                self.stats['segments_processed'] += 1

                pbar.update(len(data_types) * len(decimations))

                # Save progress periodically
                if self.stats['segments_processed'] % 10 == 0:
                    self.save_progress()

        # Final save
        self.save_progress()

        self.stats['end_time'] = datetime.now()
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        # Summary
        logger.info(f"\nGeneration complete!")
        logger.info(f"Files created: {self.stats['files_created']}")
        logger.info(f"Files skipped: {self.stats['files_skipped']}")
        logger.info(f"Files failed: {self.stats['files_failed']}")
        logger.info(f"Total segments: {self.stats['segments_processed']}")
        logger.info(f"Duration: {duration:.2f} seconds")

        return {
            'success': True,
            'stats': self.stats,
            'duration': duration,
            'output_path': str(self.segment_path)
        }


def main():
    """Test the segment fileset generator"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate segment fileset for experiment')
    parser.add_argument('experiment_id', type=int, help='Experiment ID')
    parser.add_argument('--data-types', nargs='+', default=['RAW'],
                       help='Data types (RAW, ADC14, ADC12, ADC10, ADC8, ADC6)')
    parser.add_argument('--decimations', nargs='+', type=int, default=[0],
                       help='Decimation factors (0 = no decimation)')
    parser.add_argument('--amplitude-methods', nargs='+',
                       help='Amplitude normalization methods (STANDARDIZE, MINMAX, ROBUST)')
    parser.add_argument('--max-segments', type=int,
                       help='Maximum segments to process')

    args = parser.parse_args()

    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'arc_detection',
        'user': 'kjensen'
    }

    # Create generator
    generator = ExperimentSegmentFilesetGenerator(args.experiment_id, db_config)

    # Generate fileset
    result = generator.generate_segment_fileset(
        data_types=args.data_types,
        decimations=args.decimations,
        amplitude_methods=args.amplitude_methods,
        max_segments=args.max_segments
    )

    return 0 if result['success'] else 1


if __name__ == "__main__":
    sys.exit(main())