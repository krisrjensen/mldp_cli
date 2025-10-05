#!/usr/bin/env python3
"""
Filename: experiment_segment_fileset_generator_v2.py
Author(s): Kristophor Jensen
Date Created: 20250920_201500
Date Revised: 20251005_182500
File version: 1.1.0.0
Description: Generate physical segment files compatible with v2 segment selector
             with database-driven amplitude processing support
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
import importlib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentSegmentFilesetGeneratorV2:
    """
    Generate segment filesets compatible with v2 segment selector.

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

        # Load amplitude processing methods (cached for performance)
        self.amplitude_methods = self.get_experiment_amplitude_methods()
        expected_columns = 2 + (2 * len(self.amplitude_methods))

        logger.info(f"Initialized generator for experiment {experiment_id}")
        logger.info(f"Output path: {self.segment_path}")
        logger.info(f"Amplitude methods: {len(self.amplitude_methods)} configured")
        logger.info(f"Expected columns per segment file: {expected_columns}")

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
            # Convert datetime objects to strings for JSON serialization
            stats_copy = self.stats.copy()
            if stats_copy.get('start_time'):
                stats_copy['start_time'] = stats_copy['start_time'].isoformat()
            if stats_copy.get('end_time'):
                stats_copy['end_time'] = stats_copy['end_time'].isoformat()

            with open(self.progress_file, 'w') as f:
                json.dump({
                    'completed': list(self.completed),
                    'stats': stats_copy,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")

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

            # Get segments with all necessary information from data_segments
            query = f"""
                SELECT
                    st.segment_id,
                    st.file_id,
                    st.segment_index,
                    st.segment_code_type,
                    st.segment_code_number,
                    st.segment_label_id,
                    st.file_label_id,
                    ds.beginning_index as segment_start,
                    ds.beginning_index + ds.segment_length as segment_end,
                    ds.segment_length,
                    f.binary_data_path
                FROM {table_name} st
                JOIN data_segments ds ON st.segment_id = ds.segment_id
                JOIN files f ON st.file_id = f.file_id
                WHERE st.experiment_id = %s
                ORDER BY st.file_id, st.segment_index
            """

            cursor.execute(query, (self.experiment_id,))
            segments = [dict(row) for row in cursor]

            logger.info(f"Found {len(segments)} segments to process")

            # Log segment code type distribution
            code_types = {}
            for seg in segments:
                code_type = seg.get('segment_code_type', 'Unknown')
                code_types[code_type] = code_types.get(code_type, 0) + 1

            logger.info(f"Segment code type distribution:")
            for code_type, count in sorted(code_types.items()):
                logger.info(f"  {code_type}: {count}")

            return segments

        finally:
            cursor.close()
            conn.close()

    def load_source_data(self, file_id: int, data_type: str = 'RAW') -> Optional[np.ndarray]:
        """Load source data file from appropriate folder based on data type"""
        try:
            # RAW data comes from fileset folder
            if data_type == 'RAW':
                file_path = self.fileset_path / f"{file_id:08d}.npy"
            # All ADC types come from adc_data folder (pre-quantized with real system parameters)
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

        # Handle different data formats
        # RAW data: 2 channels (voltage, current)
        # ADC data: 4 channels (may include additional quantization info)
        if segment.ndim == 1:
            # If 1D, duplicate to create 2 channels
            segment = np.column_stack([segment, segment])
        elif segment.shape[1] == 4:
            # ADC data with 4 channels - use first 2 for voltage/current
            segment = segment[:, :2]
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

    def get_adc_data(self, source_data: np.ndarray, data_type: str) -> np.ndarray:
        """
        Get ADC data for the specific data type.
        Note: ADC data is pre-quantized and stored in adc_data folder,
        so we don't convert here - we load the correct file instead.
        This function is kept for potential future use.
        """
        # This function is no longer needed as we load from the correct source
        # But keeping it as placeholder for potential post-processing
        return source_data

    def get_segment_file_path(self, segment_info: Dict, original_size: int,
                             resulting_size: int, data_type: str, decimation: int) -> Path:
        """Generate the output file path following the directory structure

        Naming convention: SID{segment_id:08d}_F{file_id:08d}_D{decimation:06d}_T{data_type}_S{original_size}_R{resulting_size}.npy
        """
        segment_id = segment_info['segment_id']
        file_id = segment_info['file_id']

        # Build directory structure
        size_dir = f"S{resulting_size:06d}"
        type_dir = f"T{data_type}"
        dec_dir = f"D{decimation:06d}"

        # Create full directory path
        full_dir = self.segment_path / size_dir / type_dir / dec_dir
        full_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with all metadata
        # SID = Segment ID, F = File, D = Decimation, T = Type, S = original Size, R = Resulting size
        filename = f"SID{segment_id:08d}_F{file_id:08d}_D{decimation:06d}_T{data_type}_S{original_size:06d}_R{resulting_size:06d}.npy"

        return full_dir / filename

    def process_segment(self, segment_info: Dict, data_types: List[str],
                       decimations: List[int]) -> Dict[str, int]:
        """Process a single segment with all variations"""
        results = {'created': 0, 'skipped': 0, 'failed': 0}

        file_id = segment_info['file_id']
        segment_id = segment_info['segment_id']
        start = segment_info['segment_start']
        end = segment_info['segment_end']

        # Skip if already processed
        segment_key = f"{segment_id}_{file_id}"
        if segment_key in self.completed:
            results['skipped'] += 1
            return results

        # Process each data type
        for data_type in data_types:
            # Load the correct source data based on data type
            # RAW from fileset/, ADC* from adc_data/
            source_data = self.load_source_data(file_id, data_type)
            if source_data is None:
                logger.error(f"Failed to load {data_type} data for file {file_id}")
                results['failed'] += 1
                continue

            # Extract segment from the source data
            try:
                segment_data = self.extract_segment(source_data, start, end)
                original_size = segment_data.shape[0]
            except Exception as e:
                logger.error(f"Failed to extract segment {segment_id} from {data_type}: {e}")
                results['failed'] += 1
                continue

            # Process each decimation
            for decimation in decimations:
                # Apply decimation
                if decimation > 0:
                    dec_data = self.apply_decimation(segment_data, decimation)
                    actual_size = dec_data.shape[0]
                else:
                    dec_data = segment_data
                    actual_size = original_size

                # Apply amplitude processing to create multi-column output
                # Format: [raw_voltage, raw_current, method1_voltage, method1_current, method2_voltage, method2_current, ...]
                try:
                    output_columns = []

                    # Always start with raw voltage/current pair
                    output_columns.extend([dec_data[:, 0], dec_data[:, 1]])

                    # Add each amplitude processing method
                    for method_config in self.amplitude_methods:
                        amplitude_processed = self._apply_amplitude_processing(dec_data, method_config)
                        output_columns.extend([amplitude_processed[:, 0], amplitude_processed[:, 1]])

                    # Combine into final array
                    final_data = np.column_stack(output_columns)

                    # Validate shape
                    expected_cols = 2 + (2 * len(self.amplitude_methods))
                    if final_data.shape[1] != expected_cols:
                        logger.error(f"Segment {segment_id}: Expected {expected_cols} columns, got {final_data.shape[1]}")
                        results['failed'] += 1
                        continue

                except Exception as e:
                    logger.error(f"Failed to apply amplitude processing to segment {segment_id}: {e}")
                    results['failed'] += 1
                    continue

                # Generate output path
                output_path = self.get_segment_file_path(
                    segment_info, original_size, actual_size, data_type, decimation
                )

                # Save segment
                try:
                    np.save(output_path, final_data)
                    results['created'] += 1
                except Exception as e:
                    logger.error(f"Failed to save segment {output_path}: {e}")
                    results['failed'] += 1

        # Mark as completed
        self.completed.add(segment_key)

        return results

    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration from database"""
        conn = self.connect_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Get data types from ml_experiments_data_types
            cursor.execute("""
                SELECT dt.data_type_name
                FROM ml_experiments_data_types edt
                JOIN ml_data_types_lut dt ON edt.data_type_id = dt.data_type_id
                WHERE edt.experiment_id = %s
                ORDER BY dt.data_type_id
            """, (self.experiment_id,))

            data_types = []
            for row in cursor:
                dt = row['data_type_name'].upper()
                # Handle 'raw' -> 'RAW' properly
                if dt == 'RAW':
                    data_types.append('RAW')
                else:
                    # Convert adc8 -> ADC8, etc.
                    data_types.append(dt.upper())

            # Get decimation factors from ml_experiment_decimation_junction
            cursor.execute("""
                SELECT d.decimation_factor
                FROM ml_experiment_decimation_junction ed
                JOIN ml_experiment_decimation_lut d ON ed.decimation_id = d.decimation_id
                WHERE ed.experiment_id = %s
                ORDER BY d.decimation_factor
            """, (self.experiment_id,))

            decimations = [row['decimation_factor'] for row in cursor]

            return {
                'data_types': data_types if data_types else ['RAW'],
                'decimations': decimations if decimations else [0]
            }

        finally:
            cursor.close()
            conn.close()

    def get_experiment_amplitude_methods(self) -> List[Dict[str, Any]]:
        """Get amplitude processing methods configured for this experiment from PostgreSQL"""
        conn = self.connect_db()
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        try:
            # Query PostgreSQL database for amplitude methods
            cursor.execute("""
                SELECT
                    eam.method_id,
                    am.method_name,
                    am.function_name,
                    am.function_args
                FROM ml_experiments_amplitude_methods eam
                JOIN ml_amplitude_normalization_lut am ON eam.method_id = am.method_id
                WHERE eam.experiment_id = %s
                ORDER BY eam.method_id
            """, (self.experiment_id,))

            amplitude_methods = []
            for row in cursor:
                # Parse JSON args if present
                args = json.loads(row['function_args']) if row['function_args'] else {}

                amplitude_methods.append({
                    'method_id': row['method_id'],
                    'name': row['method_name'],
                    'function': row['function_name'],
                    'args': args
                })

            if not amplitude_methods:
                logger.warning(f"No amplitude methods configured for experiment {self.experiment_id}")
            else:
                logger.info(f"Loaded {len(amplitude_methods)} amplitude methods: {[m['name'] for m in amplitude_methods]}")

            return amplitude_methods

        finally:
            cursor.close()
            conn.close()

    def _apply_amplitude_processing(self, data: np.ndarray, method_config: Dict[str, Any]) -> np.ndarray:
        """Apply amplitude processing method using DATABASE-DRIVEN dynamic loading"""
        function_name = method_config["function"]
        args = method_config["args"]

        try:
            # DATABASE-DRIVEN: Load functions dynamically based on function_name
            if function_name.startswith('sklearn.'):
                return self._apply_sklearn_function(data, function_name, args)
            elif function_name == "chunk_standardize":
                return self._apply_chunk_processing(data, args)
            else:
                # Future: Load custom functions from ml_code library
                logger.warning(f"Unknown amplitude processing function: {function_name}, returning original data")
                return data

        except Exception as e:
            logger.error(f"Error applying amplitude processing {function_name}: {e}")
            return data  # Fallback to original data

    def _apply_sklearn_function(self, data: np.ndarray, function_name: str, args: Dict) -> np.ndarray:
        """Dynamically load and apply sklearn functions"""
        try:
            # Split module and class name
            module_path, class_name = function_name.rsplit('.', 1)

            # Dynamic import
            module = importlib.import_module(module_path)
            func_class = getattr(module, class_name)

            # Handle special parameter conversions
            processed_args = args.copy()
            if 'quantile_range' in processed_args and isinstance(processed_args['quantile_range'], list):
                processed_args['quantile_range'] = tuple(processed_args['quantile_range'])

            # Create and apply scaler
            scaler = func_class(**processed_args)

            # Special handling for PowerTransformer
            if class_name == "PowerTransformer":
                try:
                    return scaler.fit_transform(data)
                except ValueError as e:
                    logger.warning(f"PowerTransformer failed: {e}, returning normalized data")
                    return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)
            else:
                return scaler.fit_transform(data)

        except Exception as e:
            logger.error(f"Error in sklearn function {function_name}: {e}")
            return data

    def _apply_chunk_processing(self, data: np.ndarray, args: Dict) -> np.ndarray:
        """Apply chunk-based processing with fallback implementation"""
        chunk_size = args.get("chunk_size", 8192)

        # Fallback chunk standardization implementation
        try:
            rows, cols = data.shape
            if rows < chunk_size:
                logger.warning(f"Data too small for chunk size {chunk_size}, using standard scaling")
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                return scaler.fit_transform(data)

            # Chunk the data
            chunks = []
            for i in range(0, rows, chunk_size):
                chunk = data[i:i+chunk_size]
                if chunk.shape[0] >= 100:  # Only process chunks with enough data points
                    # Standardize each chunk independently
                    chunk_mean = np.mean(chunk, axis=0)
                    chunk_std = np.std(chunk, axis=0)
                    chunk_std[chunk_std == 0] = 1  # Avoid division by zero
                    standardized_chunk = (chunk - chunk_mean) / chunk_std
                    chunks.append(standardized_chunk)
                else:
                    chunks.append(chunk)  # Keep small chunks as-is

            return np.vstack(chunks) if chunks else data

        except Exception as e:
            logger.error(f"Error in chunk processing: {e}")
            return data

    def generate_segment_fileset(self,
                                data_types: List[str] = None,
                                decimations: List[int] = None,
                                max_segments: int = None,
                                parallel_workers: int = 1) -> Dict[str, Any]:
        """
        Generate segment fileset for the experiment

        Args:
            data_types: List of data types to generate (RAW, ADC14, ADC12, etc.)
            decimations: List of decimation factors (0 = no decimation)
            max_segments: Maximum number of segments to process
            parallel_workers: Number of parallel workers

        Returns:
            Dictionary with generation statistics
        """
        self.stats['start_time'] = datetime.now()

        # Get experiment configuration if not provided
        if data_types is None or decimations is None:
            config = self.get_experiment_config()
            if data_types is None:
                data_types = config['data_types']
            if decimations is None:
                decimations = config['decimations']

        logger.info(f"Starting segment fileset generation for experiment {self.experiment_id}")
        logger.info(f"Using experiment configuration:")
        logger.info(f"  Data types: {data_types}")
        logger.info(f"  Decimations: {decimations}")
        logger.info(f"  Total variations: {len(data_types)} types × {len(decimations)} decimations = {len(data_types) * len(decimations)}")

        # Get segments to process
        segments = self.get_segments_to_generate()
        if not segments:
            logger.warning("No segments to process")
            return self.stats

        # Limit segments if requested
        if max_segments:
            segments = segments[:max_segments]

        logger.info(f"Processing {len(segments)} segments")

        # Process segments with progress bar
        total_operations = len(segments) * len(data_types) * len(decimations)

        with tqdm(total=total_operations, desc="Generating segments") as pbar:
            for segment in segments:
                results = self.process_segment(segment, data_types, decimations)

                self.stats['files_created'] += results['created']
                self.stats['files_skipped'] += results['skipped']
                self.stats['files_failed'] += results['failed']

                if results['created'] > 0:
                    self.stats['segments_processed'] += 1

                # Update progress bar
                pbar.update(len(data_types) * len(decimations))

                # Save progress periodically
                if self.stats['segments_processed'] % 100 == 0:
                    self.save_progress()

        # Final statistics
        self.stats['end_time'] = datetime.now()
        duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()

        logger.info(f"\n{'='*50}")
        logger.info(f"SEGMENT FILESET GENERATION COMPLETE")
        logger.info(f"Experiment: {self.experiment_id}")
        logger.info(f"Duration: {duration:.2f} seconds")
        logger.info(f"Segments processed: {self.stats['segments_processed']}")
        logger.info(f"Files created: {self.stats['files_created']}")
        logger.info(f"Files skipped: {self.stats['files_skipped']}")
        logger.info(f"Files failed: {self.stats['files_failed']}")
        logger.info(f"Output directory: {self.segment_path}")

        # Save final progress
        self.save_progress()

        return self.stats

    def validate_output(self) -> Dict[str, Any]:
        """Validate generated segment files"""
        validation_results = {
            'total_files': 0,
            'valid_files': 0,
            'corrupt_files': 0,
            'missing_files': 0,
            'size_distribution': {}
        }

        logger.info("Validating generated segment files...")

        # Walk through output directory
        for npy_file in self.segment_path.glob("**/*.npy"):
            validation_results['total_files'] += 1

            try:
                # Try loading the file
                data = np.load(npy_file)
                validation_results['valid_files'] += 1

                # Track size distribution
                size = data.shape[0]
                if size not in validation_results['size_distribution']:
                    validation_results['size_distribution'][size] = 0
                validation_results['size_distribution'][size] += 1

            except Exception as e:
                logger.error(f"Corrupt file {npy_file}: {e}")
                validation_results['corrupt_files'] += 1

        logger.info(f"Validation complete:")
        logger.info(f"  Total files: {validation_results['total_files']}")
        logger.info(f"  Valid files: {validation_results['valid_files']}")
        logger.info(f"  Corrupt files: {validation_results['corrupt_files']}")

        return validation_results


def main():
    """Test the segment fileset generator"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate segment fileset for ML training')
    parser.add_argument('experiment_id', type=int, help='Experiment ID')
    parser.add_argument('--data-types', default='RAW',
                       help='Comma-separated data types (RAW,ADC14,ADC12,ADC10,ADC8)')
    parser.add_argument('--decimations', default='0',
                       help='Comma-separated decimation factors (0=none)')
    parser.add_argument('--max-segments', type=int, default=None,
                       help='Maximum segments to process')
    parser.add_argument('--validate', action='store_true',
                       help='Validate output files after generation')

    args = parser.parse_args()

    # Parse arguments
    data_types = args.data_types.split(',')
    decimations = [int(d) for d in args.decimations.split(',')]

    # Database config
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'arc_detection',
        'user': 'kjensen'
    }

    # Create generator
    generator = ExperimentSegmentFilesetGeneratorV2(args.experiment_id, db_config)

    # Generate fileset
    result = generator.generate_segment_fileset(
        data_types=data_types,
        decimations=decimations,
        max_segments=args.max_segments
    )

    # Validate if requested
    if args.validate:
        validation = generator.validate_output()
        print(f"\nValidation results: {validation}")

    return 0 if result['files_created'] > 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())