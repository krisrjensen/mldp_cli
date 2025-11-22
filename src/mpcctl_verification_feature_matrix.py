#!/usr/bin/env python3
"""
Filename: mpcctl_verification_feature_matrix.py
Author(s): Kristophor Jensen
Date Created: 20251112_120000
Date Revised: 20251119_000001
File version: 1.0.0.10
Description: MPCCTL-based verification feature matrix generator
             Generates pre-computed feature matrices for classifier verification testing

Changelog:
v1.0.0.10 - Changed to use ALL segments from training files
  - Query now gets ALL segments from experiment_042_file_training_data
  - Gives 7,500 segments with all 8 compound classes represented
  - Provides dense visualizations while maintaining class coverage

v1.0.0.9 - Fixed verification segment query to use classifier data_splits table
  - Changed from experiment_042_file_training_data (6,600 segs, 5 classes)
  - To experiment_042_classifier_001_data_splits (90 segs, 8 classes)
  - Ensures all compound classes from training set are represented in verification
  - Fixes missing arc.arc_initiation, negative_transient.negative_load_transient,
    parallel_motor_arc.parallel_motor_arc_transient classes

v1.0.0.8 - Replaced hardcoded feature extraction with FeatureFunctionLoader
  - Removed 70 lines of hardcoded if/elif chains for feature extraction
  - Integrated database-driven FeatureFunctionLoader for all feature computation
  - Channel routing, function imports, and feature logic now 100% database-driven
  - All function names loaded from ml_functions table (NO hardcoded names)
  - Simplifies maintenance: add new features via database only, NO code changes
  - Reduces code from ~800 to ~730 lines by removing hardcoded logic

v1.0.0.7 - Fixed channel selection for voltage/current features
  - Implemented proper channel routing: channel 0=voltage, channel 1=current
  - Features with "calc_voltage" or "voltage" in name use voltage channel (0)
  - Features with "calc_current" or "current" in name use current channel (1)
  - Other features (e.g., volatility_dxdt_n1) default to current channel (1)
  - Eliminates warnings about unknown computation for voltage/current features

v1.0.0.6 - Fixed database query for feature metadata
  - Added computation_function and behavior_type to SELECT clause
  - Query was only selecting feature_id and feature_name, causing KeyError

v1.0.0.5 - Fixed import syntax error
  - Moved feature function imports to module level (from worker_function)
  - Python doesn't allow "import *" inside functions, only at module level
  - Removed duplicate import statements from worker_function

v1.0.0.4 - Fixed feature extraction implementation
  - Added feature metadata dict to store computation_function and behavior_type for each feature
  - Imported feature function modules (derivative, temporal, spectral, composite)
  - Replaced non-existent extract_single_feature() call with direct feature computation
  - Implemented feature computation logic supporting standard numpy functions and custom features
  - Added proper error handling with traceback logging for feature extraction failures

v1.0.0.3 - Fixed data type case mismatch
  - Added .upper() to data_type_map creation in worker and manager
  - Ensures database lowercase 'adc8' converts to segment_processor uppercase 'ADC8'

v1.0.0.2 - Fixed race condition
  - Added wait loop in worker_function for todo file creation
  - Workers now wait up to 30 seconds for manager to create todo file

v1.0.0.1 - Fixed argument parsing
  - Changed loop start from i=1 to i=0 to process all arguments correctly

v1.0.0.0 - Initial implementation
  - MPCCTL-based parallel architecture for verification feature generation
  - Distributes all configurations across workers in single job

ARCHITECTURE:
- Manager process coordinates parallel workers
- Worker processes build features for assigned work units
- Work units: (segment_id, data_type_id, decimation, amplitude_method) tuples
- ALL configurations distributed across workers in parallel
- Output: Consolidated feature matrix files (with batching support)
- Uses MPCCTL protocol with {PID}_todo.dat and {PID}_done.dat files
- Post-processing groups results by configuration and stitches into final files

Output Format:
- Structured numpy array with named fields
- Shape: (N_segments, M_features) where M = len(feature_ids)
- Filename: features_S{seg_size:06d}_D{dec:06d}_R{result_size:06d}_{dtype}_A{amp}.npy
- Batch files: features_S{}_D{}_R{}_{}_A{}_worker{:02d}_batch_{:04d}.npy
"""

import os
import sys
import time
import json
import logging
import psycopg2
import psycopg2.extras
import numpy as np
import multiprocessing as mp
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add ml_code to path for FeatureFunctionLoader
ml_code_path = Path(__file__).parent.parent.parent / 'ml_code' / 'src'
if str(ml_code_path) not in sys.path:
    sys.path.insert(0, str(ml_code_path))

# Import FeatureFunctionLoader for database-driven feature extraction
try:
    from feature_loader import FeatureFunctionLoader
except ImportError as e:
    logger.error(f"Could not import FeatureFunctionLoader: {e}")
    sys.exit(1)


class PostgreSQLConfig:
    """PostgreSQL database configuration."""

    def __init__(self, host='localhost', port=5432, database='arc_detection', user='kjensen'):
        self.host = host
        self.port = port
        self.database = database
        self.user = user

    def get_connection(self):
        """Create and return database connection."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user
        )


def worker_function(worker_id: int, experiment_id: int, segment_sizes: List[int],
                   feature_ids: List[int], mpcctl_dir: Path, output_dir: Path,
                   batch_size_mb: float, db_config: Dict,
                   raw_data_folder: str, adc_data_folder: str):
    """
    Worker process that generates feature vectors for assigned work units.

    Work units are tuples: (segment_id, data_type_id, decimation, amplitude_method)

    Args:
        worker_id: Unique worker ID
        experiment_id: Experiment ID
        segment_sizes: List of allowed segment sizes for filtering
        feature_ids: List of feature IDs to compute
        mpcctl_dir: Path to .mpcctl directory
        output_dir: Output directory for batch files
        batch_size_mb: Maximum batch file size in MB
        db_config: Database configuration dictionary
        raw_data_folder: Path to raw data folder
        adc_data_folder: Path to ADC data folder
    """
    pid = os.getpid()
    todo_file = mpcctl_dir / f"{pid}_todo.dat"
    done_file = mpcctl_dir / f"{pid}_done.dat"
    error_log = mpcctl_dir / f"{pid}_error.log"

    # Setup worker error logging
    worker_logger = logging.getLogger(f"worker_{pid}")
    worker_handler = logging.FileHandler(error_log)
    worker_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    worker_logger.addHandler(worker_handler)
    worker_logger.setLevel(logging.DEBUG)

    worker_logger.info(f"Worker {worker_id} (PID {pid}) starting")

    try:
        # Wait for todo file to be created by manager (with timeout)
        wait_time = 0
        max_wait = 30  # Maximum 30 seconds
        while not todo_file.exists() and wait_time < max_wait:
            time.sleep(0.1)
            wait_time += 0.1

        if not todo_file.exists():
            worker_logger.error(f"Worker {worker_id} (PID {pid}) timeout waiting for todo file: {todo_file}")
            return

        worker_logger.info(f"Worker {worker_id} (PID {pid}) found todo file after {wait_time:.1f}s")

        # Connect to database
        db_conn = psycopg2.connect(**db_config)
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Load work units from todo file
        # Format: segment_id,data_type_id,decimation,amplitude_method
        work_units = []
        with open(todo_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) == 4:
                        work_units.append((
                            int(parts[0]),  # segment_id
                            int(parts[1]),  # data_type_id
                            int(parts[2]),  # decimation
                            int(parts[3])   # amplitude_method
                        ))

        worker_logger.info(f"Loaded {len(work_units)} work units from {todo_file}")

        if not work_units:
            worker_logger.warning("No work units to process")
            return

        # Cache data type names (convert to uppercase for segment_processor compatibility)
        cursor.execute("SELECT data_type_id, data_type_name FROM ml_data_types_lut")
        data_type_map = {row['data_type_id']: row['data_type_name'].upper() for row in cursor.fetchall()}

        # Initialize FeatureFunctionLoader for database-driven feature extraction
        # This replaces 70 lines of hardcoded if/elif chains with database-driven architecture
        loader = FeatureFunctionLoader(db_conn, feature_ids=feature_ids)
        worker_logger.info(f"FeatureFunctionLoader initialized with {loader.feature_count} features")

        # Get feature names for structured array dtype
        feature_names = [loader.get_feature_metadata(fid)['feature_name'] for fid in feature_ids]

        # Create structured array dtype
        # Each row: segment_id (uint32), feature_1 (float32), feature_2 (float32), ...
        dtype_list = [('segment_id', np.uint32)]
        for fname in feature_names:
            # Sanitize feature name for numpy field (replace spaces/special chars with _)
            field_name = fname.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
            dtype_list.append((field_name, np.float32))

        structured_dtype = np.dtype(dtype_list)

        # Batch management per configuration
        # Key: (segment_size, data_type_id, decimation, amplitude_method)
        # Value: (batch_num, batch_rows list)
        config_batches = {}
        max_rows_per_batch = int((batch_size_mb * 1024 * 1024) / structured_dtype.itemsize)

        worker_logger.info(f"Batch size: {batch_size_mb} MB = {max_rows_per_batch} rows")

        # Import feature extraction modules
        sys.path.insert(0, os.path.dirname(__file__))
        from experiment_feature_extractor import ExperimentFeatureExtractor
        from segment_processor import SegmentFilesetProcessor

        # Initialize segment processor
        segment_processor = SegmentFilesetProcessor(experiment_id=experiment_id)

        # Process each work unit
        processed_count = 0
        for work_unit in work_units:
            segment_id, data_type_id, decimation, amplitude_method = work_unit
            processed_count += 1

            if processed_count % 100 == 0:
                worker_logger.info(f"Processed {processed_count}/{len(work_units)} work units")

            try:
                # Get segment info
                cursor.execute("""
                    SELECT experiment_file_id, beginning_index, segment_length
                    FROM data_segments
                    WHERE segment_id = %s
                """, (segment_id,))
                seg_info = cursor.fetchone()

                if not seg_info:
                    worker_logger.warning(f"Segment {segment_id} not found in database")
                    with open(done_file, 'a') as f:
                        f.write(f"{segment_id},{data_type_id},{decimation},{amplitude_method}\n")
                    continue

                file_id = seg_info['experiment_file_id']
                start_idx = seg_info['beginning_index']
                seg_length = seg_info['segment_length']

                # Check if segment size matches allowed sizes
                if segment_sizes and seg_length not in segment_sizes:
                    with open(done_file, 'a') as f:
                        f.write(f"{segment_id},{data_type_id},{decimation},{amplitude_method}\n")
                    continue

                # Load file data
                if data_type_id == 1:  # RAW
                    file_path = Path(raw_data_folder) / f"{file_id:08d}.npy"
                    if not file_path.exists():
                        worker_logger.warning(f"Raw file not found: {file_path}")
                        with open(done_file, 'a') as f:
                            f.write(f"{segment_id},{data_type_id},{decimation},{amplitude_method}\n")
                        continue
                    file_data = np.load(file_path)
                else:  # ADC
                    file_path = Path(adc_data_folder) / f"{file_id:08d}.npy"
                    if not file_path.exists():
                        worker_logger.warning(f"ADC file not found: {file_path}")
                        with open(done_file, 'a') as f:
                            f.write(f"{segment_id},{data_type_id},{decimation},{amplitude_method}\n")
                        continue
                    file_data = np.load(file_path)

                # Extract segment
                end_idx = start_idx + seg_length
                segment_data = file_data[start_idx:end_idx].copy()

                # Apply decimation
                if decimation > 0:
                    segment_data = segment_processor.apply_decimation(segment_data, decimation)

                # Get data type name for conversion
                dtype_name = data_type_map.get(data_type_id, 'RAW')

                # Apply data type conversion
                if data_type_id != 1:  # Not RAW
                    segment_data = segment_processor.apply_data_type_conversion(segment_data, dtype_name)

                # Apply amplitude processing
                if amplitude_method == 2:  # Z-score
                    mean = segment_data.mean(axis=0)
                    std = segment_data.std(axis=0)
                    if std.any():
                        segment_data = (segment_data - mean) / std
                elif amplitude_method == 3:  # Min-max normalization
                    min_val = segment_data.min(axis=0)
                    max_val = segment_data.max(axis=0)
                    range_val = max_val - min_val
                    if range_val.any():
                        segment_data = (segment_data - min_val) / range_val

                # Extract features using FeatureFunctionLoader (database-driven)
                # Replaces 70 lines of hardcoded if/elif chains with simple loader call
                feature_values = []
                for feature_id in feature_ids:
                    try:
                        feature_result = loader.extract_feature(feature_id, segment_data, sample_rate=5000000)
                        feature_values.append(feature_result)
                    except Exception as feat_err:
                        worker_logger.error(f"Feature extraction error (seg={segment_id}, feat={feature_id}): {feat_err}")
                        import traceback
                        worker_logger.error(traceback.format_exc())
                        feature_values.append(np.nan)

                # Create row for structured array
                row_data = [segment_id] + feature_values

                # Get configuration key for batch management
                resulting_size = seg_length // (decimation + 1)
                config_key = (seg_length, data_type_id, decimation, amplitude_method)

                # Initialize batch tracking for this configuration if needed
                if config_key not in config_batches:
                    config_batches[config_key] = {'batch_num': 0, 'batch_rows': []}

                config_batches[config_key]['batch_rows'].append(tuple(row_data))

                # Check if batch is full for this configuration
                if len(config_batches[config_key]['batch_rows']) >= max_rows_per_batch:
                    # Save batch
                    batch_num = config_batches[config_key]['batch_num']
                    batch_rows = config_batches[config_key]['batch_rows']

                    batch_array = np.array(batch_rows, dtype=structured_dtype)
                    batch_filename = (f"features_S{seg_length:06d}_D{decimation:06d}_"
                                     f"R{resulting_size:06d}_{dtype_name}_A{amplitude_method:02d}_"
                                     f"worker{worker_id:02d}_batch_{batch_num:04d}.npy")
                    batch_filepath = output_dir / batch_filename
                    np.save(batch_filepath, batch_array)
                    worker_logger.info(f"Saved batch {batch_num} for config {config_key}: {len(batch_rows)} rows")

                    config_batches[config_key]['batch_num'] += 1
                    config_batches[config_key]['batch_rows'] = []

                # Mark as done
                with open(done_file, 'a') as f:
                    f.write(f"{segment_id},{data_type_id},{decimation},{amplitude_method}\n")

            except Exception as e:
                worker_logger.error(f"Failed to process work unit {work_unit}: {e}")
                import traceback
                worker_logger.error(traceback.format_exc())

                # Mark as done even if failed
                with open(done_file, 'a') as f:
                    f.write(f"{segment_id},{data_type_id},{decimation},{amplitude_method}\n")

        # Save final batches for all configurations
        for config_key, config_data in config_batches.items():
            batch_rows = config_data['batch_rows']
            if batch_rows:
                batch_num = config_data['batch_num']
                seg_length, data_type_id, decimation, amplitude_method = config_key
                resulting_size = seg_length // (decimation + 1)
                dtype_name = data_type_map.get(data_type_id, 'RAW')

                batch_array = np.array(batch_rows, dtype=structured_dtype)
                batch_filename = (f"features_S{seg_length:06d}_D{decimation:06d}_"
                                 f"R{resulting_size:06d}_{dtype_name}_A{amplitude_method:02d}_"
                                 f"worker{worker_id:02d}_batch_{batch_num:04d}.npy")
                batch_filepath = output_dir / batch_filename
                np.save(batch_filepath, batch_array)
                worker_logger.info(f"Saved final batch {batch_num} for config {config_key}: {len(batch_rows)} rows")

        cursor.close()
        db_conn.close()
        worker_logger.info(f"Worker {worker_id} (PID {pid}) completed successfully")

    except Exception as e:
        worker_logger.error(f"Worker {worker_id} (PID {pid}) fatal error: {e}")
        import traceback
        worker_logger.error(traceback.format_exc())
        logger.error(f"Worker {worker_id} (PID {pid}) error: {e}")


def manager_process(experiment_id: int, data_type_ids: List[int], decimations: List[int],
                   segment_sizes: List[int], amplitude_methods: List[int], feature_ids: List[int],
                   workers_count: int, output_dir: Path, mpcctl_dir: Path,
                   batch_size_mb: float, db_config: Dict,
                   raw_data_folder: str, adc_data_folder: str):
    """
    Manager process that coordinates worker processes and monitors progress.

    Distributes ALL configuration combinations across workers in parallel.

    Args:
        experiment_id: Experiment ID
        data_type_ids: List of data type IDs
        decimations: List of decimation factors
        segment_sizes: List of segment sizes to process
        amplitude_methods: List of amplitude processing method IDs
        feature_ids: List of feature IDs to compute
        workers_count: Number of worker processes
        output_dir: Output directory
        mpcctl_dir: Path to .mpcctl directory
        batch_size_mb: Maximum batch file size in MB
        db_config: Database configuration dictionary
        raw_data_folder: Path to raw data folder
        adc_data_folder: Path to ADC data folder
    """
    import itertools

    logger.info(f"Manager process starting for experiment {experiment_id}")
    logger.info(f"Configurations:")
    logger.info(f"  Data types: {data_type_ids}")
    logger.info(f"  Decimations: {decimations}")
    logger.info(f"  Segment sizes: {segment_sizes}")
    logger.info(f"  Amplitude methods: {amplitude_methods}")
    logger.info(f"  Features: {feature_ids}")

    try:
        # Connect to database
        db_conn = psycopg2.connect(**db_config)
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Query ALL segments from training files
        # This gives us full dataset with all classes represented
        file_table = f"experiment_{experiment_id:03d}_file_training_data"

        logger.info(f"Querying ALL segments from training files with segment_lengths={segment_sizes}...")
        cursor.execute(f"""
            SELECT DISTINCT ds.segment_id, ds.segment_length
            FROM data_segments ds
            JOIN {file_table} eftd ON ds.experiment_file_id = eftd.file_id
            WHERE eftd.experiment_id = %s
              AND ds.segment_length = ANY(%s)
            ORDER BY ds.segment_id
        """, (experiment_id, segment_sizes))

        segment_rows = cursor.fetchall()
        total_segments = len(segment_rows)

        logger.info(f"Found {total_segments:,} verification segments")

        if total_segments == 0:
            logger.warning("No segments found to process")
            return

        # Generate all configuration combinations
        configurations = list(itertools.product(data_type_ids, decimations, amplitude_methods))
        logger.info(f"Total configurations: {len(configurations)}")

        # Generate all work units: (segment_id, data_type_id, decimation, amplitude_method)
        work_units = []
        for seg_row in segment_rows:
            seg_id = seg_row['segment_id']
            for data_type_id, decimation, amplitude_method in configurations:
                work_units.append((seg_id, data_type_id, decimation, amplitude_method))

        total_work_units = len(work_units)
        logger.info(f"Total work units: {total_work_units:,} ({total_segments} segments × {len(configurations)} configs)")

        if total_work_units == 0:
            logger.warning("No work units to process")
            return

        # Distribute work units among workers
        work_per_worker = total_work_units // workers_count

        worker_processes = []
        for worker_id in range(workers_count):
            start_idx = worker_id * work_per_worker
            if worker_id == workers_count - 1:
                end_idx = total_work_units
            else:
                end_idx = start_idx + work_per_worker

            worker_work_units = work_units[start_idx:end_idx]

            # Create worker process
            p = mp.Process(target=worker_function,
                          args=(worker_id, experiment_id, segment_sizes,
                                feature_ids, mpcctl_dir, output_dir,
                                batch_size_mb, db_config,
                                raw_data_folder, adc_data_folder))
            p.start()

            # Wait for process to start and get real PID
            time.sleep(0.1)
            pid = p.pid

            # Create todo file with work units (segment_id,data_type_id,decimation,amplitude_method)
            todo_file = mpcctl_dir / f"{pid}_todo.dat"
            done_file = mpcctl_dir / f"{pid}_done.dat"

            with open(todo_file, 'w') as f:
                for work_unit in worker_work_units:
                    seg_id, dt_id, dec, amp = work_unit
                    f.write(f"{seg_id},{dt_id},{dec},{amp}\n")

            # Create empty done file
            done_file.touch()

            worker_processes.append((worker_id, p, pid, len(worker_work_units)))
            logger.info(f"Worker {worker_id} (PID {pid}): {len(worker_work_units):,} work units")

        # Monitor progress
        start_time = time.time()
        last_report_time = start_time

        while True:
            time.sleep(5)  # Check every 5 seconds

            # Count completed work
            total_done = 0
            for worker_id, p, pid, work_count in worker_processes:
                done_file = mpcctl_dir / f"{pid}_done.dat"
                if done_file.exists():
                    with open(done_file, 'r') as f:
                        done_count = len(f.readlines())
                    total_done += done_count

            # Report progress every 30 seconds
            current_time = time.time()
            if current_time - last_report_time >= 30:
                elapsed = current_time - start_time
                progress_pct = (total_done / total_work_units) * 100
                rate = total_done / elapsed if elapsed > 0 else 0

                logger.info(f"Progress: {total_done:,}/{total_work_units:,} ({progress_pct:.1f}%) | "
                          f"Rate: {rate:.1f} work units/sec | Elapsed: {elapsed:.0f}s")

                last_report_time = current_time

            # Check if all workers finished
            all_done = all(not p.is_alive() for _, p, _, _ in worker_processes)
            if all_done:
                break

        # Wait for all workers
        for worker_id, p, pid, work_count in worker_processes:
            p.join()

        elapsed_time = time.time() - start_time
        logger.info(f"All workers completed in {elapsed_time:.1f} seconds")
        logger.info(f"Total work units processed: {total_done:,}/{total_work_units:,}")

        # Stitch batch files together for all configurations
        logger.info("Stitching batch files together...")

        # Build data type map (convert to uppercase for consistency)
        cursor.execute("SELECT data_type_id, data_type_name FROM ml_data_types_lut")
        data_type_map = {row['data_type_id']: row['data_type_name'].upper() for row in cursor.fetchall()}

        # Process each configuration
        files_created = 0
        for seg_length in segment_sizes:
            for data_type_id, decimation, amplitude_method in configurations:
                dtype_name = data_type_map.get(data_type_id, 'RAW')
                resulting_size = seg_length // (decimation + 1)

                # Find all batch files for this configuration
                batch_pattern = f"features_S{seg_length:06d}_D{decimation:06d}_R{resulting_size:06d}_{dtype_name}_A{amplitude_method:02d}_worker*_batch_*.npy"
                batch_files = sorted(output_dir.glob(batch_pattern))

                if not batch_files:
                    logger.warning(f"No batch files found for config: S{seg_length}, D{decimation}, {dtype_name}, A{amplitude_method}")
                    continue

                logger.info(f"Processing config: S{seg_length:06d}_D{decimation:06d}_{dtype_name}_A{amplitude_method:02d}")
                logger.info(f"  Found {len(batch_files)} batch files")

                # Load and concatenate all batches
                all_arrays = []
                total_rows = 0
                for batch_file in batch_files:
                    batch_data = np.load(batch_file)
                    all_arrays.append(batch_data)
                    total_rows += len(batch_data)

                logger.info(f"  Total rows: {total_rows:,}")

                if all_arrays:
                    final_array = np.concatenate(all_arrays)
                    logger.info(f"  Final array shape: {final_array.shape}")

                    # Save final file
                    final_filename = f"features_S{seg_length:06d}_D{decimation:06d}_R{resulting_size:06d}_{dtype_name}_A{amplitude_method:02d}.npy"
                    final_filepath = output_dir / final_filename
                    np.save(final_filepath, final_array)
                    logger.info(f"  Saved: {final_filepath.name}")
                    files_created += 1

                    # Clean up batch files
                    for batch_file in batch_files:
                        batch_file.unlink()

        cursor.close()
        db_conn.close()

        logger.info(f"Feature matrix generation complete")
        logger.info(f"  Total final files created: {files_created}")
        logger.info(f"  Expected configurations: {len(segment_sizes) * len(configurations)}")

    except Exception as e:
        logger.error(f"Manager process error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for MPCCTL verification feature matrix generator."""
    parser = argparse.ArgumentParser(description='MPCCTL Verification Feature Matrix Generator')
    parser.add_argument('--experiment-id', type=int, required=True)
    parser.add_argument('--data-type', type=str, required=True,
                       help='Comma-separated data type IDs or names (e.g., 2,6,8 or ADC2,ADC6,ADC8)')
    parser.add_argument('--decimation', type=str, required=True,
                       help='Comma-separated decimation factors (e.g., 0,7,15,31,63,127)')
    parser.add_argument('--segment-size', type=str, required=True,
                       help='Comma-separated segment sizes (e.g., 512,8192)')
    parser.add_argument('--amplitude-method', type=str, required=True,
                       help='Comma-separated amplitude method IDs (e.g., 1,2 for raw,zscore)')
    parser.add_argument('--feature-id', type=str, required=True,
                       help='Comma-separated list of feature IDs from ml_features_lut')
    parser.add_argument('--output-folder', type=str, required=False,
                       help='Output folder (default: experiment/classifier_files/verification_features)')
    parser.add_argument('--input-raw-data-folder', type=str, default='/Volumes/ArcData/V3_database/fileset',
                       help='Path to raw data folder')
    parser.add_argument('--input-adc-data-folder', type=str, default='/Volumes/ArcData/V3_database/adc_data',
                       help='Path to ADC data folder')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size-mb', type=float, default=100.0,
                       help='Maximum batch file size in MB (default: 100)')
    parser.add_argument('--mpcctl-dir', type=str, required=True)
    parser.add_argument('--db-host', type=str, default='localhost')
    parser.add_argument('--db-port', type=int, default=5432)
    parser.add_argument('--db-name', type=str, default='arc_detection')
    parser.add_argument('--db-user', type=str, default='kjensen')

    args = parser.parse_args()

    # Parse data types
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user
    }

    db_conn = psycopg2.connect(**db_config)
    cursor = db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Parse comma-separated data types
    data_type_ids = []
    for dt_str in args.data_type.split(','):
        dt_str = dt_str.strip()
        if dt_str.isdigit():
            data_type_ids.append(int(dt_str))
        else:
            cursor.execute("""
                SELECT data_type_id
                FROM ml_data_types_lut
                WHERE data_type_name = %s
            """, (dt_str,))
            result = cursor.fetchone()
            if not result:
                logger.error(f"Data type not found: {dt_str}")
                return
            data_type_ids.append(result['data_type_id'])

    # Parse comma-separated decimations
    decimations = [int(x.strip()) for x in args.decimation.split(',')]

    # Parse comma-separated segment sizes
    segment_sizes = [int(x.strip()) for x in args.segment_size.split(',')]

    # Parse comma-separated amplitude methods
    amplitude_methods = [int(x.strip()) for x in args.amplitude_method.split(',')]

    # Parse feature IDs
    feature_ids = [int(x.strip()) for x in args.feature_id.split(',')]

    # Set output directory
    if args.output_folder:
        output_dir = Path(args.output_folder)
    else:
        output_dir = Path(f'/Volumes/ArcData/V3_database/experiment{args.experiment_id:03d}/classifier_files/verification_features')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Set mpcctl directory
    mpcctl_dir = Path(args.mpcctl_dir)
    mpcctl_dir.mkdir(parents=True, exist_ok=True)

    cursor.close()
    db_conn.close()

    manager_process(
        experiment_id=args.experiment_id,
        data_type_ids=data_type_ids,
        decimations=decimations,
        segment_sizes=segment_sizes,
        amplitude_methods=amplitude_methods,
        feature_ids=feature_ids,
        workers_count=args.workers,
        output_dir=output_dir,
        mpcctl_dir=mpcctl_dir,
        batch_size_mb=args.batch_size_mb,
        db_config=db_config,
        raw_data_folder=args.input_raw_data_folder,
        adc_data_folder=args.input_adc_data_folder
    )


if __name__ == '__main__':
    main()
