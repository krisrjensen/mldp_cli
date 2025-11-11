#!/usr/bin/env python3
"""
Filename: mpcctl_svm_feature_builder.py
Author(s): Kristophor Jensen
Date Created: 20251110_114000
Date Revised: 20251110_121000
File version: 2.1.0.3
Description: MPCCTL-based SVM feature vector builder with parallel worker processing

ARCHITECTURE:
- Manager process runs in background (daemon)
- Worker processes coordinated via .mpcctl files
- Pause/Resume/Stop control via multiprocessing.Event
- State file (.mpcctl_state.json) for progress and control
- Work units: (segment_id, dec, dtype, amp, efs) combinations
- Each worker gets a batch of work units via {PID}_todo.dat

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
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics.pairwise import pairwise_distances

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def worker_function(worker_id: int, experiment_id: int, classifier_id: int,
                   config_id: int, mpcctl_dir: Path, db_config: Dict):
    """
    Worker process that builds SVM features for assigned work units.

    Args:
        worker_id: Unique worker ID
        experiment_id: Experiment ID
        classifier_id: Classifier ID
        config_id: Configuration ID
        mpcctl_dir: Path to .mpcctl directory
        db_config: Database configuration dictionary
    """
    pid = os.getpid()
    todo_file = mpcctl_dir / f"{pid}_todo.dat"
    done_file = mpcctl_dir / f"{pid}_done.dat"

    # Silent worker - no output except errors
    try:
        # Connect to database
        db_conn = psycopg2.connect(**db_config)
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Load work units from todo file
        work_units = []
        with open(todo_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    segment_id, dec, dtype, amp, efs = map(int, parts)
                    work_units.append((segment_id, dec, dtype, amp, efs))

        if not work_units:
            return

        # Get feature builder configuration
        cursor.execute("""
            SELECT include_original_feature,
                   compute_baseline_distances_inter,
                   compute_baseline_distances_intra
            FROM ml_classifier_feature_builder
            WHERE config_id = %s
        """, (config_id,))
        fb_row = cursor.fetchone()
        if not fb_row:
            return

        include_original = fb_row['include_original_feature']
        compute_inter = fb_row['compute_baseline_distances_inter']
        compute_intra = fb_row['compute_baseline_distances_intra']
        needs_references = compute_inter or compute_intra

        # Get feature base path
        cursor.execute("""
            SELECT feature_data_base_path
            FROM ml_experiments
            WHERE experiment_id = %s
        """, (experiment_id,))
        result = cursor.fetchone()
        if result and result['feature_data_base_path']:
            feature_base_path = Path(result['feature_data_base_path'])
        else:
            # Use default path
            feature_base_path = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}/feature_files')

        # Get table names
        features_table = f"experiment_{experiment_id:03d}_classifier_{classifier_id:03d}_svm_features"
        feature_fileset_table = f"experiment_{experiment_id:03d}_feature_fileset"
        ref_table = f"experiment_{experiment_id:03d}_classifier_{classifier_id:03d}_reference_segments"

        # Get distance functions if needed
        distance_metrics = []
        if needs_references:
            cursor.execute("""
                SELECT df.pairwise_metric_name
                FROM ml_classifier_config_distance_functions cdf
                JOIN ml_distance_functions_lut df ON cdf.distance_function_id = df.distance_function_id
                WHERE cdf.config_id = %s
                ORDER BY df.distance_function_id
            """, (config_id,))
            distance_metrics = [row['pairwise_metric_name'] for row in cursor.fetchall()]

        # Get number of classes
        cursor.execute("""
            SELECT COUNT(DISTINCT segment_label_id) as num_classes
            FROM experiment_{exp_id:03d}_segment_training_data
        """.format(exp_id=experiment_id))
        num_classes = cursor.fetchone()['num_classes']

        # Process each work unit
        for work_unit in work_units:
            segment_id, dec, dtype, amp, efs = work_unit

            try:
                start_time = time.time()

                # Check if already exists
                cursor.execute(f"""
                    SELECT 1 FROM {features_table}
                    WHERE segment_id = %s AND decimation_factor = %s
                      AND data_type_id = %s AND amplitude_processing_method_id = %s
                      AND experiment_feature_set_id = %s
                """, (segment_id, dec, dtype, amp, efs))

                if cursor.fetchone():
                    # Already exists, mark as done
                    with open(done_file, 'a') as f:
                        f.write(f"{segment_id},{dec},{dtype},{amp},{efs}\n")
                    continue

                # Get segment label
                cursor.execute("""
                    SELECT segment_label_id
                    FROM data_segments
                    WHERE segment_id = %s
                """, (segment_id,))
                segment_label_id = cursor.fetchone()['segment_label_id']

                # Build feature vector path
                svm_feature_dir = feature_base_path / "svm_features" / f"S{segment_id:08d}"
                svm_feature_dir.mkdir(parents=True, exist_ok=True)
                svm_feature_file = svm_feature_dir / f"svm_feature_{segment_id}_{dec}_{dtype}_{amp}_{efs}.npy"

                # Build feature vector
                if needs_references:
                    # Load reference segments for this configuration
                    cursor.execute(f"""
                        SELECT reference_segment_id, segment_label_id
                        FROM {ref_table}
                        WHERE decimation_factor = %s AND data_type_id = %s
                          AND amplitude_processing_method_id = %s
                          AND experiment_feature_set_id = %s
                        ORDER BY segment_label_id
                    """, (dec, dtype, amp, efs))
                    references = cursor.fetchall()

                    if not references:
                        # No references, mark as failed (status 2)
                        cursor.execute(f"""
                            INSERT INTO {features_table} (
                                global_classifier_id, classifier_id, segment_id, segment_label_id,
                                decimation_factor, data_type_id, amplitude_processing_method_id,
                                experiment_feature_set_id, svm_feature_file_path,
                                feature_vector_dimensions, num_classes, num_distance_metrics,
                                extraction_status_id, extraction_time_seconds
                            ) VALUES (
                                (SELECT global_classifier_id FROM ml_experiment_classifiers
                                 WHERE experiment_id = %s AND classifier_id = %s),
                                %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, 0, 2, %s
                            )
                        """, (experiment_id, classifier_id, classifier_id, segment_id, segment_label_id,
                              dec, dtype, amp, efs, str(svm_feature_file), num_classes, 0.0))
                        db_conn.commit()
                        # Mark as done even though failed
                        with open(done_file, 'a') as f:
                            f.write(f"{segment_id},{dec},{dtype},{amp},{efs}\n")
                        continue

                    # Get target segment features
                    cursor.execute(f"""
                        SELECT feature_value_mean, feature_value_std, feature_value_max
                        FROM {feature_fileset_table}
                        WHERE segment_id = %s AND decimation_factor = %s
                          AND data_type_id = %s AND amplitude_processing_method_id = %s
                          AND experiment_feature_set_id = %s
                        ORDER BY feature_set_feature_id
                    """, (segment_id, dec, dtype, amp, efs))
                    target_features = cursor.fetchall()

                    if not target_features:
                        # No features, mark as failed
                        cursor.execute(f"""
                            INSERT INTO {features_table} (
                                global_classifier_id, classifier_id, segment_id, segment_label_id,
                                decimation_factor, data_type_id, amplitude_processing_method_id,
                                experiment_feature_set_id, svm_feature_file_path,
                                feature_vector_dimensions, num_classes, num_distance_metrics,
                                extraction_status_id, extraction_time_seconds
                            ) VALUES (
                                (SELECT global_classifier_id FROM ml_experiment_classifiers
                                 WHERE experiment_id = %s AND classifier_id = %s),
                                %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, 0, 2, %s
                            )
                        """, (experiment_id, classifier_id, classifier_id, segment_id, segment_label_id,
                              dec, dtype, amp, efs, str(svm_feature_file), num_classes, 0.0))
                        db_conn.commit()
                        # Mark as done even though failed
                        with open(done_file, 'a') as f:
                            f.write(f"{segment_id},{dec},{dtype},{amp},{efs}\n")
                        continue

                    # Build target feature array
                    target_array = np.array([[f['feature_value_mean'], f['feature_value_std'], f['feature_value_max']]
                                            for f in target_features]).flatten()

                    # Compute distances to each reference
                    distance_vectors = []
                    for ref_row in references:
                        ref_seg_id = ref_row['reference_segment_id']

                        # Get reference features
                        cursor.execute(f"""
                            SELECT feature_value_mean, feature_value_std, feature_value_max
                            FROM {feature_fileset_table}
                            WHERE segment_id = %s AND decimation_factor = %s
                              AND data_type_id = %s AND amplitude_processing_method_id = %s
                              AND experiment_feature_set_id = %s
                            ORDER BY feature_set_feature_id
                        """, (ref_seg_id, dec, dtype, amp, efs))
                        ref_features = cursor.fetchall()

                        if not ref_features:
                            continue

                        ref_array = np.array([[f['feature_value_mean'], f['feature_value_std'], f['feature_value_max']]
                                             for f in ref_features]).flatten()

                        # Compute distances using all metrics
                        for metric in distance_metrics:
                            dist = pairwise_distances([target_array], [ref_array], metric=metric)[0, 0]
                            distance_vectors.append(dist)

                    # Build final feature vector
                    feature_vector = np.array(distance_vectors, dtype=np.float32)
                    feature_dims = len(feature_vector)

                else:
                    # Just use raw features
                    cursor.execute(f"""
                        SELECT feature_value_mean, feature_value_std, feature_value_max
                        FROM {feature_fileset_table}
                        WHERE segment_id = %s AND decimation_factor = %s
                          AND data_type_id = %s AND amplitude_processing_method_id = %s
                          AND experiment_feature_set_id = %s
                        ORDER BY feature_set_feature_id
                    """, (segment_id, dec, dtype, amp, efs))
                    features = cursor.fetchall()

                    if not features:
                        cursor.execute(f"""
                            INSERT INTO {features_table} (
                                global_classifier_id, classifier_id, segment_id, segment_label_id,
                                decimation_factor, data_type_id, amplitude_processing_method_id,
                                experiment_feature_set_id, svm_feature_file_path,
                                feature_vector_dimensions, num_classes, num_distance_metrics,
                                extraction_status_id, extraction_time_seconds
                            ) VALUES (
                                (SELECT global_classifier_id FROM ml_experiment_classifiers
                                 WHERE experiment_id = %s AND classifier_id = %s),
                                %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, 0, 2, %s
                            )
                        """, (experiment_id, classifier_id, classifier_id, segment_id, segment_label_id,
                              dec, dtype, amp, efs, str(svm_feature_file), num_classes, 0.0))
                        db_conn.commit()
                        # Mark as done even though failed
                        with open(done_file, 'a') as f:
                            f.write(f"{segment_id},{dec},{dtype},{amp},{efs}\n")
                        continue

                    feature_vector = np.array([[f['feature_value_mean'], f['feature_value_std'], f['feature_value_max']]
                                              for f in features]).flatten().astype(np.float32)
                    feature_dims = len(feature_vector)

                # Save feature vector
                np.save(svm_feature_file, feature_vector)

                extraction_time = time.time() - start_time

                # Insert database record
                cursor.execute(f"""
                    INSERT INTO {features_table} (
                        global_classifier_id, classifier_id, segment_id, segment_label_id,
                        decimation_factor, data_type_id, amplitude_processing_method_id,
                        experiment_feature_set_id, svm_feature_file_path,
                        feature_vector_dimensions, num_classes, num_distance_metrics,
                        extraction_status_id, extraction_time_seconds
                    ) VALUES (
                        (SELECT global_classifier_id FROM ml_experiment_classifiers
                         WHERE experiment_id = %s AND classifier_id = %s),
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 3, %s
                    )
                """, (experiment_id, classifier_id, classifier_id, segment_id, segment_label_id,
                      dec, dtype, amp, efs, str(svm_feature_file),
                      feature_dims, num_classes, len(distance_metrics), extraction_time))
                db_conn.commit()

                # Mark as done
                with open(done_file, 'a') as f:
                    f.write(f"{segment_id},{dec},{dtype},{amp},{efs}\n")

            except Exception as e:
                # Mark as failed (status 2)
                try:
                    cursor.execute(f"""
                        INSERT INTO {features_table} (
                            global_classifier_id, classifier_id, segment_id, segment_label_id,
                            decimation_factor, data_type_id, amplitude_processing_method_id,
                            experiment_feature_set_id, svm_feature_file_path,
                            feature_vector_dimensions, num_classes, num_distance_metrics,
                            extraction_status_id, extraction_time_seconds
                        ) VALUES (
                            (SELECT global_classifier_id FROM ml_experiment_classifiers
                             WHERE experiment_id = %s AND classifier_id = %s),
                            %s, %s, %s, %s, %s, %s, %s, %s, 0, %s, 0, 2, 0.0
                        ) ON CONFLICT DO NOTHING
                    """, (experiment_id, classifier_id, classifier_id, segment_id, segment_label_id,
                          dec, dtype, amp, efs, "", num_classes))
                    db_conn.commit()
                except:
                    pass

                # CRITICAL: Mark as done even if failed, otherwise progress stalls
                with open(done_file, 'a') as f:
                    f.write(f"{segment_id},{dec},{dtype},{amp},{efs}\n")

        cursor.close()
        db_conn.close()

    except Exception as e:
        logger.error(f"Worker {worker_id} (PID {pid}) error: {e}")


def manager_process(experiment_id: int, classifier_id: int, config_id: int,
                   workers_count: int, db_config: Dict, mpcctl_dir: Path):
    """
    Manager process that coordinates worker processes and monitors progress.

    Args:
        experiment_id: Experiment ID
        classifier_id: Classifier ID
        config_id: Configuration ID
        workers_count: Number of worker processes
        db_config: Database configuration dictionary
        mpcctl_dir: Path to .mpcctl directory
    """
    logger.info(f"Manager process starting for experiment {experiment_id}, classifier {classifier_id}")

    try:
        # Connect to database
        db_conn = psycopg2.connect(**db_config)
        cursor = db_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        # Get all work units (segment_id, dec, dtype, amp, efs combinations)
        segment_table = f"experiment_{experiment_id:03d}_segment_training_data"

        cursor.execute(f"""
            SELECT DISTINCT
                std.segment_id,
                cdec.decimation_factor,
                cdt.data_type_id,
                cam.amplitude_processing_method_id,
                cefs.experiment_feature_set_id
            FROM {segment_table} std
            CROSS JOIN ml_classifier_config_decimation_factors cdec
            CROSS JOIN ml_classifier_config_data_types cdt
            CROSS JOIN ml_classifier_config_amplitude_methods cam
            CROSS JOIN ml_classifier_config_experiment_feature_sets cefs
            WHERE cdec.config_id = %s
              AND cdt.config_id = %s
              AND cam.config_id = %s
              AND cefs.config_id = %s
            ORDER BY std.segment_id, cdec.decimation_factor, cdt.data_type_id,
                     cam.amplitude_processing_method_id, cefs.experiment_feature_set_id
        """, (config_id, config_id, config_id, config_id))

        work_units = cursor.fetchall()
        total_work = len(work_units)

        logger.info(f"Total work units: {total_work:,}")

        # Distribute work among workers
        work_per_worker = total_work // workers_count

        worker_processes = []
        for worker_id in range(workers_count):
            start_idx = worker_id * work_per_worker
            if worker_id == workers_count - 1:
                end_idx = total_work
            else:
                end_idx = start_idx + work_per_worker

            worker_work = work_units[start_idx:end_idx]

            # Create worker files
            # Start worker with a fake PID, it will use its real PID
            p = mp.Process(target=worker_function,
                          args=(worker_id, experiment_id, classifier_id, config_id,
                                mpcctl_dir, db_config))
            p.start()

            # Wait for process to start and get real PID
            time.sleep(0.1)
            pid = p.pid

            # Create todo file
            todo_file = mpcctl_dir / f"{pid}_todo.dat"
            done_file = mpcctl_dir / f"{pid}_done.dat"

            with open(todo_file, 'w') as f:
                for wu in worker_work:
                    f.write(f"{wu['segment_id']},{wu['decimation_factor']},{wu['data_type_id']},"
                           f"{wu['amplitude_processing_method_id']},{wu['experiment_feature_set_id']}\n")

            # Create empty done file
            done_file.touch()

            worker_processes.append((worker_id, p, pid, len(worker_work)))
            logger.info(f"Worker {worker_id} (PID {pid}): {len(worker_work):,} work units")

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
                progress_pct = (total_done / total_work) * 100
                rate = total_done / elapsed if elapsed > 0 else 0

                logger.info(f"Progress: {total_done:,}/{total_work:,} ({progress_pct:.1f}%) | "
                          f"Rate: {rate:.1f} features/sec | Elapsed: {elapsed:.0f}s")

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
        logger.info(f"Total features built: {total_done:,}/{total_work:,}")

        cursor.close()
        db_conn.close()

    except Exception as e:
        logger.error(f"Manager process error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for mpcctl SVM feature builder."""
    parser = argparse.ArgumentParser(description='MPCCTL SVM Feature Builder')
    parser.add_argument('--experiment-id', type=int, required=True)
    parser.add_argument('--classifier-id', type=int, required=True)
    parser.add_argument('--config-id', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--mpcctl-dir', type=str, required=True)
    parser.add_argument('--db-host', type=str, default='localhost')
    parser.add_argument('--db-port', type=int, default=5432)
    parser.add_argument('--db-name', type=str, default='arc_detection')
    parser.add_argument('--db-user', type=str, default='kjensen')

    args = parser.parse_args()

    mpcctl_dir = Path(args.mpcctl_dir)
    mpcctl_dir.mkdir(parents=True, exist_ok=True)

    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user
    }

    manager_process(
        experiment_id=args.experiment_id,
        classifier_id=args.classifier_id,
        config_id=args.config_id,
        workers_count=args.workers,
        db_config=db_config,
        mpcctl_dir=mpcctl_dir
    )


if __name__ == '__main__':
    main()
