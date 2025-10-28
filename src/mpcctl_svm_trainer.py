#!/usr/bin/env python3
"""
Filename: mpcctl_svm_trainer.py
Author(s): Kristophor Jensen
Date Created: 20251027_163000
Date Revised: 20251028_144123
File version: 1.0.0.19
Description: MPCCTL-based SVM training with summary generation and file export

ARCHITECTURE (follows mpcctl_cli_distance_calculator.py pattern):
- Manager creates .mpcctl/ directory with {PID}_todo.dat files
- Each worker reads from its own {PID}_todo.dat file
- Workers write completed task IDs to {PID}_done.dat for resume capability
- Workers write results directly to database with autocommit=True
- Manager monitors done files for progress tracking
- State file (.mpcctl_state.json) for CLI monitoring and control
- Pause/Resume/Stop via multiprocessing.Event flags
"""

import os
import sys
import json
import time
import warnings
import shutil
import logging
import numpy as np
import psycopg2
import psycopg2.extras
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def worker_process(worker_id: int, pause_flag: mp.Event, stop_flag: mp.Event,
                  mpcctl_dir: Path, db_config: Dict, experiment_id: int,
                  classifier_id: int, global_classifier_id: int,
                  log_file: Optional[Path], verbose: bool, cache_size_mb: int,
                  use_linear_svc: bool = False):
    """
    Worker process for SVM training using .mpcctl file-based coordination.

    Args:
        worker_id: Worker ID (PID)
        pause_flag: Shared pause flag
        stop_flag: Shared stop flag
        mpcctl_dir: .mpcctl directory path
        db_config: Database configuration dict
        experiment_id: Experiment ID
        classifier_id: Classifier ID
        global_classifier_id: Global classifier ID
        log_file: Optional log file path
        verbose: Verbose output flag
        cache_size_mb: Cache size in MB for SVC kernel computations
        use_linear_svc: Use LinearSVC for linear kernel (10-100x faster)
    """
    # Suppress all warnings
    warnings.filterwarnings('ignore')

    # Setup logging
    if log_file:
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] Worker {}: %(message)s'.format(worker_id)
        )

    def log(message: str):
        """Log to file only (no console output - progress shown via status files)."""
        if log_file:
            logging.info(message)

    # Load worker assignments
    todo_file = mpcctl_dir / f"{worker_id}_todo.dat"
    done_file = mpcctl_dir / f"{worker_id}_done.dat"

    if not todo_file.exists():
        log(f"ERROR: Todo file not found: {todo_file}")
        return

    # Read assigned configs (format: dec dtype amp efs kernel C gamma)
    configs = []
    with open(todo_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 6:
                    dec, dtype, amp, efs, kernel, c_val = parts[:6]
                    gamma_val = parts[6] if len(parts) > 6 and parts[6] != 'None' else None

                    # Parse gamma: can be numeric, 'scale', 'auto', or None
                    gamma = None
                    if gamma_val:
                        try:
                            gamma = float(gamma_val)
                        except ValueError:
                            gamma = gamma_val  # Keep as string for 'scale', 'auto'

                    configs.append({
                        'dec': int(dec),
                        'dtype': int(dtype),
                        'amp': int(amp),
                        'efs': int(efs),
                        'kernel': kernel,
                        'C': float(c_val),
                        'gamma': gamma
                    })

    log(f"Loaded {len(configs)} config assignments")

    # Read already completed configs
    completed_configs = set()
    if done_file.exists():
        with open(done_file, 'r') as f:
            for line in f:
                if line.strip():
                    completed_configs.add(line.strip())
        log(f"Resuming: {len(completed_configs)} already completed")

    # Filter out completed configs
    remaining_configs = []
    for cfg in configs:
        config_str = f"{cfg['dec']} {cfg['dtype']} {cfg['amp']} {cfg['efs']} {cfg['kernel']} {cfg['C']} {cfg['gamma'] if cfg['gamma'] else 'None'}"
        if config_str not in completed_configs:
            remaining_configs.append(cfg)

    log(f"Processing {len(remaining_configs)} remaining configs")

    # Initialize status file for progress tracking
    status_file = mpcctl_dir / f"{worker_id}_status.json"

    def update_status(completed: int, current_task: str = ""):
        """Update worker status file for monitor display."""
        status = {
            'worker_id': worker_id,
            'total_tasks': len(remaining_configs),
            'completed_tasks': completed,
            'current_task': current_task,
            'last_update': datetime.now().isoformat()
        }
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)

    # Write initial status
    update_status(0, "Initializing...")

    # Connect to database with autocommit
    conn = psycopg2.connect(**db_config)
    conn.autocommit = True
    cursor = conn.cursor()

    # Load label categories
    cursor.execute("SELECT label_id, category FROM segment_labels WHERE category IS NOT NULL")
    label_categories = {row[0]: row[1] for row in cursor.fetchall()}

    # Import ML libraries
    from sklearn.svm import SVC, LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix, roc_auc_score, average_precision_score,
                                 roc_curve, precision_recall_curve, auc)
    import joblib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Table names
    features_table = f"experiment_{experiment_id:03d}_classifier_{classifier_id:03d}_svm_features"
    splits_table = f"experiment_{experiment_id:03d}_classifier_{classifier_id:03d}_data_splits"
    results_table = f"experiment_{experiment_id:03d}_classifier_{classifier_id:03d}_svm_results"

    # Process configs
    processed = 0
    errors = 0

    for cfg in remaining_configs:
        # Check stop flag
        if stop_flag.is_set():
            log("Stop signal received, exiting")
            break

        # Check pause flag
        while pause_flag.is_set():
            time.sleep(0.1)

        dec = cfg['dec']
        dtype = cfg['dtype']
        amp = cfg['amp']
        efs = cfg['efs']
        svm_params = {'kernel': cfg['kernel'], 'C': cfg['C'], 'gamma': cfg['gamma']}

        # Update status at start of task
        task_desc = f"dec={dec} dt={dtype} amp={amp} efs={efs} k={svm_params['kernel']}"
        update_status(processed, task_desc)

        try:
            start_time = time.time()

            # Load features for all splits
            def load_split(split_type):
                cursor.execute(f"""
                    SELECT sf.segment_id, sf.segment_label_id, sf.svm_feature_file_path
                    FROM {features_table} sf
                    JOIN {splits_table} ds
                        ON sf.segment_id = ds.segment_id
                        AND sf.decimation_factor = ds.decimation_factor
                        AND sf.data_type_id = ds.data_type_id
                    WHERE sf.decimation_factor = %s
                      AND sf.data_type_id = %s
                      AND sf.amplitude_processing_method_id = %s
                      AND sf.experiment_feature_set_id = %s
                      AND ds.split_type = %s
                      AND sf.extraction_status_id = 2
                    ORDER BY sf.segment_id
                """, (dec, dtype, amp, efs, split_type))

                rows = cursor.fetchall()
                if len(rows) == 0:
                    raise ValueError(f"No features for {split_type}")

                X = []
                y = []
                for segment_id, label_id, file_path in rows:
                    try:
                        features = np.load(file_path)
                        X.append(features)
                        y.append(label_id)
                    except Exception:
                        pass

                if len(X) == 0:
                    raise ValueError(f"No valid features loaded for {split_type}")

                return np.array(X), np.array(y)

            X_train, y_train = load_split('training')
            X_test, y_test = load_split('test')
            X_verify, y_verify = load_split('verification')

            # Diagnostic: Check label distribution
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            unique_verify, counts_verify = np.unique(y_verify, return_counts=True)

            log(f"Label distribution - Train: {len(unique_train)} classes {dict(zip(unique_train, counts_train))}")
            log(f"Label distribution - Test: {len(unique_test)} classes {dict(zip(unique_test, counts_test))}")
            log(f"Label distribution - Verify: {len(unique_verify)} classes {dict(zip(unique_verify, counts_verify))}")

            if len(unique_train) < 2:
                log(f"ERROR: Training set has only {len(unique_train)} class(es)! Cannot train classifier.")
                raise ValueError(f"Insufficient classes in training data: only {len(unique_train)} class(es) found")

            # Train SVM
            # LinearSVC option: 10-100x faster than SVC for linear kernel, but different implementation
            # User can choose between LinearSVC (fast) or SVC(kernel='linear') (consistent with rbf/poly)
            if svm_params['kernel'] == 'linear' and use_linear_svc:
                # Use LinearSVC (optimized linear solver)
                base_svm = LinearSVC(
                    C=svm_params['C'],
                    class_weight='balanced',
                    random_state=42,
                    max_iter=10000,
                    dual='auto'  # Let sklearn choose based on n_samples vs n_features
                )

                # Train base LinearSVC WITHOUT calibration first to check predictions
                log(f"Training LinearSVC (no calibration) with C={svm_params['C']}")
                base_svm.fit(X_train, y_train)

                # Check predictions from base LinearSVC
                y_pred_base_train = base_svm.predict(X_train)
                y_pred_base_test = base_svm.predict(X_test)
                unique_pred_base_train = np.unique(y_pred_base_train)
                unique_pred_base_test = np.unique(y_pred_base_test)
                log(f"LinearSVC base predictions - Train: {len(unique_pred_base_train)} classes {list(unique_pred_base_train)}")
                log(f"LinearSVC base predictions - Test: {len(unique_pred_base_test)} classes {list(unique_pred_base_test)}")

                if len(unique_pred_base_test) == 1:
                    log(f"ERROR: LinearSVC base model predicting only 1 class: {unique_pred_base_test[0]}")
                    log(f"  This is a LinearSVC bug - trying with different parameters")
                    # Try without class_weight='balanced'
                    base_svm = LinearSVC(
                        C=svm_params['C'],
                        random_state=42,
                        max_iter=10000,
                        dual='auto'
                    )
                    base_svm.fit(X_train, y_train)
                    y_pred_retry = base_svm.predict(X_test)
                    unique_pred_retry = np.unique(y_pred_retry)
                    log(f"LinearSVC retry (no class_weight) - Test: {len(unique_pred_retry)} classes {list(unique_pred_retry)}")

                # Wrap in calibrator for probability estimates (cv=2 means 3 models total)
                svm = CalibratedClassifierCV(base_svm, cv=2, method='sigmoid')
                log(f"Using LinearSVC with calibration (fast linear solver)")
            else:
                # Use standard SVC for all kernels (linear, rbf, poly)
                # Enforce minimum cache sizes for kernel matrix operations
                # Linear kernel: min 2GB (n^2 dot products, iterative optimization)
                # RBF/Poly kernels: min 2GB (n^2 expensive kernel evals, iterative optimization)
                if svm_params['kernel'] == 'linear':
                    min_cache = 2000  # 2GB minimum for linear kernel
                elif svm_params['kernel'] in ['rbf', 'poly']:
                    min_cache = 2000  # 2GB minimum for non-linear kernels
                else:
                    min_cache = 2000  # 2GB default

                actual_cache = max(cache_size_mb, min_cache)
                if actual_cache > cache_size_mb:
                    log(f"WARNING: Requested cache {cache_size_mb}MB too small, using {actual_cache}MB minimum for kernel='{svm_params['kernel']}'")

                svm_kwargs = {
                    'kernel': svm_params['kernel'],
                    'C': svm_params['C'],
                    'class_weight': 'balanced',
                    'random_state': 42,
                    'probability': True,
                    'cache_size': actual_cache,  # Enforced minimum cache size
                    'shrinking': False  # Disable shrinking to use full cache (increases memory, may improve speed)
                }
                if svm_params['kernel'] in ['rbf', 'poly'] and svm_params.get('gamma'):
                    svm_kwargs['gamma'] = svm_params['gamma']
                svm = SVC(**svm_kwargs)
                log(f"Using SVC(kernel='{svm_params['kernel']}') with {actual_cache}MB cache, shrinking=False")

            svm.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Predictions
            y_pred_train = svm.predict(X_train)
            y_pred_test = svm.predict(X_test)
            y_pred_verify = svm.predict(X_verify)

            # Diagnostic: Check prediction distribution
            unique_pred_train = np.unique(y_pred_train)
            unique_pred_test = np.unique(y_pred_test)
            unique_pred_verify = np.unique(y_pred_verify)

            log(f"Predictions - Train: {len(unique_pred_train)} unique classes {list(unique_pred_train)}")
            log(f"Predictions - Test: {len(unique_pred_test)} unique classes {list(unique_pred_test)}")
            log(f"Predictions - Verify: {len(unique_pred_verify)} unique classes {list(unique_pred_verify)}")

            if len(unique_pred_test) == 1:
                log(f"WARNING: Model predicting only 1 class on test set: {unique_pred_test[0]}")
                log(f"  This may indicate: (1) severe class imbalance, (2) poor features, or (3) hyperparameter issues")

            y_proba_train = svm.predict_proba(X_train)
            y_proba_test = svm.predict_proba(X_test)
            y_proba_verify = svm.predict_proba(X_verify)

            # Compute metrics
            def compute_multiclass_metrics(y_true, y_pred):
                accuracy = accuracy_score(y_true, y_pred)
                precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='macro', zero_division=0
                )
                precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='weighted', zero_division=0
                )
                return {
                    'accuracy': accuracy,
                    'precision_macro': precision_macro,
                    'recall_macro': recall_macro,
                    'f1_macro': f1_macro,
                    'precision_weighted': precision_weighted,
                    'recall_weighted': recall_weighted,
                    'f1_weighted': f1_weighted
                }

            def compute_binary_metrics(y_true, y_pred, y_proba):
                y_true_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_true])
                y_pred_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_pred])

                unique_labels = np.unique(y_true)
                arc_label_indices = [i for i, label in enumerate(unique_labels)
                                   if label_categories.get(int(label), 'unknown') == 'arc']

                if len(arc_label_indices) > 0 and y_proba.shape[1] >= len(unique_labels):
                    y_proba_arc = np.sum(y_proba[:, arc_label_indices], axis=1)
                else:
                    y_proba_arc = np.zeros(len(y_true))

                accuracy = accuracy_score(y_true_binary, y_pred_binary)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true_binary, y_pred_binary, average='binary', zero_division=0
                )

                cm = confusion_matrix(y_true_binary, y_pred_binary)
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    specificity = 0.0

                try:
                    if len(np.unique(y_true_binary)) > 1:
                        roc_auc = roc_auc_score(y_true_binary, y_proba_arc)
                        pr_auc = average_precision_score(y_true_binary, y_proba_arc)
                    else:
                        roc_auc = 0.0
                        pr_auc = 0.0
                except:
                    roc_auc = 0.0
                    pr_auc = 0.0

                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'specificity': specificity,
                    'roc_auc': roc_auc,
                    'pr_auc': pr_auc
                }

            metrics_train = compute_multiclass_metrics(y_train, y_pred_train)
            metrics_test = compute_multiclass_metrics(y_test, y_pred_test)
            metrics_verify = compute_multiclass_metrics(y_verify, y_pred_verify)

            binary_train = compute_binary_metrics(y_train, y_pred_train, y_proba_train)
            binary_test = compute_binary_metrics(y_test, y_pred_test, y_proba_test)
            binary_verify = compute_binary_metrics(y_verify, y_pred_verify, y_proba_verify)

            # Save model
            base_dir = f"/Volumes/ArcData/V3_database/experiment{experiment_id:03d}/classifier_files"
            model_dir = f"{base_dir}/svm_models/classifier_{classifier_id:03d}/D{dec:06d}_TADC{dtype}_A{amp}_EFS{efs:03d}"
            os.makedirs(model_dir, exist_ok=True)

            model_filename = f"svm_{svm_params['kernel']}_C{svm_params['C']}"
            if svm_params.get('gamma'):
                model_filename += f"_G{svm_params['gamma']}"
            model_filename += ".pkl"
            model_path = os.path.join(model_dir, model_filename)

            joblib.dump(svm, model_path)

            # Save visualizations
            viz_dir = f"{base_dir}/svm_results/classifier_{classifier_id:03d}/D{dec:06d}_TADC{dtype}_A{amp}_EFS{efs:03d}"
            svm_viz_dir = f"{viz_dir}/{svm_params['kernel']}_C{svm_params['C']}"
            if svm_params.get('gamma'):
                svm_viz_dir += f"_G{svm_params['gamma']}"
            os.makedirs(svm_viz_dir, exist_ok=True)

            # Save confusion matrices (simplified to avoid large file I/O)
            for split_name, y_true, y_pred in [
                ('train', y_train, y_pred_train),
                ('test', y_test, y_pred_test),
                ('verify', y_verify, y_pred_verify)
            ]:
                # 13-class
                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(12, 10))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'13-Class Confusion Matrix ({split_name.title()})')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(f"{svm_viz_dir}/confusion_matrix_13class_{split_name}.png", dpi=150, bbox_inches='tight')
                plt.close()

                # Binary
                y_true_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_true])
                y_pred_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_pred])
                cm = confusion_matrix(y_true_binary, y_pred_binary)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                           xticklabels=['Non-arc', 'Arc'], yticklabels=['Non-arc', 'Arc'])
                plt.title(f'Binary Arc Detection ({split_name.title()})')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(f"{svm_viz_dir}/confusion_matrix_binary_{split_name}.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Generate ROC and PR curves for binary classification
            for split_name, y_true, y_proba_split in [
                ('train', y_train, y_proba_train),
                ('test', y_test, y_proba_test),
                ('verify', y_verify, y_proba_verify)
            ]:
                # Get binary labels and arc probabilities
                y_true_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_true])

                # Get arc class probability
                unique_labels = np.unique(y_train)
                arc_label_indices = [i for i, lbl in enumerate(unique_labels) if label_categories.get(int(lbl), 'unknown') == 'arc']

                if len(arc_label_indices) > 0 and y_proba_split.shape[1] >= len(unique_labels):
                    y_proba_arc = np.sum(y_proba_split[:, arc_label_indices], axis=1)
                else:
                    y_proba_arc = np.zeros(len(y_true))

                # ROC Curve
                if len(np.unique(y_true_binary)) > 1:  # Need both classes present
                    fpr, tpr, _ = roc_curve(y_true_binary, y_proba_arc)
                    roc_auc_val = auc(fpr, tpr)

                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - Arc Detection ({split_name.title()})')
                    plt.legend(loc="lower right")
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f"{svm_viz_dir}/roc_curve_binary_{split_name}.png", dpi=150, bbox_inches='tight')
                    plt.close()

                    # PR Curve
                    precision, recall, _ = precision_recall_curve(y_true_binary, y_proba_arc)
                    pr_auc_val = auc(recall, precision)

                    plt.figure(figsize=(8, 6))
                    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc_val:.3f})')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.title(f'Precision-Recall Curve - Arc Detection ({split_name.title()})')
                    plt.legend(loc="lower left")
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f"{svm_viz_dir}/pr_curve_binary_{split_name}.png", dpi=150, bbox_inches='tight')
                    plt.close()

                    # F1 Score vs Threshold Curve
                    # Calculate F1 scores at different thresholds
                    thresholds = np.linspace(0, 1, 101)  # 101 points from 0.0 to 1.0
                    f1_scores = []

                    for threshold in thresholds:
                        y_pred_threshold = (y_proba_arc >= threshold).astype(int)
                        # Calculate F1 for this threshold
                        if len(np.unique(y_pred_threshold)) > 1:  # At least 2 classes predicted
                            prec, rec, f1, _ = precision_recall_fscore_support(
                                y_true_binary, y_pred_threshold, average='binary', zero_division=0
                            )
                            f1_scores.append(f1)
                        else:
                            f1_scores.append(0.0)

                    f1_scores = np.array(f1_scores)
                    best_threshold_idx = np.argmax(f1_scores)
                    best_threshold = thresholds[best_threshold_idx]
                    best_f1 = f1_scores[best_threshold_idx]

                    plt.figure(figsize=(8, 6))
                    plt.plot(thresholds, f1_scores, color='green', lw=2, label=f'F1 Score')
                    plt.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5,
                               label=f'Best: F1={best_f1:.3f} @ threshold={best_threshold:.3f}')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('Classification Threshold')
                    plt.ylabel('F1 Score')
                    plt.title(f'F1 Score vs Threshold - Arc Detection ({split_name.title()})')
                    plt.legend(loc="best")
                    plt.grid(True, alpha=0.3)
                    plt.savefig(f"{svm_viz_dir}/f1_threshold_curve_binary_{split_name}.png", dpi=150, bbox_inches='tight')
                    plt.close()

            # Convert numpy types to Python types
            def to_python_type(value):
                if value is None:
                    return None
                if isinstance(value, (np.floating, np.integer)):
                    if np.isnan(value) or np.isinf(value):
                        return None
                    return float(value)
                return value

            # Insert results into database
            cursor.execute(f"""
                INSERT INTO {results_table}
                    (global_classifier_id, classifier_id, decimation_factor, data_type_id,
                     amplitude_processing_method_id, experiment_feature_set_id,
                     svm_kernel, svm_c_parameter, svm_gamma,
                     class_weight, random_state,
                     train_ratio, test_ratio, verification_ratio, cv_folds,
                     accuracy_train, precision_macro_train, recall_macro_train, f1_macro_train,
                     precision_weighted_train, recall_weighted_train, f1_weighted_train,
                     cv_mean_accuracy, cv_std_accuracy,
                     accuracy_test, precision_macro_test, recall_macro_test, f1_macro_test,
                     precision_weighted_test, recall_weighted_test, f1_weighted_test,
                     accuracy_verify, precision_macro_verify, recall_macro_verify, f1_macro_verify,
                     precision_weighted_verify, recall_weighted_verify, f1_weighted_verify,
                     arc_accuracy_train, arc_precision_train, arc_recall_train, arc_f1_train,
                     arc_specificity_train, arc_roc_auc_train, arc_pr_auc_train,
                     arc_accuracy_test, arc_precision_test, arc_recall_test, arc_f1_test,
                     arc_specificity_test, arc_roc_auc_test, arc_pr_auc_test,
                     arc_accuracy_verify, arc_precision_verify, arc_recall_verify, arc_f1_verify,
                     arc_specificity_verify, arc_roc_auc_verify, arc_pr_auc_verify,
                     model_path, training_time_seconds, prediction_time_seconds)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s)
            """, (
                global_classifier_id, classifier_id, dec, dtype, amp, efs,
                svm_params['kernel'], svm_params['C'], svm_params.get('gamma'),
                'balanced', 42, 0.70, 0.20, 0.10, 5,
                to_python_type(metrics_train['accuracy']), to_python_type(metrics_train['precision_macro']),
                to_python_type(metrics_train['recall_macro']), to_python_type(metrics_train['f1_macro']),
                to_python_type(metrics_train['precision_weighted']), to_python_type(metrics_train['recall_weighted']),
                to_python_type(metrics_train['f1_weighted']), 0.0, 0.0,
                to_python_type(metrics_test['accuracy']), to_python_type(metrics_test['precision_macro']),
                to_python_type(metrics_test['recall_macro']), to_python_type(metrics_test['f1_macro']),
                to_python_type(metrics_test['precision_weighted']), to_python_type(metrics_test['recall_weighted']),
                to_python_type(metrics_test['f1_weighted']),
                to_python_type(metrics_verify['accuracy']), to_python_type(metrics_verify['precision_macro']),
                to_python_type(metrics_verify['recall_macro']), to_python_type(metrics_verify['f1_macro']),
                to_python_type(metrics_verify['precision_weighted']), to_python_type(metrics_verify['recall_weighted']),
                to_python_type(metrics_verify['f1_weighted']),
                to_python_type(binary_train['accuracy']), to_python_type(binary_train['precision']),
                to_python_type(binary_train['recall']), to_python_type(binary_train['f1']),
                to_python_type(binary_train['specificity']), to_python_type(binary_train['roc_auc']), to_python_type(binary_train['pr_auc']),
                to_python_type(binary_test['accuracy']), to_python_type(binary_test['precision']),
                to_python_type(binary_test['recall']), to_python_type(binary_test['f1']),
                to_python_type(binary_test['specificity']), to_python_type(binary_test['roc_auc']), to_python_type(binary_test['pr_auc']),
                to_python_type(binary_verify['accuracy']), to_python_type(binary_verify['precision']),
                to_python_type(binary_verify['recall']), to_python_type(binary_verify['f1']),
                to_python_type(binary_verify['specificity']), to_python_type(binary_verify['roc_auc']), to_python_type(binary_verify['pr_auc']),
                model_path, training_time, 0.0
            ))

            # Mark as done
            config_str = f"{dec} {dtype} {amp} {efs} {svm_params['kernel']} {svm_params['C']} {svm_params['gamma'] if svm_params['gamma'] else 'None'}"
            with open(done_file, 'a') as f:
                f.write(config_str + '\n')

            processed += 1
            log(f"COMPLETED [{processed}/{len(remaining_configs)}]: dec={dec}, dtype={dtype}, amp={amp}, efs={efs}, acc={metrics_test['accuracy']:.4f}, time={training_time:.1f}s")

            # Update status after completion
            update_status(processed, "")

        except Exception as e:
            errors += 1
            log(f"ERROR: dec={dec}, dtype={dtype}, amp={amp}, efs={efs} - {str(e)}")
            # Update status after error (keep completed count unchanged)
            update_status(processed, "")

    cursor.close()
    conn.close()

    # Final status update
    update_status(processed, "Finished")

    log(f"Worker finished: {processed} processed, {errors} errors")


def monitor_progress(mpcctl_dir: Path, total_configs: int, state_file: Path,
                     log_file: Optional[Path], verbose: bool, stop_flag: mp.Event):
    """Monitor progress with static per-worker progress bars."""

    def log(msg):
        """Log to file only (console shows progress bars)."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_msg = f"[{timestamp}] {msg}"
        if log_file:
            with open(log_file, 'a') as f:
                f.write(full_msg + '\n')

    from datetime import datetime

    start_time = time.time()
    last_completed = 0

    # ANSI escape codes
    CLEAR_SCREEN = '\033[2J'
    CURSOR_HOME = '\033[H'
    HIDE_CURSOR = '\033[?25l'
    SHOW_CURSOR = '\033[?25h'

    try:
        # Hide cursor for cleaner display
        print(HIDE_CURSOR, end='', flush=True)

        while not stop_flag.is_set():
            try:
                # Read all worker status files
                worker_statuses = []
                for status_file in sorted(mpcctl_dir.glob('*_status.json')):
                    try:
                        with open(status_file, 'r') as f:
                            status = json.load(f)
                            worker_statuses.append(status)
                    except Exception:
                        pass

                # Calculate overall progress
                total_completed = sum(w.get('completed_tasks', 0) for w in worker_statuses)
                progress_pct = (total_completed / total_configs * 100) if total_configs > 0 else 0
                elapsed = time.time() - start_time
                rate = total_completed / elapsed if elapsed > 0 else 0
                remaining = total_configs - total_completed
                eta_seconds = remaining / rate if rate > 0 else 0

                # Clear screen and move cursor to home on every update
                print(CLEAR_SCREEN + CURSOR_HOME, end='', flush=True)

                # Display header
                print("=" * 80)
                print(f"SVM Training Progress - {total_completed:,}/{total_configs:,} configs ({progress_pct:.1f}%)")
                print(f"Rate: {rate:.2f} configs/sec | ETA: {eta_seconds/60:.1f} min")
                print("=" * 80)
                print()

                # Display per-worker progress bars
                bar_width = 50
                stalled_workers = []
                for worker_status in worker_statuses:
                    worker_id = worker_status.get('worker_id', 0)
                    completed = worker_status.get('completed_tasks', 0)
                    total = worker_status.get('total_tasks', 1)
                    current_task = worker_status.get('current_task', '')
                    last_update_str = worker_status.get('last_update', '')

                    worker_pct = (completed / total * 100) if total > 0 else 0
                    filled = int(bar_width * completed / total) if total > 0 else 0
                    bar = '█' * filled + '░' * (bar_width - filled)

                    # Check if worker is stalled (no update in 2 minutes)
                    is_stalled = False
                    time_since_update = 0
                    if last_update_str and current_task != "Finished":
                        try:
                            last_update = datetime.fromisoformat(last_update_str)
                            time_since_update = (datetime.now() - last_update).total_seconds()
                            if time_since_update > 120:  # 2 minutes
                                is_stalled = True
                                stalled_workers.append(worker_id)
                        except Exception:
                            pass

                    # Display worker progress
                    print(f"Worker {worker_id}: [{bar}] {worker_pct:5.1f}%")

                    # Show current task or status
                    if is_stalled:
                        print(f"    ⚠ STALLED (no update for {int(time_since_update)}s) - may have crashed")
                    elif current_task and current_task != "Finished":
                        task_display = current_task[:60] if len(current_task) > 60 else current_task
                        print(f"    {task_display}")
                    elif current_task == "Finished":
                        print(f"    ✓ Complete")
                    else:
                        print()

                print()

                # Show warning if workers are stalled
                if stalled_workers:
                    print("⚠ WARNING: Some workers appear to have crashed or stalled")
                    print(f"   Stalled workers: {', '.join(map(str, stalled_workers))}")
                    print(f"   These workers may have been killed by OS (out of memory)")
                    print()

                print("=" * 80)
                print(f"Log: {log_file if log_file else 'None'}")
                print("Press Ctrl+C to detach (training continues in background)")
                print("=" * 80)

                # Update state file
                if state_file.exists():
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)

                        state['progress'] = {
                            'total_tasks': total_configs,
                            'completed_tasks': total_completed,
                            'percent_complete': round(progress_pct, 2),
                            'configs_per_second': round(rate, 3),
                            'estimated_time_remaining_seconds': int(eta_seconds)
                        }

                        with open(state_file, 'w') as f:
                            json.dump(state, f, indent=2)
                    except Exception:
                        pass

                if total_completed != last_completed:
                    log(f"Progress: {total_completed}/{total_configs} configs ({progress_pct:.1f}%)")
                    last_completed = total_completed

                if total_completed >= total_configs:
                    log("All configs completed")
                    break

                time.sleep(1)  # Update every 1 second

            except Exception as e:
                log(f"Monitor error: {e}")
                time.sleep(1)

    finally:
        # Show cursor again
        print(SHOW_CURSOR, end='', flush=True)


def manager_process(experiment_id: int, classifier_id: int, workers_count: int,
                   db_config: Dict, filters: Dict, log_file: Optional[Path],
                   verbose: bool, mpcctl_base_dir: Path, max_memory_mb: int):
    """Manager process - creates .mpcctl files and spawns workers."""

    def log(msg):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_msg = f"[{timestamp}] {msg}"
        if verbose:
            print(full_msg)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(full_msg + '\n')

    # Calculate cache size per worker
    cache_size_mb = max_memory_mb // workers_count

    # Extract use_linear_svc flag from filters
    use_linear_svc = filters.get('use_linear_svc', False)

    log(f"Manager started: Experiment {experiment_id}, Classifier {classifier_id}, Workers {workers_count}")
    log(f"Memory configuration: {max_memory_mb} MB total, {cache_size_mb} MB cache per worker")
    if use_linear_svc:
        log(f"Using LinearSVC for linear kernel (fast solver)")
    else:
        log(f"Using SVC for all kernels (consistent implementation)")

    try:
        # Create .mpcctl directory
        mpcctl_dir = mpcctl_base_dir / ".mpcctl"
        if mpcctl_dir.exists():
            log("Removing existing .mpcctl directory...")
            shutil.rmtree(mpcctl_dir)
        mpcctl_dir.mkdir(parents=True, exist_ok=True)
        log(f"Created .mpcctl directory: {mpcctl_dir}")

        # Create state file
        state_file = mpcctl_base_dir / f".mpcctl_state.json"
        state = {
            'status': 'running',
            'experiment_id': experiment_id,
            'classifier_id': classifier_id,
            'workers_count': workers_count,
            'start_time': datetime.now().isoformat(),
            'progress': {
                'total_tasks': 0,
                'completed_tasks': 0,
                'percent_complete': 0.0
            }
        }
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Connect to database and load configuration
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Get global classifier ID
        cursor.execute("""
            SELECT global_classifier_id
            FROM ml_experiment_classifiers
            WHERE experiment_id = %s AND classifier_id = %s
        """, (experiment_id, classifier_id))
        result = cursor.fetchone()
        if not result:
            log(f"ERROR: Classifier not found")
            return
        global_classifier_id = result[0]

        # Get active configuration
        cursor.execute("""
            SELECT config_id FROM ml_classifier_configs
            WHERE global_classifier_id = %s AND is_active = true
            LIMIT 1
        """, (global_classifier_id,))
        result = cursor.fetchone()
        if not result:
            log("ERROR: No active configuration")
            return
        config_id = result[0]

        # Query hyperparameters
        cursor.execute("""
            SELECT DISTINCT lut.decimation_factor
            FROM ml_classifier_config_decimation_factors cfg
            JOIN ml_experiment_decimation_lut lut ON cfg.decimation_factor = lut.decimation_id
            WHERE cfg.config_id = %s
            ORDER BY lut.decimation_factor
        """, (config_id,))
        decimation_factors = [row[0] for row in cursor.fetchall()]

        cursor.execute("""
            SELECT DISTINCT data_type_id
            FROM ml_classifier_config_data_types
            WHERE config_id = %s
            ORDER BY data_type_id
        """, (config_id,))
        data_type_ids = [row[0] for row in cursor.fetchall()]

        cursor.execute("""
            SELECT DISTINCT amplitude_processing_method_id
            FROM ml_classifier_config_amplitude_methods
            WHERE config_id = %s
            ORDER BY amplitude_processing_method_id
        """, (config_id,))
        amplitude_methods = [row[0] for row in cursor.fetchall()]

        cursor.execute("""
            SELECT DISTINCT experiment_feature_set_id
            FROM ml_classifier_config_experiment_feature_sets
            WHERE config_id = %s
            ORDER BY experiment_feature_set_id
        """, (config_id,))
        experiment_feature_sets = [row[0] for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        # Apply filters
        if filters.get('decimation_factor') is not None:
            decimation_factors = [filters['decimation_factor']]
        if filters.get('data_type') is not None:
            data_type_ids = [filters['data_type']]
        if filters.get('amplitude_method') is not None:
            amplitude_methods = [filters['amplitude_method']]
        if filters.get('experiment_feature_set') is not None:
            experiment_feature_sets = [filters['experiment_feature_set']]

        # Build SVM parameter grid (hardcoded like original implementation)
        if filters.get('svm_kernel') is None:
            kernels = ['linear', 'rbf', 'poly']
        else:
            kernels = [filters['svm_kernel']]

        if filters.get('svm_C') is None:
            C_values = [0.1, 1.0, 10.0, 100.0]
        else:
            C_values = [filters['svm_C']]

        gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1]

        # Build svm_hyperparams list: (kernel, c_value, gamma_value)
        svm_hyperparams = []
        for k in kernels:
            for c in C_values:
                if k in ['rbf', 'poly']:
                    for g in gamma_values:
                        svm_hyperparams.append((k, c, g))
                else:
                    svm_hyperparams.append((k, c, None))

        # Build all configs
        all_configs = []
        for dec in decimation_factors:
            for dtype in data_type_ids:
                for amp in amplitude_methods:
                    for efs in experiment_feature_sets:
                        for kernel, c_val, gamma_val in svm_hyperparams:
                            config_line = f"{dec} {dtype} {amp} {efs} {kernel} {c_val} {gamma_val if gamma_val else 'None'}"
                            all_configs.append(config_line)

        total_configs = len(all_configs)
        log(f"Total training configs: {total_configs}")

        if total_configs == 0:
            log("ERROR: No configs to process")
            return

        # Distribute configs among workers using round-robin for balanced workload
        # Round-robin ensures each worker gets a mix of all parameter combinations
        # (avoids assigning all slow dec=0 tasks to first workers)
        worker_assignments = [[] for _ in range(workers_count)]

        for idx, config_line in enumerate(all_configs):
            worker_idx = idx % workers_count
            worker_assignments[worker_idx].append(config_line)

        log(f"Round-robin distribution: {len(all_configs)} configs across {workers_count} workers")

        worker_pids = []
        for worker_idx in range(workers_count):
            assigned_configs = worker_assignments[worker_idx]

            # Generate unique worker ID (use incremental IDs, actual PID will be different)
            worker_id = 10000 + worker_idx

            # Create todo file
            todo_file = mpcctl_dir / f"{worker_id}_todo.dat"
            done_file = mpcctl_dir / f"{worker_id}_done.dat"

            with open(todo_file, 'w') as f:
                for config_line in assigned_configs:
                    f.write(config_line + '\n')

            done_file.touch()  # Create empty done file

            worker_pids.append(worker_id)

            # Log distribution stats for this worker
            # Extract decimation factors from assigned configs
            dec_counts = {}
            for config_line in assigned_configs:
                dec = config_line.split()[0]
                dec_counts[dec] = dec_counts.get(dec, 0) + 1
            dec_summary = ', '.join([f"dec{k}={v}" for k, v in sorted(dec_counts.items())])
            log(f"Worker {worker_id}: {len(assigned_configs)} configs ({dec_summary})")

        # Update state file with total tasks
        state['progress']['total_tasks'] = total_configs
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        # Create shared flags for control
        pause_flag = mp.Event()
        stop_flag = mp.Event()

        # Spawn workers
        workers = []
        for worker_id in worker_pids:
            worker = mp.Process(
                target=worker_process,
                args=(worker_id, pause_flag, stop_flag, mpcctl_dir, db_config,
                      experiment_id, classifier_id, global_classifier_id,
                      log_file, verbose, cache_size_mb, use_linear_svc)
            )
            worker.start()
            workers.append(worker)
            log(f"Spawned worker {worker_id} (PID {worker.pid})")

        # Start monitor
        monitor = mp.Process(
            target=monitor_progress,
            args=(mpcctl_dir, total_configs, state_file, log_file, verbose, stop_flag)
        )
        monitor.start()
        log("Monitor process started")

        # Wait for workers
        for worker in workers:
            worker.join()

        # Signal monitor to stop
        stop_flag.set()
        monitor.join(timeout=5)
        if monitor.is_alive():
            monitor.terminate()

        # Generate and save training summary
        try:
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Get results from database for this training run
            results_table = f"experiment_{experiment_id:03d}_classifier_{classifier_id:03d}_svm_results"

            # Build WHERE clause based on filters
            where_conditions = []
            where_params = []

            if filters.get('decimation_factor') is not None:
                where_conditions.append("decimation_factor = %s")
                where_params.append(filters['decimation_factor'])
            if filters.get('data_type') is not None:
                where_conditions.append("data_type_id = %s")
                where_params.append(filters['data_type'])
            if filters.get('amplitude_method') is not None:
                where_conditions.append("amplitude_processing_method_id = %s")
                where_params.append(filters['amplitude_method'])
            if filters.get('experiment_feature_set') is not None:
                where_conditions.append("experiment_feature_set_id = %s")
                where_params.append(filters['experiment_feature_set'])
            if filters.get('svm_kernel') is not None:
                where_conditions.append("svm_kernel = %s")
                where_params.append(filters['svm_kernel'])

            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

            query = f"""
                SELECT
                    COUNT(*) as total_configs,
                    AVG(accuracy_test) as avg_test_acc,
                    MAX(accuracy_test) as max_test_acc,
                    AVG(arc_f1_test) as avg_arc_f1,
                    MAX(arc_f1_test) as max_arc_f1,
                    AVG(arc_roc_auc_test) as avg_roc_auc,
                    MAX(arc_roc_auc_test) as max_roc_auc,
                    AVG(training_time_seconds) as avg_train_time
                FROM {results_table}
                WHERE {where_clause}
            """

            cursor.execute(query, where_params)
            row = cursor.fetchone()

            summary_lines = []
            summary_lines.append("=" * 80)
            summary_lines.append("TRAINING SUMMARY")
            summary_lines.append("=" * 80)
            summary_lines.append(f"Experiment: {experiment_id}, Classifier: {classifier_id}")
            summary_lines.append(f"Total Configurations Trained: {row[0]}")
            summary_lines.append("")
            summary_lines.append("Test Set Performance:")
            summary_lines.append(f"  Average Accuracy: {row[1]:.4f}")
            summary_lines.append(f"  Best Accuracy: {row[2]:.4f}")
            summary_lines.append("")
            summary_lines.append("Binary Arc Detection (Test Set):")
            summary_lines.append(f"  Average F1 Score: {row[3]:.4f}")
            summary_lines.append(f"  Best F1 Score: {row[4]:.4f}")
            summary_lines.append(f"  Average ROC AUC: {row[5]:.4f}")
            summary_lines.append(f"  Best ROC AUC: {row[6]:.4f}")
            summary_lines.append("")
            summary_lines.append(f"Average Training Time: {row[7]:.2f} seconds per config")
            summary_lines.append("=" * 80)

            summary_text = "\n".join(summary_lines)

            # Display summary
            print("\n" + summary_text)

            # Save summary to file
            summary_file = mpcctl_base_dir / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(summary_file, 'w') as f:
                f.write(summary_text + "\n")

            log(f"Training summary saved to: {summary_file}")

            cursor.close()
            conn.close()

        except Exception as e:
            log(f"Error generating summary: {e}")

        # Final state update
        state['status'] = 'completed'
        state['progress']['percent_complete'] = 100.0
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

        log("Training complete")

    except Exception as e:
        import traceback
        log(f"MANAGER ERROR: {e}")
        log(traceback.format_exc())


if __name__ == '__main__':
    print("This module is designed to be imported and used via manager_process()")
    print("Use: from mpcctl_svm_trainer import manager_process")
