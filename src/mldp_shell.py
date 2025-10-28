#!/usr/bin/env python3
"""
Filename: mldp_shell.py
Author(s): Kristophor Jensen
Date Created: 20250901_240000
Date Revised: 20251028_150000
File version: 2.0.10.14
Description: Advanced interactive shell for MLDP with prompt_toolkit

Version Format: MAJOR.MINOR.COMMIT.CHANGE
- MAJOR: User-controlled major releases (currently 2)
- MINOR: User-controlled minor releases (currently 0)
- COMMIT: Increments on every git commit/push (currently 10)
- CHANGE: Tracks changes within current commit cycle (currently 1)

Changes in this version (10.14):
1. PHASE 4 SVM CONFIGURATION - Added --use-linear-svc flag for algorithm choice
   - v2.0.10.14: Added --use-linear-svc flag to classifier-train-svm command
                 By default, uses SVC(kernel='linear') for consistency across all kernels
                 With --use-linear-svc, uses LinearSVC for 10-100x faster linear training
                 User can choose between speed (LinearSVC) and consistency (SVC)
                 Allows testing both implementations for comparison

2. PHASE 4 RANDOM FOREST - Added Random Forest classifier training support
   - v2.0.10.13: Added classifier-train-rf command with MPCCTL architecture
                 Supports same configuration as SVM (decimation, data_type, amplitude, feature_set)
                 RF hyperparameters: n_estimators, max_depth, min_samples_split, max_features
                 Grid search: [50,100,200,500] trees × [10,20,30,None] depth × [2,5,10] split × ['sqrt','log2',None] features
                 Uses same feature vectors as SVM for direct comparison
                 Results stored in experiment_{exp}_classifier_{cls}_rf_results tables

2. PHASE 4 ARCHITECTURAL CHANGE - Implemented MPCCTL architecture for SVM training
   - v2.0.10.1: COMPLETE REWRITE of classifier-train-svm using mpcctl architecture
                Created new module: mpcctl_svm_trainer.py with file-based worker coordination
                Manager creates .mpcctl/ directory with {PID}_todo.dat and {PID}_done.dat files
                Workers read from todo files, write completed configs to done files
                Workers insert results directly to database with autocommit=True
                Main process monitors progress via .mpcctl_state.json file
                Live progress monitor with ETA, rate, and progress bar
                Detachable monitoring (Ctrl+C to detach, training continues)
                Resumable: Workers skip already-completed configs in done files
                NO DEADLOCKS: File-based coordination prevents pipe buffer issues
                Based on proven mpcctl_cli_distance_calculator.py pattern
   - Previous issues (v2.0.9.31-32): Pool.imap_unordered caused deadlocks
                Workers spinning CPU or hanging after 8-24 tasks
                Pipe buffer overflow from worker output or large return values
                Worker debug output was filling multiprocessing pipe buffers
                Caused deadlock after 8-24 tasks with main process blocked in imap_unordered
                Workers now run silently, main process reports all progress
   - v2.0.9.30: Fixed THIRD location of decimation query (line 16580)
                Previous fix in v2.0.9.27 only fixed 2 of 3 occurrences
                NOW FIXED: classifier-train-svm will query all 7 decimations
                Training should now work for all decimation factors
   - v2.0.9.29: Added DEBUG prints to show actual hyperparameter values
                Investigating why training uses IDs (1,4,5,6,7,8,9) instead of factors (0,7,15,31,63,127,255)
   - v2.0.9.28: Fixed classifier-train-svm WHERE clause column names
                Changed kernel_type -> svm_kernel
                Changed C_parameter -> svm_c_parameter
                Fixes "column does not exist" error in existing results check
   - v2.0.9.27: Fixed decimation_factor storage in features and training
                Config stores decimation IDs (1,4,5,6,7,8,9)
                Now converts to actual factors (0,7,15,31,63,127,255) via JOIN with LUT
                Fixes bug where only decimation 7 was trained (ID 7 = factor 63)
                BREAKING: Must rebuild features and retrain models for classifier 2
   - v2.0.9.26: Fixed classifier-train-svm existing results check
                Now properly filters by --amplitude-method, --decimation-factor, etc.
                Allows training different amplitude methods without --force
                Also fixed force deletion to respect filters
   - v2.0.9.25: Fixed classifier-build-features existing feature check
                Now properly filters by --amplitude-method, --decimation-factor, etc.
                Allows building different amplitude methods without --force
   - v2.0.9.24: Fixed classifier-build-features to properly support raw features
                Fixed filename generation to use vector_dims (not hardcoded)
                Fixed database INSERT to use vector_dims and num_metrics variables
                Raw features now work correctly (tested with include_original=True)
   - v2.0.9.23: Added classifier-copy-reference-segments command
                Fixed classifier-build-features to check feature_builder settings
                Only requires reference segments when computing distances
                Supports raw features only (no distance calculations)
   - v2.0.9.22: Added classifier-copy-splits-from command to copy train/test/verify
                splits from one classifier to another for fair model comparison
                Ensures identical data splits when comparing different classifiers
   - v2.0.9.21: DISABLED 5-fold cross-validation in train_svm_worker
                CV was taking 20-30 minutes per task (5x training time!)
                With 84 tasks, CV alone would take 28-42 hours
                We have separate test/verify sets, so CV is unnecessary
                Expected speedup: 46 hours → 14 hours (3.3x faster)
   - v2.0.9.20: Added real-time debug output to train_svm_worker
                Shows progress during feature loading (train/test/verify)
                Shows progress during SVM training and cross-validation
                Helps identify exactly where the process hangs
                Added sys.stdout.flush() to force immediate output
   - v2.0.9.19: Fixed segment_labels table reference in classifier-test-svm-single
                Changed FROM experiment_041_segment_labels to FROM segment_labels
                Added WHERE active = TRUE filter
                Matches actual database schema (global table, not per-experiment)
   - v2.0.9.18: Fixed SQL column name error in classifier-test-svm-single
                Changed local_classifier_id to classifier_id in query
                Matches actual ml_experiment_classifiers table schema
   - v2.0.9.17: Fixed AttributeError in classifier-test-svm-single
                Changed self.current_classifier to self.current_classifier_id
                Now matches convention used by all other classifier commands
   - v2.0.9.16: Added detailed timing instrumentation to train_svm_worker
                Added classifier-test-svm-single diagnostic command
                Shows timing breakdown for each operation (load, train, CV, predict, save)
                Identifies bottleneck operation automatically
                Helps diagnose performance issues in SVM training
   - v2.0.9.15: Enhanced classifier-train-svm progress reporting
                Shows EACH task completion with config and test accuracy
                Shows summary statistics every 10 tasks with avg time/task
   - v2.0.9.14: Added --amplitude-method option to classifier-train-svm command
                Allows training only specific amplitude methods (e.g., --amplitude-method 2)
   - v2.0.9.13: Added classifier-config-remove-hyperparameters command (~230 lines)
                Allows removing amplitude methods, decimation factors, data types,
                distance functions, and experiment feature sets from existing configs
   - v2.0.9.12: Fixed classifier-config-add-hyperparameters to include global_classifier_id
                and experiment_id in all 5 junction table INSERT statements
   - v2.0.9.11: Added classifier-config-add-hyperparameters command (~250 lines)
                Allows adding amplitude methods, decimation factors, data types,
                distance functions, and experiment feature sets to existing configs
   - v2.0.9.10: Added classifier-clean-svm-results and classifier-clean-features commands
                (~350 lines total for both commands with full filtering support)
   - v2.0.9.9: Added missing per_class_table_name definition in classifier-train-svm
   - v2.0.9.8: Convert numpy types to Python types before DB insertion
   - v2.0.9.7: Fixed queries to use correct Phase 0b table names
   - v2.0.9.6: Implemented database insertion and summary statistics (~250 lines)
   - v2.0.9.5: Implemented classifier-train-svm main training loop (~370 lines)
   - v2.0.9.4: Implemented SVM worker function and helpers (~470 lines)
   - v2.0.9.3: Implemented SVM training helper functions (~427 lines)
   - v2.0.9.2: Fixed NameError for 'force' variable in classifier-train-svm-init
   - v2.0.9.1: Renamed classifier-train-svm to classifier-train-svm-init
   - v2.0.9.0: Implementing classifier-train-svm command
   - Train SVM classifiers on distance-based feature vectors
   - Two confusion matrices: 13-class and binary arc detection
   - Parallel processing with --workers N (default: 7, max: 28)
   - Precision-recall curves in addition to ROC curves
   - Evaluate on train/test/verification splits
   - Binary arc detection: map 13 classes → 2 (arc vs. non-arc)
   - Category-based classification using segment_labels.category

Changes in previous version (8.3):
1. BUGFIX - Include custom function_type in distance function query
   - v2.0.8.3: Removed function_type='builtin' filter
   - Now includes Pearson (function_type='custom') that uses sklearn.metrics.pairwise
   - Query filters only by library_name='sklearn.metrics.pairwise'

Changes in previous version (8.2):
1. ENHANCEMENT - Use sklearn pairwise distances from ml_distance_functions_lut
   - v2.0.8.2: Replaced manual distance calculations with sklearn.metrics.pairwise.pairwise_distances
   - Queries distance functions from ml_distance_functions_lut (builtin, sklearn.metrics.pairwise)
   - Uses configured pairwise_metric_name (manhattan, euclidean, cosine, correlation)
   - Converts NaN values to 0 (handles zero-variance/zero-length vectors)
   - Removed manual L1/L2/Cosine/Pearson functions
   - Removed NaN check (now handled in compute_distance)

Changes in previous version (8.1):
1. BUGFIX - classifier-build-features junction table name
   - v2.0.8.1: Fixed query to use ml_classifier_config_experiment_feature_sets
   - Was using incorrect table name ml_classifier_config_feature_sets
   - Now correctly queries experiment_feature_set_id from config

Changes in previous version (8.0):
1. PHASE 3 START - Feature Vector Construction
   - v2.0.8.0: Implemented classifier-build-features command (590 lines)
   - Builds distance-based SVM feature vectors for ALL segments
   - Feature dimensions: num_classes × 4 distance metrics
   - For experiment 041: 13 × 4 = 52-dimensional vectors
   - Distance metrics: L1 (Manhattan), L2 (Euclidean), Cosine, Pearson
   - Creates experiment_{exp}_classifier_{cls}_svm_features table
   - Saves feature vectors to .npy files
   - Progress tracking with ETA display
   - Batch commits for performance
   - Error handling with extraction_status_id tracking
   - Filters: --decimation-factor, --data-type, --amplitude-method, --feature-set
   - Options: --force, --batch-size
   - Estimated runtime: 4-6 hours for full dataset

Changes in previous version (7.9):
1. ENHANCEMENT - Added label names to feature plots
   - v2.0.7.9: Query segment_labels table to get actual label name
   - Display in plot title: "Class: arc_discharge (ID=1)"
   - Include in filename: exp041_cls001_arc_discharge_id1_seg12345_...
   - Sanitize label names for filesystem (spaces -> underscores)
   - Shows meaningful class names instead of just numeric IDs

Changes in previous version (7.8):
1. NEW COMMAND - classifier-plot-reference-features
   - v2.0.7.8: Added command to plot ACTUAL FEATURE DATA of selected segments
   - Shows the actual waveforms/time series of selected reference segments
   - One plot per reference segment showing all features (voltage, current, etc.)
   - Multi-subplot layout with one subplot per feature
   - Filters: --decimation-factor, --data-type, --amplitude-method, --feature-set
   - THIS is what the user wanted - plots of the actual selected segments!

Changes in previous version (7.7):
1. NEW COMMAND - classifier-plot-references
   - v2.0.7.7: Added separate command to generate PCA plots from existing references
   - Reads reference segments from database
   - Regenerates PCA analysis and visualization
   - Allows plot generation without re-running expensive reference selection
   - Options: --plot-dir <path>, --pca-components <n>
   - Useful for changing plot directory or regenerating after data issues

Changes in previous version (7.6):
1. BUG FIX - Fixed self.current_experiment access
   - v2.0.7.6: Fixed classifier-drop-references-table command
   - Changed: exp_id = self.current_experiment['experiment_id']
   - To: exp_id = self.current_experiment
   - self.current_experiment is an integer, not a dictionary
   - Error: "'int' object is not subscriptable"

Changes in previous version (7.5):
1. REGISTRATION FIX - Registered classifier-drop-references-table command
   - v2.0.7.5: Added command to self.commands dictionary
   - Added tab completion support with --confirm and --help flags
   - Command was implemented but not registered (lazy mistake!)

Changes in previous version (7.4):
1. SCHEMA FIX - Updated reference_segments table schema
   - v2.0.7.4: Made feature_set_feature_id NULLABLE (was NOT NULL)
   - Removed feature_set_feature_id from UNIQUE constraint
   - Added classifier-drop-references-table command to recreate table
   - UNIQUE constraint now: (segment_label_id, decimation_factor,
     data_type_id, amplitude_processing_method_id, experiment_feature_set_id)
   - Allows NULL feature_set_feature_id for multi-feature feature_sets

Changes in previous version (7.3):
1. CRITICAL ARCHITECTURE FIX - Feature concatenation within feature_sets
   - v2.0.7.3: Rewrote loop structure to properly handle multi-feature feature_sets
   - REMOVED inner loop over individual features
   - Now queries ALL features belonging to a feature_set
   - Loads each feature separately and concatenates in order
   - Result vectors:
     * efs=1 (current_only, feature 18): (8192,)
     * efs=2 (voltage_only, feature 16): (8192,)
     * efs=5 (all_electrical, features 16+18): (16384,) concatenated
   - Fixed plot grouping to use experiment_feature_set_id only
   - Removed obsolete feature_set_feature_id from plot titles/filenames

Changes in previous version (7.2):
1. CRITICAL FIX - Feature file amplitude column selection
   - v2.0.7.2: Fixed feature loading to select correct amplitude method column
   - Feature files have shape (n_samples, 2) with TWO amplitude methods:
     * Column 0: amplitude_method_id 1 (minmax normalization)
     * Column 1: amplitude_method_id 2 (zscore normalization)
   - Now selects correct column: column_idx = amplitude_method_id - 1
   - Prevents PCA "Found array with dim 3" error
   - Each feature vector is now 1D: (8192,) instead of (8192, 2)

Changes in previous version (7.1):
1. PHASE 2 START - Reference Segment Selection
   - v2.0.7.1: Implemented classifier-select-references command
   - Uses PCA dimensionality reduction to 2D + centroid analysis
   - Selects one representative reference segment per class
   - Creates experiment_{exp}_classifier_{cls}_reference_segments table
   - Loads feature vectors from .npy files
   - Performs stratified selection by segment_label_id
   - Optional PCA visualization plots (--plot option)
   - Configurable minimum segments per class (--min-segments)
   - Stores PCA metadata (centroid coordinates, explained variance)
   - Ready for Phase 3 (Feature Vector Construction)

Changes in previous version (7.0):
1. MILESTONE RELEASE - Phase 0b and Phase 1 Complete
   - v2.0.7.0: Phase 0b (Configuration Management) fully implemented and tested
   - v2.0.7.0: Phase 1 (Data Split Assignment) fully implemented and tested
   - All three Phase 1 commands working: classifier-create-splits-table,
     classifier-assign-splits, classifier-show-splits
   - Feature builder integration completed
   - Stratified sampling verified
   - Ready to proceed to Phase 2 (Reference Selection)

Changes in previous version (6.36):
1. REFACTOR - Integrated feature builder into classifier-config-show
   - v2.0.6.36: Removed standalone classifier-config-show-feature-builder command
   - Integrated feature builder settings into classifier-config-show output
   - classifier-config-show now displays:
     * All hyperparameters (decimation factors, data types, etc.)
     * Feature builder settings section
     * Shows if no feature builder is configured
   - Removed redundant command from registration, tab completion, and help
   - Updated help text to indicate classifier-config-show includes feature builder
   - Cleaner, more unified interface

Changes in previous version (6.35):
1. FEATURE BUILDER MANAGEMENT - Phase 0b Enhancement
   - v2.0.6.35: Implemented feature builder management commands
   - classifier-config-set-feature-builder: Create/update feature builder flags
   - Updated classifier-config-list: Added "FeatBuilder" column (Yes/No)
   - Feature builder controls which feature types are included in X matrix:
     * include_original_feature: Raw feature values
     * compute_baseline_distances_inter: Distances to OTHER class baselines
     * compute_baseline_distances_intra: Distances to SAME class baseline
     * statistical_features: Reserved for future use
     * external_function: Reserved for future use
   - Updated tab completion and help system

Changes in previous version (6.34):
1. CRITICAL FIX - Command Registration
   - v2.0.6.34: Registered Phase 0b and Phase 1 commands in command handler
   - Added missing Phase 0b commands: classifier-config-list, classifier-config-activate,
     classifier-config-show, classifier-create-feature-builder-table
   - Added Phase 1 commands: classifier-create-splits-table, classifier-assign-splits,
     classifier-show-splits
   - Commands were implemented but not registered in self.commands dictionary

Changes in previous version (6.33):
1. PHASE 1 COMPLETION - Data Split Assignment System
   - v2.0.6.33: Updated help system with Phase 1 documentation
   - Added comprehensive help for all three split commands
   - Documented all command options and default values
   - Included usage examples for common scenarios
   - Phase 1 implementation complete (ready for commit)

Changes in previous version (6.32):
1. PHASE 1 - Data Split Assignment System (continued)
   - v2.0.6.32: Updated tab completion for Phase 1 commands
   - Added classifier-create-splits-table completion
   - Added classifier-assign-splits completion with all options
   - Added classifier-show-splits completion with filters

Changes in previous version (6.31):
1. PHASE 1 - Data Split Assignment System (continued)
   - v2.0.6.31: Implemented classifier-show-splits command
   - Displays split statistics with tabulated output
   - Summary view shows segments and classes per split type
   - Detail view shows per-class breakdown for each combination
   - Supports filtering by decimation_factor and data_type
   - Displays split percentages and totals

Changes in previous version (6.30):
1. PHASE 1 - Data Split Assignment System (continued)
   - v2.0.6.30: Implemented classifier-assign-splits command
   - Stratified sampling using sklearn.model_selection.train_test_split
   - Configurable train/test/verification ratios (default: 70/20/10)
   - Uses active configuration to determine decimation_factors and data_types
   - Processes all (decimation_factor, data_type_id) combinations
   - Per-class stratification ensures balanced representation
   - Random seed for reproducibility

Changes in previous version (6.29):
1. PHASE 1 START - Data Split Assignment System
   - v2.0.6.29: Implemented classifier-create-splits-table command
   - Creates per-classifier data_splits table for train/test/verification assignments
   - Table schema includes stratification fields (segment_label_id)
   - UNIQUE constraint on (segment_id, decimation_factor, data_type_id)
   - Four indexes for efficient querying by split_type, segment, label, and experiment data

Changes in previous version (6.28):
1. PHASE 0b COMPLETION - Configuration Management System
   - v2.0.6.26: Implemented classifier-config-list command with ARRAY_AGG queries
   - v2.0.6.27: Implemented classifier-config-activate command
   - v2.0.6.28: Implemented classifier-config-show command with detailed hyperparameter display
   - Updated classifier-new to retrieve and display global_classifier_id
   - Updated tab completion for all Phase 0b commands
   - Updated help system with comprehensive Phase 0b documentation

Changes in previous version (6.25):
1. Add classifier-create-feature-builder-table command
   - Creates ml_classifier_feature_builder table for feature vector construction
   - Boolean flags: include_original_feature, compute_baseline_distances_inter/intra
   - FK constraint to ml_classifier_configs with CASCADE delete
   - UNIQUE constraint on config_id (one feature builder per config)
   - References per-classifier reference_segments tables from Phase 2

Changes in previous version (6.24):
1. Add classifier-config-add-feature-sets command
   - Helper command to add feature sets to existing configs
   - Looks up experiment_feature_set_ids from feature_set_ids
   - Inserts into ml_classifier_config_experiment_feature_sets junction table

Changes in previous version (6.23):
1. Add classifier-migrate-configs-to-global command
   - Migrates data from old per-classifier tables to global ml_classifier_configs
   - Preserves config_id values for junction table compatibility
   - Must run BEFORE adding FK constraints

Changes in previous version (6.22):
1. CRITICAL FIX: Change config table design from per-classifier to GLOBAL
   - Add classifier-create-global-config-table command
   - Creates ml_classifier_configs (global for ALL classifiers)
   - Add classifier-add-config-foreign-keys command
   - Adds FK constraints from junction tables to global config table

Changes in previous version (6.21):
1. Phase 0b Task 0b.7: Implement classifier-config-delete command (DEPRECATED - used per-classifier tables)

Changes in previous version (6.20):
1. FEATURE: Add feature set support to classifier-config-create
   - Query all feature sets when --feature-sets all (default)
   - Validate feature set IDs against experiment configuration
   - Insert into ml_classifier_config_experiment_feature_sets junction table

Changes in previous version (6.19):
1. BUGFIX: Fixed argument parsing in classifier-config-create and classifier-migrate-config-tables
   - Remove incorrect shlex.split() usage (args is already a list)

Changes in previous version (6.18):
1. Phase 0b Task 0b.3: Rewrite classifier-config-create with junction tables
   - Complete rewrite to use normalized junction tables
   - Get global_classifier_id from ml_experiment_classifiers
   - Insert config metadata into per-classifier table
   - Insert hyperparameters into 6 global junction tables

Changes in previous version (6.17):
1. Phase 0b Task 0b.2: Drop/recreate config tables with new schema
   - Add classifier-migrate-config-tables command
   - Drop old array-based config tables

Changes in previous version (6.16):
1. Phase 0b Task 0b.1: Create 6 global junction tables
   - Add classifier-create-junction-tables command
   - Create all 6 junction tables with 4-part composite PKs
   - Enable CASCADE deletes and foreign key validation

Changes in previous version (6.15):
1. Phase 0b Task 0b.0: Migrate ml_experiment_classifiers
   - Add classifier-migrate-registry command
   - Add global_classifier_id as new PRIMARY KEY
   - Maintain (experiment_id, classifier_id) as UNIQUE constraint

Changes in previous version (6.14):
1. Phase 0b Task 1: Implement classifier-config-create command (DEPRECATED - needs rewrite)
   - Initial array-based implementation
   - Fixed SQL column name bug (df.function_name)
   - Added data type name-to-ID conversion

Changes in previous version (6.13):
1. LOCAL COMMIT: Phase 0a Classifier Registry Setup complete

Phase 0a Summary (versions 6.6-6.12):
1. Created ml_experiment_classifiers table (global registry)
2. Implemented all classifier lifecycle commands
3. Session context and prompt updates
4. Tab completion and help system

Changes in previous version (6.12):
1. Task 9: Added classifier commands to help system

Changes in version (6.11):
1. Task 7: Added set classifier command
2. Task 8: Updated MLDPCompleter with tab completion

Changes in version (6.9):
1. Added classifier-list command (Phase 0a Task 4)

Changes in version (6.8):
1. Added classifier-remove command (Phase 0a Task 3)

Changes in version (6.7):
1. Added classifier-new command (Phase 0a Task 2)

COMPLETE PIPELINE AUTOMATION: All 7 steps now run unattended!
The pipeline is now perfect for automation:
  clean-experiment --force
  mpcctl-execute-experiment --workers 20 --log --verbose --force
"""

# Version tracking
VERSION = "2.0.10.12"  # MAJOR.MINOR.COMMIT.CHANGE

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter, Completer, Completion
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from prompt_toolkit.shortcuts import clear
import subprocess
import sys
import os
from pathlib import Path
import psycopg2
from tabulate import tabulate
import json
from datetime import datetime
import shlex
import argparse

# Path to MLDP main project (mldp_cli is now a submodule inside mldp)
MLDP_ROOT = Path(__file__).parent.parent.parent

# Define style for the prompt
style = Style.from_dict({
    'prompt': '#00aa00 bold',
    'experiment': '#0088ff',
    'distance': '#ff8800',
    'separator': '#666666',
})


class MLDPCompleter(Completer):
    """Custom completer for MLDP commands"""
    
    def __init__(self):
        self.commands = {
            # Database commands
            'connect': ['localhost', '5432', 'arc_detection', 'kjensen'],
            'query': ['SELECT', 'FROM', 'WHERE', 'LIMIT', 'ORDER BY', 'GROUP BY'],
            'sql': ['SELECT', 'DROP', 'TRUNCATE', 'UPDATE', 'INSERT', 'DELETE', 'ALTER', 'CREATE', 'FROM', 'WHERE', 'CASCADE'],
            'tables': [],
            'browser': [],
            
            # Experiment commands
            'experiment-list': [],
            'experiment-info': ['17', '18', '19', '20'],
            'experiment-config': ['17', '18', '19', '20', '--json'],
            'experiment-summary': ['17', '18', '19', '20'],
            'experiment-generate': ['balanced', 'small', 'large', '--dry-run'],
            'experiment-create': ['--name', '--max-files', '--segment-sizes', '--data-types', '--help'],
            
            # Distance commands
            'calculate': ['--segment-size', '--distance-type', '--workers', '8192', '16384', '32768', 'euclidean', 'l1', 'l2', 'cosine'],
            'insert_distances': ['--input-folder', '--distance-type', 'l1', 'l2', 'cosine', 'pearson'],
            'mpcctl-distance-function': ['--start', '--status', '--pause', '--continue', '--stop', '--workers', '--feature_sets', '--log', '--verbose', '--force', '--clean', '--resume'],
            'mpcctl-distance-insert': ['--start', '--status', '--pause', '--continue', '--stop', '--list-processes', '--kill', '--kill-all', '--workers', '--distances', '--method', '--batch-size', '--log', '--verbose', '--force'],
            'mpcctl-execute-experiment': ['--workers', '--log', '--verbose', '--force', '--skip-file-selection', '--skip-segment-selection', '--skip-segment-fileset', '--skip-segment-pairs', '--skip-feature-fileset', '--skip-distance-calc', '--skip-distance-insert', '--help'],

            # Visualization
            'heatmap': ['--version', '--output-dir', '1', '2', '3', '4', '5', '6', '7'],
            'histogram': ['--version', '--bins', '1_0', '1_1', '1_2', '1_3', '50', '100'],
            'visualize': ['--segment-id', '--file-id'],
            'segment-plot': ['--amplitude-method', '--original-segment', '--result-segment-size', '--types',
                           '--decimations', '--output-folder', 'raw', 'minmax', 'zscore', 'amplitude_0', 'amplitude_1',
                           'RAW', 'ADC6', 'ADC8', 'ADC10', 'ADC12', 'ADC14', '0', '7', '15'],
            'feature-plot': ['--file', '--save', '--output-folder'],

            # Distance calculations
            'init-distance-tables': ['--drop-existing', '--help'],
            'show-distance-metrics': [],
            'add-distance-metric': ['--metric', 'L1', 'L2', 'cosine', 'pearson', 'euclidean', 'manhattan', 'wasserstein'],
            'remove-distance-metric': ['--metric', '--all-except', 'L1', 'L2', 'cosine', 'pearson'],
            'clean-distance-tables': ['--dry-run', '--force'],
            'show-distance-functions': ['--active-only'],
            'update-distance-function': ['--pairwise-metric', '--library', '--function-import', '--description', '--active'],
            'mpcctl-distance-function': ['--start', '--pause', '--continue', '--stop', '--status', '--workers', '--log', '--verbose', '--help'],

            # Analysis
            'stats': ['l1', 'l2', 'cosine', 'pearson'],
            'closest': ['10', '20', '50', '100'],
            
            # Experiments
            'select-segments': ['--strategy', '--segments-per-type', '--seed', '--clean', '--force', '--help', '41', '42', '43'],
            'clean-segment-table': ['--force', '41', '42', '43'],
            'clean-segment-pairs': ['41', '42', '43'],
            'clean-feature-files': ['41', '42', '43'],
            'generate-segment-pairs': ['--strategy', '--max-pairs-per-segment', '--same-label-ratio', '--seed', '--clean', '--help'],
            'generate-feature-fileset': ['--feature-sets', '--max-segments', '--force', '--clean', '--help'],
            'update-decimations': ['0', '1', '3', '7', '15', '31', '63', '127', '255', '511'],
            'add-decimation': ['0', '1', '3', '7', '15', '31', '63', '127', '255', '511'],
            'remove-decimation': ['0', '1', '3', '7', '15', '31', '63', '127', '255', '511'],
            'update-segment-sizes': ['128', '256', '512', '1024', '2048', '4096', '8192', '16384', '32768', '65536', '131072', '262144'],
            'update-amplitude-methods': ['minmax', 'zscore', 'maxabs', 'robust', 'TRAW', 'TADC14', 'TADC12', 'TADC10', 'TADC8', 'TADC6'],
            'create-feature-set': ['--name', '--features', '--n-value', 'voltage', 'current', 'impedance', 'power'],
            'remove-feature-set': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
            'clear-feature-sets': [],
            'list-feature-sets': [],
            'remove-data-type': ['1', '2', '3', '4', '5', '6', '7'],
            'add-data-type': ['1', '2', '3', '4', '5', '6', '7'],
            'list-data-types': [],
            'list-all-data-types': [],
            'list-amplitude-methods': [],
            'list-all-amplitude-methods': [],
            'show-all-feature-sets': [],
            # New feature management commands
            'create-feature': ['--name', '--category', '--behavior', '--description', 'electrical', 'statistical', 'spectral', 'temporal', 'compute', 'driver', 'derived', 'aggregate', 'transform'],
            'list-features': ['--category', 'electrical', 'statistical', 'spectral', 'temporal', 'compute'],
            'show-feature': [],
            'update-feature': ['--name', '--category', '--description'],
            'delete-feature': [],
            'create-global-feature-set': ['--name', '--category', '--description', 'electrical', 'statistical', 'custom'],
            'add-features-to-set': ['--features'],
            'remove-features-from-set': ['--features'],
            'clone-feature-set': ['--name'],
            'link-feature-set': ['--n-value', '--channel', '--priority', 'load_voltage', 'source_current'],
            'bulk-link-feature-sets': ['--sets', '--n-values'],
            'update-feature-link': ['--n-value', '--priority', '--active'],
            'show-feature-config': [],
            'update-selection-config': ['--max-files', '--seed', '--strategy', '--balanced', '10', '25', '50', '100'],
            'select-files': ['--max-files', '--label', '--seed', '50', '100'],
            'remove-files': ['--label', '--file-ids'],
            'remove-file-labels': ['trash', 'voltage_only', 'arc_short_gap', 'arc_extinguish', 'other'],
            'remove-segments': ['--label', '--segment-ids'],

            # Data management commands
            'get-experiment-data-path': [],
            'set-experiment-data-path': ['--reset'],
            'clean-segment-files': ['--dry-run'],
            'clean-feature-files': ['--dry-run', '--force', '--files-and-tables', '--files-only', '--tables-only'],
            'clean-distance-work-files': ['--dry-run', '--force'],

            # Settings
            'set': ['experiment', 'distance', 'classifier', '18', 'l1', 'l2', 'cosine', 'none'],
            'show': [],
            
            # Server Management
            'servers': ['start', 'stop', 'restart', 'status', 'logs'],
            'start': [],
            'stop': [],
            'restart': [],
            'status': [],
            'logs': ['real_time_sync_hub', 'database_browser', 'data_cleaning_tool',
                    'transient_viewer', 'segment_visualizer', 'distance_visualizer',
                    'experiment_generator', 'jupyter_integration', 'segment_verifier'],

            # Classifier Management (Phase 0a)
            'classifier-create-registry': [],
            'classifier-migrate-registry': ['--force', '--help'],
            'classifier-create-junction-tables': ['--force', '--help'],
            'classifier-migrate-config-tables': ['--experiment-id', '--classifier-id', '--all', '--force', '--help'],
            'classifier-new': ['--name', '--description', '--type', '--auto-select', '--no-auto-select', '--help'],
            'classifier-remove': ['--classifier-id', '--confirm', '--archive-instead', '--help'],
            'classifier-list': ['--include-archived', '--show-tables', '--help'],

            # Classifier Configuration Management (Phase 0b)
            'classifier-config-create': ['--config-name', '--decimation-factors', '--data-types',
                                        '--amplitude-methods', '--feature-sets', '--features',
                                        '--distance-functions', '--set-active', '--notes', '--help'],
            'classifier-config-list': ['--all', '--experiment-id', '--classifier-id', '--help'],
            'classifier-config-activate': ['--config-name', '--config-id', '--help'],
            'classifier-config-show': ['--config-name', '--config-id', '--active', '--help'],
            'classifier-config-delete': ['--config-name', '--config-id', '--confirm', '--help'],
            'classifier-config-add-feature-sets': ['--config-id', '--feature-sets', '--experiment-id', '--help'],
            'classifier-config-add-hyperparameters': ['--config', '--amplitude-methods', '--decimation-factors',
                                                     '--data-types', '--distance-functions',
                                                     '--experiment-feature-sets', '--help'],
            'classifier-config-remove-hyperparameters': ['--config', '--amplitude-methods', '--decimation-factors',
                                                        '--data-types', '--distance-functions',
                                                        '--experiment-feature-sets', '--force', '--help'],
            'classifier-create-global-config-table': ['--force', '--help'],
            'classifier-add-config-foreign-keys': ['--help'],
            'classifier-migrate-configs-to-global': ['--experiment-id', '--classifier-id', '--help'],
            'classifier-create-feature-builder-table': ['--force', '--help'],
            'classifier-config-set-feature-builder': ['--config-id', '--include-original', '--no-include-original',
                                                     '--compute-distances-inter', '--no-compute-distances-inter',
                                                     '--compute-distances-intra', '--no-compute-distances-intra',
                                                     '--statistical-features', '--no-statistical-features',
                                                     '--external-function', '--no-external-function',
                                                     '--notes', '--help'],

            # Classifier Data Split Assignment (Phase 1)
            'classifier-create-splits-table': ['--force', '--help'],
            'classifier-assign-splits': ['--train-ratio', '--test-ratio', '--verification-ratio',
                                        '--seed', '--force', '--help'],
            'classifier-copy-splits-from': ['--source-classifier', '--force', '--help'],
            'classifier-show-splits': ['--decimation-factor', '--data-type', '--detail', '--help'],

            # Classifier Reference Selection (Phase 2)
            'classifier-drop-references-table': ['--confirm', '--help'],
            'classifier-select-references': ['--force', '--plot', '--plot-dir', '--min-segments',
                                            '--pca-components', '--help'],
            'classifier-copy-reference-segments': ['--source-classifier', '--force', '--help'],
            'classifier-plot-references': ['--plot-dir', '--pca-components', '--help'],
            'classifier-plot-reference-features': ['--plot-dir', '--decimation-factor', '--data-type',
                                                   '--amplitude-method', '--feature-set', '--help'],

            # Classifier Feature Construction (Phase 3)
            'classifier-build-features': ['--force', '--batch-size', '--decimation-factor', '--data-type',
                                         '--amplitude-method', '--feature-set', '--help'],

            # Classifier SVM Training (Phase 4)
            'classifier-train-svm-init': ['--help'],
            'classifier-train-svm': ['--workers', '--decimation-factor', '--data-type', '--amplitude-method',
                                    '--feature-set', '--kernel', '--C', '--gamma', '--force', '--help'],
            'classifier-test-svm-single': ['--decimation-factor', '--data-type', '--amplitude-method',
                                          '--feature-set', '--kernel', '--C', '--gamma', '--help'],
            'classifier-clean-svm-results': ['--amplitude-method', '--decimation-factor', '--data-type',
                                            '--feature-set', '--force', '--help'],
            'classifier-clean-features': ['--amplitude-method', '--decimation-factor', '--data-type',
                                         '--feature-set', '--force', '--help'],

            # Utilities
            'verify': [],
            'clear': [],
            'export': [],
            'time': [],
            'help': [],  # Will be populated with all command names
            'exit': [],
            'quit': [],
        }
        # Add all command names to help completions
        self.commands['help'] = list(self.commands.keys())
    
    def get_completions(self, document, complete_event):
        """Get completions for current input"""
        text = document.text_before_cursor
        words = text.split()
        
        if not words:
            # Complete commands
            for cmd in self.commands.keys():
                yield Completion(cmd, start_position=0)
        elif len(words) == 1:
            # Still completing the command
            word = words[0]
            for cmd in self.commands.keys():
                if cmd.startswith(word):
                    yield Completion(cmd, start_position=-len(word))
        else:
            # Complete command arguments
            cmd = words[0]
            if cmd in self.commands:
                current_word = words[-1] if len(text) > 0 and not text.endswith(' ') else ''
                start_pos = -len(current_word) if current_word else 0
                
                for option in self.commands[cmd]:
                    if not current_word or option.startswith(current_word):
                        yield Completion(option, start_position=start_pos)



# ========== SVM Training Worker Function (Phase 4 Step 3) ==========
# This function must be defined at module level (not inside a class) for multiprocessing

def train_svm_worker(config_tuple):
    """
    Train one SVM model for a specific configuration

    This function runs in a subprocess via multiprocessing.Pool.

    Args:
        config_tuple: Tuple containing:
            - dec: Decimation factor
            - dtype: Data type ID
            - amp: Amplitude processing method ID
            - efs: Experiment feature set ID
            - svm_params: Dictionary with 'kernel', 'C', 'gamma' (optional)
            - db_config: Dictionary with database connection parameters
            - label_categories: Dictionary mapping label_id to category ('arc', 'normal', 'transient')
            - exp_id: Experiment ID
            - cls_id: Classifier ID
            - global_classifier_id: Global classifier ID

    Returns:
        Dictionary with training results and metrics
    """
    import warnings
    warnings.filterwarnings('ignore')  # Suppress all warnings to prevent pipe buffer overflow

    import numpy as np
    import psycopg2
    import time
    import os
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix, roc_auc_score, roc_curve,
                                 precision_recall_curve, average_precision_score)
    from sklearn.model_selection import cross_val_score
    import joblib
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Unpack configuration
    dec, dtype, amp, efs, svm_params, db_config, label_categories, exp_id, cls_id, global_classifier_id = config_tuple

    # Timing dictionary for diagnostics
    timings = {}
    t_total_start = time.time()

    try:
        # CHECKPOINT 1: Database connection
        t_db_start = time.time()
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        timings['db_connection'] = time.time() - t_db_start

        # CHECKPOINT 2: Feature loading (split by train/test/verify)
        t_load_train_start = time.time()
        X_train, y_train = load_feature_vectors_from_db_worker(
            cursor, exp_id, cls_id, dec, dtype, amp, efs, 'training'
        )
        timings['feature_load_train'] = time.time() - t_load_train_start

        t_load_test_start = time.time()
        X_test, y_test = load_feature_vectors_from_db_worker(
            cursor, exp_id, cls_id, dec, dtype, amp, efs, 'test'
        )
        timings['feature_load_test'] = time.time() - t_load_test_start

        t_load_verify_start = time.time()
        X_verify, y_verify = load_feature_vectors_from_db_worker(
            cursor, exp_id, cls_id, dec, dtype, amp, efs, 'verification'
        )
        timings['feature_load_verify'] = time.time() - t_load_verify_start

        timings['feature_load_total'] = (timings['feature_load_train'] +
                                          timings['feature_load_test'] +
                                          timings['feature_load_verify'])

        # CHECKPOINT 3: SVM training
        t_svm_start = time.time()

        # Build SVM parameters
        svm_kwargs = {
            'kernel': svm_params['kernel'],
            'C': svm_params['C'],
            'class_weight': 'balanced',
            'random_state': 42,
            'probability': True  # Required for ROC/PR curves
        }

        # Add gamma for rbf and poly kernels
        if svm_params['kernel'] in ['rbf', 'poly'] and svm_params.get('gamma'):
            svm_kwargs['gamma'] = svm_params['gamma']

        svm = SVC(**svm_kwargs)
        svm.fit(X_train, y_train)
        training_time = time.time() - t_svm_start
        timings['svm_training'] = training_time

        # CHECKPOINT 4: Cross-validation (SKIPPED - too slow with large datasets)
        # Cross-validation was taking 20-30 minutes per task (5x the training time)
        # With 84 tasks, this would add 28-42 hours to total training time
        # We have a separate test set, so CV is not necessary
        cv_mean = 0.0
        cv_std = 0.0
        timings['cross_validation'] = 0.0

        # CHECKPOINT 5: Predictions
        t_pred_start = time.time()
        y_pred_train = svm.predict(X_train)
        y_pred_test = svm.predict(X_test)
        y_pred_verify = svm.predict(X_verify)

        # Get probability scores for ROC/PR curves
        y_proba_train = svm.predict_proba(X_train)
        y_proba_test = svm.predict_proba(X_test)
        y_proba_verify = svm.predict_proba(X_verify)
        prediction_time = time.time() - t_pred_start
        timings['prediction'] = prediction_time

        # CHECKPOINT 6: Metrics computation
        t_metrics_start = time.time()
        # Compute 13-class metrics using helper function
        metrics_train = compute_multiclass_metrics_worker(y_train, y_pred_train)
        metrics_test = compute_multiclass_metrics_worker(y_test, y_pred_test)
        metrics_verify = compute_multiclass_metrics_worker(y_verify, y_pred_verify)

        # Compute binary arc detection metrics using helper function
        binary_metrics_train = compute_binary_arc_metrics_worker(
            y_train, y_pred_train, y_proba_train, label_categories
        )
        binary_metrics_test = compute_binary_arc_metrics_worker(
            y_test, y_pred_test, y_proba_test, label_categories
        )
        binary_metrics_verify = compute_binary_arc_metrics_worker(
            y_verify, y_pred_verify, y_proba_verify, label_categories
        )
        timings['metrics_computation'] = time.time() - t_metrics_start

        # CHECKPOINT 7: Model saving
        t_save_model_start = time.time()
        model_path = save_svm_model_worker(svm, exp_id, cls_id, dec, dtype, amp, efs, svm_params)
        timings['save_model'] = time.time() - t_save_model_start

        # CHECKPOINT 8: Visualization saving
        t_save_viz_start = time.time()
        # Save confusion matrices using helper function
        cm_paths = save_confusion_matrices_worker(
            y_train, y_pred_train, y_test, y_pred_test,
            y_verify, y_pred_verify, label_categories,
            exp_id, cls_id, dec, dtype, amp, efs, svm_params
        )

        # Save ROC and PR curves using helper function
        curve_paths = save_curves_worker(
            y_train, y_proba_train, y_test, y_proba_test,
            y_verify, y_proba_verify, label_categories,
            exp_id, cls_id, dec, dtype, amp, efs, svm_params
        )
        timings['save_visualizations'] = time.time() - t_save_viz_start

        # Close database connection
        cursor.close()
        conn.close()

        # Calculate total time
        timings['total'] = time.time() - t_total_start

        # Calculate data sizes for diagnostics
        data_sizes = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'verify_samples': len(X_verify),
            'feature_dim': X_train.shape[1] if len(X_train) > 0 else 0
        }

        # Return results
        return {
            'success': True,
            'config': (dec, dtype, amp, efs),
            'svm_params': svm_params,
            'global_classifier_id': global_classifier_id,
            'training_time': training_time,
            'prediction_time': prediction_time,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'metrics_train': metrics_train,
            'metrics_test': metrics_test,
            'metrics_verify': metrics_verify,
            'binary_metrics_train': binary_metrics_train,
            'binary_metrics_test': binary_metrics_test,
            'binary_metrics_verify': binary_metrics_verify,
            'model_path': model_path,
            'cm_paths': cm_paths,
            'curve_paths': curve_paths,
            'unique_labels': np.unique(y_train).tolist(),
            'timings': timings,
            'data_sizes': data_sizes
        }

    except Exception as e:
        import traceback
        # Calculate partial timings if available
        if 'timings' in locals():
            timings['total'] = time.time() - t_total_start
        else:
            timings = {'total': time.time() - t_total_start if 't_total_start' in locals() else 0}

        return {
            'success': False,
            'config': (dec, dtype, amp, efs),
            'svm_params': svm_params,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timings': timings
        }


# ========== Worker Helper Functions (duplicated for multiprocessing) ==========
# These functions are copies of the class methods but standalone for worker processes

def load_feature_vectors_from_db_worker(cursor, exp_id, cls_id, dec, dtype, amp, efs, split_type):
    """Load feature vectors for worker process"""
    import numpy as np

    cursor.execute(f"""
        SELECT sf.segment_id, sf.segment_label_id, sf.svm_feature_file_path
        FROM experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_features sf
        JOIN experiment_{exp_id:03d}_classifier_{cls_id:03d}_data_splits ds
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
        raise ValueError(f"No feature vectors found for config: dec={dec}, dtype={dtype}, amp={amp}, efs={efs}, split={split_type}")

    X = []
    y = []

    for segment_id, label_id, file_path in rows:
        try:
            features = np.load(file_path)
            X.append(features)
            y.append(label_id)
        except Exception:
            pass  # Skip failed loads silently in worker

    if len(X) == 0:
        raise ValueError(f"No valid feature vectors loaded for split {split_type}")

    return np.array(X), np.array(y)


def compute_multiclass_metrics_worker(y_true, y_pred):
    """Compute multiclass metrics for worker process"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(y_true, y_pred, zero_division=0)

    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'per_class': {
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class,
            'support': support_per_class
        }
    }


def compute_binary_arc_metrics_worker(y_true, y_pred, y_proba, label_categories):
    """Compute binary arc metrics for worker process"""
    import numpy as np
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix, roc_auc_score, average_precision_score)

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
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0

    try:
        if len(np.unique(y_true_binary)) > 1:
            roc_auc = roc_auc_score(y_true_binary, y_proba_arc)
        else:
            roc_auc = np.nan
    except Exception:
        roc_auc = np.nan

    try:
        if len(np.unique(y_true_binary)) > 1:
            pr_auc = average_precision_score(y_true_binary, y_proba_arc)
        else:
            pr_auc = np.nan
    except Exception:
        pr_auc = np.nan

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def save_confusion_matrices_worker(y_train, y_pred_train, y_test, y_pred_test,
                                   y_verify, y_pred_verify, label_categories,
                                   exp_id, cls_id, dec, dtype, amp, efs, svm_params):
    """Save confusion matrices for worker process"""
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    base_dir = f"/Volumes/ArcData/V3_database/experiment{exp_id:03d}/classifier_files/svm_results"
    classifier_dir = f"classifier_{cls_id:03d}"
    config_dir = f"D{dec:06d}_TADC{dtype}_A{amp}_EFS{efs:03d}"
    svm_dir = f"{svm_params['kernel']}_C{svm_params['C']}"
    if svm_params.get('gamma'):
        svm_dir += f"_G{svm_params['gamma']}"

    full_dir = os.path.join(base_dir, classifier_dir, config_dir, svm_dir)
    os.makedirs(full_dir, exist_ok=True)

    paths = {}

    # 13-class confusion matrices
    for split_name, y_true, y_pred in [
        ('train', y_train, y_pred_train),
        ('test', y_test, y_pred_test),
        ('verify', y_verify, y_pred_verify)
    ]:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'13-Class Confusion Matrix ({split_name.title()} Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        filename = f"confusion_matrix_13class_{split_name}.png"
        filepath = os.path.join(full_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        paths[f'cm_13class_{split_name}'] = filepath

    # Binary arc detection confusion matrices
    for split_name, y_true, y_pred in [
        ('train', y_train, y_pred_train),
        ('test', y_test, y_pred_test),
        ('verify', y_verify, y_pred_verify)
    ]:
        y_true_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_true])
        y_pred_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_pred])
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                   xticklabels=['Non-arc', 'Arc'],
                   yticklabels=['Non-arc', 'Arc'])
        plt.title(f'Binary Arc Detection ({split_name.title()} Set)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        filename = f"confusion_matrix_binary_{split_name}.png"
        filepath = os.path.join(full_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        paths[f'cm_binary_{split_name}'] = filepath

    return paths


def save_curves_worker(y_train, y_proba_train, y_test, y_proba_test,
                      y_verify, y_proba_verify, label_categories,
                      exp_id, cls_id, dec, dtype, amp, efs, svm_params):
    """Save ROC and PR curves for worker process"""
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

    base_dir = f"/Volumes/ArcData/V3_database/experiment{exp_id:03d}/classifier_files/svm_results"
    classifier_dir = f"classifier_{cls_id:03d}"
    config_dir = f"D{dec:06d}_TADC{dtype}_A{amp}_EFS{efs:03d}"
    svm_dir = f"{svm_params['kernel']}_C{svm_params['C']}"
    if svm_params.get('gamma'):
        svm_dir += f"_G{svm_params['gamma']}"

    full_dir = os.path.join(base_dir, classifier_dir, config_dir, svm_dir)
    os.makedirs(full_dir, exist_ok=True)

    paths = {}

    unique_labels = np.unique(y_train)
    arc_label_indices = [i for i, label in enumerate(unique_labels)
                         if label_categories.get(int(label), 'unknown') == 'arc']

    for split_name, y_true, y_proba in [
        ('train', y_train, y_proba_train),
        ('test', y_test, y_proba_test),
        ('verify', y_verify, y_proba_verify)
    ]:
        y_true_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_true])

        if len(arc_label_indices) > 0 and y_proba.shape[1] >= len(unique_labels):
            y_proba_arc = np.sum(y_proba[:, arc_label_indices], axis=1)
        else:
            y_proba_arc = np.zeros(len(y_true))

        if len(np.unique(y_true_binary)) <= 1:
            continue

        try:
            fpr, tpr, _ = roc_curve(y_true_binary, y_proba_arc)
            roc_auc = roc_auc_score(y_true_binary, y_proba_arc)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - Arc Detection ({split_name.title()} Set)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            filename = f"roc_curve_binary_{split_name}.png"
            filepath = os.path.join(full_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            paths[f'roc_binary_{split_name}'] = filepath
        except Exception:
            pass

        try:
            precision, recall, _ = precision_recall_curve(y_true_binary, y_proba_arc)
            pr_auc = average_precision_score(y_true_binary, y_proba_arc)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - Arc Detection ({split_name.title()} Set)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            filename = f"pr_curve_binary_{split_name}.png"
            filepath = os.path.join(full_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            paths[f'pr_binary_{split_name}'] = filepath
        except Exception:
            pass

    return paths


def save_svm_model_worker(svm_model, exp_id, cls_id, dec, dtype, amp, efs, svm_params):
    """Save SVM model for worker process"""
    import os
    import joblib

    base_dir = f"/Volumes/ArcData/V3_database/experiment{exp_id:03d}/classifier_files/svm_models"
    classifier_dir = f"classifier_{cls_id:03d}"
    config_dir = f"D{dec:06d}_TADC{dtype}_A{amp}_EFS{efs:03d}"

    full_dir = os.path.join(base_dir, classifier_dir, config_dir)
    os.makedirs(full_dir, exist_ok=True)

    filename = f"svm_{svm_params['kernel']}_C{svm_params['C']}"
    if svm_params.get('gamma'):
        filename += f"_G{svm_params['gamma']}"
    filename += ".pkl"

    filepath = os.path.join(full_dir, filename)
    joblib.dump(svm_model, filepath, compress=3)

    return filepath


class MLDPShell:
    """Advanced MLDP Interactive Shell"""

    def __init__(self, auto_connect=False, auto_experiment=None):
        self.session = PromptSession(
            history=FileHistory(os.path.expanduser('~/.mldp_shell_history')),
            auto_suggest=AutoSuggestFromHistory(),
            completer=MLDPCompleter(),
            style=style,
            message=self.get_prompt,
            vi_mode=False,  # Set to True if you prefer vi mode
        )

        self.db_conn = None
        self.current_experiment = 18
        self.current_distance_type = 'l2'
        self.current_classifier_id = None
        self.current_classifier_name = None
        self.last_result = None
        self.running = True
        self.auto_connect = auto_connect
        self.auto_experiment = auto_experiment
        
        # Command handlers
        self.commands = {
            'connect': self.cmd_connect,
            'query': self.cmd_query,
            'sql': self.cmd_sql,
            'tables': self.cmd_tables,
            'browser': self.cmd_browser,
            # Experiment commands
            'experiment-list': self.cmd_experiment_list,
            'experiment-info': self.cmd_experiment_info,
            'experiment-config': self.cmd_experiment_config,
            'experiment-summary': self.cmd_experiment_summary,
            'experiment-generate': self.cmd_experiment_generate,
            'experiment-create': self.cmd_experiment_create,
            # Distance commands
            'calculate': self.cmd_calculate,
            'insert_distances': self.cmd_insert_distances,
            'heatmap': self.cmd_heatmap,
            'histogram': self.cmd_histogram,
            'visualize': self.cmd_visualize,
            'stats': self.cmd_stats,
            'closest': self.cmd_closest,
            'set': self.cmd_set,
            'show': self.cmd_show,
            'verify': self.cmd_verify,
            'clear': self.cmd_clear,
            'export': self.cmd_export,
            'time': self.cmd_time,
            'select-segments': self.cmd_select_segments,
            'clean-segment-table': self.cmd_clean_segment_table,
            'clean-segment-pairs': self.cmd_clean_segment_pairs,
            'clean-feature-files': self.cmd_clean_feature_files,
            'update-decimations': self.cmd_update_decimations,
            'add-decimation': self.cmd_add_decimation,
            'remove-decimation': self.cmd_remove_decimation,
            'update-segment-sizes': self.cmd_update_segment_sizes,
            'update-amplitude-methods': self.cmd_update_amplitude_methods,
            'create-feature-set': self.cmd_create_feature_set,
            'add-feature-set': self.cmd_add_feature_set,
            'remove-feature-set': self.cmd_remove_feature_set,
            'clear-feature-sets': self.cmd_clear_feature_sets,
            'list-feature-sets': self.cmd_list_feature_sets,
            'remove-data-type': self.cmd_remove_data_type,
            'add-data-type': self.cmd_add_data_type,
            'list-data-types': self.cmd_list_data_types,
            'list-all-data-types': self.cmd_list_all_data_types,
            'list-amplitude-methods': self.cmd_list_amplitude_methods,
            'list-all-amplitude-methods': self.cmd_list_all_amplitude_methods,
            'show-all-feature-sets': self.cmd_show_all_feature_sets,
            # New feature management commands
            'create-feature': self.cmd_create_feature,
            'list-features': self.cmd_list_features,
            'show-feature': self.cmd_show_feature,
            'update-feature': self.cmd_update_feature,
            'delete-feature': self.cmd_delete_feature,
            'create-global-feature-set': self.cmd_create_global_feature_set,
            'add-features-to-set': self.cmd_add_features_to_set,
            'remove-features-from-set': self.cmd_remove_features_from_set,
            'clone-feature-set': self.cmd_clone_feature_set,
            'link-feature-set': self.cmd_link_feature_set,
            'bulk-link-feature-sets': self.cmd_bulk_link_feature_sets,
            'update-feature-link': self.cmd_update_feature_link,
            'show-feature-config': self.cmd_show_feature_config,
            'update-selection-config': self.cmd_update_selection_config,
            'select-files': self.cmd_select_files,
            'remove-files': self.cmd_remove_files,
            'remove-file-labels': self.cmd_remove_file_labels,
            'remove-segments': self.cmd_remove_segments,
            'generate-training-data': self.cmd_generate_training_data,
            'generate-segment-pairs': self.cmd_generate_segment_pairs,
            'generate-feature-fileset': self.cmd_generate_feature_fileset,
            'help': self.cmd_help,
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            # Server management commands
            'servers': self.cmd_servers,
            'start': self.cmd_servers_start,
            'stop': self.cmd_servers_stop,
            'restart': self.cmd_servers_restart,
            'status': self.cmd_servers_status,
            'logs': self.cmd_servers_logs,
            # Segment generation commands
            'segment-generate': self.cmd_segment_generate,
            'generate-segment-fileset': self.cmd_generate_segment_fileset,
            'show-segment-status': self.cmd_show_segment_status,
            'segment-test': self.cmd_segment_test,
            'validate-segments': self.cmd_validate_segments,
            'segment-plot': self.cmd_segment_plot,
            'feature-plot': self.cmd_feature_plot,
            # Data path and cleanup commands
            'get-experiment-data-path': self.cmd_get_experiment_data_path,
            'set-experiment-data-path': self.cmd_set_experiment_data_path,
            'clean-segment-files': self.cmd_clean_segment_files,
            'clean-feature-files': self.cmd_clean_feature_files,
            'clean-distance-work-files': self.cmd_clean_distance_work_files,
            # Distance calculation commands
            'init-distance-tables': self.cmd_init_distance_tables,
            'show-distance-metrics': self.cmd_show_distance_metrics,
            'add-distance-metric': self.cmd_add_distance_metric,
            'remove-distance-metric': self.cmd_remove_distance_metric,
            'clean-distance-tables': self.cmd_clean_distance_tables,
            # Phase 6 cleanup command aliases
            'clean-distance-calculate': self.cmd_clean_distance_work_files,  # Alias
            'clean-features': self.cmd_clean_feature_files,  # Alias
            # Phase 6 new cleanup commands
            'clean-distance-insert': self.cmd_clean_distance_insert,
            'clean-segments': self.cmd_clean_segments,
            'clean-files': self.cmd_clean_files,
            'clean-experiment': self.cmd_clean_experiment,
            # Distance function LUT management
            'show-distance-functions': self.cmd_show_distance_functions,
            'update-distance-function': self.cmd_update_distance_function,
            # MPCCTL distance calculation
            'mpcctl-distance-function': self.cmd_mpcctl_distance_function,
            'mpcctl-distance-insert': self.cmd_mpcctl_distance_insert,
            # Phase 7 execution pipeline
            'mpcctl-execute-experiment': self.cmd_mpcctl_execute_experiment,
            # Classifier management commands (Phase 0a)
            'classifier-create-registry': self.cmd_classifier_create_registry,
            'classifier-migrate-registry': self.cmd_classifier_migrate_registry,
            'classifier-create-junction-tables': self.cmd_classifier_create_junction_tables,
            'classifier-create-global-config-table': self.cmd_classifier_create_global_config_table,
            'classifier-migrate-configs-to-global': self.cmd_classifier_migrate_configs_to_global,
            'classifier-add-config-foreign-keys': self.cmd_classifier_add_config_foreign_keys,
            'classifier-migrate-config-tables': self.cmd_classifier_migrate_config_tables,
            'classifier-new': self.cmd_classifier_new,
            'classifier-remove': self.cmd_classifier_remove,
            'classifier-list': self.cmd_classifier_list,
            # Classifier configuration commands (Phase 0b)
            'classifier-config-create': self.cmd_classifier_config_create,
            'classifier-config-add-feature-sets': self.cmd_classifier_config_add_feature_sets,
            'classifier-config-add-hyperparameters': self.cmd_classifier_config_add_hyperparameters,
            'classifier-config-remove-hyperparameters': self.cmd_classifier_config_remove_hyperparameters,
            'classifier-config-delete': self.cmd_classifier_config_delete,
            'classifier-config-list': self.cmd_classifier_config_list,
            'classifier-config-activate': self.cmd_classifier_config_activate,
            'classifier-config-show': self.cmd_classifier_config_show,
            'classifier-create-feature-builder-table': self.cmd_classifier_create_feature_builder_table,
            'classifier-config-set-feature-builder': self.cmd_classifier_config_set_feature_builder,
            # Classifier data split assignment commands (Phase 1)
            'classifier-create-splits-table': self.cmd_classifier_create_splits_table,
            'classifier-assign-splits': self.cmd_classifier_assign_splits,
            'classifier-copy-splits-from': self.cmd_classifier_copy_splits_from,
            'classifier-show-splits': self.cmd_classifier_show_splits,
            # Classifier reference selection commands (Phase 2)
            'classifier-drop-references-table': self.cmd_classifier_drop_references_table,
            'classifier-select-references': self.cmd_classifier_select_references,
            'classifier-copy-reference-segments': self.cmd_classifier_copy_reference_segments,
            'classifier-plot-references': self.cmd_classifier_plot_references,
            'classifier-plot-reference-features': self.cmd_classifier_plot_reference_features,
            # Classifier feature construction commands (Phase 3)
            'classifier-build-features': self.cmd_classifier_build_features,
            # Classifier SVM training commands (Phase 4)
            'classifier-train-svm-init': self.cmd_classifier_train_svm_init,
            'classifier-train-svm': self.cmd_classifier_train_svm,
            'classifier-test-svm-single': self.cmd_classifier_test_svm_single,
            'classifier-clean-svm-results': self.cmd_classifier_clean_svm_results,
            'classifier-clean-features': self.cmd_classifier_clean_features,
        }
    
    def get_prompt(self):
        """Generate dynamic prompt with current settings"""
        # Build prompt based on current context
        # Priority: classifier > distance > experiment only
        prompt_parts = [('class:prompt', 'mldp')]

        if self.current_experiment:
            prompt_parts.append(('class:separator', '['))
            prompt_parts.append(('class:experiment', f'exp{self.current_experiment}'))

            # Show classifier if selected (takes precedence over distance)
            if self.current_classifier_id:
                prompt_parts.append(('class:separator', ':'))
                prompt_parts.append(('class:classifier', f'cls{self.current_classifier_id}'))
            # Otherwise show distance if selected
            elif self.current_distance_type:
                prompt_parts.append(('class:separator', ':'))
                prompt_parts.append(('class:distance', self.current_distance_type))

            prompt_parts.append(('class:separator', ']'))

        prompt_parts.append(('class:prompt', '> '))
        return FormattedText(prompt_parts)
    
    def print_banner(self):
        """Print welcome banner"""
        clear()
        # Calculate padding for centered version
        version_text = f"MLDP Interactive Shell v{VERSION}"
        version_padding = (78 - len(version_text)) // 2

        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║{' ' * version_padding}{version_text}{' ' * (78 - len(version_text) - version_padding)}║
║         Machine Learning Data Processing Platform - Arc Data Version         ║
║                           Author: Kris Jensen                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  • Tab completion and auto-suggestions available                             ║
║  • Type 'help' for commands or 'help <command>' for details                  ║
║  • Current settings shown in prompt: mldp[exp18:l2]>                         ║
║  • Type 'exit' or Ctrl-D to leave                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        
        if MLDP_ROOT.exists():
            print(f"✅ Connected to MLDP ecosystem at: {MLDP_ROOT}")
        else:
            print(f"⚠️  Warning: MLDP not found at {MLDP_ROOT}")
        print()
    
    def _auto_set_experiment(self, experiment_id):
        """Auto-set experiment on startup with fallback to first experiment"""
        if not self.db_conn:
            print("⚠️  Cannot set experiment: Not connected to database")
            return

        try:
            cursor = self.db_conn.cursor()

            # Check if requested experiment exists
            cursor.execute("""
                SELECT experiment_id
                FROM ml_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))

            if cursor.fetchone():
                # Requested experiment exists
                self.current_experiment = experiment_id
                print(f"✅ Current experiment set to: {self.current_experiment}")
            else:
                # Requested experiment doesn't exist, use first one
                print(f"⚠️  Experiment {experiment_id} not found")
                cursor.execute("""
                    SELECT experiment_id
                    FROM ml_experiments
                    ORDER BY experiment_id
                    LIMIT 1
                """)
                result = cursor.fetchone()
                if result:
                    self.current_experiment = result[0]
                    print(f"✅ Using first experiment: {self.current_experiment}")
                else:
                    print("❌ No experiments found in database")

            cursor.close()
        except Exception as e:
            print(f"❌ Error setting experiment: {e}")

    def run(self):
        """Main shell loop"""
        self.print_banner()

        # Handle auto-connect flag
        if self.auto_connect:
            self.cmd_connect([])

        # Handle auto-experiment flag
        if self.auto_experiment is not None:
            self._auto_set_experiment(self.auto_experiment)

        while self.running:
            try:
                # Get user input
                text = self.session.prompt()

                if not text.strip():
                    continue

                # Check for bash command (starts with !)
                if text.strip().startswith('!'):
                    bash_cmd = text.strip()[1:].strip()
                    if bash_cmd:
                        try:
                            result = subprocess.run(
                                bash_cmd,
                                shell=True,
                                capture_output=True,
                                text=True,
                                timeout=300  # 5 minute timeout
                            )
                            if result.stdout:
                                print(result.stdout, end='')
                            if result.stderr:
                                print(result.stderr, end='', file=sys.stderr)
                            if result.returncode != 0:
                                print(f"⚠️  Command exited with code {result.returncode}")
                        except subprocess.TimeoutExpired:
                            print("❌ Command timed out after 5 minutes")
                        except Exception as e:
                            print(f"❌ Error executing bash command: {e}")
                    continue

                # Parse command
                parts = shlex.split(text)
                if not parts:
                    continue

                cmd = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []

                # Execute command
                if cmd in self.commands:
                    self.commands[cmd](args)
                else:
                    print(f"❌ Unknown command: {cmd}")
                    print("Type 'help' for available commands")
                
            except KeyboardInterrupt:
                print("\nUse 'exit' or Ctrl-D to quit")
                continue
            except EOFError:
                self.cmd_exit([])
                break
            except Exception as e:
                print(f"Error: {e}")
    
    # ========== Command Handlers ==========
    
    def cmd_connect(self, args):
        """Connect to database"""
        host = args[0] if len(args) > 0 else 'localhost'
        port = args[1] if len(args) > 1 else '5432'
        database = args[2] if len(args) > 2 else 'arc_detection'
        user = args[3] if len(args) > 3 else 'kjensen'
        
        try:
            if self.db_conn:
                self.db_conn.close()
            
            self.db_conn = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user
            )
            print(f"✅ Connected to {database}@{host}:{port}")
        except Exception as e:
            print(f"❌ Connection failed: {e}")
    
    def cmd_query(self, args):
        """Execute SQL query"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        if not args:
            print("Usage: query <SQL statement>")
            return
        
        sql = ' '.join(args)
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(sql)
            
            if cursor.description:
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                if rows:
                    # Limit display to 100 rows
                    display_rows = rows[:100]
                    print(tabulate(display_rows, headers=columns, tablefmt='grid'))
                    
                    if len(rows) > 100:
                        print(f"\n... showing first 100 of {len(rows)} rows")
                    
                    self.last_result = rows
                    print(f"\n📊 {len(rows)} rows returned")
                else:
                    print("No results found")
            else:
                self.db_conn.commit()
                print(f"✅ Query executed: {cursor.rowcount} rows affected")
            
            cursor.close()
        except Exception as e:
            print(f"❌ Query error: {e}")
            self.db_conn.rollback()
    
    def cmd_tables(self, args):
        """List database tables"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        pattern = args[0] if args else '%'
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE %s
                ORDER BY table_name
            """, (pattern,))
            
            tables = cursor.fetchall()
            
            if tables:
                print("\n📋 Available tables:")
                for i, (table,) in enumerate(tables, 1):
                    print(f"  {i:3d}. {table}")
                print(f"\nTotal: {len(tables)} tables")
            else:
                print("No tables found")
            
            cursor.close()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_browser(self, args):
        """Launch database browser"""
        browser_path = MLDP_ROOT / "database_browser" / "database_browser.py"
        
        if not browser_path.exists():
            print(f"❌ Database browser not found")
            return
        
        print("🚀 Launching database browser...")
        subprocess.Popen([sys.executable, str(browser_path)])
        print("✅ Database browser launched in background")
    
    # ========== Experiment Commands ==========
    
    def cmd_experiment_list(self, args):
        """List all experiments in the database"""
        try:
            from experiment_query_pg import ExperimentQueryPG
            query = ExperimentQueryPG()
            experiments = query.list_experiments()
            
            if not experiments:
                print("No experiments found")
                return
            
            print(f"\n📋 Available Experiments ({len(experiments)} total):")
            print("-" * 80)
            
            for exp in experiments:
                status_emoji = {
                    'completed': '✅',
                    'in_progress': '🔄',
                    'failed': '❌',
                    'initialized': '🆕'
                }.get(exp.get('status', ''), '❓')
                
                print(f"{status_emoji} Experiment {exp['experiment_id']:3d}: {exp['name'][:50]}")
                if exp.get('description'):
                    print(f"   {exp['description'][:70]}")
            
            query.disconnect()
            
        except Exception as e:
            print(f"❌ Error listing experiments: {e}")
    
    def cmd_experiment_info(self, args):
        """Show detailed information about an experiment"""
        if not args:
            # Use current experiment if no ID provided
            exp_id = self.current_experiment
        else:
            try:
                exp_id = int(args[0])
            except ValueError:
                print(f"❌ Invalid experiment ID: {args[0]}")
                return
        
        try:
            from experiment_query_pg import ExperimentQueryPG
            
            query = ExperimentQueryPG()
            query.print_experiment_summary(exp_id)
            
            # Also check for file training data
            # Create connection if needed
            db_conn = self.db_conn
            if not db_conn:
                try:
                    import psycopg2
                    db_conn = psycopg2.connect(
                        host='localhost',
                        port=5432,
                        database='arc_detection',
                        user='kjensen'
                    )
                    temp_conn = True
                except:
                    db_conn = None
                    temp_conn = False
            else:
                temp_conn = False
            
            if db_conn:
                table_name = f"experiment_{exp_id:03d}_file_training_data"
                cursor = db_conn.cursor()
                try:
                    # Check if training data table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT 1 FROM information_schema.tables 
                            WHERE table_name = %s
                        )
                    """, (table_name,))
                    
                    if cursor.fetchone()[0]:
                        # Check which column name is used for labels
                        # Note: Experiment 18 uses 'assigned_label' (published data, cannot change)
                        # All other experiments should use 'file_label_name' (standard)
                        cursor.execute("""
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_name = %s
                            AND column_name IN ('assigned_label', 'file_label_name')
                            LIMIT 1
                        """, (table_name,))

                        label_column_result = cursor.fetchone()
                        if label_column_result:
                            label_column = label_column_result[0]

                            # Get file label statistics using the correct column
                            cursor.execute(f"""
                                SELECT
                                    {label_column} as file_label_name,
                                    COUNT(*) as count
                                FROM {table_name}
                                WHERE experiment_id = %s
                                GROUP BY {label_column}
                                ORDER BY count DESC, {label_column}
                            """, (exp_id,))

                            labels = cursor.fetchall()

                            if labels:
                                print("\n📁 FILE TRAINING DATA:")
                                print("=" * 60)

                                # Get total counts using the correct column
                                cursor.execute(f"""
                                    SELECT
                                        COUNT(DISTINCT file_id) as total_files,
                                        COUNT(DISTINCT {label_column}) as unique_labels
                                    FROM {table_name}
                                    WHERE experiment_id = %s
                                """, (exp_id,))

                                stats = cursor.fetchone()
                                print(f"Total files: {stats[0]}")
                                print(f"Unique labels: {stats[1]}")

                                # Show label distribution
                                print("\nLabel Distribution:")
                                for label_name, count in labels:
                                    bar_length = int(count / max(l[1] for l in labels) * 30)
                                    bar = '█' * bar_length
                                    print(f"  {label_name:30} {count:4} {bar}")

                                # Check for segment training data too
                                seg_table = f"experiment_{exp_id:03d}_segment_training_data"
                                cursor.execute("""
                                    SELECT EXISTS (
                                        SELECT 1 FROM information_schema.tables
                                        WHERE table_name = %s
                                    )
                                """, (seg_table,))

                                if cursor.fetchone()[0]:
                                    cursor.execute(f"""
                                        SELECT COUNT(*) FROM {seg_table}
                                        WHERE experiment_id = %s
                                    """, (exp_id,))
                                    seg_count = cursor.fetchone()[0]
                                    if seg_count > 0:
                                        print(f"\n📊 SEGMENT TRAINING DATA:")
                                        print("=" * 60)
                                        print(f"Total segments: {seg_count}")

                                        # Get segment label distribution by joining with data_segments
                                        cursor.execute(f"""
                                            SELECT
                                                COALESCE(sl.label_name, 'unlabeled') as label_name,
                                                COUNT(*) as count
                                            FROM {seg_table} st
                                            JOIN data_segments ds ON st.segment_id = ds.segment_id
                                            LEFT JOIN segment_labels sl ON ds.segment_label_id = sl.label_id
                                            GROUP BY sl.label_name
                                            ORDER BY count DESC, label_name
                                        """)

                                        seg_labels = cursor.fetchall()
                                        if seg_labels:
                                            print(f"Unique segment labels: {len(seg_labels)}")
                                            print("\nSegment Label Distribution:")
                                            max_seg_count = max(l[1] for l in seg_labels)
                                            for label_name, count in seg_labels:
                                                bar_length = int(count / max_seg_count * 30)
                                                bar = '█' * bar_length
                                                print(f"  {label_name:35} {count:4} {bar}")

                                        # Get position distribution from segment_selection_log if available
                                        cursor.execute("""
                                            SELECT EXISTS (
                                                SELECT 1 FROM information_schema.tables
                                                WHERE table_name = 'segment_selection_log'
                                            )
                                        """)

                                        if cursor.fetchone()[0]:
                                            cursor.execute("""
                                                SELECT
                                                    position_type,
                                                    COUNT(*) as count
                                                FROM segment_selection_log
                                                WHERE experiment_id = %s
                                                GROUP BY position_type
                                                ORDER BY position_type
                                            """, (exp_id,))

                                            positions = cursor.fetchall()
                                            if positions:
                                                print("\nPosition Distribution:")
                                                for pos, count in positions:
                                                    print(f"  {pos:10}: {count} segments")

                                        # Also get segment type distribution
                                        cursor.execute(f"""
                                            SELECT
                                                ds.segment_type,
                                                COUNT(*) as count
                                            FROM {seg_table} st
                                            JOIN data_segments ds ON st.segment_id = ds.segment_id
                                            GROUP BY ds.segment_type
                                            ORDER BY ds.segment_type
                                        """)

                                        seg_types = cursor.fetchall()
                                        if seg_types:
                                            print("\nSegment Type Distribution:")
                                            for seg_type, count in seg_types:
                                                print(f"  {seg_type:10}: {count} segments")

                                # Check for segment pairs too
                                pairs_table = f"experiment_{exp_id:03d}_segment_pairs"
                                cursor.execute("""
                                    SELECT EXISTS (
                                        SELECT 1 FROM information_schema.tables
                                        WHERE table_name = %s
                                    )
                                """, (pairs_table,))

                                if cursor.fetchone()[0]:
                                    cursor.execute(f"""
                                        SELECT COUNT(*) FROM {pairs_table}
                                        WHERE experiment_id = %s
                                    """, (exp_id,))
                                    pairs_count = cursor.fetchone()[0]
                                    if pairs_count > 0:
                                        print(f"🔗 Segment Pairs: {pairs_count} pairs generated")
                            
                except Exception as e:
                    # Silently continue if there's an error (table might not exist)
                    pass
                finally:
                    cursor.close()
                    # Close temporary connection if we created one
                    if temp_conn and db_conn:
                        db_conn.close()
            
            query.disconnect()
            
        except Exception as e:
            print(f"❌ Error getting experiment info: {e}")
    
    def cmd_experiment_config(self, args):
        """Get experiment configuration from database"""
        if not args:
            # Use current experiment if no ID provided
            exp_id = self.current_experiment
        else:
            try:
                exp_id = int(args[0])
            except ValueError:
                print(f"❌ Invalid experiment ID: {args[0]}")
                return
        
        output_json = '--json' in args
        
        try:
            # Try new configurator first for more detailed info
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(exp_id, db_config)
            config = configurator.get_current_config()
            
            if output_json:
                import json
                print(json.dumps(config, indent=2))
            else:
                print(f"\n📊 Configuration for Experiment {exp_id}:")
                print("-" * 60)
                
                # Show decimations
                if config.get('decimations'):
                    print(f"Decimations: {config['decimations']}")
                
                # Show segment sizes
                if config.get('segment_sizes'):
                    print(f"Segment sizes: {config['segment_sizes']}")
                
                # Show amplitude methods
                if config.get('amplitude_methods'):
                    print(f"Amplitude methods: {config['amplitude_methods']}")
                
                # Show feature sets
                if config.get('feature_sets'):
                    print(f"\nFeature Sets:")
                    for fs in config['feature_sets']:
                        print(f"  • {fs['name']}")
                        print(f"    Features: {fs['features']}")
                        if fs['n_values']:
                            print(f"    N values: {fs['n_values']}")
                
                print("-" * 60)
                for key, value in config.items():
                    if isinstance(value, list) and len(value) > 5:
                        print(f"{key:20}: {value[:5]} ... ({len(value)} items)")
                    elif isinstance(value, dict):
                        print(f"{key:20}: {len(value)} entries")
                    else:
                        print(f"{key:20}: {value}")
            
            query.disconnect()
            
        except ValueError:
            print(f"❌ Invalid experiment ID: {args[0]}")
        except Exception as e:
            print(f"❌ Error getting experiment config: {e}")
    
    def cmd_experiment_summary(self, args):
        """Show experiment summary with junction table data"""
        if not args:
            # Show summary of all experiments
            self.cmd_experiment_list([])
        else:
            # Show detailed summary of specific experiment
            self.cmd_experiment_info(args)
    
    def cmd_experiment_create(self, args):
        """Create a new experiment with full CLI specification"""
        try:
            from experiment_cli_builder import ExperimentCLIBuilder
            from experiment_creator import ExperimentCreator
            
            # Check for help
            if not args or '--help' in args:
                print("Usage: experiment-create --name <name> [options]")
                print("\nRequired:")
                print("  --name NAME                    Experiment name")
                print("\nFile Selection:")
                print("  --file-selection {random,all}  File selection strategy (default: random)")
                print("  --max-files N                  Maximum files to select (default: 50)")
                print("  --random-seed N                Random seed (default: 42)")
                print("  --min-examples N               Min examples per class (default: 25)")
                print("  --exclude-labels LABELS        Labels to exclude (default: trash current_only voltage_only other)")
                print("  --target-labels IDS            Specific label IDs (auto-detect if not specified)")
                print("\nSegment Configuration:")
                print("  --segment-sizes SIZES          Segment sizes (default: 8192)")
                print("  --decimations FACTORS          Decimation factors (default: 0)")
                print("  --data-types TYPES             Data types: raw adc6 adc8 adc10 adc12 adc14")
                print("\nProcessing Methods:")
                print("  --amplitude-methods METHODS    Amplitude methods (use 'all' for all available)")
                print("  --distance-functions FUNCS     Distance functions (use 'all' for all available)")
                print("\nSegment Selection:")
                print("  --min-segments-per-position N  Min segments per position (default: 1)")
                print("  --min-segments-per-file N      Min segments per file (default: 3)")
                print("  --position-balance-mode MODE   Balance mode: at_least_one, equal, proportional")
                print("\nOptions:")
                print("  --dry-run                      Validate without creating")
                print("  --force                        Skip confirmation")
                print("\nExample:")
                print("  experiment-create --name random_50files \\")
                print("    --max-files 50 --segment-sizes 128 1024 8192 \\")
                print("    --decimations 0 7 15 --data-types raw adc6 adc8 adc10 adc12 adc14 \\")
                print("    --amplitude-methods all --distance-functions all")
                return
            
            # Build configuration from CLI arguments
            builder = ExperimentCLIBuilder()
            config = builder.create_from_cli(args)
            
            # Validate
            if not config.validate():
                print("❌ Configuration validation failed")
                builder.close()
                return
            
            # Check dry-run
            if config.dry_run:
                print("\n✅ Configuration validated (dry-run mode)")
                builder.close()
                return
            
            # Confirm creation
            force = '--force' in args
            if not force:
                response = input("\nCreate experiment? (y/n): ")
                if response.lower() != 'y':
                    print("❌ Creation cancelled")
                    builder.close()
                    return
            
            # Create experiment
            creator = ExperimentCreator()
            experiment_id = creator.create_experiment(config)
            
            print(f"\n✅ Successfully created experiment {experiment_id}")
            print(f"📊 Experiment: {config.experiment_name}")
            
            # Show what was created
            info = creator.get_experiment_info(experiment_id)
            print(f"\nConfiguration applied:")
            print(f"  • Data Types: {len(info.get('data_types', []))}")
            print(f"  • Amplitude Methods: {len(info.get('amplitude_methods', []))}")
            print(f"  • Decimations: {len(info.get('decimations', []))}")
            print(f"  • Distance Functions: {len(info.get('distance_functions', []))}")
            
            builder.close()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    def cmd_experiment_generate(self, args):
        """Generate a new experiment with configurable parameters"""
        try:
            from experiment_generation_config import (
                ExperimentGenerationConfig,
                BALANCED_18CLASS_CONFIG,
                SMALL_TEST_CONFIG,
                LARGE_UNBALANCED_CONFIG
            )
            from experiment_query_pg import ExperimentQueryPG
            import json
            
            # Parse arguments
            if not args:
                print("Usage: experiment-generate <config_name|config_file> [--dry-run]")
                print("\nAvailable configs:")
                print("  balanced    - 18 classes × 750 instances each")
                print("  small       - 3 classes × 100 instances (test)")
                print("  large       - 18 classes × 1000 instances (unbalanced)")
                print("  <file.json> - Load from JSON file")
                print("\nFor dynamic configuration, use: experiment-create --help")
                print("\nOptions:")
                print("  --dry-run   - Validate configuration without creating experiment")
                return
            
            config_name = args[0]
            dry_run = '--dry-run' in args
            
            # Load configuration
            if config_name == 'balanced':
                config = BALANCED_18CLASS_CONFIG
            elif config_name == 'small':
                config = SMALL_TEST_CONFIG
            elif config_name == 'large':
                config = LARGE_UNBALANCED_CONFIG
            elif config_name.endswith('.json'):
                try:
                    with open(config_name, 'r') as f:
                        config_data = json.load(f)
                    config = ExperimentGenerationConfig.from_dict(config_data)
                except FileNotFoundError:
                    print(f"❌ Configuration file not found: {config_name}")
                    return
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON in configuration file: {config_name}")
                    return
            else:
                print(f"❌ Unknown configuration: {config_name}")
                return
            
            # Set dry run mode
            config.dry_run = dry_run
            
            # Validate configuration
            if not config.validate():
                print("❌ Configuration validation failed")
                return
            
            # Display configuration summary
            print("\nExperiment Generation Configuration:")
            print("=" * 60)
            print(config.summary())
            print("=" * 60)
            
            if dry_run:
                print("\n🔍 DRY RUN MODE - No changes will be made")
            
            # Confirm generation
            if not dry_run:
                response = input("\nGenerate experiment? (y/n): ")
                if response.lower() != 'y':
                    print("❌ Generation cancelled")
                    return
            
            # Connect to database
            query_tool = ExperimentQueryPG()
            
            # Check if experiment name already exists
            existing = query_tool.execute_query(
                "SELECT experiment_id FROM ml_experiments WHERE experiment_name = %s",
                (config.experiment_name,)
            )
            
            if existing:
                print(f"❌ Experiment '{config.experiment_name}' already exists (ID: {existing[0][0]})")
                return
            
            print(f"\n✅ Configuration validated")
            print(f"📊 Will create experiment: {config.experiment_name}")
            print(f"📁 Target: {len(config.target_labels)} labels × {config.instances_per_label} instances")
            print(f"🎲 Selection: {config.selection_strategy} (seed={config.random_seed})")
            
            if dry_run:
                print("\n✅ Dry run completed successfully")
            else:
                # Create the experiment
                try:
                    from experiment_creator import ExperimentCreator
                    
                    creator = ExperimentCreator()
                    experiment_id = creator.create_experiment(config)
                    
                    print(f"\n✅ Successfully created experiment {experiment_id}")
                    print(f"📊 Experiment: {config.experiment_name}")
                    
                    # Show what was created
                    info = creator.get_experiment_info(experiment_id)
                    print(f"\nConfiguration applied:")
                    print(f"  • Data Types: {len(info.get('data_types', []))}")
                    print(f"  • Amplitude Methods: {len(info.get('amplitude_methods', []))}")
                    print(f"  • Decimations: {len(info.get('decimations', []))}")
                    print(f"  • Distance Functions: {len(info.get('distance_functions', []))}")
                    
                    print(f"\n📁 Next steps:")
                    print(f"  1. Run segment selection: experiment-select {experiment_id}")
                    print(f"  2. Generate segment files: experiment-generate-files {experiment_id}")
                    print(f"  3. Calculate distances: experiment-calculate-distances {experiment_id}")
                    print(f"  4. View progress: experiment-info {experiment_id}")
                    
                except ImportError as e:
                    print(f"❌ Failed to import experiment creator: {e}")
                except Exception as e:
                    print(f"❌ Failed to create experiment: {e}")
            
        except ImportError as e:
            print(f"❌ Failed to import required modules: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_calculate(self, args):
        """Calculate distances"""
        calculator_path = MLDP_ROOT / "mldp_exp18_distance" / "mpcctl_distance_calculator.py"
        
        if not calculator_path.exists():
            print(f"❌ Distance calculator not found")
            return
        
        # Parse arguments
        segment_size = None
        distance_type = self.current_distance_type
        workers = 16
        
        i = 0
        while i < len(args):
            if args[i] == '--segment-size' and i + 1 < len(args):
                segment_size = args[i + 1]
                i += 2
            elif args[i] == '--distance-type' and i + 1 < len(args):
                distance_type = args[i + 1]
                i += 2
            elif args[i] == '--workers' and i + 1 < len(args):
                workers = args[i + 1]
                i += 2
            else:
                i += 1
        
        cmd = [sys.executable, str(calculator_path)]
        cmd.extend(['--input-folder', '/Volumes/ArcData/V3_database/experiment18/segment_files'])
        
        if segment_size:
            cmd.extend(['--segment-size', str(segment_size)])
        
        cmd.append(f'--{distance_type}')
        cmd.extend(['--workers', str(workers)])
        
        print(f"🔄 Running distance calculation ({distance_type})...")
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Distance calculation complete!")
            else:
                print(f"❌ Distance calculation failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_insert_distances(self, args):
        """Insert distances to database"""
        insert_path = MLDP_ROOT / "mldp_distance_db_insert" / "mpcctl_distance_db_insert.py"
        
        if not insert_path.exists():
            print(f"❌ Distance insert tool not found")
            return
        
        # Parse arguments
        input_folder = None
        distance_type = self.current_distance_type
        
        i = 0
        while i < len(args):
            if args[i] == '--input-folder' and i + 1 < len(args):
                input_folder = args[i + 1]
                i += 2
            elif args[i] == '--distance-type' and i + 1 < len(args):
                distance_type = args[i + 1]
                i += 2
            else:
                i += 1
        
        cmd = [sys.executable, str(insert_path)]
        if input_folder:
            cmd.extend(['--input-folder', input_folder])
        cmd.extend(['--distance-type', distance_type])
        
        print("🔄 Inserting distances to database...")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Database insertion complete!")
            else:
                print(f"❌ Insertion failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_heatmap(self, args):
        """Generate heatmap"""
        version = 7
        output_dir = None
        
        i = 0
        while i < len(args):
            if args[i] == '--version' and i + 1 < len(args):
                version = args[i + 1]
                i += 2
            elif args[i] == '--output-dir' and i + 1 < len(args):
                output_dir = args[i + 1]
                i += 2
            else:
                i += 1
        
        heatmap_path = MLDP_ROOT / "experiment_generator" / "src" / "heatmaps" / f"generate_exp18_heatmaps_v{version}.py"
        
        if not heatmap_path.exists():
            heatmap_path = MLDP_ROOT / "experiment_generator" / "src" / "heatmaps" / "generate_exp18_heatmaps.py"
        
        if not heatmap_path.exists():
            print(f"❌ Heatmap generator not found")
            return
        
        cmd = [sys.executable, str(heatmap_path)]
        cmd.extend(['--distance-type', self.current_distance_type])
        
        if output_dir:
            cmd.extend(['--output-dir', output_dir])
        
        print(f"🎨 Generating {self.current_distance_type} heatmap (v{version})...")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Heatmap generated!")
        except subprocess.CalledProcessError:
            print("❌ Heatmap generation failed")
    
    def cmd_histogram(self, args):
        """Generate histogram"""
        version = '1_3'
        bins = 50
        
        i = 0
        while i < len(args):
            if args[i] == '--version' and i + 1 < len(args):
                version = args[i + 1]
                i += 2
            elif args[i] == '--bins' and i + 1 < len(args):
                bins = args[i + 1]
                i += 2
            else:
                i += 1
        
        histogram_path = MLDP_ROOT / "experiment_generator" / "src" / "heatmaps" / f"histogram_plot_generator_v{version}.py"
        
        if not histogram_path.exists():
            histogram_path = MLDP_ROOT / "experiment_generator" / "src" / "heatmaps" / "simple_histogram_generator.py"
        
        if not histogram_path.exists():
            print(f"❌ Histogram generator not found")
            return
        
        cmd = [sys.executable, str(histogram_path)]
        cmd.extend(['--distance-type', self.current_distance_type])
        cmd.extend(['--bins', str(bins)])
        
        print(f"📊 Generating {self.current_distance_type} histogram...")
        
        try:
            subprocess.run(cmd, check=True)
            print("✅ Histogram generated!")
        except subprocess.CalledProcessError:
            print("❌ Histogram generation failed")
    
    def cmd_visualize(self, args):
        """Visualize segment"""
        visualizer_path = MLDP_ROOT / "segment_visualizer" / "segment_visualizer.py"
        
        if not visualizer_path.exists():
            print(f"❌ Segment visualizer not found")
            return
        
        segment_id = None
        file_id = None
        
        i = 0
        while i < len(args):
            if args[i] == '--segment-id' and i + 1 < len(args):
                segment_id = args[i + 1]
                i += 2
            elif args[i] == '--file-id' and i + 1 < len(args):
                file_id = args[i + 1]
                i += 2
            else:
                i += 1
        
        if not segment_id and not file_id:
            print("Usage: visualize --segment-id ID [--file-id ID]")
            return
        
        cmd = [sys.executable, str(visualizer_path)]
        if segment_id:
            cmd.extend(['--segment-id', str(segment_id)])
        if file_id:
            cmd.extend(['--file-id', str(file_id)])
        
        print("🔍 Launching segment visualizer...")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("❌ Visualization failed")
    
    def cmd_stats(self, args):
        """Show distance statistics"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        distance_type = args[0] if args else self.current_distance_type
        table_name = f"experiment_{self.current_experiment:03d}_distance_{distance_type.lower()}"
        
        try:
            cursor = self.db_conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            if not cursor.fetchone()[0]:
                print(f"❌ Table {table_name} does not exist")
                cursor.close()
                return
            
            # Count records
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            # Get statistics
            cursor.execute(f"""
                SELECT 
                    MIN(distance_s) as min_dist,
                    MAX(distance_s) as max_dist,
                    AVG(distance_s) as avg_dist,
                    STDDEV(distance_s) as std_dist,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY distance_s) as q1,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY distance_s) as median,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY distance_s) as q3
                FROM {table_name}
            """)
            
            stats = cursor.fetchone()
            
            print(f"\n📊 Statistics for {table_name}:")
            print(f"{'─' * 50}")
            print(f"  Total records:  {count:,}")
            print(f"  Min distance:   {stats[0]:.6f}")
            print(f"  Q1 (25%):       {stats[4]:.6f}")
            print(f"  Median (50%):   {stats[5]:.6f}")
            print(f"  Q3 (75%):       {stats[6]:.6f}")
            print(f"  Max distance:   {stats[1]:.6f}")
            print(f"  Mean distance:  {stats[2]:.6f}")
            print(f"  Std deviation:  {stats[3]:.6f}")
            
            cursor.close()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_closest(self, args):
        """Find closest pairs"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        n = int(args[0]) if args else 10
        table_name = f"experiment_{self.current_experiment:03d}_distance_{self.current_distance_type.lower()}"
        
        try:
            cursor = self.db_conn.cursor()
            cursor.execute(f"""
                SELECT 
                    segment_id_1,
                    segment_id_2,
                    distance_s,
                    file_id_1,
                    file_id_2
                FROM {table_name}
                ORDER BY distance_s ASC
                LIMIT {n}
            """)
            
            rows = cursor.fetchall()
            headers = ['Segment 1', 'Segment 2', 'Distance', 'File 1', 'File 2']
            
            print(f"\n🔍 Top {n} closest pairs ({self.current_distance_type} distance):")
            print(tabulate(rows, headers=headers, tablefmt='grid'))
            
            cursor.close()
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def cmd_set(self, args):
        """Set configuration"""
        if len(args) != 2:
            print("Usage: set <parameter> <value>")
            print("Parameters: experiment, distance, classifier")
            print("Examples:")
            print("  set experiment 41")
            print("  set distance l2")
            print("  set classifier 1")
            print("  set classifier none")
            return

        param, value = args

        if param == 'experiment':
            self.current_experiment = int(value)
            print(f"[SUCCESS] Current experiment set to: {self.current_experiment}")
        elif param == 'distance':
            self.current_distance_type = value
            print(f"[SUCCESS] Current distance type set to: {self.current_distance_type}")
        elif param == 'classifier':
            # Handle classifier selection
            if value.lower() == 'none':
                self.current_classifier_id = None
                self.current_classifier_name = None
                print("[SUCCESS] Classifier deselected")
            else:
                # Validate classifier ID
                try:
                    classifier_id = int(value)
                except ValueError:
                    print(f"[ERROR] Invalid classifier ID: {value}")
                    print("Usage: set classifier <id> or set classifier none")
                    return

                # Check if experiment is selected
                if not self.current_experiment:
                    print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
                    return

                # Check if classifier exists
                if not self.db_conn:
                    print("[ERROR] Not connected to database. Use 'connect' first.")
                    return

                try:
                    cursor = self.db_conn.cursor()
                    cursor.execute("""
                        SELECT classifier_name, is_archived
                        FROM ml_experiment_classifiers
                        WHERE experiment_id = %s AND classifier_id = %s
                    """, (self.current_experiment, classifier_id))
                    result = cursor.fetchone()

                    if not result:
                        print(f"[ERROR] Classifier {classifier_id} not found for experiment {self.current_experiment}")
                        print("Use 'classifier-list' to see available classifiers")
                        return

                    classifier_name, is_archived = result

                    if is_archived:
                        print(f"[WARNING] Classifier {classifier_id} is archived")

                    self.current_classifier_id = classifier_id
                    self.current_classifier_name = classifier_name
                    print(f"[SUCCESS] Current classifier set to: {classifier_id} ({classifier_name})")

                except Exception as e:
                    print(f"[ERROR] Failed to set classifier: {e}")
        else:
            print(f"[ERROR] Unknown parameter: {param}")
            print("Parameters: experiment, distance, classifier")
    
    def cmd_show(self, args):
        """Show current settings"""
        print("\n⚙️  Current Settings:")
        print(f"{'─' * 40}")
        print(f"  Experiment ID:  {self.current_experiment}")
        print(f"  Distance Type:  {self.current_distance_type}")
        print(f"  MLDP Root:      {MLDP_ROOT}")
        print(f"  Database:       {'✅ Connected' if self.db_conn else '❌ Not connected'}")
        
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT current_database(), current_user")
                db, user = cursor.fetchone()
                print(f"  DB Name:        {db}")
                print(f"  DB User:        {user}")
                cursor.close()
            except:
                pass
    
    def cmd_verify(self, args):
        """Verify MLDP tools"""
        print("\n🔍 Verifying MLDP tools...")
        print(f"{'─' * 50}")
        
        tools = [
            ("mldp_exp18_distance/mpcctl_distance_calculator.py", "Distance Calculator"),
            ("mldp_distance_db_insert/mpcctl_distance_db_insert.py", "Distance DB Insert"),
            ("segment_visualizer/segment_visualizer.py", "Segment Visualizer"),
            ("database_browser/database_browser.py", "Database Browser"),
            ("experiment_generator", "Experiment Generator"),
            ("segment_verifier", "Segment Verifier"),
            ("data_cleaning_tool", "Data Cleaning Tool"),
        ]
        
        found = 0
        missing = 0
        
        for tool_path, tool_name in tools:
            full_path = MLDP_ROOT / tool_path
            if full_path.exists():
                print(f"  ✅ {tool_name:25s} Found")
                found += 1
            else:
                print(f"  ❌ {tool_name:25s} Not found")
                missing += 1
        
        print(f"{'─' * 50}")
        print(f"Summary: {found} found, {missing} missing")
        
        if missing == 0:
            print("✅ All tools verified successfully!")
        else:
            print("⚠️  Some tools are missing")
    
    def cmd_clear(self, args):
        """Clear screen"""
        clear()
        self.print_banner()
    
    def cmd_export(self, args):
        """Export last query result"""
        if not self.last_result:
            print("❌ No results to export. Run a query first.")
            return
        
        if not args:
            print("Usage: export <filename>")
            return
        
        filename = args[0]
        
        try:
            if filename.endswith('.json'):
                with open(filename, 'w') as f:
                    json.dump(self.last_result, f, indent=2, default=str)
            else:
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(self.last_result)
            
            print(f"✅ Exported {len(self.last_result)} rows to {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def cmd_time(self, args):
        """Show current time"""
        now = datetime.now()
        print(f"🕐 Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Unix timestamp: {int(now.timestamp())}")
    
    def cmd_help(self, args):
        """Show help"""
        if args:
            cmd = args[0]
            if cmd in self.commands:
                print(f"\nHelp for '{cmd}':")
                print(f"  {self.commands[cmd].__doc__}")
            else:
                print(f"❌ Unknown command: {cmd}")
        else:
            print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              MLDP Commands                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 DATABASE COMMANDS:
  connect [host] [port] [db] [user]  Connect to PostgreSQL database
  query <SQL>                         Execute SQL query
  tables [pattern]                    List database tables
  browser                             Launch database browser GUI

🧪 EXPERIMENT COMMANDS:
  experiment-list                     List all experiments in database
  experiment-info <id>                Show detailed experiment information
  experiment-config <id> [--json]     Get experiment configuration
  experiment-summary [id]             Show experiment summary
  experiment-generate <config>        Generate new experiment (balanced|small|large)
  experiment-create --name <name>     Create experiment with full CLI specification

🔧 EXPERIMENT CONFIGURATION:
  update-decimations <d1> <d2>...     Update decimation factors
  update-segment-sizes <s1> <s2>...   Update segment sizes
  update-amplitude-methods <m1>...    Update amplitude/ADC methods
  update-selection-config [options]   Update segment selection parameters
  create-feature-set --name <n>       Create custom feature set
  add-feature-set <ids> [options]     Add feature sets (--n N --channel source_current|load_voltage)
  list-feature-sets                   List feature sets for current experiment
  show-all-feature-sets                Show ALL feature sets in database
  remove-feature-set <id>              Remove a feature set from experiment
  clear-feature-sets                   Remove ALL feature sets from experiment
  list-data-types                      List data types for current experiment
  add-data-type <id>                   Add a data type to current experiment
  remove-data-type <id>                Remove a data type from current experiment
  select-files [--max-files N]        Select files for training data
  remove-file-labels <label1>...      Remove files with specified labels from training data
  remove-files <id1> <id2>...         Remove specific files by ID from training data
  remove-segments <id1> <id2>...      Remove specific segments by ID from training data

📐 DISTANCE OPERATIONS:
  calculate [options]                 Calculate distances using mpcctl
  insert_distances [options]          Insert distances into database
  stats [distance_type]               Show distance statistics
  closest [N]                         Find N closest segment pairs

🔄 MPCCTL DISTANCE PIPELINE:
  mpcctl-distance-function            Calculate distances (--start/--status/--pause/--continue/--stop)
  mpcctl-distance-insert              Insert distances to DB (--start/--status/--pause/--continue/--stop)
  mpcctl-execute-experiment           Execute complete pipeline (--workers N --log --verbose)

🎨 VISUALIZATION:
  heatmap [--version N]               Generate distance heatmap
  histogram [--version] [--bins]      Generate distance histogram
  visualize --segment-id ID           Visualize segment data

🔬 ML PIPELINE COMMANDS:
  select-files                        Select files for training (DB table)
  select-segments                     Select segments for training (DB table)
  generate-segment-pairs              Generate segment pairs (DB table)
  generate-segment-fileset            Generate physical segment files from raw data
  generate-feature-fileset            Extract features and save to disk

🤖 CLASSIFIER MANAGEMENT:
  classifier-create-registry          Create ml_experiment_classifiers table (one-time setup)
  classifier-migrate-registry         Migrate to global_classifier_id schema (Phase 0b)
    [--force]                         Skip confirmation prompt
  classifier-create-junction-tables   Create 6 junction tables for normalized config storage
    [--force]                         Recreate tables if they exist
  classifier-migrate-config-tables    Drop/recreate config tables with new schema
    [--all]                           Migrate all classifiers in experiment
    [--force]                         Skip confirmation prompt
  classifier-new --name <name>        Create new classifier instance
    [--description <desc>]            Optional description
    [--type <type>]                   Classifier type (default: svm)
    [--auto-select]                   Auto-select after creation (default: True)
  classifier-remove --classifier-id <id> --confirm  Delete classifier
    [--archive-instead]               Archive instead of deleting
  classifier-list                     List all classifiers for current experiment
    [--include-archived]              Show archived classifiers
    [--show-tables]                   Show table counts for each classifier
  set classifier <N>                  Select classifier N for current experiment
  set classifier none                 Deselect current classifier

  Examples:
    classifier-new --name "baseline_svm" --description "Baseline configuration"
    set classifier 1
    classifier-list --include-archived
    classifier-remove --classifier-id 2 --confirm

⚙️  CLASSIFIER CONFIGURATION (Phase 0b):
  classifier-config-create            Create a new configuration for current classifier
    --config-name <name>              Configuration name (required)
    --decimation-factors <list>       Comma-separated decimation factors or "all"
    --data-types <list>               Comma-separated data type names (e.g., adc6,adc8,adc10,adc12)
    --amplitude-methods <list>        Comma-separated amplitude method IDs or "all"
    --distance-functions <list>       Comma-separated distance function names or "all"
    --feature-sets <list>             Comma-separated feature set IDs or "all" (default)
    --set-active                      Set as active configuration
    --notes <text>                    Optional notes

  classifier-config-list              List all configurations for current classifier
    [--all]                           Show configs for all classifiers
    [--experiment-id <id>]            Specify experiment ID
    [--classifier-id <id>]            Specify classifier ID

  classifier-config-activate          Activate a configuration
    --config-name <name>              Configuration name
      or --config-id <id>             Configuration ID

  classifier-config-show              Show detailed configuration information
                                      (includes feature builder settings)
    [--config-name <name>]            Show specific config by name
    [--config-id <id>]                Show specific config by ID
    [--active]                        Show active config (default)

  classifier-config-delete            Delete a configuration
    --config-name <name>              Configuration name
      or --config-id <id>             Configuration ID
    [--confirm]                       Skip confirmation prompt

  classifier-config-add-feature-sets  Add feature sets to existing config
    --config-id <id>                  Configuration ID
    --feature-sets <list>             Comma-separated feature_set_ids from ml_feature_sets_lut

  classifier-config-add-hyperparameters  Add hyperparameters to existing config
    --config <name|id>                Config name or ID (required)
    --amplitude-methods <list>        Comma-separated amplitude method IDs
    --decimation-factors <list>       Comma-separated decimation factors
    --data-types <list>               Comma-separated data type names or IDs
    --distance-functions <list>       Comma-separated distance function names
    --experiment-feature-sets <list>  Comma-separated experiment feature set IDs

  classifier-config-remove-hyperparameters  Remove hyperparameters from existing config
    --config <name|id>                Config name or ID (required)
    --amplitude-methods <list>        Comma-separated amplitude method IDs
    --decimation-factors <list>       Comma-separated decimation factors
    --data-types <list>               Comma-separated data type names or IDs
    --distance-functions <list>       Comma-separated distance function names
    --experiment-feature-sets <list>  Comma-separated experiment feature set IDs
    --force                           Required to confirm deletion (REQUIRED)

  classifier-create-global-config-table       Create global ml_classifier_configs table (one-time)
  classifier-create-feature-builder-table     Create ml_classifier_feature_builder table (one-time)

  classifier-config-set-feature-builder       Set feature builder flags for a configuration
    --config-id <id>                          Configuration ID (required)
    [--include-original]                      Include raw feature values in X matrix
    [--no-include-original]                   Exclude raw feature values
    [--compute-distances-inter]               Compute distances to OTHER class baselines
    [--no-compute-distances-inter]            Don't compute inter-class distances
    [--compute-distances-intra]               Compute distances to SAME class baseline
    [--no-compute-distances-intra]            Don't compute intra-class distances
    [--statistical-features]                  Enable statistical features (reserved)
    [--external-function]                     Enable external function (reserved)
    [--notes <text>]                          Optional notes

  Note: Feature builder settings are displayed in classifier-config-show

  Examples:
    classifier-config-create --config-name "baseline" --decimation-factors all \\
                             --data-types adc6,adc8,adc10,adc12 --feature-sets all \\
                             --distance-functions all --set-active
    classifier-config-list
    classifier-config-show
    classifier-config-activate --config-name "baseline"
    classifier-config-add-feature-sets --config-id 1 --feature-sets 1,2,5
    classifier-config-add-hyperparameters --config baseline --amplitude-methods 2
    classifier-config-add-hyperparameters --config baseline --decimation-factors 31,63
    classifier-config-remove-hyperparameters --config baseline --amplitude-methods 1 --force
    classifier-config-remove-hyperparameters --config 1 --decimation-factors 255 --force
    classifier-config-set-feature-builder --config-id 1 --include-original --compute-distances-inter
    classifier-config-show    # Shows hyperparameters AND feature builder settings
    classifier-config-delete --config-name "test_config" --confirm

⚙️  DATA SPLIT ASSIGNMENT (Phase 1):
  classifier-create-splits-table  Create data_splits table (one-time setup)
    [--force]                     Recreate table if exists

  classifier-assign-splits        Assign train/test/verification splits
    [--train-ratio <float>]       Training set ratio (default: 0.70)
    [--test-ratio <float>]        Test set ratio (default: 0.20)
    [--verification-ratio <float>] Verification set ratio (default: 0.10)
    [--seed <int>]                Random seed (default: 42)
    [--force]                     Overwrite existing splits

  classifier-copy-splits-from     Copy splits from another classifier
    --source-classifier <id>      Source classifier ID (required)
    [--force]                     Overwrite existing splits

  classifier-show-splits          Display split statistics
    [--decimation-factor <n>]     Filter by decimation factor
    [--data-type <id>]            Filter by data type
    [--detail]                    Show per-class breakdown

  Examples:
    classifier-create-splits-table
    classifier-assign-splits --train-ratio 0.80 --test-ratio 0.15 --verification-ratio 0.05
    classifier-copy-splits-from --source-classifier 1
    classifier-show-splits --detail

📊 REFERENCE SELECTION (Phase 2):
  classifier-select-references   Select reference segments using PCA + centroid
    [--force]                    Overwrite existing references
    [--plot]                     Generate PCA visualization plots
    [--plot-dir <path>]          Directory for plots (default: ~/plots)
    [--min-segments <n>]         Minimum segments per class (default: 5)
    [--pca-components <n>]       PCA components (default: 2)

  classifier-copy-reference-segments  Copy reference segments from another classifier
    --source-classifier <id>     Source classifier ID (required)
    [--force]                    Overwrite existing references

  Examples:
    classifier-select-references --plot
    classifier-copy-reference-segments --source-classifier 1
    classifier-select-references --plot --plot-dir ~/plots/exp041_cls001_refs

🔧 FEATURE CONSTRUCTION (Phase 3):
  classifier-build-features      Build distance-based feature vectors for SVM
    [--force]                    Overwrite existing features
    [--batch-size <n>]           Batch size for processing (default: 1000)
    [--decimation-factor <n>]    Build only for specific decimation factor
    [--data-type <id>]           Build only for specific data type
    [--amplitude-method <id>]    Build only for specific amplitude method
    [--feature-set <id>]         Build only for specific feature set
    [--workers <n>]              Parallel workers (default: 7, max: 28)

  Examples:
    classifier-build-features
    classifier-build-features --amplitude-method 2 --workers 21
    classifier-build-features --decimation-factor 255 --data-type 6

🤖 SVM TRAINING (Phase 4):
  classifier-train-svm-init      Create SVM results tables (one-time setup)

  classifier-train-svm           Train SVM models on distance-based features
    [--workers <n>]              Parallel workers (default: 7, max: 28)
    [--decimation-factor <n>]    Train only for specific decimation factor
    [--data-type <id>]           Train only for specific data type
    [--amplitude-method <id>]    Train only for specific amplitude method
    [--feature-set <id>]         Train only for specific feature set
    [--kernel <type>]            SVM kernel: linear, rbf, poly (default: all)
    [--C <value>]                SVM C parameter (default: grid search)
    [--gamma <value>]            SVM gamma parameter (default: grid search)
    [--force]                    Overwrite existing results

  classifier-clean-svm-results   Clear SVM training results from database
    [--amplitude-method <id>]    Delete only this amplitude method
    [--decimation-factor <n>]    Delete only this decimation factor
    [--data-type <id>]           Delete only this data type
    [--feature-set <id>]         Delete only this feature set
    [--force]                    Required to confirm deletion (REQUIRED)

  classifier-clean-features      Clear feature vectors from database
    [--amplitude-method <id>]    Delete only this amplitude method
    [--decimation-factor <n>]    Delete only this decimation factor
    [--data-type <id>]           Delete only this data type
    [--feature-set <id>]         Delete only this feature set
    [--force]                    Required to confirm deletion (REQUIRED)

  Examples:
    classifier-train-svm-init
    classifier-train-svm --workers 21
    classifier-train-svm --amplitude-method 2 --kernel rbf
    classifier-clean-svm-results --amplitude-method 1 --force
    classifier-clean-features --amplitude-method 2 --force

📂 DATA MANAGEMENT:
  get-experiment-data-path            Show paths and file counts for experiment data
  set-experiment-data-path <path>     Set custom data storage paths (or --reset for default)
  clean-segment-files                 Delete segment files (supports --dry-run, --force)
  clean-feature-files                 Delete feature files and truncate DB table

🔍 SEGMENT COMMANDS:
  segment-generate                    Generate segments from raw data
  show-segment-status                 Show segment generation status
  segment-test                        Test segment generation with small dataset
  validate-segments                   Validate generated segments
  segment-plot                        Plot segment data

⚙️  SETTINGS:
  set <param> <value>                 Set configuration (experiment, distance, classifier)
  show                                Show current settings

🖥️  SERVER MANAGEMENT:
  start                               Start all MLDP servers
  stop                                Stop all MLDP servers
  restart                             Restart all MLDP servers
  status                              Check status of all servers
  logs [service] [lines]              View server logs
  servers <command>                   Server management (start/stop/status/etc)

🛠️  UTILITIES:
  sql <query>                         Execute SQL query (SELECT/DROP/UPDATE/INSERT)
  verify                              Verify MLDP tools
  clear                               Clear screen
  export <filename>                   Export query results (.csv or .json)
  time                                Show current time
  help [command]                      Show help
  exit/quit                           Exit shell

⚡ BASH COMMANDS:
  !<command>                          Execute bash command (e.g., !ls, !python3 script.py)
  Examples:
    !ls -la                           List files in current directory
    !rm -rf /path/to/files/*          Remove files
    !python3 -c "import numpy"        Run Python code
    !find . -name "*.npy"             Find files

💡 TIPS:
  • Use Tab for command completion
  • Use ↑/↓ arrows for command history
  • Current settings shown in prompt: mldp[exp41:cls1]> (or mldp[exp41:l2]>)
  • Prompt shows classifier if selected, otherwise distance type
  • SQL queries support all PostgreSQL syntax
  • Export supports .csv and .json formats
  • Bash commands timeout after 5 minutes
""")
    
    def cmd_select_segments(self, args):
        """Select segments for training with proper segment code balancing"""
        # Parse experiment ID if provided, otherwise use current
        if args and args[0].isdigit():
            experiment_id = int(args[0])
            args = args[1:]  # Remove experiment ID from args
        else:
            experiment_id = self.current_experiment

        if not experiment_id:
            print("❌ No experiment specified. Use: select-segments <experiment_id> [options]")
            print("   Or set current experiment: set experiment <id>")
            return

        # Parse options
        strategy = 'balanced'  # Default
        segments_per_type = 3  # Default for fixed_per_type strategy
        seed = 42
        clean_first = False  # Default
        force = False  # Default

        i = 0
        while i < len(args):
            if args[i] == '--strategy' and i + 1 < len(args):
                strategy = args[i + 1]
                i += 2
            elif args[i] == '--segments-per-type' and i + 1 < len(args):
                segments_per_type = int(args[i + 1])
                i += 2
            elif args[i] == '--seed' and i + 1 < len(args):
                seed = int(args[i + 1])
                i += 2
            elif args[i] == '--clean':
                clean_first = True
                i += 1
            elif args[i] == '--force':
                force = True
                i += 1
            elif args[i] == '--help':
                print("\nUsage: select-segments [experiment_id] [options]")
                print("\nOptions:")
                print("  --strategy STRAT           Selection strategy (default: balanced)")
                print("                             balanced: Find min count across segment types,")
                print("                                      select that number from EACH type")
                print("                             fixed_per_type: Select N segments from each type")
                print("                             proportional: Select proportionally from each type")
                print("  --segments-per-type N      For fixed_per_type: segments to select per type (default: 3)")
                print("  --seed N                   Random seed (default: 42)")
                print("  --clean                    Clear existing segment training data before selection")
                print("  --force                    Skip confirmation prompts (for unattended execution)")
                print("\n📊 BALANCED STRATEGY (recommended):")
                print("  Per file: Groups segments by code type (L, R, C, Cm, Cl, Cr, etc.)")
                print("  Example: File has L=45, R=40, C=5, Cm=25, Cl=3, Cr=2 segments")
                print("  → Finds minimum: min(45,40,5,25,3,2) = 2")
                print("  → Selects 2 from EACH type: 2L + 2R + 2C + 2Cm + 2Cl + 2Cr = 12 total")
                print("\nExamples:")
                print("  select-segments 41 --strategy balanced")
                print("  select-segments 41 --strategy balanced --clean")
                print("  select-segments 41 --strategy fixed_per_type --segments-per-type 5")
                print("  select-segments --strategy balanced  (uses current experiment)")
                return
            else:
                i += 1

        # Clean existing data if requested
        if clean_first:
            print(f"\n🗑️  Cleaning existing segment training data...")
            cleanup_args = [str(experiment_id)]
            if force:
                cleanup_args.append('--force')
            self.cmd_clean_segment_table(cleanup_args)
            print()

        print(f"🔄 Selecting segments for experiment {experiment_id}...")
        print(f"   Strategy: {strategy}")
        if strategy == 'fixed_per_type':
            print(f"   Segments per type: {segments_per_type}")
        elif strategy == 'balanced':
            print(f"   Will select minimum count across all segment types from EACH type")
        print(f"   Random seed: {seed}")

        try:
            # Try to use the improved v2 selector first
            try:
                from experiment_segment_selector_v2 import SegmentSelectorV2
                use_v2 = True
            except ImportError:
                # Fallback to original if v2 not available
                from experiment_segment_selector import ExperimentSegmentSelector
                use_v2 = False
                print("⚠️  Using legacy selector. For better results, ensure experiment_segment_selector_v2.py is available.")

            # Database configuration
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            # Run selection with appropriate selector
            if use_v2:
                selector = SegmentSelectorV2(experiment_id, db_config)
                result = selector.run_selection(
                    strategy=strategy,
                    segments_per_type=segments_per_type
                )
            else:
                # Fallback to old selector
                selector = ExperimentSegmentSelector(experiment_id, db_config)
                result = selector.run_selection()

            # Display results
            if result and 'total_segments' in result:
                print(f"\n✅ Successfully selected {result['total_segments']} segments")
                print(f"   From {result.get('total_files', 0)} files")

                # Show average per file
                if result.get('total_files', 0) > 0:
                    avg_per_file = result['total_segments'] / result['total_files']
                    print(f"   Average per file: {avg_per_file:.1f}")

                # Show segment type distribution from v2 selector
                if 'segments_by_type' in result:
                    print("\n📊 Segment type distribution:")
                    for code_type, count in sorted(result['segments_by_type'].items()):
                        print(f"     {code_type}: {count} segments")

                # Show strategy used
                if 'strategy' in result:
                    print(f"\n📋 Selection strategy: {result['strategy']}")

                print(f"\n💾 Data saved to:")
                print(f"   experiment_{experiment_id:03d}_segment_training_data")
            else:
                print(f"❌ Failed to select segments")
                if isinstance(result, dict) and 'error' in result:
                    print(f"   Error: {result['error']}")

        except ImportError as e:
            print(f"❌ Could not import segment selector: {e}")
            print("Make sure experiment_segment_selector_v2.py is in the same directory")
        except Exception as e:
            print(f"❌ Error during segment selection: {e}")

    def cmd_clean_segment_table(self, args):
        """Clean (delete all rows from) the segment training data table for an experiment"""
        # Parse experiment ID and flags
        experiment_id = None
        force = '--force' in args

        # Extract experiment ID if provided
        for arg in args:
            if arg.isdigit():
                experiment_id = int(arg)
                break

        # Use current experiment if not specified
        if not experiment_id:
            experiment_id = self.current_experiment

        if not experiment_id:
            print("❌ No experiment specified. Use: clean-segment-table <experiment_id> [--force]")
            print("   Or set current experiment: set experiment <id>")
            return

        table_name = f"experiment_{experiment_id:03d}_segment_training_data"

        # Connect to database
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        try:
            cursor = self.db_conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (table_name,))

            table_exists = cursor.fetchone()[0]

            if not table_exists:
                print(f"ℹ️  Table {table_name} does not exist (nothing to clean)")
                cursor.close()
                return

            # Get count before deletion
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count_before = cursor.fetchone()[0]

            if count_before == 0:
                print(f"ℹ️  Table {table_name} is already empty")
                cursor.close()
                return

            # Show what will be deleted
            print(f"\n📊 Segment training data table: {table_name}")
            print(f"   Current rows: {count_before:,}")

            # Confirmation (skip if --force)
            if not force:
                print(f"\n⚠️  WARNING: This will delete all {count_before:,} rows from {table_name}")
                print(f"⚠️  This action CANNOT be undone!")
                response = input(f"\nType 'DELETE' to confirm: ").strip()

                if response != 'DELETE':
                    print("❌ Cancelled")
                    cursor.close()
                    return
            else:
                print("\n⚠️  --force flag set: Skipping confirmation")

            # Delete all rows
            print(f"\n🗑️  Deleting all rows from {table_name}...")
            cursor.execute(f"DELETE FROM {table_name}")
            self.db_conn.commit()

            # Verify deletion
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count_after = cursor.fetchone()[0]

            if count_after == 0:
                print(f"✅ Deleted {count_before:,} rows")
                print(f"✅ Table {table_name} is now empty")
            else:
                print(f"⚠️  Warning: {count_after} rows remaining")

            cursor.close()

        except Exception as e:
            print(f"❌ Error cleaning segment table: {e}")
            if self.db_conn:
                self.db_conn.rollback()

    def cmd_update_decimations(self, args):
        """Update decimation factors for current experiment"""
        if not args:
            print("Usage: update-decimations <decimation1> <decimation2> ...")
            print("Example: update-decimations 0 7 15")
            return

        try:
            decimations = [int(arg) for arg in args]
        except ValueError:
            print(f"❌ Invalid decimation values. Must be integers.")
            return

        try:
            from experiment_configurator import ExperimentConfigurator

            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            configurator = ExperimentConfigurator(self.current_experiment, db_config)

            print(f"🔄 Updating decimations for experiment {self.current_experiment}...")
            if configurator.update_decimations(decimations):
                print(f"✅ Decimations updated: {decimations}")
            else:
                print(f"❌ Failed to update decimations")

        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error updating decimations: {e}")

    def cmd_add_decimation(self, args):
        """Add a single decimation factor to current experiment"""
        if not args or len(args) != 1:
            print("Usage: add-decimation <decimation_factor>")
            print("Example: add-decimation 31")
            return

        try:
            decimation = int(args[0])
        except ValueError:
            print(f"❌ Invalid decimation value. Must be an integer.")
            return

        try:
            from experiment_configurator import ExperimentConfigurator

            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            configurator = ExperimentConfigurator(self.current_experiment, db_config)

            print(f"➕ Adding decimation {decimation} to experiment {self.current_experiment}...")
            if configurator.add_decimation(decimation):
                print(f"✅ Decimation {decimation} added successfully")
            else:
                print(f"⚠️  Decimation {decimation} already exists or failed to add")

        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error adding decimation: {e}")

    def cmd_remove_decimation(self, args):
        """Remove a single decimation factor from current experiment"""
        if not args or len(args) != 1:
            print("Usage: remove-decimation <decimation_factor>")
            print("Example: remove-decimation 15")
            return

        try:
            decimation = int(args[0])
        except ValueError:
            print(f"❌ Invalid decimation value. Must be an integer.")
            return

        try:
            from experiment_configurator import ExperimentConfigurator

            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            configurator = ExperimentConfigurator(self.current_experiment, db_config)

            print(f"➖ Removing decimation {decimation} from experiment {self.current_experiment}...")
            if configurator.remove_decimation(decimation):
                print(f"✅ Decimation {decimation} removed successfully")
            else:
                print(f"⚠️  Decimation {decimation} not found in experiment or failed to remove")

        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error removing decimation: {e}")

    def cmd_update_segment_sizes(self, args):
        """Update segment sizes for current experiment"""
        if not args:
            print("Usage: update-segment-sizes <size1> <size2> ...")
            print("Example: update-segment-sizes 128 1024 8192")
            return
        
        try:
            sizes = [int(arg) for arg in args]
        except ValueError:
            print(f"❌ Invalid segment sizes. Must be integers.")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Updating segment sizes for experiment {self.current_experiment}...")
            if configurator.update_segment_sizes(sizes):
                print(f"✅ Segment sizes updated: {sizes}")
            else:
                print(f"❌ Failed to update segment sizes")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error updating segment sizes: {e}")
    
    def cmd_update_amplitude_methods(self, args):
        """Update amplitude methods for current experiment"""
        if not args:
            print("Usage: update-amplitude-methods <method1> <method2> ...")
            print("Example: update-amplitude-methods minmax zscore")
            print("Available: minmax, zscore, maxabs, robust, TRAW, TADC14, TADC12, TADC10, TADC8, TADC6")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Updating amplitude methods for experiment {self.current_experiment}...")
            if configurator.update_amplitude_methods(args):
                print(f"✅ Amplitude methods updated: {args}")
            else:
                print(f"❌ Failed to update amplitude methods")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error updating amplitude methods: {e}")
    
    def cmd_create_feature_set(self, args):
        """Create a custom feature set for current experiment"""
        if not args or '--name' not in args or '--features' not in args:
            print("Usage: create-feature-set --name <name> --features <feature1,feature2,...> [--n-value <n>]")
            print("Example: create-feature-set --name voltage_variance --features voltage,variance(voltage) --n-value 128")
            return
        
        try:
            # Parse arguments
            name = None
            features = None
            n_value = 128  # Default
            
            i = 0
            while i < len(args):
                if args[i] == '--name' and i + 1 < len(args):
                    name = args[i + 1]
                    i += 2
                elif args[i] == '--features' and i + 1 < len(args):
                    features = args[i + 1].split(',')
                    i += 2
                elif args[i] == '--n-value' and i + 1 < len(args):
                    n_value = int(args[i + 1])
                    i += 2
                else:
                    i += 1
            
            if not name or not features:
                print("❌ Both --name and --features are required")
                return
            
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Creating feature set '{name}' for experiment {self.current_experiment}...")
            feature_set_id = configurator.create_feature_set(name, features, n_value)
            
            if feature_set_id:
                print(f"✅ Feature set created (ID: {feature_set_id})")
                print(f"   Name: {name}")
                print(f"   Features: {', '.join(features)}")
                print(f"   N value: {n_value}")
            else:
                print(f"❌ Failed to create feature set")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error creating feature set: {e}")
    
    def cmd_add_feature_set(self, args):
        """Add existing feature set(s) to current experiment"""
        if not args or '--help' in args:
            print("Usage: add-feature-set <feature_set_id> [options]")
            print("   or: add-feature-set <id1,id2,id3,...> [options]")
            print("\nOptions:")
            print("  --n <value>          N value for chunk size")
            print("  --channel <channel>  Data channel: source_current or load_voltage (default: load_voltage)")
            print("\nExamples:")
            print("  add-feature-set 3                              # Add with defaults")
            print("  add-feature-set 3 --n 1024                     # With N=1024")
            print("  add-feature-set 3 --channel source_current     # From source current")
            print("  add-feature-set 1,2,3,4 --channel load_voltage --n 8192")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            config = ExperimentConfigurator(self.current_experiment, db_config)
            
            # Parse arguments
            ids_arg = args[0]
            n_value = None
            data_channel = 'load_voltage'
            
            # Parse optional arguments
            i = 1
            while i < len(args):
                if args[i] == '--n' and i + 1 < len(args):
                    n_value = int(args[i + 1])
                    i += 2
                elif args[i] == '--channel' and i + 1 < len(args):
                    data_channel = args[i + 1]
                    if data_channel not in ['source_current', 'load_voltage']:
                        print(f"❌ Invalid channel: {data_channel}")
                        print("   Must be 'source_current' or 'load_voltage'")
                        return
                    i += 2
                else:
                    # Legacy support for positional N value
                    if i == 1 and args[i].isdigit():
                        n_value = int(args[i])
                    i += 1
            
            # Check if comma-separated list
            if ',' in ids_arg:
                # Multiple feature sets
                feature_set_ids = [int(id.strip()) for id in ids_arg.split(',')]
                
                print(f"🔄 Adding {len(feature_set_ids)} feature sets to experiment {self.current_experiment}...")
                print(f"   Data channel: {data_channel}")
                if n_value:
                    print(f"   Using N value: {n_value}")
                
                results = config.add_multiple_feature_sets(feature_set_ids, n_value, data_channel)
                
                # Report results
                success_count = sum(1 for success in results.values() if success)
                print(f"\n✅ Successfully added {success_count}/{len(feature_set_ids)} feature sets")
                
                for fs_id, success in results.items():
                    if not success:
                        print(f"   ⚠️  Feature set {fs_id} was already linked or doesn't exist")
            else:
                # Single feature set
                feature_set_id = int(ids_arg)
                
                print(f"🔄 Adding feature set {feature_set_id} to experiment {self.current_experiment}...")
                print(f"   Data channel: {data_channel}")
                if n_value:
                    print(f"   Using N value: {n_value}")
                
                if config.add_feature_set(feature_set_id, n_value, data_channel):
                    print(f"✅ Feature set {feature_set_id} added successfully")
                else:
                    print(f"⚠️  Feature set {feature_set_id} is already linked or doesn't exist")
            
            config.disconnect()
            
        except ValueError as e:
            print(f"❌ Invalid input: {e}")
            print("Feature set IDs and N value must be integers")
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error adding feature set: {e}")
    
    def cmd_remove_feature_set(self, args):
        """Remove a feature set from current experiment"""
        if not args:
            print("Usage: remove-feature-set <feature_set_id>")
            print("Use 'list-feature-sets' to see IDs")
            return
        
        try:
            feature_set_id = int(args[0])
        except ValueError:
            print(f"❌ Invalid feature set ID: {args[0]}")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Removing feature set {feature_set_id} from experiment {self.current_experiment}...")
            if configurator.remove_feature_set(feature_set_id):
                print(f"✅ Feature set {feature_set_id} removed")
            else:
                print(f"❌ Failed to remove feature set")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error removing feature set: {e}")
    
    def cmd_clear_feature_sets(self, args):
        """Remove all feature sets from current experiment"""
        response = input(f"⚠️  Remove ALL feature sets from experiment {self.current_experiment}? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
        
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Clearing all feature sets from experiment {self.current_experiment}...")
            if configurator.clear_all_feature_sets():
                print(f"✅ All feature sets cleared")
            else:
                print(f"❌ Failed to clear feature sets")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error clearing feature sets: {e}")

    def cmd_remove_data_type(self, args):
        """Remove a data type from current experiment"""
        if not args:
            print("Usage: remove-data-type <data_type_id>")
            print("\nData Type IDs:")
            print("  1 = raw")
            print("  2 = adc8")
            print("  3 = adc10")
            print("  4 = adc12")
            print("  5 = adc24")
            print("  6 = adc6")
            print("  7 = adc14")
            return

        try:
            data_type_id = int(args[0])
        except ValueError:
            print(f"❌ Invalid data type ID: {args[0]}")
            return

        try:
            from experiment_configurator import ExperimentConfigurator

            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            configurator = ExperimentConfigurator(self.current_experiment, db_config)

            print(f"🔄 Removing data type {data_type_id} from experiment {self.current_experiment}...")
            if configurator.remove_data_type(data_type_id):
                print(f"✅ Data type {data_type_id} removed")
            else:
                print(f"❌ Failed to remove data type")

        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error removing data type: {e}")

    def cmd_add_data_type(self, args):
        """Add a data type to current experiment"""
        if not args:
            print("Usage: add-data-type <data_type_id>")
            print("\nData Type IDs:")
            print("  1 = raw")
            print("  2 = adc8")
            print("  3 = adc10")
            print("  4 = adc12")
            print("  5 = adc24")
            print("  6 = adc6")
            print("  7 = adc14")
            return

        try:
            data_type_id = int(args[0])
        except ValueError:
            print(f"❌ Invalid data type ID: {args[0]}")
            return

        try:
            from experiment_configurator import ExperimentConfigurator

            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            configurator = ExperimentConfigurator(self.current_experiment, db_config)

            print(f"🔄 Adding data type {data_type_id} to experiment {self.current_experiment}...")
            if configurator.add_data_type(data_type_id):
                print(f"✅ Data type {data_type_id} added")
            else:
                print(f"❌ Data type already exists or failed to add")

        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error adding data type: {e}")

    def cmd_list_data_types(self, args):
        """List data types for current experiment"""
        try:
            import psycopg2

            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("""
                SELECT edt.data_type_id, dt.data_type_name, dt.description
                FROM ml_experiments_data_types edt
                JOIN ml_data_types_lut dt ON edt.data_type_id = dt.data_type_id
                WHERE edt.experiment_id = %s
                ORDER BY edt.data_type_id
            """, (self.current_experiment,))

            data_types = cursor.fetchall()

            if not data_types:
                print(f"\n❌ No data types configured for experiment {self.current_experiment}")
                return

            print(f"\n📊 Data Types for Experiment {self.current_experiment}:")
            print(f"\n{'ID':<5} {'Name':<15} {'Description':<50}")
            print("-" * 72)
            for dt_id, dt_name, dt_desc in data_types:
                desc = (dt_desc[:47] + '...') if dt_desc and len(dt_desc) > 50 else (dt_desc or '')
                print(f"{dt_id:<5} {dt_name:<15} {desc:<50}")

            print(f"\nTotal: {len(data_types)} data types")
            print("\nUse 'remove-data-type <id>' to remove a specific type")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error listing data types: {e}")

    def cmd_list_all_data_types(self, args):
        """List ALL available data types from ml_data_types_lut"""
        try:
            import psycopg2

            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("""
                SELECT data_type_id, data_type_name, description
                FROM ml_data_types_lut
                ORDER BY data_type_id
            """)

            data_types = cursor.fetchall()

            if not data_types:
                print(f"\n❌ No data types found in ml_data_types_lut")
                return

            print(f"\n📊 All Available Data Types:")
            print(f"\n{'ID':<5} {'Name':<15} {'Description':<50}")
            print("-" * 72)
            for dt_id, dt_name, dt_desc in data_types:
                desc = (dt_desc[:47] + '...') if dt_desc and len(dt_desc) > 50 else (dt_desc or '')
                print(f"{dt_id:<5} {dt_name:<15} {desc:<50}")

            print(f"\nTotal: {len(data_types)} data types")
            print("\nUse 'add-data-type <id>' to add a type to current experiment")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error listing all data types: {e}")

    def cmd_list_amplitude_methods(self, args):
        """List amplitude methods for current experiment"""
        try:
            import psycopg2

            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("""
                SELECT eam.method_id, am.method_name, am.function_name, am.description
                FROM ml_experiments_amplitude_methods eam
                JOIN ml_amplitude_normalization_lut am ON eam.method_id = am.method_id
                WHERE eam.experiment_id = %s
                ORDER BY eam.method_id
            """, (self.current_experiment,))

            methods = cursor.fetchall()

            if not methods:
                print(f"\n❌ No amplitude methods configured for experiment {self.current_experiment}")
                return

            print(f"\n📊 Amplitude Methods for Experiment {self.current_experiment}:")
            print(f"\n{'ID':<5} {'Name':<15} {'Function':<40} {'Description':<30}")
            print("-" * 95)
            for method_id, method_name, func_name, desc in methods:
                func_str = (func_name[:37] + '...') if func_name and len(func_name) > 40 else (func_name or '')
                desc_str = (desc[:27] + '...') if desc and len(desc) > 30 else (desc or '')
                print(f"{method_id:<5} {method_name:<15} {func_str:<40} {desc_str:<30}")

            print(f"\nTotal: {len(methods)} amplitude methods")
            print("\nUse 'update-amplitude-methods <name1> <name2> ...' to update")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error listing amplitude methods: {e}")

    def cmd_list_all_amplitude_methods(self, args):
        """List ALL available amplitude methods from ml_amplitude_normalization_lut"""
        try:
            import psycopg2

            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("""
                SELECT method_id, method_name, function_name, description
                FROM ml_amplitude_normalization_lut
                ORDER BY method_id
            """)

            methods = cursor.fetchall()

            if not methods:
                print(f"\n❌ No amplitude methods found in ml_amplitude_normalization_lut")
                return

            print(f"\n📊 All Available Amplitude Methods:")
            print(f"\n{'ID':<5} {'Name':<15} {'Function':<40} {'Description':<30}")
            print("-" * 95)
            for method_id, method_name, func_name, desc in methods:
                func_str = (func_name[:37] + '...') if func_name and len(func_name) > 40 else (func_name or '')
                desc_str = (desc[:27] + '...') if desc and len(desc) > 30 else (desc or '')
                print(f"{method_id:<5} {method_name:<15} {func_str:<40} {desc_str:<30}")

            print(f"\nTotal: {len(methods)} amplitude methods")
            print("\nUse 'update-amplitude-methods <name1> <name2> ...' to configure for current experiment")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error listing all amplitude methods: {e}")

    def cmd_list_feature_sets(self, args):
        """List feature sets for current experiment"""
        try:
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            config = configurator.get_current_config()
            
            feature_sets = config.get('feature_sets', [])
            
            if not feature_sets:
                print(f"No feature sets linked to experiment {self.current_experiment}")
                return
            
            print(f"\n🧬 Feature Sets for Experiment {self.current_experiment}:")
            print("-" * 60)
            
            for fs in feature_sets:
                print(f"• ID {fs.get('id', '?')}: {fs['name']}")
                print(f"  Features: {fs['features']}")
                print(f"  Data channel: {fs.get('data_channel', 'load_voltage')}")
                n_value = fs.get('n_value')
                if n_value:
                    print(f"  N value: {n_value}")
            
            print("-" * 60)
            print(f"Total: {len(feature_sets)} feature sets")
            print("\nUse 'remove-feature-set <id>' to remove a specific set")
            print("Use 'clear-feature-sets' to remove all")
                
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error listing feature sets: {e}")
    
    def cmd_show_all_feature_sets(self, args):
        """Show all available feature sets in the database"""
        try:
            import psycopg2
            import psycopg2.extras
            
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get all feature sets from the lookup table
            cursor.execute("""
                SELECT 
                    fsl.feature_set_id,
                    fsl.feature_set_name,
                    fsl.num_features,
                    fsl.category,
                    fsl.description,
                    STRING_AGG(fl.feature_name || ' (' || fl.behavior_type || ')', ', ' ORDER BY fsf.feature_order) as features
                FROM ml_feature_sets_lut fsl
                LEFT JOIN ml_feature_set_features fsf ON fsl.feature_set_id = fsf.feature_set_id
                LEFT JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                GROUP BY fsl.feature_set_id, fsl.feature_set_name, fsl.num_features, fsl.category, fsl.description
                ORDER BY fsl.feature_set_id
            """)
            
            results = cursor.fetchall()
            
            if not results:
                print("No feature sets found in database")
                return
            
            print(f"\n📚 ALL AVAILABLE FEATURE SETS IN DATABASE:")
            print("=" * 70)
            
            for fs in results:
                print(f"\n📦 ID {fs['feature_set_id']}: {fs['feature_set_name']}")
                print(f"   Category: {fs['category']}")
                if fs['description']:
                    print(f"   Description: {fs['description']}")
                print(f"   Number of features: {fs['num_features']}")
                
                if fs['features']:
                    features_str = fs['features']
                    if len(features_str) > 150:
                        # Truncate long feature lists
                        feature_list = features_str.split(', ')[:3]
                        print(f"   Features: {', '.join(feature_list)}...")
                        print(f"             (and {len(features_str.split(', ')) - 3} more)")
                    else:
                        print(f"   Features: {features_str}")
                
                # Check which experiments use this feature set
                cursor.execute("""
                    SELECT ARRAY_AGG(DISTINCT experiment_id ORDER BY experiment_id) as experiments
                    FROM ml_experiments_feature_sets
                    WHERE feature_set_id = %s
                """, (fs['feature_set_id'],))
                exp_result = cursor.fetchone()
                if exp_result and exp_result['experiments']:
                    print(f"   Used by experiments: {exp_result['experiments']}")
            
            print("\n" + "=" * 70)
            print(f"Total: {len(results)} feature sets available")
            print("\nTo link a feature set to current experiment, create it with:")
            print("  create-feature-set --name <name> --features <f1,f2,...>")
            
            cursor.close()
            conn.close()
                
        except psycopg2.Error as e:
            print(f"❌ Database error: {e}")
        except Exception as e:
            print(f"❌ Error showing feature sets: {e}")

    def cmd_create_feature(self, args):
        """Create a new feature in ml_features_lut"""
        if not args or '--name' not in args:
            print("Usage: create-feature --name <name> --category <category> --behavior <behavior> [--description <desc>]")
            print("\nCategories: electrical, statistical, spectral, temporal, compute")
            print("Behaviors: driver, derived, aggregate, transform")
            print("\nExample: create-feature --name impedance --category electrical --behavior derived --description 'Electrical impedance Z=V/I'")
            return

        name = None
        category = 'electrical'
        behavior = 'driver'
        description = None

        i = 0
        while i < len(args):
            if args[i] == '--name' and i + 1 < len(args):
                name = args[i + 1]
                i += 2
            elif args[i] == '--category' and i + 1 < len(args):
                category = args[i + 1]
                i += 2
            elif args[i] == '--behavior' and i + 1 < len(args):
                behavior = args[i + 1]
                i += 2
            elif args[i] == '--description' and i + 1 < len(args):
                description = args[i + 1]
                i += 2
            else:
                i += 1

        if not name:
            print("❌ Feature name is required")
            return

        valid_categories = ['electrical', 'statistical', 'spectral', 'temporal', 'compute']
        if category not in valid_categories:
            print(f"❌ Invalid category: {category}")
            print(f"   Must be one of: {', '.join(valid_categories)}")
            return

        valid_behaviors = ['driver', 'derived', 'aggregate', 'transform']
        if behavior not in valid_behaviors:
            print(f"❌ Invalid behavior: {behavior}")
            print(f"   Must be one of: {', '.join(valid_behaviors)}")
            return

        try:
            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("SELECT feature_id FROM ml_features_lut WHERE feature_name = %s", (name,))
            if cursor.fetchone():
                print(f"❌ Feature '{name}' already exists")
                conn.close()
                return

            cursor.execute("SELECT COALESCE(MAX(feature_id), 0) + 1 FROM ml_features_lut")
            feature_id = cursor.fetchone()[0]

            cursor.execute("""
                INSERT INTO ml_features_lut
                (feature_id, feature_name, feature_category, behavior_type, description, is_active, created_at)
                VALUES (%s, %s, %s, %s, %s, true, CURRENT_TIMESTAMP)
            """, (feature_id, name, category, behavior, description or f"{name} feature"))

            conn.commit()
            print(f"✅ Created feature '{name}' (ID: {feature_id})")
            print(f"   Category: {category}")
            print(f"   Behavior: {behavior}")
            if description:
                print(f"   Description: {description}")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error creating feature: {e}")

    def cmd_list_features(self, args):
        """List all available features"""
        category_filter = None
        if args and '--category' in args:
            idx = args.index('--category')
            if idx + 1 < len(args):
                category_filter = args[idx + 1]

        try:
            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            query = """
                SELECT
                    feature_id,
                    feature_name,
                    feature_category,
                    behavior_type,
                    description,
                    is_active
                FROM ml_features_lut
            """
            params = []

            if category_filter:
                query += " WHERE feature_category = %s"
                params.append(category_filter)

            query += " ORDER BY feature_category, feature_id"

            cursor.execute(query, params)
            features = cursor.fetchall()

            if not features:
                if category_filter:
                    print(f"No features found in category '{category_filter}'")
                else:
                    print("No features found in database")
                return

            from collections import defaultdict
            by_category = defaultdict(list)
            for f in features:
                by_category[f['feature_category']].append(f)

            print("\n📊 Available Features:")
            print("=" * 80)

            for category in sorted(by_category.keys()):
                print(f"\n🏷️  {category.upper()} Features:")
                print("-" * 40)

                for f in by_category[category]:
                    status = "✓" if f['is_active'] else "✗"
                    print(f"  {status} ID {f['feature_id']:3d}: {f['feature_name']:20s} ({f['behavior_type']:10s})")
                    if f['description'] and f['description'] != f'{f["feature_name"]} feature':
                        print(f"           {f['description'][:60]}")

            print("\n" + "=" * 80)
            print(f"Total: {len(features)} features")

            if not category_filter:
                print("\nFilter by category: list-features --category <category>")
                print("Categories: electrical, statistical, spectral, temporal, compute")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error listing features: {e}")

    def cmd_show_feature(self, args):
        """Show details of a specific feature"""
        if not args:
            print("Usage: show-feature <feature_id|feature_name>")
            return

        try:
            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            feature_arg = args[0]
            if feature_arg.isdigit():
                cursor.execute("SELECT * FROM ml_features_lut WHERE feature_id = %s", (int(feature_arg),))
            else:
                cursor.execute("SELECT * FROM ml_features_lut WHERE feature_name = %s", (feature_arg,))

            feature = cursor.fetchone()

            if not feature:
                print(f"❌ Feature '{feature_arg}' not found")
                return

            print(f"\n📊 Feature Details:")
            print("=" * 60)
            print(f"ID:           {feature['feature_id']}")
            print(f"Name:         {feature['feature_name']}")
            print(f"Category:     {feature.get('feature_category', 'N/A')}")
            print(f"Behavior:     {feature.get('behavior_type', 'N/A')}")
            print(f"Active:       {'✓' if feature.get('is_active', False) else '✗'}")
            print(f"Description:  {feature.get('description', 'N/A')}")
            print(f"Created:      {feature.get('created_at', 'N/A')}")

            cursor.execute("""
                SELECT
                    fs.feature_set_id,
                    fs.feature_set_name
                FROM ml_feature_set_features fsf
                JOIN ml_feature_sets_lut fs ON fsf.feature_set_id = fs.feature_set_id
                WHERE fsf.feature_id = %s
                ORDER BY fs.feature_set_id
            """, (feature['feature_id'],))

            feature_sets = cursor.fetchall()
            if feature_sets:
                print(f"\nUsed in {len(feature_sets)} feature set(s):")
                for fs in feature_sets:
                    print(f"  • ID {fs['feature_set_id']}: {fs['feature_set_name']}")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error showing feature: {e}")

    def cmd_update_feature(self, args):
        """Update feature properties"""
        if not args or len(args) < 2:
            print("Usage: update-feature <feature_id> [--name <name>] [--category <category>] [--description <desc>]")
            return

        try:
            feature_id = int(args[0])

            updates = {}
            i = 1
            while i < len(args):
                if args[i] == '--name' and i + 1 < len(args):
                    updates['feature_name'] = args[i + 1]
                    i += 2
                elif args[i] == '--category' and i + 1 < len(args):
                    updates['feature_category'] = args[i + 1]
                    i += 2
                elif args[i] == '--description' and i + 1 < len(args):
                    updates['description'] = args[i + 1]
                    i += 2
                else:
                    i += 1

            if not updates:
                print("❌ No updates specified")
                return

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            set_clauses = []
            params = []
            for key, value in updates.items():
                set_clauses.append(f"{key} = %s")
                params.append(value)
            params.append(feature_id)

            query = f"UPDATE ml_features_lut SET {', '.join(set_clauses)} WHERE feature_id = %s"
            cursor.execute(query, params)

            if cursor.rowcount == 0:
                print(f"❌ Feature {feature_id} not found")
            else:
                conn.commit()
                print(f"✅ Updated feature {feature_id}")
                for key, value in updates.items():
                    print(f"   {key}: {value}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid feature ID")
        except Exception as e:
            print(f"❌ Error updating feature: {e}")

    def cmd_delete_feature(self, args):
        """Delete a feature if not in use"""
        if not args:
            print("Usage: delete-feature <feature_id>")
            return

        try:
            feature_id = int(args[0])

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COUNT(*) FROM ml_feature_set_features WHERE feature_id = %s
            """, (feature_id,))

            count = cursor.fetchone()[0]
            if count > 0:
                print(f"❌ Cannot delete feature {feature_id}: used in {count} feature set(s)")
                print("   Remove from feature sets first using 'remove-features-from-set'")
                return

            cursor.execute("DELETE FROM ml_features_lut WHERE feature_id = %s", (feature_id,))

            if cursor.rowcount == 0:
                print(f"❌ Feature {feature_id} not found")
            else:
                conn.commit()
                print(f"✅ Deleted feature {feature_id}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid feature ID")
        except Exception as e:
            print(f"❌ Error deleting feature: {e}")

    def cmd_create_global_feature_set(self, args):
        """Create a feature set without linking to any experiment"""
        if not args or '--name' not in args:
            print("Usage: create-global-feature-set --name <name> [--category <category>] [--description <desc>]")
            print("\nExample: create-global-feature-set --name basic_electrical --category electrical --description 'Basic electrical measurements'")
            return

        name = None
        category = 'custom'
        description = None

        i = 0
        while i < len(args):
            if args[i] == '--name' and i + 1 < len(args):
                name = args[i + 1]
                i += 2
            elif args[i] == '--category' and i + 1 < len(args):
                category = args[i + 1]
                i += 2
            elif args[i] == '--description' and i + 1 < len(args):
                description = args[i + 1]
                i += 2
            else:
                i += 1

        if not name:
            print("❌ Feature set name is required")
            return

        try:
            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("SELECT feature_set_id FROM ml_feature_sets_lut WHERE feature_set_name = %s", (name,))
            if cursor.fetchone():
                print(f"❌ Feature set '{name}' already exists")
                conn.close()
                return

            cursor.execute("SELECT COALESCE(MAX(feature_set_id), 0) + 1 FROM ml_feature_sets_lut")
            feature_set_id = cursor.fetchone()[0]

            cursor.execute("""
                INSERT INTO ml_feature_sets_lut
                (feature_set_id, feature_set_name, category, description, is_active, created_at)
                VALUES (%s, %s, %s, %s, true, CURRENT_TIMESTAMP)
            """, (feature_set_id, name, category, description or f"{name} feature set"))

            conn.commit()
            print(f"✅ Created global feature set '{name}' (ID: {feature_set_id})")
            print(f"   Category: {category}")
            if description:
                print(f"   Description: {description}")
            print(f"\nNext: Add features using: add-features-to-set {feature_set_id} --features <id1,id2,...>")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error creating feature set: {e}")

    def cmd_add_features_to_set(self, args):
        """Add features to an existing feature set with optional per-feature overrides"""
        if not args or len(args) < 2 or '--features' not in args:
            print("Usage: add-features-to-set <feature_set_id> --features <feature_id1,feature_id2,...> [--channels <ch1,ch2,...>] [--n-values <n1,n2,...>]")
            print("\nExample: add-features-to-set 15 --features 1,2,3,4")
            print("         add-features-to-set 15 --features 2,2,2,2 --channels load_voltage,source_current,impedance,power")
            print("         add-features-to-set 15 --features 2,5 --channels impedance,null --n-values 128,null")
            print("\nUse 'list-features' to see available feature IDs")
            print("\nChannels: source_current, load_voltage, impedance, power, null (inherit from set)")
            return

        try:
            feature_set_id = int(args[0])

            features = []
            channels = []
            n_values = []

            # Parse --features
            if '--features' in args:
                idx = args.index('--features')
                if idx + 1 < len(args):
                    features = [int(f.strip()) for f in args[idx + 1].split(',')]

            # Parse --channels
            if '--channels' in args:
                idx = args.index('--channels')
                if idx + 1 < len(args):
                    channels = [ch.strip() if ch.strip().lower() not in ['default', 'null', 'none'] else None
                               for ch in args[idx + 1].split(',')]

            # Parse --n-values
            if '--n-values' in args:
                idx = args.index('--n-values')
                if idx + 1 < len(args):
                    n_values = [int(n.strip()) if n.strip().lower() not in ['default', 'null', 'none'] else None
                               for n in args[idx + 1].split(',')]

            if not features:
                print("❌ No features specified")
                return

            # Validate counts match
            if channels and len(channels) != len(features):
                print(f"❌ Channel count ({len(channels)}) must match feature count ({len(features)})")
                return

            if n_values and len(n_values) != len(features):
                print(f"❌ N-value count ({len(n_values)}) must match feature count ({len(features)})")
                return

            # Validate channels
            valid_channels = ['source_current', 'load_voltage', 'impedance', 'power', 'source_current,load_voltage']
            for ch in channels:
                if ch is not None and ch not in valid_channels:
                    print(f"❌ Invalid channel: {ch}")
                    print(f"   Must be one of: {', '.join(valid_channels)}")
                    return

            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute("SELECT feature_set_name FROM ml_feature_sets_lut WHERE feature_set_id = %s", (feature_set_id,))
            result = cursor.fetchone()
            if not result:
                print(f"❌ Feature set {feature_set_id} does not exist")
                conn.close()
                return

            feature_set_name = result['feature_set_name']

            cursor.execute("SELECT feature_id, feature_name FROM ml_features_lut WHERE feature_id = ANY(%s)", (features,))
            valid_features = {row['feature_id']: row['feature_name'] for row in cursor}

            invalid = [f for f in features if f not in valid_features]
            if invalid:
                print(f"❌ Invalid feature IDs: {invalid}")
                conn.close()
                return

            cursor.execute("""
                SELECT COALESCE(MAX(feature_order), 0) as max_order
                FROM ml_feature_set_features
                WHERE feature_set_id = %s
            """, (feature_set_id,))
            max_order = cursor.fetchone()['max_order']

            added = []
            skipped = []
            for i, feature_id in enumerate(features, 1):
                channel = channels[i-1] if channels and (i-1) < len(channels) else None
                n_value = n_values[i-1] if n_values and (i-1) < len(n_values) else None

                try:
                    cursor.execute("""
                        INSERT INTO ml_feature_set_features
                        (feature_set_id, feature_id, feature_order, data_channel, n_value_override)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (feature_set_id, feature_id, max_order + i, channel, n_value))

                    override_info = []
                    if channel:
                        override_info.append(f"ch={channel}")
                    if n_value:
                        override_info.append(f"n={n_value}")
                    info = f" [{', '.join(override_info)}]" if override_info else ""
                    added.append(f"{valid_features[feature_id]}{info}")
                    conn.commit()
                except psycopg2.IntegrityError:
                    skipped.append(valid_features[feature_id])
                    conn.rollback()

            print(f"✅ Updated feature set '{feature_set_name}' (ID: {feature_set_id})")
            if added:
                print(f"   Added {len(added)} features: {', '.join(added)}")
            if skipped:
                print(f"   Skipped {len(skipped)} (already in set): {', '.join(skipped)}")

            cursor.execute("""
                SELECT fl.feature_name
                FROM ml_feature_set_features fsf
                JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                WHERE fsf.feature_set_id = %s
                ORDER BY fsf.feature_order
            """, (feature_set_id,))

            all_features = [row['feature_name'] for row in cursor]
            print(f"\n   Total features in set: {len(all_features)}")
            print(f"   Features: {', '.join(all_features)}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid feature set ID or feature IDs")
        except Exception as e:
            print(f"❌ Error adding features: {e}")

    def cmd_remove_features_from_set(self, args):
        """Remove features from a feature set"""
        if not args or len(args) < 2 or '--features' not in args:
            print("Usage: remove-features-from-set <feature_set_id> --features <feature_id1,feature_id2,...>")
            return

        try:
            feature_set_id = int(args[0])

            features = []
            idx = args.index('--features')
            if idx + 1 < len(args):
                features = [int(f.strip()) for f in args[idx + 1].split(',')]

            if not features:
                print("❌ No features specified")
                return

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM ml_feature_set_features
                WHERE feature_set_id = %s AND feature_id = ANY(%s)
            """, (feature_set_id, features))

            removed = cursor.rowcount
            if removed > 0:
                conn.commit()
                print(f"✅ Removed {removed} feature(s) from feature set {feature_set_id}")

                cursor.execute("""
                    WITH reordered AS (
                        SELECT feature_set_id, feature_id,
                               ROW_NUMBER() OVER (PARTITION BY feature_set_id ORDER BY feature_order) as new_order
                        FROM ml_feature_set_features
                        WHERE feature_set_id = %s
                    )
                    UPDATE ml_feature_set_features fsf
                    SET feature_order = r.new_order
                    FROM reordered r
                    WHERE fsf.feature_set_id = r.feature_set_id
                      AND fsf.feature_id = r.feature_id
                """, (feature_set_id,))
                conn.commit()
                print("   Reordered remaining features")
            else:
                print(f"❌ No features removed (not found in set)")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid feature set ID or feature IDs")
        except Exception as e:
            print(f"❌ Error removing features: {e}")

    def cmd_update_feature_in_set(self, args):
        """Update feature assignment in a feature set"""
        if not args or len(args) < 2:
            print("Usage: update-feature-in-set <feature_set_id> <feature_id> [--channel <ch>] [--n-value <n>] [--order <order>]")
            print("\nExamples:")
            print("  update-feature-in-set 15 2 --channel impedance")
            print("  update-feature-in-set 15 2 --n-value 256")
            print("  update-feature-in-set 15 2 --channel null  (clear override, inherit from set)")
            print("  update-feature-in-set 15 2 --channel power --n-value 512 --order 3")
            return

        try:
            feature_set_id = int(args[0])
            feature_id = int(args[1])

            updates = {}
            i = 2
            while i < len(args):
                if args[i] == '--channel' and i + 1 < len(args):
                    value = args[i + 1]
                    updates['data_channel'] = None if value.lower() in ['null', 'none', 'default'] else value
                    i += 2
                elif args[i] == '--n-value' and i + 1 < len(args):
                    value = args[i + 1]
                    updates['n_value_override'] = None if value.lower() in ['null', 'none', 'default'] else int(value)
                    i += 2
                elif args[i] == '--order' and i + 1 < len(args):
                    updates['feature_order'] = int(args[i + 1])
                    i += 2
                else:
                    i += 1

            if not updates:
                print("❌ No updates specified")
                return

            # Validate channel if provided
            if 'data_channel' in updates and updates['data_channel'] is not None:
                valid_channels = ['source_current', 'load_voltage', 'impedance', 'power', 'source_current,load_voltage']
                if updates['data_channel'] not in valid_channels:
                    print(f"❌ Invalid channel: {updates['data_channel']}")
                    print(f"   Must be one of: {', '.join(valid_channels)}")
                    return

            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            # Build UPDATE query
            set_clause = ', '.join([f"{k} = %s" for k in updates.keys()])
            values = list(updates.values())

            cursor.execute(f"""
                UPDATE ml_feature_set_features
                SET {set_clause}
                WHERE feature_set_id = %s AND feature_id = %s
            """, values + [feature_set_id, feature_id])

            if cursor.rowcount == 0:
                print(f"❌ Feature {feature_id} not found in set {feature_set_id}")
                cursor.close()
                conn.close()
                return

            conn.commit()

            # Show updated configuration
            cursor.execute("""
                SELECT
                    fl.feature_name,
                    fsf.feature_order,
                    fsf.data_channel as feature_channel,
                    fsf.n_value_override,
                    efs.data_channel as set_channel,
                    efs.n_value as set_n_value
                FROM ml_feature_set_features fsf
                JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                JOIN ml_experiments_feature_sets efs ON fsf.feature_set_id = efs.feature_set_id
                WHERE fsf.feature_set_id = %s AND fsf.feature_id = %s
                LIMIT 1
            """, (feature_set_id, feature_id))

            row = cursor.fetchone()
            if row:
                effective_channel = row['feature_channel'] or row['set_channel']
                effective_n = row['n_value_override'] or row['set_n_value']

                print(f"✅ Updated {row['feature_name']} in set {feature_set_id} (order {row['feature_order']})")
                print(f"   Channel: {effective_channel} {'(override)' if row['feature_channel'] else '(inherit)'}")
                print(f"   N-value: {effective_n} {'(override)' if row['n_value_override'] else '(inherit)'}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid feature set ID or feature ID")
        except Exception as e:
            print(f"❌ Error updating feature: {e}")

    def cmd_clone_feature_set(self, args):
        """Create a copy of an existing feature set"""
        if not args or len(args) < 2 or '--name' not in args:
            print("Usage: clone-feature-set <source_feature_set_id> --name <new_name>")
            return

        try:
            source_id = int(args[0])

            new_name = None
            idx = args.index('--name')
            if idx + 1 < len(args):
                new_name = args[idx + 1]

            if not new_name:
                print("❌ New name is required")
                return

            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            cursor.execute("""
                SELECT * FROM ml_feature_sets_lut WHERE feature_set_id = %s
            """, (source_id,))
            source = cursor.fetchone()

            if not source:
                print(f"❌ Source feature set {source_id} not found")
                return

            cursor.execute("SELECT 1 FROM ml_feature_sets_lut WHERE feature_set_name = %s", (new_name,))
            if cursor.fetchone():
                print(f"❌ Feature set '{new_name}' already exists")
                return

            cursor.execute("SELECT COALESCE(MAX(feature_set_id), 0) + 1 FROM ml_feature_sets_lut")
            new_id = cursor.fetchone()['next_id']

            cursor.execute("""
                INSERT INTO ml_feature_sets_lut
                (feature_set_id, feature_set_name, category, description, is_active)
                VALUES (%s, %s, %s, %s, %s)
            """, (new_id, new_name, source['category'],
                  f"Clone of {source['feature_set_name']}: {source.get('description', '')}",
                  True))

            cursor.execute("""
                INSERT INTO ml_feature_set_features (feature_set_id, feature_id, feature_order)
                SELECT %s, feature_id, feature_order
                FROM ml_feature_set_features
                WHERE feature_set_id = %s
            """, (new_id, source_id))

            conn.commit()

            print(f"✅ Cloned feature set '{source['feature_set_name']}' (ID: {source_id})")
            print(f"   New set: '{new_name}' (ID: {new_id})")

            cursor.execute("""
                SELECT COUNT(*) as count FROM ml_feature_set_features WHERE feature_set_id = %s
            """, (new_id,))
            count = cursor.fetchone()['count']
            print(f"   Copied {count} features")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid source feature set ID")
        except Exception as e:
            print(f"❌ Error cloning feature set: {e}")

    def cmd_link_feature_set(self, args):
        """Link a feature set to an experiment with configuration"""
        if not args or len(args) < 2:
            print("Usage: link-feature-set <experiment_id> <feature_set_id> [--n-value <n>] [--channel <channel>] [--priority <p>] [--windowing <strategy>]")
            print("\nChannels: load_voltage, source_current, impedance, power")
            print("Windowing: non_overlapping (default), sliding_window")
            print("\nExample: link-feature-set 41 6 --n-value 64 --channel load_voltage --priority 1 --windowing non_overlapping")
            return

        try:
            experiment_id = int(args[0])
            feature_set_id = int(args[1])

            n_value = None
            channel = 'load_voltage'
            priority = None
            windowing_strategy = 'non_overlapping'

            i = 2
            while i < len(args):
                if args[i] == '--n-value' and i + 1 < len(args):
                    n_value = int(args[i + 1])
                    i += 2
                elif args[i] == '--channel' and i + 1 < len(args):
                    channel = args[i + 1]
                    i += 2
                elif args[i] == '--priority' and i + 1 < len(args):
                    priority = int(args[i + 1])
                    i += 2
                elif args[i] == '--windowing' and i + 1 < len(args):
                    windowing_strategy = args[i + 1]
                    if windowing_strategy not in ['non_overlapping', 'sliding_window']:
                        print(f"❌ Invalid windowing strategy: {windowing_strategy}")
                        print("   Must be 'non_overlapping' or 'sliding_window'")
                        return
                    i += 2
                else:
                    i += 1

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("""
                SELECT COALESCE(MAX(experiment_feature_set_id), 0) + 1 FROM ml_experiments_feature_sets
            """)
            efs_id = cursor.fetchone()[0]

            if priority is None:
                cursor.execute("""
                    SELECT COALESCE(MAX(priority_order), 0) + 1
                    FROM ml_experiments_feature_sets
                    WHERE experiment_id = %s
                """, (experiment_id,))
                priority = cursor.fetchone()[0]

            cursor.execute("""
                INSERT INTO ml_experiments_feature_sets
                (experiment_feature_set_id, experiment_id, feature_set_id, n_value, priority_order, is_active, data_channel, windowing_strategy)
                VALUES (%s, %s, %s, %s, %s, true, %s, %s)
            """, (efs_id, experiment_id, feature_set_id, n_value, priority, channel, windowing_strategy))

            conn.commit()
            print(f"✅ Linked feature set {feature_set_id} to experiment {experiment_id}")
            print(f"   Channel: {channel}")
            if n_value:
                print(f"   N-value: {n_value}")
            print(f"   Priority: {priority}")
            print(f"   Windowing: {windowing_strategy}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid experiment ID or feature set ID")
        except psycopg2.IntegrityError as e:
            print(f"❌ Link already exists or invalid IDs: {e}")
        except Exception as e:
            print(f"❌ Error linking feature set: {e}")

    def cmd_bulk_link_feature_sets(self, args):
        """Link multiple feature sets to an experiment"""
        if not args or len(args) < 2 or '--sets' not in args:
            print("Usage: bulk-link-feature-sets <experiment_id> --sets <id1,id2,id3,...> [--n-values <n1,n2,n3,...>]")
            print("\nExample: bulk-link-feature-sets 41 --sets 1,2,3,4,5 --n-values null,null,null,null,null")
            print("         bulk-link-feature-sets 41 --sets 6,7,8,9 --n-values 64,64,64,64")
            return

        try:
            experiment_id = int(args[0])

            sets = []
            idx = args.index('--sets')
            if idx + 1 < len(args):
                sets = [int(s.strip()) for s in args[idx + 1].split(',')]

            if not sets:
                print("❌ No feature sets specified")
                return

            n_values = [None] * len(sets)
            if '--n-values' in args:
                idx = args.index('--n-values')
                if idx + 1 < len(args):
                    n_val_strs = args[idx + 1].split(',')
                    for i, val in enumerate(n_val_strs[:len(sets)]):
                        if val.strip().lower() != 'null' and val.strip():
                            n_values[i] = int(val.strip())

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            cursor.execute("SELECT COALESCE(MAX(experiment_feature_set_id), 0) + 1 FROM ml_experiments_feature_sets")
            next_efs_id = cursor.fetchone()[0]

            cursor.execute("""
                SELECT COALESCE(MAX(priority_order), 0) + 1
                FROM ml_experiments_feature_sets
                WHERE experiment_id = %s
            """, (experiment_id,))
            next_priority = cursor.fetchone()[0]

            success = 0
            failed = 0

            for i, fs_id in enumerate(sets):
                try:
                    cursor.execute("""
                        INSERT INTO ml_experiments_feature_sets
                        (experiment_feature_set_id, experiment_id, feature_set_id, n_value, priority_order, is_active, data_channel)
                        VALUES (%s, %s, %s, %s, %s, true, 'load_voltage')
                    """, (next_efs_id, experiment_id, fs_id, n_values[i], next_priority))

                    next_efs_id += 1
                    next_priority += 1
                    success += 1
                    conn.commit()
                except psycopg2.IntegrityError:
                    failed += 1
                    conn.rollback()

            print(f"✅ Linked {success}/{len(sets)} feature sets to experiment {experiment_id}")
            if failed > 0:
                print(f"   ⚠️  {failed} feature sets were already linked or don't exist")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid experiment ID or feature set IDs")
        except Exception as e:
            print(f"❌ Error linking feature sets: {e}")

    def cmd_update_feature_link(self, args):
        """Update properties of an experiment-feature set link"""
        if not args or len(args) < 2:
            print("Usage: update-feature-link <experiment_id> <feature_set_id> [--n-value <n>] [--priority <p>] [--active <bool>] [--windowing <strategy>]")
            print("\nWindowing: non_overlapping, sliding_window")
            return

        try:
            experiment_id = int(args[0])
            feature_set_id = int(args[1])

            updates = {}
            i = 2
            while i < len(args):
                if args[i] == '--n-value' and i + 1 < len(args):
                    val = args[i + 1]
                    updates['n_value'] = None if val.lower() == 'null' else int(val)
                    i += 2
                elif args[i] == '--priority' and i + 1 < len(args):
                    updates['priority_order'] = int(args[i + 1])
                    i += 2
                elif args[i] == '--active' and i + 1 < len(args):
                    updates['is_active'] = args[i + 1].lower() in ['true', '1', 'yes']
                    i += 2
                elif args[i] == '--windowing' and i + 1 < len(args):
                    windowing_strategy = args[i + 1]
                    if windowing_strategy not in ['non_overlapping', 'sliding_window']:
                        print(f"❌ Invalid windowing strategy: {windowing_strategy}")
                        print("   Must be 'non_overlapping' or 'sliding_window'")
                        return
                    updates['windowing_strategy'] = windowing_strategy
                    i += 2
                else:
                    i += 1

            if not updates:
                print("❌ No updates specified")
                return

            import psycopg2
            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            set_clauses = []
            params = []
            for key, value in updates.items():
                set_clauses.append(f"{key} = %s")
                params.append(value)
            params.extend([experiment_id, feature_set_id])

            query = f"""
                UPDATE ml_experiments_feature_sets
                SET {', '.join(set_clauses)}
                WHERE experiment_id = %s AND feature_set_id = %s
            """
            cursor.execute(query, params)

            if cursor.rowcount == 0:
                print(f"❌ Link between experiment {experiment_id} and feature set {feature_set_id} not found")
            else:
                conn.commit()
                print(f"✅ Updated link between experiment {experiment_id} and feature set {feature_set_id}")
                for key, value in updates.items():
                    print(f"   {key}: {value}")

            cursor.close()
            conn.close()

        except ValueError:
            print("❌ Invalid experiment ID or feature set ID")
        except Exception as e:
            print(f"❌ Error updating feature link: {e}")

    def cmd_show_feature_config(self, args):
        """Show complete feature configuration for an experiment"""
        experiment_id = None
        if args and args[0].isdigit():
            experiment_id = int(args[0])
        else:
            experiment_id = self.current_experiment

        try:
            import psycopg2
            import psycopg2.extras

            conn = psycopg2.connect(
                host='localhost',
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

            print(f"\n🧬 Feature Configuration for Experiment {experiment_id}:")
            print("=" * 80)

            # First get feature sets
            cursor.execute("""
                SELECT
                    efs.priority_order,
                    fs.feature_set_id,
                    fs.feature_set_name,
                    fs.category,
                    efs.n_value,
                    efs.data_channel,
                    efs.is_active,
                    efs.windowing_strategy,
                    fs.description
                FROM ml_experiments_feature_sets efs
                JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
                WHERE efs.experiment_id = %s
                ORDER BY efs.priority_order
            """, (experiment_id,))

            feature_sets = cursor.fetchall()

            # Get features for each set with overrides
            feature_details = {}
            for fs in feature_sets:
                cursor.execute("""
                    SELECT
                        fl.feature_name,
                        fsf.feature_order,
                        fsf.data_channel as feature_channel,
                        fsf.n_value_override,
                        COALESCE(fsf.data_channel, %s) as effective_channel,
                        COALESCE(fsf.n_value_override, %s) as effective_n_value
                    FROM ml_feature_set_features fsf
                    JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                    WHERE fsf.feature_set_id = %s
                    ORDER BY fsf.feature_order
                """, (fs['data_channel'], fs['n_value'], fs['feature_set_id']))
                feature_details[fs['feature_set_id']] = cursor.fetchall()

            if not feature_sets:
                print("No feature sets configured for this experiment")
                print("\nAdd feature sets using:")
                print("  link-feature-set <exp_id> <fs_id> [options]")
                print("  bulk-link-feature-sets <exp_id> --sets <ids> [options]")
                return

            # Display feature sets with detailed features
            for row in feature_sets:
                status = "✓" if row['is_active'] else "✗"
                print(f"\n[{row['feature_set_id']}] {row['feature_set_name']} (Priority: {row['priority_order']}, Status: {status})")
                print(f"    Default Channel: {row['data_channel'] or 'N/A'}")
                print(f"    Default N-value: {row['n_value'] or 'N/A'}")
                print(f"    Windowing: {row['windowing_strategy']}")

                if row['description'] and row['description'] != f"{row['feature_set_name']} feature set":
                    print(f"    Description: {row['description']}")

                # Show features with overrides
                features = feature_details.get(row['feature_set_id'], [])
                if features:
                    print(f"    Features ({len(features)}):")
                    for feat in features:
                        # Build override indicators
                        overrides = []
                        if feat['feature_channel']:
                            overrides.append(f"ch={feat['feature_channel']}")
                        if feat['n_value_override']:
                            overrides.append(f"n={feat['n_value_override']}")

                        override_str = f" [{', '.join(overrides)}]" if overrides else ""
                        print(f"      {feat['feature_order']}. {feat['feature_name']}({feat['effective_channel']}, n={feat['effective_n_value']}){override_str}")
                else:
                    print(f"    Features: None configured")

            print("\n" + "=" * 80)
            print(f"Total: {len(feature_sets)} feature sets configured")

            active = sum(1 for r in feature_sets if r['is_active'])
            with_n = sum(1 for r in feature_sets if r['n_value'])

            print(f"Status: {active} active, {len(feature_sets) - active} inactive")
            print(f"N-values: {with_n} sets have window sizes configured")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"❌ Error showing feature configuration: {e}")

    def cmd_update_selection_config(self, args):
        """Update segment selection configuration"""
        if not args or all(not arg.startswith('--') for arg in args):
            print("Usage: update-selection-config [options]")
            print("Options:")
            print("  --max-files <n>    Max files per label (e.g., 50)")
            print("  --seed <n>         Random seed for reproducibility")
            print("  --strategy <s>     Selection strategy (e.g., position_balanced_per_file)")
            print("  --balanced <bool>  Enable balanced segments (true/false)")
            print("\nExample: update-selection-config --max-files 50 --seed 42")
            return
        
        try:
            # Parse arguments
            config_updates = {}
            i = 0
            while i < len(args):
                if args[i] == '--max-files' and i + 1 < len(args):
                    config_updates['max_files_per_label'] = int(args[i + 1])
                    i += 2
                elif args[i] == '--seed' and i + 1 < len(args):
                    config_updates['random_seed'] = int(args[i + 1])
                    i += 2
                elif args[i] == '--strategy' and i + 1 < len(args):
                    config_updates['selection_strategy'] = args[i + 1]
                    i += 2
                elif args[i] == '--balanced' and i + 1 < len(args):
                    config_updates['balanced_segments'] = args[i + 1].lower() == 'true'
                    i += 2
                else:
                    i += 1
            
            if not config_updates:
                print("❌ No valid parameters provided")
                return
            
            from experiment_configurator import ExperimentConfigurator
            
            db_config = {
                'host': 'localhost',
                'database': 'arc_detection',
                'user': 'kjensen'
            }
            
            configurator = ExperimentConfigurator(self.current_experiment, db_config)
            
            print(f"🔄 Updating segment selection config for experiment {self.current_experiment}...")
            if configurator.update_segment_selection_config(config_updates):
                print(f"✅ Segment selection config updated:")
                for key, value in config_updates.items():
                    print(f"   {key}: {value}")
            else:
                print(f"❌ Failed to update segment selection config")
                
        except ValueError as e:
            print(f"❌ Invalid value: {e}")
        except ImportError as e:
            print(f"❌ Could not import configurator: {e}")
        except Exception as e:
            print(f"❌ Error updating selection config: {e}")
    
    def cmd_select_files(self, args):
        """Select files for experiment training data"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
            
        max_files = 50  # Default
        seed = 42  # Default for experiment 41
        strategy = 'random'  # Default strategy
        min_quality = None
        dry_run = False
        
        # Parse arguments
        i = 0
        while i < len(args):
            if args[i] == '--max-files' and i + 1 < len(args):
                max_files = int(args[i + 1])
                i += 2
            elif args[i] == '--seed' and i + 1 < len(args):
                seed = int(args[i + 1])
                i += 2
            elif args[i] == '--strategy' and i + 1 < len(args):
                strategy = args[i + 1]
                i += 2
            elif args[i] == '--min-quality' and i + 1 < len(args):
                min_quality = float(args[i + 1])
                i += 2
            elif args[i] == '--dry-run':
                dry_run = True
                i += 1
            elif args[i] == '--help':
                print("\nUsage: select-files [options]")
                print("\nOptions:")
                print("  --strategy STRATEGY    Selection strategy: random|balanced|quality_first (default: random)")
                print("  --max-files N         Maximum files per label (default: 50)")
                print("  --seed N              Random seed for reproducibility (default: 42)")
                print("  --min-quality N       Minimum quality score for quality_first strategy")
                print("  --dry-run             Preview selection without saving to database")
                print("\nExample:")
                print("  select-files --strategy random --max-files 50 --seed 42")
                return
            else:
                i += 1
        
        print(f"🔄 Selecting files for experiment {self.current_experiment}...")
        print(f"   Strategy: {strategy}")
        print(f"   Max files per label: {max_files}")
        print(f"   Random seed: {seed}")
        if min_quality:
            print(f"   Minimum quality: {min_quality}")
        
        try:
            from experiment_file_selector import ExperimentFileSelector
            
            selector = ExperimentFileSelector(self.current_experiment, self.db_conn)
            
            if dry_run:
                # Preview available files
                files_by_label = selector.get_available_files()
                print(f"\n📊 Available files by label:")
                total_available = 0
                for label, files in files_by_label.items():
                    print(f"   {label}: {len(files)} files")
                    total_available += len(files)
                print(f"   Total: {total_available} files")
                print("\n💡 Run without --dry-run to save selection")
                return
            
            # Perform selection
            result = selector.select_files(
                strategy=strategy,
                max_files_per_label=max_files,
                seed=seed,
                min_quality=min_quality
            )
            
            if result['success']:
                print(f"\n✅ Successfully selected {result['total_selected']} files")
                
                # Display statistics
                stats = result['statistics']
                if stats and 'label_counts' in stats:
                    print("\n📊 Files selected per label:")
                    for label, count in stats['label_counts'].items():
                        print(f"   {label}: {count} files")
                    print(f"\n   Total unique files: {stats['unique_files']}")
                    print(f"   Total unique labels: {stats['unique_labels']}")
                
                print(f"\n💾 Data saved to: experiment_{self.current_experiment:03d}_file_training_data")
            else:
                print(f"❌ Failed to select files: {result.get('error', 'Unknown error')}")
            
        except ImportError:
            print("❌ ExperimentFileSelector module not found")
        except Exception as e:
            print(f"❌ Error selecting files: {e}")
    
    def cmd_remove_file_labels(self, args):
        """Remove specific file labels from experiment training data"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return
        
        if not args:
            print("Usage: remove-file-labels <label1> [label2] [label3] ...")
            print("\nExample:")
            print("  remove-file-labels trash voltage_only arc_short_gap")
            print("\nThis removes all files with the specified labels from the training data.")
            return
        
        # Parse labels from arguments
        labels_to_remove = args
        
        table_name = f"experiment_{self.current_experiment:03d}_file_training_data"
        
        print(f"🗑️  Removing file labels from experiment {self.current_experiment}...")
        print(f"   Labels to remove: {', '.join(labels_to_remove)}")
        
        cursor = self.db_conn.cursor()
        try:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = %s
                )
            """, (table_name,))
            
            if not cursor.fetchone()[0]:
                print(f"❌ Table {table_name} does not exist")
                return
            
            # Check which column name is used for labels
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                AND column_name IN ('assigned_label', 'file_label_name')
                LIMIT 1
            """, (table_name,))

            label_column_result = cursor.fetchone()
            if not label_column_result:
                print("❌ No label column found in the table")
                return

            label_column = label_column_result[0]

            # Get counts before deletion using correct column
            cursor.execute(f"""
                SELECT {label_column}, COUNT(*) as count
                FROM {table_name}
                WHERE {label_column} = ANY(%s)
                GROUP BY {label_column}
            """, (labels_to_remove,))
            
            labels_found = {}
            for row in cursor:
                labels_found[row[0]] = row[1]
            
            if not labels_found:
                print("⚠️  No files found with the specified labels")
                return
            
            print("\n📊 Files to be removed:")
            total_to_remove = 0
            for label, count in labels_found.items():
                print(f"   {label}: {count} files")
                total_to_remove += count
            
            # Ask for confirmation
            response = input(f"\n⚠️  Remove {total_to_remove} files? (y/n): ")
            if response.lower() != 'y':
                print("❌ Removal cancelled")
                return
            
            # Delete the files using correct column
            cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE {label_column} = ANY(%s)
            """, (labels_to_remove,))
            
            deleted = cursor.rowcount
            self.db_conn.commit()
            
            print(f"\n✅ Successfully removed {deleted} files")
            
            # Show remaining statistics
            cursor.execute(f"""
                SELECT
                    COUNT(DISTINCT file_id) as total_files,
                    COUNT(DISTINCT {label_column}) as unique_labels
                FROM {table_name}
                WHERE experiment_id = %s
            """, (self.current_experiment,))

            stats = cursor.fetchone()
            print(f"\n📊 Remaining in training data:")
            print(f"   Total files: {stats[0]}")
            print(f"   Unique labels: {stats[1]}")

            # Show remaining label distribution
            cursor.execute(f"""
                SELECT {label_column}, COUNT(*) as count
                FROM {table_name}
                WHERE experiment_id = %s
                GROUP BY {label_column}
                ORDER BY count DESC
            """, (self.current_experiment,))
            
            print("\n📊 Remaining label distribution:")
            for row in cursor:
                print(f"   {row[0]}: {row[1]} files")
            
        except Exception as e:
            self.db_conn.rollback()
            print(f"❌ Error removing file labels: {e}")
        finally:
            cursor.close()

    def cmd_remove_files(self, args):
        """Remove specific files from experiment training data"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        if not args:
            print("Usage: remove-files <file_id1> [file_id2] [file_id3] ...")
            print("\nExample:")
            print("  remove-files 1234 5678 9012")
            print("\nThis removes specific files by ID from the training data.")
            return

        # Parse file IDs from arguments
        file_ids = []
        for arg in args:
            try:
                file_ids.append(int(arg))
            except ValueError:
                print(f"⚠️ Skipping invalid file ID: {arg}")

        if not file_ids:
            print("❌ No valid file IDs provided")
            return

        table_name = f"experiment_{self.current_experiment:03d}_file_training_data"

        print(f"🗑️  Removing {len(file_ids)} files from experiment {self.current_experiment}...")

        cursor = self.db_conn.cursor()
        try:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (table_name,))

            if not cursor.fetchone()[0]:
                print(f"❌ Table {table_name} does not exist")
                return

            # Delete the files
            cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE file_id = ANY(%s) AND experiment_id = %s
            """, (file_ids, self.current_experiment))

            deleted = cursor.rowcount
            self.db_conn.commit()

            print(f"✅ Successfully removed {deleted} files")

            # Show remaining statistics
            cursor.execute(f"""
                SELECT COUNT(DISTINCT file_id) FROM {table_name}
                WHERE experiment_id = %s
            """, (self.current_experiment,))

            remaining = cursor.fetchone()[0]
            print(f"📊 Remaining files in training data: {remaining}")

        except Exception as e:
            self.db_conn.rollback()
            print(f"❌ Error removing files: {e}")
        finally:
            cursor.close()

    def cmd_remove_segments(self, args):
        """Remove specific segments from experiment training data"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        if not args:
            print("Usage: remove-segments <segment_id1> [segment_id2] [segment_id3] ...")
            print("\nExample:")
            print("  remove-segments 104075 104076 104077")
            print("\nThis removes specific segments by ID from the training data.")
            return

        # Parse segment IDs from arguments
        segment_ids = []
        for arg in args:
            try:
                segment_ids.append(int(arg))
            except ValueError:
                print(f"⚠️ Skipping invalid segment ID: {arg}")

        if not segment_ids:
            print("❌ No valid segment IDs provided")
            return

        table_name = f"experiment_{self.current_experiment:03d}_segment_training_data"

        print(f"🗑️  Removing {len(segment_ids)} segments from experiment {self.current_experiment}...")

        cursor = self.db_conn.cursor()
        try:
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (table_name,))

            if not cursor.fetchone()[0]:
                print(f"❌ Table {table_name} does not exist")
                print("   Run 'select-segments' first to create segment training data")
                return

            # Delete the segments
            cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE segment_id = ANY(%s) AND experiment_id = %s
            """, (segment_ids, self.current_experiment))

            deleted = cursor.rowcount
            self.db_conn.commit()

            print(f"✅ Successfully removed {deleted} segments")

            # Show remaining statistics
            cursor.execute(f"""
                SELECT COUNT(DISTINCT segment_id) FROM {table_name}
                WHERE experiment_id = %s
            """, (self.current_experiment,))

            remaining = cursor.fetchone()[0]
            print(f"📊 Remaining segments in training data: {remaining}")

        except Exception as e:
            self.db_conn.rollback()
            print(f"❌ Error removing segments: {e}")
        finally:
            cursor.close()

    def cmd_generate_training_data(self, args):
        """Deprecated - use select-segments instead"""
        print("⚠️  This command has been replaced by 'select-segments' for clarity.")
        print("\nUse: select-segments [experiment_id] [options]")
        print("\nExample:")
        print("  select-segments 41 --strategy balanced")
        print("  select-segments --help  (for all options)")
        print("\nRedirecting to select-segments...")
        print()

        # Redirect to the proper command
        self.cmd_select_segments(args)

    def cmd_clean_segment_pairs(self, args):
        """Clean (delete all rows from) the segment pairs table for current experiment"""
        # Parse experiment ID if provided, otherwise use current
        if args and args[0].isdigit():
            experiment_id = int(args[0])
        else:
            experiment_id = self.current_experiment

        if not experiment_id:
            print("❌ No experiment specified. Use: clean-segment-pairs <experiment_id>")
            print("   Or set current experiment: set experiment <id>")
            return

        table_name = f"experiment_{experiment_id:03d}_segment_pairs"

        # Connect to database
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        try:
            cursor = self.db_conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (table_name,))

            table_exists = cursor.fetchone()[0]

            if not table_exists:
                print(f"ℹ️  Table {table_name} does not exist (nothing to clean)")
                cursor.close()
                return

            # Get count before deletion
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count_before = cursor.fetchone()[0]

            if count_before == 0:
                print(f"ℹ️  Table {table_name} is already empty")
                cursor.close()
                return

            # Show what will be deleted
            print(f"\n📊 Segment pairs table: {table_name}")
            print(f"   Current rows: {count_before:,}")

            # Confirmation
            print(f"\n⚠️  WARNING: This will delete all {count_before:,} pairs from {table_name}")
            print(f"⚠️  This action CANNOT be undone!")
            response = input(f"\nType 'DELETE' to confirm: ").strip()

            if response != 'DELETE':
                print("❌ Cancelled")
                cursor.close()
                return

            # Delete all rows
            print(f"\n🗑️  Deleting all rows from {table_name}...")
            cursor.execute(f"DELETE FROM {table_name}")
            self.db_conn.commit()

            # Verify deletion
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count_after = cursor.fetchone()[0]

            if count_after == 0:
                print(f"✅ Deleted {count_before:,} pairs")
                print(f"✅ Table {table_name} is now empty")
            else:
                print(f"⚠️  Warning: {count_after} pairs remaining")

            cursor.close()

        except Exception as e:
            print(f"❌ Error cleaning segment pairs table: {e}")
            if self.db_conn:
                self.db_conn.rollback()

    def cmd_generate_segment_pairs(self, args):
        """Generate segment pairs for distance calculations"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        pairing_strategy = 'match_lengths_all_combinations'  # Default (safe for same-size comparison)
        max_pairs_per_segment = None
        same_label_ratio = 0.5
        seed = 42
        clean_first = False  # Default

        # Parse arguments
        i = 0
        while i < len(args):
            if args[i] == '--strategy' and i + 1 < len(args):
                pairing_strategy = args[i + 1]
                i += 2
            elif args[i] == '--max-pairs-per-segment' and i + 1 < len(args):
                max_pairs_per_segment = int(args[i + 1])
                i += 2
            elif args[i] == '--same-label-ratio' and i + 1 < len(args):
                same_label_ratio = float(args[i + 1])
                i += 2
            elif args[i] == '--seed' and i + 1 < len(args):
                seed = int(args[i + 1])
                i += 2
            elif args[i] == '--clean':
                clean_first = True
                i += 1
            elif args[i] == '--help':
                print("\nUsage: generate-segment-pairs [options]")
                print("\nOptions:")
                print("  --strategy STRAT            Pairing strategy (default: match_lengths_all_combinations)")
                print("                              Options: match_lengths_all_combinations, all_combinations,")
                print("                                       balanced, code_type_balanced, random_sample")
                print("  --max-pairs-per-segment N   Maximum pairs per segment")
                print("  --same-label-ratio RATIO    Ratio of same-label pairs for balanced strategy (0.0-1.0)")
                print("  --seed N                    Random seed (default: 42)")
                print("  --clean                     Clear existing segment pairs before generation")
                print("\nStrategies:")
                print("  match_lengths_all_combinations - Generate all pairs ONLY for segments with same length (RECOMMENDED)")
                print("  all_combinations              - Generate all possible pairs regardless of length")
                print("  balanced                      - Balance same/different label pairs")
                print("  code_type_balanced            - Balance pairs by segment code type (L, R, C, etc.)")
                print("  random_sample                 - Random sample of possible pairs")
                print("\nExample:")
                print("  generate-segment-pairs --strategy match_lengths_all_combinations")
                print("  generate-segment-pairs --clean --strategy all_combinations")
                print("  generate-segment-pairs --clean --strategy code_type_balanced --max-pairs-per-segment 100")
                print("  generate-segment-pairs --strategy balanced --same-label-ratio 0.3")
                return
            else:
                i += 1

        # Clean existing pairs if requested
        if clean_first:
            print(f"\n🗑️  Cleaning existing segment pairs...")
            self.cmd_clean_segment_pairs([])
            print()

        print(f"🔄 Generating segment pairs for experiment {self.current_experiment}...")
        print(f"   Strategy: {pairing_strategy}")
        if max_pairs_per_segment:
            print(f"   Max pairs per segment: {max_pairs_per_segment}")
        print(f"   Same label ratio: {same_label_ratio}")
        print(f"   Random seed: {seed}")

        try:
            # Import the v2 segment pair generator module (compatible with v2 selector)
            from experiment_segment_pair_generator_v2 import ExperimentSegmentPairGeneratorV2

            # Create generator instance
            generator = ExperimentSegmentPairGeneratorV2(self.current_experiment, self.db_conn)

            # Generate pairs with progress indicator
            print(f"\n⏳ Generating pairs (this may take several minutes)...")
            import time
            start_time = time.time()

            result = generator.generate_pairs(
                strategy=pairing_strategy,
                max_pairs_per_segment=max_pairs_per_segment,
                same_label_ratio=same_label_ratio,
                seed=seed
            )

            elapsed_time = time.time() - start_time
            print(f"✓ Pair generation completed in {elapsed_time:.1f} seconds")
            
            if result['success']:
                print(f"\n✅ Successfully generated segment pairs!")
                print(f"   Total segments: {result['total_segments']}")
                print(f"   Total pairs: {result['total_pairs']}")
                
                # Display statistics
                if 'statistics' in result and result['statistics']:
                    stats = result['statistics']
                    print("\n📊 Pair Statistics:")
                    print(f"   Same segment label pairs: {stats.get('same_segment_label_pairs', 0)}")
                    print(f"   Same file label pairs: {stats.get('same_file_label_pairs', 0)}")
                    print(f"   Same code type pairs: {stats.get('same_code_type_pairs', 0)}")

                    if 'type_distribution' in stats:
                        print("\n   Pair type distribution:")
                        for pair_type, count in sorted(stats['type_distribution'].items()):
                            print(f"     {pair_type}: {count}")

                    if 'top_code_type_pairs' in stats and stats['top_code_type_pairs']:
                        print("\n   Top code type combinations:")
                        for pair in stats['top_code_type_pairs'][:5]:
                            print(f"     {pair}")
            else:
                print(f"\n❌ Failed to generate pairs: {result.get('error', 'Unknown error')}")
                
        except ImportError:
            print("❌ ExperimentSegmentPairGeneratorV2 module not found")
            print("   Make sure experiment_segment_pair_generator_v2.py is in the same directory")
        except Exception as e:
            print(f"❌ Error generating segment pairs: {e}")

    def cmd_init_distance_tables(self, args):
        """Initialize distance result tables for current experiment

        Usage: init-distance-tables [options]

        Options:
            --drop-existing    Drop existing tables before creating (WARNING: destroys data)
            --help             Show this help message

        This command creates all necessary distance result tables for the current experiment
        based on the distance functions configured in ml_distance_functions_lut.

        Examples:
            init-distance-tables
            init-distance-tables --drop-existing
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        drop_existing = False
        if '--help' in args:
            print(self.cmd_init_distance_tables.__doc__)
            return
        if '--drop-existing' in args:
            drop_existing = True

        print(f"\n🔄 Initializing distance tables for experiment {self.current_experiment}...")

        try:
            import psycopg2

            cursor = self.db_conn.cursor()

            # Get distance functions configured for this experiment
            cursor.execute("""
                SELECT df.distance_function_id, df.function_name, df.result_table_prefix, df.display_name
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s AND df.is_active = true
                ORDER BY df.distance_function_id
            """, (self.current_experiment,))

            distance_functions = cursor.fetchall()

            if not distance_functions:
                print(f"❌ No distance functions configured for experiment {self.current_experiment}")
                print("   Check ml_experiments_distance_measurements table")
                return

            print(f"📊 Found {len(distance_functions)} active distance functions")
            print()

            # If --drop-existing is specified, first check which tables exist and get confirmation
            if drop_existing:
                tables_to_drop = []
                for func_id, func_name, table_prefix, display_name in distance_functions:
                    table_name = f"experiment_{self.current_experiment:03d}_{table_prefix}".lower()

                    # Check if table exists and get row count
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = %s
                        )
                    """, (table_name,))

                    if cursor.fetchone()[0]:
                        # Get row count
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        row_count = cursor.fetchone()[0]
                        tables_to_drop.append((table_name, display_name, row_count))

                if tables_to_drop:
                    print(f"\n⚠️  WARNING: The following tables will be PERMANENTLY DELETED:")
                    print()
                    total_rows = 0
                    for table_name, display_name, row_count in tables_to_drop:
                        print(f"   📊 {table_name}")
                        print(f"      ({display_name}): {row_count:,} records")
                        total_rows += row_count
                    print()
                    print(f"   🔢 Total records to delete: {total_rows:,}")
                    print()
                    print(f"⚠️  This action CANNOT be undone!")
                    print(f"⚠️  ALL distance data for experiment {self.current_experiment} will be lost!")
                    print()
                    response = input("Type 'DROP' to confirm deletion: ").strip()

                    if response != 'DROP':
                        print("❌ Cancelled - no tables were dropped")
                        return
                    print()

            created_count = 0
            skipped_count = 0
            error_count = 0

            for func_id, func_name, table_prefix, display_name in distance_functions:
                # PostgreSQL stores table names in lowercase, so normalize
                table_name = f"experiment_{self.current_experiment:03d}_{table_prefix}".lower()

                try:
                    # Check if table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = %s
                        )
                    """, (table_name,))

                    table_exists = cursor.fetchone()[0]

                    if table_exists:
                        if drop_existing:
                            print(f"🗑️  Dropping existing table: {table_name}")
                            cursor.execute(f"DROP TABLE {table_name} CASCADE")
                            self.db_conn.commit()
                        else:
                            print(f"⏭️  Skipping {table_name} (already exists)")
                            skipped_count += 1
                            continue

                    # Create distance result table with composite primary key
                    create_sql = f"""
                        CREATE TABLE {table_name} (
                            pair_id INTEGER NOT NULL,
                            decimation_factor INTEGER NOT NULL,
                            data_type_id INTEGER NOT NULL,
                            amplitude_processing_method_id INTEGER NOT NULL,
                            experiment_feature_set_id BIGINT NOT NULL,
                            feature_set_feature_id BIGINT NOT NULL,
                            distance_s DOUBLE PRECISION,
                            distance_i DOUBLE PRECISION,
                            distance_j DOUBLE PRECISION,
                            distance_k DOUBLE PRECISION,
                            created_at TIMESTAMP DEFAULT NOW(),
                            PRIMARY KEY (pair_id, decimation_factor, data_type_id,
                                       amplitude_processing_method_id, experiment_feature_set_id,
                                       feature_set_feature_id)
                        )
                    """

                    cursor.execute(create_sql)

                    # Create indexes for common query patterns
                    cursor.execute(f"CREATE INDEX idx_{table_name}_pair ON {table_name}(pair_id)")
                    cursor.execute(f"CREATE INDEX idx_{table_name}_decimation ON {table_name}(decimation_factor)")
                    cursor.execute(f"CREATE INDEX idx_{table_name}_data_type ON {table_name}(data_type_id)")
                    cursor.execute(f"CREATE INDEX idx_{table_name}_feature_set ON {table_name}(experiment_feature_set_id)")
                    cursor.execute(f"CREATE INDEX idx_{table_name}_pair_decimation ON {table_name}(pair_id, decimation_factor)")

                    self.db_conn.commit()

                    print(f"✅ Created table: {table_name} ({display_name})")
                    created_count += 1

                except Exception as e:
                    print(f"❌ Error creating {table_name}: {e}")
                    self.db_conn.rollback()
                    error_count += 1

            print()
            print(f"📊 Summary:")
            print(f"   Created: {created_count}")
            print(f"   Skipped: {skipped_count}")
            print(f"   Errors: {error_count}")
            print()

            if created_count > 0:
                print(f"✅ Distance tables initialized for experiment {self.current_experiment}")
            elif skipped_count > 0:
                print(f"ℹ️  All tables already exist. Use --drop-existing to recreate them.")

        except Exception as e:
            print(f"❌ Error initializing distance tables: {e}")
            import traceback
            traceback.print_exc()

    def cmd_show_distance_metrics(self, args):
        """Show distance metrics configured for current experiment

        Usage: show-distance-metrics

        Displays all distance metrics configured in ml_experiments_distance_measurements
        for the current experiment.
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        try:
            cursor = self.db_conn.cursor()

            # Get distance functions configured for this experiment
            cursor.execute("""
                SELECT df.distance_function_id, df.function_name, df.display_name, df.result_table_prefix
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s
                ORDER BY df.function_name
            """, (self.current_experiment,))

            metrics = cursor.fetchall()

            if not metrics:
                print(f"\n❌ No distance metrics configured for experiment {self.current_experiment}")
                return

            print(f"\n📊 Distance metrics configured for experiment {self.current_experiment}:")
            print(f"\nID  | Function Name        | Display Name                      | Table Prefix")
            print("-" * 90)
            for metric in metrics:
                print(f"{metric[0]:<4}| {metric[1]:<20} | {metric[2]:<33} | {metric[3]}")

            print(f"\nTotal: {len(metrics)} metrics")

        except Exception as e:
            print(f"❌ Error showing distance metrics: {e}")

    def cmd_add_distance_metric(self, args):
        """Add distance metric to current experiment

        Usage: add-distance-metric --metric <metric_name>

        Options:
            --metric <name>    Metric name (e.g., L1, L2, cosine, pearson, wasserstein)

        Examples:
            add-distance-metric --metric wasserstein
            add-distance-metric --metric euclidean
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        metric_name = None
        if '--metric' in args:
            idx = args.index('--metric')
            if idx + 1 < len(args):
                metric_name = args[idx + 1]

        if not metric_name:
            print("❌ Error: --metric is required")
            print("\nUsage: add-distance-metric --metric <metric_name>")
            print("\nExample: add-distance-metric --metric wasserstein")
            return

        try:
            cursor = self.db_conn.cursor()

            # Find distance function by name
            cursor.execute("""
                SELECT distance_function_id, function_name, display_name
                FROM ml_distance_functions_lut
                WHERE function_name = %s AND is_active = true
            """, (metric_name,))

            function = cursor.fetchone()

            if not function:
                print(f"❌ Distance function '{metric_name}' not found or not active")
                print("\nAvailable metrics:")
                cursor.execute("SELECT function_name FROM ml_distance_functions_lut WHERE is_active = true ORDER BY function_name")
                available = cursor.fetchall()
                for avail in available:
                    print(f"  - {avail[0]}")
                return

            func_id, func_name, display_name = function

            # Check if already configured
            cursor.execute("""
                SELECT COUNT(*) FROM ml_experiments_distance_measurements
                WHERE experiment_id = %s AND distance_function_id = %s
            """, (self.current_experiment, func_id))

            if cursor.fetchone()[0] > 0:
                print(f"⚠️  {func_name} ({display_name}) is already configured for experiment {self.current_experiment}")
                return

            # Add to experiment
            cursor.execute("""
                INSERT INTO ml_experiments_distance_measurements (experiment_id, distance_function_id)
                VALUES (%s, %s)
            """, (self.current_experiment, func_id))

            self.db_conn.commit()

            print(f"✅ Added {func_name} ({display_name}) to experiment {self.current_experiment}")

        except Exception as e:
            print(f"❌ Error adding distance metric: {e}")
            self.db_conn.rollback()

    def cmd_remove_distance_metric(self, args):
        """Remove distance metric from current experiment

        Usage: remove-distance-metric [options]

        Options:
            --metric <name>       Remove specific metric (e.g., wasserstein)
            --all-except <list>   Remove all except specified metrics (comma-separated)

        Examples:
            remove-distance-metric --metric wasserstein
            remove-distance-metric --all-except L1,L2,cosine,pearson
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        metric_name = None
        keep_only = None

        if '--metric' in args:
            idx = args.index('--metric')
            if idx + 1 < len(args):
                metric_name = args[idx + 1]

        if '--all-except' in args:
            idx = args.index('--all-except')
            if idx + 1 < len(args):
                keep_only = [m.strip() for m in args[idx + 1].split(',')]

        if not metric_name and not keep_only:
            print("❌ Error: --metric or --all-except is required")
            print("\nUsage:")
            print("  remove-distance-metric --metric <metric_name>")
            print("  remove-distance-metric --all-except L1,L2,cosine,pearson")
            return

        try:
            cursor = self.db_conn.cursor()

            if keep_only:
                # Remove all except specified metrics
                print(f"\n🔄 Removing all distance metrics except: {', '.join(keep_only)}")

                # Get IDs of metrics to keep
                placeholders = ','.join(['%s'] * len(keep_only))
                cursor.execute(f"""
                    SELECT distance_function_id, function_name
                    FROM ml_distance_functions_lut
                    WHERE function_name IN ({placeholders})
                """, keep_only)

                keep_ids = cursor.fetchall()

                if not keep_ids:
                    print(f"❌ None of the specified metrics found: {', '.join(keep_only)}")
                    return

                print(f"ℹ️  Keeping {len(keep_ids)} metrics:")
                for func_id, func_name in keep_ids:
                    print(f"   - {func_name}")

                # Delete all except these
                keep_id_list = [func_id for func_id, _ in keep_ids]
                placeholders = ','.join(['%s'] * len(keep_id_list))
                cursor.execute(f"""
                    DELETE FROM ml_experiments_distance_measurements
                    WHERE experiment_id = %s
                    AND distance_function_id NOT IN ({placeholders})
                    RETURNING distance_function_id
                """, [self.current_experiment] + keep_id_list)

                deleted = cursor.fetchall()
                self.db_conn.commit()

                print(f"\n✅ Removed {len(deleted)} distance metrics from experiment {self.current_experiment}")

            else:
                # Remove specific metric
                cursor.execute("""
                    SELECT distance_function_id, function_name, display_name
                    FROM ml_distance_functions_lut
                    WHERE function_name = %s
                """, (metric_name,))

                function = cursor.fetchone()

                if not function:
                    print(f"❌ Distance function '{metric_name}' not found")
                    return

                func_id, func_name, display_name = function

                cursor.execute("""
                    DELETE FROM ml_experiments_distance_measurements
                    WHERE experiment_id = %s AND distance_function_id = %s
                    RETURNING experiment_distance_id
                """, (self.current_experiment, func_id))

                deleted = cursor.fetchone()

                if not deleted:
                    print(f"⚠️  {func_name} ({display_name}) was not configured for experiment {self.current_experiment}")
                    return

                self.db_conn.commit()

                print(f"✅ Removed {func_name} ({display_name}) from experiment {self.current_experiment}")

        except Exception as e:
            print(f"❌ Error removing distance metric: {e}")
            self.db_conn.rollback()
            import traceback
            traceback.print_exc()

    def cmd_clean_distance_tables(self, args):
        """Clean unconfigured empty distance tables for current experiment

        Usage: clean-distance-tables [options]

        Options:
            --dry-run    Show what would be deleted without actually deleting
            --force      Skip confirmation prompt

        This command removes distance result tables that are:
        1. NOT configured in ml_experiments_distance_measurements for current experiment
        2. Have 0 rows (empty tables)

        Tables with data are NEVER deleted (safety check).

        Examples:
            clean-distance-tables                # Interactive mode
            clean-distance-tables --dry-run      # Show what would be deleted
            clean-distance-tables --force        # Skip confirmation
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        dry_run = '--dry-run' in args
        force = '--force' in args

        print(f"\n🔄 Scanning distance tables for experiment {self.current_experiment}...")

        try:
            cursor = self.db_conn.cursor()

            # Get configured distance metrics for this experiment
            cursor.execute("""
                SELECT df.result_table_prefix
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s
            """, (self.current_experiment,))

            configured_prefixes = [row[0] for row in cursor.fetchall()]
            configured_tables = [f"experiment_{self.current_experiment:03d}_{prefix}" for prefix in configured_prefixes]
            # PostgreSQL lowercases table names, so normalize for comparison
            configured_tables_lower = [t.lower() for t in configured_tables]

            print(f"📊 Found {len(configured_tables)} configured distance tables")

            # Get all distance tables for this experiment
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name LIKE %s
                ORDER BY table_name
            """, (f"experiment_{self.current_experiment:03d}_distance_%",))

            all_tables = [row[0] for row in cursor.fetchall()]

            print(f"📁 Found {len(all_tables)} total distance tables in database")

            # Find unconfigured tables (case-insensitive comparison)
            unconfigured_tables = [t for t in all_tables if t.lower() not in configured_tables_lower]

            if not unconfigured_tables:
                print("\n✅ No unconfigured distance tables found. All tables match configuration.")
                return

            print(f"\n⚠️  Found {len(unconfigured_tables)} unconfigured tables:")

            # Check row counts and categorize
            empty_tables = []
            non_empty_tables = []

            for table_name in unconfigured_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]

                if row_count == 0:
                    empty_tables.append(table_name)
                    print(f"   🗑️  {table_name}: 0 rows (can be deleted)")
                else:
                    non_empty_tables.append((table_name, row_count))
                    print(f"   ⚠️  {table_name}: {row_count:,} rows (WILL NOT DELETE - has data)")

            if not empty_tables:
                print("\n✅ No empty unconfigured tables to clean.")
                if non_empty_tables:
                    print(f"\nℹ️  {len(non_empty_tables)} tables have data and were not deleted.")
                return

            print(f"\n📋 Summary:")
            print(f"   Empty tables to delete: {len(empty_tables)}")
            print(f"   Tables with data (protected): {len(non_empty_tables)}")

            if dry_run:
                print("\n🔍 DRY RUN - No tables will be deleted")
                print("\nWould delete:")
                for table in empty_tables:
                    print(f"   - {table}")
                return

            # Confirmation prompt
            if not force:
                print(f"\n⚠️  About to delete {len(empty_tables)} empty unconfigured tables")
                response = input("Continue? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("❌ Cancelled")
                    return

            # Delete empty unconfigured tables
            deleted_count = 0
            for table_name in empty_tables:
                try:
                    cursor.execute(f"DROP TABLE {table_name} CASCADE")
                    self.db_conn.commit()
                    print(f"✅ Deleted: {table_name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Error deleting {table_name}: {e}")
                    self.db_conn.rollback()

            print(f"\n✅ Cleaned {deleted_count} empty unconfigured distance tables")

            if non_empty_tables:
                print(f"\nℹ️  {len(non_empty_tables)} tables with data were preserved")

        except Exception as e:
            print(f"❌ Error cleaning distance tables: {e}")
            import traceback
            traceback.print_exc()

    def cmd_get_experiment_data_path(self, args):
        """Get the data path for an experiment

        Usage: get-experiment-data-path [experiment_id]

        Shows the configured data paths for segment files and feature files.
        If no experiment_id is provided, uses current experiment.

        Examples:
            get-experiment-data-path         # Current experiment
            get-experiment-data-path 41      # Experiment 41
        """
        # Determine experiment_id
        if args and args[0].isdigit():
            experiment_id = int(args[0])
        elif self.current_experiment:
            experiment_id = self.current_experiment
        else:
            print("❌ No experiment specified. Use 'set experiment <id>' first or provide experiment_id as argument.")
            return

        from pathlib import Path

        # Check database for custom paths
        custom_segment_path = None
        custom_feature_path = None

        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    SELECT segment_data_base_path, feature_data_base_path
                    FROM ml_experiments
                    WHERE experiment_id = %s
                """, (experiment_id,))
                result = cursor.fetchone()
                if result:
                    custom_segment_path = result[0]
                    custom_feature_path = result[1]
                cursor.close()
            except Exception as e:
                print(f"⚠️  Warning: Could not read custom paths from database: {e}")

        # Use custom paths if configured, otherwise use defaults
        if custom_segment_path and custom_feature_path:
            segment_path = Path(custom_segment_path)
            feature_path = Path(custom_feature_path)
            base_path = segment_path.parent
            using_custom = True
        else:
            base_path = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}')
            segment_path = base_path / 'segment_files'
            feature_path = base_path / 'feature_files'
            using_custom = False

        print(f"\n📁 Data paths for experiment {experiment_id}:")
        if using_custom:
            print(f"   ⚙️  Using CUSTOM paths from database")
        else:
            print(f"   ⚙️  Using DEFAULT path pattern")
        print(f"   Base:     {base_path}")
        print(f"   Segments: {segment_path}")
        print(f"   Features: {feature_path}")
        print()
        print(f"Status:")
        print(f"   Base exists:     {'✅' if base_path.exists() else '❌'}")
        print(f"   Segments exist:  {'✅' if segment_path.exists() else '❌'}")
        print(f"   Features exist:  {'✅' if feature_path.exists() else '❌'}")

        # Count files if directories exist
        if segment_path.exists():
            segment_count = sum(1 for _ in segment_path.glob('**/*.npy'))
            print(f"   Segment files:   {segment_count:,}")

        if feature_path.exists():
            feature_count = sum(1 for _ in feature_path.glob('**/*.npy'))
            print(f"   Feature files:   {feature_count:,}")

    def cmd_set_experiment_data_path(self, args):
        """Set/configure the data path for an experiment

        Usage: set-experiment-data-path <path> [experiment_id]
                set-experiment-data-path --reset [experiment_id]

        This command sets a custom base data path for an experiment.
        By default, experiments use: /Volumes/ArcData/V3_database/experiment{NNN}/

        The path is stored in the database and used by generators.
        Use --reset to clear custom path and use default.

        Examples:
            set-experiment-data-path /custom/path 41
            set-experiment-data-path --reset 41
        """
        if not args:
            print("❌ Usage: set-experiment-data-path <path> [experiment_id]")
            print("   Or:    set-experiment-data-path --reset [experiment_id]")
            return

        # Check for --reset flag
        reset_mode = args[0] == '--reset'

        if reset_mode:
            custom_path = None
            args = args[1:]  # Remove --reset from args
        else:
            custom_path = args[0]
            args = args[1:]

        # Get experiment ID
        if args and args[0].isdigit():
            experiment_id = int(args[0])
        elif self.current_experiment:
            experiment_id = self.current_experiment
        else:
            print("❌ No experiment specified. Use 'set experiment <id>' first or provide experiment_id as argument.")
            return

        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        try:
            cursor = self.db_conn.cursor()

            if reset_mode:
                # Reset to default (NULL = use default path pattern)
                cursor.execute("""
                    UPDATE ml_experiments
                    SET segment_data_base_path = NULL,
                        feature_data_base_path = NULL
                    WHERE experiment_id = %s
                """, (experiment_id,))
                self.db_conn.commit()

                print(f"\n✅ Reset experiment {experiment_id} to use default data paths")
                print(f"   Default pattern: /Volumes/ArcData/V3_database/experiment{experiment_id:03d}/")
            else:
                # Validate path format
                from pathlib import Path
                path_obj = Path(custom_path)

                if not path_obj.is_absolute():
                    print(f"❌ Path must be absolute: {custom_path}")
                    return

                # Set custom paths
                segment_path = f"{custom_path}/experiment{experiment_id:03d}/segment_files"
                feature_path = f"{custom_path}/experiment{experiment_id:03d}/feature_files"

                cursor.execute("""
                    UPDATE ml_experiments
                    SET segment_data_base_path = %s,
                        feature_data_base_path = %s
                    WHERE experiment_id = %s
                """, (segment_path, feature_path, experiment_id))
                self.db_conn.commit()

                print(f"\n✅ Updated experiment {experiment_id} data paths:")
                print(f"   Segment path: {segment_path}")
                print(f"   Feature path: {feature_path}")
                print()
                print(f"📝 Note: Paths are stored in database and will be used by generators.")
                print(f"   Make sure the parent directory exists and is writable.")

            cursor.close()

        except Exception as e:
            print(f"❌ Error updating data path: {e}")
            if self.db_conn:
                self.db_conn.rollback()

    def cmd_clean_segment_files(self, args):
        """Delete segment files for an experiment

        Usage: clean-segment-files [options] [experiment_id]

        Options:
            --dry-run    Show what would be deleted without actually deleting
            --force      Skip confirmation prompt

        This command deletes all segment files AND directories for an experiment.
        USE WITH CAUTION - This cannot be undone!

        You will be shown what will be deleted and must type 'DELETE' to confirm.

        Examples:
            clean-segment-files                    # Current experiment, requires 'DELETE'
            clean-segment-files --dry-run          # Show what would be deleted
            clean-segment-files --force            # Skip confirmation
            clean-segment-files 41                 # Delete experiment 41 segments
        """
        # Parse arguments
        dry_run = '--dry-run' in args
        force = '--force' in args

        # Remove flags from args to find experiment_id
        args_clean = [a for a in args if not a.startswith('--')]

        # Determine experiment_id
        if args_clean and args_clean[0].isdigit():
            experiment_id = int(args_clean[0])
        elif self.current_experiment:
            experiment_id = self.current_experiment
        else:
            print("❌ No experiment specified. Use 'set experiment <id>' first or provide experiment_id as argument.")
            return

        from pathlib import Path

        # Read custom paths from database if configured
        custom_segment_path = None
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    SELECT segment_data_base_path
                    FROM ml_experiments
                    WHERE experiment_id = %s
                """, (experiment_id,))
                result = cursor.fetchone()
                if result:
                    custom_segment_path = result[0]
                cursor.close()
            except Exception as e:
                self.db_conn.rollback()
                print(f"⚠️  Warning: Could not read custom path from database: {e}")

        # Use custom path if configured, otherwise use default
        if custom_segment_path:
            segment_path = Path(custom_segment_path)
        else:
            segment_path = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}/segment_files')

        if not segment_path.exists():
            print(f"ℹ️  No segment files directory found for experiment {experiment_id}")
            print(f"   Path: {segment_path}")
            return

        # Count ALL files (not just .npy) to ensure complete cleanup
        all_files = [f for f in segment_path.rglob('*') if f.is_file()]
        file_count = len(all_files)

        # Also count just .npy files for reporting
        npy_files = [f for f in all_files if f.suffix == '.npy']
        npy_count = len(npy_files)
        other_files = file_count - npy_count

        # Count directories (excluding the base path itself)
        all_dirs = [d for d in segment_path.rglob('*') if d.is_dir()]
        dir_count = len(all_dirs)

        # Check if there's anything to clean
        if file_count == 0 and dir_count == 0:
            print(f"✅ Segment folder already empty for experiment {experiment_id}")
            print(f"   Path: {segment_path}")
            return

        # Calculate total size (only if files exist)
        if file_count > 0:
            total_size = sum(f.stat().st_size for f in all_files)
            size_mb = total_size / (1024 * 1024)
            size_gb = size_mb / 1024
        else:
            total_size = 0
            size_mb = 0
            size_gb = 0

        print(f"\n📁 Segment files for experiment {experiment_id}:")
        print(f"   Path: {segment_path}")
        print(f"   Files: {file_count:,}")
        if npy_count > 0:
            print(f"   - Segment files (.npy): {npy_count:,}")
        if other_files > 0:
            print(f"   - Other files (.DS_Store, etc.): {other_files:,}")
        if file_count > 0:
            print(f"   Size: {size_gb:.2f} GB ({size_mb:.2f} MB)")
        print(f"   Directories: {dir_count:,}")

        if dry_run:
            print("\n🔍 DRY RUN - No files or directories will be deleted")
            if file_count > 0:
                print("\nSample files that would be deleted:")
                for f in all_files[:10]:
                    print(f"   - {f.name}")
                if len(all_files) > 10:
                    print(f"   ... and {len(all_files) - 10:,} more files")
            if dir_count > 0:
                print(f"\nWould remove {dir_count:,} directories")
            return

        # Require confirmation for destructive operations (unless --force)
        print(f"\n⚠️  WARNING: This will permanently delete:")
        if file_count > 0:
            print(f"   - {file_count:,} files ({size_gb:.2f} GB)")
        if dir_count > 0:
            print(f"   - {dir_count:,} directories")
        print(f"⚠️  This action CANNOT be undone!")

        if not force:
            response = input("\nType 'DELETE' to confirm: ").strip()
            if response != 'DELETE':
                print("❌ Cancelled")
                return
        else:
            print("\n⚠️  --force flag set: Skipping confirmation")

        # Delete ALL files (including .DS_Store, etc.)
        if file_count > 0:
            print(f"\n🗑️  Deleting all files...")
            deleted_count = 0
            failed_count = 0

            for file_path in all_files:
                try:
                    file_path.unlink()
                    deleted_count += 1
                    if deleted_count % 1000 == 0:
                        print(f"   Deleted {deleted_count:,} / {file_count:,} files...")
                except Exception as e:
                    print(f"❌ Error deleting {file_path.name}: {e}")
                    failed_count += 1

            print(f"✅ Deleted {deleted_count:,} files")
            if npy_count > 0:
                print(f"   - Segment files (.npy): {npy_count:,}")
            if other_files > 0:
                print(f"   - Other files: {other_files:,}")
            if failed_count > 0:
                print(f"⚠️  Failed to delete {failed_count} files")

        # Also delete progress checkpoint
        progress_file = segment_path / 'generation_progress.json'
        if progress_file.exists():
            try:
                progress_file.unlink()
                print(f"✅ Deleted progress checkpoint")
            except Exception as e:
                print(f"⚠️  Could not delete progress checkpoint: {e}")

        # Delete all empty directories to completely clean the folder structure
        print(f"\n🗑️  Removing empty directories...")
        import shutil
        dirs_removed = 0

        # Get all subdirectories, sorted by depth (deepest first)
        all_dirs = sorted([d for d in segment_path.rglob('*') if d.is_dir()],
                         key=lambda p: len(p.parts), reverse=True)

        for directory in all_dirs:
            try:
                # Only remove if empty
                if not any(directory.iterdir()):
                    directory.rmdir()
                    dirs_removed += 1
            except Exception as e:
                # Ignore errors (directory might not be empty or might be in use)
                pass

        if dirs_removed > 0:
            print(f"✅ Removed {dirs_removed} empty directories")

        # Verify folder is completely empty
        remaining_items = list(segment_path.iterdir())
        if remaining_items:
            print(f"\n⚠️  Warning: {len(remaining_items)} items remaining in {segment_path}:")
            for item in remaining_items[:5]:
                print(f"   - {item.name}")
            if len(remaining_items) > 5:
                print(f"   ... and {len(remaining_items) - 5} more items")
        else:
            print(f"\n✅ Segment folder completely empty: {segment_path}")

    def cmd_clean_feature_files(self, args):
        """Delete feature files for an experiment

        Usage: clean-feature-files [options] [experiment_id]

        Options:
            --dry-run              Show what would be deleted without actually deleting
            --force                Skip confirmation prompt
            --files-and-tables     Delete files AND truncate table (default)
            --files-only           Delete files only, leave table
            --tables-only          Truncate table only, leave files

        This command deletes all feature files for an experiment.
        USE WITH CAUTION - This cannot be undone!

        Examples:
            clean-feature-files                    # Current experiment, files and tables
            clean-feature-files --dry-run          # Show what would be deleted
            clean-feature-files --force 41         # Delete experiment 41 features
            clean-feature-files --files-only       # Delete files, keep table
            clean-feature-files --tables-only      # Truncate table, keep files
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        dry_run = '--dry-run' in args
        force = '--force' in args
        files_only = '--files-only' in args
        tables_only = '--tables-only' in args
        files_and_tables = '--files-and-tables' in args

        # Default behavior: clean both files and tables
        if not files_only and not tables_only and not files_and_tables:
            files_and_tables = True

        # Validate mutually exclusive options
        mode_count = sum([files_only, tables_only, files_and_tables])
        if mode_count > 1:
            print("❌ Error: --files-only, --tables-only, and --files-and-tables are mutually exclusive")
            return

        # Determine what to clean
        clean_files = files_only or files_and_tables
        clean_tables = tables_only or files_and_tables

        # Remove flags from args to find experiment_id
        args_clean = [a for a in args if not a.startswith('--')]

        # Determine experiment_id
        if args_clean and args_clean[0].isdigit():
            experiment_id = int(args_clean[0])
        elif self.current_experiment:
            experiment_id = self.current_experiment
        else:
            print("❌ No experiment specified. Use 'set experiment <id>' first or provide experiment_id as argument.")
            return

        from pathlib import Path

        # Read custom paths from database if configured
        custom_feature_path = None
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT feature_data_base_path
                FROM ml_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))
            result = cursor.fetchone()
            if result:
                custom_feature_path = result[0]
            cursor.close()
        except Exception as e:
            self.db_conn.rollback()
            print(f"⚠️  Warning: Could not read custom path from database: {e}")

        # Use custom path if configured, otherwise use default
        if custom_feature_path:
            feature_path = Path(custom_feature_path)
        else:
            feature_path = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}/feature_files')

        # Count files
        file_count = 0
        total_size = 0
        if feature_path.exists():
            feature_files = list(feature_path.glob('**/*.npy'))
            file_count = len(feature_files)
            if file_count > 0:
                total_size = sum(f.stat().st_size for f in feature_files)

        # Check database records
        cursor = self.db_conn.cursor()
        table_name = f"experiment_{experiment_id:03d}_feature_fileset"

        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            db_count = cursor.fetchone()[0]
        except Exception:
            db_count = 0
            print(f"ℹ️  No feature table found: {table_name}")

        size_mb = total_size / (1024 * 1024)
        size_gb = size_mb / 1024

        print(f"\n📁 Feature files for experiment {experiment_id}:")
        print(f"   Path: {feature_path}")
        print(f"   Files on disk: {file_count:,}")
        print(f"   Database records: {db_count:,}")
        if file_count > 0:
            print(f"   Size: {size_gb:.2f} GB ({size_mb:.2f} MB)")

        # Check if anything to clean based on mode
        if clean_files and file_count == 0 and clean_tables and db_count == 0:
            print(f"\nℹ️  No feature files or database records found")
            return
        if clean_files and not clean_tables and file_count == 0:
            print(f"\nℹ️  No feature files found")
            return
        if clean_tables and not clean_files and db_count == 0:
            print(f"\nℹ️  No database records found")
            return

        if dry_run:
            print("\n🔍 DRY RUN - No files or data will be deleted")
            if clean_files and file_count > 0:
                print("\nSample files that would be deleted:")
                for f in feature_files[:10]:
                    print(f"   - {f.name}")
                if len(feature_files) > 10:
                    print(f"   ... and {len(feature_files) - 10:,} more files")
            if clean_tables and db_count > 0:
                print(f"\nWould truncate table: {table_name} ({db_count:,} records)")
            return

        # Confirmation prompt
        if not force:
            print(f"\n⚠️  WARNING: This will permanently delete:")
            if clean_files and file_count > 0:
                print(f"   - {file_count:,} feature files ({size_gb:.2f} GB)")
            if clean_tables and db_count > 0:
                print(f"   - {db_count:,} database records from {table_name}")
            print(f"⚠️  This action CANNOT be undone!")
            response = input("\nType 'DELETE' to confirm: ").strip()
            if response != 'DELETE':
                print("❌ Cancelled")
                return

        # Delete files
        if clean_files and file_count > 0:
            print(f"\n🗑️  Deleting feature files directory...")
            import shutil

            try:
                # Delete the ENTIRE feature_files directory
                shutil.rmtree(feature_path)
                print(f"✅ Deleted entire directory: {feature_path}")
                print(f"   Removed {file_count:,} files ({size_gb:.2f} GB)")

                # Recreate empty directory
                feature_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Recreated empty directory: {feature_path}")

            except Exception as e:
                print(f"❌ Error deleting directory: {e}")
                print(f"   Falling back to file-by-file deletion...")

                # Fallback: delete files individually
                deleted_count = 0
                failed_count = 0

                for feature_file in feature_files:
                    try:
                        feature_file.unlink()
                        deleted_count += 1
                        if deleted_count % 1000 == 0:
                            print(f"   Deleted {deleted_count:,} / {file_count:,} files...")
                    except Exception as e:
                        print(f"❌ Error deleting {feature_file.name}: {e}")
                        failed_count += 1

                print(f"✅ Deleted {deleted_count:,} feature files")
                if failed_count > 0:
                    print(f"⚠️  Failed to delete {failed_count} files")

        # Truncate database table
        if clean_tables and db_count > 0:
            try:
                cursor.execute(f"TRUNCATE TABLE {table_name}")
                self.db_conn.commit()
                print(f"✅ Truncated table {table_name} ({db_count:,} records)")
            except Exception as e:
                print(f"❌ Error truncating table: {e}")
                self.db_conn.rollback()

    def cmd_sql(self, args):
        """Execute SQL query with appropriate confirmation

        Usage: sql <query>

        Confirmation rules:
        - SELECT: No confirmation needed
        - DROP/TRUNCATE: Must type 'DROP' or 'TRUNCATE' to confirm
        - INSERT/UPDATE: Requires (N/y) confirmation
        - Other queries: Requires (N/y) confirmation

        Examples:
            sql SELECT * FROM ml_experiments LIMIT 5
            sql DROP TABLE experiment_041_feature_fileset CASCADE
            sql UPDATE ml_experiments SET is_active = true WHERE experiment_id = 41
            sql INSERT INTO ml_experiments (experiment_name) VALUES ('test')
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        if not args:
            print("❌ No SQL query provided.")
            print("Usage: sql <query>")
            return

        # Join all args into a single query
        query = ' '.join(args).strip()

        # Remove trailing semicolon if present
        if query.endswith(';'):
            query = query[:-1]

        # Determine query type by looking at first word
        query_upper = query.upper().strip()
        first_word = query_upper.split()[0] if query_upper else ''

        # Confirmation logic based on query type
        if first_word == 'SELECT':
            # No confirmation needed for SELECT
            pass
        elif first_word in ('DROP', 'TRUNCATE'):
            # Require typing the word for destructive operations
            print(f"\n⚠️  WARNING: You are about to execute a {first_word} query:")
            print(f"   {query}")
            print(f"\n⚠️  This action CANNOT be undone!")
            response = input(f"\nType '{first_word}' to confirm: ").strip()
            if response != first_word:
                print("❌ Cancelled")
                return
        elif first_word in ('INSERT', 'UPDATE', 'DELETE'):
            # Require (N/y) confirmation for data modification
            print(f"\n⚠️  You are about to execute a {first_word} query:")
            print(f"   {query}")
            response = input("\nContinue? (N/y): ").strip().lower()
            if response not in ('y', 'yes'):
                print("❌ Cancelled")
                return
        else:
            # Other queries require confirmation
            print(f"\n⚠️  You are about to execute:")
            print(f"   {query}")
            response = input("\nContinue? (N/y): ").strip().lower()
            if response not in ('y', 'yes'):
                print("❌ Cancelled")
                return

        # Execute query
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(query)

            # If it's a SELECT query, fetch and display results
            if first_word == 'SELECT':
                results = cursor.fetchall()
                if results:
                    # Get column names
                    col_names = [desc[0] for desc in cursor.description]

                    # Print header
                    print(f"\n📊 Results ({len(results)} rows):")
                    print("─" * 80)
                    print(" | ".join(col_names))
                    print("─" * 80)

                    # Print rows
                    for row in results[:100]:  # Limit to first 100 rows
                        print(" | ".join(str(val) for val in row))

                    if len(results) > 100:
                        print(f"\n... and {len(results) - 100} more rows")
                    print("─" * 80)
                else:
                    print("\n✅ Query returned 0 rows")
            else:
                # For non-SELECT queries, commit and show affected rows
                self.db_conn.commit()
                if cursor.rowcount >= 0:
                    print(f"\n✅ Query executed successfully. Rows affected: {cursor.rowcount}")
                else:
                    print(f"\n✅ Query executed successfully")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n❌ Error executing query: {e}")
        finally:
            cursor.close()

    def cmd_clean_distance_work_files(self, args):
        """Delete mpcctl distance calculation work files

        Usage: clean-distance-work-files [options] [experiment_id]

        Options:
            --dry-run    Show what would be deleted without actually deleting
            --force      Skip confirmation prompt

        This command deletes mpcctl state and work files:
            - .mpcctl_state.json
            - .mpcctl/ directory
            - .processed/ directory
            - distance_insert/state.json
            - distance_insert/ directory (if empty)

        These files are created during distance calculation and can be
        safely deleted after distances are computed and inserted into the database.

        Examples:
            clean-distance-work-files                    # Current experiment, interactive
            clean-distance-work-files --dry-run          # Show what would be deleted
            clean-distance-work-files --force 41         # Delete experiment 41 work files
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        dry_run = '--dry-run' in args
        force = '--force' in args

        # Remove flags from args to find experiment_id
        args_clean = [a for a in args if not a.startswith('--')]

        # Determine experiment_id
        if args_clean and args_clean[0].isdigit():
            experiment_id = int(args_clean[0])
        elif self.current_experiment:
            experiment_id = self.current_experiment
        else:
            print("❌ No experiment specified. Use 'set experiment <id>' first or provide experiment_id as argument.")
            return

        from pathlib import Path
        import shutil

        # Read custom paths from database if configured
        # MPCCTL files are in the experiment root (parent of feature_files)
        custom_feature_path = None
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT feature_data_base_path
                FROM ml_experiments
                WHERE experiment_id = %s
            """, (experiment_id,))
            result = cursor.fetchone()
            if result:
                custom_feature_path = result[0]
            cursor.close()
        except Exception as e:
            print(f"⚠️  Warning: Could not read custom path from database: {e}")

        # Use custom path if configured, otherwise use default
        # MPCCTL work files are in the parent directory of feature_files
        if custom_feature_path:
            experiment_root = Path(custom_feature_path).parent
        else:
            experiment_root = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}')

        if not experiment_root.exists():
            print(f"❌ Experiment root path does not exist: {experiment_root}")
            return

        # Check for mpcctl work files (in experiment root)
        state_file = experiment_root / '.mpcctl_state.json'
        mpcctl_dir = experiment_root / '.mpcctl'
        processed_dir = experiment_root / '.processed'
        distance_insert_dir = experiment_root / 'distance_insert'
        distance_insert_state = distance_insert_dir / 'state.json'

        items_to_delete = []
        total_size = 0

        if state_file.exists():
            size = state_file.stat().st_size
            items_to_delete.append(('.mpcctl_state.json', state_file, size, 'file'))
            total_size += size

        if mpcctl_dir.exists():
            dir_size = sum(f.stat().st_size for f in mpcctl_dir.glob('**/*') if f.is_file())
            file_count = len(list(mpcctl_dir.glob('**/*')))
            items_to_delete.append(('.mpcctl/', mpcctl_dir, dir_size, f'directory ({file_count} files)'))
            total_size += dir_size

        if processed_dir.exists():
            dir_size = sum(f.stat().st_size for f in processed_dir.glob('**/*') if f.is_file())
            file_count = len(list(processed_dir.glob('**/*')))
            items_to_delete.append(('.processed/', processed_dir, dir_size, f'directory ({file_count} files)'))
            total_size += dir_size

        if distance_insert_dir.exists() and distance_insert_state.exists():
            size = distance_insert_state.stat().st_size
            items_to_delete.append(('distance_insert/state.json', distance_insert_state, size, 'file'))
            total_size += size

        if distance_insert_dir.exists() and distance_insert_dir.is_dir():
            # Check if directory would be empty after state.json deletion
            remaining_files = [f for f in distance_insert_dir.glob('*') if f != distance_insert_state]
            if not remaining_files:
                # Directory will be empty, add it to deletion list
                items_to_delete.append(('distance_insert/', distance_insert_dir, 0, 'directory (empty)'))


        # Show what will be deleted
        if not items_to_delete:
            print(f"\n✅ No mpcctl work files found in {experiment_root}")
            return

        print(f"\n📂 Location: {experiment_root}")
        print(f"\n🗑️  The following mpcctl work files will be deleted:")
        print(f"\n{'Name':<30} {'Type':<25} {'Size':<15}")
        print("-" * 72)

        for name, path, size, item_type in items_to_delete:
            size_mb = size / (1024 * 1024)
            print(f"{name:<30} {item_type:<25} {size_mb:>10.2f} MB")

        print("-" * 72)
        print(f"{'Total:':<30} {len(items_to_delete)} items {total_size / (1024 * 1024):>10.2f} MB")

        if dry_run:
            print("\n✅ Dry run complete - no files were deleted")
            return

        # Confirmation
        if not force:
            print(f"\n⚠️  This will permanently delete {len(items_to_delete)} items from experiment {experiment_id}")
            response = input("Type 'DELETE' to confirm: ").strip()
            if response != 'DELETE':
                print("❌ Cancelled")
                return

        # Delete items
        print(f"\n🗑️  Deleting mpcctl work files...")
        deleted_count = 0
        failed_count = 0

        for name, path, size, item_type in items_to_delete:
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                deleted_count += 1
                print(f"   ✅ Deleted {name}")
            except Exception as e:
                print(f"   ❌ Error deleting {name}: {e}")
                failed_count += 1

        print(f"\n✅ Deleted {deleted_count} items")
        if failed_count > 0:
            print(f"⚠️  Failed to delete {failed_count} items")

    def cmd_clean_distance_insert(self, args):
        """Truncate distance tables for current experiment

        Usage: clean-distance-insert [options]

        Options:
            --dry-run    Show what would be truncated without actually truncating
            --force      Skip confirmation prompt

        This command TRUNCATES all configured distance result tables.
        ALL DISTANCE DATA WILL BE DELETED - Cannot be undone!

        Examples:
            clean-distance-insert                # Interactive mode, requires confirmation
            clean-distance-insert --dry-run      # Show what would be truncated
            clean-distance-insert --force        # Skip confirmation
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("❌ No experiment selected. Use 'set experiment <id>' first.")
            return

        dry_run = '--dry-run' in args
        force = '--force' in args

        try:
            cursor = self.db_conn.cursor()

            # Get configured distance functions for this experiment
            cursor.execute("""
                SELECT df.function_name, df.result_table_prefix
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s
            """, (self.current_experiment,))

            distance_funcs = cursor.fetchall()

            if not distance_funcs:
                print(f"ℹ️  No distance functions configured for experiment {self.current_experiment}")
                return

            # Build table names
            tables = [f"experiment_{self.current_experiment:03d}_{row[1]}" for row in distance_funcs]

            print(f"\n{'='*80}")
            print(f"🗑️  TRUNCATE DISTANCE TABLES - Experiment {self.current_experiment}")
            print(f"{'='*80}\n")

            # Get record counts
            total_records = 0
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                total_records += count
                print(f"   {table}: {count:,} records")

            print(f"\n📊 Total records to delete: {total_records:,}")
            print(f"\n⚠️  WARNING: This will DELETE ALL DISTANCE DATA for experiment {self.current_experiment}")
            print(f"   This operation CANNOT be undone!")

            if dry_run:
                print(f"\n🔍 DRY RUN: No tables would be truncated")
                cursor.close()
                return

            if not force:
                print(f"\n❓ Type 'TRUNCATE' to confirm deletion: ", end='')
                confirmation = input().strip()
                if confirmation != 'TRUNCATE':
                    print("❌ Operation cancelled")
                    cursor.close()
                    return

            # Truncate tables
            print(f"\n🗑️  Truncating tables...")
            for table in tables:
                cursor.execute(f"TRUNCATE TABLE {table}")
                print(f"   ✅ Truncated {table}")

            self.db_conn.commit()
            cursor.close()

            print(f"\n✅ Successfully truncated {len(tables)} distance tables")

        except Exception as e:
            self.db_conn.rollback()
            print(f"❌ Error truncating distance tables: {e}")

    def cmd_clean_files(self, args):
        """Truncate file training data table for current experiment

        Usage: clean-files [options]

        Options:
            --dry-run    Show what would be deleted without actually deleting
            --force      Skip confirmation prompt

        This command truncates experiment_NNN_file_training_data table
        for the current experiment. This removes the experiment's file selection.

        Examples:
            clean-files                  # Interactive mode, requires confirmation
            clean-files --dry-run        # Show what would be deleted
            clean-files --force          # Skip confirmation
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("❌ No experiment selected. Use 'set experiment <id>' first.")
            return

        dry_run = '--dry-run' in args
        force = '--force' in args

        try:
            cursor = self.db_conn.cursor()

            # Check if experiment-specific table exists
            table_name = f"experiment_{self.current_experiment:03d}_file_training_data"
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                )
            """, (table_name,))
            table_exists = cursor.fetchone()[0]

            if not table_exists:
                print(f"ℹ️  Table {table_name} does not exist")
                print(f"   Skipping file training data cleanup")
                cursor.close()
                return

            # Get file count
            cursor.execute(f"""
                SELECT COUNT(*) FROM {table_name}
            """)
            file_count = cursor.fetchone()[0]

            if file_count == 0:
                print(f"ℹ️  No file training data for experiment {self.current_experiment}")
                cursor.close()
                return

            print(f"\n{'='*80}")
            print(f"🗑️  TRUNCATE FILE TRAINING DATA - Experiment {self.current_experiment}")
            print(f"{'='*80}\n")
            print(f"   {table_name}: {file_count:,} rows")
            print(f"\n⚠️  WARNING: This will TRUNCATE ALL FILE TRAINING DATA for experiment {self.current_experiment}")

            if dry_run:
                print(f"\n🔍 DRY RUN: No rows would be deleted")
                cursor.close()
                return

            if not force:
                print(f"\n❓ Type 'TRUNCATE' to confirm: ", end='')
                confirmation = input().strip()
                if confirmation != 'TRUNCATE':
                    print("❌ Operation cancelled")
                    cursor.close()
                    return

            # Truncate table
            print(f"\n🗑️  Truncating table...")
            cursor.execute(f"TRUNCATE TABLE {table_name}")

            self.db_conn.commit()
            cursor.close()

            print(f"   ✅ Truncated {table_name} ({file_count:,} rows)")
            print(f"\n✅ Successfully cleaned file training data")

        except Exception as e:
            self.db_conn.rollback()
            print(f"❌ Error cleaning file training data: {e}")

    def cmd_clean_segments(self, args):
        """Delete segment files and truncate segment tables for current experiment

        Usage: clean-segments [options]

        Options:
            --dry-run              Show what would be deleted without actually deleting
            --force                Skip confirmation prompt
            --files-only           Delete files only, leave tables
            --tables-only          Truncate tables only, leave files
            --files-and-tables     Delete files AND truncate tables (default)

        This command:
        1. Deletes all segment files for the experiment
        2. Truncates experiment_{exp}_segment_pairs table
        3. Deletes data_segments rows for this experiment

        USE WITH CAUTION - This cannot be undone!

        Examples:
            clean-segments                       # Interactive mode, files and tables
            clean-segments --dry-run             # Show what would be deleted
            clean-segments --files-only          # Delete files, keep tables
            clean-segments --tables-only         # Truncate tables, keep files
        """
        if not self.current_experiment:
            print("❌ No experiment selected. Use 'set experiment <id>' first.")
            return

        # Parse options
        dry_run = '--dry-run' in args
        force = '--force' in args
        files_only = '--files-only' in args
        tables_only = '--tables-only' in args
        files_and_tables = '--files-and-tables' in args

        # Default: both files and tables
        if not files_only and not tables_only and not files_and_tables:
            files_and_tables = True

        clean_files = files_only or files_and_tables
        clean_tables = tables_only or files_and_tables

        # 1. Clean segment files (using existing clean-segment-files logic)
        if clean_files:
            # Call existing clean_segment_files implementation
            self.cmd_clean_segment_files(args)

        # 2. Clean segment tables
        if clean_tables:
            if not self.db_conn:
                print("❌ Not connected to database. Use 'connect' first.")
                return

            try:
                cursor = self.db_conn.cursor()

                # Check which experiment-specific tables exist
                segment_training_table = f"experiment_{self.current_experiment:03d}_segment_training_data"
                segment_pairs_table = f"experiment_{self.current_experiment:03d}_segment_pairs"

                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    )
                """, (segment_training_table,))
                segment_training_exists = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    )
                """, (segment_pairs_table,))
                segment_pairs_exists = cursor.fetchone()[0]

                # Get counts
                segment_training_count = 0
                if segment_training_exists:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {segment_training_table}
                    """)
                    segment_training_count = cursor.fetchone()[0]

                pairs_count = 0
                if segment_pairs_exists:
                    cursor.execute(f"""
                        SELECT COUNT(*) FROM {segment_pairs_table}
                    """)
                    pairs_count = cursor.fetchone()[0]

                print(f"\n{'='*80}")
                print(f"🗑️  TRUNCATE SEGMENT TABLES - Experiment {self.current_experiment}")
                print(f"{'='*80}\n")

                if segment_training_exists:
                    print(f"   {segment_training_table}: {segment_training_count:,} rows")
                else:
                    print(f"   {segment_training_table}: table not found (skipping)")

                if segment_pairs_exists:
                    print(f"   {segment_pairs_table}: {pairs_count:,} rows")
                else:
                    print(f"   {segment_pairs_table}: table not found (skipping)")

                if not segment_training_exists and not segment_pairs_exists:
                    print(f"\nℹ️  No segment tables found to clean")
                    cursor.close()
                    return

                if dry_run:
                    print(f"\n🔍 DRY RUN: No tables would be truncated")
                    cursor.close()
                    return

                if not force:
                    print(f"\n❓ Type 'TRUNCATE' to confirm deletion: ", end='')
                    confirmation = input().strip()
                    if confirmation != 'TRUNCATE':
                        print("❌ Operation cancelled")
                        cursor.close()
                        return

                # Truncate tables
                print(f"\n🗑️  Truncating tables...")

                if segment_training_exists:
                    cursor.execute(f"TRUNCATE TABLE {segment_training_table}")
                    print(f"   ✅ Truncated {segment_training_table}")

                if segment_pairs_exists:
                    cursor.execute(f"TRUNCATE TABLE {segment_pairs_table}")
                    print(f"   ✅ Truncated {segment_pairs_table}")

                self.db_conn.commit()
                cursor.close()

                print(f"\n✅ Successfully cleaned segment tables")

            except Exception as e:
                self.db_conn.rollback()
                print(f"❌ Error cleaning segment tables: {e}")

    def cmd_clean_experiment(self, args):
        """Complete experiment cleanup - removes ALL data and files

        Usage: clean-experiment [options]

        Options:
            --dry-run    Show what would be deleted without actually deleting
            --force      Skip ALL confirmation prompts (DANGEROUS!)

        This command executes a complete experiment cleanup in the following order:
            1. clean-distance-insert      (truncate distance tables)
            2. clean-distance-calculate   (remove distance work files)
            3. clean-features             (remove feature files and truncate table)
            4. clean-segments             (remove segment files and truncate tables)
            5. clean-files                (truncate training files table)

        ⚠️  EXTREME CAUTION: This will DELETE ALL EXPERIMENT DATA!

        After this command, the experiment will be in a clean state as if
        it had just been created. You can then re-run the entire pipeline.

        Examples:
            clean-experiment                 # Interactive mode, confirm each step
            clean-experiment --dry-run       # Show what would be deleted
            clean-experiment --force         # Skip all confirmations (DANGEROUS!)
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("❌ No experiment selected. Use 'set experiment <id>' first.")
            return

        dry_run = '--dry-run' in args
        force = '--force' in args

        print(f"\n{'='*80}")
        print(f"🗑️  COMPLETE EXPERIMENT CLEANUP - Experiment {self.current_experiment}")
        print(f"{'='*80}\n")
        print(f"This will execute the following cleanup commands:")
        print(f"   1. clean-distance-insert      (truncate distance tables)")
        print(f"   2. clean-distance-calculate   (remove distance work files)")
        print(f"   3. clean-features             (remove feature files and truncate table)")
        print(f"   4. clean-segments             (remove segment files and truncate tables)")
        print(f"   5. clean-files                (truncate training files table)")
        print(f"\n⚠️  EXTREME CAUTION: This will DELETE ALL EXPERIMENT DATA!")

        if dry_run:
            print(f"\n🔍 DRY RUN MODE: No data will be deleted\n")

        if not force and not dry_run:
            print(f"\n❓ Type 'DELETE ALL' to confirm complete experiment cleanup: ", end='')
            confirmation = input().strip()
            if confirmation != 'DELETE ALL':
                print("❌ Operation cancelled")
                return

        # Build args for sub-commands
        sub_args = []
        if dry_run:
            sub_args.append('--dry-run')
        if force:
            sub_args.append('--force')

        success_count = 0
        failed_commands = []

        # 1. clean-distance-insert
        print(f"\n{'='*80}")
        print(f"Step 1/5: Cleaning distance tables...")
        print(f"{'='*80}\n")
        try:
            self.cmd_clean_distance_insert(sub_args)
            success_count += 1
        except Exception as e:
            print(f"❌ Error in clean-distance-insert: {e}")
            failed_commands.append('clean-distance-insert')

        # 2. clean-distance-calculate
        print(f"\n{'='*80}")
        print(f"Step 2/5: Cleaning distance work files...")
        print(f"{'='*80}\n")
        try:
            self.cmd_clean_distance_work_files(sub_args)
            success_count += 1
        except Exception as e:
            print(f"❌ Error in clean-distance-calculate: {e}")
            failed_commands.append('clean-distance-calculate')

        # 3. clean-features
        print(f"\n{'='*80}")
        print(f"Step 3/5: Cleaning feature files and table...")
        print(f"{'='*80}\n")
        try:
            self.cmd_clean_feature_files(sub_args + ['--files-and-tables'])
            success_count += 1
        except Exception as e:
            print(f"❌ Error in clean-features: {e}")
            failed_commands.append('clean-features')

        # 4. clean-segments
        print(f"\n{'='*80}")
        print(f"Step 4/5: Cleaning segment files and tables...")
        print(f"{'='*80}\n")
        try:
            self.cmd_clean_segments(sub_args + ['--files-and-tables'])
            success_count += 1
        except Exception as e:
            print(f"❌ Error in clean-segments: {e}")
            failed_commands.append('clean-segments')

        # 5. clean-files
        print(f"\n{'='*80}")
        print(f"Step 5/5: Cleaning training file selections...")
        print(f"{'='*80}\n")
        try:
            self.cmd_clean_files(sub_args)
            success_count += 1
        except Exception as e:
            print(f"❌ Error in clean-files: {e}")
            failed_commands.append('clean-files')

        # Summary
        print(f"\n{'='*80}")
        print(f"CLEANUP SUMMARY")
        print(f"{'='*80}\n")
        print(f"   Successful: {success_count}/5 commands")
        if failed_commands:
            print(f"   Failed: {len(failed_commands)} commands")
            for cmd in failed_commands:
                print(f"      - {cmd}")

        if success_count == 5:
            print(f"\n✅ Complete experiment cleanup finished successfully")
            print(f"   Experiment {self.current_experiment} is now in a clean state")
        else:
            print(f"\n⚠️  Some cleanup commands failed. Check errors above.")

    def cmd_show_distance_functions(self, args):
        """Show all distance functions in ml_distance_functions_lut

        Usage: show-distance-functions [--active-only]

        Options:
            --active-only    Only show functions where is_active = true

        Displays all distance functions available in the system.
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        active_only = '--active-only' in args

        try:
            cursor = self.db_conn.cursor()

            # Check if pairwise_metric_name column exists
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'ml_distance_functions_lut'
                AND column_name = 'pairwise_metric_name'
            """)

            has_pairwise_column = cursor.fetchone() is not None

            if has_pairwise_column:
                # Query with pairwise_metric_name
                query = """
                    SELECT distance_function_id, function_name, display_name,
                           library_name, function_import, pairwise_metric_name,
                           result_table_prefix, is_active
                    FROM ml_distance_functions_lut
                """
            else:
                # Query without pairwise_metric_name (backward compatible)
                query = """
                    SELECT distance_function_id, function_name, display_name,
                           library_name, function_import,
                           result_table_prefix, is_active
                    FROM ml_distance_functions_lut
                """

            if active_only:
                query += " WHERE is_active = true"

            query += " ORDER BY distance_function_id"

            cursor.execute(query)
            functions = cursor.fetchall()

            if not functions:
                print("\n❌ No distance functions found in ml_distance_functions_lut")
                return

            # Show warning if pairwise_metric_name column doesn't exist
            if not has_pairwise_column:
                print("\n⚠️  WARNING: pairwise_metric_name column not found!")
                print("   Run this SQL script to add it:")
                print("   psql -h localhost -p 5432 -d arc_detection -f /Users/kjensen/Documents/GitHub/mldp/mldp_cli/sql/update_distance_functions_lut.sql")
                print()

            print(f"\n📊 Distance Functions in ml_distance_functions_lut:")
            if active_only:
                print("(Showing only active functions)")
            print()

            if has_pairwise_column:
                print(f"{'ID':<4} | {'Name':<20} | {'Display Name':<30} | {'Pairwise Metric':<15} | {'Active':<6}")
                print("-" * 95)

                for func in functions:
                    func_id, name, display, library, func_import, pairwise, prefix, active = func
                    pairwise_str = pairwise or 'N/A'
                    active_str = '✅' if active else '❌'
                    print(f"{func_id:<4} | {name:<20} | {display:<30} | {pairwise_str:<15} | {active_str:<6}")
            else:
                # Without pairwise_metric_name column
                print(f"{'ID':<4} | {'Name':<20} | {'Display Name':<30} | {'Library':<30} | {'Active':<6}")
                print("-" * 100)

                for func in functions:
                    func_id, name, display, library, func_import, prefix, active = func
                    library_str = library or 'N/A'
                    active_str = '✅' if active else '❌'
                    print(f"{func_id:<4} | {name:<20} | {display:<30} | {library_str:<30} | {active_str:<6}")

            print(f"\nTotal: {len(functions)} functions")

            # Show additional details if not many
            if len(functions) <= 5:
                print("\nDetailed Information:")
                for func in functions:
                    if has_pairwise_column:
                        func_id, name, display, library, func_import, pairwise, prefix, active = func
                    else:
                        func_id, name, display, library, func_import, prefix, active = func
                        pairwise = None

                    print(f"\n{name} (ID: {func_id}):")
                    print(f"  Display: {display}")
                    print(f"  Library: {library or 'N/A'}")
                    print(f"  Function: {func_import or 'N/A'}")
                    if has_pairwise_column:
                        print(f"  Pairwise Metric: {pairwise or 'N/A'}")
                    print(f"  Table Prefix: {prefix}")
                    print(f"  Active: {'Yes' if active else 'No'}")

        except Exception as e:
            print(f"❌ Error showing distance functions: {e}")
            import traceback
            traceback.print_exc()

    def cmd_update_distance_function(self, args):
        """Update distance function in ml_distance_functions_lut

        Usage: update-distance-function <function_name> [options]

        Options:
            --pairwise-metric <name>    Set pairwise metric name for sklearn.metrics.pairwise_distances
            --library <name>            Set library name
            --function-import <name>    Set function import name
            --description <text>        Set description
            --active <true|false>       Set is_active flag

        Examples:
            update-distance-function pearson --pairwise-metric correlation
            update-distance-function manhattan --pairwise-metric manhattan --library sklearn.metrics.pairwise --function-import pairwise_distances
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        if not args or args[0].startswith('--'):
            print("❌ Error: function_name is required")
            print("\nUsage: update-distance-function <function_name> [options]")
            return

        function_name = args[0]
        updates = {}

        i = 1
        while i < len(args):
            if args[i] == '--pairwise-metric' and i + 1 < len(args):
                updates['pairwise_metric_name'] = args[i + 1]
                i += 2
            elif args[i] == '--library' and i + 1 < len(args):
                updates['library_name'] = args[i + 1]
                i += 2
            elif args[i] == '--function-import' and i + 1 < len(args):
                updates['function_import'] = args[i + 1]
                i += 2
            elif args[i] == '--description' and i + 1 < len(args):
                updates['description'] = args[i + 1]
                i += 2
            elif args[i] == '--active' and i + 1 < len(args):
                updates['is_active'] = args[i + 1].lower() in ['true', '1', 'yes']
                i += 2
            else:
                i += 1

        if not updates:
            print("❌ Error: At least one update option is required")
            print("\nAvailable options: --pairwise-metric, --library, --function-import, --description, --active")
            return

        try:
            cursor = self.db_conn.cursor()

            # Check if function exists
            cursor.execute("""
                SELECT distance_function_id, function_name, display_name
                FROM ml_distance_functions_lut
                WHERE function_name = %s
            """, (function_name,))

            function = cursor.fetchone()

            if not function:
                print(f"❌ Distance function '{function_name}' not found in ml_distance_functions_lut")
                return

            func_id, func_name, display_name = function

            # Build UPDATE query
            set_clauses = []
            values = []

            for key, value in updates.items():
                set_clauses.append(f"{key} = %s")
                values.append(value)

            values.append(func_id)

            update_query = f"""
                UPDATE ml_distance_functions_lut
                SET {', '.join(set_clauses)}, updated_at = NOW()
                WHERE distance_function_id = %s
            """

            cursor.execute(update_query, values)
            self.db_conn.commit()

            print(f"✅ Updated {func_name} ({display_name})")
            print("\nUpdated fields:")
            for key, value in updates.items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"❌ Error updating distance function: {e}")
            self.db_conn.rollback()
            import traceback
            traceback.print_exc()

    def cmd_mpcctl_distance_function(self, args):
        """Control MPCCTL distance calculation with background execution."""

        if '--help' in args:
            print("\nUsage: mpcctl-distance-function [options]")
            print("\nBackground distance calculation with pause/resume/stop control.")
            print("\nCommands:")
            print("  --start                  Start distance calculation in background")
            print("  --pause                  Pause running calculation")
            print("  --continue               Resume paused calculation")
            print("  --stop                   Stop calculation")
            print("  --status                 Show progress")
            print("  --kill-all               Forcefully terminate all workers")
            print("\nOptions for --start:")
            print("  --workers N              Number of worker processes (default: 16)")
            print("  --feature_sets 1,2,3     Comma-separated list of feature set IDs to use")
            print("  --log                    Create log file (yyyymmdd_hhmmss_mpcctl_distance_calculation.log)")
            print("  --verbose                Show verbose output in CLI")
            print("  --clean                  Start fresh (delete .mpcctl and .processed) [DEFAULT]")
            print("  --resume                 Resume from existing progress")
            print("  --force                  Skip confirmation prompt (for automation)")
            print("\nExamples:")
            print("  mpcctl-distance-function --start --workers 20")
            print("  mpcctl-distance-function --start --workers 20 --resume")
            print("  mpcctl-distance-function --start --workers 2 --feature_sets 1,2,3,4,5")
            print("  mpcctl-distance-function --start --workers 20 --log --verbose")
            print("  mpcctl-distance-function --status")
            print("  mpcctl-distance-function --pause")
            print("  mpcctl-distance-function --continue")
            print("  mpcctl-distance-function --stop")
            print("  mpcctl-distance-function --kill-all")
            return

        if '--start' in args:
            # Start distance calculation in background
            if not self.current_experiment:
                print("❌ No experiment selected. Use 'set experiment <id>' first.")
                return

            # Parse options
            workers = 16
            log_enabled = '--log' in args
            verbose = '--verbose' in args
            force = '--force' in args
            feature_set_filter = None
            clean_mode = True  # Default: start fresh

            # Parse --clean/--resume (mutually exclusive, --clean is default)
            if '--resume' in args:
                clean_mode = False

            for i, arg in enumerate(args):
                if arg == '--workers' and i + 1 < len(args):
                    try:
                        workers = int(args[i + 1])
                    except ValueError:
                        print(f"❌ Invalid workers value: {args[i + 1]}")
                        return
                elif arg == '--feature_sets' and i + 1 < len(args):
                    try:
                        feature_set_filter = [int(x.strip()) for x in args[i + 1].split(',')]
                    except ValueError:
                        print(f"❌ Invalid feature_sets value: {args[i + 1]}")
                        print("   Expected format: --feature_sets 1,2,3,4,5")
                        return

            # Import required modules
            import multiprocessing as mp
            from pathlib import Path
            from datetime import datetime
            import sys
            import psycopg2
            sys.path.insert(0, '/Users/kjensen/Documents/GitHub/mldp/mldp_distance')
            from mpcctl_cli_distance_calculator import manager_process

            # Prepare configuration
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            feature_base_path = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/feature_files')
            mpcctl_base_dir = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}')

            # Show pre-flight plan
            print(f"\n{'='*80}")
            print(f"📋 DISTANCE CALCULATION PLAN - Experiment {self.current_experiment}")
            print(f"{'='*80}\n")

            # Query configuration
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Get pair count
            try:
                cursor.execute(f"""
                    SELECT COUNT(*) FROM experiment_{self.current_experiment:03d}_segment_pairs
                """)
                total_pairs = cursor.fetchone()[0]
            except:
                total_pairs = 0

            # Get feature file count
            feature_file_count = len(list(feature_base_path.glob('**/*.npy'))) if feature_base_path.exists() else 0

            # Get distance functions
            cursor.execute("""
                SELECT df.function_name, df.display_name
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s
                ORDER BY df.function_name
            """, (self.current_experiment,))
            distance_functions = cursor.fetchall()

            # Get amplitude methods
            cursor.execute("""
                SELECT method_name
                FROM ml_experiments_amplitude_methods eam
                JOIN ml_amplitude_normalization_lut am ON eam.method_id = am.method_id
                WHERE eam.experiment_id = %s
                ORDER BY eam.method_id
            """, (self.current_experiment,))
            amplitude_methods = [row[0] for row in cursor.fetchall()]

            cursor.close()
            conn.close()

            print(f"📊 Configuration:")
            print(f"   Mode: {'--clean (starting fresh)' if clean_mode else '--resume (continuing existing)'}")
            print(f"   Workers: {workers}")
            if feature_set_filter:
                print(f"   Feature sets: {feature_set_filter} (filtered)")
            else:
                print(f"   Feature sets: All configured sets")

            print(f"\n📁 Input:")
            print(f"   Total pairs: {total_pairs:,}")
            print(f"   Feature files: {feature_file_count:,}")

            print(f"\n🎯 Distance Functions ({len(distance_functions)}):")
            for func_name, display_name in distance_functions:
                print(f"   - {display_name} ({func_name})")

            print(f"\n📐 Amplitude Methods ({len(amplitude_methods)}):")
            for method in amplitude_methods:
                print(f"   - {method}")

            # Calculate total computations
            pairs_to_compute = total_pairs if clean_mode else 0  # Resume mode = unknown remaining
            computations_per_pair = len(distance_functions) * len(amplitude_methods)
            if clean_mode:
                total_computations = pairs_to_compute * computations_per_pair
                print(f"\n📄 Expected Computations:")
                print(f"   {total_pairs:,} pairs × {len(distance_functions)} functions × {len(amplitude_methods)} methods")
                print(f"   = {total_computations:,} total distance calculations")
            else:
                print(f"\n📄 Resume Mode:")
                print(f"   Will continue from existing progress")

            print(f"\n💾 Output:")
            print(f"   .processed/{'{'}function_name{'}'}/worker_*_distance_{'{'}function_name{'}'}_batch_*.npy")
            if clean_mode:
                print(f"   ⚠️  Existing .mpcctl and .processed directories will be deleted")

            print(f"\n{'='*80}\n")

            # Confirmation prompt (skip if --force)
            if not force:
                response = input("Do you wish to continue? (Y/n): ").strip().lower()
                if response and response != 'y':
                    print("❌ Cancelled")
                    return
            else:
                print("⚠️  --force flag set: Skipping confirmation prompt\n")

            print(f"\n🚀 Starting distance calculation...\n")

            # Clean up state file BEFORE spawning manager if in clean mode
            # This prevents race condition where shell reads old state file before manager deletes it
            if clean_mode:
                state_file_path = mpcctl_base_dir / ".mpcctl_state.json"
                if state_file_path.exists():
                    try:
                        state_file_path.unlink()
                        print(f"🗑️  Removed old state file: {state_file_path}")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not remove old state file: {e}")
                        print(f"   Manager will attempt to remove it during cleanup")

            # Create log file if requested
            log_file = None
            if log_enabled:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = Path(f"{timestamp}_mpcctl_distance_calculation.log")

            # Spawn manager process in background (non-daemon so it can spawn workers)
            manager = mp.Process(
                target=manager_process,
                args=(self.current_experiment, workers, feature_base_path,
                      db_config, log_file, verbose, mpcctl_base_dir, feature_set_filter, clean_mode)
            )
            manager.start()

            print(f"🚀 Distance calculation started in background")
            print(f"   Experiment: {self.current_experiment}")
            print(f"   Workers: {workers}")
            print(f"   Mode: {'Clean' if clean_mode else 'Resume'}")
            if feature_set_filter:
                print(f"   Feature sets: {feature_set_filter}")
            if log_file:
                print(f"   Log file: {log_file}")
            print(f"\n⏳ Waiting for manager to initialize...")

            # Wait for state file to be created
            import time
            import json
            state_file = mpcctl_base_dir / ".mpcctl_state.json"
            max_wait = 10  # seconds
            waited = 0
            while not state_file.exists() and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5

            if not state_file.exists():
                print(f"⚠️  State file not created yet. Monitor progress with:")
                print(f"   mpcctl-distance-function --status")
                return

            print(f"\n📊 Live Progress Monitor (Press Ctrl+C to detach)\n")

            # Live progress monitoring loop
            try:
                last_status = None
                while True:
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)

                        progress = state.get('progress', {})
                        status = state.get('status', 'unknown')

                        # Progress bar
                        bar_width = 50
                        percent = progress.get('percent_complete', 0)
                        filled = int(bar_width * percent / 100)
                        bar = '█' * filled + '░' * (bar_width - filled)

                        # Format time
                        eta_seconds = progress.get('estimated_time_remaining_seconds', 0)
                        eta_minutes = eta_seconds // 60
                        eta_seconds_remainder = eta_seconds % 60

                        # Clear previous output (move cursor up and clear lines)
                        if last_status is not None:
                            # Move cursor up 6 lines and clear to end of screen
                            print('\033[6A\033[J', end='')

                        # Display progress
                        print(f"Status: {status}")
                        print(f"[{bar}] {percent:.1f}%")
                        print(f"Completed: {progress.get('completed_pairs', 0):,} / {progress.get('total_pairs', 0):,} pairs")
                        print(f"Rate: {progress.get('pairs_per_second', 0):.0f} pairs/sec")
                        print(f"ETA: {eta_minutes} min {eta_seconds_remainder} sec")
                        print(f"Workers: {state.get('workers_count', 0)}")

                        last_status = status

                        # Check if completed or stopped
                        if status in ['completed', 'stopped', 'killed']:
                            print(f"\n✅ Calculation {status}")
                            break

                        time.sleep(1.0)

                    except (json.JSONDecodeError, FileNotFoundError):
                        # State file might be being written
                        time.sleep(0.5)
                        continue

            except KeyboardInterrupt:
                print(f"\n\n⏸️  Detached from monitoring (calculation continues in background)")
                print(f"\n📊 Monitor progress:")
                print(f"   mpcctl-distance-function --status")
                print(f"\n⏸️  Control:")
                print(f"   mpcctl-distance-function --pause")
                print(f"   mpcctl-distance-function --continue")
                print(f"   mpcctl-distance-function --stop")
                print(f"   mpcctl-distance-function --kill-all")

        elif '--status' in args:
            # Show progress
            from pathlib import Path
            import json

            if not self.current_experiment:
                print("❌ No experiment selected. Use 'set experiment <id>' first.")
                return

            state_file = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/.mpcctl_state.json')
            if not state_file.exists():
                print("❌ No active distance calculation found")
                print("   Start with: mpcctl-distance-function --start --workers 20")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                progress = state.get('progress', {})
                status = state.get('status', 'unknown')

                # Progress bar
                bar_width = 50
                percent = progress.get('percent_complete', 0)
                filled = int(bar_width * percent / 100)
                bar = '█' * filled + '░' * (bar_width - filled)

                # Format time
                eta_seconds = progress.get('estimated_time_remaining_seconds', 0)
                eta_minutes = eta_seconds // 60
                eta_seconds_remainder = eta_seconds % 60

                print(f"\n📊 Distance Calculation Progress")
                print(f"   Status: {status}")
                print(f"   [{bar}] {percent:.1f}%")
                print(f"   Completed: {progress.get('completed_pairs', 0):,} / {progress.get('total_pairs', 0):,} pairs")
                print(f"   Rate: {progress.get('pairs_per_second', 0):.0f} pairs/sec")
                print(f"   ETA: {eta_minutes} min {eta_seconds_remainder} sec")
                print(f"   Workers: {state.get('workers_count', 0)}")

                if state.get('log_file'):
                    print(f"   Log: {state['log_file']}")

            except Exception as e:
                print(f"❌ Error reading status: {e}")

        elif '--pause' in args:
            # Send pause command
            from pathlib import Path
            import json
            from datetime import datetime

            state_file = Path('/Volumes/ArcData/V3_database/experiment041/.mpcctl_state.json')
            if not state_file.exists():
                print("❌ No active distance calculation found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                state['control'] = {
                    'command': 'pause',
                    'command_time': datetime.now().strftime('%Y%m%d_%H%M%S')
                }

                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)

                print("⏸️  Pause signal sent")
                print("   Workers will pause after current pair")
                print("   Use 'mpcctl-distance-function --status' to verify")

            except Exception as e:
                print(f"❌ Error sending pause signal: {e}")

        elif '--continue' in args:
            # Send resume command
            from pathlib import Path
            import json
            from datetime import datetime

            state_file = Path('/Volumes/ArcData/V3_database/experiment041/.mpcctl_state.json')
            if not state_file.exists():
                print("❌ No active distance calculation found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                state['control'] = {
                    'command': 'resume',
                    'command_time': datetime.now().strftime('%Y%m%d_%H%M%S')
                }

                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)

                print("▶️  Resume signal sent")
                print("   Workers will continue processing")
                print("   Use 'mpcctl-distance-function --status' to verify")

            except Exception as e:
                print(f"❌ Error sending resume signal: {e}")

        elif '--stop' in args:
            # Send stop command
            from pathlib import Path
            import json
            from datetime import datetime

            state_file = Path('/Volumes/ArcData/V3_database/experiment041/.mpcctl_state.json')
            if not state_file.exists():
                print("❌ No active distance calculation found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                state['control'] = {
                    'command': 'stop',
                    'command_time': datetime.now().strftime('%Y%m%d_%H%M%S')
                }

                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)

                print("⏹️  Stop signal sent")
                print("   Workers will exit gracefully after current pair")
                print("   Use 'mpcctl-distance-function --status' to verify")

            except Exception as e:
                print(f"❌ Error sending stop signal: {e}")

        elif '--kill-all' in args:
            # Kill all workers forcefully
            from pathlib import Path
            import json
            import signal
            import psutil

            if not self.current_experiment:
                print("❌ No experiment selected. Use 'set experiment <id>' first.")
                return

            state_file = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/.mpcctl_state.json')
            if not state_file.exists():
                print("❌ No active distance calculation found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                manager_pid = state.get('manager_pid')

                print(f"⚠️  WARNING: This will forcefully terminate all workers!")
                print(f"   Manager process PID: {manager_pid}")
                print()
                response = input("Type 'KILL' to confirm: ").strip()

                if response != 'KILL':
                    print("❌ Cancelled")
                    return

                killed_count = 0

                # Kill manager
                if manager_pid:
                    try:
                        import os
                        os.kill(manager_pid, signal.SIGTERM)
                        print(f"✅ Sent SIGTERM to manager (PID {manager_pid})")
                        killed_count += 1
                    except ProcessLookupError:
                        print(f"⚠️  Manager process {manager_pid} not found")
                    except Exception as e:
                        print(f"⚠️  Error killing manager: {e}")

                # Kill workers by finding python processes with mpcctl in command line
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        cmdline = proc.info.get('cmdline', [])
                        if cmdline and any('mpcctl_cli_distance_calculator' in str(arg) for arg in cmdline):
                            proc.terminate()
                            print(f"✅ Sent SIGTERM to worker (PID {proc.info['pid']})")
                            killed_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                print(f"\n✅ Terminated {killed_count} processes")

                # Update state file
                state['status'] = 'killed'
                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)

            except Exception as e:
                print(f"❌ Error killing processes: {e}")

        else:
            print("❌ Unknown option. Use --help for usage information.")

    def cmd_mpcctl_distance_insert(self, args):
        """Control MPCCTL distance database insertion with background execution."""

        if '--help' in args:
            print("\nUsage: mpcctl-distance-insert [options]")
            print("\nBackground database insertion with pause/resume/stop control.")
            print("\nCommands:")
            print("  --start                  Start database insertion in background")
            print("  --pause                  Pause running insertion")
            print("  --continue               Resume paused insertion")
            print("  --stop                   Stop insertion")
            print("  --status                 Show progress")
            print("  --list-processes         List active mpcctl processes with PIDs")
            print("  --kill <PID>             Kill specific process by PID")
            print("  --kill-all               Kill all mpcctl distance insert processes")
            print("\nOptions for --start:")
            print("  --workers N              Number of worker processes (default: 4)")
            print("  --distances <list>       Comma-separated distance functions (default: all)")
            print("                           Options: manhattan, euclidean, cosine, pearson")
            print("  --method <type>          Insertion method: copy (fast) or insert (safe)")
            print("                           Default: copy")
            print("  --batch-size N           Files per batch (default: 100)")
            print("  --log                    Create log file")
            print("  --verbose                Show verbose output")
            print("  --force                  Skip confirmation prompt (for automation)")
            print("\nExamples:")
            print("  mpcctl-distance-insert --start --workers 4")
            print("  mpcctl-distance-insert --start --workers 2 --distances manhattan,euclidean")
            print("  mpcctl-distance-insert --start --method copy --log --verbose")
            print("  mpcctl-distance-insert --status")
            print("  mpcctl-distance-insert --list-processes")
            print("  mpcctl-distance-insert --kill 12345")
            print("  mpcctl-distance-insert --kill-all")
            print("  mpcctl-distance-insert --pause")
            print("  mpcctl-distance-insert --continue")
            print("  mpcctl-distance-insert --stop")
            print("\nMethods:")
            print("  copy   - PostgreSQL COPY (10x faster, requires clean data)")
            print("  insert - INSERT with ON CONFLICT (slower, handles duplicates)")
            print("\nExpected Records:")
            print("  Per distance table: 2,083,954,560 records")
            print("  Total (4 tables):   8,335,818,240 records")
            return

        if '--start' in args:
            # Start database insertion in background
            if not self.current_experiment:
                print("❌ No experiment selected. Use 'set experiment <id>' first.")
                return

            # Parse options
            workers = 4
            distances = None
            method = 'copy'
            batch_size = 100
            log_enabled = '--log' in args
            verbose = '--verbose' in args
            force = '--force' in args

            for i, arg in enumerate(args):
                if arg == '--workers' and i + 1 < len(args):
                    try:
                        workers = int(args[i + 1])
                    except ValueError:
                        print(f"❌ Invalid workers value: {args[i + 1]}")
                        return
                elif arg == '--distances' and i + 1 < len(args):
                    distances = args[i + 1]
                elif arg == '--method' and i + 1 < len(args):
                    method = args[i + 1]
                    if method not in ['copy', 'insert']:
                        print(f"❌ Invalid method: {method}")
                        print("   Use 'copy' or 'insert'")
                        return
                elif arg == '--batch-size' and i + 1 < len(args):
                    try:
                        batch_size = int(args[i + 1])
                    except ValueError:
                        print(f"❌ Invalid batch size: {args[i + 1]}")
                        return

            # Import required modules
            import multiprocessing as mp
            from pathlib import Path
            from datetime import datetime
            import sys
            import psycopg2
            sys.path.insert(0, '/Users/kjensen/Documents/GitHub/mldp/mldp_distance')
            from mpcctl_distance_db_insert import manager_process

            # Prepare configuration
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            processed_dir = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/.processed')
            mpcctl_base_dir = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}')

            # Show pre-flight plan
            print(f"\n{'='*80}")
            print(f"📋 DISTANCE INSERTION PLAN - Experiment {self.current_experiment}")
            print(f"{'='*80}\n")

            # Count distance files in .processed directory
            distance_file_counts = {}
            total_files = 0
            if processed_dir.exists():
                for dist_dir in processed_dir.iterdir():
                    if dist_dir.is_dir():
                        file_count = len(list(dist_dir.glob('*.npy')))
                        distance_file_counts[dist_dir.name] = file_count
                        total_files += file_count

            # Query database for current record counts
            conn = psycopg2.connect(**db_config)
            cursor = conn.cursor()

            # Get distance functions
            cursor.execute("""
                SELECT df.function_name, df.display_name, df.result_table_prefix
                FROM ml_experiments_distance_measurements edm
                JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                WHERE edm.experiment_id = %s
                ORDER BY df.function_name
            """, (self.current_experiment,))
            distance_functions = cursor.fetchall()

            # Get current record counts in database
            db_record_counts = {}
            for func_name, display_name, table_prefix in distance_functions:
                table_name = f"experiment_{self.current_experiment:03d}_distance_{func_name}"
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    db_record_counts[func_name] = count
                except:
                    db_record_counts[func_name] = 0

            cursor.close()
            conn.close()

            print(f"📊 Configuration:")
            print(f"   Workers: {workers}")
            print(f"   Method: {method.upper()} {'(fast, PostgreSQL COPY)' if method == 'copy' else '(safe, INSERT with ON CONFLICT)'}")
            print(f"   Batch size: {batch_size} files")
            if distances:
                print(f"   Distance functions: {distances} (filtered)")
            else:
                print(f"   Distance functions: All configured")

            print(f"\n📁 Input (.processed directory):")
            if distance_file_counts:
                for func_name, count in sorted(distance_file_counts.items()):
                    print(f"   {func_name}: {count:,} files")
                print(f"   Total files: {total_files:,}")
            else:
                print(f"   ⚠️  No distance files found in {processed_dir}")

            print(f"\n💾 Current Database Records:")
            for func_name, display_name, table_prefix in distance_functions:
                count = db_record_counts.get(func_name, 0)
                table_name = f"experiment_{self.current_experiment:03d}_distance_{func_name}"
                print(f"   {display_name} ({table_name}): {count:,} records")

            if method == 'copy':
                print(f"\n⚠️  WARNING: COPY method will fail if:")
                print(f"   - Distance tables already contain data")
                print(f"   - Distance files contain duplicate keys")
                print(f"   Use INSERT method for safer incremental inserts")

            print(f"\n📄 Expected Action:")
            print(f"   Process {total_files:,} distance files from .processed/")
            print(f"   Insert records into 4 distance tables")

            print(f"\n{'='*80}\n")

            # Confirmation prompt (skip if --force)
            if not force:
                response = input("Do you wish to continue? (Y/n): ").strip().lower()
                if response and response != 'y':
                    print("❌ Cancelled")
                    return
            else:
                print("⚠️  --force flag set: Skipping confirmation prompt\n")

            print(f"\n🚀 Starting distance insertion...\n")

            # Clean up state file BEFORE spawning manager
            # This prevents race condition where shell reads old state file before manager deletes it
            state_file_path = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/distance_insert/state.json')
            if state_file_path.exists():
                try:
                    state_file_path.unlink()
                    print(f"🗑️  Removed old state file: {state_file_path}")
                except Exception as e:
                    print(f"⚠️  Warning: Could not remove old state file: {e}")
                    print(f"   Manager will attempt to remove it during cleanup")

            # Create log file if requested
            log_file = None
            if log_enabled:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = Path(f"{timestamp}_mpcctl_distance_insert.log")

            # Spawn manager process in background (non-daemon so it can spawn workers)
            manager = mp.Process(
                target=manager_process,
                args=(self.current_experiment, workers, processed_dir, db_config,
                      distances, method, batch_size, log_file, verbose, mpcctl_base_dir)
            )
            manager.start()

            print(f"🚀 Distance insertion started in background")
            print(f"   Experiment: {self.current_experiment}")
            print(f"   Workers: {workers}")
            if distances:
                print(f"   Distance functions: {distances}")
            print(f"   Method: {method}")
            print(f"   Batch size: {batch_size}")
            if log_file:
                print(f"   Log file: {log_file}")
            print(f"\n⏳ Waiting for manager to initialize...")

            # Wait for state file to be created
            import time
            import json
            state_file = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/distance_insert/state.json')
            max_wait = 10  # seconds
            waited = 0
            while not state_file.exists() and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5

            if not state_file.exists():
                print(f"⚠️  State file not created yet. Monitor progress with:")
                print(f"   mpcctl-distance-insert --status")
                return

            print(f"\n📊 Live Progress Monitor (Press Ctrl+C to detach)\n")

            # Live progress monitoring loop
            try:
                last_status = None
                while True:
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)

                        status = state.get('status', 'unknown')
                        prog = state.get('progress', {})

                        # Progress bar
                        bar_width = 50
                        percent = prog.get('percent_complete', 0)
                        filled = int(bar_width * percent / 100)
                        bar = '█' * filled + '░' * (bar_width - filled)

                        # Format ETA
                        eta_seconds = prog.get('estimated_time_remaining_seconds', 0)
                        eta_minutes = eta_seconds // 60
                        eta_seconds_remainder = eta_seconds % 60

                        # Clear previous output (move cursor up and clear lines)
                        if last_status is not None:
                            # Move cursor up 7 lines and clear to end of screen
                            print('\033[7A\033[J', end='')

                        # Display progress
                        print(f"Status: {status}")
                        print(f"[{bar}] {percent:.1f}%")
                        print(f"Completed: {prog.get('completed_files', 0):,} / {prog.get('total_files', 0):,} files")
                        print(f"Records inserted: {prog.get('records_inserted', 0):,}")
                        if 'files_per_second' in prog:
                            print(f"Rate: {prog.get('files_per_second', 0):.1f} files/sec")
                        print(f"ETA: {eta_minutes} min {eta_seconds_remainder} sec")
                        print(f"Workers: {state.get('workers_count', 0)}")

                        last_status = status

                        # Check if completed or stopped
                        if status in ['completed', 'stopped', 'killed']:
                            print(f"\n✅ Insertion {status}")
                            break

                        time.sleep(1.0)

                    except (json.JSONDecodeError, FileNotFoundError):
                        # State file might be being written
                        time.sleep(0.5)
                        continue

            except KeyboardInterrupt:
                print(f"\n\n⏸️  Detached from monitoring (insertion continues in background)")
                print(f"\n📊 Monitor progress:")
                print(f"   mpcctl-distance-insert --status")
                print(f"\n⏸️  Control:")
                print(f"   mpcctl-distance-insert --pause")
                print(f"   mpcctl-distance-insert --continue")
                print(f"   mpcctl-distance-insert --stop")

        elif '--status' in args:
            # Show status from state file
            from pathlib import Path
            import json

            state_file = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/distance_insert/state.json')
            if not state_file.exists():
                print("❌ No active distance insertion found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                status = state.get('status', 'unknown')

                print(f"\n📊 Distance Insertion Status - Experiment {state.get('experiment_id', 'N/A')}")
                print(f"   Status: {status}")

                if 'progress' in state:
                    prog = state['progress']

                    # Progress bar
                    bar_width = 50
                    percent = prog.get('percent_complete', 0)
                    filled = int(bar_width * percent / 100)
                    bar = '█' * filled + '░' * (bar_width - filled)

                    # Format ETA
                    eta_seconds = prog.get('estimated_time_remaining_seconds', 0)
                    eta_minutes = eta_seconds // 60
                    eta_seconds_remainder = eta_seconds % 60

                    print(f"   [{bar}] {percent:.1f}%")
                    print(f"   Completed: {prog.get('completed_files', 0):,} / {prog.get('total_files', 0):,} files")
                    print(f"   Records inserted: {prog.get('records_inserted', 'N/A'):,}")
                    if 'files_per_second' in prog:
                        print(f"   Rate: {prog.get('files_per_second', 0):.1f} files/sec")
                    if eta_seconds > 0:
                        print(f"   ETA: {eta_minutes} min {eta_seconds_remainder} sec")
                    print(f"   Workers: {state.get('workers_count', 'N/A')}")
                    print(f"   Manager PID: {state.get('manager_pid', 'N/A')}")

                if 'metrics' in state:
                    metrics = state['metrics']
                    print(f"\n🔧 Configuration:")
                    print(f"   Distance functions: {metrics.get('distance_functions', [])}")
                    print(f"   Method: {metrics.get('method', 'N/A')}")
                    print(f"   Batch size: {metrics.get('batch_size', 'N/A')}")

            except Exception as e:
                print(f"❌ Error reading state: {e}")

        elif '--pause' in args:
            # Send pause command
            from pathlib import Path
            import json
            from datetime import datetime

            state_file = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/.mpcctl_insert_state.json')
            if not state_file.exists():
                print("❌ No active distance insertion found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                state['control'] = {
                    'command': 'pause',
                    'command_time': datetime.now().strftime('%Y%m%d_%H%M%S')
                }

                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)

                print("⏸️  Pause signal sent")
                print("   Workers will pause after current batch")
                print("   Use 'mpcctl-distance-insert --status' to verify")

            except Exception as e:
                print(f"❌ Error sending pause signal: {e}")

        elif '--continue' in args:
            # Send resume command
            from pathlib import Path
            import json
            from datetime import datetime

            state_file = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/.mpcctl_insert_state.json')
            if not state_file.exists():
                print("❌ No active distance insertion found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                state['control'] = {
                    'command': 'resume',
                    'command_time': datetime.now().strftime('%Y%m%d_%H%M%S')
                }

                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)

                print("▶️  Resume signal sent")
                print("   Workers will continue processing")
                print("   Use 'mpcctl-distance-insert --status' to verify")

            except Exception as e:
                print(f"❌ Error sending resume signal: {e}")

        elif '--stop' in args:
            # Send stop command
            from pathlib import Path
            import json
            from datetime import datetime

            state_file = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/.mpcctl_insert_state.json')
            if not state_file.exists():
                print("❌ No active distance insertion found")
                return

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)

                state['control'] = {
                    'command': 'stop',
                    'command_time': datetime.now().strftime('%Y%m%d_%H%M%S')
                }

                with open(state_file, 'w') as f:
                    json.dump(state, f, indent=2)

                print("⏹️  Stop signal sent")
                print("   Workers will exit gracefully after current batch")
                print("   Use 'mpcctl-distance-insert --status' to verify")

            except Exception as e:
                print(f"❌ Error sending stop signal: {e}")

        elif '--list-processes' in args:
            # List all mpcctl distance insert processes
            import subprocess
            from pathlib import Path
            import json

            state_file = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/.mpcctl_insert_state.json')

            print(f"\n🔍 MPCCTL Distance Insert Processes (Experiment {self.current_experiment}):")
            print("=" * 70)

            # Check state file for manager PID
            manager_pid = None
            if state_file.exists():
                try:
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    manager_pid = state.get('manager_pid')
                    print(f"\nManager Process:")
                    print(f"  PID: {manager_pid}")
                    print(f"  Status: {state.get('status', 'unknown')}")
                    print(f"  Workers: {state.get('workers_count', 0)}")
                    print(f"  Start Time: {state.get('start_time', 'unknown')}")
                except Exception as e:
                    print(f"  ⚠️  Error reading state file: {e}")
            else:
                print(f"\n  ℹ️  No active manager process (no state file)")

            # List all Python processes containing mpcctl_distance_db_insert
            try:
                result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                processes = []
                for line in result.stdout.split('\n'):
                    if 'mpcctl_distance_db_insert' in line or (manager_pid and str(manager_pid) in line):
                        parts = line.split()
                        if len(parts) >= 2:
                            processes.append({'pid': parts[1], 'line': line})

                if processes:
                    print(f"\n\nActive Processes:")
                    for proc in processes:
                        print(f"  PID {proc['pid']}: {proc['line'][80:]}" if len(proc['line']) > 80 else f"  PID {proc['pid']}")
                else:
                    print(f"\n  ℹ️  No active worker processes found")

            except Exception as e:
                print(f"  ⚠️  Error listing processes: {e}")

            print()

        elif '--kill' in args:
            # Kill specific PID
            import signal
            import os

            # Get PID from arguments
            pid = None
            for i, arg in enumerate(args):
                if arg == '--kill' and i + 1 < len(args):
                    try:
                        pid = int(args[i + 1])
                    except ValueError:
                        print(f"❌ Invalid PID: {args[i + 1]}")
                        return

            if pid is None:
                print("❌ No PID specified. Usage: mpcctl-distance-insert --kill <PID>")
                return

            try:
                os.kill(pid, signal.SIGTERM)
                print(f"⚠️  Sent SIGTERM to PID {pid}")
                print(f"   Process will terminate gracefully")
            except ProcessLookupError:
                print(f"❌ No process found with PID {pid}")
            except PermissionError:
                print(f"❌ Permission denied to kill PID {pid}")
            except Exception as e:
                print(f"❌ Error killing process {pid}: {e}")

        elif '--kill-all' in args:
            # Kill all mpcctl distance insert processes
            import subprocess
            import signal
            import os
            from pathlib import Path
            import json

            state_file = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/.mpcctl_insert_state.json')

            # Get manager PID from state file
            manager_pid = None
            if state_file.exists():
                try:
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                    manager_pid = state.get('manager_pid')
                except:
                    pass

            # Find all related processes (manager + all spawn_main workers)
            pids_to_kill = []
            try:
                result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    # Kill manager process, all spawn_main workers, and distance insert processes
                    if ('mpcctl_distance_db_insert' in line or
                        'spawn_main' in line or
                        (manager_pid and str(manager_pid) in line)):
                        parts = line.split()
                        if len(parts) >= 2:
                            try:
                                pid = int(parts[1])
                                # Avoid killing ourselves (the CLI process)
                                if pid != os.getpid():
                                    pids_to_kill.append(pid)
                            except ValueError:
                                continue
            except Exception as e:
                print(f"❌ Error finding processes: {e}")
                return

            if not pids_to_kill:
                print("ℹ️  No mpcctl-distance-insert processes found")
                return

            print(f"⚠️  Found {len(pids_to_kill)} process(es) to kill:")
            for pid in pids_to_kill:
                print(f"   PID {pid}")

            # Confirm
            response = input("\nKill all these processes? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("❌ Cancelled")
                return

            # Kill all processes
            killed = 0
            for pid in pids_to_kill:
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"✅ Killed PID {pid}")
                    killed += 1
                except ProcessLookupError:
                    print(f"⚠️  PID {pid} already terminated")
                except PermissionError:
                    print(f"❌ Permission denied for PID {pid}")
                except Exception as e:
                    print(f"❌ Error killing PID {pid}: {e}")

            print(f"\n✅ Killed {killed}/{len(pids_to_kill)} processes")

            # Clean up state file
            if state_file.exists():
                try:
                    os.remove(state_file)
                    print(f"✅ Removed state file")
                except Exception as e:
                    print(f"⚠️  Could not remove state file: {e}")

        else:
            print("❌ Unknown option. Use --help for usage information.")

    def cmd_mpcctl_execute_experiment(self, args):
        """Execute complete MPCCTL experiment pipeline from file selection to distance insertion

        Usage: mpcctl-execute-experiment [options]

        Options:
            --workers N         Number of worker processes for distance calculation (default: 20)
            --log               Create log files for distance calculation and insertion
            --verbose           Show verbose output during execution
            --force             Skip confirmation prompt (DANGEROUS - for automation only!)
            --skip-file-selection       Skip select-files step
            --skip-segment-selection    Skip select-segments step
            --skip-segment-fileset      Skip generate-segment-fileset step
            --skip-segment-pairs        Skip generate-segment-pairs step
            --skip-feature-fileset      Skip generate-feature-fileset step
            --skip-distance-calc        Skip mpcctl-distance-function step
            --skip-distance-insert      Skip mpcctl-distance-insert step

        This command executes the complete MPCCTL pipeline in the following order:
            1. select-files              (file selection for training)
            2. select-segments           (segment selection for training)
            3. generate-segment-fileset  (generate segment files on disk)
            4. generate-segment-pairs    (generate all segment pairs)
            5. generate-feature-fileset  (extract features from segments)
            6. mpcctl-distance-function  (calculate distances with N workers)
            7. mpcctl-distance-insert    (insert distances into database with N workers)

        Each step's execution time is tracked and reported in the final summary.

        Examples:
            mpcctl-execute-experiment --workers 20 --log --verbose
            mpcctl-execute-experiment --workers 10 --skip-file-selection
            mpcctl-execute-experiment --workers 20 --log --verbose --force
        """
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("❌ No experiment selected. Use 'set experiment <id>' first.")
            return

        if '--help' in args:
            print("\nUsage: mpcctl-execute-experiment [options]")
            print("\nExecute complete MPCCTL experiment pipeline from file selection to distance insertion")
            print("\nOptions:")
            print("  --workers N                     Number of worker processes (default: 20)")
            print("  --log                           Create log files for distance operations")
            print("  --verbose                       Show verbose output")
            print("  --force                         Skip confirmation prompt (DANGEROUS!)")
            print("  --skip-file-selection           Skip select-files step")
            print("  --skip-segment-selection        Skip select-segments step")
            print("  --skip-segment-fileset          Skip generate-segment-fileset step")
            print("  --skip-segment-pairs            Skip generate-segment-pairs step")
            print("  --skip-feature-fileset          Skip generate-feature-fileset step")
            print("  --skip-distance-calc            Skip mpcctl-distance-function step")
            print("  --skip-distance-insert          Skip mpcctl-distance-insert step")
            print("\nPipeline Steps:")
            print("  1. select-files              File selection for training")
            print("  2. select-segments           Segment selection for training")
            print("  3. generate-segment-fileset  Generate segment files on disk")
            print("  4. generate-segment-pairs    Generate all segment pairs")
            print("  5. generate-feature-fileset  Extract features from segments")
            print("  6. mpcctl-distance-function  Calculate distances (parallel)")
            print("  7. mpcctl-distance-insert    Insert distances into database (parallel)")
            print("\nExamples:")
            print("  mpcctl-execute-experiment --workers 20 --log --verbose")
            print("  mpcctl-execute-experiment --workers 10 --skip-file-selection")
            print("  mpcctl-execute-experiment --workers 20 --log --verbose --force")
            return

        # Parse options
        workers = 20
        log_enabled = '--log' in args
        verbose = '--verbose' in args
        force = '--force' in args
        skip_file_selection = '--skip-file-selection' in args
        skip_segment_selection = '--skip-segment-selection' in args
        skip_segment_fileset = '--skip-segment-fileset' in args
        skip_segment_pairs = '--skip-segment-pairs' in args
        skip_feature_fileset = '--skip-feature-fileset' in args
        skip_distance_calc = '--skip-distance-calc' in args
        skip_distance_insert = '--skip-distance-insert' in args

        for i, arg in enumerate(args):
            if arg == '--workers' and i + 1 < len(args):
                try:
                    workers = int(args[i + 1])
                except ValueError:
                    print(f"❌ Invalid workers value: {args[i + 1]}")
                    return

        # Banner
        print(f"\n{'='*80}")
        print(f"🚀 MPCCTL EXPERIMENT PIPELINE - Experiment {self.current_experiment}")
        print(f"{'='*80}\n")
        print(f"📋 Configuration:")
        print(f"   Workers: {workers}")
        print(f"   Log files: {'Enabled' if log_enabled else 'Disabled'}")
        print(f"   Verbose: {'Enabled' if verbose else 'Disabled'}")
        print(f"\n📊 Pipeline Steps:")
        step_num = 1
        if not skip_file_selection:
            print(f"   {step_num}. select-files")
            step_num += 1
        else:
            print(f"   ⏭️  Skipping: select-files")

        if not skip_segment_selection:
            print(f"   {step_num}. select-segments")
            step_num += 1
        else:
            print(f"   ⏭️  Skipping: select-segments")

        if not skip_segment_fileset:
            print(f"   {step_num}. generate-segment-fileset")
            step_num += 1
        else:
            print(f"   ⏭️  Skipping: generate-segment-fileset")

        if not skip_segment_pairs:
            print(f"   {step_num}. generate-segment-pairs")
            step_num += 1
        else:
            print(f"   ⏭️  Skipping: generate-segment-pairs")

        if not skip_feature_fileset:
            print(f"   {step_num}. generate-feature-fileset")
            step_num += 1
        else:
            print(f"   ⏭️  Skipping: generate-feature-fileset")

        if not skip_distance_calc:
            print(f"   {step_num}. mpcctl-distance-function (workers={workers})")
            step_num += 1
        else:
            print(f"   ⏭️  Skipping: mpcctl-distance-function")

        if not skip_distance_insert:
            print(f"   {step_num}. mpcctl-distance-insert (workers={workers})")
        else:
            print(f"   ⏭️  Skipping: mpcctl-distance-insert")

        print(f"\n{'='*80}\n")

        # Confirmation prompt (skip if --force)
        if not force:
            response = input("Do you wish to continue with pipeline execution? (Y/n): ").strip().lower()
            if response and response != 'y':
                print("❌ Pipeline execution cancelled")
                return
        else:
            print("⚠️  --force flag set: Skipping confirmation prompt\n")

        # Execute pipeline with timing
        import time
        from datetime import datetime

        success_count = 0
        failed_commands = []
        step_times = {}
        total_start = time.time()

        # Step 1: select-files
        if not skip_file_selection:
            print(f"\n{'='*80}")
            print(f"Step 1: Running select-files...")
            print(f"{'='*80}\n")
            step_start = time.time()
            try:
                self.cmd_select_files([])
                step_times['select-files'] = time.time() - step_start
                success_count += 1
            except Exception as e:
                print(f"❌ Error in select-files: {e}")
                failed_commands.append('select-files')
                step_times['select-files'] = time.time() - step_start

        # Step 2: select-segments
        if not skip_segment_selection:
            print(f"\n{'='*80}")
            print(f"Step 2: Running select-segments...")
            print(f"{'='*80}\n")
            step_start = time.time()
            try:
                self.cmd_select_segments([])
                step_times['select-segments'] = time.time() - step_start
                success_count += 1
            except Exception as e:
                print(f"❌ Error in select-segments: {e}")
                failed_commands.append('select-segments')
                step_times['select-segments'] = time.time() - step_start

        # Step 3: generate-segment-fileset
        if not skip_segment_fileset:
            print(f"\n{'='*80}")
            print(f"Step 3: Running generate-segment-fileset...")
            print(f"{'='*80}\n")
            step_start = time.time()
            try:
                self.cmd_generate_segment_fileset([])
                step_times['generate-segment-fileset'] = time.time() - step_start
                success_count += 1
            except Exception as e:
                print(f"❌ Error in generate-segment-fileset: {e}")
                failed_commands.append('generate-segment-fileset')
                step_times['generate-segment-fileset'] = time.time() - step_start

        # Step 4: generate-segment-pairs
        if not skip_segment_pairs:
            print(f"\n{'='*80}")
            print(f"Step 4: Running generate-segment-pairs...")
            print(f"{'='*80}\n")
            step_start = time.time()
            try:
                self.cmd_generate_segment_pairs([])
                step_times['generate-segment-pairs'] = time.time() - step_start
                success_count += 1
            except Exception as e:
                print(f"❌ Error in generate-segment-pairs: {e}")
                failed_commands.append('generate-segment-pairs')
                step_times['generate-segment-pairs'] = time.time() - step_start

        # Step 5: generate-feature-fileset
        if not skip_feature_fileset:
            print(f"\n{'='*80}")
            print(f"Step 5: Running generate-feature-fileset...")
            print(f"{'='*80}\n")
            step_start = time.time()
            try:
                self.cmd_generate_feature_fileset([])
                step_times['generate-feature-fileset'] = time.time() - step_start
                success_count += 1
            except Exception as e:
                print(f"❌ Error in generate-feature-fileset: {e}")
                failed_commands.append('generate-feature-fileset')
                step_times['generate-feature-fileset'] = time.time() - step_start

        # Step 6: mpcctl-distance-function
        if not skip_distance_calc:
            print(f"\n{'='*80}")
            print(f"Step 6: Running mpcctl-distance-function...")
            print(f"{'='*80}\n")
            step_start = time.time()
            try:
                distance_args = ['--start', '--workers', str(workers), '--force']
                if log_enabled:
                    distance_args.append('--log')
                if verbose:
                    distance_args.append('--verbose')
                self.cmd_mpcctl_distance_function(distance_args)
                step_times['mpcctl-distance-function'] = time.time() - step_start
                success_count += 1
            except Exception as e:
                print(f"❌ Error in mpcctl-distance-function: {e}")
                failed_commands.append('mpcctl-distance-function')
                step_times['mpcctl-distance-function'] = time.time() - step_start

        # Step 7: mpcctl-distance-insert
        if not skip_distance_insert:
            print(f"\n{'='*80}")
            print(f"Step 7: Running mpcctl-distance-insert...")
            print(f"{'='*80}\n")
            step_start = time.time()
            try:
                insert_args = ['--start', '--workers', str(workers), '--force']
                if log_enabled:
                    insert_args.append('--log')
                if verbose:
                    insert_args.append('--verbose')
                self.cmd_mpcctl_distance_insert(insert_args)
                step_times['mpcctl-distance-insert'] = time.time() - step_start
                success_count += 1
            except Exception as e:
                print(f"❌ Error in mpcctl-distance-insert: {e}")
                failed_commands.append('mpcctl-distance-insert')
                step_times['mpcctl-distance-insert'] = time.time() - step_start

        # Calculate total time
        total_time = time.time() - total_start
        total_steps = 7 - sum([skip_file_selection, skip_segment_selection, skip_segment_fileset,
                               skip_segment_pairs, skip_feature_fileset, skip_distance_calc,
                               skip_distance_insert])

        # Summary
        print(f"\n{'='*80}")
        print(f"PIPELINE EXECUTION SUMMARY")
        print(f"{'='*80}\n")
        print(f"   Successful: {success_count}/{total_steps} steps")
        if failed_commands:
            print(f"   Failed: {len(failed_commands)} steps")
            for cmd in failed_commands:
                print(f"      - {cmd}")

        print(f"\n⏱️  Execution Times:")
        for step, duration in step_times.items():
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            print(f"   {step}: {time_str}")

        # Total time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        if hours > 0:
            total_time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            total_time_str = f"{minutes}m {seconds}s"
        else:
            total_time_str = f"{seconds}s"
        print(f"\n   Total time: {total_time_str}")

        if success_count == total_steps:
            print(f"\n✅ Pipeline execution completed successfully")
            print(f"   Experiment {self.current_experiment} is ready for analysis")
        else:
            print(f"\n⚠️  Some pipeline steps failed. Check errors above.")

    def cmd_generate_feature_fileset(self, args):
        """Generate feature files from segment data"""
        if not self.db_conn:
            print("❌ Not connected to database. Use 'connect' first.")
            return

        # Show help if requested or if no experiment set
        if '--help' in args or not self.current_experiment:
            print("\nUsage: generate-feature-fileset [options]")
            print("\nThis command extracts features from ALL segment files (all decimation levels).")
            print("\nBy default, processes ALL segment files and ALL active feature sets.")
            print("\nOptions:")
            print("  --feature-sets <list>    Comma-separated feature set IDs (default: all active)")
            print("  --max-segments N         Maximum segment FILES to process (default: all)")
            print("  --force                  Force re-extraction of existing features")
            print("  --clean                  Clear existing feature files before generation")
            print("\nExamples:")
            print("  generate-feature-fileset")
            print("  generate-feature-fileset --feature-sets 1,2,3")
            print("  generate-feature-fileset --max-segments 1000")
            print("  generate-feature-fileset --force")
            print("  generate-feature-fileset --clean")
            print("\n📝 Pipeline Order:")
            print("  1. select-files          - Select files for training (DB)")
            print("  2. select-segments       - Select segments for training (DB)")
            print("  3. generate-training-data - Create training data tables (DB)")
            print("  4. generate-segment-fileset - Create physical segment files (Disk)")
            print("  5. generate-feature-fileset - Extract features from segments (Disk)")
            print("\n📁 Input Structure:")
            print("  experiment{NNN}/segment_files/S{size}/T{type}/D{decimation}/*.npy")
            print("\n📁 Output Structure:")
            print("  experiment{NNN}/feature_files/S{size}/T{type}/D{decimation}/*_FS{id}[_N_{n}].npy")
            print("\n📊 Processing Details:")
            print("  - Processes ALL decimation levels (S000512 to S524288)")
            print("  - Processes ALL ADC types (TRAW, TADC6, TADC8, TADC10, TADC12, TADC14)")
            print("  - Mirrors segment_files/ directory structure exactly")
            print("  - Tracks original_segment_length AND stored_segment_length")
            print("  - Enables decimation/information-loss analysis")
            print("\n⚙️  Database Records:")
            print("  Each extraction creates a record in experiment_{NNN}_feature_fileset with:")
            print("    - segment_id, file_id, feature_set_id, n_value")
            print("    - original_segment_length (from data_segments)")
            print("    - stored_segment_length (from filesystem path)")
            print("    - adc_type, adc_division")
            print("    - feature_file_path, num_chunks, extraction_time")
            if not self.current_experiment:
                print("\n⚠️  No experiment selected. Use 'set experiment <id>' first.")
            return

        feature_set_ids = None  # Default: all configured feature sets
        max_segments = None
        force_reextract = False
        clean_first = False  # Default

        # Parse arguments
        i = 0
        while i < len(args):
            if args[i] == '--feature-sets' and i + 1 < len(args):
                feature_set_ids = [int(x) for x in args[i + 1].split(',')]
                i += 2
            elif args[i] == '--max-segments' and i + 1 < len(args):
                max_segments = int(args[i + 1])
                i += 2
            elif args[i] == '--force':
                force_reextract = True
                i += 1
            elif args[i] == '--clean':
                clean_first = True
                i += 1
            else:
                i += 1

        # Clean existing feature files if requested
        if clean_first:
            print(f"\n🗑️  Cleaning existing feature files...")
            self.cmd_clean_feature_files([])
            print()

        # Show pre-flight plan
        print(f"\n{'='*80}")
        print(f"📋 FEATURE EXTRACTION PLAN - Experiment {self.current_experiment}")
        print(f"{'='*80}\n")

        # Query configuration with proper error handling
        try:
            # Rollback any existing transaction errors
            self.db_conn.rollback()

            cursor = self.db_conn.cursor()

            # Get segment file count
            try:
                segment_path = Path(f'/Volumes/ArcData/V3_database/experiment{self.current_experiment:03d}/segment_files')
                segment_file_count = len(list(segment_path.glob('**/*.npy'))) if segment_path.exists() else 0
            except:
                segment_file_count = 0

            # Get feature set info
            if feature_set_ids:
                placeholders = ','.join(['%s'] * len(feature_set_ids))
                cursor.execute(f"""
                    SELECT fs.feature_set_id, fs.feature_set_name
                    FROM ml_experiments_feature_sets efs
                    JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
                    WHERE efs.experiment_id = %s AND fs.feature_set_id IN ({placeholders})
                    ORDER BY fs.feature_set_id
                """, (self.current_experiment, *feature_set_ids))
            else:
                cursor.execute("""
                    SELECT fs.feature_set_id, fs.feature_set_name
                    FROM ml_experiments_feature_sets efs
                    JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
                    WHERE efs.experiment_id = %s AND efs.is_active = true
                    ORDER BY fs.feature_set_id
                """, (self.current_experiment,))

            feature_sets = cursor.fetchall()
            cursor.close()

        except psycopg2.Error as e:
            print(f"\n❌ Database error: {e}")
            print(f"⚠️  Rolling back transaction...")
            self.db_conn.rollback()
            return
        except Exception as e:
            print(f"\n❌ Error during configuration query: {e}")
            return

        print(f"📊 Input:")
        print(f"   Segment files available: {segment_file_count:,}")
        if max_segments:
            print(f"   Will process: {max_segments:,} files (limited)")
        else:
            print(f"   Will process: All {segment_file_count:,} files")

        print(f"\n🎯 Feature Sets to Extract ({len(feature_sets)}):")
        for fs_id, fs_name in feature_sets:
            print(f"   - ID {fs_id}: {fs_name}")

        print(f"\n📄 Expected Output:")
        files_to_create = (max_segments if max_segments else segment_file_count) * len(feature_sets)
        print(f"   Feature files to create: ~{files_to_create:,}")
        if force_reextract:
            print(f"   Mode: FORCE re-extraction (will overwrite existing)")
        else:
            print(f"   Mode: Skip existing files")

        print(f"\n{'='*80}\n")

        # Confirmation prompt (skip if --force)
        if not force_reextract:
            response = input("Do you wish to continue? (Y/n): ").strip().lower()
            if response and response != 'y':
                print("❌ Cancelled")
                return
        else:
            print("⚠️  --force flag set: Skipping confirmation prompt\n")

        print(f"\n🔄 Starting feature extraction...")

        try:
            # Import the feature extractor module
            from experiment_feature_extractor import ExperimentFeatureExtractor

            # Create extractor instance
            extractor = ExperimentFeatureExtractor(self.current_experiment, self.db_conn)

            # Extract features
            result = extractor.extract_features(
                feature_set_ids=feature_set_ids,
                max_segments=max_segments,
                force_reextract=force_reextract
            )
            
            if result['success']:
                print(f"\n✅ Successfully extracted features!")
                print(f"   Total segments: {result['total_segments']}")
                print(f"   Total feature sets: {result['total_feature_sets']}")
                print(f"   Total extracted: {result['total_extracted']}")
                
                if result['failed_count'] > 0:
                    print(f"\n⚠️  Failed extractions: {result['failed_count']}")
                    if result.get('failed_extractions'):
                        print("   First few failures:")
                        for fail in result['failed_extractions'][:5]:
                            print(f"     Segment {fail['segment_id']}, FS {fail['feature_set_id']}: {fail['error']}")
                
                if result.get('average_extraction_time'):
                    print(f"\n⏱️  Performance:")
                    print(f"   Average time per extraction: {result['average_extraction_time']:.2f}s")
                    print(f"   Total extraction time: {result['total_extraction_time']:.2f}s")
            else:
                print(f"\n❌ Failed to extract features: {result.get('error', 'Unknown error')}")
                
        except ImportError:
            print("❌ ExperimentFeatureExtractor module not found")
            print("   Make sure experiment_feature_extractor.py is in the same directory")
        except Exception as e:
            print(f"❌ Error generating feature fileset: {e}")
    
    def _create_segment_selector_module(self):
        """Create the segment selector module if it doesn't exist"""
        print("\nCreating ExperimentSegmentSelector module...")
        # The module has been created separately
        print("   Module should be available at: experiment_segment_selector.py")

    # ========== Classifier Management Commands (Phase 0a) ==========

    def cmd_classifier_create_registry(self, args):
        """
        Create ml_experiment_classifiers registry table

        Usage: classifier-create-registry

        Creates the global registry table for tracking classifier instances
        across all experiments. This table stores metadata about each
        classifier including name, type, creation date, and status.

        Table Schema:
        - experiment_id + classifier_id: Primary key
        - classifier_name: Unique within experiment
        - classifier_type: svm, random_forest, xgboost, etc.
        - is_active, is_archived: Status flags
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        try:
            cursor = self.db_conn.cursor()

            # Create ml_experiment_classifiers table
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS ml_experiment_classifiers (
                    experiment_id INTEGER NOT NULL,
                    classifier_id INTEGER NOT NULL,
                    classifier_name VARCHAR(255) NOT NULL,
                    classifier_description TEXT,
                    classifier_type VARCHAR(50) DEFAULT 'svm',
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    created_by VARCHAR(100),
                    is_active BOOLEAN DEFAULT TRUE,
                    is_archived BOOLEAN DEFAULT FALSE,
                    notes TEXT,
                    PRIMARY KEY (experiment_id, classifier_id),
                    FOREIGN KEY (experiment_id) REFERENCES ml_experiments(experiment_id) ON DELETE CASCADE,
                    UNIQUE (experiment_id, classifier_name)
                );
            """

            cursor.execute(create_table_sql)

            # Create indexes
            index_sqls = [
                "CREATE INDEX IF NOT EXISTS idx_exp_classifiers_experiment ON ml_experiment_classifiers(experiment_id);",
                "CREATE INDEX IF NOT EXISTS idx_exp_classifiers_active ON ml_experiment_classifiers(is_active);",
                "CREATE INDEX IF NOT EXISTS idx_exp_classifiers_type ON ml_experiment_classifiers(classifier_type);"
            ]

            for index_sql in index_sqls:
                cursor.execute(index_sql)

            self.db_conn.commit()

            print("[SUCCESS] Created ml_experiment_classifiers table")
            print("  - Primary key: (experiment_id, classifier_id)")
            print("  - Unique constraint: (experiment_id, classifier_name)")
            print("  - Indexes: experiment_id, is_active, classifier_type")
            print("  - Foreign key: experiment_id -> ml_experiments")

        except Exception as e:
            self.db_conn.rollback()
            print(f"[ERROR] Failed to create ml_experiment_classifiers table: {e}")

    def cmd_classifier_migrate_registry(self, args):
        """
        Migrate ml_experiment_classifiers to add global_classifier_id

        Usage: classifier-migrate-registry [--force]

        This migration adds global_classifier_id as the new PRIMARY KEY.
        The global_classifier_id is a SERIAL field that uniquely identifies
        classifiers across ALL experiments, enabling proper CASCADE deletes
        and junction table relationships.

        Changes:
        1. Add global_classifier_id SERIAL column
        2. Backfill existing rows with sequential IDs
        3. Change PRIMARY KEY from (experiment_id, classifier_id) to global_classifier_id
        4. Add UNIQUE constraint on (experiment_id, classifier_id)
        5. Create index on global_classifier_id

        Options:
            --force    Skip confirmation prompt
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        force = '--force' in args.split() if args else False

        try:
            cursor = self.db_conn.cursor()

            # Check if migration already done
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'ml_experiment_classifiers'
                AND column_name = 'global_classifier_id'
            """)

            if cursor.fetchone():
                print("[INFO] Migration already complete - global_classifier_id exists")
                return

            # Show warning and get confirmation
            if not force:
                print("\n[WARNING] This migration will modify ml_experiment_classifiers")
                print("          - Add global_classifier_id as new PRIMARY KEY")
                print("          - Change (experiment_id, classifier_id) to UNIQUE constraint")
                print("          - This is a one-way migration")
                print("\nType 'MIGRATE REGISTRY' to confirm:")

                confirmation = input().strip()
                if confirmation != 'MIGRATE REGISTRY':
                    print("[CANCELLED] Migration aborted")
                    return

            print("\n[STEP 1/6] Adding global_classifier_id column...")
            cursor.execute("""
                ALTER TABLE ml_experiment_classifiers
                ADD COLUMN global_classifier_id SERIAL
            """)
            print("  ✓ Column added")

            print("\n[STEP 2/6] Backfilling existing rows...")
            # The SERIAL type automatically assigns values via the sequence
            # But we need to update rows that were added before the column existed
            cursor.execute("""
                SELECT COUNT(*) FROM ml_experiment_classifiers
                WHERE global_classifier_id IS NULL
            """)
            null_count = cursor.fetchone()[0]

            if null_count > 0:
                cursor.execute("""
                    UPDATE ml_experiment_classifiers
                    SET global_classifier_id = nextval('ml_experiment_classifiers_global_classifier_id_seq')
                    WHERE global_classifier_id IS NULL
                """)
                print(f"  ✓ Backfilled {null_count} rows")
            else:
                print("  ✓ No backfill needed")

            print("\n[STEP 3/6] Dropping old PRIMARY KEY constraint...")
            cursor.execute("""
                ALTER TABLE ml_experiment_classifiers
                DROP CONSTRAINT ml_experiment_classifiers_pkey
            """)
            print("  ✓ Old PRIMARY KEY dropped")

            print("\n[STEP 4/6] Adding new PRIMARY KEY on global_classifier_id...")
            cursor.execute("""
                ALTER TABLE ml_experiment_classifiers
                ADD PRIMARY KEY (global_classifier_id)
            """)
            print("  ✓ New PRIMARY KEY added")

            print("\n[STEP 5/6] Adding UNIQUE constraint on (experiment_id, classifier_id)...")
            cursor.execute("""
                ALTER TABLE ml_experiment_classifiers
                ADD CONSTRAINT ml_experiment_classifiers_exp_cls_unique
                UNIQUE (experiment_id, classifier_id)
            """)
            print("  ✓ UNIQUE constraint added")

            print("\n[STEP 6/6] Creating index on global_classifier_id...")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_exp_classifiers_global
                ON ml_experiment_classifiers(global_classifier_id)
            """)
            print("  ✓ Index created")

            self.db_conn.commit()

            print("\n[SUCCESS] Migration complete!")
            print("  - global_classifier_id is now PRIMARY KEY")
            print("  - (experiment_id, classifier_id) is now UNIQUE")
            print("  - Ready for junction table creation")

            # Show current state
            cursor.execute("""
                SELECT global_classifier_id, experiment_id, classifier_id, classifier_name
                FROM ml_experiment_classifiers
                ORDER BY global_classifier_id
            """)
            rows = cursor.fetchall()

            if rows:
                print("\nCurrent classifiers:")
                headers = ['Global ID', 'Exp ID', 'Cls ID', 'Name']
                print(tabulate(rows, headers=headers, tablefmt='simple'))

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Migration failed: {e}")
            print("  Database rolled back to previous state")

    def cmd_classifier_create_junction_tables(self, args):
        """
        Create 6 global junction tables for classifier configuration hyperparameters

        Usage: classifier-create-junction-tables [--force]

        Creates the normalized junction tables that replace array-based storage.
        Each junction table has a 4-part composite PRIMARY KEY:
        (global_classifier_id, experiment_id, config_id, element_id)

        Junction Tables Created:
        1. ml_classifier_config_decimation_factors
        2. ml_classifier_config_data_types
        3. ml_classifier_config_amplitude_methods
        4. ml_classifier_config_experiment_feature_sets
        5. ml_classifier_config_feature_set_features
        6. ml_classifier_config_distance_functions

        Benefits:
        - Referential integrity with foreign keys
        - Proper 1NF compliance (no arrays)
        - Easy querying: "find all configs using decimation_factor=7"
        - CASCADE deletes work properly
        - Validates all hyperparameter values against lookup tables

        Options:
            --force    Recreate tables if they already exist
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        force = '--force' in args.split() if args else False

        try:
            cursor = self.db_conn.cursor()

            # Check if global_classifier_id exists
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'ml_experiment_classifiers'
                AND column_name = 'global_classifier_id'
            """)

            if not cursor.fetchone():
                print("[ERROR] ml_experiment_classifiers not migrated yet")
                print("  Run: classifier-migrate-registry")
                return

            # Define all 6 junction tables
            junction_tables = [
                {
                    'name': 'ml_classifier_config_decimation_factors',
                    'element_column': 'decimation_factor',
                    'element_type': 'INTEGER',
                    'foreign_keys': []
                },
                {
                    'name': 'ml_classifier_config_data_types',
                    'element_column': 'data_type_id',
                    'element_type': 'INTEGER',
                    'foreign_keys': [
                        ('data_type_id', 'ml_data_types_lut', 'data_type_id')
                    ]
                },
                {
                    'name': 'ml_classifier_config_amplitude_methods',
                    'element_column': 'amplitude_processing_method_id',
                    'element_type': 'INTEGER',
                    'foreign_keys': []
                },
                {
                    'name': 'ml_classifier_config_experiment_feature_sets',
                    'element_column': 'experiment_feature_set_id',
                    'element_type': 'BIGINT',
                    'foreign_keys': []
                },
                {
                    'name': 'ml_classifier_config_feature_set_features',
                    'element_column': 'feature_set_feature_id',
                    'element_type': 'BIGINT',
                    'foreign_keys': []
                },
                {
                    'name': 'ml_classifier_config_distance_functions',
                    'element_column': 'distance_function_id',
                    'element_type': 'INTEGER',
                    'foreign_keys': [
                        ('distance_function_id', 'ml_distance_functions_lut', 'distance_function_id')
                    ]
                }
            ]

            tables_created = []
            tables_existed = []

            for table_def in junction_tables:
                table_name = table_def['name']
                element_column = table_def['element_column']
                element_type = table_def['element_type']
                foreign_keys = table_def['foreign_keys']

                # Check if table exists
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name = %s
                """, (table_name,))

                exists = cursor.fetchone()

                if exists and not force:
                    tables_existed.append(table_name)
                    continue

                if exists and force:
                    print(f"  Dropping {table_name}...")
                    cursor.execute(f"DROP TABLE {table_name} CASCADE")

                # Create junction table
                print(f"  Creating {table_name}...")

                # Build foreign key clauses
                fk_clauses = [
                    """FOREIGN KEY (global_classifier_id)
                        REFERENCES ml_experiment_classifiers(global_classifier_id)
                        ON DELETE CASCADE"""
                ]

                for fk_col, fk_table, fk_ref_col in foreign_keys:
                    fk_clauses.append(f"""FOREIGN KEY ({fk_col})
                        REFERENCES {fk_table}({fk_ref_col})""")

                fk_sql = ',\n    '.join(fk_clauses)

                create_sql = f"""
                    CREATE TABLE {table_name} (
                        global_classifier_id INTEGER NOT NULL,
                        experiment_id INTEGER NOT NULL,
                        config_id INTEGER NOT NULL,
                        {element_column} {element_type} NOT NULL,
                        PRIMARY KEY (global_classifier_id, experiment_id, config_id, {element_column}),
                        {fk_sql}
                    )
                """

                cursor.execute(create_sql)

                # Create indexes
                index_prefix = table_name.replace('ml_classifier_config_', 'idx_clf_config_')
                indexes = [
                    f"CREATE INDEX {index_prefix}_global ON {table_name}(global_classifier_id)",
                    f"CREATE INDEX {index_prefix}_exp ON {table_name}(experiment_id)",
                    f"CREATE INDEX {index_prefix}_config ON {table_name}(config_id)"
                ]

                # Add index on element column for lookup tables
                if foreign_keys:
                    element_short = element_column.replace('_id', '')
                    if element_short.endswith('_function'):
                        element_short = 'func'
                    elif element_short.endswith('_type'):
                        element_short = 'type'
                    elif element_short.endswith('_method'):
                        element_short = 'method'

                    indexes.append(f"CREATE INDEX {index_prefix}_{element_short} ON {table_name}({element_column})")

                for index_sql in indexes:
                    cursor.execute(index_sql)

                tables_created.append(table_name)

            self.db_conn.commit()

            # Print summary
            if tables_created:
                print(f"\n[SUCCESS] Created {len(tables_created)} junction tables")
                for table_name in tables_created:
                    print(f"  ✓ {table_name}")

            if tables_existed:
                print(f"\n[INFO] {len(tables_existed)} tables already exist (use --force to recreate)")
                for table_name in tables_existed:
                    print(f"  - {table_name}")

            if tables_created:
                print("\nJunction tables ready for normalized config storage!")
                print("  - 4-part composite PRIMARY KEY")
                print("  - CASCADE deletes enabled")
                print("  - Foreign key validation")
                print("\nNext step: classifier-config-create (with junction tables)")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to create junction tables: {e}")
            print("  Database rolled back to previous state")

    def cmd_classifier_create_global_config_table(self, args):
        """
        Create the global ml_classifier_configs table

        Usage: classifier-create-global-config-table [--force]

        Creates a single GLOBAL config table for ALL classifiers across ALL experiments.
        This replaces the per-classifier config tables and enables proper CASCADE deletes
        from junction tables.

        Schema:
        - config_id: SERIAL PRIMARY KEY (unique across all configs)
        - global_classifier_id: Links to ml_experiment_classifiers
        - experiment_id: Denormalized for query optimization
        - classifier_id: Denormalized for query optimization
        - config_name: Must be unique within a classifier
        - is_active: Boolean flag
        - created_at, updated_at, notes

        Options:
            --force    Recreate table if it exists

        IMPORTANT: This is a one-time setup command. After creating this table,
        junction tables can have proper FK constraints with CASCADE delete.
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        force = '--force' in args if args else False

        try:
            cursor = self.db_conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = 'ml_classifier_configs'
            """)

            exists = cursor.fetchone()

            if exists and not force:
                print("[INFO] ml_classifier_configs table already exists")
                print("  Use --force to recreate")
                return

            if exists and force:
                print("[WARNING] Dropping existing ml_classifier_configs table...")
                cursor.execute("DROP TABLE ml_classifier_configs CASCADE")
                print("  ✓ Dropped")

            # Create global config table
            print("Creating ml_classifier_configs table...")

            create_sql = """
                CREATE TABLE ml_classifier_configs (
                    config_id SERIAL PRIMARY KEY,
                    global_classifier_id INTEGER NOT NULL,
                    experiment_id INTEGER NOT NULL,
                    classifier_id INTEGER NOT NULL,
                    config_name VARCHAR(255) NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    notes TEXT,
                    FOREIGN KEY (global_classifier_id)
                        REFERENCES ml_experiment_classifiers(global_classifier_id)
                        ON DELETE CASCADE,
                    UNIQUE(global_classifier_id, config_name)
                );

                CREATE INDEX idx_classifier_configs_global ON ml_classifier_configs(global_classifier_id);
                CREATE INDEX idx_classifier_configs_exp ON ml_classifier_configs(experiment_id);
                CREATE INDEX idx_classifier_configs_active ON ml_classifier_configs(is_active);
                CREATE INDEX idx_classifier_configs_lookup ON ml_classifier_configs(experiment_id, classifier_id);
            """

            cursor.execute(create_sql)
            self.db_conn.commit()

            print("\n[SUCCESS] Created ml_classifier_configs table")
            print("  - config_id: SERIAL PRIMARY KEY (global)")
            print("  - Foreign key to ml_experiment_classifiers")
            print("  - Unique constraint on (global_classifier_id, config_name)")
            print("  - Indexes for efficient querying")
            print("\nNext: Update junction tables with FK constraints")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to create global config table: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_migrate_configs_to_global(self, args):
        """
        Migrate configs from per-classifier tables to global ml_classifier_configs table

        Usage: classifier-migrate-configs-to-global [--experiment-id <N>] [--all]

        This command finds existing per-classifier config tables and migrates
        their data to the global ml_classifier_configs table. Run this BEFORE
        adding foreign key constraints.

        Steps:
        1. Find all experiment_NNN_classifier_MMM_config tables
        2. For each config row, copy to ml_classifier_configs
        3. Preserve config_id so junction table data remains valid

        Options:
            --experiment-id <N>    Migrate only configs for specific experiment
            --all                  Migrate configs from all experiments (default)

        IMPORTANT: Run this BEFORE classifier-add-config-foreign-keys!
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        experiment_id = None
        migrate_all = True

        i = 0
        while i < len(args):
            if args[i] == '--experiment-id' and i + 1 < len(args):
                experiment_id = int(args[i + 1])
                migrate_all = False
                i += 2
            elif args[i] == '--all':
                migrate_all = True
                i += 1
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        try:
            cursor = self.db_conn.cursor()

            # Check if global config table exists
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = 'ml_classifier_configs'
            """)

            if not cursor.fetchone():
                print("[ERROR] ml_classifier_configs table does not exist")
                print("  Run: classifier-create-global-config-table")
                return

            # Find all per-classifier config tables
            pattern = 'experiment_%_classifier_%_config'
            if not migrate_all:
                pattern = f'experiment_{experiment_id:03d}_classifier_%_config'

            cursor.execute("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename LIKE %s
                ORDER BY tablename
            """, (pattern,))

            tables = [row[0] for row in cursor.fetchall()]

            if not tables:
                print(f"[INFO] No per-classifier config tables found matching: {pattern}")
                return

            print(f"\n[MIGRATION] Found {len(tables)} config table(s) to migrate:")
            for table in tables:
                print(f"  - {table}")

            # Migrate each table
            total_migrated = 0
            for table_name in tables:
                # Parse experiment_id and classifier_id from table name
                # Format: experiment_041_classifier_001_config
                parts = table_name.split('_')
                exp_id = int(parts[1])
                cls_id = int(parts[3])

                # Get global_classifier_id
                cursor.execute("""
                    SELECT global_classifier_id
                    FROM ml_experiment_classifiers
                    WHERE experiment_id = %s AND classifier_id = %s
                """, (exp_id, cls_id))

                result = cursor.fetchone()
                if not result:
                    print(f"\n[WARNING] Skipping {table_name} - classifier not found")
                    continue

                global_cls_id = result[0]

                # Select all configs from per-classifier table
                cursor.execute(f"""
                    SELECT config_id, config_name, is_active, created_at, updated_at, notes
                    FROM {table_name}
                    ORDER BY config_id
                """)

                configs = cursor.fetchall()

                if not configs:
                    print(f"\n  {table_name}: No configs to migrate")
                    continue

                print(f"\n  Migrating {len(configs)} config(s) from {table_name}...")

                for config_row in configs:
                    config_id, config_name, is_active, created_at, updated_at, notes = config_row

                    # Check if config_id already exists in global table
                    cursor.execute("""
                        SELECT config_id
                        FROM ml_classifier_configs
                        WHERE config_id = %s
                    """, (config_id,))

                    if cursor.fetchone():
                        print(f"    - Config {config_id} ({config_name}): Already exists, skipping")
                        continue

                    # Insert into global table, preserving config_id
                    cursor.execute("""
                        INSERT INTO ml_classifier_configs
                            (config_id, global_classifier_id, experiment_id, classifier_id,
                             config_name, is_active, created_at, updated_at, notes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (config_id, global_cls_id, exp_id, cls_id, config_name,
                          is_active, created_at, updated_at, notes))

                    print(f"    ✓ Migrated config {config_id}: {config_name}")
                    total_migrated += 1

                # Update sequence to prevent conflicts
                cursor.execute("""
                    SELECT setval('ml_classifier_configs_config_id_seq',
                                  (SELECT MAX(config_id) FROM ml_classifier_configs))
                """)

            self.db_conn.commit()

            print(f"\n[SUCCESS] Migration complete")
            print(f"  - Total configs migrated: {total_migrated}")
            print(f"  - Tables processed: {len(tables)}")
            print("\nNext steps:")
            print("  1. Run: classifier-add-config-foreign-keys")
            print("  2. Optionally drop old per-classifier tables")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Migration failed: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_add_config_foreign_keys(self, args):
        """
        Add foreign key constraints from junction tables to ml_classifier_configs

        Usage: classifier-add-config-foreign-keys

        This adds FK constraints with CASCADE delete to all junction tables,
        pointing to the global ml_classifier_configs table. This enables
        automatic cleanup when configs are deleted.

        FK Constraints Added:
        1. ml_classifier_config_decimation_factors -> ml_classifier_configs(config_id)
        2. ml_classifier_config_data_types -> ml_classifier_configs(config_id)
        3. ml_classifier_config_amplitude_methods -> ml_classifier_configs(config_id)
        4. ml_classifier_config_distance_functions -> ml_classifier_configs(config_id)
        5. ml_classifier_config_experiment_feature_sets -> ml_classifier_configs(config_id)
        6. ml_classifier_config_feature_set_features -> ml_classifier_configs(config_id)

        IMPORTANT: Run classifier-create-global-config-table first!
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        try:
            cursor = self.db_conn.cursor()

            # Check if global config table exists
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = 'ml_classifier_configs'
            """)

            if not cursor.fetchone():
                print("[ERROR] ml_classifier_configs table does not exist")
                print("  Run: classifier-create-global-config-table")
                return

            # Define junction tables
            junction_tables = [
                'ml_classifier_config_decimation_factors',
                'ml_classifier_config_data_types',
                'ml_classifier_config_amplitude_methods',
                'ml_classifier_config_distance_functions',
                'ml_classifier_config_experiment_feature_sets',
                'ml_classifier_config_feature_set_features'
            ]

            added = 0
            skipped = 0

            for table_name in junction_tables:
                # Check if FK already exists
                fk_name = f"fk_{table_name}_config"

                cursor.execute("""
                    SELECT constraint_name
                    FROM information_schema.table_constraints
                    WHERE table_name = %s
                    AND constraint_type = 'FOREIGN KEY'
                    AND constraint_name = %s
                """, (table_name, fk_name))

                if cursor.fetchone():
                    print(f"  - {table_name}: FK already exists")
                    skipped += 1
                    continue

                # Add FK constraint
                print(f"  Adding FK to {table_name}...")

                alter_sql = f"""
                    ALTER TABLE {table_name}
                    ADD CONSTRAINT {fk_name}
                    FOREIGN KEY (config_id)
                    REFERENCES ml_classifier_configs(config_id)
                    ON DELETE CASCADE
                """

                cursor.execute(alter_sql)
                print(f"    ✓ Added FK constraint: {fk_name}")
                added += 1

            self.db_conn.commit()

            print(f"\n[SUCCESS] Foreign key constraints updated")
            print(f"  - Added: {added}")
            print(f"  - Skipped (already exists): {skipped}")
            print("\nCASCADE delete now enabled:")
            print("  Deleting a config from ml_classifier_configs will automatically")
            print("  delete all associated entries from junction tables")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to add foreign keys: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_migrate_config_tables(self, args):
        """
        Drop old array-based config tables and recreate with normalized schema

        Usage: classifier-migrate-config-tables [--experiment-id <N>] [--classifier-id <M>] [--all] [--force]

        This command handles migration from array-based config storage to
        normalized junction tables. It drops old config tables that used
        PostgreSQL arrays and recreates them with the new schema (no arrays).

        The new schema stores only metadata (config_name, is_active, etc.)
        while hyperparameters are stored in the 6 global junction tables.

        Options:
            --experiment-id <N>       Target specific experiment (default: current)
            --classifier-id <M>       Target specific classifier (default: current)
            --all                     Migrate all config tables for current experiment
            --force                   Skip confirmation prompt

        Examples:
            classifier-migrate-config-tables                    # Migrate current classifier
            classifier-migrate-config-tables --all              # Migrate all classifiers in experiment
            classifier-migrate-config-tables --experiment-id 41 --classifier-id 1
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        # Parse arguments (args is already a list)
        experiment_id = self.current_experiment
        classifier_id = self.current_classifier_id
        migrate_all = False
        force = False

        i = 0
        while i < len(args):
            if args[i] == '--experiment-id' and i + 1 < len(args):
                experiment_id = int(args[i + 1])
                i += 2
            elif args[i] == '--classifier-id' and i + 1 < len(args):
                classifier_id = int(args[i + 1])
                i += 2
            elif args[i] == '--all':
                migrate_all = True
                i += 1
            elif args[i] == '--force':
                force = True
                i += 1
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if not experiment_id:
            print("[ERROR] No experiment selected. Use 'set experiment <N>' or --experiment-id")
            return

        try:
            cursor = self.db_conn.cursor()

            # Determine which config tables to migrate
            if migrate_all:
                # Get all classifiers for experiment
                cursor.execute("""
                    SELECT experiment_id, classifier_id, classifier_name, global_classifier_id
                    FROM ml_experiment_classifiers
                    WHERE experiment_id = %s
                    ORDER BY classifier_id
                """, (experiment_id,))
                classifiers = cursor.fetchall()

                if not classifiers:
                    print(f"[ERROR] No classifiers found for experiment {experiment_id}")
                    return

                tables_to_migrate = [
                    (row[0], row[1], row[2], row[3])  # exp_id, cls_id, cls_name, global_cls_id
                    for row in classifiers
                ]
            else:
                if not classifier_id:
                    print("[ERROR] No classifier selected. Use 'set classifier <N>' or --classifier-id")
                    return

                # Get single classifier
                cursor.execute("""
                    SELECT experiment_id, classifier_id, classifier_name, global_classifier_id
                    FROM ml_experiment_classifiers
                    WHERE experiment_id = %s AND classifier_id = %s
                """, (experiment_id, classifier_id))

                row = cursor.fetchone()
                if not row:
                    print(f"[ERROR] Classifier {classifier_id} not found for experiment {experiment_id}")
                    return

                tables_to_migrate = [(row[0], row[1], row[2], row[3])]

            # Show migration plan
            print(f"\n[MIGRATION PLAN] Will migrate {len(tables_to_migrate)} config table(s):")
            for exp_id, cls_id, cls_name, global_cls_id in tables_to_migrate:
                table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_config"
                print(f"  - {table_name} (classifier: {cls_name}, global_id: {global_cls_id})")

            # Get confirmation
            if not force:
                print("\n[WARNING] This will DROP existing config tables and recreate with new schema")
                print("          ALL configuration data in these tables will be LOST")
                print("          Junction table data is unaffected")
                print("\nType 'MIGRATE CONFIG' to confirm:")

                confirmation = input().strip()
                if confirmation != 'MIGRATE CONFIG':
                    print("[CANCELLED] Migration aborted")
                    return

            # Migrate each table
            migrated = 0
            for exp_id, cls_id, cls_name, global_cls_id in tables_to_migrate:
                table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_config"

                # Check if table exists
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_name = %s
                """, (table_name,))

                exists = cursor.fetchone()

                if exists:
                    print(f"\n  Dropping {table_name}...")
                    cursor.execute(f"DROP TABLE {table_name} CASCADE")
                    print("    ✓ Dropped")

                # Create new schema (simplified, no arrays)
                print(f"  Creating {table_name} (new schema)...")

                create_sql = f"""
                    CREATE TABLE {table_name} (
                        config_id SERIAL PRIMARY KEY,
                        global_classifier_id INTEGER NOT NULL,
                        experiment_id INTEGER NOT NULL,
                        config_name VARCHAR(255) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        notes TEXT,
                        FOREIGN KEY (global_classifier_id)
                            REFERENCES ml_experiment_classifiers(global_classifier_id)
                            ON DELETE CASCADE,
                        UNIQUE(global_classifier_id, config_name)
                    )
                """

                cursor.execute(create_sql)

                # Create indexes
                cursor.execute(f"""
                    CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_config_active
                    ON {table_name}(is_active)
                """)

                cursor.execute(f"""
                    CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_config_global
                    ON {table_name}(global_classifier_id)
                """)

                print("    ✓ Created with new schema")
                migrated += 1

            self.db_conn.commit()

            print(f"\n[SUCCESS] Migrated {migrated} config table(s)")
            print("  - Array columns removed")
            print("  - global_classifier_id added")
            print("  - Foreign key constraints enabled")
            print("  - Ready for junction table-based config creation")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Migration failed: {e}")
            print("  Database rolled back to previous state")

    def cmd_classifier_new(self, args):
        """
        Create new classifier instance for current experiment

        Usage: classifier-new --name <name> [OPTIONS]

        Options:
            --name <name>              Unique classifier name (required)
            --description <desc>       Detailed description
            --type <type>              Classifier type (default: svm)
            --auto-select              Auto-select after creation (default: True)
            --no-auto-select           Do not auto-select after creation

        Example:
            classifier-new --name "baseline_svm" --description "Baseline configuration"
            classifier-new --name "multi_decimation" --type svm --no-auto-select
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        # Parse arguments
        name = None
        description = None
        classifier_type = 'svm'
        auto_select = True

        i = 0
        while i < len(args):
            if args[i] == '--name' and i + 1 < len(args):
                name = args[i + 1]
                i += 2
            elif args[i] == '--description' and i + 1 < len(args):
                description = args[i + 1]
                i += 2
            elif args[i] == '--type' and i + 1 < len(args):
                classifier_type = args[i + 1]
                i += 2
            elif args[i] == '--auto-select':
                auto_select = True
                i += 1
            elif args[i] == '--no-auto-select':
                auto_select = False
                i += 1
            elif args[i] == '--help':
                print("\nUsage: classifier-new --name <name> [OPTIONS]")
                print("\nOptions:")
                print("  --name <name>              Unique classifier name (required)")
                print("  --description <desc>       Detailed description")
                print("  --type <type>              Classifier type (default: svm)")
                print("  --auto-select              Auto-select after creation (default)")
                print("  --no-auto-select           Do not auto-select after creation")
                print("\nExamples:")
                print("  classifier-new --name \"baseline_svm\" --description \"Baseline config\"")
                print("  classifier-new --name \"multi_decimation\" --type svm")
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        # Validate required arguments
        if not name:
            print("[ERROR] --name is required")
            print("Usage: classifier-new --name <name> [OPTIONS]")
            print("Try: classifier-new --help")
            return

        try:
            cursor = self.db_conn.cursor()

            # Get next available classifier_id for this experiment
            cursor.execute("""
                SELECT COALESCE(MAX(classifier_id), 0) + 1 as next_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s
            """, (self.current_experiment,))
            next_id = cursor.fetchone()[0]

            # Check if name is unique for this experiment
            cursor.execute("""
                SELECT COUNT(*) FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_name = %s
            """, (self.current_experiment, name))
            if cursor.fetchone()[0] > 0:
                print(f"[ERROR] Classifier name '{name}' already exists for experiment {self.current_experiment}")
                print("       Use a different name or remove the existing classifier")
                return

            # Insert new classifier and get global_classifier_id
            cursor.execute("""
                INSERT INTO ml_experiment_classifiers
                    (experiment_id, classifier_id, classifier_name, classifier_description, classifier_type)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING global_classifier_id
            """, (self.current_experiment, next_id, name, description, classifier_type))

            global_classifier_id = cursor.fetchone()[0]

            self.db_conn.commit()

            print(f"[SUCCESS] Created classifier {next_id} for experiment {self.current_experiment}")
            print(f"  - Name: {name}")
            print(f"  - Type: {classifier_type}")
            print(f"  - Global Classifier ID: {global_classifier_id}")
            if description:
                print(f"  - Description: {description}")

            # Auto-select if requested
            if auto_select:
                self.current_classifier_id = next_id
                self.current_classifier_name = name
                print(f"[INFO] Auto-selected classifier {next_id}")
                print(f"       Prompt will update on next command")

        except Exception as e:
            self.db_conn.rollback()
            print(f"[ERROR] Failed to create classifier: {e}")

    def cmd_classifier_remove(self, args):
        """
        Remove classifier instance (delete or archive)

        Usage: classifier-remove --classifier-id <id> --confirm [OPTIONS]

        Options:
            --classifier-id <id>       Classifier ID to remove (required)
            --confirm                  Confirm deletion (required)
            --archive-instead          Archive instead of delete

        Safety:
            - Requires explicit --confirm flag
            - Shows all tables that will be deleted
            - Requires typed confirmation "DELETE CLASSIFIER <N>"
            - Option to archive for safety

        Example:
            classifier-remove --classifier-id 2 --confirm
            classifier-remove --classifier-id 1 --confirm --archive-instead
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        # Parse arguments
        classifier_id = None
        confirm = False
        archive_instead = False

        i = 0
        while i < len(args):
            if args[i] == '--classifier-id' and i + 1 < len(args):
                try:
                    classifier_id = int(args[i + 1])
                except ValueError:
                    print(f"[ERROR] Invalid classifier ID: {args[i + 1]}")
                    return
                i += 2
            elif args[i] == '--confirm':
                confirm = True
                i += 1
            elif args[i] == '--archive-instead':
                archive_instead = True
                i += 1
            elif args[i] == '--help':
                print(self.cmd_classifier_remove.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        # Validation
        if classifier_id is None:
            print("[ERROR] --classifier-id is required")
            print("Usage: classifier-remove --classifier-id <id> --confirm [OPTIONS]")
            return

        if not confirm:
            print("[ERROR] --confirm flag is required for safety")
            print("Usage: classifier-remove --classifier-id <id> --confirm [OPTIONS]")
            return

        try:
            cursor = self.db_conn.cursor()

            # Check if classifier exists
            cursor.execute("""
                SELECT classifier_name, classifier_type, is_archived
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (self.current_experiment, classifier_id))
            result = cursor.fetchone()

            if not result:
                print(f"[ERROR] Classifier {classifier_id} not found for experiment {self.current_experiment}")
                return

            classifier_name, classifier_type, is_archived = result

            # Query for related tables
            cursor.execute("""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
                AND tablename LIKE %s
                ORDER BY tablename
            """, (f"experiment_{self.current_experiment}_classifier_{classifier_id}_%",))

            related_tables = [row[0] for row in cursor.fetchall()]

            # Display warning
            print("\n" + "=" * 80)
            if archive_instead:
                print("[WARNING] ARCHIVE CLASSIFIER OPERATION")
            else:
                print("[WARNING] DELETE CLASSIFIER OPERATION")
            print("=" * 80)
            print(f"Experiment: {self.current_experiment}")
            print(f"Classifier ID: {classifier_id}")
            print(f"Classifier Name: {classifier_name}")
            print(f"Classifier Type: {classifier_type}")
            print(f"Currently Archived: {is_archived}")
            print()

            if archive_instead:
                print("This operation will:")
                print("  - Set is_archived = TRUE in ml_experiment_classifiers")
                print("  - Preserve all data and tables")
                print("  - Classifier will be hidden from default listings")
                print()
            else:
                print("This operation will:")
                print("  - DELETE the classifier from ml_experiment_classifiers")
                print("  - CASCADE DELETE all related tables")
                print("  - ALL DATA WILL BE PERMANENTLY LOST")
                print()

                if related_tables:
                    print(f"Related tables to be deleted ({len(related_tables)}):")
                    for table in related_tables:
                        print(f"  - {table}")
                else:
                    print("No related tables found (safe to delete)")
                print()

            # Require typed confirmation
            if archive_instead:
                required_text = f"ARCHIVE CLASSIFIER {classifier_id}"
            else:
                required_text = f"DELETE CLASSIFIER {classifier_id}"

            print(f"To proceed, type: {required_text}")
            print("=" * 80)

            # Get confirmation from user
            try:
                user_input = input("Confirmation: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[INFO] Operation cancelled")
                return

            if user_input != required_text:
                print("[ERROR] Confirmation text does not match. Operation cancelled.")
                print(f"Expected: {required_text}")
                print(f"Received: {user_input}")
                return

            # Perform operation
            if archive_instead:
                cursor.execute("""
                    UPDATE ml_experiment_classifiers
                    SET is_archived = TRUE, updated_at = NOW()
                    WHERE experiment_id = %s AND classifier_id = %s
                """, (self.current_experiment, classifier_id))
                self.db_conn.commit()
                print(f"[SUCCESS] Classifier {classifier_id} archived successfully")
                print("  - is_archived set to TRUE")
                print("  - All data preserved")
            else:
                cursor.execute("""
                    DELETE FROM ml_experiment_classifiers
                    WHERE experiment_id = %s AND classifier_id = %s
                """, (self.current_experiment, classifier_id))
                self.db_conn.commit()
                print(f"[SUCCESS] Classifier {classifier_id} deleted successfully")
                if related_tables:
                    print(f"  - Deleted {len(related_tables)} related tables")

            # Deselect if currently selected classifier removed
            if self.current_classifier_id == classifier_id:
                self.current_classifier_id = None
                self.current_classifier_name = None
                print("[INFO] Deselected removed classifier")
                print("       Prompt will update on next command")

        except Exception as e:
            self.db_conn.rollback()
            print(f"[ERROR] Failed to remove classifier: {e}")

    def cmd_classifier_list(self, args):
        """
        List all classifiers for current experiment

        Usage: classifier-list [OPTIONS]

        Options:
            --include-archived         Show archived classifiers
            --show-tables              Show table names for each classifier

        Example:
            classifier-list
            classifier-list --include-archived
            classifier-list --show-tables
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        # Parse arguments
        include_archived = False
        show_tables = False

        i = 0
        while i < len(args):
            if args[i] == '--include-archived':
                include_archived = True
                i += 1
            elif args[i] == '--show-tables':
                show_tables = True
                i += 1
            elif args[i] == '--help':
                print(self.cmd_classifier_list.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        try:
            cursor = self.db_conn.cursor()

            # Build query
            where_clause = "WHERE experiment_id = %s"
            params = [self.current_experiment]

            if not include_archived:
                where_clause += " AND is_archived = FALSE"

            query = f"""
                SELECT
                    classifier_id,
                    classifier_name,
                    classifier_type,
                    is_active,
                    is_archived,
                    created_at,
                    classifier_description
                FROM ml_experiment_classifiers
                {where_clause}
                ORDER BY classifier_id
            """

            cursor.execute(query, params)
            results = cursor.fetchall()

            if not results:
                if include_archived:
                    print(f"[INFO] No classifiers found for experiment {self.current_experiment}")
                else:
                    print(f"[INFO] No active classifiers found for experiment {self.current_experiment}")
                    print("       Use --include-archived to show archived classifiers")
                return

            # Prepare table data
            table_data = []
            for row in results:
                classifier_id, name, ctype, is_active, is_archived, created_at, description = row

                # Format status
                status_parts = []
                if is_active:
                    status_parts.append("Active")
                if is_archived:
                    status_parts.append("Archived")
                if not status_parts:
                    status_parts.append("Inactive")
                status = ", ".join(status_parts)

                # Format created date
                created_str = created_at.strftime("%Y-%m-%d %H:%M") if created_at else "N/A"

                # Mark currently selected classifier
                selected = "[SELECTED]" if self.current_classifier_id == classifier_id else ""

                row_data = [
                    classifier_id,
                    name,
                    ctype,
                    status,
                    created_str,
                    selected
                ]

                # Add table count if requested
                if show_tables:
                    cursor.execute("""
                        SELECT COUNT(*)
                        FROM pg_tables
                        WHERE schemaname = 'public'
                        AND tablename LIKE %s
                    """, (f"experiment_{self.current_experiment}_classifier_{classifier_id}_%",))
                    table_count = cursor.fetchone()[0]
                    row_data.append(table_count)

                table_data.append(row_data)

            # Print table
            headers = ["ID", "Name", "Type", "Status", "Created", "Selected"]
            if show_tables:
                headers.append("Tables")

            print()
            print(f"Classifiers for experiment {self.current_experiment}:")
            print()
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print()
            print(f"Total: {len(results)} classifier(s)")

            if not include_archived:
                # Check if there are archived classifiers
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM ml_experiment_classifiers
                    WHERE experiment_id = %s AND is_archived = TRUE
                """, (self.current_experiment,))
                archived_count = cursor.fetchone()[0]
                if archived_count > 0:
                    print(f"Note: {archived_count} archived classifier(s) hidden. Use --include-archived to show.")

        except Exception as e:
            print(f"[ERROR] Failed to list classifiers: {e}")

    def cmd_classifier_config_create(self, args):
        """
        Create new configuration for currently selected classifier (JUNCTION TABLE VERSION)

        Usage: classifier-config-create --config-name <name> [OPTIONS]

        This command stores configuration metadata in the per-classifier config table
        and hyperparameter values in the 6 global junction tables.

        Options:
            --config-name <name>           Unique configuration name (required)
            --decimation-factors <list>    Comma-separated or 'all' (default: 0)
            --data-types <list>            Comma-separated names/IDs or 'all' (default: adc12)
            --amplitude-methods <list>     Comma-separated IDs or 'all' (default: 1)
            --feature-sets <list>          Comma-separated IDs or 'all' (default: all)
            --features <list>              Comma-separated IDs or 'all' (default: all)
            --distance-functions <list>    Comma-separated names or 'all' (default: all)
            --set-active                   Set as active configuration
            --notes <text>                 Configuration description

        Example:
            classifier-config-create --config-name "baseline" --decimation-factors 0,7,15 \
                --data-types adc6,adc8,adc10,adc12 --distance-functions l1,l2,wasserstein --set-active
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments (args is already a list)
        config_name = None
        decimation_factors = '0'
        data_types = 'adc12'
        amplitude_methods = '1'
        feature_sets = 'all'
        features = 'all'
        distance_functions = 'all'
        set_active = False
        notes = None

        i = 0
        while i < len(args):
            if args[i] == '--config-name' and i + 1 < len(args):
                config_name = args[i + 1]
                i += 2
            elif args[i] == '--decimation-factors' and i + 1 < len(args):
                decimation_factors = args[i + 1]
                i += 2
            elif args[i] == '--data-types' and i + 1 < len(args):
                data_types = args[i + 1]
                i += 2
            elif args[i] == '--amplitude-methods' and i + 1 < len(args):
                amplitude_methods = args[i + 1]
                i += 2
            elif args[i] == '--feature-sets' and i + 1 < len(args):
                feature_sets = args[i + 1]
                i += 2
            elif args[i] == '--features' and i + 1 < len(args):
                features = args[i + 1]
                i += 2
            elif args[i] == '--distance-functions' and i + 1 < len(args):
                distance_functions = args[i + 1]
                i += 2
            elif args[i] == '--notes' and i + 1 < len(args):
                notes = args[i + 1]
                i += 2
            elif args[i] == '--set-active':
                set_active = True
                i += 1
            elif args[i] == '--help':
                print(self.cmd_classifier_config_create.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if not config_name:
            print("[ERROR] --config-name is required")
            print("Usage: classifier-config-create --config-name <name> [OPTIONS]")
            return

        try:
            cursor = self.db_conn.cursor()

            # STEP 1: Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (self.current_experiment, self.current_classifier_id))

            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {self.current_classifier_id} not found")
                return

            global_classifier_id = result[0]

            # STEP 2: Use global ml_classifier_configs table (junction tables reference this)
            table_name = "ml_classifier_configs"

            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = %s
                )
            """, (table_name,))

            if not cursor.fetchone()[0]:
                # Create config table with NEW schema (no arrays)
                create_table_sql = f"""
                    CREATE TABLE {table_name} (
                        config_id SERIAL PRIMARY KEY,
                        global_classifier_id INTEGER NOT NULL,
                        experiment_id INTEGER NOT NULL,
                        config_name VARCHAR(255) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        notes TEXT,
                        FOREIGN KEY (global_classifier_id)
                            REFERENCES ml_experiment_classifiers(global_classifier_id)
                            ON DELETE CASCADE,
                        UNIQUE(global_classifier_id, config_name)
                    );

                    CREATE INDEX idx_exp{self.current_experiment:03d}_cls{self.current_classifier_id:03d}_config_active
                        ON {table_name}(is_active);
                    CREATE INDEX idx_exp{self.current_experiment:03d}_cls{self.current_classifier_id:03d}_config_global
                        ON {table_name}(global_classifier_id);
                """
                cursor.execute(create_table_sql)
                print(f"[INFO] Created config table: {table_name}")

            # STEP 3: Process decimation factors
            decimation_factors_list = []
            if decimation_factors.lower() != 'all':
                try:
                    decimation_factors_list = [int(x.strip()) for x in decimation_factors.split(',')]
                except ValueError:
                    print(f"[ERROR] Invalid decimation factors: {decimation_factors}")
                    return

            # STEP 4: Process data types (convert names to IDs)
            data_type_ids_list = []
            if data_types.lower() != 'all':
                # Try to parse as integers first
                try:
                    data_type_ids_list = [int(x.strip()) for x in data_types.split(',')]
                except ValueError:
                    # Names like 'adc6', 'adc12' - look up IDs
                    cursor.execute("""
                        SELECT edt.data_type_id, dt.data_type_name
                        FROM ml_experiments_data_types edt
                        JOIN ml_data_types_lut dt ON edt.data_type_id = dt.data_type_id
                        WHERE edt.experiment_id = %s
                    """, (self.current_experiment,))

                    available_types = {row[1]: row[0] for row in cursor.fetchall()}

                    if not available_types:
                        print(f"[ERROR] No data types configured for experiment {self.current_experiment}")
                        return

                    requested_types = [name.strip() for name in data_types.split(',')]
                    for name in requested_types:
                        if name not in available_types:
                            print(f"[ERROR] Data type '{name}' not configured")
                            print(f"Available: {', '.join(available_types.keys())}")
                            return
                        data_type_ids_list.append(available_types[name])

            # STEP 5: Process distance functions (convert names to IDs)
            distance_function_ids_list = []
            if distance_functions.lower() != 'all':
                cursor.execute("""
                    SELECT df.distance_function_id, df.function_name
                    FROM ml_experiments_distance_measurements edm
                    JOIN ml_distance_functions_lut df ON edm.distance_function_id = df.distance_function_id
                    WHERE edm.experiment_id = %s
                """, (self.current_experiment,))

                available_functions = {row[1]: row[0] for row in cursor.fetchall()}

                if not available_functions:
                    print(f"[ERROR] No distance functions configured for experiment {self.current_experiment}")
                    return

                requested_names = [name.strip() for name in distance_functions.split(',')]
                for name in requested_names:
                    if name not in available_functions:
                        print(f"[ERROR] Distance function '{name}' not configured")
                        print(f"Available: {', '.join(available_functions.keys())}")
                        return
                    distance_function_ids_list.append(available_functions[name])

            # STEP 6: Process amplitude methods
            amplitude_methods_list = []
            if amplitude_methods.lower() != 'all':
                try:
                    amplitude_methods_list = [int(x.strip()) for x in amplitude_methods.split(',')]
                except ValueError:
                    print(f"[ERROR] Invalid amplitude methods: {amplitude_methods}")
                    return

            # STEP 6b: Process feature sets
            feature_sets_list = []
            if feature_sets.lower() == 'all':
                # Query all feature sets for this experiment
                cursor.execute("""
                    SELECT experiment_feature_set_id
                    FROM ml_experiments_feature_sets
                    WHERE experiment_id = %s
                    ORDER BY experiment_feature_set_id
                """, (self.current_experiment,))

                feature_sets_list = [row[0] for row in cursor.fetchall()]

                if not feature_sets_list:
                    print(f"[ERROR] No feature sets configured for experiment {self.current_experiment}")
                    print("  You need to configure feature sets before creating classifier configs")
                    return
            else:
                # Parse specific feature set IDs
                try:
                    feature_sets_list = [int(x.strip()) for x in feature_sets.split(',')]
                except ValueError:
                    print(f"[ERROR] Invalid feature sets: {feature_sets}")
                    return

                # Validate feature sets exist for this experiment
                for fs_id in feature_sets_list:
                    cursor.execute("""
                        SELECT 1
                        FROM ml_experiments_feature_sets
                        WHERE experiment_id = %s AND experiment_feature_set_id = %s
                    """, (self.current_experiment, fs_id))

                    if not cursor.fetchone():
                        print(f"[ERROR] Feature set {fs_id} not configured for experiment {self.current_experiment}")
                        return

            # STEP 7: If set_active, deactivate all other configs
            if set_active:
                cursor.execute(f"""
                    UPDATE {table_name}
                    SET is_active = FALSE, updated_at = NOW()
                    WHERE global_classifier_id = %s
                """, (global_classifier_id,))

            # STEP 8: Insert config row
            cursor.execute(f"""
                INSERT INTO {table_name}
                    (global_classifier_id, experiment_id, classifier_id, config_name, is_active, notes)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING config_id
            """, (global_classifier_id, self.current_experiment, self.current_classifier_id, config_name, set_active, notes))

            config_id = cursor.fetchone()[0]

            # STEP 9: Insert into junction tables
            inserted_counts = {}

            # Insert decimation factors
            if decimation_factors_list:
                for df in decimation_factors_list:
                    cursor.execute("""
                        INSERT INTO ml_classifier_config_decimation_factors
                            (global_classifier_id, experiment_id, config_id, decimation_factor)
                        VALUES (%s, %s, %s, %s)
                    """, (global_classifier_id, self.current_experiment, config_id, df))
                inserted_counts['decimation_factors'] = len(decimation_factors_list)

            # Insert data types
            if data_type_ids_list:
                for dt_id in data_type_ids_list:
                    cursor.execute("""
                        INSERT INTO ml_classifier_config_data_types
                            (global_classifier_id, experiment_id, config_id, data_type_id)
                        VALUES (%s, %s, %s, %s)
                    """, (global_classifier_id, self.current_experiment, config_id, dt_id))
                inserted_counts['data_types'] = len(data_type_ids_list)

            # Insert amplitude methods
            if amplitude_methods_list:
                for am_id in amplitude_methods_list:
                    cursor.execute("""
                        INSERT INTO ml_classifier_config_amplitude_methods
                            (global_classifier_id, experiment_id, config_id, amplitude_processing_method_id)
                        VALUES (%s, %s, %s, %s)
                    """, (global_classifier_id, self.current_experiment, config_id, am_id))
                inserted_counts['amplitude_methods'] = len(amplitude_methods_list)

            # Insert distance functions
            if distance_function_ids_list:
                for dfunc_id in distance_function_ids_list:
                    cursor.execute("""
                        INSERT INTO ml_classifier_config_distance_functions
                            (global_classifier_id, experiment_id, config_id, distance_function_id)
                        VALUES (%s, %s, %s, %s)
                    """, (global_classifier_id, self.current_experiment, config_id, dfunc_id))
                inserted_counts['distance_functions'] = len(distance_function_ids_list)

            # Insert feature sets
            if feature_sets_list:
                for fs_id in feature_sets_list:
                    cursor.execute("""
                        INSERT INTO ml_classifier_config_experiment_feature_sets
                            (global_classifier_id, experiment_id, config_id, experiment_feature_set_id)
                        VALUES (%s, %s, %s, %s)
                    """, (global_classifier_id, self.current_experiment, config_id, fs_id))
                inserted_counts['feature_sets'] = len(feature_sets_list)

            self.db_conn.commit()

            # Success message
            print(f"\n[SUCCESS] Created configuration '{config_name}'")
            print(f"  Config ID: {config_id}")
            print(f"  Global Classifier ID: {global_classifier_id}")
            print(f"  Experiment ID: {self.current_experiment}")
            print(f"  Classifier ID: {self.current_classifier_id}")
            print("\nHyperparameters stored in junction tables:")
            print(f"  - Decimation factors: {decimation_factors} ({inserted_counts.get('decimation_factors', 0)} values)")
            print(f"  - Data types: {data_types} ({inserted_counts.get('data_types', 0)} values)")
            print(f"  - Amplitude methods: {amplitude_methods} ({inserted_counts.get('amplitude_methods', 0)} values)")
            print(f"  - Distance functions: {distance_functions} ({inserted_counts.get('distance_functions', 0)} values)")
            print(f"  - Feature sets: {feature_sets} ({inserted_counts.get('feature_sets', 0)} values)")

            if set_active:
                print("\n  Status: ACTIVE configuration")
            if notes:
                print(f"  Notes: {notes}")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to create configuration: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_config_add_feature_sets(self, args):
        """
        Add feature sets to an existing configuration

        Usage: classifier-config-add-feature-sets --config-id <id> --feature-sets <list>

        This command adds feature sets to a configuration by looking up the
        experiment_feature_set_ids from feature_set_ids and inserting into
        the ml_classifier_config_experiment_feature_sets junction table.

        Options:
            --config-id <id>           Config ID (required)
            --feature-sets <list>      Comma-separated feature_set_ids from ml_feature_sets_lut (required)
            --experiment-id <id>       Experiment ID (default: current experiment)

        Examples:
            classifier-config-add-feature-sets --config-id 1 --feature-sets 1,2,5
            classifier-config-add-feature-sets --config-id 1 --feature-sets 3 --experiment-id 41
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        config_id = None
        feature_sets = None
        experiment_id = self.current_experiment

        i = 0
        while i < len(args):
            if args[i] == '--config-id' and i + 1 < len(args):
                config_id = int(args[i + 1])
                i += 2
            elif args[i] == '--feature-sets' and i + 1 < len(args):
                feature_sets = args[i + 1]
                i += 2
            elif args[i] == '--experiment-id' and i + 1 < len(args):
                experiment_id = int(args[i + 1])
                i += 2
            elif args[i] == '--help':
                print(self.cmd_classifier_config_add_feature_sets.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if not config_id:
            print("[ERROR] --config-id is required")
            return

        if not feature_sets:
            print("[ERROR] --feature-sets is required")
            return

        if not experiment_id:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' or --experiment-id")
            return

        try:
            cursor = self.db_conn.cursor()

            # Parse feature set IDs
            feature_set_ids = [int(x.strip()) for x in feature_sets.split(',')]

            # Get config info including global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id, experiment_id, classifier_id, config_name
                FROM ml_classifier_configs
                WHERE config_id = %s
            """, (config_id,))

            config_row = cursor.fetchone()
            if not config_row:
                print(f"[ERROR] Config ID {config_id} not found in ml_classifier_configs")
                return

            global_classifier_id, config_exp_id, classifier_id, config_name = config_row

            # Verify experiment matches
            if config_exp_id != experiment_id:
                print(f"[ERROR] Config {config_id} belongs to experiment {config_exp_id}, not {experiment_id}")
                return

            print(f"\n[INFO] Adding feature sets to config '{config_name}' (ID: {config_id})")
            print(f"  Experiment ID: {experiment_id}")
            print(f"  Global Classifier ID: {global_classifier_id}")
            print(f"  Requested feature_set_ids: {feature_set_ids}")

            # Look up experiment_feature_set_ids
            cursor.execute("""
                SELECT efs.experiment_feature_set_id, efs.feature_set_id, fs.feature_set_name
                FROM ml_experiments_feature_sets efs
                JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
                WHERE efs.experiment_id = %s
                AND efs.feature_set_id = ANY(%s)
                ORDER BY efs.feature_set_id
            """, (experiment_id, feature_set_ids))

            feature_set_mappings = cursor.fetchall()

            if not feature_set_mappings:
                print(f"\n[ERROR] No feature sets found for experiment {experiment_id}")
                print(f"  Requested: {feature_set_ids}")
                return

            found_ids = {row[1] for row in feature_set_mappings}
            missing_ids = set(feature_set_ids) - found_ids

            if missing_ids:
                print(f"\n[WARNING] Feature set IDs not configured for experiment {experiment_id}: {sorted(missing_ids)}")

            # Insert feature sets
            inserted = 0
            skipped = 0

            print(f"\nInserting feature sets into junction table...")
            for exp_fs_id, fs_id, fs_name in feature_set_mappings:
                # Check if already exists
                cursor.execute("""
                    SELECT 1
                    FROM ml_classifier_config_experiment_feature_sets
                    WHERE global_classifier_id = %s
                    AND experiment_id = %s
                    AND config_id = %s
                    AND experiment_feature_set_id = %s
                """, (global_classifier_id, experiment_id, config_id, exp_fs_id))

                if cursor.fetchone():
                    print(f"  - Feature set {fs_id} ({fs_name}): Already exists, skipping")
                    skipped += 1
                    continue

                # Insert
                cursor.execute("""
                    INSERT INTO ml_classifier_config_experiment_feature_sets
                        (global_classifier_id, experiment_id, config_id, experiment_feature_set_id)
                    VALUES (%s, %s, %s, %s)
                """, (global_classifier_id, experiment_id, config_id, exp_fs_id))

                print(f"  ✓ Added feature set {fs_id} ({fs_name})")
                inserted += 1

            self.db_conn.commit()

            print(f"\n[SUCCESS] Feature sets updated")
            print(f"  - Inserted: {inserted}")
            print(f"  - Skipped (already exists): {skipped}")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to add feature sets: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_config_add_hyperparameters(self, args):
        """
        Add hyperparameters to an existing configuration

        Usage: classifier-config-add-hyperparameters --config <name|id> [OPTIONS]

        Adds hyperparameters to an existing configuration by inserting into
        the appropriate junction tables. Can add multiple types at once.

        Options:
            --config <name|id>            Config name or ID (required)
            --amplitude-methods <list>    Comma-separated amplitude method IDs
            --decimation-factors <list>   Comma-separated decimation factors
            --data-types <list>           Comma-separated data type names or IDs
            --distance-functions <list>   Comma-separated distance function names
            --experiment-feature-sets <list>  Comma-separated experiment feature set IDs

        Examples:
            classifier-config-add-hyperparameters --config baseline --amplitude-methods 2
            classifier-config-add-hyperparameters --config 1 --amplitude-methods 2,3
            classifier-config-add-hyperparameters --config baseline --decimation-factors 31,63
            classifier-config-add-hyperparameters --config baseline --data-types adc2,adc3
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        config_name_or_id = None
        amplitude_methods = None
        decimation_factors = None
        data_types = None
        distance_functions = None
        experiment_feature_sets = None

        i = 0
        while i < len(args):
            if args[i] == '--config' and i + 1 < len(args):
                config_name_or_id = args[i + 1]
                i += 2
            elif args[i] == '--amplitude-methods' and i + 1 < len(args):
                amplitude_methods = args[i + 1]
                i += 2
            elif args[i] == '--decimation-factors' and i + 1 < len(args):
                decimation_factors = args[i + 1]
                i += 2
            elif args[i] == '--data-types' and i + 1 < len(args):
                data_types = args[i + 1]
                i += 2
            elif args[i] == '--distance-functions' and i + 1 < len(args):
                distance_functions = args[i + 1]
                i += 2
            elif args[i] == '--experiment-feature-sets' and i + 1 < len(args):
                experiment_feature_sets = args[i + 1]
                i += 2
            elif args[i] == '--help':
                print(self.cmd_classifier_config_add_hyperparameters.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if not config_name_or_id:
            print("[ERROR] --config is required")
            return

        if not any([amplitude_methods, decimation_factors, data_types, distance_functions, experiment_feature_sets]):
            print("[ERROR] At least one hyperparameter type must be specified")
            print("Options: --amplitude-methods, --decimation-factors, --data-types,")
            print("         --distance-functions, --experiment-feature-sets")
            return

        try:
            cursor = self.db_conn.cursor()

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (self.current_experiment, self.current_classifier_id))
            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {self.current_classifier_id} not found")
                return
            global_classifier_id = result[0]

            # Resolve config_id from name or ID
            if config_name_or_id.isdigit():
                config_id = int(config_name_or_id)
            else:
                cursor.execute("""
                    SELECT config_id FROM ml_classifier_configs
                    WHERE global_classifier_id = %s AND config_name = %s
                """, (global_classifier_id, config_name_or_id))
                result = cursor.fetchone()
                if not result:
                    print(f"[ERROR] Configuration '{config_name_or_id}' not found")
                    return
                config_id = result[0]

            added_count = 0

            # Add amplitude methods
            if amplitude_methods:
                amp_ids = [int(x.strip()) for x in amplitude_methods.split(',')]
                for amp_id in amp_ids:
                    # Check if already exists
                    cursor.execute("""
                        SELECT 1 FROM ml_classifier_config_amplitude_methods
                        WHERE config_id = %s AND amplitude_processing_method_id = %s
                    """, (config_id, amp_id))
                    if cursor.fetchone():
                        print(f"[INFO] Amplitude method {amp_id} already exists in config, skipping")
                        continue

                    cursor.execute("""
                        INSERT INTO ml_classifier_config_amplitude_methods
                            (global_classifier_id, experiment_id, config_id, amplitude_processing_method_id)
                        VALUES (%s, %s, %s, %s)
                    """, (global_classifier_id, self.current_experiment, config_id, amp_id))
                    added_count += 1
                    print(f"[SUCCESS] Added amplitude method {amp_id} to configuration")

            # Add decimation factors
            if decimation_factors:
                dec_values = [int(x.strip()) for x in decimation_factors.split(',')]
                for dec in dec_values:
                    # Check if already exists
                    cursor.execute("""
                        SELECT 1 FROM ml_classifier_config_decimation_factors
                        WHERE config_id = %s AND decimation_factor = %s
                    """, (config_id, dec))
                    if cursor.fetchone():
                        print(f"[INFO] Decimation factor {dec} already exists in config, skipping")
                        continue

                    cursor.execute("""
                        INSERT INTO ml_classifier_config_decimation_factors
                            (global_classifier_id, experiment_id, config_id, decimation_factor)
                        VALUES (%s, %s, %s, %s)
                    """, (global_classifier_id, self.current_experiment, config_id, dec))
                    added_count += 1
                    print(f"[SUCCESS] Added decimation factor {dec} to configuration")

            # Add data types
            if data_types:
                dt_list = [x.strip() for x in data_types.split(',')]
                for dt in dt_list:
                    # Try to parse as ID or name
                    if dt.isdigit():
                        dt_id = int(dt)
                    else:
                        # Look up data type ID by name
                        cursor.execute("""
                            SELECT data_type_id FROM experiment_data_types
                            WHERE experiment_id = %s AND data_type_name = %s
                        """, (self.current_experiment, dt))
                        result = cursor.fetchone()
                        if not result:
                            print(f"[WARNING] Data type '{dt}' not found for experiment {self.current_experiment}, skipping")
                            continue
                        dt_id = result[0]

                    # Check if already exists
                    cursor.execute("""
                        SELECT 1 FROM ml_classifier_config_data_types
                        WHERE config_id = %s AND data_type_id = %s
                    """, (config_id, dt_id))
                    if cursor.fetchone():
                        print(f"[INFO] Data type {dt_id} already exists in config, skipping")
                        continue

                    cursor.execute("""
                        INSERT INTO ml_classifier_config_data_types
                            (global_classifier_id, experiment_id, config_id, data_type_id)
                        VALUES (%s, %s, %s, %s)
                    """, (global_classifier_id, self.current_experiment, config_id, dt_id))
                    added_count += 1
                    print(f"[SUCCESS] Added data type {dt_id} to configuration")

            # Add distance functions
            if distance_functions:
                df_names = [x.strip() for x in distance_functions.split(',')]
                for df_name in df_names:
                    # Look up distance function ID
                    cursor.execute("""
                        SELECT function_id FROM distance_functions
                        WHERE function_name = %s
                    """, (df_name,))
                    result = cursor.fetchone()
                    if not result:
                        print(f"[WARNING] Distance function '{df_name}' not found, skipping")
                        continue
                    df_id = result[0]

                    # Check if already exists
                    cursor.execute("""
                        SELECT 1 FROM ml_classifier_config_distance_functions
                        WHERE config_id = %s AND distance_function_id = %s
                    """, (config_id, df_id))
                    if cursor.fetchone():
                        print(f"[INFO] Distance function '{df_name}' already exists in config, skipping")
                        continue

                    cursor.execute("""
                        INSERT INTO ml_classifier_config_distance_functions
                            (global_classifier_id, experiment_id, config_id, distance_function_id)
                        VALUES (%s, %s, %s, %s)
                    """, (global_classifier_id, self.current_experiment, config_id, df_id))
                    added_count += 1
                    print(f"[SUCCESS] Added distance function '{df_name}' to configuration")

            # Add experiment feature sets
            if experiment_feature_sets:
                efs_ids = [int(x.strip()) for x in experiment_feature_sets.split(',')]
                for efs_id in efs_ids:
                    # Check if already exists
                    cursor.execute("""
                        SELECT 1 FROM ml_classifier_config_experiment_feature_sets
                        WHERE config_id = %s AND experiment_feature_set_id = %s
                    """, (config_id, efs_id))
                    if cursor.fetchone():
                        print(f"[INFO] Experiment feature set {efs_id} already exists in config, skipping")
                        continue

                    cursor.execute("""
                        INSERT INTO ml_classifier_config_experiment_feature_sets
                            (global_classifier_id, experiment_id, config_id, experiment_feature_set_id)
                        VALUES (%s, %s, %s, %s)
                    """, (global_classifier_id, self.current_experiment, config_id, efs_id))
                    added_count += 1
                    print(f"[SUCCESS] Added experiment feature set {efs_id} to configuration")

            self.db_conn.commit()
            print(f"\n[SUCCESS] Added {added_count} hyperparameter(s) to configuration '{config_name_or_id}'")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to add hyperparameters: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_config_remove_hyperparameters(self, args):
        """
        Remove hyperparameters from an existing configuration

        Usage: classifier-config-remove-hyperparameters --config <name|id> [OPTIONS]

        Removes hyperparameters from an existing configuration by deleting from
        the appropriate junction tables. Can remove multiple types at once.

        Options:
            --config <name|id>            Config name or ID (required)
            --amplitude-methods <list>    Comma-separated amplitude method IDs
            --decimation-factors <list>   Comma-separated decimation factors
            --data-types <list>           Comma-separated data type names or IDs
            --distance-functions <list>   Comma-separated distance function names
            --experiment-feature-sets <list>  Comma-separated experiment feature set IDs
            --force                       Required to confirm deletion

        Examples:
            classifier-config-remove-hyperparameters --config baseline --amplitude-methods 2 --force
            classifier-config-remove-hyperparameters --config 1 --decimation-factors 31,63 --force
            classifier-config-remove-hyperparameters --config baseline --data-types adc2 --force
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        config_name_or_id = None
        amplitude_methods = None
        decimation_factors = None
        data_types = None
        distance_functions = None
        experiment_feature_sets = None
        force = False

        i = 0
        while i < len(args):
            if args[i] == '--config' and i + 1 < len(args):
                config_name_or_id = args[i + 1]
                i += 2
            elif args[i] == '--amplitude-methods' and i + 1 < len(args):
                amplitude_methods = args[i + 1]
                i += 2
            elif args[i] == '--decimation-factors' and i + 1 < len(args):
                decimation_factors = args[i + 1]
                i += 2
            elif args[i] == '--data-types' and i + 1 < len(args):
                data_types = args[i + 1]
                i += 2
            elif args[i] == '--distance-functions' and i + 1 < len(args):
                distance_functions = args[i + 1]
                i += 2
            elif args[i] == '--experiment-feature-sets' and i + 1 < len(args):
                experiment_feature_sets = args[i + 1]
                i += 2
            elif args[i] == '--force':
                force = True
                i += 1
            elif args[i] == '--help':
                print(self.cmd_classifier_config_remove_hyperparameters.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if not config_name_or_id:
            print("[ERROR] --config is required")
            return

        if not any([amplitude_methods, decimation_factors, data_types, distance_functions, experiment_feature_sets]):
            print("[ERROR] At least one hyperparameter type must be specified")
            print("Options: --amplitude-methods, --decimation-factors, --data-types,")
            print("         --distance-functions, --experiment-feature-sets")
            return

        if not force:
            print("[ERROR] --force flag required to confirm deletion")
            print("Re-run with --force to proceed")
            return

        try:
            cursor = self.db_conn.cursor()

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (self.current_experiment, self.current_classifier_id))
            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {self.current_classifier_id} not found")
                return
            global_classifier_id = result[0]

            # Resolve config_id from name or ID
            if config_name_or_id.isdigit():
                config_id = int(config_name_or_id)
            else:
                cursor.execute("""
                    SELECT config_id FROM ml_classifier_configs
                    WHERE global_classifier_id = %s AND config_name = %s
                """, (global_classifier_id, config_name_or_id))
                result = cursor.fetchone()
                if not result:
                    print(f"[ERROR] Configuration '{config_name_or_id}' not found")
                    return
                config_id = result[0]

            removed_count = 0

            # Remove amplitude methods
            if amplitude_methods:
                amp_ids = [int(x.strip()) for x in amplitude_methods.split(',')]
                for amp_id in amp_ids:
                    cursor.execute("""
                        DELETE FROM ml_classifier_config_amplitude_methods
                        WHERE config_id = %s AND amplitude_processing_method_id = %s
                    """, (config_id, amp_id))
                    if cursor.rowcount > 0:
                        removed_count += cursor.rowcount
                        print(f"[SUCCESS] Removed amplitude method {amp_id} from configuration")
                    else:
                        print(f"[INFO] Amplitude method {amp_id} not found in config, skipping")

            # Remove decimation factors
            if decimation_factors:
                dec_values = [int(x.strip()) for x in decimation_factors.split(',')]
                for dec in dec_values:
                    cursor.execute("""
                        DELETE FROM ml_classifier_config_decimation_factors
                        WHERE config_id = %s AND decimation_factor = %s
                    """, (config_id, dec))
                    if cursor.rowcount > 0:
                        removed_count += cursor.rowcount
                        print(f"[SUCCESS] Removed decimation factor {dec} from configuration")
                    else:
                        print(f"[INFO] Decimation factor {dec} not found in config, skipping")

            # Remove data types
            if data_types:
                dt_list = [x.strip() for x in data_types.split(',')]
                for dt in dt_list:
                    # Try to parse as ID or name
                    if dt.isdigit():
                        dt_id = int(dt)
                    else:
                        # Look up data type ID by name
                        cursor.execute("""
                            SELECT data_type_id FROM experiment_data_types
                            WHERE experiment_id = %s AND data_type_name = %s
                        """, (self.current_experiment, dt))
                        result = cursor.fetchone()
                        if not result:
                            print(f"[WARNING] Data type '{dt}' not found for experiment {self.current_experiment}, skipping")
                            continue
                        dt_id = result[0]

                    cursor.execute("""
                        DELETE FROM ml_classifier_config_data_types
                        WHERE config_id = %s AND data_type_id = %s
                    """, (config_id, dt_id))
                    if cursor.rowcount > 0:
                        removed_count += cursor.rowcount
                        print(f"[SUCCESS] Removed data type {dt_id} from configuration")
                    else:
                        print(f"[INFO] Data type {dt_id} not found in config, skipping")

            # Remove distance functions
            if distance_functions:
                df_names = [x.strip() for x in distance_functions.split(',')]
                for df_name in df_names:
                    # Look up distance function ID
                    cursor.execute("""
                        SELECT function_id FROM distance_functions
                        WHERE function_name = %s
                    """, (df_name,))
                    result = cursor.fetchone()
                    if not result:
                        print(f"[WARNING] Distance function '{df_name}' not found, skipping")
                        continue
                    df_id = result[0]

                    cursor.execute("""
                        DELETE FROM ml_classifier_config_distance_functions
                        WHERE config_id = %s AND distance_function_id = %s
                    """, (config_id, df_id))
                    if cursor.rowcount > 0:
                        removed_count += cursor.rowcount
                        print(f"[SUCCESS] Removed distance function '{df_name}' from configuration")
                    else:
                        print(f"[INFO] Distance function '{df_name}' not found in config, skipping")

            # Remove experiment feature sets
            if experiment_feature_sets:
                efs_ids = [int(x.strip()) for x in experiment_feature_sets.split(',')]
                for efs_id in efs_ids:
                    cursor.execute("""
                        DELETE FROM ml_classifier_config_experiment_feature_sets
                        WHERE config_id = %s AND experiment_feature_set_id = %s
                    """, (config_id, efs_id))
                    if cursor.rowcount > 0:
                        removed_count += cursor.rowcount
                        print(f"[SUCCESS] Removed experiment feature set {efs_id} from configuration")
                    else:
                        print(f"[INFO] Experiment feature set {efs_id} not found in config, skipping")

            self.db_conn.commit()
            print(f"\n[SUCCESS] Removed {removed_count} hyperparameter(s) from configuration '{config_name_or_id}'")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to remove hyperparameters: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_config_delete(self, args):
        """
        Delete a configuration for the currently selected classifier

        Usage: classifier-config-delete --config-name <name> [--confirm]

        This command deletes a configuration and all associated hyperparameter
        entries from the junction tables. Use with caution!

        Options:
            --config-name <name>    Configuration name to delete (required)
            --config-id <id>        Or specify by config ID
            --confirm               Skip confirmation prompt

        Safety:
            - Requires explicit confirmation unless --confirm flag is used
            - Shows what will be deleted before proceeding
            - Deletes from config table AND all 6 junction tables

        Examples:
            classifier-config-delete --config-name "baseline"
            classifier-config-delete --config-id 1 --confirm
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        config_name = None
        config_id = None
        confirm = False

        i = 0
        while i < len(args):
            if args[i] == '--config-name' and i + 1 < len(args):
                config_name = args[i + 1]
                i += 2
            elif args[i] == '--config-id' and i + 1 < len(args):
                config_id = int(args[i + 1])
                i += 2
            elif args[i] == '--confirm':
                confirm = True
                i += 1
            elif args[i] == '--help':
                print(self.cmd_classifier_config_delete.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if not config_name and not config_id:
            print("[ERROR] Either --config-name or --config-id is required")
            print("Usage: classifier-config-delete --config-name <name> [--confirm]")
            return

        try:
            cursor = self.db_conn.cursor()

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (self.current_experiment, self.current_classifier_id))

            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {self.current_classifier_id} not found")
                return

            global_classifier_id = result[0]

            # Build config table name
            table_name = f"experiment_{self.current_experiment:03d}_classifier_{self.current_classifier_id:03d}_config"

            # Check if config exists and get details
            if config_name:
                cursor.execute(f"""
                    SELECT config_id, config_name, is_active, created_at, notes
                    FROM {table_name}
                    WHERE global_classifier_id = %s AND config_name = %s
                """, (global_classifier_id, config_name))
            else:
                cursor.execute(f"""
                    SELECT config_id, config_name, is_active, created_at, notes
                    FROM {table_name}
                    WHERE global_classifier_id = %s AND config_id = %s
                """, (global_classifier_id, config_id))

            config_row = cursor.fetchone()
            if not config_row:
                identifier = config_name if config_name else f"ID {config_id}"
                print(f"[ERROR] Configuration '{identifier}' not found")
                return

            config_id, config_name, is_active, created_at, notes = config_row

            # Count junction table entries that will be deleted
            junction_counts = {}
            junction_tables = [
                ('decimation_factors', 'ml_classifier_config_decimation_factors'),
                ('data_types', 'ml_classifier_config_data_types'),
                ('amplitude_methods', 'ml_classifier_config_amplitude_methods'),
                ('distance_functions', 'ml_classifier_config_distance_functions'),
                ('feature_sets', 'ml_classifier_config_experiment_feature_sets'),
            ]

            for name, table in junction_tables:
                cursor.execute(f"""
                    SELECT COUNT(*)
                    FROM {table}
                    WHERE global_classifier_id = %s AND experiment_id = %s AND config_id = %s
                """, (global_classifier_id, self.current_experiment, config_id))
                junction_counts[name] = cursor.fetchone()[0]

            # Show warning
            print("\n" + "=" * 80)
            print("[WARNING] DELETE CONFIGURATION")
            print("=" * 80)
            print(f"Experiment ID: {self.current_experiment}")
            print(f"Classifier ID: {self.current_classifier_id}")
            print(f"Global Classifier ID: {global_classifier_id}")
            print(f"Config ID: {config_id}")
            print(f"Config Name: {config_name}")
            print(f"Is Active: {is_active}")
            print(f"Created: {created_at}")
            if notes:
                print(f"Notes: {notes}")
            print()
            print("This operation will DELETE:")
            print(f"  - 1 config row from {table_name}")
            print(f"  - {junction_counts['decimation_factors']} decimation factor entries")
            print(f"  - {junction_counts['data_types']} data type entries")
            print(f"  - {junction_counts['amplitude_methods']} amplitude method entries")
            print(f"  - {junction_counts['distance_functions']} distance function entries")
            print(f"  - {junction_counts['feature_sets']} feature set entries")
            total = sum(junction_counts.values())
            print(f"  TOTAL: {total} junction table entries + 1 config row")
            print()

            # Get confirmation
            if not confirm:
                print(f"To proceed, type: DELETE CONFIG {config_id}")
                print("=" * 80)

                try:
                    user_input = input("Confirmation: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n[INFO] Operation cancelled")
                    return

                required_text = f"DELETE CONFIG {config_id}"
                if user_input != required_text:
                    print("[ERROR] Confirmation text does not match. Operation cancelled.")
                    print(f"Expected: {required_text}")
                    print(f"Received: {user_input}")
                    return

            # Delete from junction tables first
            print("\nDeleting junction table entries...")
            for name, table in junction_tables:
                cursor.execute(f"""
                    DELETE FROM {table}
                    WHERE global_classifier_id = %s AND experiment_id = %s AND config_id = %s
                """, (global_classifier_id, self.current_experiment, config_id))
                deleted = cursor.rowcount
                if deleted > 0:
                    print(f"  ✓ Deleted {deleted} {name} entries")

            # Delete from config table
            cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE global_classifier_id = %s AND config_id = %s
            """, (global_classifier_id, config_id))

            self.db_conn.commit()

            print(f"\n[SUCCESS] Configuration '{config_name}' (ID: {config_id}) deleted successfully")
            print(f"  - Deleted {total} junction table entries")
            print(f"  - Deleted 1 config row")

            if is_active:
                print("\n[WARNING] You deleted the ACTIVE configuration")
                print("          Use classifier-config-activate to set a new active config")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to delete configuration: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_config_list(self, args):
        """
        List all configurations for the currently selected classifier

        Usage: classifier-config-list [OPTIONS]

        Displays all configurations with their hyperparameters, showing which one is active.

        Options:
            --all                  Show configs for all classifiers (not just current)
            --experiment-id <id>   Specify experiment ID (default: current experiment)
            --classifier-id <id>   Specify classifier ID (default: current classifier)

        Examples:
            classifier-config-list
            classifier-config-list --all
            classifier-config-list --experiment-id 41 --classifier-id 1
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        show_all = '--all' in args if args else False
        experiment_id = self.current_experiment
        classifier_id = self.current_classifier_id

        i = 0
        while i < len(args):
            if args[i] == '--experiment-id' and i + 1 < len(args):
                experiment_id = int(args[i + 1])
                i += 2
            elif args[i] == '--classifier-id' and i + 1 < len(args):
                classifier_id = int(args[i + 1])
                i += 2
            elif args[i] == '--all':
                i += 1
            elif args[i] == '--help':
                print(self.cmd_classifier_config_list.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if not show_all and not experiment_id:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' or --experiment-id")
            return

        if not show_all and not classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' or --classifier-id")
            return

        try:
            cursor = self.db_conn.cursor()

            # Build WHERE clause
            if show_all:
                where_clause = ""
                params = ()
                print("\n[INFO] Listing ALL configurations")
            else:
                # Get global_classifier_id
                cursor.execute("""
                    SELECT global_classifier_id
                    FROM ml_experiment_classifiers
                    WHERE experiment_id = %s AND classifier_id = %s
                """, (experiment_id, classifier_id))

                result = cursor.fetchone()
                if not result:
                    print(f"[ERROR] Classifier {classifier_id} not found for experiment {experiment_id}")
                    return

                global_classifier_id = result[0]
                where_clause = "WHERE c.global_classifier_id = %s"
                params = (global_classifier_id,)
                print(f"\n[INFO] Configurations for Experiment {experiment_id}, Classifier {classifier_id}")

            # Query configurations with all hyperparameters using ARRAY_AGG
            query = f"""
                SELECT
                    c.config_id,
                    c.config_name,
                    c.is_active,
                    c.created_at,
                    c.notes,
                    c.experiment_id,
                    c.classifier_id,
                    ARRAY_AGG(DISTINCT df.decimation_factor ORDER BY df.decimation_factor)
                        FILTER (WHERE df.decimation_factor IS NOT NULL) as decimation_factors,
                    ARRAY_AGG(DISTINCT dt.data_type_id ORDER BY dt.data_type_id)
                        FILTER (WHERE dt.data_type_id IS NOT NULL) as data_type_ids,
                    ARRAY_AGG(DISTINCT am.amplitude_processing_method_id ORDER BY am.amplitude_processing_method_id)
                        FILTER (WHERE am.amplitude_processing_method_id IS NOT NULL) as amplitude_methods,
                    ARRAY_AGG(DISTINCT efs.experiment_feature_set_id ORDER BY efs.experiment_feature_set_id)
                        FILTER (WHERE efs.experiment_feature_set_id IS NOT NULL) as feature_set_ids,
                    ARRAY_AGG(DISTINCT dfunc.distance_function_id ORDER BY dfunc.distance_function_id)
                        FILTER (WHERE dfunc.distance_function_id IS NOT NULL) as distance_function_ids,
                    CASE WHEN fb.feature_builder_id IS NOT NULL THEN true ELSE false END as has_feature_builder
                FROM ml_classifier_configs c
                LEFT JOIN ml_classifier_config_decimation_factors df
                    ON c.config_id = df.config_id
                LEFT JOIN ml_classifier_config_data_types dt
                    ON c.config_id = dt.config_id
                LEFT JOIN ml_classifier_config_amplitude_methods am
                    ON c.config_id = am.config_id
                LEFT JOIN ml_classifier_config_experiment_feature_sets efs
                    ON c.config_id = efs.config_id
                LEFT JOIN ml_classifier_config_distance_functions dfunc
                    ON c.config_id = dfunc.config_id
                LEFT JOIN ml_classifier_feature_builder fb
                    ON c.config_id = fb.config_id
                {where_clause}
                GROUP BY c.config_id, c.config_name, c.is_active, c.created_at, c.notes,
                         c.experiment_id, c.classifier_id, fb.feature_builder_id
                ORDER BY c.config_id
            """

            cursor.execute(query, params)
            configs = cursor.fetchall()

            if not configs:
                print("\n[INFO] No configurations found")
                return

            # Display configurations
            from tabulate import tabulate

            table_data = []
            for config in configs:
                config_id, config_name, is_active, created_at, notes, exp_id, cls_id, \
                    decimation_factors, data_type_ids, amplitude_methods, \
                    feature_set_ids, distance_function_ids, has_feature_builder = config

                # Format arrays
                df_str = ','.join(map(str, decimation_factors)) if decimation_factors else 'None'
                dt_str = ','.join(map(str, data_type_ids)) if data_type_ids else 'None'
                am_str = ','.join(map(str, amplitude_methods)) if amplitude_methods else 'None'
                fs_str = ','.join(map(str, feature_set_ids)) if feature_set_ids else 'None'
                dfunc_str = ','.join(map(str, distance_function_ids)) if distance_function_ids else 'None'

                active_marker = '[ACTIVE]' if is_active else ''
                fb_marker = 'Yes' if has_feature_builder else 'No'

                table_data.append([
                    config_id,
                    f"{config_name} {active_marker}",
                    f"E{exp_id}/C{cls_id}",
                    df_str,
                    dt_str,
                    am_str,
                    fs_str,
                    dfunc_str,
                    fb_marker,
                    created_at.strftime('%Y-%m-%d %H:%M') if created_at else ''
                ])

            headers = ['ID', 'Config Name', 'Exp/Cls', 'Dec. Factors', 'Data Types',
                      'Amp Methods', 'Feature Sets', 'Dist Funcs', 'FeatBuilder', 'Created']

            print(f"\n{tabulate(table_data, headers=headers, tablefmt='grid')}")
            print(f"\nTotal configurations: {len(configs)}")

        except Exception as e:
            print(f"\n[ERROR] Failed to list configurations: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_config_activate(self, args):
        """
        Activate a configuration for the currently selected classifier

        Usage: classifier-config-activate --config-name <name>
           or: classifier-config-activate --config-id <id>

        Sets the specified configuration as active and deactivates all other
        configurations for this classifier. Only one configuration can be active
        at a time per classifier.

        Options:
            --config-name <name>    Configuration name to activate (required if not using --config-id)
            --config-id <id>        Configuration ID to activate (required if not using --config-name)

        Examples:
            classifier-config-activate --config-name "baseline"
            classifier-config-activate --config-id 1
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        config_name = None
        config_id = None

        i = 0
        while i < len(args):
            if args[i] == '--config-name' and i + 1 < len(args):
                config_name = args[i + 1]
                i += 2
            elif args[i] == '--config-id' and i + 1 < len(args):
                config_id = int(args[i + 1])
                i += 2
            elif args[i] == '--help':
                print(self.cmd_classifier_config_activate.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if not config_name and not config_id:
            print("[ERROR] Either --config-name or --config-id is required")
            return

        try:
            cursor = self.db_conn.cursor()

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (self.current_experiment, self.current_classifier_id))

            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {self.current_classifier_id} not found for experiment {self.current_experiment}")
                return

            global_classifier_id = result[0]

            # Verify config exists
            if config_name:
                cursor.execute("""
                    SELECT config_id, config_name
                    FROM ml_classifier_configs
                    WHERE global_classifier_id = %s AND config_name = %s
                """, (global_classifier_id, config_name))
            else:
                cursor.execute("""
                    SELECT config_id, config_name
                    FROM ml_classifier_configs
                    WHERE global_classifier_id = %s AND config_id = %s
                """, (global_classifier_id, config_id))

            result = cursor.fetchone()
            if not result:
                identifier = config_name if config_name else f"ID {config_id}"
                print(f"[ERROR] Configuration '{identifier}' not found for this classifier")
                return

            config_id, config_name = result

            # Deactivate all configs for this classifier
            cursor.execute("""
                UPDATE ml_classifier_configs
                SET is_active = FALSE, updated_at = NOW()
                WHERE global_classifier_id = %s
            """, (global_classifier_id,))

            deactivated_count = cursor.rowcount

            # Activate the specified config
            cursor.execute("""
                UPDATE ml_classifier_configs
                SET is_active = TRUE, updated_at = NOW()
                WHERE global_classifier_id = %s AND config_id = %s
            """, (global_classifier_id, config_id))

            self.db_conn.commit()

            print(f"\n[SUCCESS] Configuration '{config_name}' (ID: {config_id}) activated")
            if deactivated_count > 1:
                print(f"  - Deactivated {deactivated_count - 1} other configuration(s)")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to activate configuration: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_config_show(self, args):
        """
        Show detailed information about a configuration

        Usage: classifier-config-show [OPTIONS]

        Displays detailed hyperparameters for the active configuration or a specified configuration.

        Options:
            --config-name <name>    Show specific configuration by name
            --config-id <id>        Show specific configuration by ID
            --active                Show active configuration (default)

        Examples:
            classifier-config-show
            classifier-config-show --config-name "baseline"
            classifier-config-show --config-id 1
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        config_name = None
        config_id = None
        show_active = True  # Default to showing active config

        i = 0
        while i < len(args):
            if args[i] == '--config-name' and i + 1 < len(args):
                config_name = args[i + 1]
                show_active = False
                i += 2
            elif args[i] == '--config-id' and i + 1 < len(args):
                config_id = int(args[i + 1])
                show_active = False
                i += 2
            elif args[i] == '--active':
                show_active = True
                i += 1
            elif args[i] == '--help':
                print(self.cmd_classifier_config_show.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        try:
            cursor = self.db_conn.cursor()

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (self.current_experiment, self.current_classifier_id))

            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {self.current_classifier_id} not found for experiment {self.current_experiment}")
                return

            global_classifier_id = result[0]

            # Build query based on arguments
            if show_active:
                where_clause = "WHERE c.global_classifier_id = %s AND c.is_active = TRUE"
                params = (global_classifier_id,)
            elif config_name:
                where_clause = "WHERE c.global_classifier_id = %s AND c.config_name = %s"
                params = (global_classifier_id, config_name)
            else:
                where_clause = "WHERE c.global_classifier_id = %s AND c.config_id = %s"
                params = (global_classifier_id, config_id)

            # Query configuration with all hyperparameters using ARRAY_AGG
            query = f"""
                SELECT
                    c.config_id,
                    c.config_name,
                    c.is_active,
                    c.created_at,
                    c.updated_at,
                    c.notes,
                    c.experiment_id,
                    c.classifier_id,
                    ARRAY_AGG(DISTINCT df.decimation_factor ORDER BY df.decimation_factor)
                        FILTER (WHERE df.decimation_factor IS NOT NULL) as decimation_factors,
                    ARRAY_AGG(DISTINCT dt.data_type_id ORDER BY dt.data_type_id)
                        FILTER (WHERE dt.data_type_id IS NOT NULL) as data_type_ids,
                    ARRAY_AGG(DISTINCT am.amplitude_processing_method_id ORDER BY am.amplitude_processing_method_id)
                        FILTER (WHERE am.amplitude_processing_method_id IS NOT NULL) as amplitude_methods,
                    ARRAY_AGG(DISTINCT efs.experiment_feature_set_id ORDER BY efs.experiment_feature_set_id)
                        FILTER (WHERE efs.experiment_feature_set_id IS NOT NULL) as feature_set_ids,
                    ARRAY_AGG(DISTINCT dfunc.distance_function_id ORDER BY dfunc.distance_function_id)
                        FILTER (WHERE dfunc.distance_function_id IS NOT NULL) as distance_function_ids
                FROM ml_classifier_configs c
                LEFT JOIN ml_classifier_config_decimation_factors df
                    ON c.config_id = df.config_id
                LEFT JOIN ml_classifier_config_data_types dt
                    ON c.config_id = dt.config_id
                LEFT JOIN ml_classifier_config_amplitude_methods am
                    ON c.config_id = am.config_id
                LEFT JOIN ml_classifier_config_experiment_feature_sets efs
                    ON c.config_id = efs.config_id
                LEFT JOIN ml_classifier_config_distance_functions dfunc
                    ON c.config_id = dfunc.config_id
                {where_clause}
                GROUP BY c.config_id, c.config_name, c.is_active, c.created_at, c.updated_at,
                         c.notes, c.experiment_id, c.classifier_id
            """

            cursor.execute(query, params)
            config = cursor.fetchone()

            if not config:
                identifier = "active configuration" if show_active else \
                            (config_name if config_name else f"ID {config_id}")
                print(f"[INFO] No {identifier} found")
                return

            # Unpack configuration
            config_id, config_name, is_active, created_at, updated_at, notes, exp_id, cls_id, \
                decimation_factors, data_type_ids, amplitude_methods, \
                feature_set_ids, distance_function_ids = config

            # Display configuration details
            print(f"\n{'='*60}")
            print(f"Configuration: {config_name} (ID: {config_id})")
            print(f"{'='*60}")
            print(f"Experiment ID:  {exp_id}")
            print(f"Classifier ID:  {cls_id}")
            print(f"Status:         {'ACTIVE' if is_active else 'INACTIVE'}")
            print(f"Created:        {created_at.strftime('%Y-%m-%d %H:%M:%S') if created_at else 'N/A'}")
            print(f"Updated:        {updated_at.strftime('%Y-%m-%d %H:%M:%S') if updated_at else 'N/A'}")
            if notes:
                print(f"Notes:          {notes}")

            print(f"\n{'Hyperparameters':-^60}")

            # Decimation factors
            if decimation_factors:
                df_str = ', '.join(map(str, decimation_factors))
                print(f"\nDecimation Factors ({len(decimation_factors)}):")
                print(f"  {df_str}")

            # Data types (look up names)
            if data_type_ids:
                cursor.execute("""
                    SELECT data_type_id, data_type_name
                    FROM ml_data_types_lut
                    WHERE data_type_id = ANY(%s)
                    ORDER BY data_type_id
                """, (data_type_ids,))
                dt_rows = cursor.fetchall()
                print(f"\nData Types ({len(dt_rows)}):")
                for dt_id, dt_name in dt_rows:
                    print(f"  [{dt_id}] {dt_name}")

            # Amplitude methods
            if amplitude_methods:
                am_str = ', '.join(map(str, amplitude_methods))
                print(f"\nAmplitude Processing Methods ({len(amplitude_methods)}):")
                print(f"  {am_str}")

            # Feature sets (look up names)
            if feature_set_ids:
                cursor.execute("""
                    SELECT efs.experiment_feature_set_id, fs.feature_set_id, fs.feature_set_name
                    FROM ml_experiments_feature_sets efs
                    JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
                    WHERE efs.experiment_feature_set_id = ANY(%s)
                    ORDER BY efs.experiment_feature_set_id
                """, (feature_set_ids,))
                fs_rows = cursor.fetchall()
                print(f"\nFeature Sets ({len(fs_rows)}):")
                for exp_fs_id, fs_id, fs_name in fs_rows:
                    print(f"  [EFS:{exp_fs_id}, FS:{fs_id}] {fs_name}")

            # Distance functions (look up names)
            if distance_function_ids:
                cursor.execute("""
                    SELECT distance_function_id, function_name
                    FROM ml_distance_functions_lut
                    WHERE distance_function_id = ANY(%s)
                    ORDER BY distance_function_id
                """, (distance_function_ids,))
                dfunc_rows = cursor.fetchall()
                print(f"\nDistance Functions ({len(dfunc_rows)}):")
                for dfunc_id, dfunc_name in dfunc_rows:
                    print(f"  [{dfunc_id}] {dfunc_name}")

            # Query feature builder settings
            cursor.execute("""
                SELECT include_original_feature, compute_baseline_distances_inter,
                       compute_baseline_distances_intra, statistical_features,
                       external_function, notes, created_at, updated_at
                FROM ml_classifier_feature_builder
                WHERE config_id = %s
            """, (config_id,))

            fb_row = cursor.fetchone()

            if fb_row:
                inc_orig, comp_inter, comp_intra, stat_feat, ext_func, fb_notes, fb_created, fb_updated = fb_row

                print(f"\n{'Feature Builder Settings':-^60}")
                print(f"\nInclude original features:             {inc_orig}")
                print(f"Compute inter-class baseline distances: {comp_inter}")
                print(f"Compute intra-class baseline distances: {comp_intra}")
                print(f"Statistical features (reserved):        {stat_feat}")
                print(f"External function (reserved):           {ext_func}")

                if fb_notes:
                    print(f"\nFeature Builder Notes: {fb_notes}")
            else:
                print(f"\n{'Feature Builder Settings':-^60}")
                print("\n[INFO] No feature builder configured for this configuration")
                print("  Use 'classifier-config-set-feature-builder' to configure")

            print(f"\n{'='*60}\n")

        except Exception as e:
            print(f"\n[ERROR] Failed to show configuration: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_create_feature_builder_table(self, args):
        """
        Create the ml_classifier_feature_builder table

        Usage: classifier-create-feature-builder-table [--force]

        Creates a table to control feature vector construction. This table specifies
        which feature types should be included when building the X matrix for training.

        Schema:
        - feature_builder_id: SERIAL PRIMARY KEY
        - config_id: Links to ml_classifier_configs (one feature builder per config)
        - experiment_id: Denormalized for query optimization
        - include_original_feature: Boolean - include raw feature values in X
        - compute_baseline_distances_inter: Boolean - compute distances to OTHER class baselines
        - compute_baseline_distances_intra: Boolean - compute distances to SAME class baseline
        - statistical_features: Boolean - reserved for future statistical features
        - external_function: Boolean - reserved for future external functions
        - created_at, updated_at, notes

        Baseline/reference segments are stored in per-classifier tables:
        experiment_{exp}_classifier_{cls}_reference_segments (created in Phase 2)

        Options:
            --force    Recreate table if it exists

        Examples:
            classifier-create-feature-builder-table
            classifier-create-feature-builder-table --force
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        force = '--force' in args if args else False

        try:
            cursor = self.db_conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = 'ml_classifier_feature_builder'
            """)

            exists = cursor.fetchone()

            if exists and not force:
                print("[INFO] ml_classifier_feature_builder table already exists")
                print("  Use --force to recreate")
                return

            if exists and force:
                print("[WARNING] Dropping existing ml_classifier_feature_builder table...")
                cursor.execute("DROP TABLE ml_classifier_feature_builder CASCADE")
                print("  ✓ Dropped")

            # Create feature builder table
            print("Creating ml_classifier_feature_builder table...")

            create_sql = """
                CREATE TABLE ml_classifier_feature_builder (
                    feature_builder_id SERIAL PRIMARY KEY,
                    config_id INTEGER NOT NULL,
                    experiment_id INTEGER NOT NULL,

                    -- Feature inclusion flags
                    include_original_feature BOOLEAN DEFAULT FALSE,
                    compute_baseline_distances_inter BOOLEAN DEFAULT FALSE,
                    compute_baseline_distances_intra BOOLEAN DEFAULT FALSE,
                    statistical_features BOOLEAN DEFAULT FALSE,
                    external_function BOOLEAN DEFAULT FALSE,

                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    notes TEXT,

                    FOREIGN KEY (config_id)
                        REFERENCES ml_classifier_configs(config_id)
                        ON DELETE CASCADE,
                    UNIQUE(config_id)
                );

                CREATE INDEX idx_feature_builder_config ON ml_classifier_feature_builder(config_id);
                CREATE INDEX idx_feature_builder_exp ON ml_classifier_feature_builder(experiment_id);
            """

            cursor.execute(create_sql)
            self.db_conn.commit()

            print("  ✓ Table created: ml_classifier_feature_builder")
            print("  ✓ Indexes created")
            print("  ✓ Foreign key constraint to ml_classifier_configs added")
            print("  ✓ UNIQUE constraint on config_id enforced")

            print("\n[SUCCESS] ml_classifier_feature_builder table created successfully")
            print("\nFeature builder flags:")
            print("  - include_original_feature: Include raw feature values in X matrix")
            print("  - compute_baseline_distances_inter: Compute distances to OTHER class baselines")
            print("  - compute_baseline_distances_intra: Compute distances to SAME class baseline")
            print("  - statistical_features: Reserved for future statistical features")
            print("  - external_function: Reserved for future external functions")
            print("\nBaseline segments are stored in per-classifier tables:")
            print("  experiment_{exp}_classifier_{cls}_reference_segments (Phase 2)")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to create ml_classifier_feature_builder table: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_config_set_feature_builder(self, args):
        """
        Set feature builder flags for a configuration

        Usage: classifier-config-set-feature-builder --config-id <id> [OPTIONS]

        Creates or updates the feature builder entry for a configuration.
        The feature builder controls which feature types are included when
        building the X matrix for SVM training.

        Options:
            --config-id <id>                      Config ID (required)
            --include-original                    Include raw feature values
            --no-include-original                 Exclude raw feature values
            --compute-distances-inter             Compute distances to OTHER class baselines
            --no-compute-distances-inter          Don't compute inter-class distances
            --compute-distances-intra             Compute distances to SAME class baseline
            --no-compute-distances-intra          Don't compute intra-class distances
            --statistical-features                Enable statistical features (reserved)
            --no-statistical-features             Disable statistical features
            --external-function                   Enable external function (reserved)
            --no-external-function                Disable external function
            --notes <text>                        Optional notes

        Examples:
            classifier-config-set-feature-builder --config-id 1 --include-original --compute-distances-inter
            classifier-config-set-feature-builder --config-id 1 --no-include-original
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        # Parse arguments
        config_id = None
        include_original = None
        compute_inter = None
        compute_intra = None
        statistical = None
        external = None
        notes = None

        i = 0
        while i < len(args):
            if args[i] == '--config-id' and i + 1 < len(args):
                config_id = int(args[i + 1])
                i += 2
            elif args[i] == '--include-original':
                include_original = True
                i += 1
            elif args[i] == '--no-include-original':
                include_original = False
                i += 1
            elif args[i] == '--compute-distances-inter':
                compute_inter = True
                i += 1
            elif args[i] == '--no-compute-distances-inter':
                compute_inter = False
                i += 1
            elif args[i] == '--compute-distances-intra':
                compute_intra = True
                i += 1
            elif args[i] == '--no-compute-distances-intra':
                compute_intra = False
                i += 1
            elif args[i] == '--statistical-features':
                statistical = True
                i += 1
            elif args[i] == '--no-statistical-features':
                statistical = False
                i += 1
            elif args[i] == '--external-function':
                external = True
                i += 1
            elif args[i] == '--no-external-function':
                external = False
                i += 1
            elif args[i] == '--notes' and i + 1 < len(args):
                notes = args[i + 1]
                i += 2
            elif args[i] == '--help':
                print(self.cmd_classifier_config_set_feature_builder.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if config_id is None:
            print("[ERROR] --config-id is required")
            return

        try:
            cursor = self.db_conn.cursor()

            # Get config info
            cursor.execute("""
                SELECT experiment_id, config_name
                FROM ml_classifier_configs
                WHERE config_id = %s
            """, (config_id,))

            config_row = cursor.fetchone()
            if not config_row:
                print(f"[ERROR] Config ID {config_id} not found")
                return

            experiment_id, config_name = config_row

            # Check if feature builder entry exists
            cursor.execute("""
                SELECT feature_builder_id
                FROM ml_classifier_feature_builder
                WHERE config_id = %s
            """, (config_id,))

            existing = cursor.fetchone()

            if existing:
                # UPDATE existing entry
                update_fields = []
                update_values = []

                if include_original is not None:
                    update_fields.append("include_original_feature = %s")
                    update_values.append(include_original)

                if compute_inter is not None:
                    update_fields.append("compute_baseline_distances_inter = %s")
                    update_values.append(compute_inter)

                if compute_intra is not None:
                    update_fields.append("compute_baseline_distances_intra = %s")
                    update_values.append(compute_intra)

                if statistical is not None:
                    update_fields.append("statistical_features = %s")
                    update_values.append(statistical)

                if external is not None:
                    update_fields.append("external_function = %s")
                    update_values.append(external)

                if notes is not None:
                    update_fields.append("notes = %s")
                    update_values.append(notes)

                update_fields.append("updated_at = NOW()")

                if not update_fields:
                    print("[WARNING] No changes specified")
                    return

                update_values.append(config_id)

                cursor.execute(f"""
                    UPDATE ml_classifier_feature_builder
                    SET {', '.join(update_fields)}
                    WHERE config_id = %s
                """, update_values)

                print(f"[SUCCESS] Updated feature builder for config '{config_name}' (ID: {config_id})")

            else:
                # INSERT new entry
                cursor.execute("""
                    INSERT INTO ml_classifier_feature_builder
                        (config_id, experiment_id, include_original_feature,
                         compute_baseline_distances_inter, compute_baseline_distances_intra,
                         statistical_features, external_function, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    config_id,
                    experiment_id,
                    include_original if include_original is not None else False,
                    compute_inter if compute_inter is not None else False,
                    compute_intra if compute_intra is not None else False,
                    statistical if statistical is not None else False,
                    external if external is not None else False,
                    notes
                ))

                print(f"[SUCCESS] Created feature builder for config '{config_name}' (ID: {config_id})")

            self.db_conn.commit()

            # Display current settings
            cursor.execute("""
                SELECT include_original_feature, compute_baseline_distances_inter,
                       compute_baseline_distances_intra, statistical_features,
                       external_function, notes
                FROM ml_classifier_feature_builder
                WHERE config_id = %s
            """, (config_id,))

            row = cursor.fetchone()
            if row:
                print("\nCurrent Feature Builder Settings:")
                print(f"  Include original features: {row[0]}")
                print(f"  Compute inter-class baseline distances: {row[1]}")
                print(f"  Compute intra-class baseline distances: {row[2]}")
                print(f"  Statistical features: {row[3]} (reserved)")
                print(f"  External function: {row[4]} (reserved)")
                if row[5]:
                    print(f"  Notes: {row[5]}")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to set feature builder: {e}")
            import traceback
            traceback.print_exc()

    # ========== Phase 1: Data Split Assignment Commands ==========

    def cmd_classifier_create_splits_table(self, args):
        """
        Create data_splits table for current classifier

        Usage: classifier-create-splits-table [--force]

        Creates experiment_{exp}_classifier_{cls}_data_splits table for storing
        train/test/verification split assignments.

        Schema:
        - split_id: BIGSERIAL PRIMARY KEY
        - classifier_id: Links to current classifier
        - segment_id: Links to data_segments
        - split_type: 'training', 'test', or 'verification'
        - split_assignment_date: When split was assigned
        - random_seed: For reproducibility
        - decimation_factor: Configuration parameter
        - data_type_id: Configuration parameter
        - is_experiment_data: Boolean flag
        - segment_label_id: For stratified sampling
        - created_at: Timestamp

        Options:
            --force    Recreate table if it exists (WARNING: deletes existing splits)

        Examples:
            classifier-create-splits-table
            classifier-create-splits-table --force
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        force = '--force' in args if args else False

        try:
            cursor = self.db_conn.cursor()

            # Build table name
            exp_id = self.current_experiment
            cls_id = self.current_classifier_id
            table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_data_splits"

            # Check if table exists
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = %s
            """, (table_name,))

            exists = cursor.fetchone()

            if exists and not force:
                print(f"[INFO] Table {table_name} already exists")
                print("  Use --force to recreate (WARNING: will delete existing splits)")
                return

            if exists and force:
                print(f"[WARNING] Dropping existing table {table_name}...")
                cursor.execute(f"DROP TABLE {table_name} CASCADE")
                print("  ✓ Dropped")

            # Create data splits table
            print(f"Creating table {table_name}...")

            create_sql = f"""
                CREATE TABLE {table_name} (
                    split_id BIGSERIAL PRIMARY KEY,
                    classifier_id INTEGER NOT NULL DEFAULT {cls_id},
                    segment_id INTEGER NOT NULL,
                    split_type VARCHAR(20) NOT NULL,
                    split_assignment_date TIMESTAMP DEFAULT NOW(),
                    random_seed INTEGER,
                    decimation_factor INTEGER NOT NULL,
                    data_type_id INTEGER NOT NULL,
                    is_experiment_data BOOLEAN NOT NULL DEFAULT TRUE,
                    segment_label_id INTEGER,
                    created_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE (segment_id, decimation_factor, data_type_id),
                    FOREIGN KEY (segment_id) REFERENCES data_segments(segment_id) ON DELETE CASCADE
                );

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_splits_type
                    ON {table_name}(split_type);

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_splits_segment
                    ON {table_name}(segment_id);

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_splits_label
                    ON {table_name}(segment_label_id);

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_splits_exp_data
                    ON {table_name}(is_experiment_data);
            """

            cursor.execute(create_sql)
            self.db_conn.commit()

            print(f"  ✓ Table created: {table_name}")
            print("  ✓ Indexes created:")
            print(f"    - idx_exp{exp_id:03d}_cls{cls_id:03d}_splits_type")
            print(f"    - idx_exp{exp_id:03d}_cls{cls_id:03d}_splits_segment")
            print(f"    - idx_exp{exp_id:03d}_cls{cls_id:03d}_splits_label")
            print(f"    - idx_exp{exp_id:03d}_cls{cls_id:03d}_splits_exp_data")
            print("  ✓ Foreign key constraint to data_segments added")
            print("  ✓ UNIQUE constraint on (segment_id, decimation_factor, data_type_id) enforced")

            print(f"\n[SUCCESS] Data splits table created successfully")
            print("\nNext steps:")
            print("  1. classifier-assign-splits - Assign segments to training/test/verification splits")
            print("  2. classifier-show-splits - View split statistics")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to create data splits table: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_assign_splits(self, args):
        """
        Assign segments to train/test/verification splits

        Usage: classifier-assign-splits [OPTIONS]

        Assigns segments from experiment_NNN_segment_training_data to splits using
        stratified sampling by segment_label_id. Uses active configuration to determine
        which decimation_factors and data_types to process.

        Options:
            --train-ratio <float>        Training set ratio (default: 0.70)
            --test-ratio <float>         Test set ratio (default: 0.20)
            --verification-ratio <float> Verification set ratio (default: 0.10)
            --seed <int>                 Random seed for reproducibility (default: 42)
            --force                      Overwrite existing splits

        Examples:
            classifier-assign-splits
            classifier-assign-splits --train-ratio 0.80 --test-ratio 0.15 --verification-ratio 0.05
            classifier-assign-splits --seed 123 --force
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        train_ratio = 0.70
        test_ratio = 0.20
        verification_ratio = 0.10
        seed = 42
        force = False

        i = 0
        while args and i < len(args):
            if args[i] == '--train-ratio' and i + 1 < len(args):
                train_ratio = float(args[i + 1])
                i += 2
            elif args[i] == '--test-ratio' and i + 1 < len(args):
                test_ratio = float(args[i + 1])
                i += 2
            elif args[i] == '--verification-ratio' and i + 1 < len(args):
                verification_ratio = float(args[i + 1])
                i += 2
            elif args[i] == '--seed' and i + 1 < len(args):
                seed = int(args[i + 1])
                i += 2
            elif args[i] == '--force':
                force = True
                i += 1
            else:
                i += 1

        # Validate split ratios
        total_ratio = train_ratio + test_ratio + verification_ratio
        if abs(total_ratio - 1.0) > 0.001:
            print(f"[ERROR] Split ratios must sum to 1.0 (got {total_ratio})")
            return

        if not (0.0 <= train_ratio <= 1.0 and 0.0 <= test_ratio <= 1.0 and 0.0 <= verification_ratio <= 1.0):
            print("[ERROR] All ratios must be between 0.0 and 1.0")
            return

        try:
            cursor = self.db_conn.cursor()

            exp_id = self.current_experiment
            cls_id = self.current_classifier_id
            table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_data_splits"

            # Check if splits table exists
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = %s
            """, (table_name,))

            if not cursor.fetchone():
                print(f"[ERROR] Table {table_name} does not exist")
                print("  Run 'classifier-create-splits-table' first")
                return

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (exp_id, cls_id))

            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {cls_id} not found for experiment {exp_id}")
                return

            global_classifier_id = result[0]

            # Check for active configuration
            cursor.execute("""
                SELECT config_id
                FROM ml_classifier_configs
                WHERE global_classifier_id = %s AND is_active = TRUE
            """, (global_classifier_id,))

            config_result = cursor.fetchone()
            if not config_result:
                print("[ERROR] No active configuration found for current classifier")
                print("  Use 'classifier-config-activate' to activate a configuration")
                return

            config_id = config_result[0]

            # Get decimation factors from active config
            cursor.execute("""
                SELECT DISTINCT decimation_factor
                FROM ml_classifier_config_decimation_factors
                WHERE config_id = %s
                ORDER BY decimation_factor
            """, (config_id,))

            decimation_factors = [row[0] for row in cursor.fetchall()]

            if not decimation_factors:
                print("[ERROR] Active configuration has no decimation factors")
                return

            # Get data types from active config
            cursor.execute("""
                SELECT DISTINCT data_type_id
                FROM ml_classifier_config_data_types
                WHERE config_id = %s
                ORDER BY data_type_id
            """, (config_id,))

            data_type_ids = [row[0] for row in cursor.fetchall()]

            if not data_type_ids:
                print("[ERROR] Active configuration has no data types")
                return

            print(f"Using active configuration {config_id}:")
            print(f"  Decimation factors: {decimation_factors}")
            print(f"  Data type IDs: {data_type_ids}")
            print(f"\nSplit ratios:")
            print(f"  Training: {train_ratio * 100:.1f}%")
            print(f"  Test: {test_ratio * 100:.1f}%")
            print(f"  Verification: {verification_ratio * 100:.1f}%")
            print(f"  Random seed: {seed}\n")

            # Check for existing splits
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            existing_count = cursor.fetchone()[0]

            if existing_count > 0 and not force:
                print(f"[ERROR] Table already contains {existing_count} split assignments")
                print("  Use --force to overwrite existing splits")
                return

            if existing_count > 0 and force:
                print(f"[WARNING] Deleting {existing_count} existing split assignments...")
                cursor.execute(f"DELETE FROM {table_name}")
                print("  ✓ Deleted")

            # Import sklearn for stratified sampling
            try:
                from sklearn.model_selection import train_test_split
            except ImportError:
                print("[ERROR] scikit-learn not installed")
                print("  Install with: pip install scikit-learn")
                return

            # Process each (decimation_factor, data_type_id) combination
            from collections import defaultdict

            total_assigned = 0
            combination_count = 0

            for dec_factor in decimation_factors:
                for data_type_id in data_type_ids:
                    print(f"\nProcessing decimation_factor={dec_factor}, data_type_id={data_type_id}...")

                    # Query all segments for this combination
                    query = f"""
                        SELECT DISTINCT
                            s.segment_id,
                            s.segment_label_id
                        FROM experiment_{exp_id:03d}_segment_training_data s
                        WHERE EXISTS (
                            SELECT 1 FROM experiment_{exp_id:03d}_feature_fileset f
                            WHERE f.segment_id = s.segment_id
                            AND f.decimation_factor = %s
                            AND f.data_type_id = %s
                        )
                        ORDER BY s.segment_label_id, s.segment_id
                    """

                    cursor.execute(query, (dec_factor, data_type_id))
                    segments = cursor.fetchall()

                    if not segments:
                        print(f"  ⚠ No segments found - skipping")
                        continue

                    # Group segments by label_id
                    segments_by_label = defaultdict(list)
                    for segment_id, label_id in segments:
                        segments_by_label[label_id].append(segment_id)

                    print(f"  Found {len(segments)} segments across {len(segments_by_label)} classes")

                    # Perform stratified split for each class
                    all_train_ids = []
                    all_test_ids = []
                    all_verify_ids = []

                    for label_id, segment_ids in segments_by_label.items():
                        # First split: training vs. (test + verification)
                        train_ids, temp_ids = train_test_split(
                            segment_ids,
                            train_size=train_ratio,
                            random_state=seed,
                            shuffle=True
                        )

                        # Second split: test vs. verification
                        if verification_ratio > 0:
                            test_size_adjusted = test_ratio / (test_ratio + verification_ratio)
                            test_ids, verify_ids = train_test_split(
                                temp_ids,
                                train_size=test_size_adjusted,
                                random_state=seed,
                                shuffle=True
                            )
                        else:
                            test_ids = temp_ids
                            verify_ids = []

                        all_train_ids.extend([(sid, label_id) for sid in train_ids])
                        all_test_ids.extend([(sid, label_id) for sid in test_ids])
                        all_verify_ids.extend([(sid, label_id) for sid in verify_ids])

                        print(f"    Class {label_id}: {len(segment_ids)} segments -> "
                              f"{len(train_ids)} train, {len(test_ids)} test, {len(verify_ids)} verify")

                    # Insert split assignments
                    for segment_id, label_id in all_train_ids:
                        cursor.execute(f"""
                            INSERT INTO {table_name}
                                (classifier_id, segment_id, split_type, random_seed,
                                 decimation_factor, data_type_id, is_experiment_data, segment_label_id)
                            VALUES (%s, %s, 'training', %s, %s, %s, TRUE, %s)
                        """, (cls_id, segment_id, seed, dec_factor, data_type_id, label_id))

                    for segment_id, label_id in all_test_ids:
                        cursor.execute(f"""
                            INSERT INTO {table_name}
                                (classifier_id, segment_id, split_type, random_seed,
                                 decimation_factor, data_type_id, is_experiment_data, segment_label_id)
                            VALUES (%s, %s, 'test', %s, %s, %s, TRUE, %s)
                        """, (cls_id, segment_id, seed, dec_factor, data_type_id, label_id))

                    for segment_id, label_id in all_verify_ids:
                        cursor.execute(f"""
                            INSERT INTO {table_name}
                                (classifier_id, segment_id, split_type, random_seed,
                                 decimation_factor, data_type_id, is_experiment_data, segment_label_id)
                            VALUES (%s, %s, 'verification', %s, %s, %s, TRUE, %s)
                        """, (cls_id, segment_id, seed, dec_factor, data_type_id, label_id))

                    combination_total = len(all_train_ids) + len(all_test_ids) + len(all_verify_ids)
                    total_assigned += combination_total
                    combination_count += 1

                    print(f"  ✓ Assigned {combination_total} segments")

            self.db_conn.commit()

            print(f"\n[SUCCESS] Split assignment complete!")
            print(f"  Total segments assigned: {total_assigned}")
            print(f"  Combinations processed: {combination_count}")
            print(f"  Random seed: {seed}")
            print("\nNext steps:")
            print("  classifier-show-splits - View split statistics")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to assign splits: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_copy_splits_from(self, args):
        """
        Copy data splits from another classifier to current classifier

        Usage: classifier-copy-splits-from --source-classifier <id> [OPTIONS]

        Copies all split assignments (training/test/verification) from a source
        classifier to the current classifier. This ensures fair comparison when
        evaluating different models on identical train/test splits.

        Options:
            --source-classifier <id>  Source classifier ID (required)
            --force                   Overwrite existing splits if present

        Examples:
            classifier-copy-splits-from --source-classifier 1
            classifier-copy-splits-from --source-classifier 1 --force
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        source_classifier_id = None
        force = False

        i = 0
        while i < len(args):
            if args[i] == '--source-classifier' and i + 1 < len(args):
                try:
                    source_classifier_id = int(args[i + 1])
                except ValueError:
                    print(f"[ERROR] Invalid source classifier ID: {args[i + 1]}")
                    return
                i += 2
            elif args[i] == '--force':
                force = True
                i += 1
            elif args[i] == '--help':
                print(self.cmd_classifier_copy_splits_from.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if source_classifier_id is None:
            print("[ERROR] --source-classifier is required")
            print("Usage: classifier-copy-splits-from --source-classifier <id> [--force]")
            return

        if source_classifier_id == self.current_classifier_id:
            print("[ERROR] Source and destination classifiers cannot be the same")
            return

        try:
            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            src_cls_id = source_classifier_id
            dest_cls_id = self.current_classifier_id

            source_table = f"experiment_{exp_id:03d}_classifier_{src_cls_id:03d}_data_splits"
            dest_table = f"experiment_{exp_id:03d}_classifier_{dest_cls_id:03d}_data_splits"

            # Check if source table exists
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            """, (source_table,))

            if not cursor.fetchone():
                print(f"[ERROR] Source table {source_table} does not exist")
                print(f"  Classifier {src_cls_id} has no splits table")
                return

            # Check if source table has data
            cursor.execute(f"SELECT COUNT(*) FROM {source_table}")
            source_count = cursor.fetchone()[0]

            if source_count == 0:
                print(f"[ERROR] Source table {source_table} is empty")
                print(f"  Classifier {src_cls_id} has no splits assigned")
                return

            # Check if destination table exists
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            """, (dest_table,))

            if not cursor.fetchone():
                print(f"[ERROR] Destination table {dest_table} does not exist")
                print(f"  Run 'classifier-create-splits-table' first")
                return

            # Check if destination already has splits
            cursor.execute(f"SELECT COUNT(*) FROM {dest_table}")
            dest_count = cursor.fetchone()[0]

            if dest_count > 0 and not force:
                print(f"[ERROR] Destination table {dest_table} already has {dest_count} splits")
                print("  Use --force to overwrite existing splits")
                return

            # Clear existing splits if force is enabled
            if dest_count > 0:
                cursor.execute(f"DELETE FROM {dest_table}")
                print(f"[INFO] Cleared {dest_count} existing splits from destination table")

            # Copy splits from source to destination
            cursor.execute(f"""
                INSERT INTO {dest_table}
                    (classifier_id, segment_id, split_type, random_seed,
                     decimation_factor, data_type_id, is_experiment_data, segment_label_id)
                SELECT
                    %s as classifier_id, segment_id, split_type, random_seed,
                    decimation_factor, data_type_id, is_experiment_data, segment_label_id
                FROM {source_table}
            """, (dest_cls_id,))

            copied_count = cursor.rowcount

            # Get split statistics
            cursor.execute(f"""
                SELECT split_type, COUNT(*)
                FROM {dest_table}
                GROUP BY split_type
                ORDER BY split_type
            """)
            split_stats = cursor.fetchall()

            # Get random seed
            cursor.execute(f"""
                SELECT DISTINCT random_seed
                FROM {dest_table}
                LIMIT 1
            """)
            seed_result = cursor.fetchone()
            random_seed = seed_result[0] if seed_result else None

            self.db_conn.commit()

            print(f"\n[SUCCESS] Copied splits from classifier {src_cls_id} to classifier {dest_cls_id}")
            print(f"  Total segments copied: {copied_count}")
            print(f"  Random seed: {random_seed}")
            print("\nSplit distribution:")
            for split_type, count in split_stats:
                print(f"  - {split_type}: {count}")

            print("\nNext steps:")
            print("  classifier-show-splits - View split statistics")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to copy splits: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_show_splits(self, args):
        """
        Display split statistics for current classifier

        Usage: classifier-show-splits [OPTIONS]

        Shows train/test/verification split counts and distribution by class.

        Options:
            --decimation-factor <n>  Show splits for specific decimation factor
            --data-type <id>         Show splits for specific data type
            --detail                 Show detailed per-class breakdown

        Examples:
            classifier-show-splits
            classifier-show-splits --decimation-factor 0 --data-type 4
            classifier-show-splits --detail
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        decimation_factor_filter = None
        data_type_filter = None
        detail = False

        i = 0
        while args and i < len(args):
            if args[i] == '--decimation-factor' and i + 1 < len(args):
                decimation_factor_filter = int(args[i + 1])
                i += 2
            elif args[i] == '--data-type' and i + 1 < len(args):
                data_type_filter = int(args[i + 1])
                i += 2
            elif args[i] == '--detail':
                detail = True
                i += 1
            else:
                i += 1

        try:
            cursor = self.db_conn.cursor()

            exp_id = self.current_experiment
            cls_id = self.current_classifier_id
            table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_data_splits"

            # Check if splits table exists
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = %s
            """, (table_name,))

            if not cursor.fetchone():
                print(f"[ERROR] Table {table_name} does not exist")
                print("  Run 'classifier-create-splits-table' first")
                return

            # Query random seed
            cursor.execute(f"""
                SELECT DISTINCT random_seed
                FROM {table_name}
                LIMIT 1
            """)

            seed_result = cursor.fetchone()
            if seed_result:
                random_seed = seed_result[0]
                print(f"Random seed: {random_seed}\n")
            else:
                print("[INFO] No splits assigned yet\n")
                print("  Use 'classifier-assign-splits' to assign splits")
                return

            # Build WHERE clause for filters
            where_conditions = []
            params = []

            if decimation_factor_filter is not None:
                where_conditions.append("decimation_factor = %s")
                params.append(decimation_factor_filter)

            if data_type_filter is not None:
                where_conditions.append("data_type_id = %s")
                params.append(data_type_filter)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            # Query split summary
            query = f"""
                SELECT
                    split_type,
                    decimation_factor,
                    data_type_id,
                    COUNT(*) as segment_count,
                    COUNT(DISTINCT segment_label_id) as class_count
                FROM {table_name}
                {where_clause}
                GROUP BY split_type, decimation_factor, data_type_id
                ORDER BY decimation_factor, data_type_id, split_type
            """

            cursor.execute(query, params)
            summary = cursor.fetchall()

            if not summary:
                print("[INFO] No splits found matching the specified filters")
                return

            # Display summary using tabulate
            from tabulate import tabulate

            headers = ["Split Type", "Dec Factor", "Data Type", "Segments", "Classes"]
            table_data = []

            for split_type, dec_factor, data_type_id, segment_count, class_count in summary:
                table_data.append([split_type, dec_factor, data_type_id, segment_count, class_count])

            print("Split Summary:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

            # Calculate totals
            total_segments = sum(row[3] for row in summary)
            unique_combinations = len(set((row[1], row[2]) for row in summary))

            print(f"\nTotals:")
            print(f"  Total segments: {total_segments}")
            print(f"  Unique (decimation_factor, data_type) combinations: {unique_combinations}")

            # If --detail, show per-class breakdown
            if detail:
                print("\nPer-Class Breakdown:")

                for dec_factor, data_type_id in sorted(set((row[1], row[2]) for row in summary)):
                    print(f"\n  Decimation Factor: {dec_factor}, Data Type: {data_type_id}")

                    # Query per-class breakdown for this combination
                    detail_query = f"""
                        SELECT
                            split_type,
                            segment_label_id,
                            COUNT(*) as segment_count
                        FROM {table_name}
                        WHERE decimation_factor = %s AND data_type_id = %s
                        GROUP BY split_type, segment_label_id
                        ORDER BY segment_label_id, split_type
                    """

                    cursor.execute(detail_query, (dec_factor, data_type_id))
                    detail_results = cursor.fetchall()

                    # Group by class
                    from collections import defaultdict
                    class_data = defaultdict(dict)

                    for split_type, label_id, count in detail_results:
                        class_data[label_id][split_type] = count

                    # Build detail table
                    detail_headers = ["Class", "Training", "Test", "Verification", "Total"]
                    detail_table = []

                    for label_id in sorted(class_data.keys()):
                        train = class_data[label_id].get('training', 0)
                        test = class_data[label_id].get('test', 0)
                        verify = class_data[label_id].get('verification', 0)
                        total = train + test + verify

                        detail_table.append([label_id, train, test, verify, total])

                    # Add totals row
                    total_train = sum(row[1] for row in detail_table)
                    total_test = sum(row[2] for row in detail_table)
                    total_verify = sum(row[3] for row in detail_table)
                    total_all = sum(row[4] for row in detail_table)

                    detail_table.append(["TOTAL", total_train, total_test, total_verify, total_all])

                    print(tabulate(detail_table, headers=detail_headers, tablefmt="grid"))

                    # Show percentages
                    if total_all > 0:
                        train_pct = (total_train / total_all) * 100
                        test_pct = (total_test / total_all) * 100
                        verify_pct = (total_verify / total_all) * 100

                        print(f"    Percentages: Training={train_pct:.1f}%, Test={test_pct:.1f}%, Verification={verify_pct:.1f}%")

        except Exception as e:
            print(f"\n[ERROR] Failed to show splits: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_drop_references_table(self, args):
        """
        Drop and recreate reference_segments table with updated schema

        Usage: classifier-drop-references-table --confirm

        Drops experiment_{exp}_classifier_{cls}_reference_segments table and
        recreates it with the current schema. Use this to fix schema issues
        after code updates.

        WARNING: This will DELETE all existing reference segments!

        Options:
            --confirm    Required confirmation flag

        Example:
            classifier-drop-references-table --confirm
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        confirm = '--confirm' in args

        if not confirm:
            print("[ERROR] This command requires --confirm flag")
            print("WARNING: This will DELETE all existing reference segments!")
            return

        exp_id = self.current_experiment
        cls_id = self.current_classifier_id
        table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_reference_segments"

        try:
            cursor = self.db_conn.cursor()

            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE schemaname = 'public'
                    AND tablename = %s
                )
            """, (table_name,))

            table_exists = cursor.fetchone()[0]

            if table_exists:
                print(f"[INFO] Dropping table {table_name}...")
                cursor.execute(f"DROP TABLE {table_name} CASCADE")
                self.db_conn.commit()
                print(f"[SUCCESS] Table dropped")

            # Recreate table with updated schema
            print(f"[INFO] Creating table {table_name} with updated schema...")

            create_sql = f"""
            CREATE TABLE {table_name} (
                reference_id BIGSERIAL PRIMARY KEY,
                global_classifier_id INTEGER NOT NULL,
                classifier_id INTEGER NOT NULL DEFAULT {cls_id},
                segment_label_id INTEGER NOT NULL,
                decimation_factor INTEGER NOT NULL,
                data_type_id INTEGER NOT NULL,
                amplitude_processing_method_id INTEGER NOT NULL,
                experiment_feature_set_id BIGINT NOT NULL,
                feature_set_feature_id BIGINT,
                reference_segment_id INTEGER NOT NULL,
                centroid_x DOUBLE PRECISION,
                centroid_y DOUBLE PRECISION,
                distance_to_centroid DOUBLE PRECISION,
                total_segments_in_class INTEGER,
                pca_explained_variance_ratio_1 DOUBLE PRECISION,
                pca_explained_variance_ratio_2 DOUBLE PRECISION,
                created_at TIMESTAMP DEFAULT NOW(),

                UNIQUE (segment_label_id, decimation_factor, data_type_id,
                        amplitude_processing_method_id, experiment_feature_set_id),

                FOREIGN KEY (global_classifier_id)
                    REFERENCES ml_experiment_classifiers(global_classifier_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (segment_label_id)
                    REFERENCES segment_labels(label_id),
                FOREIGN KEY (reference_segment_id)
                    REFERENCES data_segments(segment_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (data_type_id)
                    REFERENCES ml_data_types_lut(data_type_id),
                FOREIGN KEY (amplitude_processing_method_id)
                    REFERENCES ml_amplitude_normalization_lut(method_id),
                FOREIGN KEY (experiment_feature_set_id)
                    REFERENCES ml_experiments_feature_sets(experiment_feature_set_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_refs_global
                ON {table_name}(global_classifier_id);

            CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_refs_label
                ON {table_name}(segment_label_id);

            CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_refs_dec
                ON {table_name}(decimation_factor);

            CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_refs_dtype
                ON {table_name}(data_type_id);

            CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_refs_segment
                ON {table_name}(reference_segment_id);
            """

            cursor.execute(create_sql)
            self.db_conn.commit()
            print(f"[SUCCESS] Table {table_name} created with updated schema")
            print(f"\nSchema changes:")
            print(f"  - feature_set_feature_id: NOW NULLABLE (was NOT NULL)")
            print(f"  - UNIQUE constraint: Removed feature_set_feature_id")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to drop/recreate table: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_select_references(self, args):
        """
        Select reference segments for each class using PCA + centroid analysis

        Usage: classifier-select-references [OPTIONS]

        For each class in experiment_NNN_segment_training_data, performs PCA
        dimensionality reduction to 2D and selects the segment nearest to the
        centroid as the reference segment. Uses active configuration to determine
        which hyperparameter combinations to process.

        Options:
            --force                Overwrite existing reference selections
            --plot                 Generate PCA visualization plots
            --plot-dir <path>      Directory for plots (default: ~/plots)
            --min-segments <n>     Minimum segments per class (default: 5)
            --pca-components <n>   PCA components (default: 2)

        Examples:
            classifier-select-references
            classifier-select-references --plot --plot-dir ~/plots/exp041_cls001_refs
            classifier-select-references --force --min-segments 3
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        force = False
        plot = False
        plot_dir = None
        min_segments = 5
        pca_components = 2

        i = 0
        while args and i < len(args):
            if args[i] == '--force':
                force = True
                i += 1
            elif args[i] == '--plot':
                plot = True
                i += 1
            elif args[i] == '--plot-dir' and i + 1 < len(args):
                plot_dir = args[i + 1]
                i += 2
            elif args[i] == '--min-segments' and i + 1 < len(args):
                min_segments = int(args[i + 1])
                i += 2
            elif args[i] == '--pca-components' and i + 1 < len(args):
                pca_components = int(args[i + 1])
                i += 2
            else:
                i += 1

        if plot_dir is None:
            import os
            plot_dir = os.path.expanduser('~/plots')

        try:
            import numpy as np
            from sklearn.decomposition import PCA
            from collections import defaultdict

            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            cls_id = self.current_classifier_id

            # Get global_classifier_id from registry
            cursor.execute("""
                SELECT global_classifier_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (exp_id, cls_id))

            row = cursor.fetchone()
            if not row:
                print(f"[ERROR] Classifier {cls_id} not found for experiment {exp_id}")
                return

            global_classifier_id = row[0]

            # Query active configuration
            cursor.execute("""
                SELECT config_id, config_name
                FROM ml_classifier_configs
                WHERE global_classifier_id = %s AND is_active = TRUE
            """, (global_classifier_id,))

            config_row = cursor.fetchone()
            if not config_row:
                print(f"[ERROR] No active configuration for classifier {cls_id}")
                print("Use 'classifier-config-activate --config-name <name>' to activate a configuration")
                return

            config_id, config_name = config_row

            print(f"Using active configuration: {config_name}")

            # Query hyperparameters from active configuration
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
            """, (config_id,))
            amplitude_methods = [row[0] for row in cursor.fetchall()]

            cursor.execute("""
                SELECT DISTINCT experiment_feature_set_id
                FROM ml_classifier_config_experiment_feature_sets
                WHERE config_id = %s
            """, (config_id,))
            experiment_feature_sets = [row[0] for row in cursor.fetchall()]

            print(f"\nProcessing {len(decimation_factors)} decimation factor(s)")
            print(f"Processing {len(data_type_ids)} data type(s)")
            print(f"Processing {len(amplitude_methods)} amplitude method(s)")
            print(f"Processing {len(experiment_feature_sets)} experiment feature set(s)")

            # Check if reference_segments table exists
            table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_reference_segments"

            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE schemaname = 'public'
                    AND tablename = %s
                )
            """, (table_name,))

            table_exists = cursor.fetchone()[0]

            if not table_exists:
                print(f"\n[INFO] Creating table {table_name}...")

                create_sql = f"""
                CREATE TABLE {table_name} (
                    reference_id BIGSERIAL PRIMARY KEY,
                    global_classifier_id INTEGER NOT NULL,
                    classifier_id INTEGER NOT NULL DEFAULT {cls_id},
                    segment_label_id INTEGER NOT NULL,
                    decimation_factor INTEGER NOT NULL,
                    data_type_id INTEGER NOT NULL,
                    amplitude_processing_method_id INTEGER NOT NULL,
                    experiment_feature_set_id BIGINT NOT NULL,
                    feature_set_feature_id BIGINT,
                    reference_segment_id INTEGER NOT NULL,
                    centroid_x DOUBLE PRECISION,
                    centroid_y DOUBLE PRECISION,
                    distance_to_centroid DOUBLE PRECISION,
                    total_segments_in_class INTEGER,
                    pca_explained_variance_ratio_1 DOUBLE PRECISION,
                    pca_explained_variance_ratio_2 DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE (segment_label_id, decimation_factor, data_type_id,
                            amplitude_processing_method_id, experiment_feature_set_id),

                    FOREIGN KEY (global_classifier_id)
                        REFERENCES ml_experiment_classifiers(global_classifier_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (segment_label_id)
                        REFERENCES segment_labels(label_id),
                    FOREIGN KEY (reference_segment_id)
                        REFERENCES data_segments(segment_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (data_type_id)
                        REFERENCES ml_data_types_lut(data_type_id),
                    FOREIGN KEY (amplitude_processing_method_id)
                        REFERENCES ml_amplitude_normalization_lut(method_id),
                    FOREIGN KEY (experiment_feature_set_id)
                        REFERENCES ml_experiments_feature_sets(experiment_feature_set_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_refs_global
                    ON {table_name}(global_classifier_id);

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_refs_label
                    ON {table_name}(segment_label_id);

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_refs_dec
                    ON {table_name}(decimation_factor);

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_refs_dtype
                    ON {table_name}(data_type_id);

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_refs_segment
                    ON {table_name}(reference_segment_id);
                """

                cursor.execute(create_sql)
                self.db_conn.commit()
                print(f"[SUCCESS] Table {table_name} created")

            # Check for existing references
            if table_exists or force:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                existing_count = cursor.fetchone()[0]

                if existing_count > 0 and not force:
                    print(f"\n[WARNING] {existing_count} existing reference segments found")
                    print("Use --force to overwrite existing references")
                    return

                if existing_count > 0 and force:
                    print(f"[INFO] Deleting {existing_count} existing reference segments...")
                    cursor.execute(f"DELETE FROM {table_name}")
                    self.db_conn.commit()

            # Query all segment labels
            cursor.execute(f"""
                SELECT DISTINCT segment_label_id
                FROM experiment_{exp_id:03d}_segment_training_data
                ORDER BY segment_label_id
            """)
            segment_labels = [row[0] for row in cursor.fetchall()]

            print(f"\nFound {len(segment_labels)} unique classes (segment labels)")

            total_references = 0
            reference_selections = []
            skipped_count = 0

            # Process each hyperparameter combination
            for decimation_factor in decimation_factors:
                for data_type_id in data_type_ids:
                    for amplitude_method_id in amplitude_methods:
                        for exp_feature_set_id in experiment_feature_sets:

                            # Query all features that belong to this experiment_feature_set
                            cursor.execute("""
                                SELECT f.feature_id, f.feature_name, fsf.feature_order
                                FROM ml_experiments_feature_sets efs
                                JOIN ml_feature_set_features fsf ON efs.feature_set_id = fsf.feature_set_id
                                JOIN ml_features_lut f ON fsf.feature_id = f.feature_id
                                WHERE efs.experiment_feature_set_id = %s
                                ORDER BY fsf.feature_order
                            """, (exp_feature_set_id,))
                            features_in_set = cursor.fetchall()  # [(feature_id, feature_name, order), ...]

                            feature_ids = [row[0] for row in features_in_set]
                            feature_names = [row[1] for row in features_in_set]

                            print(f"\nProcessing: dec={decimation_factor}, dtype={data_type_id}, "
                                  f"amp={amplitude_method_id}, efs={exp_feature_set_id}, "
                                  f"features={feature_names}")

                            # For each class (segment label)
                            for label_id in segment_labels:

                                # Query all segments in this class
                                # We'll load feature files for each feature_id separately and concatenate
                                cursor.execute(f"""
                                    SELECT DISTINCT s.segment_id
                                    FROM experiment_{exp_id:03d}_segment_training_data s
                                    WHERE s.segment_label_id = %s
                                    ORDER BY s.segment_id
                                """, (label_id,))
                                segment_ids_query = [row[0] for row in cursor.fetchall()]

                                if len(segment_ids_query) < min_segments:
                                    print(f"  [SKIP] Class {label_id}: Only {len(segment_ids_query)} segments "
                                          f"(min {min_segments} required)")
                                    skipped_count += 1
                                    continue

                                # Load and concatenate features for all segments
                                segment_feature_vectors = {}  # {segment_id: [features]}

                                # For each feature_id in this feature_set
                                for feature_id in feature_ids:
                                    # Query feature files for this specific feature
                                    cursor.execute(f"""
                                        SELECT f.segment_id, f.feature_file_path
                                        FROM experiment_{exp_id:03d}_feature_fileset f
                                        WHERE f.segment_id = ANY(%s)
                                          AND f.decimation_factor = %s
                                          AND f.data_type_id = %s
                                          AND f.amplitude_processing_method_id = %s
                                          AND f.experiment_feature_set_id = %s
                                          AND f.feature_set_feature_id = %s
                                        ORDER BY f.segment_id
                                    """, (segment_ids_query, decimation_factor, data_type_id,
                                          amplitude_method_id, exp_feature_set_id, feature_id))

                                    feature_files = cursor.fetchall()

                                    # Load each feature file
                                    for segment_id, feature_file_path in feature_files:
                                        try:
                                            features = np.load(feature_file_path)
                                            # Select correct amplitude column
                                            column_idx = amplitude_method_id - 1
                                            features_1d = features[:, column_idx]  # Shape: (8192,)

                                            # Initialize or append to this segment's feature list
                                            if segment_id not in segment_feature_vectors:
                                                segment_feature_vectors[segment_id] = []
                                            segment_feature_vectors[segment_id].append(features_1d)
                                        except Exception as e:
                                            print(f"  [WARNING] Failed to load {feature_file_path}: {e}")
                                            continue

                                # Now concatenate features for each segment
                                feature_vectors = []
                                segment_ids = []

                                for segment_id in segment_ids_query:
                                    if segment_id in segment_feature_vectors:
                                        # Check if all features were loaded for this segment
                                        if len(segment_feature_vectors[segment_id]) == len(feature_ids):
                                            # Concatenate all features in order
                                            concatenated = np.concatenate(segment_feature_vectors[segment_id])
                                            feature_vectors.append(concatenated)
                                            segment_ids.append(segment_id)

                                if len(feature_vectors) < min_segments:
                                    print(f"  [SKIP] Class {label_id}: Only {len(feature_vectors)} "
                                          f"valid complete feature vectors (min {min_segments} required)")
                                    skipped_count += 1
                                    continue

                                # Convert to numpy array
                                X = np.array(feature_vectors)  # Shape: (n_segments, concatenated_feature_length)

                                # Apply PCA to reduce to 2D
                                pca = PCA(n_components=pca_components)
                                X_pca = pca.fit_transform(X)  # Shape: (n_segments, 2)

                                # Calculate centroid in PCA space
                                centroid = np.mean(X_pca, axis=0)  # Shape: (2,)

                                # Find segment nearest to centroid (L2 distance)
                                distances_to_centroid = np.linalg.norm(X_pca - centroid, axis=1)
                                nearest_idx = np.argmin(distances_to_centroid)

                                reference_segment_id = segment_ids[nearest_idx]
                                distance_to_centroid = distances_to_centroid[nearest_idx]

                                # Store reference selection
                                ref_data = {
                                    'global_classifier_id': global_classifier_id,
                                    'classifier_id': cls_id,
                                    'segment_label_id': label_id,
                                    'decimation_factor': decimation_factor,
                                    'data_type_id': data_type_id,
                                    'amplitude_processing_method_id': amplitude_method_id,
                                    'experiment_feature_set_id': exp_feature_set_id,
                                    'feature_set_feature_id': feature_ids[0] if len(feature_ids) == 1 else None,
                                    'reference_segment_id': reference_segment_id,
                                    'centroid_x': float(centroid[0]),
                                    'centroid_y': float(centroid[1]),
                                    'distance_to_centroid': float(distance_to_centroid),
                                    'total_segments_in_class': len(segment_ids),
                                    'pca_explained_variance_ratio_1': float(pca.explained_variance_ratio_[0]),
                                    'pca_explained_variance_ratio_2': float(pca.explained_variance_ratio_[1]),
                                    'X_pca': X_pca,  # For plotting
                                    'nearest_idx': nearest_idx
                                }
                                reference_selections.append(ref_data)

                                print(f"  [OK] Class {label_id}: Selected segment {reference_segment_id} "
                                      f"({len(segment_ids)} total segments, "
                                      f"dist={distance_to_centroid:.4f})")

                                total_references += 1

            # Insert reference segments into database
            if reference_selections:
                print(f"\n[INFO] Inserting {len(reference_selections)} reference segments into database...")

                for ref in reference_selections:
                    cursor.execute(f"""
                        INSERT INTO {table_name} (
                            global_classifier_id, classifier_id, segment_label_id,
                            decimation_factor, data_type_id, amplitude_processing_method_id,
                            experiment_feature_set_id, feature_set_feature_id,
                            reference_segment_id, centroid_x, centroid_y,
                            distance_to_centroid, total_segments_in_class,
                            pca_explained_variance_ratio_1, pca_explained_variance_ratio_2
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        ref['global_classifier_id'], ref['classifier_id'], ref['segment_label_id'],
                        ref['decimation_factor'], ref['data_type_id'], ref['amplitude_processing_method_id'],
                        ref['experiment_feature_set_id'], ref['feature_set_feature_id'],
                        ref['reference_segment_id'], ref['centroid_x'], ref['centroid_y'],
                        ref['distance_to_centroid'], ref['total_segments_in_class'],
                        ref['pca_explained_variance_ratio_1'], ref['pca_explained_variance_ratio_2']
                    ))

                self.db_conn.commit()
                print(f"[SUCCESS] Inserted {len(reference_selections)} reference segments")

            # Generate PCA plots if requested
            if plot and reference_selections:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt
                import os

                os.makedirs(plot_dir, exist_ok=True)

                print(f"\n[INFO] Generating PCA plots in {plot_dir}...")

                # Group by hyperparameter combination
                plots_by_config = defaultdict(list)

                for ref in reference_selections:
                    key = (ref['decimation_factor'], ref['data_type_id'],
                           ref['amplitude_processing_method_id'],
                           ref['experiment_feature_set_id'])
                    plots_by_config[key].append(ref)

                # Generate one plot per hyperparameter combination
                plot_count = 0
                for (dec, dtype, amp, efs), refs in plots_by_config.items():

                    fig, ax = plt.subplots(figsize=(12, 10))

                    # Plot each class
                    for ref in refs:
                        X_pca = ref['X_pca']
                        label_id = ref['segment_label_id']
                        centroid = np.array([ref['centroid_x'], ref['centroid_y']])
                        nearest_idx = ref['nearest_idx']

                        # Scatter all segments in class
                        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, label=f'Class {label_id}', s=30)

                        # Mark centroid
                        ax.scatter(centroid[0], centroid[1], marker='X', s=200,
                                  edgecolors='black', linewidths=2, zorder=10)

                        # Mark selected reference segment
                        ax.scatter(X_pca[nearest_idx, 0], X_pca[nearest_idx, 1],
                                  marker='*', s=300, edgecolors='red', linewidths=2, zorder=11)

                    ax.set_xlabel(f'PC1 ({refs[0]["pca_explained_variance_ratio_1"]*100:.1f}% variance)')
                    ax.set_ylabel(f'PC2 ({refs[0]["pca_explained_variance_ratio_2"]*100:.1f}% variance)')
                    ax.set_title(f'PCA Reference Selection\n'
                                f'Exp {exp_id}, Classifier {cls_id}\n'
                                f'Dec={dec}, DataType={dtype}, Amp={amp}, EFS={efs}')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    ax.grid(True, alpha=0.3)

                    plot_filename = (f'exp{exp_id:03d}_cls{cls_id:03d}_'
                                    f'dec{dec}_dtype{dtype}_amp{amp}_efs{efs}_pca_references.png')
                    plot_path = os.path.join(plot_dir, plot_filename)

                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"  Saved: {plot_filename}")
                    plot_count += 1

                print(f"[SUCCESS] Generated {plot_count} PCA plots")

            # Display summary
            print(f"\n{'='*60}")
            print(f"Reference Selection Summary")
            print(f"{'='*60}")
            print(f"Experiment:              {exp_id}")
            print(f"Classifier:              {cls_id}")
            print(f"Configuration:           {config_name}")
            print(f"Total references:        {total_references}")
            print(f"Classes processed:       {len(segment_labels)}")
            print(f"Skipped (insufficient):  {skipped_count}")
            print(f"Table:                   {table_name}")
            if plot:
                print(f"Plots saved to:          {plot_dir}")
            print(f"{'='*60}")

        except ImportError as e:
            print(f"\n[ERROR] Missing required package: {e}")
            print("Install with: pip install numpy scikit-learn matplotlib")
        except Exception as e:
            print(f"\n[ERROR] Failed to select references: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_copy_reference_segments(self, args):
        """
        Copy reference segments from another classifier to current classifier

        Usage: classifier-copy-reference-segments --source-classifier <id> [OPTIONS]

        Copies all reference segment assignments from a source classifier to the
        current classifier. This is useful when multiple classifiers should use
        the same reference segments for fair comparison.

        Options:
            --source-classifier <id>  Source classifier ID (required)
            --force                   Overwrite existing references if present

        Examples:
            classifier-copy-reference-segments --source-classifier 1
            classifier-copy-reference-segments --source-classifier 1 --force
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        source_classifier_id = None
        force = False

        i = 0
        while i < len(args):
            if args[i] == '--source-classifier' and i + 1 < len(args):
                try:
                    source_classifier_id = int(args[i + 1])
                except ValueError:
                    print(f"[ERROR] Invalid source classifier ID: {args[i + 1]}")
                    return
                i += 2
            elif args[i] == '--force':
                force = True
                i += 1
            elif args[i] == '--help':
                print(self.cmd_classifier_copy_reference_segments.__doc__)
                return
            else:
                print(f"[WARNING] Unknown option: {args[i]}")
                i += 1

        if source_classifier_id is None:
            print("[ERROR] --source-classifier is required")
            print("Usage: classifier-copy-reference-segments --source-classifier <id> [--force]")
            return

        if source_classifier_id == self.current_classifier_id:
            print("[ERROR] Source and destination classifiers cannot be the same")
            return

        try:
            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            src_cls_id = source_classifier_id
            dest_cls_id = self.current_classifier_id

            source_table = f"experiment_{exp_id:03d}_classifier_{src_cls_id:03d}_reference_segments"
            dest_table = f"experiment_{exp_id:03d}_classifier_{dest_cls_id:03d}_reference_segments"

            # Check if source table exists
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            """, (source_table,))

            if not cursor.fetchone():
                print(f"[ERROR] Source table {source_table} does not exist")
                print(f"  Classifier {src_cls_id} has no reference segments table")
                return

            # Check if source table has data
            cursor.execute(f"SELECT COUNT(*) FROM {source_table}")
            source_count = cursor.fetchone()[0]

            if source_count == 0:
                print(f"[ERROR] Source table {source_table} is empty")
                print(f"  Classifier {src_cls_id} has no reference segments")
                return

            # Check if destination table exists, create if not
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = %s
            """, (dest_table,))

            if not cursor.fetchone():
                print(f"[INFO] Creating destination table {dest_table}...")

                # Create destination table with same structure as source
                cursor.execute(f"""
                    CREATE TABLE {dest_table} (LIKE {source_table} INCLUDING ALL)
                """)
                self.db_conn.commit()
                print(f"  ✓ Table created")

            # Check if destination already has references
            cursor.execute(f"SELECT COUNT(*) FROM {dest_table}")
            dest_count = cursor.fetchone()[0]

            if dest_count > 0 and not force:
                print(f"[ERROR] Destination table {dest_table} already has {dest_count} references")
                print("  Use --force to overwrite existing references")
                return

            # Clear existing references if force is enabled
            if dest_count > 0:
                cursor.execute(f"DELETE FROM {dest_table}")
                print(f"[INFO] Cleared {dest_count} existing references from destination table")

            # Copy references from source to destination
            cursor.execute(f"""
                INSERT INTO {dest_table}
                SELECT * FROM {source_table}
            """)

            copied_count = cursor.rowcount

            # Get reference statistics
            cursor.execute(f"""
                SELECT segment_label_id, COUNT(*)
                FROM {dest_table}
                GROUP BY segment_label_id
                ORDER BY segment_label_id
            """)
            ref_stats = cursor.fetchall()

            self.db_conn.commit()

            print(f"\n[SUCCESS] Copied reference segments from classifier {src_cls_id} to classifier {dest_cls_id}")
            print(f"  Total references copied: {copied_count}")
            print("\nReference distribution by class:")
            for label_id, count in ref_stats:
                print(f"  - Class {label_id}: {count} references")

            print("\nNext steps:")
            print("  classifier-build-features - Build feature vectors using these references")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to copy reference segments: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_plot_references(self, args):
        """
        Generate PCA plots for existing reference segments

        Usage: classifier-plot-references [OPTIONS]

        Reads reference segments from the database and generates PCA visualization
        plots showing all segments in each class, the centroid, and the selected
        reference segment.

        Options:
            --plot-dir <path>      Directory for plots (default: ~/plots)
            --pca-components <n>   PCA components (default: 2)

        Examples:
            classifier-plot-references
            classifier-plot-references --plot-dir ~/plots/exp041_cls001_refs
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        plot_dir = os.path.expanduser('~/plots')
        pca_components = 2

        i = 0
        while args and i < len(args):
            if args[i] == '--plot-dir':
                if i + 1 < len(args):
                    plot_dir = os.path.expanduser(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --plot-dir requires a path argument")
                    return
            elif args[i] == '--pca-components':
                if i + 1 < len(args):
                    pca_components = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --pca-components requires a number argument")
                    return
            else:
                i += 1

        try:
            import numpy as np
            from sklearn.decomposition import PCA
            from collections import defaultdict
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            cls_id = self.current_classifier_id
            table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_reference_segments"

            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE schemaname = 'public'
                    AND tablename = %s
                )
            """, (table_name,))

            if not cursor.fetchone()[0]:
                print(f"[ERROR] Reference segments table does not exist: {table_name}")
                print("  Run 'classifier-select-references' first")
                return

            # Query all reference segments
            cursor.execute(f"""
                SELECT
                    segment_label_id,
                    decimation_factor,
                    data_type_id,
                    amplitude_processing_method_id,
                    experiment_feature_set_id,
                    reference_segment_id,
                    centroid_x,
                    centroid_y,
                    total_segments_in_class
                FROM {table_name}
                ORDER BY decimation_factor, data_type_id, amplitude_processing_method_id,
                         experiment_feature_set_id, segment_label_id
            """)

            reference_rows = cursor.fetchall()

            if not reference_rows:
                print("[ERROR] No reference segments found in database")
                print("  Run 'classifier-select-references' first")
                return

            print(f"[INFO] Found {len(reference_rows)} reference segments")
            print(f"[INFO] Generating PCA plots in {plot_dir}...")

            os.makedirs(plot_dir, exist_ok=True)

            # Group by hyperparameter combination
            refs_by_config = defaultdict(list)
            for row in reference_rows:
                key = (row[1], row[2], row[3], row[4])  # dec, dtype, amp, efs
                refs_by_config[key].append({
                    'segment_label_id': row[0],
                    'reference_segment_id': row[5],
                    'centroid_x': row[6],
                    'centroid_y': row[7],
                    'total_segments': row[8]
                })

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (exp_id, cls_id))
            global_classifier_id = cursor.fetchone()[0]

            # Query active configuration for feature information
            cursor.execute("""
                SELECT config_id
                FROM ml_classifier_configs
                WHERE global_classifier_id = %s AND is_active = TRUE
            """, (global_classifier_id,))
            config_row = cursor.fetchone()
            if not config_row:
                print("[ERROR] No active configuration found")
                return
            config_id = config_row[0]

            plot_count = 0

            # Generate one plot per hyperparameter combination
            for (dec, dtype, amp, efs), refs in refs_by_config.items():
                print(f"\nProcessing: dec={dec}, dtype={dtype}, amp={amp}, efs={efs}")

                # Query all features in this feature_set
                cursor.execute("""
                    SELECT f.feature_id, f.feature_name, fsf.feature_order
                    FROM ml_experiments_feature_sets efs_tbl
                    JOIN ml_feature_set_features fsf ON efs_tbl.feature_set_id = fsf.feature_set_id
                    JOIN ml_features_lut f ON fsf.feature_id = f.feature_id
                    WHERE efs_tbl.experiment_feature_set_id = %s
                    ORDER BY fsf.feature_order
                """, (efs,))
                features_in_set = cursor.fetchall()
                feature_ids = [row[0] for row in features_in_set]

                # For each class, load features and perform PCA
                class_data = {}

                for ref in refs:
                    label_id = ref['segment_label_id']
                    ref_segment_id = ref['reference_segment_id']

                    # Query all segments in this class
                    cursor.execute(f"""
                        SELECT DISTINCT s.segment_id
                        FROM experiment_{exp_id:03d}_segment_training_data s
                        WHERE s.segment_label_id = %s
                        ORDER BY s.segment_id
                    """, (label_id,))
                    segment_ids_query = [row[0] for row in cursor.fetchall()]

                    # Load and concatenate features for all segments
                    segment_feature_vectors = {}

                    for feature_id in feature_ids:
                        cursor.execute(f"""
                            SELECT f.segment_id, f.feature_file_path
                            FROM experiment_{exp_id:03d}_feature_fileset f
                            WHERE f.segment_id = ANY(%s)
                              AND f.decimation_factor = %s
                              AND f.data_type_id = %s
                              AND f.amplitude_processing_method_id = %s
                              AND f.experiment_feature_set_id = %s
                              AND f.feature_set_feature_id = %s
                            ORDER BY f.segment_id
                        """, (segment_ids_query, dec, dtype, amp, efs, feature_id))

                        feature_files = cursor.fetchall()

                        for segment_id, feature_file_path in feature_files:
                            try:
                                features = np.load(feature_file_path)
                                column_idx = amp - 1
                                features_1d = features[:, column_idx]

                                if segment_id not in segment_feature_vectors:
                                    segment_feature_vectors[segment_id] = []
                                segment_feature_vectors[segment_id].append(features_1d)
                            except Exception as e:
                                continue

                    # Concatenate features for each segment
                    feature_vectors = []
                    segment_ids = []

                    for segment_id in segment_ids_query:
                        if segment_id in segment_feature_vectors:
                            if len(segment_feature_vectors[segment_id]) == len(feature_ids):
                                concatenated = np.concatenate(segment_feature_vectors[segment_id])
                                feature_vectors.append(concatenated)
                                segment_ids.append(segment_id)

                    if len(feature_vectors) >= 2:
                        X = np.array(feature_vectors)
                        pca = PCA(n_components=pca_components)
                        X_pca = pca.fit_transform(X)

                        # Find index of reference segment
                        try:
                            ref_idx = segment_ids.index(ref_segment_id)
                        except ValueError:
                            print(f"  [WARNING] Reference segment {ref_segment_id} not found in class {label_id}")
                            continue

                        class_data[label_id] = {
                            'X_pca': X_pca,
                            'ref_idx': ref_idx,
                            'centroid': np.array([ref['centroid_x'], ref['centroid_y']]),
                            'pca': pca
                        }

                # Generate plot for this configuration
                if class_data:
                    fig, ax = plt.subplots(figsize=(12, 10))

                    for label_id, data in class_data.items():
                        X_pca = data['X_pca']
                        ref_idx = data['ref_idx']
                        centroid = data['centroid']

                        # Scatter all segments in class
                        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, label=f'Class {label_id}', s=30)

                        # Mark centroid
                        ax.scatter(centroid[0], centroid[1], marker='X', s=200,
                                  edgecolors='black', linewidths=2, zorder=10)

                        # Mark selected reference segment
                        ax.scatter(X_pca[ref_idx, 0], X_pca[ref_idx, 1],
                                  marker='*', s=300, edgecolors='red', linewidths=2, zorder=11)

                    # Use explained variance from first class
                    first_pca = list(class_data.values())[0]['pca']
                    ax.set_xlabel(f'PC1 ({first_pca.explained_variance_ratio_[0]*100:.1f}% variance)')
                    ax.set_ylabel(f'PC2 ({first_pca.explained_variance_ratio_[1]*100:.1f}% variance)')
                    ax.set_title(f'PCA Reference Selection\n'
                                f'Exp {exp_id}, Classifier {cls_id}\n'
                                f'Dec={dec}, DataType={dtype}, Amp={amp}, EFS={efs}')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                    ax.grid(True, alpha=0.3)

                    plot_filename = (f'exp{exp_id:03d}_cls{cls_id:03d}_'
                                    f'dec{dec}_dtype{dtype}_amp{amp}_efs{efs}_pca_references.png')
                    plot_path = os.path.join(plot_dir, plot_filename)

                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()

                    print(f"  Saved: {plot_filename}")
                    plot_count += 1

            print(f"\n[SUCCESS] Generated {plot_count} PCA plots in {plot_dir}")

        except ImportError as e:
            print(f"\n[ERROR] Missing required package: {e}")
            print("Install with: pip install numpy scikit-learn matplotlib")
        except Exception as e:
            print(f"\n[ERROR] Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_plot_reference_features(self, args):
        """
        Plot the actual feature data of selected reference segments

        Usage: classifier-plot-reference-features [OPTIONS]

        Reads reference segments from the database and plots the actual feature
        data (waveforms) for each selected reference segment. Shows what the
        representative segment for each class actually looks like.

        Options:
            --plot-dir <path>      Directory for plots (default: ~/plots)
            --decimation-factor <n>  Filter by decimation factor
            --data-type <id>         Filter by data type
            --amplitude-method <id>  Filter by amplitude method
            --feature-set <id>       Filter by experiment feature set

        Examples:
            classifier-plot-reference-features --plot-dir ~/plots/exp041_cls001_refs
            classifier-plot-reference-features --decimation-factor 0 --data-type 4
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        plot_dir = os.path.expanduser('~/plots')
        filter_dec = None
        filter_dtype = None
        filter_amp = None
        filter_efs = None

        i = 0
        while args and i < len(args):
            if args[i] == '--plot-dir':
                if i + 1 < len(args):
                    plot_dir = os.path.expanduser(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --plot-dir requires a path argument")
                    return
            elif args[i] == '--decimation-factor':
                if i + 1 < len(args):
                    filter_dec = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --decimation-factor requires a number")
                    return
            elif args[i] == '--data-type':
                if i + 1 < len(args):
                    filter_dtype = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --data-type requires an ID")
                    return
            elif args[i] == '--amplitude-method':
                if i + 1 < len(args):
                    filter_amp = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --amplitude-method requires an ID")
                    return
            elif args[i] == '--feature-set':
                if i + 1 < len(args):
                    filter_efs = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --feature-set requires an ID")
                    return
            else:
                i += 1

        try:
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            cls_id = self.current_classifier_id
            table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_reference_segments"

            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE schemaname = 'public'
                    AND tablename = %s
                )
            """, (table_name,))

            if not cursor.fetchone()[0]:
                print(f"[ERROR] Reference segments table does not exist: {table_name}")
                print("  Run 'classifier-select-references' first")
                return

            # Build WHERE clause with filters
            where_conditions = []
            params = []
            if filter_dec is not None:
                where_conditions.append("decimation_factor = %s")
                params.append(filter_dec)
            if filter_dtype is not None:
                where_conditions.append("data_type_id = %s")
                params.append(filter_dtype)
            if filter_amp is not None:
                where_conditions.append("amplitude_processing_method_id = %s")
                params.append(filter_amp)
            if filter_efs is not None:
                where_conditions.append("experiment_feature_set_id = %s")
                params.append(filter_efs)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            # Query all reference segments
            cursor.execute(f"""
                SELECT
                    segment_label_id,
                    decimation_factor,
                    data_type_id,
                    amplitude_processing_method_id,
                    experiment_feature_set_id,
                    reference_segment_id,
                    total_segments_in_class
                FROM {table_name}
                {where_clause}
                ORDER BY decimation_factor, data_type_id, amplitude_processing_method_id,
                         experiment_feature_set_id, segment_label_id
            """, params)

            reference_rows = cursor.fetchall()

            if not reference_rows:
                print("[ERROR] No reference segments found")
                print("  Run 'classifier-select-references' first or adjust filters")
                return

            print(f"[INFO] Found {len(reference_rows)} reference segments")
            print(f"[INFO] Generating feature plots in {plot_dir}...")

            os.makedirs(plot_dir, exist_ok=True)

            plot_count = 0

            for row in reference_rows:
                label_id = row[0]
                dec = row[1]
                dtype = row[2]
                amp = row[3]
                efs = row[4]
                ref_segment_id = row[5]
                total_segments = row[6]

                # Get label name from segment_labels table
                cursor.execute("""
                    SELECT label_name
                    FROM segment_labels
                    WHERE label_id = %s
                """, (label_id,))
                label_name_row = cursor.fetchone()
                label_name = label_name_row[0] if label_name_row else f"Label_{label_id}"

                print(f"\nProcessing: Class {label_id} ({label_name}), dec={dec}, dtype={dtype}, amp={amp}, efs={efs}, segment={ref_segment_id}")

                # Query all features in this feature_set
                cursor.execute("""
                    SELECT f.feature_id, f.feature_name, fsf.feature_order
                    FROM ml_experiments_feature_sets efs_tbl
                    JOIN ml_feature_set_features fsf ON efs_tbl.feature_set_id = fsf.feature_set_id
                    JOIN ml_features_lut f ON fsf.feature_id = f.feature_id
                    WHERE efs_tbl.experiment_feature_set_id = %s
                    ORDER BY fsf.feature_order
                """, (efs,))
                features_in_set = cursor.fetchall()
                feature_ids = [row[0] for row in features_in_set]
                feature_names = [row[1] for row in features_in_set]

                # Load feature files for this reference segment
                feature_data = []
                for feature_id, feature_name in zip(feature_ids, feature_names):
                    cursor.execute(f"""
                        SELECT f.feature_file_path
                        FROM experiment_{exp_id:03d}_feature_fileset f
                        WHERE f.segment_id = %s
                          AND f.decimation_factor = %s
                          AND f.data_type_id = %s
                          AND f.amplitude_processing_method_id = %s
                          AND f.experiment_feature_set_id = %s
                          AND f.feature_set_feature_id = %s
                    """, (ref_segment_id, dec, dtype, amp, efs, feature_id))

                    result = cursor.fetchone()
                    if result:
                        feature_file_path = result[0]
                        try:
                            features = np.load(feature_file_path)
                            # Select correct amplitude column
                            column_idx = amp - 1
                            features_1d = features[:, column_idx]
                            feature_data.append((feature_name, features_1d))
                        except Exception as e:
                            print(f"  [WARNING] Failed to load {feature_file_path}: {e}")
                            continue

                if not feature_data:
                    print(f"  [SKIP] No feature data found for segment {ref_segment_id}")
                    continue

                # Create plot
                num_features = len(feature_data)
                fig, axes = plt.subplots(num_features, 1, figsize=(12, 3 * num_features))
                if num_features == 1:
                    axes = [axes]

                for ax, (feature_name, data) in zip(axes, feature_data):
                    ax.plot(data, linewidth=0.5)
                    ax.set_ylabel(feature_name)
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim(0, len(data))

                axes[-1].set_xlabel('Sample Index')

                fig.suptitle(f'Reference Segment Features\n'
                            f'Exp {exp_id}, Classifier {cls_id}\n'
                            f'Class: {label_name} (ID={label_id})\n'
                            f'Segment {ref_segment_id} (1 of {total_segments} in class)\n'
                            f'Dec={dec}, DataType={dtype}, Amp={amp}, EFS={efs}',
                            fontsize=12)

                # Sanitize label name for filename
                safe_label_name = label_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                plot_filename = (f'exp{exp_id:03d}_cls{cls_id:03d}_'
                                f'{safe_label_name}_id{label_id}_seg{ref_segment_id}_'
                                f'dec{dec}_dtype{dtype}_amp{amp}_efs{efs}_features.png')
                plot_path = os.path.join(plot_dir, plot_filename)

                plt.tight_layout()
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()

                print(f"  Saved: {plot_filename}")
                plot_count += 1

            print(f"\n[SUCCESS] Generated {plot_count} feature plots in {plot_dir}")

        except ImportError as e:
            print(f"\n[ERROR] Missing required package: {e}")
            print("Install with: pip install numpy matplotlib")
        except Exception as e:
            print(f"\n[ERROR] Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_build_features(self, args):
        """
        Build distance-based SVM feature vectors for all segments

        Usage: classifier-build-features [OPTIONS]

        Computes feature vectors for ALL segments in experiment using reference
        segments from Phase 2. Feature dimensions = num_classes × 4 distance metrics.

        Options:
            --force                   Overwrite existing feature vectors
            --batch-size <n>          Segments per batch (default: 100)
            --decimation-factor <n>   Process only this decimation factor
            --data-type <id>          Process only this data type
            --amplitude-method <id>   Process only this amplitude method
            --feature-set <id>        Process only this feature set

        Examples:
            classifier-build-features
            classifier-build-features --decimation-factor 0 --data-type 4
            classifier-build-features --force
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database. Use 'connect' first.")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        force = False
        batch_size = 100
        filter_dec = None
        filter_dtype = None
        filter_amp = None
        filter_efs = None

        i = 0
        while args and i < len(args):
            if args[i] == '--force':
                force = True
                i += 1
            elif args[i] == '--batch-size':
                if i + 1 < len(args):
                    batch_size = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --batch-size requires a number")
                    return
            elif args[i] == '--decimation-factor':
                if i + 1 < len(args):
                    filter_dec = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --decimation-factor requires a number")
                    return
            elif args[i] == '--data-type':
                if i + 1 < len(args):
                    filter_dtype = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --data-type requires an ID")
                    return
            elif args[i] == '--amplitude-method':
                if i + 1 < len(args):
                    filter_amp = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --amplitude-method requires an ID")
                    return
            elif args[i] == '--feature-set':
                if i + 1 < len(args):
                    filter_efs = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --feature-set requires an ID")
                    return
            else:
                i += 1

        try:
            import numpy as np
            from sklearn.metrics.pairwise import pairwise_distances
            import time

            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            cls_id = self.current_classifier_id

            print(f"\n[INFO] Building SVM feature vectors for Experiment {exp_id}, Classifier {cls_id}")
            print(f"[INFO] This process may take several hours depending on data size...\n")

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (exp_id, cls_id))
            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {cls_id} not found for experiment {exp_id}")
                return
            global_classifier_id = result[0]

            # Check active configuration exists
            cursor.execute("""
                SELECT config_id, config_name
                FROM ml_classifier_configs
                WHERE global_classifier_id = %s AND is_active = TRUE
            """, (global_classifier_id,))
            config_row = cursor.fetchone()
            if not config_row:
                print("[ERROR] No active configuration found")
                print("  Run 'classifier-config-activate' first")
                return
            config_id, config_name = config_row

            print(f"[INFO] Using configuration: {config_name}")

            # Query feature builder settings
            cursor.execute("""
                SELECT include_original_feature,
                       compute_baseline_distances_inter,
                       compute_baseline_distances_intra,
                       statistical_features,
                       external_function
                FROM ml_classifier_feature_builder
                WHERE config_id = %s
            """, (config_id,))
            fb_row = cursor.fetchone()

            if not fb_row:
                print("[ERROR] No feature builder configuration found")
                print("  Run 'classifier-config-set-feature-builder' first")
                return

            include_original, compute_inter, compute_intra, statistical, external = fb_row

            print(f"[INFO] Feature builder settings:")
            print(f"  - Include original features: {include_original}")
            print(f"  - Compute inter-class distances: {compute_inter}")
            print(f"  - Compute intra-class distances: {compute_intra}")

            # Check if distances are needed
            needs_references = compute_inter or compute_intra

            # Only check for reference segments if distances are needed
            if needs_references:
                ref_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_reference_segments"
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM pg_tables
                        WHERE schemaname = 'public'
                        AND tablename = %s
                    )
                """, (ref_table_name,))
                if not cursor.fetchone()[0]:
                    print(f"[ERROR] Reference segments table does not exist: {ref_table_name}")
                    print("  Run 'classifier-select-references' first")
                    return

                cursor.execute(f"SELECT COUNT(*) FROM {ref_table_name}")
                ref_count = cursor.fetchone()[0]
                if ref_count == 0:
                    print("[ERROR] No reference segments found")
                    print("  Run 'classifier-select-references' first")
                    return

                print(f"[INFO] Found {ref_count} reference segments")
            else:
                ref_table_name = None
                print(f"[INFO] Using raw features only (no distance calculations)")

            # Create svm_features table if needed
            features_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_features"

            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE schemaname = 'public'
                    AND tablename = %s
                )
            """, (features_table_name,))

            table_exists = cursor.fetchone()[0]

            if not table_exists:
                print(f"[INFO] Creating table {features_table_name}...")

                create_sql = f"""
                CREATE TABLE {features_table_name} (
                    feature_vector_id BIGSERIAL PRIMARY KEY,
                    global_classifier_id INTEGER NOT NULL,
                    classifier_id INTEGER NOT NULL DEFAULT {cls_id},
                    segment_id INTEGER NOT NULL,
                    segment_label_id INTEGER NOT NULL,
                    decimation_factor INTEGER NOT NULL,
                    data_type_id INTEGER NOT NULL,
                    amplitude_processing_method_id INTEGER NOT NULL,
                    experiment_feature_set_id BIGINT NOT NULL,
                    svm_feature_file_path TEXT NOT NULL,
                    feature_vector_dimensions INTEGER NOT NULL,
                    num_classes INTEGER NOT NULL,
                    num_distance_metrics INTEGER NOT NULL DEFAULT 4,
                    extraction_status_id INTEGER NOT NULL DEFAULT 1,
                    extraction_time_seconds DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE (segment_id, decimation_factor, data_type_id,
                            amplitude_processing_method_id, experiment_feature_set_id),

                    FOREIGN KEY (global_classifier_id)
                        REFERENCES ml_experiment_classifiers(global_classifier_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (segment_id)
                        REFERENCES data_segments(segment_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (segment_label_id)
                        REFERENCES segment_labels(label_id),
                    FOREIGN KEY (data_type_id)
                        REFERENCES ml_data_types_lut(data_type_id),
                    FOREIGN KEY (amplitude_processing_method_id)
                        REFERENCES ml_amplitude_normalization_lut(method_id),
                    FOREIGN KEY (experiment_feature_set_id)
                        REFERENCES ml_experiments_feature_sets(experiment_feature_set_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (extraction_status_id)
                        REFERENCES ml_extraction_status_lut(status_id)
                );

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_svm_features_segment
                    ON {features_table_name}(segment_id);
                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_svm_features_decimation
                    ON {features_table_name}(decimation_factor);
                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_svm_features_dtype
                    ON {features_table_name}(data_type_id);
                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_svm_features_label
                    ON {features_table_name}(segment_label_id);
                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_svm_features_status
                    ON {features_table_name}(extraction_status_id);
                """

                cursor.execute(create_sql)
                self.db_conn.commit()
                print(f"[SUCCESS] Table {features_table_name} created")

            print("[INFO] Feature construction starting...")
            print(f"[INFO] Batch size: {batch_size} segments")

            # Query hyperparameters from active configuration
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

            # Query distance functions from configuration (only if computing distances)
            distance_functions = []
            metric_names = []

            if needs_references:
                cursor.execute("""
                    SELECT DISTINCT df.distance_function_id, df.function_name,
                           df.pairwise_metric_name, df.display_name
                    FROM ml_classifier_config_distance_functions cdf
                    JOIN ml_distance_functions_lut df ON cdf.distance_function_id = df.distance_function_id
                    WHERE cdf.config_id = %s
                      AND df.library_name = 'sklearn.metrics.pairwise'
                    ORDER BY df.distance_function_id
                """, (config_id,))
                distance_functions = cursor.fetchall()

                if len(distance_functions) != 4:
                    print(f"[ERROR] Expected 4 distance functions (L1, L2, Cosine, Pearson), found {len(distance_functions)}")
                    return

                # Extract metric names for pairwise_distances
                metric_names = [df[2] for df in distance_functions]  # pairwise_metric_name
                print(f"[INFO] Distance metrics: {[df[3] for df in distance_functions]}")  # display_name
            else:
                print(f"[INFO] Distance metrics: None (using raw features only)")

            # Apply filters if specified
            if filter_dec is not None:
                decimation_factors = [d for d in decimation_factors if d == filter_dec]
            if filter_dtype is not None:
                data_type_ids = [d for d in data_type_ids if d == filter_dtype]
            if filter_amp is not None:
                amplitude_methods = [a for a in amplitude_methods if a == filter_amp]
            if filter_efs is not None:
                experiment_feature_sets = [e for e in experiment_feature_sets if e == filter_efs]

            print(f"[INFO] Hyperparameters:")
            print(f"  Decimation factors: {decimation_factors}")
            print(f"  Data types: {data_type_ids}")
            print(f"  Amplitude methods: {amplitude_methods}")
            print(f"  Feature sets: {experiment_feature_sets}")

            # Check for existing features with current filters
            if table_exists and not force:
                # Build WHERE clause for filters
                where_conditions = []
                params = []

                if filter_dec is not None:
                    where_conditions.append("decimation_factor = %s")
                    params.append(filter_dec)
                if filter_dtype is not None:
                    where_conditions.append("data_type_id = %s")
                    params.append(filter_dtype)
                if filter_amp is not None:
                    where_conditions.append("amplitude_processing_method_id = %s")
                    params.append(filter_amp)
                if filter_efs is not None:
                    where_conditions.append("experiment_feature_set_id = %s")
                    params.append(filter_efs)

                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)

                cursor.execute(f"SELECT COUNT(*) FROM {features_table_name} {where_clause}", params)
                existing_count = cursor.fetchone()[0]

                if existing_count > 0:
                    print(f"\n[WARNING] {existing_count} existing feature vectors found for specified filters")
                    print("Use --force to overwrite existing vectors")
                    return

            # Delete existing features if force is enabled
            if force and table_exists:
                # Build WHERE clause for filters
                where_conditions = []
                params = []

                if filter_dec is not None:
                    where_conditions.append("decimation_factor = %s")
                    params.append(filter_dec)
                if filter_dtype is not None:
                    where_conditions.append("data_type_id = %s")
                    params.append(filter_dtype)
                if filter_amp is not None:
                    where_conditions.append("amplitude_processing_method_id = %s")
                    params.append(filter_amp)
                if filter_efs is not None:
                    where_conditions.append("experiment_feature_set_id = %s")
                    params.append(filter_efs)

                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)

                cursor.execute(f"SELECT COUNT(*) FROM {features_table_name} {where_clause}", params)
                existing_count = cursor.fetchone()[0]

                if existing_count > 0:
                    print(f"[INFO] Deleting {existing_count} existing feature vectors...")
                    cursor.execute(f"DELETE FROM {features_table_name} {where_clause}", params)
                    self.db_conn.commit()

            # Query all segment labels (classes)
            cursor.execute(f"""
                SELECT DISTINCT segment_label_id
                FROM experiment_{exp_id:03d}_segment_training_data
                ORDER BY segment_label_id
            """)
            segment_labels = [row[0] for row in cursor.fetchall()]
            num_classes = len(segment_labels)

            print(f"\n[INFO] Found {num_classes} classes: {segment_labels}")

            if needs_references:
                print(f"[INFO] Feature vector dimensions: {num_classes} classes × 4 metrics = {num_classes * 4}")
            else:
                print(f"[INFO] Feature vector dimensions: Variable (depends on feature set)")

            # Query all segments to process
            cursor.execute(f"""
                SELECT segment_id, segment_label_id
                FROM experiment_{exp_id:03d}_segment_training_data
                ORDER BY segment_id
            """)
            all_segments = cursor.fetchall()
            total_segments = len(all_segments)

            print(f"\n[INFO] Total segments to process: {total_segments}")

            # Distance computation function using sklearn pairwise_distances
            def compute_distance(x, y, metric):
                """
                Compute pairwise distance using sklearn.metrics.pairwise.pairwise_distances

                Args:
                    x: 1D numpy array (segment features)
                    y: 1D numpy array (reference features)
                    metric: metric name for pairwise_distances (e.g., 'manhattan', 'euclidean', 'cosine', 'correlation')

                Returns:
                    float: distance value (NaN converted to 0)
                """
                try:
                    # Reshape to 2D for pairwise_distances (expects shape (n_samples, n_features))
                    X = x.reshape(1, -1)
                    Y = y.reshape(1, -1)

                    # Compute pairwise distance
                    dist = pairwise_distances(X, Y, metric=metric)[0, 0]

                    # Convert NaN to 0 (handles zero-variance vectors)
                    if np.isnan(dist):
                        return 0.0

                    return dist
                except Exception as e:
                    # Any error returns 0
                    return 0.0

            # Process each hyperparameter combination
            total_vectors_created = 0
            total_errors = 0
            start_time = time.time()

            for dec in decimation_factors:
                for dtype in data_type_ids:
                    for amp in amplitude_methods:
                        for efs in experiment_feature_sets:

                            print(f"\n{'='*60}")
                            print(f"Processing: dec={dec}, dtype={dtype}, amp={amp}, efs={efs}")
                            print(f"{'='*60}")

                            # Query features in this feature_set
                            cursor.execute("""
                                SELECT f.feature_id, f.feature_name, fsf.feature_order
                                FROM ml_experiments_feature_sets efs_tbl
                                JOIN ml_feature_set_features fsf ON efs_tbl.feature_set_id = fsf.feature_set_id
                                JOIN ml_features_lut f ON fsf.feature_id = f.feature_id
                                WHERE efs_tbl.experiment_feature_set_id = %s
                                ORDER BY fsf.feature_order
                            """, (efs,))
                            features_in_set = cursor.fetchall()
                            feature_ids = [row[0] for row in features_in_set]
                            feature_names = [row[1] for row in features_in_set]

                            print(f"Features in set: {feature_names}")

                            # Initialize reference data structures
                            reference_segments = []
                            reference_map = {}
                            reference_features = {}

                            # Only load reference segments if computing distances
                            if needs_references:
                                # Query reference segments for this configuration
                                cursor.execute(f"""
                                    SELECT segment_label_id, reference_segment_id
                                    FROM {ref_table_name}
                                    WHERE decimation_factor = %s
                                      AND data_type_id = %s
                                      AND amplitude_processing_method_id = %s
                                      AND experiment_feature_set_id = %s
                                    ORDER BY segment_label_id
                                """, (dec, dtype, amp, efs))
                                reference_segments = cursor.fetchall()

                                if len(reference_segments) != num_classes:
                                    print(f"[WARNING] Expected {num_classes} reference segments, found {len(reference_segments)}")
                                    print(f"[SKIP] Skipping this configuration")
                                    continue

                                # Build reference_map: {label_id: reference_segment_id}
                                reference_map = {label_id: ref_seg_id for label_id, ref_seg_id in reference_segments}

                                print(f"Reference segments: {len(reference_segments)}")

                                # Load ALL reference features (once per configuration)
                                for label_id, ref_segment_id in reference_segments:
                                    ref_feature_parts = []

                                    for feature_id in feature_ids:
                                        cursor.execute(f"""
                                            SELECT feature_file_path
                                            FROM experiment_{exp_id:03d}_feature_fileset
                                            WHERE segment_id = %s
                                              AND decimation_factor = %s
                                              AND data_type_id = %s
                                              AND amplitude_processing_method_id = %s
                                              AND experiment_feature_set_id = %s
                                              AND feature_set_feature_id = %s
                                        """, (ref_segment_id, dec, dtype, amp, efs, feature_id))

                                        result = cursor.fetchone()
                                        if result:
                                            feature_file_path = result[0]
                                            try:
                                                features = np.load(feature_file_path)
                                                column_idx = amp - 1
                                                features_1d = features[:, column_idx]
                                                ref_feature_parts.append(features_1d)
                                            except Exception as e:
                                                print(f"[ERROR] Failed to load reference {ref_segment_id} feature {feature_id}: {e}")
                                                ref_feature_parts = None
                                                break

                                    if ref_feature_parts and len(ref_feature_parts) == len(feature_ids):
                                        reference_features[label_id] = np.concatenate(ref_feature_parts)
                                    else:
                                        print(f"[ERROR] Failed to load complete reference for label {label_id}")

                                if len(reference_features) != num_classes:
                                    print(f"[ERROR] Failed to load all reference features")
                                    print(f"[SKIP] Skipping this configuration")
                                    continue

                                print(f"[INFO] Loaded {len(reference_features)} reference feature vectors")
                            else:
                                print(f"[INFO] Using raw features (no reference segments needed)")

                            # Process all segments
                            segments_processed = 0
                            segments_created = 0
                            segments_failed = 0

                            for segment_id, segment_label_id in all_segments:
                                segments_processed += 1

                                # Progress tracking
                                if segments_processed % 10 == 0 or segments_processed == total_segments:
                                    elapsed = time.time() - start_time
                                    rate = segments_processed / elapsed if elapsed > 0 else 0
                                    remaining = (total_segments - segments_processed) / rate if rate > 0 else 0
                                    print(f"  Progress: {segments_processed}/{total_segments} "
                                          f"({segments_processed/total_segments*100:.1f}%) - "
                                          f"{rate:.1f} seg/sec - "
                                          f"ETA: {remaining/60:.1f} min", end='\r')

                                segment_start_time = time.time()

                                try:
                                    # Load segment features
                                    segment_feature_parts = []

                                    for feature_id in feature_ids:
                                        cursor.execute(f"""
                                            SELECT feature_file_path
                                            FROM experiment_{exp_id:03d}_feature_fileset
                                            WHERE segment_id = %s
                                              AND decimation_factor = %s
                                              AND data_type_id = %s
                                              AND amplitude_processing_method_id = %s
                                              AND experiment_feature_set_id = %s
                                              AND feature_set_feature_id = %s
                                        """, (segment_id, dec, dtype, amp, efs, feature_id))

                                        result = cursor.fetchone()
                                        if result:
                                            feature_file_path = result[0]
                                            features = np.load(feature_file_path)
                                            column_idx = amp - 1
                                            features_1d = features[:, column_idx]
                                            segment_feature_parts.append(features_1d)
                                        else:
                                            raise ValueError(f"Missing feature file for segment {segment_id}, feature {feature_id}")

                                    if len(segment_feature_parts) != len(feature_ids):
                                        raise ValueError(f"Incomplete features for segment {segment_id}")

                                    segment_features = np.concatenate(segment_feature_parts)

                                    # Build feature vector: either raw features or distance-based
                                    if needs_references:
                                        # Distance-based features: compute distances to all references
                                        feature_vector = np.zeros(num_classes * 4)

                                        for class_idx, label_id in enumerate(segment_labels):
                                            if label_id not in reference_features:
                                                raise ValueError(f"Missing reference for label {label_id}")

                                            ref_features = reference_features[label_id]

                                            # Compute distances using configured metrics
                                            base_idx = class_idx * 4
                                            for metric_idx, metric_name in enumerate(metric_names):
                                                feature_vector[base_idx + metric_idx] = compute_distance(
                                                    segment_features, ref_features, metric_name
                                                )

                                        vector_dims = num_classes * 4
                                        num_metrics = 4
                                    else:
                                        # Raw features: use concatenated features directly
                                        feature_vector = segment_features
                                        vector_dims = len(feature_vector)
                                        num_metrics = 0

                                    # Construct file path
                                    base_dir = f"/Volumes/ArcData/V3_database/experiment{exp_id:03d}/classifier_files/svm_features"
                                    classifier_dir = f"classifier_{cls_id:03d}"
                                    dec_dir = f"D{dec:06d}"
                                    dtype_name = f"TADC{dtype}"  # Assuming data_type_id maps to ADC bit depth
                                    amp_name = f"A{amp}"
                                    efs_dir = f"EFS{efs:03d}"

                                    full_dir = os.path.join(base_dir, classifier_dir, dec_dir, dtype_name, amp_name, efs_dir)
                                    os.makedirs(full_dir, exist_ok=True)

                                    filename = f"SID{segment_id:08d}_SVM_FEATURES_{vector_dims:03d}.npy"
                                    file_path = os.path.join(full_dir, filename)

                                    # Save feature vector
                                    np.save(file_path, feature_vector)

                                    # Insert database record
                                    extraction_time = time.time() - segment_start_time

                                    cursor.execute(f"""
                                        INSERT INTO {features_table_name} (
                                            global_classifier_id, classifier_id, segment_id, segment_label_id,
                                            decimation_factor, data_type_id, amplitude_processing_method_id,
                                            experiment_feature_set_id, svm_feature_file_path,
                                            feature_vector_dimensions, num_classes, num_distance_metrics,
                                            extraction_status_id, extraction_time_seconds
                                        ) VALUES (
                                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                                        )
                                    """, (
                                        global_classifier_id, cls_id, segment_id, segment_label_id,
                                        dec, dtype, amp, efs, file_path,
                                        vector_dims, num_classes, num_metrics,
                                        2, extraction_time  # status_id=2 (complete)
                                    ))

                                    segments_created += 1

                                except Exception as e:
                                    # Mark as error in database
                                    try:
                                        cursor.execute(f"""
                                            INSERT INTO {features_table_name} (
                                                global_classifier_id, classifier_id, segment_id, segment_label_id,
                                                decimation_factor, data_type_id, amplitude_processing_method_id,
                                                experiment_feature_set_id, svm_feature_file_path,
                                                feature_vector_dimensions, num_classes, num_distance_metrics,
                                                extraction_status_id
                                            ) VALUES (
                                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                                            )
                                        """, (
                                            global_classifier_id, cls_id, segment_id, segment_label_id,
                                            dec, dtype, amp, efs, '',
                                            num_classes * 4, num_classes, 4,
                                            3  # status_id=3 (error)
                                        ))
                                    except:
                                        pass

                                    segments_failed += 1
                                    total_errors += 1

                                # Commit every batch_size segments
                                if segments_processed % batch_size == 0:
                                    self.db_conn.commit()

                            # Final commit for this configuration
                            self.db_conn.commit()

                            print(f"\n  Created: {segments_created}, Failed: {segments_failed}")
                            total_vectors_created += segments_created

            # Display final summary
            elapsed_total = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"Feature Construction Summary")
            print(f"{'='*60}")
            print(f"Total feature vectors created: {total_vectors_created}")
            print(f"Total errors: {total_errors}")
            print(f"Total time: {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
            print(f"Average time per vector: {elapsed_total/total_vectors_created:.2f} seconds")
            print(f"Table: {features_table_name}")
            print(f"{'='*60}")

        except ImportError as e:
            print(f"\n[ERROR] Missing required package: {e}")
            print("Install with: pip install numpy scipy")
        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to build features: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_train_svm_init(self, args):
        """
        Initialize SVM training tables and validate prerequisites

        Usage: classifier-train-svm-init

        Creates database tables for storing SVM training results and validates
        that all prerequisites are met (data splits, feature vectors, etc.).

        This is a one-time setup command. Run this before classifier-train-svm.

        Examples:
            classifier-train-svm-init
        """
        # Check session context
        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        try:
            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            cls_id = self.current_classifier_id

            print(f"\n[INFO] Initializing SVM training for Experiment {exp_id}, Classifier {cls_id}")

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id, classifier_name
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (exp_id, cls_id))
            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {cls_id} not found for experiment {exp_id}")
                return

            global_classifier_id, classifier_name = result
            print(f"[INFO] Using classifier: {classifier_name}")

            # Check if active configuration exists
            cursor.execute("""
                SELECT config_id, config_name
                FROM ml_classifier_configs
                WHERE global_classifier_id = %s AND is_active = TRUE
            """, (global_classifier_id,))
            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] No active configuration found for classifier {cls_id}")
                print("Use 'classifier-config-activate <config_id>' to activate a configuration")
                return

            config_id, config_name = result
            print(f"[INFO] Using configuration: {config_name}")

            # Check if data splits exist
            splits_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_data_splits"
            cursor.execute(f"""
                SELECT COUNT(*) FROM {splits_table_name}
            """)
            split_count = cursor.fetchone()[0]
            if split_count == 0:
                print(f"[ERROR] No data splits found in {splits_table_name}")
                print("Run 'classifier-assign-splits' first")
                return

            print(f"[INFO] Found {split_count} split assignments")

            # Check if feature vectors exist
            features_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_features"
            cursor.execute(f"""
                SELECT COUNT(*) FROM {features_table_name}
                WHERE extraction_status_id = 2
            """)
            feature_count = cursor.fetchone()[0]
            if feature_count == 0:
                print(f"[ERROR] No feature vectors found in {features_table_name}")
                print("Run 'classifier-build-features' first")
                return

            print(f"[INFO] Found {feature_count} feature vectors")

            # Create svm_results table if needed
            results_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_results"
            per_class_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_per_class_results"

            # Check if tables exist
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (results_table_name,))
            table_exists = cursor.fetchone()[0]

            if not table_exists:
                print(f"[INFO] Creating table {results_table_name}...")

                # Create svm_results table
                create_results_sql = f"""
                CREATE TABLE {results_table_name} (
                    result_id BIGSERIAL PRIMARY KEY,
                    global_classifier_id INTEGER NOT NULL,
                    classifier_id INTEGER NOT NULL DEFAULT {cls_id},
                    decimation_factor INTEGER NOT NULL,
                    data_type_id INTEGER NOT NULL,
                    amplitude_processing_method_id INTEGER NOT NULL,
                    experiment_feature_set_id BIGINT NOT NULL,

                    -- SVM hyperparameters
                    svm_kernel VARCHAR(20) NOT NULL,
                    svm_c_parameter DOUBLE PRECISION NOT NULL,
                    svm_gamma VARCHAR(20),
                    class_weight VARCHAR(20) DEFAULT 'balanced',
                    random_state INTEGER DEFAULT 42,

                    -- Training configuration
                    train_ratio DOUBLE PRECISION DEFAULT 0.70,
                    test_ratio DOUBLE PRECISION DEFAULT 0.20,
                    verification_ratio DOUBLE PRECISION DEFAULT 0.10,
                    cv_folds INTEGER DEFAULT 5,

                    -- 13-class multiclass metrics (TRAINING SET)
                    accuracy_train DOUBLE PRECISION,
                    precision_macro_train DOUBLE PRECISION,
                    recall_macro_train DOUBLE PRECISION,
                    f1_macro_train DOUBLE PRECISION,
                    precision_weighted_train DOUBLE PRECISION,
                    recall_weighted_train DOUBLE PRECISION,
                    f1_weighted_train DOUBLE PRECISION,
                    cv_mean_accuracy DOUBLE PRECISION,
                    cv_std_accuracy DOUBLE PRECISION,

                    -- 13-class multiclass metrics (TEST SET)
                    accuracy_test DOUBLE PRECISION,
                    precision_macro_test DOUBLE PRECISION,
                    recall_macro_test DOUBLE PRECISION,
                    f1_macro_test DOUBLE PRECISION,
                    precision_weighted_test DOUBLE PRECISION,
                    recall_weighted_test DOUBLE PRECISION,
                    f1_weighted_test DOUBLE PRECISION,

                    -- 13-class multiclass metrics (VERIFICATION SET)
                    accuracy_verify DOUBLE PRECISION,
                    precision_macro_verify DOUBLE PRECISION,
                    recall_macro_verify DOUBLE PRECISION,
                    f1_macro_verify DOUBLE PRECISION,
                    precision_weighted_verify DOUBLE PRECISION,
                    recall_weighted_verify DOUBLE PRECISION,
                    f1_weighted_verify DOUBLE PRECISION,

                    -- Binary arc detection metrics (TRAINING SET)
                    arc_accuracy_train DOUBLE PRECISION,
                    arc_precision_train DOUBLE PRECISION,
                    arc_recall_train DOUBLE PRECISION,
                    arc_f1_train DOUBLE PRECISION,
                    arc_specificity_train DOUBLE PRECISION,
                    arc_roc_auc_train DOUBLE PRECISION,
                    arc_pr_auc_train DOUBLE PRECISION,

                    -- Binary arc detection metrics (TEST SET)
                    arc_accuracy_test DOUBLE PRECISION,
                    arc_precision_test DOUBLE PRECISION,
                    arc_recall_test DOUBLE PRECISION,
                    arc_f1_test DOUBLE PRECISION,
                    arc_specificity_test DOUBLE PRECISION,
                    arc_roc_auc_test DOUBLE PRECISION,
                    arc_pr_auc_test DOUBLE PRECISION,

                    -- Binary arc detection metrics (VERIFICATION SET)
                    arc_accuracy_verify DOUBLE PRECISION,
                    arc_precision_verify DOUBLE PRECISION,
                    arc_recall_verify DOUBLE PRECISION,
                    arc_f1_verify DOUBLE PRECISION,
                    arc_specificity_verify DOUBLE PRECISION,
                    arc_roc_auc_verify DOUBLE PRECISION,
                    arc_pr_auc_verify DOUBLE PRECISION,

                    -- File paths
                    model_path TEXT NOT NULL,
                    confusion_matrix_13class_train_path TEXT,
                    confusion_matrix_13class_test_path TEXT,
                    confusion_matrix_13class_verify_path TEXT,
                    confusion_matrix_binary_train_path TEXT,
                    confusion_matrix_binary_test_path TEXT,
                    confusion_matrix_binary_verify_path TEXT,
                    roc_curve_binary_train_path TEXT,
                    roc_curve_binary_test_path TEXT,
                    roc_curve_binary_verify_path TEXT,
                    pr_curve_binary_train_path TEXT,
                    pr_curve_binary_test_path TEXT,
                    pr_curve_binary_verify_path TEXT,
                    classification_report_path TEXT,

                    -- Performance tracking
                    training_time_seconds DOUBLE PRECISION,
                    prediction_time_seconds DOUBLE PRECISION,
                    created_at TIMESTAMP DEFAULT NOW(),

                    UNIQUE (decimation_factor, data_type_id, amplitude_processing_method_id,
                            experiment_feature_set_id, svm_kernel, svm_c_parameter, svm_gamma),

                    FOREIGN KEY (global_classifier_id)
                        REFERENCES ml_experiment_classifiers(global_classifier_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_svm_results_config
                    ON {results_table_name}(
                        decimation_factor, data_type_id, amplitude_processing_method_id,
                        experiment_feature_set_id);

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_svm_results_accuracy
                    ON {results_table_name}(accuracy_test DESC);

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_svm_results_arc_f1
                    ON {results_table_name}(arc_f1_test DESC);
                """

                cursor.execute(create_results_sql)

                # Create per_class_results table
                create_per_class_sql = f"""
                CREATE TABLE {per_class_table_name} (
                    result_id BIGINT NOT NULL,
                    segment_label_id INTEGER NOT NULL,
                    split_type VARCHAR(20) NOT NULL,

                    -- Per-class metrics
                    precision DOUBLE PRECISION,
                    recall DOUBLE PRECISION,
                    f1_score DOUBLE PRECISION,
                    support INTEGER,

                    PRIMARY KEY (result_id, segment_label_id, split_type),

                    FOREIGN KEY (result_id)
                        REFERENCES {results_table_name}(result_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (segment_label_id)
                        REFERENCES segment_labels(label_id)
                );

                CREATE INDEX idx_exp{exp_id:03d}_cls{cls_id:03d}_per_class_label
                    ON {per_class_table_name}(segment_label_id);
                """

                cursor.execute(create_per_class_sql)
                self.db_conn.commit()
                print(f"[SUCCESS] Tables created: {results_table_name}, {per_class_table_name}")
            else:
                print(f"[INFO] Tables already exist: {results_table_name}, {per_class_table_name}")

            print("\n[SUCCESS] Initialization complete!")
            print("[INFO] You can now run 'classifier-train-svm' to train the SVM classifier.")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to initialize SVM training: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_test_svm_single(self, args):
        """
        Test single SVM training with detailed diagnostics

        Usage: classifier-test-svm-single [OPTIONS]

        Diagnostic command to test ONE SVM training task with detailed timing output.
        Helps identify performance bottlenecks without multiprocessing complexity.

        Options:
          --decimation-factor <n>   Test with this decimation factor (default: 0)
          --data-type <id>          Test with this data type (default: 6)
          --amplitude-method <id>   Test with this amplitude method (default: 2)
          --feature-set <id>        Test with this feature set (default: 1)
          --kernel <name>           Test with this kernel (default: linear)
          --C <value>               SVM C parameter (default: 1.0)
          --gamma <value>           SVM gamma parameter (default: scale)

        This command is for DIAGNOSTICS ONLY. It does NOT insert results into database.
        Use classifier-train-svm for actual training.
        """
        if not self.db_conn:
            print("[ERROR] Not connected to database")
            return

        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>'")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>'")
            return

        exp_id = self.current_experiment
        cls_id = self.current_classifier_id

        # Parse arguments
        dec = 0
        dtype = 6
        amp = 2
        efs = 1
        kernel = 'linear'
        C = 1.0
        gamma = 'scale'

        i = 0
        while i < len(args):
            if args[i] == '--decimation-factor':
                if i + 1 < len(args):
                    dec = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --decimation-factor requires a value")
                    return
            elif args[i] == '--data-type':
                if i + 1 < len(args):
                    dtype = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --data-type requires a value")
                    return
            elif args[i] == '--amplitude-method':
                if i + 1 < len(args):
                    amp = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --amplitude-method requires a value")
                    return
            elif args[i] == '--feature-set':
                if i + 1 < len(args):
                    efs = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --feature-set requires a value")
                    return
            elif args[i] == '--kernel':
                if i + 1 < len(args):
                    kernel = args[i + 1]
                    if kernel not in ['linear', 'rbf', 'poly']:
                        print(f"[ERROR] Invalid kernel: {kernel}. Must be linear, rbf, or poly")
                        return
                    i += 2
                else:
                    print("[ERROR] --kernel requires a value")
                    return
            elif args[i] == '--C':
                if i + 1 < len(args):
                    try:
                        C = float(args[i + 1])
                    except ValueError:
                        print(f"[ERROR] Invalid C value: {args[i + 1]}")
                        return
                    i += 2
                else:
                    print("[ERROR] --C requires a value")
                    return
            elif args[i] == '--gamma':
                if i + 1 < len(args):
                    gamma_val = args[i + 1]
                    if gamma_val not in ['scale', 'auto']:
                        try:
                            gamma = float(gamma_val)
                        except ValueError:
                            print(f"[ERROR] Invalid gamma value: {gamma_val}. Must be 'scale', 'auto', or a number")
                            return
                    else:
                        gamma = gamma_val
                    i += 2
                else:
                    print("[ERROR] --gamma requires a value")
                    return
            else:
                print(f"[ERROR] Unknown option: {args[i]}")
                return

        # Get global_classifier_id
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT global_classifier_id
            FROM ml_experiment_classifiers
            WHERE experiment_id = %s AND classifier_id = %s
        """, (exp_id, cls_id))
        row = cursor.fetchone()
        if not row:
            print(f"[ERROR] Classifier {cls_id} not found for experiment {exp_id}")
            return
        global_classifier_id = row[0]

        # Get label categories
        cursor.execute("""
            SELECT label_id, category
            FROM segment_labels
            WHERE active = TRUE
            ORDER BY label_id
        """)
        label_categories = {label_id: category for label_id, category in cursor.fetchall()}

        # Build SVM parameters
        svm_params = {
            'kernel': kernel,
            'C': C
        }
        if kernel in ['rbf', 'poly']:
            svm_params['gamma'] = gamma

        # Build database config for worker
        db_config = {
            'dbname': 'arc_detection',
            'user': 'kjensen',
            'password': '',
            'host': 'localhost',
            'port': 5432
        }

        # Build work item tuple
        config_tuple = (
            dec, dtype, amp, efs, svm_params, db_config, label_categories,
            exp_id, cls_id, global_classifier_id
        )

        print(f"\n[INFO] Testing SVM Training - DIAGNOSTIC MODE")
        print(f"[INFO] Experiment {exp_id}, Classifier {cls_id}")
        print(f"[INFO] Config: dec={dec}, dtype={dtype}, amp={amp}, efs={efs}")
        print(f"[INFO] SVM: kernel={kernel}, C={C}", end="")
        if kernel in ['rbf', 'poly']:
            print(f", gamma={gamma}")
        else:
            print()
        print(f"\n[INFO] This is a diagnostic test. Results will NOT be inserted into database.")
        print(f"[INFO] Starting single SVM training task...")
        print(f"[DEBUG] Step 1: Calling worker function...")
        print(f"[DEBUG] If this hangs, the problem is in the worker function itself\n")

        import sys
        sys.stdout.flush()  # Force output to display

        # Call worker function directly (no multiprocessing)
        result = train_svm_worker(config_tuple)

        print(f"\n[DEBUG] Step 2: Worker function returned")

        # Display results
        if result['success']:
            print(f"\n{'='*80}")
            print(f"[SUCCESS] SVM Training Completed")
            print(f"{'='*80}\n")

            # Show test accuracy
            acc = result['metrics_test']['accuracy']
            print(f"Test Accuracy: {acc:.4f}\n")

            # Show detailed timing breakdown
            if 'timings' in result:
                t = result['timings']
                print(f"TIMING BREAKDOWN:")
                print(f"  Database Connection:     {t.get('db_connection', 0):>8.2f}s")
                print(f"  Feature Loading (train): {t.get('feature_load_train', 0):>8.2f}s")
                print(f"  Feature Loading (test):  {t.get('feature_load_test', 0):>8.2f}s")
                print(f"  Feature Loading (verify):{t.get('feature_load_verify', 0):>8.2f}s")
                print(f"  Feature Loading (TOTAL): {t.get('feature_load_total', 0):>8.2f}s  <-- May be bottleneck")
                print(f"  SVM Training:            {t.get('svm_training', 0):>8.2f}s")
                print(f"  Cross-Validation:        {t.get('cross_validation', 0):>8.2f}s")
                print(f"  Prediction:              {t.get('prediction', 0):>8.2f}s")
                print(f"  Metrics Computation:     {t.get('metrics_computation', 0):>8.2f}s")
                print(f"  Save Model:              {t.get('save_model', 0):>8.2f}s")
                print(f"  Save Visualizations:     {t.get('save_visualizations', 0):>8.2f}s")
                print(f"  {'-'*40}")
                print(f"  TOTAL TIME:              {t.get('total', 0):>8.2f}s\n")

                # Identify bottleneck
                max_time = 0
                max_step = ""
                for step, time_val in t.items():
                    if step != 'total' and time_val > max_time:
                        max_time = time_val
                        max_step = step

                if max_time > 0:
                    pct = (max_time / t.get('total', 1)) * 100
                    print(f"BOTTLENECK: '{max_step}' took {max_time:.2f}s ({pct:.1f}% of total time)\n")

            # Show data sizes
            if 'data_sizes' in result:
                ds = result['data_sizes']
                print(f"DATA SIZES:")
                print(f"  Training samples:   {ds.get('train_samples', 0):>6}")
                print(f"  Test samples:       {ds.get('test_samples', 0):>6}")
                print(f"  Verification samples:{ds.get('verify_samples', 0):>6}")
                print(f"  Feature dimensions: {ds.get('feature_dim', 0):>6}\n")

            # Show metrics summary
            print(f"METRICS SUMMARY:")
            print(f"  Training accuracy:   {result['metrics_train']['accuracy']:.4f}")
            print(f"  Test accuracy:       {result['metrics_test']['accuracy']:.4f}")
            print(f"  Verification accuracy:{result['metrics_verify']['accuracy']:.4f}")
            print(f"  Cross-val mean:      {result['cv_mean']:.4f} ± {result['cv_std']:.4f}\n")

            # Show binary arc detection
            print(f"BINARY ARC DETECTION:")
            print(f"  Training accuracy:   {result['binary_metrics_train']['accuracy']:.4f}")
            print(f"  Test accuracy:       {result['binary_metrics_test']['accuracy']:.4f}")
            print(f"  Verification accuracy:{result['binary_metrics_verify']['accuracy']:.4f}\n")

            print(f"Model saved to: {result['model_path']}\n")
            print(f"NOTE: Results NOT inserted into database (diagnostic mode)")

        else:
            print(f"\n{'='*80}")
            print(f"[FAILED] SVM Training Failed")
            print(f"{'='*80}\n")
            print(f"Error: {result['error']}\n")
            if 'traceback' in result:
                print(f"Traceback:")
                print(result['traceback'])
            if 'timings' in result:
                print(f"\nFailed after {result['timings'].get('total', 0):.2f}s")

    def cmd_classifier_train_svm(self, args):
        """
        Train SVM classifier using distance-based feature vectors

        Usage: classifier-train-svm [OPTIONS]

        Trains SVM on feature vectors from Phase 3 using data splits from Phase 1.
        Evaluates on train/test/verification splits with both 13-class and binary
        arc detection metrics.

        Options:
            --workers <n>             Parallel workers (default: 7, max: 28)
            --memory <mb>             Total memory budget in MB (default: 16000)
            --decimation-factor <n>   Train only this decimation factor
            --data-type <id>          Train only this data type
            --amplitude-method <id>   Train only this amplitude method
            --feature-set <id>        Train only this feature set
            --kernel <type>           SVM kernel: linear, rbf, poly (default: all)
            --C <value>               SVM C parameter (default: grid search)
            --gamma <value>           SVM gamma parameter (default: grid search)
            --use-linear-svc          Use LinearSVC for linear kernel (10-100x faster)
            --force                   Overwrite existing results

        Examples:
            classifier-train-svm
            classifier-train-svm --workers 21 --memory 80000
            classifier-train-svm --decimation-factor 0 --data-type 4
            classifier-train-svm --amplitude-method 2 --workers 20 --memory 90000
            classifier-train-svm --kernel linear --use-linear-svc
            classifier-train-svm --kernel rbf --C 10.0 --gamma 0.01
        """
        # Check session context
        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        num_workers = 7
        max_memory_mb = 16000  # Default 16GB total memory budget
        filter_dec = None
        filter_dtype = None
        filter_amp = None
        filter_efs = None
        svm_kernel = None
        svm_C = None
        svm_gamma = None
        use_linear_svc = False  # Use LinearSVC for linear kernel (10-100x faster)
        force = False

        i = 0
        while i < len(args):
            if args[i] == '--workers':
                if i + 1 < len(args):
                    num_workers = int(args[i + 1])
                    if num_workers < 1 or num_workers > 28:
                        print("[ERROR] --workers must be between 1 and 28")
                        return
                    i += 2
                else:
                    print("[ERROR] --workers requires a value")
                    return
            elif args[i] == '--memory':
                if i + 1 < len(args):
                    max_memory_mb = int(args[i + 1])
                    if max_memory_mb < 1000:
                        print("[ERROR] --memory must be at least 1000 MB (1GB)")
                        return
                    i += 2
                else:
                    print("[ERROR] --memory requires a value")
                    return
            elif args[i] == '--decimation-factor':
                if i + 1 < len(args):
                    filter_dec = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --decimation-factor requires a value")
                    return
            elif args[i] == '--data-type':
                if i + 1 < len(args):
                    filter_dtype = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --data-type requires a value")
                    return
            elif args[i] == '--amplitude-method':
                if i + 1 < len(args):
                    filter_amp = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --amplitude-method requires a value")
                    return
            elif args[i] == '--feature-set':
                if i + 1 < len(args):
                    filter_efs = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --feature-set requires a value")
                    return
            elif args[i] == '--kernel':
                if i + 1 < len(args):
                    svm_kernel = args[i + 1]
                    if svm_kernel not in ['linear', 'rbf', 'poly']:
                        print("[ERROR] --kernel must be linear, rbf, or poly")
                        return
                    i += 2
                else:
                    print("[ERROR] --kernel requires a value")
                    return
            elif args[i] == '--C':
                if i + 1 < len(args):
                    svm_C = float(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --C requires a value")
                    return
            elif args[i] == '--gamma':
                if i + 1 < len(args):
                    svm_gamma = args[i + 1]
                    # Can be 'scale', 'auto', or numeric
                    if svm_gamma not in ['scale', 'auto']:
                        try:
                            svm_gamma = float(svm_gamma)
                        except ValueError:
                            print("[ERROR] --gamma must be 'scale', 'auto', or numeric")
                            return
                    i += 2
                else:
                    print("[ERROR] --gamma requires a value")
                    return
            elif args[i] == '--use-linear-svc':
                use_linear_svc = True
                i += 1
            elif args[i] == '--force':
                force = True
                i += 1
            else:
                print(f"[ERROR] Unknown option: {args[i]}")
                return

        try:
            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            cls_id = self.current_classifier_id

            print(f"\n[INFO] Training SVM for Experiment {exp_id}, Classifier {cls_id}")
            print(f"[INFO] Using {num_workers} parallel workers")

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id, classifier_name
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (exp_id, cls_id))
            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {cls_id} not found for experiment {exp_id}")
                return

            global_classifier_id, classifier_name = result
            print(f"[INFO] Using classifier: {classifier_name}")

            # Check if active configuration exists
            cursor.execute("""
                SELECT config_id, config_name
                FROM ml_classifier_configs
                WHERE global_classifier_id = %s AND is_active = TRUE
            """, (global_classifier_id,))
            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] No active configuration found for classifier {cls_id}")
                print("Use 'classifier-config-activate <config_id>' to activate a configuration")
                return

            config_id, config_name = result
            print(f"[INFO] Using configuration: {config_name}")

            # Check if results tables exist
            results_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_results"
            per_class_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_per_class_results"
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (results_table_name,))
            table_exists = cursor.fetchone()[0]

            if not table_exists:
                print(f"[ERROR] Results table {results_table_name} does not exist")
                print("Run 'classifier-train-svm-init' first to create tables")
                return

            # Check for existing results with current filters
            if not force:
                # Build WHERE clause for filters
                where_conditions = []
                params = []

                if filter_dec is not None:
                    where_conditions.append("decimation_factor = %s")
                    params.append(filter_dec)
                if filter_dtype is not None:
                    where_conditions.append("data_type_id = %s")
                    params.append(filter_dtype)
                if filter_amp is not None:
                    where_conditions.append("amplitude_processing_method_id = %s")
                    params.append(filter_amp)
                if filter_efs is not None:
                    where_conditions.append("experiment_feature_set_id = %s")
                    params.append(filter_efs)
                if svm_kernel is not None:
                    where_conditions.append("svm_kernel = %s")
                    params.append(svm_kernel)
                if svm_C is not None:
                    where_conditions.append("svm_c_parameter = %s")
                    params.append(svm_C)

                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)

                cursor.execute(f"SELECT COUNT(*) FROM {results_table_name} {where_clause}", params)
                existing_count = cursor.fetchone()[0]

                if existing_count > 0:
                    print(f"\n[WARNING] {existing_count} existing results found for specified filters")
                    print("Use --force to overwrite and retrain these models")
                    return

            # Delete existing results if force is enabled
            if force:
                # Build WHERE clause for filters
                where_conditions = []
                params = []

                if filter_dec is not None:
                    where_conditions.append("decimation_factor = %s")
                    params.append(filter_dec)
                if filter_dtype is not None:
                    where_conditions.append("data_type_id = %s")
                    params.append(filter_dtype)
                if filter_amp is not None:
                    where_conditions.append("amplitude_processing_method_id = %s")
                    params.append(filter_amp)
                if filter_efs is not None:
                    where_conditions.append("experiment_feature_set_id = %s")
                    params.append(filter_efs)
                if svm_kernel is not None:
                    where_conditions.append("svm_kernel = %s")
                    params.append(svm_kernel)
                if svm_C is not None:
                    where_conditions.append("svm_c_parameter = %s")
                    params.append(svm_C)

                where_clause = ""
                if where_conditions:
                    where_clause = "WHERE " + " AND ".join(where_conditions)

                cursor.execute(f"SELECT COUNT(*) FROM {results_table_name} {where_clause}", params)
                existing_count = cursor.fetchone()[0]

                if existing_count > 0:
                    print(f"[INFO] Deleting {existing_count} existing results...")
                    cursor.execute(f"DELETE FROM {results_table_name} {where_clause}", params)
                    self.db_conn.commit()

            # Check if data splits exist
            splits_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_data_splits"
            cursor.execute(f"SELECT COUNT(*) FROM {splits_table_name}")
            split_count = cursor.fetchone()[0]
            if split_count == 0:
                print(f"[ERROR] No data splits found in {splits_table_name}")
                print("Run 'classifier-assign-splits' first")
                return

            print(f"[INFO] Found {split_count} split assignments")

            # Check if feature vectors exist
            features_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_features"
            cursor.execute(f"""
                SELECT COUNT(*) FROM {features_table_name}
                WHERE extraction_status_id = 2
            """)
            feature_count = cursor.fetchone()[0]
            if feature_count == 0:
                print(f"[ERROR] No feature vectors found in {features_table_name}")
                print("Run 'classifier-build-features' first")
                return

            print(f"[INFO] Found {feature_count} feature vectors")

            # Query hyperparameter combinations from active configuration
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

            # Apply filters if specified
            if filter_dec is not None:
                decimation_factors = [filter_dec] if filter_dec in decimation_factors else []
            if filter_dtype is not None:
                data_type_ids = [filter_dtype] if filter_dtype in data_type_ids else []
            if filter_amp is not None:
                amplitude_methods = [filter_amp] if filter_amp in amplitude_methods else []
            if filter_efs is not None:
                experiment_feature_sets = [filter_efs] if filter_efs in experiment_feature_sets else []

            if not decimation_factors or not data_type_ids or not amplitude_methods or not experiment_feature_sets:
                print("[ERROR] No matching hyperparameter combinations found with specified filters")
                return

            print(f"[INFO] Hyperparameters: {len(decimation_factors)} decimations, {len(data_type_ids)} data types, "
                  f"{len(amplitude_methods)} amp methods, {len(experiment_feature_sets)} feature sets")
            print(f"[DEBUG] Decimation factors: {decimation_factors}")
            print(f"[DEBUG] Data types: {data_type_ids}")
            print(f"[DEBUG] Amplitude methods: {amplitude_methods}")
            print(f"[DEBUG] Feature sets: {experiment_feature_sets}")

            # Build SVM parameter grid
            if svm_kernel is None:
                kernels = ['linear', 'rbf', 'poly']
            else:
                kernels = [svm_kernel]

            if svm_C is None:
                C_values = [0.1, 1.0, 10.0, 100.0]
            else:
                C_values = [svm_C]

            if svm_gamma is None:
                gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1]
            else:
                gamma_values = [svm_gamma]

            # Build grid
            svm_grid = []
            for k in kernels:
                for c in C_values:
                    if k in ['rbf', 'poly']:
                        for g in gamma_values:
                            svm_grid.append({'kernel': k, 'C': c, 'gamma': g})
                    else:
                        svm_grid.append({'kernel': k, 'C': c, 'gamma': None})

            print(f"[INFO] SVM parameter grid: {len(svm_grid)} combinations")

            # Query category mapping for binary classification
            cursor.execute("""
                SELECT label_id, category
                FROM segment_labels
                WHERE active = TRUE
                ORDER BY label_id
            """)
            label_categories = {row[0]: row[1] for row in cursor.fetchall()}
            print(f"[INFO] Loaded {len(label_categories)} label categories for binary arc detection")

            # Build work queue
            work_queue = []
            for dec in decimation_factors:
                for dtype in data_type_ids:
                    for amp in amplitude_methods:
                        for efs in experiment_feature_sets:
                            for svm_params in svm_grid:
                                work_queue.append((dec, dtype, amp, efs, svm_params))

            total_tasks = len(work_queue)
            print(f"\n[INFO] Total SVM training tasks: {total_tasks}")
            print(f"[INFO] Estimated time: {total_tasks * 1.5 / num_workers / 60:.1f} - {total_tasks * 3.0 / num_workers / 60:.1f} minutes")

            # Prepare database config for workers
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            # Use MPCCTL architecture for deadlock-free parallel training
            import multiprocessing as mp
            from pathlib import Path
            import sys
            import time
            import json

            # Add mpcctl_svm_trainer to path
            sys.path.insert(0, str(Path(__file__).parent))
            from mpcctl_svm_trainer import manager_process

            # Prepare mpcctl base directory
            mpcctl_base_dir = Path(f'/Volumes/ArcData/V3_database/experiment{exp_id:03d}')

            # Prepare filter dictionary
            filters = {
                'decimation_factor': filter_dec,
                'data_type': filter_dtype,
                'amplitude_method': filter_amp,
                'experiment_feature_set': filter_efs,
                'svm_kernel': svm_kernel,
                'svm_C': svm_C,
                'use_linear_svc': use_linear_svc
            }

            # Create timestamped log file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = mpcctl_base_dir / f"svm_training_{timestamp}.log"

            cache_size_mb = max_memory_mb // num_workers
            print(f"\n[INFO] Starting parallel SVM training with {num_workers} workers...")
            print("[INFO] Using MPCCTL architecture for deadlock-proof training")
            print(f"[INFO] Memory budget: {max_memory_mb:,} MB total ({cache_size_mb:,} MB per worker)")
            print(f"[INFO] Expected tasks: {total_tasks}")
            print(f"[INFO] Estimated time: {total_tasks * 1.5 / num_workers / 60:.1f} - {total_tasks * 3.0 / num_workers / 60:.1f} minutes\n")

            # Spawn manager process in background
            manager = mp.Process(
                target=manager_process,
                args=(exp_id, cls_id, num_workers, db_config, filters, log_file, True, mpcctl_base_dir, max_memory_mb)
            )
            manager.start()

            print(f"[INFO] Training manager started in background (PID {manager.pid})")
            print(f"[INFO] Waiting for initialization...\n")

            # Wait for state file to be created
            state_file = mpcctl_base_dir / ".mpcctl_state.json"
            max_wait = 10
            waited = 0
            while not state_file.exists() and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5

            if not state_file.exists():
                print("[WARNING] State file not created yet")
                print("Training is running in background")
                print(f"Monitor progress: cat {state_file}")
                return

            print("[INFO] Live Progress Monitor (Press Ctrl+C to detach)\n")

            # Live progress monitoring loop (similar to mpcctl_distance_insert pattern)
            try:
                last_status = None
                last_completed = 0
                while True:
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)

                        progress = state.get('progress', {})
                        status = state.get('status', 'unknown')
                        completed = progress.get('completed_tasks', 0)

                        # Only show update if progress changed
                        if completed != last_completed or status != last_status:
                            # Progress bar
                            bar_width = 50
                            percent = progress.get('percent_complete', 0)
                            filled = int(bar_width * percent / 100)
                            bar = '█' * filled + '░' * (bar_width - filled)

                            # Format ETA
                            eta_seconds = progress.get('estimated_time_remaining_seconds', 0)
                            eta_minutes = eta_seconds // 60
                            eta_seconds_remainder = eta_seconds % 60

                            # Clear previous output (move cursor up and clear lines)
                            if last_status is not None:
                                print('\033[6A\033[J', end='')

                            # Display progress
                            print(f"Status: {status}")
                            print(f"[{bar}] {percent:.1f}%")
                            print(f"Completed: {completed:,} / {progress.get('total_tasks', 0):,} configs")
                            print(f"Rate: {progress.get('configs_per_second', 0):.3f} configs/sec")
                            print(f"ETA: {eta_minutes} min {eta_seconds_remainder} sec")
                            print(f"Workers: {state.get('workers_count', 0)}")

                            last_status = status
                            last_completed = completed

                        # Check if completed or stopped
                        if status in ['completed', 'stopped', 'killed']:
                            print(f"\n[SUCCESS] Training {status}")
                            break

                        time.sleep(1.0)

                    except (json.JSONDecodeError, FileNotFoundError):
                        # State file might be being written
                        time.sleep(0.5)
                        continue

            except KeyboardInterrupt:
                print(f"\n\n[INFO] Detached from monitoring (training continues in background)")
                print(f"\n[INFO] Monitor progress:")
                print(f"   cat {state_file}")
                print(f"\n[INFO] Check .mpcctl folder:")
                print(f"   ls -la {mpcctl_base_dir}/.mpcctl/")
                return

            # Wait for manager to finish
            manager.join(timeout=5)
            if manager.is_alive():
                print("[WARNING] Manager still running, detaching...")

            cursor.close()

        except Exception as e:
            import traceback
            print(f"[ERROR] {e}")
            print(traceback.format_exc())

    def cmd_classifier_train_rf(self, args):
        """
        Train Random Forest classifier using distance-based feature vectors

        Usage: classifier-train-rf [OPTIONS]

        Trains Random Forest on feature vectors from Phase 3 using data splits from Phase 1.
        Evaluates on train/test/verification splits with both 13-class and binary
        arc detection metrics.

        Options:
            --workers <n>             Parallel workers (default: 7, max: 28)
            --decimation-factor <n>   Train only this decimation factor
            --data-type <id>          Train only this data type
            --amplitude-method <id>   Train only this amplitude method
            --feature-set <id>        Train only this feature set
            --n-estimators <n>        Number of trees (default: grid search)
            --max-depth <n>           Max tree depth (default: grid search)
            --min-samples-split <n>   Min samples to split (default: grid search)
            --max-features <val>      Max features per split (default: grid search)
            --force                   Overwrite existing results

        Examples:
            classifier-train-rf
            classifier-train-rf --workers 21
            classifier-train-rf --decimation-factor 0 --data-type 4
            classifier-train-rf --amplitude-method 2 --workers 20
            classifier-train-rf --n-estimators 200 --max-depth 30
        """
        # Check session context
        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        num_workers = 7
        filter_dec = None
        filter_dtype = None
        filter_amp = None
        filter_efs = None
        n_estimators = None
        max_depth = None
        min_samples_split = None
        max_features = None
        force = False

        i = 0
        while i < len(args):
            if args[i] == '--workers':
                if i + 1 < len(args):
                    num_workers = int(args[i + 1])
                    if num_workers < 1 or num_workers > 28:
                        print("[ERROR] --workers must be between 1 and 28")
                        return
                    i += 2
                else:
                    print("[ERROR] --workers requires a value")
                    return
            elif args[i] == '--decimation-factor':
                if i + 1 < len(args):
                    filter_dec = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --decimation-factor requires a value")
                    return
            elif args[i] == '--data-type':
                if i + 1 < len(args):
                    filter_dtype = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --data-type requires a value")
                    return
            elif args[i] == '--amplitude-method':
                if i + 1 < len(args):
                    filter_amp = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --amplitude-method requires a value")
                    return
            elif args[i] == '--feature-set':
                if i + 1 < len(args):
                    filter_efs = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --feature-set requires a value")
                    return
            elif args[i] == '--n-estimators':
                if i + 1 < len(args):
                    n_estimators = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --n-estimators requires a value")
                    return
            elif args[i] == '--max-depth':
                if i + 1 < len(args):
                    if args[i + 1].lower() == 'none':
                        max_depth = None
                    else:
                        max_depth = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --max-depth requires a value")
                    return
            elif args[i] == '--min-samples-split':
                if i + 1 < len(args):
                    min_samples_split = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --min-samples-split requires a value")
                    return
            elif args[i] == '--max-features':
                if i + 1 < len(args):
                    val = args[i + 1].lower()
                    if val == 'none':
                        max_features = None
                    elif val in ['sqrt', 'log2']:
                        max_features = val
                    else:
                        try:
                            max_features = int(val)
                        except ValueError:
                            print("[ERROR] --max-features must be 'sqrt', 'log2', 'none', or an integer")
                            return
                    i += 2
                else:
                    print("[ERROR] --max-features requires a value")
                    return
            elif args[i] == '--force':
                force = True
                i += 1
            else:
                print(f"[ERROR] Unknown option: {args[i]}")
                return

        try:
            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            cls_id = self.current_classifier_id

            print(f"\n[INFO] Training Random Forest for Experiment {exp_id}, Classifier {cls_id}")
            print(f"[INFO] Using {num_workers} parallel workers")

            # Get global_classifier_id
            cursor.execute("""
                SELECT global_classifier_id, classifier_name
                FROM ml_experiment_classifiers
                WHERE experiment_id = %s AND classifier_id = %s
            """, (exp_id, cls_id))
            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] Classifier {cls_id} not found for experiment {exp_id}")
                return

            global_classifier_id, classifier_name = result
            print(f"[INFO] Using classifier: {classifier_name}")

            # Check if active configuration exists
            cursor.execute("""
                SELECT config_id, config_name
                FROM ml_classifier_configs
                WHERE global_classifier_id = %s AND is_active = TRUE
            """, (global_classifier_id,))
            result = cursor.fetchone()
            if not result:
                print(f"[ERROR] No active configuration found for classifier {cls_id}")
                print("Use 'classifier-config-activate <config_id>' to activate a configuration")
                return

            config_id, config_name = result
            print(f"[INFO] Using configuration: {config_name}")

            # Check if results tables exist
            results_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_rf_results"
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (results_table_name,))
            table_exists = cursor.fetchone()[0]

            if not table_exists:
                print(f"[ERROR] Results table {results_table_name} does not exist")
                print(f"[INFO] Please create the RF results tables manually using the schema:")
                print(f"[INFO] See: documentation/reference/rf_results_schema.sql")
                return

            # Check for existing results (simplified check)
            if not force:
                cursor.execute(f"SELECT COUNT(*) FROM {results_table_name}")
                existing_count = cursor.fetchone()[0]
                if existing_count > 0:
                    print(f"[WARNING] Found {existing_count} existing results")
                    print("[INFO] Use --force to overwrite existing results")
                    return

            # Import and prepare
            from datetime import datetime
            import multiprocessing as mp
            from pathlib import Path
            import sys
            import time
            import json

            # Add mpcctl_rf_trainer to path
            sys.path.insert(0, str(Path(__file__).parent))
            from mpcctl_rf_trainer import manager_process

            # Prepare mpcctl base directory
            mpcctl_base_dir = Path(f'/Volumes/ArcData/V3_database/experiment{exp_id:03d}')

            # Prepare filter dictionary
            filters = {
                'decimation_factor': filter_dec,
                'data_type': filter_dtype,
                'amplitude_method': filter_amp,
                'experiment_feature_set': filter_efs,
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'max_features': max_features
            }

            # Create timestamped log file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = mpcctl_base_dir / f"rf_training_{timestamp}.log"

            # Database config
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            print(f"\n[INFO] Starting parallel Random Forest training with {num_workers} workers...")
            print("[INFO] Using MPCCTL architecture for deadlock-proof training")
            print(f"[INFO] Log file: {log_file}\n")

            # Spawn manager process in background
            manager = mp.Process(
                target=manager_process,
                args=(exp_id, cls_id, num_workers, db_config, filters, log_file, True, mpcctl_base_dir)
            )
            manager.start()

            print(f"[INFO] Training manager started in background (PID {manager.pid})")
            print(f"[INFO] Waiting for initialization...\n")

            # Wait for state file to be created
            state_file = mpcctl_base_dir / ".mpcctl_state.json"
            max_wait = 10
            waited = 0
            while not state_file.exists() and waited < max_wait:
                time.sleep(0.5)
                waited += 0.5

            if not state_file.exists():
                print("[WARNING] State file not created yet")
                print("Training is running in background")
                print(f"Monitor progress: cat {state_file}")
                return

            print("[INFO] Live Progress Monitor (Press Ctrl+C to detach)\n")

            # Live progress monitoring loop
            try:
                last_status = None
                last_completed = 0
                while True:
                    try:
                        with open(state_file, 'r') as f:
                            state = json.load(f)

                        progress = state.get('progress', {})
                        status = state.get('status', 'unknown')
                        completed = progress.get('completed_tasks', 0)

                        if completed != last_completed or status != last_status:
                            total = progress.get('total_tasks', 0)
                            pct = progress.get('percent_complete', 0.0)
                            rate = progress.get('configs_per_second', 0.0)
                            eta = progress.get('estimated_time_remaining_seconds', 0)

                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {completed}/{total} ({pct:.1f}%) | Rate: {rate:.2f} configs/s | ETA: {eta//60:.0f}m {eta%60:.0f}s")

                            last_completed = completed
                            last_status = status

                        if status == 'completed':
                            print("\n[INFO] Training completed successfully!")
                            break

                        time.sleep(2)

                    except json.JSONDecodeError:
                        time.sleep(0.5)
                    except KeyboardInterrupt:
                        print("\n\n[INFO] Detached from monitor (training continues in background)")
                        print(f"[INFO] Manager PID: {manager.pid}")
                        print(f"[INFO] Monitor progress: cat {state_file}")
                        break

            except Exception as e:
                print(f"[ERROR] Monitor error: {e}")

            cursor.close()

        except Exception as e:
            import traceback
            print(f"[ERROR] {e}")
            print(traceback.format_exc())

    def cmd_classifier_clean_svm_results(self, args):
        """
        Clear SVM training results from database

        Usage: classifier-clean-svm-results [OPTIONS]

        Deletes SVM results and per-class metrics from database.
        Use filters to delete specific subsets or delete all results.

        Options:
            --amplitude-method <id>   Delete only this amplitude method
            --decimation-factor <n>   Delete only this decimation factor
            --data-type <id>          Delete only this data type
            --feature-set <id>        Delete only this feature set
            --force                   Required to confirm deletion

        Examples:
            classifier-clean-svm-results --amplitude-method 2 --force
            classifier-clean-svm-results --decimation-factor 255 --force
            classifier-clean-svm-results --force
        """
        # Check session context
        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        filter_amp = None
        filter_dec = None
        filter_dtype = None
        filter_efs = None
        force = False

        i = 0
        while i < len(args):
            if args[i] == '--amplitude-method':
                if i + 1 < len(args):
                    filter_amp = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --amplitude-method requires a value")
                    return
            elif args[i] == '--decimation-factor':
                if i + 1 < len(args):
                    filter_dec = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --decimation-factor requires a value")
                    return
            elif args[i] == '--data-type':
                if i + 1 < len(args):
                    filter_dtype = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --data-type requires a value")
                    return
            elif args[i] == '--feature-set':
                if i + 1 < len(args):
                    filter_efs = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --feature-set requires a value")
                    return
            elif args[i] == '--force':
                force = True
                i += 1
            else:
                print(f"[ERROR] Unknown option: {args[i]}")
                return

        try:
            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            cls_id = self.current_classifier_id

            results_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_results"
            per_class_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_per_class_results"

            # Check if tables exist
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (results_table_name,))
            if not cursor.fetchone()[0]:
                print(f"[ERROR] Results table {results_table_name} does not exist")
                return

            # Build WHERE clause based on filters
            where_conditions = []
            where_params = []

            if filter_amp is not None:
                where_conditions.append("amplitude_processing_method_id = %s")
                where_params.append(filter_amp)
            if filter_dec is not None:
                where_conditions.append("decimation_factor = %s")
                where_params.append(filter_dec)
            if filter_dtype is not None:
                where_conditions.append("data_type_id = %s")
                where_params.append(filter_dtype)
            if filter_efs is not None:
                where_conditions.append("experiment_feature_set_id = %s")
                where_params.append(filter_efs)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            # Count what will be deleted
            cursor.execute(f"""
                SELECT COUNT(*) FROM {results_table_name}
                {where_clause}
            """, tuple(where_params))
            result_count = cursor.fetchone()[0]

            cursor.execute(f"""
                SELECT COUNT(*) FROM {per_class_table_name} pc
                JOIN {results_table_name} r ON pc.result_id = r.result_id
                {where_clause}
            """, tuple(where_params))
            per_class_count = cursor.fetchone()[0]

            # Show what will be deleted
            print(f"\n[INFO] Cleaning SVM results for Experiment {exp_id}, Classifier {cls_id}")
            if filter_amp:
                print(f"[INFO] Filter: amplitude_method_id = {filter_amp}")
            if filter_dec:
                print(f"[INFO] Filter: decimation_factor = {filter_dec}")
            if filter_dtype:
                print(f"[INFO] Filter: data_type_id = {filter_dtype}")
            if filter_efs:
                print(f"[INFO] Filter: experiment_feature_set_id = {filter_efs}")

            print(f"\n[WARNING] This will delete:")
            print(f"  - {result_count} SVM result records")
            print(f"  - {per_class_count} per-class metric records")

            if result_count == 0:
                print("\n[INFO] No results found matching criteria. Nothing to delete.")
                return

            if not force:
                print("\n[ERROR] --force flag required to proceed with deletion")
                print("Re-run with --force to confirm deletion")
                return

            # Delete per-class results first (FK dependency)
            if per_class_count > 0:
                cursor.execute(f"""
                    DELETE FROM {per_class_table_name}
                    WHERE result_id IN (
                        SELECT result_id FROM {results_table_name}
                        {where_clause}
                    )
                """, tuple(where_params))
                print(f"[INFO] Deleted {per_class_count} per-class metric records")

            # Delete main results
            cursor.execute(f"""
                DELETE FROM {results_table_name}
                {where_clause}
            """, tuple(where_params))
            print(f"[INFO] Deleted {result_count} SVM result records")

            self.db_conn.commit()
            print(f"\n[SUCCESS] SVM results cleaned successfully")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to clean SVM results: {e}")
            import traceback
            traceback.print_exc()

    def cmd_classifier_clean_features(self, args):
        """
        Clear feature vectors from database

        Usage: classifier-clean-features [OPTIONS]

        Deletes feature vector records from database.
        Use filters to delete specific subsets or delete all features.
        WARNING: This does not delete feature files from disk.

        Options:
            --amplitude-method <id>   Delete only this amplitude method
            --decimation-factor <n>   Delete only this decimation factor
            --data-type <id>          Delete only this data type
            --feature-set <id>        Delete only this feature set
            --force                   Required to confirm deletion

        Examples:
            classifier-clean-features --amplitude-method 2 --force
            classifier-clean-features --decimation-factor 255 --force
            classifier-clean-features --force
        """
        # Check session context
        if not self.current_experiment:
            print("[ERROR] No experiment selected. Use 'set experiment <id>' first.")
            return

        if not self.current_classifier_id:
            print("[ERROR] No classifier selected. Use 'set classifier <id>' first.")
            return

        # Parse arguments
        filter_amp = None
        filter_dec = None
        filter_dtype = None
        filter_efs = None
        force = False

        i = 0
        while i < len(args):
            if args[i] == '--amplitude-method':
                if i + 1 < len(args):
                    filter_amp = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --amplitude-method requires a value")
                    return
            elif args[i] == '--decimation-factor':
                if i + 1 < len(args):
                    filter_dec = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --decimation-factor requires a value")
                    return
            elif args[i] == '--data-type':
                if i + 1 < len(args):
                    filter_dtype = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --data-type requires a value")
                    return
            elif args[i] == '--feature-set':
                if i + 1 < len(args):
                    filter_efs = int(args[i + 1])
                    i += 2
                else:
                    print("[ERROR] --feature-set requires a value")
                    return
            elif args[i] == '--force':
                force = True
                i += 1
            else:
                print(f"[ERROR] Unknown option: {args[i]}")
                return

        try:
            cursor = self.db_conn.cursor()
            exp_id = self.current_experiment
            cls_id = self.current_classifier_id

            features_table_name = f"experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_features"

            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = %s
                )
            """, (features_table_name,))
            if not cursor.fetchone()[0]:
                print(f"[ERROR] Features table {features_table_name} does not exist")
                return

            # Build WHERE clause based on filters
            where_conditions = []
            where_params = []

            if filter_amp is not None:
                where_conditions.append("amplitude_processing_method_id = %s")
                where_params.append(filter_amp)
            if filter_dec is not None:
                where_conditions.append("decimation_factor = %s")
                where_params.append(filter_dec)
            if filter_dtype is not None:
                where_conditions.append("data_type_id = %s")
                where_params.append(filter_dtype)
            if filter_efs is not None:
                where_conditions.append("experiment_feature_set_id = %s")
                where_params.append(filter_efs)

            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)

            # Count what will be deleted
            cursor.execute(f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN extraction_status_id = 2 THEN 1 END) as extracted,
                    COUNT(CASE WHEN extraction_status_id = 3 THEN 1 END) as failed
                FROM {features_table_name}
                {where_clause}
            """, tuple(where_params))
            counts = cursor.fetchone()
            total_count, extracted_count, failed_count = counts

            # Show what will be deleted
            print(f"\n[INFO] Cleaning feature vectors for Experiment {exp_id}, Classifier {cls_id}")
            if filter_amp:
                print(f"[INFO] Filter: amplitude_method_id = {filter_amp}")
            if filter_dec:
                print(f"[INFO] Filter: decimation_factor = {filter_dec}")
            if filter_dtype:
                print(f"[INFO] Filter: data_type_id = {filter_dtype}")
            if filter_efs:
                print(f"[INFO] Filter: experiment_feature_set_id = {filter_efs}")

            print(f"\n[WARNING] This will delete:")
            print(f"  - {total_count} feature vector records")
            print(f"    - {extracted_count} extracted")
            print(f"    - {failed_count} failed")
            print(f"\n[NOTE] Feature files on disk will NOT be deleted")

            if total_count == 0:
                print("\n[INFO] No features found matching criteria. Nothing to delete.")
                return

            if not force:
                print("\n[ERROR] --force flag required to proceed with deletion")
                print("Re-run with --force to confirm deletion")
                return

            # Delete feature records
            cursor.execute(f"""
                DELETE FROM {features_table_name}
                {where_clause}
            """, tuple(where_params))

            self.db_conn.commit()
            print(f"\n[INFO] Deleted {total_count} feature vector records")
            print(f"[SUCCESS] Feature vectors cleaned successfully")

        except Exception as e:
            self.db_conn.rollback()
            print(f"\n[ERROR] Failed to clean features: {e}")
            import traceback
            traceback.print_exc()


    # ========== SVM Training Helper Functions (Phase 4) ==========

    def _load_feature_vectors_from_db(self, cursor, exp_id, cls_id, dec, dtype, amp, efs, split_type):
        """
        Load feature vectors for a specific configuration and split

        Args:
            cursor: Database cursor
            exp_id: Experiment ID
            cls_id: Classifier ID
            dec: Decimation factor
            dtype: Data type ID
            amp: Amplitude processing method ID
            efs: Experiment feature set ID
            split_type: Split type ('training', 'test', 'verification')

        Returns:
            X: Feature vectors (numpy array)
            y: Labels (numpy array)
        """
        import numpy as np

        # Query feature vectors from database
        cursor.execute(f"""
            SELECT sf.segment_id, sf.segment_label_id, sf.svm_feature_file_path
            FROM experiment_{exp_id:03d}_classifier_{cls_id:03d}_svm_features sf
            JOIN experiment_{exp_id:03d}_classifier_{cls_id:03d}_data_splits ds
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
            raise ValueError(f"No feature vectors found for config: dec={dec}, dtype={dtype}, amp={amp}, efs={efs}, split={split_type}")

        # Load feature vectors from .npy files
        X = []
        y = []
        failed_count = 0

        for segment_id, label_id, file_path in rows:
            try:
                features = np.load(file_path)
                X.append(features)
                y.append(label_id)
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:  # Only print first 5 errors
                    print(f"[WARNING] Failed to load {file_path}: {e}")

        if failed_count > 0:
            print(f"[WARNING] Failed to load {failed_count}/{len(rows)} feature vectors")

        if len(X) == 0:
            raise ValueError(f"No valid feature vectors loaded for split {split_type}")

        return np.array(X), np.array(y)

    def _compute_multiclass_metrics(self, y_true, y_pred):
        """
        Compute 13-class classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(y_true, y_pred)

        # Macro-averaged metrics (equal weight to each class)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        # Weighted metrics (weight by class support)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, zero_division=0)

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class': {
                'precision': precision_per_class,
                'recall': recall_per_class,
                'f1': f1_per_class,
                'support': support_per_class
            }
        }

    def _compute_binary_arc_metrics(self, y_true, y_pred, y_proba, label_categories):
        """
        Compute binary arc detection metrics (arc vs. non-arc)

        Args:
            y_true: True labels (13-class)
            y_pred: Predicted labels (13-class)
            y_proba: Prediction probabilities (13-class)
            label_categories: Dictionary mapping label_id to category

        Returns:
            Dictionary of binary metrics
        """
        import numpy as np
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                     confusion_matrix, roc_auc_score, average_precision_score)

        # Map 13-class labels to binary (1=arc, 0=non-arc)
        y_true_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_true])
        y_pred_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_pred])

        # Get probability for arc class (sum probabilities of all arc labels)
        unique_labels = np.unique(y_true)
        arc_label_indices = [i for i, label in enumerate(unique_labels)
                             if label_categories.get(int(label), 'unknown') == 'arc']

        if len(arc_label_indices) > 0 and y_proba.shape[1] >= len(unique_labels):
            y_proba_arc = np.sum(y_proba[:, arc_label_indices], axis=1)
        else:
            y_proba_arc = np.zeros(len(y_true))

        # Compute metrics
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary', zero_division=0
        )

        # Confusion matrix for specificity (TN / (TN + FP))
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            specificity = 0.0

        # ROC AUC
        try:
            if len(np.unique(y_true_binary)) > 1:
                roc_auc = roc_auc_score(y_true_binary, y_proba_arc)
            else:
                roc_auc = np.nan
        except Exception:
            roc_auc = np.nan

        # Precision-Recall AUC
        try:
            if len(np.unique(y_true_binary)) > 1:
                pr_auc = average_precision_score(y_true_binary, y_proba_arc)
            else:
                pr_auc = np.nan
        except Exception:
            pr_auc = np.nan

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': (tn, fp, fn, tp),
            'y_true': y_true_binary,
            'y_pred': y_pred_binary,
            'y_proba': y_proba_arc
        }

    def _save_confusion_matrices(self, y_train, y_pred_train, y_test, y_pred_test,
                                  y_verify, y_pred_verify, label_categories,
                                  exp_id, cls_id, dec, dtype, amp, efs, svm_params):
        """
        Save confusion matrices as PNG images

        Args:
            y_train, y_pred_train: Training labels and predictions
            y_test, y_pred_test: Test labels and predictions
            y_verify, y_pred_verify: Verification labels and predictions
            label_categories: Dictionary mapping label_id to category
            exp_id: Experiment ID
            cls_id: Classifier ID
            dec: Decimation factor
            dtype: Data type ID
            amp: Amplitude method ID
            efs: Experiment feature set ID
            svm_params: SVM parameters dictionary

        Returns:
            Dictionary of file paths
        """
        import os
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        base_dir = f"/Volumes/ArcData/V3_database/experiment{exp_id:03d}/classifier_files/svm_results"
        classifier_dir = f"classifier_{cls_id:03d}"
        config_dir = f"D{dec:06d}_TADC{dtype}_A{amp}_EFS{efs:03d}"
        svm_dir = f"{svm_params['kernel']}_C{svm_params['C']}"
        if svm_params.get('gamma'):
            svm_dir += f"_G{svm_params['gamma']}"

        full_dir = os.path.join(base_dir, classifier_dir, config_dir, svm_dir)
        os.makedirs(full_dir, exist_ok=True)

        paths = {}

        # 13-class confusion matrices
        for split_name, y_true, y_pred in [
            ('train', y_train, y_pred_train),
            ('test', y_test, y_pred_test),
            ('verify', y_verify, y_pred_verify)
        ]:
            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=(12, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'13-Class Confusion Matrix ({split_name.title()} Set)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            filename = f"confusion_matrix_13class_{split_name}.png"
            filepath = os.path.join(full_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            paths[f'cm_13class_{split_name}'] = filepath

        # Binary arc detection confusion matrices
        for split_name, y_true, y_pred in [
            ('train', y_train, y_pred_train),
            ('test', y_test, y_pred_test),
            ('verify', y_verify, y_pred_verify)
        ]:
            y_true_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_true])
            y_pred_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_pred])

            cm = confusion_matrix(y_true_binary, y_pred_binary)

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                       xticklabels=['Non-arc', 'Arc'],
                       yticklabels=['Non-arc', 'Arc'])
            plt.title(f'Binary Arc Detection ({split_name.title()} Set)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            filename = f"confusion_matrix_binary_{split_name}.png"
            filepath = os.path.join(full_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            paths[f'cm_binary_{split_name}'] = filepath

        return paths

    def _save_curves(self, y_train, y_proba_train, y_test, y_proba_test,
                     y_verify, y_proba_verify, label_categories,
                     exp_id, cls_id, dec, dtype, amp, efs, svm_params):
        """
        Save ROC and Precision-Recall curves

        Args:
            y_train, y_proba_train: Training labels and probabilities
            y_test, y_proba_test: Test labels and probabilities
            y_verify, y_proba_verify: Verification labels and probabilities
            label_categories: Dictionary mapping label_id to category
            exp_id: Experiment ID
            cls_id: Classifier ID
            dec: Decimation factor
            dtype: Data type ID
            amp: Amplitude method ID
            efs: Experiment feature set ID
            svm_params: SVM parameters dictionary

        Returns:
            Dictionary of file paths
        """
        import os
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

        base_dir = f"/Volumes/ArcData/V3_database/experiment{exp_id:03d}/classifier_files/svm_results"
        classifier_dir = f"classifier_{cls_id:03d}"
        config_dir = f"D{dec:06d}_TADC{dtype}_A{amp}_EFS{efs:03d}"
        svm_dir = f"{svm_params['kernel']}_C{svm_params['C']}"
        if svm_params.get('gamma'):
            svm_dir += f"_G{svm_params['gamma']}"

        full_dir = os.path.join(base_dir, classifier_dir, config_dir, svm_dir)
        os.makedirs(full_dir, exist_ok=True)

        paths = {}

        # Get arc probability for each split
        unique_labels = np.unique(y_train)
        arc_label_indices = [i for i, label in enumerate(unique_labels)
                             if label_categories.get(int(label), 'unknown') == 'arc']

        for split_name, y_true, y_proba in [
            ('train', y_train, y_proba_train),
            ('test', y_test, y_proba_test),
            ('verify', y_verify, y_proba_verify)
        ]:
            y_true_binary = np.array([1 if label_categories.get(int(y), 'unknown') == 'arc' else 0 for y in y_true])

            if len(arc_label_indices) > 0 and y_proba.shape[1] >= len(unique_labels):
                y_proba_arc = np.sum(y_proba[:, arc_label_indices], axis=1)
            else:
                y_proba_arc = np.zeros(len(y_true))

            # Skip if only one class present
            if len(np.unique(y_true_binary)) <= 1:
                continue

            # ROC Curve
            try:
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba_arc)
                roc_auc = roc_auc_score(y_true_binary, y_proba_arc)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - Arc Detection ({split_name.title()} Set)')
                plt.legend()
                plt.grid(True, alpha=0.3)

                filename = f"roc_curve_binary_{split_name}.png"
                filepath = os.path.join(full_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()

                paths[f'roc_binary_{split_name}'] = filepath
            except Exception as e:
                print(f"[WARNING] Failed to create ROC curve for {split_name}: {e}")

            # Precision-Recall Curve
            try:
                precision, recall, _ = precision_recall_curve(y_true_binary, y_proba_arc)
                pr_auc = average_precision_score(y_true_binary, y_proba_arc)

                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - Arc Detection ({split_name.title()} Set)')
                plt.legend()
                plt.grid(True, alpha=0.3)

                filename = f"pr_curve_binary_{split_name}.png"
                filepath = os.path.join(full_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()

                paths[f'pr_binary_{split_name}'] = filepath
            except Exception as e:
                print(f"[WARNING] Failed to create PR curve for {split_name}: {e}")

        return paths

    def _save_svm_model(self, svm_model, exp_id, cls_id, dec, dtype, amp, efs, svm_params):
        """
        Save trained SVM model to disk

        Args:
            svm_model: Trained SVM model
            exp_id: Experiment ID
            cls_id: Classifier ID
            dec: Decimation factor
            dtype: Data type ID
            amp: Amplitude method ID
            efs: Experiment feature set ID
            svm_params: SVM parameters dictionary

        Returns:
            Model file path
        """
        import os
        import joblib

        base_dir = f"/Volumes/ArcData/V3_database/experiment{exp_id:03d}/classifier_files/svm_models"
        classifier_dir = f"classifier_{cls_id:03d}"
        config_dir = f"D{dec:06d}_TADC{dtype}_A{amp}_EFS{efs:03d}"

        full_dir = os.path.join(base_dir, classifier_dir, config_dir)
        os.makedirs(full_dir, exist_ok=True)

        # Create filename
        filename = f"svm_{svm_params['kernel']}_C{svm_params['C']}"
        if svm_params.get('gamma'):
            filename += f"_G{svm_params['gamma']}"
        filename += ".pkl"

        filepath = os.path.join(full_dir, filename)

        # Save model
        joblib.dump(svm_model, filepath, compress=3)

        return filepath


    def cmd_exit(self, args):
        """Exit the shell"""
        if self.db_conn:
            self.db_conn.close()
        print("\nGoodbye! Thank you for using MLDP.")
        self.running = False
    
    # ========== Server Management Commands ==========
    
    def cmd_servers(self, args):
        """Server management - show help for server commands"""
        if args and args[0] in ['start', 'stop', 'restart', 'status', 'logs']:
            # Handle subcommands
            if args[0] == 'start':
                self.cmd_servers_start([])
            elif args[0] == 'stop':
                self.cmd_servers_stop([])
            elif args[0] == 'restart':
                self.cmd_servers_restart([])
            elif args[0] == 'status':
                self.cmd_servers_status([])
            elif args[0] == 'logs':
                self.cmd_servers_logs(args[1:])
        else:
            print("""
🖥️  Server Management Commands:
────────────────────────────────
  servers start    - Start all MLDP servers
  servers stop     - Stop all MLDP servers
  servers restart  - Restart all MLDP servers
  servers status   - Check status of all servers
  servers logs     - View server logs
  
  Shortcuts:
  start            - Start all servers
  stop             - Stop all servers
  restart          - Restart all servers
  status           - Check server status
  logs [service]   - View logs
""")
    
    def cmd_servers_start(self, args):
        """Start all MLDP servers"""
        scripts_path = MLDP_ROOT / "scripts" / "start_services.sh"
        
        if not scripts_path.exists():
            print(f"❌ start_services.sh not found at {scripts_path}")
            return
        
        print("🚀 Starting all MLDP servers...")
        print("This may take a moment...")
        print("─" * 60)
        
        try:
            result = subprocess.run(
                ["bash", str(scripts_path)],
                capture_output=False,
                text=True,
                cwd=str(MLDP_ROOT)
            )
            if result.returncode == 0:
                print("\n✅ All servers started successfully!")
                print("\nUse 'status' to check server status")
            else:
                print("\n⚠️  Some servers may have failed to start")
                print("Use 'status' to check which services are running")
        except Exception as e:
            print(f"❌ Error starting servers: {e}")
    
    def cmd_servers_stop(self, args):
        """Stop all MLDP servers"""
        scripts_path = MLDP_ROOT / "scripts" / "stop_services.sh"
        
        if not scripts_path.exists():
            print(f"❌ stop_services.sh not found at {scripts_path}")
            return
        
        print("🛑 Stopping all MLDP servers...")
        
        try:
            result = subprocess.run(
                ["bash", str(scripts_path)],
                capture_output=True,
                text=True,
                cwd=str(MLDP_ROOT)
            )
            print(result.stdout)
            if result.returncode == 0:
                print("✅ All servers stopped successfully!")
            else:
                print("⚠️  Some servers may still be running")
                print("Use 'status' to check")
        except Exception as e:
            print(f"❌ Error stopping servers: {e}")
    
    def cmd_servers_restart(self, args):
        """Restart all MLDP servers"""
        print("🔄 Restarting all MLDP servers...")
        print("─" * 60)
        
        # Stop servers
        self.cmd_servers_stop([])
        
        # Wait
        import time
        print("\n⏳ Waiting for services to shut down...")
        time.sleep(3)
        
        # Start servers
        self.cmd_servers_start([])
    
    def cmd_servers_status(self, args):
        """Check status of all MLDP servers"""
        operation_pid_path = MLDP_ROOT / "operation" / "pid"
        
        services = [
            ("real_time_sync_hub", 5035, "Real-Time Sync Hub"),
            ("database_browser", 5020, "Database Browser"),
            ("data_cleaning_tool", 5030, "Data Cleaning Tool"),
            ("transient_viewer", 5031, "Transient Viewer"),
            ("segment_visualizer", 5032, "Segment Visualizer"),
            ("distance_visualizer", 5037, "Distance Visualizer"),
            ("experiment_generator", 5040, "ML Experiment Generator"),
            ("jupyter_integration", 5041, "Jupyter Integration"),
            ("segment_verifier", 5034, "Segment Verifier"),
        ]
        
        print("\n📊 MLDP Server Status")
        print("=" * 70)
        print(f"{'Service':<30} {'Port':<8} {'PID':<10} {'Status':<15} {'URL'}")
        print("-" * 70)
        
        running_count = 0
        total_count = len(services)
        
        for service_name, port, display_name in services:
            pid_file = operation_pid_path / f"{service_name}.pid"
            
            status = "❓ Unknown"
            pid_str = "-"
            url = f"http://localhost:{port}"
            
            if pid_file.exists():
                try:
                    with open(pid_file, 'r') as f:
                        pid_str = f.read().strip()
                    
                    # Check if process is running using ps command
                    result = subprocess.run(
                        ["ps", "-p", pid_str],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        status = "✅ Running"
                        running_count += 1
                    else:
                        status = "❌ Not Running"
                        url = "-"
                except Exception:
                    status = "❌ Error"
                    url = "-"
            else:
                status = "⏹️  Stopped"
                url = "-"
            
            print(f"{display_name:<30} {port:<8} {pid_str:<10} {status:<15} {url}")
        
        print("-" * 70)
        print(f"Summary: {running_count}/{total_count} services running")
        
        if running_count == total_count:
            print("\n🎉 All services are running!")
        elif running_count == 0:
            print("\n⚠️  No services are running. Use 'start' to start them.")
        else:
            print(f"\n⚠️  Only {running_count}/{total_count} services are running.")
            print("Use 'restart' to restart all services.")
    
    def cmd_servers_logs(self, args):
        """View server logs"""
        logs_path = MLDP_ROOT / "operation" / "logs"
        
        if args and len(args) > 0:
            service = args[0]
            lines = int(args[1]) if len(args) > 1 else 50
            
            log_file = logs_path / f"{service}.log"
            if log_file.exists():
                print(f"\n📋 Last {lines} lines of {service}.log:")
                print("=" * 60)
                result = subprocess.run(
                    ["tail", f"-{lines}", str(log_file)],
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
            else:
                print(f"❌ Log file not found: {log_file}")
                print("\nAvailable services:")
                for log_file in sorted(logs_path.glob("*.log")):
                    print(f"  • {log_file.stem}")
        else:
            # Show available log files
            print("\n📁 Available log files:")
            print("=" * 60)
            if logs_path.exists():
                log_files = list(logs_path.glob("*.log"))
                if log_files:
                    for log_file in sorted(log_files):
                        size = log_file.stat().st_size
                        size_str = f"{size / 1024:.1f}K" if size < 1024*1024 else f"{size / (1024*1024):.1f}M"
                        print(f"  {log_file.stem:<30} {size_str:>10}")
                    print("\nUsage: logs <service> [lines]")
                    print("Example: logs real_time_sync_hub 100")
                else:
                    print("No log files found")
            else:
                print("❌ Logs directory not found")
    
    def cmd_segment_generate(self, args):
        """Generate segment fileset for experiment"""
        try:
            from .segment_processor import SegmentFilesetProcessor
        except ImportError:
            # Fallback for when running as script
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from segment_processor import SegmentFilesetProcessor
        
        # Parse arguments
        # args is already a list from the shell parser
        parts = args if isinstance(args, list) else args.split()
        if not parts or parts[0] != 'exp18':
            print("Usage: segment-generate exp18 [options]")
            print("Options:")
            print("  --files <range>   File range (e.g., 200-210)")
            print("  --types <list>    Data types (comma-separated)")
            print("  --decimations <list>  Decimation factors (comma-separated)")
            print("  --sizes <list>    Segment sizes to process (comma-separated)")
            print("                    Available: 8192,32768,65536,262144,524288")
            return
        
        # Check for options
        file_range = None
        data_types = None
        decimations = None
        sizes = None
        
        for i, part in enumerate(parts):
            if part == '--files' and i + 1 < len(parts):
                file_range = parts[i + 1]
            elif part == '--types' and i + 1 < len(parts):
                data_types = parts[i + 1].split(',')
            elif part == '--decimations' and i + 1 < len(parts):
                decimations = [int(d) for d in parts[i + 1].split(',')]
            elif part == '--sizes' and i + 1 < len(parts):
                sizes = [int(s) for s in parts[i + 1].split(',')]
        
        # Use defaults for experiment 18
        if decimations is None:
            decimations = [1, 3, 7, 15, 31, 63, 127, 255, 511]
        if data_types is None:
            data_types = ['ADC14', 'ADC12', 'ADC10', 'ADC8', 'ADC6']
        
        # Note: decimation 0 means no decimation (keep all samples)
        
        print("\n" + "="*70)
        print("Starting Experiment 18 Segment Generation")
        print("="*70)
        print(f"Decimations: {decimations}")
        print(f"Data Types: {data_types}")
        print(f"File Range: {file_range if file_range else 'all files'}")
        print(f"Segment Sizes: {sizes if sizes else 'all available (8192,32768,65536,262144,524288)'}")
        
        # Estimate file count
        if file_range:
            parts = file_range.split('-')
            if len(parts) == 2:
                num_files = int(parts[1]) - int(parts[0]) + 1
            else:
                num_files = 1
        else:
            num_files = 750  # Approximate total files in experiment 18
        
        # Estimate segments per file based on sizes filter
        if sizes:
            # Rough estimate based on typical distribution when filtering by size
            segments_per_file = len(sizes) * 2  # ~2 segments per size per file on average
        else:
            segments_per_file = 13  # Average when processing all sizes
        
        estimated_files = num_files * segments_per_file * len(decimations) * len(data_types)
        print(f"\nEstimated files to generate: ~{estimated_files:,}")
        
        # Confirm
        response = input("\nProceed? (y/n): ")
        if response.lower() != 'y':
            print("Generation cancelled.")
            return
        
        # Create processor and run
        print("\nInitializing processor...")
        processor = SegmentFilesetProcessor(experiment_id=18)
        
        print("Starting generation (this may take several hours)...")
        stats = processor.generate(
            decimations=decimations,
            data_types=data_types,
            file_range=file_range,
            sizes=sizes,
            workers=16
        )
        
        print("\n✅ Generation complete!")

    def cmd_generate_segment_fileset(self, args):
        """Generate physical segment files from raw data on disk

        This command creates the actual segment files on disk by processing
        raw data files. It performs decimation and data type conversions.

        Note: This is different from generate-training-data which only
        creates database tables for tracking which segments to use.
        """
        # Show help if requested
        if '--help' in args:
            print("\nUsage: generate-segment-fileset [options]")
            print("\nThis command generates physical segment files from raw data.")
            print("\nBy default, uses the experiment's configured data types and decimations.")
            print("\nOptions:")
            print("  --data-types <list>      Override data types (RAW,ADC14,ADC12,ADC10,ADC8,ADC6)")
            print("  --decimations <list>     Override decimation factors (0=none, comma-separated)")
            print("  --max-segments N         Maximum segments to process")
            print("  --clean                  Delete progress file and regenerate all segments")
            print("  --workers N              Number of parallel workers (default: 1)")
            print("\nNote: If no --data-types or --decimations are specified, uses experiment config.")
            print("\nExamples:")
            print("  generate-segment-fileset")
            print("  generate-segment-fileset --data-types RAW")
            print("  generate-segment-fileset --data-types RAW,ADC14 --decimations 0,7,15")
            print("\n📝 Pipeline Order:")
            print("  1. select-files          - Select files for training (DB)")
            print("  2. select-segments       - Select segments for training (DB)")
            print("  3. generate-training-data - Create training data tables (DB)")
            print("  4. generate-segment-fileset - Create physical segment files (Disk)")
            print("  5. generate-feature-fileset - Extract features from segments (Disk)")
            print("\n📁 Output Structure:")
            print("  experiment{NNN}/segment_files/S{size}/T{type}/D{decimation}/*.npy")
            return

        # Determine experiment_id: use current experiment or first arg if it's a number
        experiment_id = None
        arg_offset = 0

        if args and args[0].isdigit():
            # Legacy support: first arg is experiment_id
            experiment_id = int(args[0])
            arg_offset = 1
        elif self.current_experiment:
            # Use current experiment set via 'set experiment'
            experiment_id = self.current_experiment
        else:
            print("❌ No experiment specified. Use 'set experiment <id>' first or provide experiment_id as argument.")
            return

        # Special handling for experiment 18 with legacy code
        if experiment_id == 18 and '--files' in args:
            print(f"🔄 Using legacy generator for experiment 18...")
            self.cmd_segment_generate(args)
            return

        # Use new generator for all experiments
        print(f"🔄 Generating segment fileset for experiment {experiment_id}...")

        # Parse arguments - only use if explicitly provided
        data_types = None  # Will use experiment config if not specified
        decimations = None  # Will use experiment config if not specified
        max_segments = None
        clean_mode = False
        workers = 1
        use_experiment_config = True

        i = arg_offset
        while i < len(args):
            if args[i] == '--data-types' and i + 1 < len(args):
                data_types = [dt.upper() for dt in args[i + 1].split(',')]
                use_experiment_config = False
                i += 2
            elif args[i] == '--decimations' and i + 1 < len(args):
                decimations = [int(d) for d in args[i + 1].split(',')]
                use_experiment_config = False
                i += 2
            elif args[i] == '--max-segments' and i + 1 < len(args):
                max_segments = int(args[i + 1])
                i += 2
            elif args[i] == '--clean':
                clean_mode = True
                i += 1
            elif args[i] == '--workers' and i + 1 < len(args):
                workers = int(args[i + 1])
                i += 2
            else:
                i += 1

        if use_experiment_config:
            print(f"📋 Using experiment {experiment_id} configuration (data types & decimations)")
        else:
            if data_types:
                print(f"📋 Using custom data types: {data_types}")
            if decimations is not None:
                print(f"📋 Using custom decimations: {decimations}")

        try:
            from experiment_segment_fileset_generator_v2 import ExperimentSegmentFilesetGeneratorV2
            from pathlib import Path

            # Database configuration
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'arc_detection',
                'user': 'kjensen'
            }

            # Handle --clean flag: delete progress file before creating generator
            if clean_mode:
                # Determine segment path (same logic as generator __init__)
                import psycopg2
                conn = psycopg2.connect(**db_config)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT segment_data_base_path
                    FROM ml_experiments
                    WHERE experiment_id = %s
                """, (experiment_id,))
                result = cursor.fetchone()
                cursor.close()
                conn.close()

                if result and result[0]:
                    segment_path = Path(result[0])
                else:
                    segment_path = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}/segment_files')

                progress_file = segment_path / 'generation_progress.json'
                if progress_file.exists():
                    progress_file.unlink()
                    print(f"🗑️  Deleted progress file: {progress_file}")

            # Create generator
            generator = ExperimentSegmentFilesetGeneratorV2(experiment_id, db_config)

            # Pre-flight check: show what will be generated
            if not self._show_segment_generation_plan(experiment_id, data_types, decimations):
                print("❌ Cancelled by user")
                return

            # Generate fileset - pass None to use experiment config
            result = generator.generate_segment_fileset(
                data_types=data_types,  # None = use experiment config
                decimations=decimations,  # None = use experiment config
                max_segments=max_segments,
                parallel_workers=workers
            )

            if result.get('files_created', 0) > 0:
                print(f"\n✅ Successfully generated segment files!")
                print(f"   Files created: {result['files_created']}")
                print(f"   Files skipped: {result['files_skipped']}")
                print(f"   Segments processed: {result['segments_processed']}")
                print(f"   Output path: {generator.segment_path}")
            else:
                print(f"\n❌ No segment files generated")
                print(f"   Files failed: {result.get('files_failed', 0)}")

        except ImportError:
            print("❌ ExperimentSegmentFilesetGeneratorV2 module not found")
            print("   Make sure experiment_segment_fileset_generator_v2.py is in the same directory")
        except Exception as e:
            print(f"❌ Error generating segment fileset: {e}")

    def _show_segment_generation_plan(self, experiment_id, override_data_types=None, override_decimations=None):
        """Show pre-flight information about what will be generated

        Returns True if user confirms, False if cancelled
        """
        try:
            import psycopg2

            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                database='arc_detection',
                user='kjensen'
            )
            cursor = conn.cursor()

            # Get total segment count
            cursor.execute(f"""
                SELECT COUNT(*) as total_count
                FROM experiment_{experiment_id:03d}_segment_training_data
            """)
            total_segments = cursor.fetchone()[0]

            # Verify all segments have the same configured size
            cursor.execute(f"""
                SELECT DISTINCT ds.segment_length
                FROM experiment_{experiment_id:03d}_segment_training_data std
                JOIN data_segments ds ON std.segment_id = ds.segment_id
            """)
            distinct_sizes = [row[0] for row in cursor.fetchall()]

            if len(distinct_sizes) != 1:
                print(f"⚠️  Warning: Segments have multiple sizes: {distinct_sizes}")
                print(f"   Expected all segments to be the same size")
                cursor.close()
                conn.close()
                return False

            segment_size = distinct_sizes[0]

            if total_segments == 0:
                print("❌ No segments selected for this experiment")
                print("   Run 'select-segments' first")
                cursor.close()
                conn.close()
                return False

            # Get configured data types (or use override)
            if override_data_types:
                data_types = override_data_types
            else:
                cursor.execute("""
                    SELECT DISTINCT dt.data_type_name
                    FROM ml_experiments_data_types edt
                    JOIN ml_data_types_lut dt ON edt.data_type_id = dt.data_type_id
                    WHERE edt.experiment_id = %s
                    ORDER BY dt.data_type_name
                """, (experiment_id,))
                data_types = [row[0].upper() for row in cursor.fetchall()]

            # Get configured decimations (or use override)
            if override_decimations is not None:
                decimations = override_decimations
            else:
                cursor.execute("""
                    SELECT d.decimation_factor
                    FROM ml_experiment_decimation_junction ed
                    JOIN ml_experiment_decimation_lut d ON ed.decimation_id = d.decimation_id
                    WHERE ed.experiment_id = %s
                    ORDER BY d.decimation_factor
                """, (experiment_id,))
                decimations = [row[0] for row in cursor.fetchall()]

            cursor.close()
            conn.close()

            # Calculate totals
            num_data_types = len(data_types)
            num_decimations = len(decimations)
            total_directories = num_data_types * num_decimations  # One size only
            total_files = total_segments * num_data_types * num_decimations

            # Display pre-flight information
            print(f"\n{'='*80}")
            print(f"📋 SEGMENT GENERATION PLAN - Experiment {experiment_id}")
            print(f"{'='*80}")

            print(f"\n📊 Input Configuration:")
            print(f"   Total segments selected: {total_segments:,}")
            print(f"   Segment size: {segment_size} samples")
            print(f"      - All {total_segments:,} segments are size {segment_size}")

            print(f"\n🔧 Processing Configuration:")
            print(f"   Data types ({num_data_types}): {', '.join(data_types)}")
            print(f"   Decimation factors ({num_decimations}): {', '.join(map(str, decimations))}")

            print(f"\n📁 Output Structure:")
            print(f"   Directory pattern: S{segment_size:06d}/T{{type}}/D{{decimation:06d}}/")
            print(f"   Total directories to create: {total_directories:,}")
            print(f"      ({num_data_types} data types × {num_decimations} decimations)")
            print(f"   Examples:")
            for dt in data_types[:2]:  # Show first 2 data types
                for dec in decimations[:2]:  # Show first 2 decimations
                    print(f"      - S{segment_size:06d}/T{dt}/D{dec:06d}/")
            examples_shown = min(len(data_types), 2) * min(len(decimations), 2)
            if total_directories > examples_shown:
                print(f"      ... and {total_directories - examples_shown} more directories")

            print(f"\n📄 Files to Generate:")
            print(f"   Files per directory: {total_segments:,} files")
            print(f"   (Each directory contains all {total_segments:,} segments processed with one type/decimation combination)")
            print(f"   Total files: {total_segments:,} segments × {total_directories} directories = {total_files:,} files")

            print(f"\n🎯 TOTAL FILES TO CREATE: {total_files:,}")
            print(f"   ({total_segments:,} segments × {num_data_types} data types × {num_decimations} decimations)")

            print(f"\n{'='*80}")
            response = input("\nDo you wish to continue? (Y/n): ").strip().lower()

            if response == '' or response == 'y' or response == 'yes':
                return True
            else:
                return False

        except Exception as e:
            print(f"❌ Error calculating generation plan: {e}")
            import traceback
            traceback.print_exc()
            return False

    def cmd_show_segment_status(self, args):
        """Check segment generation status"""
        from pathlib import Path
        import json
        
        base_path = Path('/Volumes/ArcData/V3_database/experiment018/segment_files')
        progress_file = base_path / 'generation_progress.json'
        
        print("\n📊 Segment Generation Status")
        print("="*60)
        
        # Check progress file
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
                completed = progress.get('completed', [])
                print(f"Segments processed: {len(completed):,}")
        else:
            print("No generation in progress")
        
        # Count existing files
        total_files = 0
        for pattern in ['S*/T*/D*/*.npy']:
            files = list(base_path.glob(pattern))
            total_files += len(files)
        
        print(f"Total segment files: {total_files:,}")
        
        # Show breakdown by size
        print("\nBreakdown by segment size:")
        for size in [8192, 32768, 65536, 131072, 262144, 524288]:
            size_files = list(base_path.glob(f"S{size:06d}/*/*/*.npy"))
            if size_files:
                print(f"  {size:7d} samples: {len(size_files):,} files")
        
        # Show breakdown by type
        print("\nBreakdown by data type:")
        for data_type in ['TRAW', 'TADC14', 'TADC12', 'TADC10', 'TADC8', 'TADC6']:
            type_files = list(base_path.glob(f"*/T{data_type}/*/*.npy"))
            if type_files:
                print(f"  {data_type}: {len(type_files):,} files")
    
    def cmd_segment_test(self, args):
        """Test segment generation with small dataset"""
        try:
            from .segment_processor import SegmentFilesetProcessor
        except ImportError:
            # Fallback for when running as script
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from segment_processor import SegmentFilesetProcessor
        
        print("\n🧪 Testing Segment Generation")
        print("="*60)
        print("Test parameters:")
        print("  Files: 200-201 (2 files)")
        print("  Decimations: [1, 3]")
        print("  Data Types: [ADC12, ADC8]")
        print("  Expected files: ~104 (2 files × 13 segments × 2 decimations × 2 types)")
        
        response = input("\nRun test? (y/n): ")
        if response.lower() != 'y':
            print("Test cancelled.")
            return
        
        print("\nRunning test...")
        processor = SegmentFilesetProcessor(experiment_id=18)
        
        stats = processor.generate(
            decimations=[1, 3],
            data_types=['ADC12', 'ADC8'],
            file_range='200-201',
            workers=2
        )
        
        print("\n✅ Test complete!")
    
    def cmd_validate_segments(self, args):
        """Validate generated segment files"""
        import numpy as np
        from pathlib import Path
        
        base_path = Path('/Volumes/ArcData/V3_database/experiment018/segment_files')
        
        print("\n🔍 Validating Segment Files")
        print("="*60)
        
        # Sample some files
        sample_files = list(base_path.glob("*/T*/*/*.npy"))[:10]
        
        if not sample_files:
            print("No segment files found to validate")
            return
        
        print(f"Validating {len(sample_files)} sample files...")
        
        for filepath in sample_files:
            try:
                data = np.load(filepath)
                size = data.shape[0]
                is_power_of_2 = (size & (size - 1)) == 0
                
                # Parse filename
                filename = filepath.name
                parts = filename.split('_')
                segment_id = parts[0]
                file_id = parts[1]
                data_type = parts[3]
                
                status = "✅" if is_power_of_2 else "❌"
                print(f"{status} {filename[:40]:<40} Shape: {data.shape}, 2^N: {is_power_of_2}")
                
            except Exception as e:
                print(f"❌ Error validating {filepath.name}: {e}")

    def cmd_feature_plot(self, args):
        """Plot feature files with statistical visualization

        Usage: feature-plot [options]

        Options:
            --file <path>           Path to feature file (.npy)
            --output-folder <path>  Output directory for plots
            --save <filename>       Save to specific filename (overrides --output-folder)

        Examples:
            feature-plot --file /path/to/feature.npy
            feature-plot --file /path/to/feature.npy --output-folder ~/plots/
            feature-plot --file /path/to/feature.npy --save ~/plots/my_feature.png
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from pathlib import Path

        # Parse arguments
        parts = args if isinstance(args, list) else args.split()

        file_path = None
        output_folder = None
        save_path = None

        i = 0
        while i < len(parts):
            if parts[i] == '--file' and i + 1 < len(parts):
                file_path = parts[i + 1]
                i += 2
            elif parts[i] == '--output-folder' and i + 1 < len(parts):
                output_folder = parts[i + 1]
                i += 2
            elif parts[i] == '--save' and i + 1 < len(parts):
                save_path = parts[i + 1]
                i += 2
            else:
                i += 1

        # Validate required parameters
        if not file_path:
            print("❌ Error: --file is required")
            print("\nUsage: feature-plot --file <path> [--output-folder <path>] [--save <filename>]")
            print("\nExample:")
            print("  feature-plot --file /Volumes/ArcData/V3_database/experiment041/feature_files/S000512/TADC8/D000015/SID00012527_F00000238_D000015_TADC8_S008192_R000512_FS0001_N_00000064.npy")
            return

        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ Error: File not found: {file_path}")
            return

        # Determine save location
        if save_path:
            save_location = Path(save_path).expanduser()
        elif output_folder:
            output_dir = Path(output_folder).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            save_location = output_dir / f"{file_path.stem}_plot.png"
        else:
            save_location = None

        # Load and validate data
        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return

        if data.ndim != 2:
            print(f"❌ Error: Expected 2D array (windows × features), got shape {data.shape}")
            return

        filename = file_path.name

        # Parse feature set ID from filename (e.g., FS0001)
        import re
        fs_match = re.search(r'_FS(\d+)', filename)
        feature_set_id = int(fs_match.group(1)) if fs_match else None

        # Get column labels from database
        column_labels = []
        if self.db_conn and feature_set_id and self.current_experiment:
            try:
                cursor = self.db_conn.cursor()

                # Get feature set features
                cursor.execute("""
                    SELECT
                        f.feature_name,
                        fsf.feature_order
                    FROM ml_feature_set_features fsf
                    JOIN ml_features_lut f ON fsf.feature_id = f.feature_id
                    WHERE fsf.feature_set_id = %s
                    ORDER BY fsf.feature_order
                """, (feature_set_id,))
                features = cursor.fetchall()

                # Get CONFIGURED amplitude methods for this experiment
                # Feature files should only contain configured methods
                cursor.execute("""
                    SELECT eam.method_id, am.method_name
                    FROM ml_experiments_amplitude_methods eam
                    JOIN ml_amplitude_normalization_lut am ON eam.method_id = am.method_id
                    WHERE eam.experiment_id = %s
                    ORDER BY eam.method_id
                """, (self.current_experiment,))
                all_amplitude_methods = cursor.fetchall()

                cursor.close()

                # Build labels based on extraction column order: [feat0_amp0, feat0_amp1, feat1_amp0, ...]
                for feat_name, feat_order in features:
                    for method_id, method_name in all_amplitude_methods:
                        column_labels.append(f"{feat_name} ({method_name})")
                        if len(column_labels) >= data.shape[1]:
                            break
                    if len(column_labels) >= data.shape[1]:
                        break

            except Exception as e:
                print(f"⚠️  Warning: Could not fetch labels from database: {e}")
                column_labels = [f"Feature {i}" for i in range(data.shape[1])]
        else:
            column_labels = [f"Feature {i}" for i in range(data.shape[1])]

        # Pad labels if needed
        while len(column_labels) < data.shape[1]:
            column_labels.append(f"Feature {len(column_labels)}")

        print(f"\n📊 Feature File: {filename}")
        print(f"   Shape: {data.shape}")
        print(f"   Windows: {data.shape[0]:,}")
        print(f"   Columns: {data.shape[1]}")
        print(f"\n   Column Labels:")
        for i, label in enumerate(column_labels[:data.shape[1]]):
            print(f"     Column {i}: {label}")
        print()

        # Create plot
        fig, axes = plt.subplots(data.shape[1], 1, figsize=(14, 2.5 * data.shape[1]), sharex=True)
        if data.shape[1] == 1:
            axes = [axes]

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

        for i in range(data.shape[1]):
            color = colors[i % len(colors)]
            axes[i].plot(data[:, i], linewidth=1, color=color)
            axes[i].set_ylabel(column_labels[i], fontsize=10, fontweight='bold')
            axes[i].grid(True, alpha=0.3, linestyle='--')

            # Add statistics
            mean = np.mean(data[:, i])
            std = np.std(data[:, i])
            min_val = np.min(data[:, i])
            max_val = np.max(data[:, i])
            axes[i].text(0.02, 0.95,
                        f'μ={mean:.2f}, σ={std:.2f}, min={min_val:.2f}, max={max_val:.2f}',
                        transform=axes[i].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                        fontsize=8)

        axes[-1].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
        plt.suptitle(f'Feature File: {filename}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_location:
            plt.savefig(save_location, dpi=150, bbox_inches='tight')
            print(f"💾 Saved plot to: {save_location}")
            plt.close()
        else:
            plt.show()

        print("\n✅ Feature plotting complete!")

    def cmd_segment_plot(self, args):
        """Plot segment files with statistical analysis

        Usage: segment-plot [options]

        Options:
            --amplitude-method <method>   Select amplitude processing: raw, minmax, zscore (default: raw)
            --original-segment <id>       Original segment ID to plot
            --result-segment-size <size>  Result segment size
            --types <types>               Data types (RAW, ADC6, ADC8, etc.)
            --decimations <list>          Decimation factors (0, 7, 15, etc.)
            --output-folder <path>        Output directory for plots (required)

        Examples:
            segment-plot --original-segment 104075 --decimations 0 --output-folder ~/plots/
            segment-plot --result-segment-size 131072 --types RAW --output-folder ~/plots/
            segment-plot --file-labels 200,201 --num-points 500 --peak-detect --output-folder ~/plots/
            segment-plot --original-segment 104075 --amplitude-method minmax --output-folder ~/plots/
            segment-plot --original-segment 104075 --amplitude-method zscore --output-folder ~/plots/
        """
        try:
            from segment_file_plotter import plot_segment_files
        except ImportError:
            import segment_file_plotter
            plot_segment_files = segment_file_plotter.plot_segment_files
        
        # Parse arguments
        parts = args if isinstance(args, list) else args.split()
        
        # Initialize parameters
        params = {
            'experiment_id': self.current_experiment,
            'original_segment': None,
            'result_segment_size': None,
            'segment_labels': None,
            'file_labels': None,
            'decimations': None,
            'types': None,
            'num_points': 1000,
            'peak_detect': False,
            'plot_actual': True,
            'plot_minimums': False,
            'plot_maximums': False,
            'plot_average': False,
            'plot_variance': False,
            'plot_stddev': False,
            'minimums_point': False,
            'minimums_line': False,
            'maximums_point': False,
            'maximums_line': False,
            'average_point': False,
            'average_line': True,
            'variance_point': False,
            'variance_line': True,
            'stddev_point': False,
            'stddev_line': True,
            'no_subplots': False,
            'subplots': 'file',
            'max_subplot': (3, 3),
            'dpi': 300,
            'format': 'png',
            'title': None,
            'plot_style': 'cleaning',
            'output_folder': None,
            'amplitude_method': 'raw'
        }
        
        # Parse command line arguments
        i = 0
        while i < len(parts):
            if parts[i] == '--original-segment' and i + 1 < len(parts):
                params['original_segment'] = int(parts[i + 1])
                i += 2
            elif parts[i] == '--result-segment-size' and i + 1 < len(parts):
                params['result_segment_size'] = int(parts[i + 1])
                i += 2
            elif parts[i] == '--segment-labels' and i + 1 < len(parts):
                if parts[i + 1].lower() == 'all':
                    params['segment_labels'] = None
                else:
                    params['segment_labels'] = [int(x) for x in parts[i + 1].split(',')]
                i += 2
            elif parts[i] == '--file-labels' and i + 1 < len(parts):
                if parts[i + 1].lower() == 'all':
                    params['file_labels'] = None
                else:
                    params['file_labels'] = [int(x) for x in parts[i + 1].split(',')]
                i += 2
            elif parts[i] == '--decimations' and i + 1 < len(parts):
                if parts[i + 1].lower() == 'all':
                    params['decimations'] = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511]
                else:
                    params['decimations'] = [int(x) for x in parts[i + 1].split(',')]
                i += 2
            elif parts[i] == '--types' and i + 1 < len(parts):
                if parts[i + 1].lower() == 'all':
                    params['types'] = ['RAW', 'ADC14', 'ADC12', 'ADC10', 'ADC8', 'ADC6']
                else:
                    params['types'] = parts[i + 1].split(',')
                i += 2
            elif parts[i] == '--num-points' and i + 1 < len(parts):
                params['num_points'] = int(parts[i + 1])
                i += 2
            elif parts[i] == '--peak-detect':
                params['peak_detect'] = True
                i += 1
            elif parts[i] == '--plot-actual':
                params['plot_actual'] = True
                i += 1
            elif parts[i] == '--plot-minimums':
                params['plot_minimums'] = True
                i += 1
            elif parts[i] == '--plot-minimums-point':
                params['plot_minimums'] = True
                params['minimums_point'] = True
                i += 1
            elif parts[i] == '--plot-minimums-line':
                params['plot_minimums'] = True
                params['minimums_line'] = True
                i += 1
            elif parts[i] == '--plot-maximums':
                params['plot_maximums'] = True
                i += 1
            elif parts[i] == '--plot-maximums-point':
                params['plot_maximums'] = True
                params['maximums_point'] = True
                i += 1
            elif parts[i] == '--plot-maximums-line':
                params['plot_maximums'] = True
                params['maximums_line'] = True
                i += 1
            elif parts[i] == '--plot-average':
                params['plot_average'] = True
                i += 1
            elif parts[i] == '--plot-average-point':
                params['plot_average'] = True
                params['average_point'] = True
                i += 1
            elif parts[i] == '--plot-average-line':
                params['plot_average'] = True
                params['average_line'] = True
                i += 1
            elif parts[i] == '--plot-variance':
                params['plot_variance'] = True
                i += 1
            elif parts[i] == '--plot-variance-point':
                params['plot_variance'] = True
                params['variance_point'] = True
                i += 1
            elif parts[i] == '--plot-variance-line':
                params['plot_variance'] = True
                params['variance_line'] = True
                i += 1
            elif parts[i] == '--plot-stddev':
                params['plot_stddev'] = True
                i += 1
            elif parts[i] == '--plot-stddev-point':
                params['plot_stddev'] = True
                params['stddev_point'] = True
                i += 1
            elif parts[i] == '--plot-stddev-line':
                params['plot_stddev'] = True
                params['stddev_line'] = True
                i += 1
            elif parts[i] == '--no-subplots':
                params['no_subplots'] = True
                i += 1
            elif parts[i] == '--subplots' and i + 1 < len(parts):
                params['subplots'] = parts[i + 1]
                i += 2
            elif parts[i] == '--max-subplot' and i + 1 < len(parts):
                rows, cols = parts[i + 1].split(',')
                params['max_subplot'] = (int(rows), int(cols))
                i += 2
            elif parts[i] == '--dpi' and i + 1 < len(parts):
                params['dpi'] = int(parts[i + 1])
                i += 2
            elif parts[i] == '--format' and i + 1 < len(parts):
                params['format'] = parts[i + 1]
                i += 2
            elif parts[i] == '--title' and i + 1 < len(parts):
                params['title'] = parts[i + 1]
                i += 2
            elif parts[i] == '--plot-style' and i + 1 < len(parts):
                params['plot_style'] = parts[i + 1]
                i += 2
            elif parts[i] == '--amplitude-method' and i + 1 < len(parts):
                params['amplitude_method'] = parts[i + 1]
                i += 2
            elif parts[i] == '--output-folder' and i + 1 < len(parts):
                params['output_folder'] = parts[i + 1]
                i += 2
            else:
                i += 1
        
        # Check required parameters
        if not params['output_folder']:
            print("❌ Error: --output-folder is required")
            print("\nUsage: segment-plot --output-folder <path> [options]")
            print("\nExample:")
            print("  segment-plot --original-segment 104075 --decimations 0 --output-folder ~/plots/")
            return
        
        # Set defaults if nothing specified
        if params['decimations'] is None:
            params['decimations'] = [0]
        if params['types'] is None:
            params['types'] = ['RAW']
        
        print(f"\n📊 Starting Segment Plot Generation")
        print(f"Output folder: {params['output_folder']}")
        print(f"Experiment: {params['experiment_id']}")
        print(f"Decimations: {params['decimations']}")
        print(f"Types: {params['types']}")
        print(f"Amplitude method: {params['amplitude_method']}")
        print(f"Num points: {params['num_points']}")
        print(f"Peak detect: {params['peak_detect']}")
        
        # Call the plotting function
        try:
            plot_segment_files(**params)
            print("\n✅ Plotting complete!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='MLDP Interactive Shell',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./mldp                          # Start normally
  ./mldp --connect                # Auto-connect to database
  ./mldp --connect --experiment 41  # Auto-connect and set experiment 41
  ./mldp --connect --experiment 999 # Auto-connect and use first experiment (999 doesn't exist)
        """
    )
    parser.add_argument(
        '--connect',
        action='store_true',
        help='Auto-connect to database on startup'
    )
    parser.add_argument(
        '--experiment',
        type=int,
        metavar='N',
        help='Set experiment to N on startup (uses first experiment if N does not exist)'
    )

    args = parser.parse_args()

    shell = MLDPShell(
        auto_connect=args.connect,
        auto_experiment=args.experiment
    )
    shell.run()


if __name__ == '__main__':
    main()