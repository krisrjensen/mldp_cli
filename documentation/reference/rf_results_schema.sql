-- Random Forest Results Table Schema
-- File: rf_results_schema.sql
-- Author: Kristophor Jensen
-- Date Created: 20251028_130000
-- Description: Database schema for Random Forest classifier results
--
-- Usage: Replace {exp:03d} and {cls:03d} with actual experiment and classifier IDs
-- Example: experiment_041_classifier_001_rf_results

CREATE TABLE IF NOT EXISTS experiment_{exp:03d}_classifier_{cls:03d}_rf_results (
    result_id BIGSERIAL PRIMARY KEY,
    global_classifier_id INTEGER NOT NULL,
    classifier_id INTEGER NOT NULL,
    decimation_factor INTEGER NOT NULL,
    data_type_id INTEGER NOT NULL,
    amplitude_processing_method_id INTEGER NOT NULL,
    experiment_feature_set_id BIGINT NOT NULL,

    -- Random Forest hyperparameters
    rf_n_estimators INTEGER NOT NULL,
    rf_max_depth INTEGER,  -- NULL = unlimited depth
    rf_min_samples_split INTEGER NOT NULL,
    rf_max_features VARCHAR(20),  -- 'sqrt', 'log2', or NULL (all features)

    -- Training configuration
    class_weight VARCHAR(20),
    random_state INTEGER,
    train_ratio DOUBLE PRECISION,
    test_ratio DOUBLE PRECISION,
    verification_ratio DOUBLE PRECISION,
    cv_folds INTEGER,

    -- 13-class multiclass metrics - Training split
    accuracy_train DOUBLE PRECISION,
    precision_macro_train DOUBLE PRECISION,
    recall_macro_train DOUBLE PRECISION,
    f1_macro_train DOUBLE PRECISION,
    precision_weighted_train DOUBLE PRECISION,
    recall_weighted_train DOUBLE PRECISION,
    f1_weighted_train DOUBLE PRECISION,
    cv_mean_accuracy DOUBLE PRECISION,
    cv_std_accuracy DOUBLE PRECISION,

    -- 13-class multiclass metrics - Test split
    accuracy_test DOUBLE PRECISION,
    precision_macro_test DOUBLE PRECISION,
    recall_macro_test DOUBLE PRECISION,
    f1_macro_test DOUBLE PRECISION,
    precision_weighted_test DOUBLE PRECISION,
    recall_weighted_test DOUBLE PRECISION,
    f1_weighted_test DOUBLE PRECISION,

    -- 13-class multiclass metrics - Verification split
    accuracy_verify DOUBLE PRECISION,
    precision_macro_verify DOUBLE PRECISION,
    recall_macro_verify DOUBLE PRECISION,
    f1_macro_verify DOUBLE PRECISION,
    precision_weighted_verify DOUBLE PRECISION,
    recall_weighted_verify DOUBLE PRECISION,
    f1_weighted_verify DOUBLE PRECISION,

    -- Binary arc detection metrics - Training split
    arc_accuracy_train DOUBLE PRECISION,
    arc_precision_train DOUBLE PRECISION,
    arc_recall_train DOUBLE PRECISION,
    arc_f1_train DOUBLE PRECISION,
    arc_specificity_train DOUBLE PRECISION,
    arc_roc_auc_train DOUBLE PRECISION,
    arc_pr_auc_train DOUBLE PRECISION,

    -- Binary arc detection metrics - Test split
    arc_accuracy_test DOUBLE PRECISION,
    arc_precision_test DOUBLE PRECISION,
    arc_recall_test DOUBLE PRECISION,
    arc_f1_test DOUBLE PRECISION,
    arc_specificity_test DOUBLE PRECISION,
    arc_roc_auc_test DOUBLE PRECISION,
    arc_pr_auc_test DOUBLE PRECISION,

    -- Binary arc detection metrics - Verification split
    arc_accuracy_verify DOUBLE PRECISION,
    arc_precision_verify DOUBLE PRECISION,
    arc_recall_verify DOUBLE PRECISION,
    arc_f1_verify DOUBLE PRECISION,
    arc_specificity_verify DOUBLE PRECISION,
    arc_roc_auc_verify DOUBLE PRECISION,
    arc_pr_auc_verify DOUBLE PRECISION,

    -- File paths and performance
    model_path TEXT NOT NULL,
    training_time_seconds DOUBLE PRECISION,
    prediction_time_seconds DOUBLE PRECISION,

    -- Ensure unique hyperparameter combinations
    UNIQUE (decimation_factor, data_type_id, amplitude_processing_method_id,
            experiment_feature_set_id, rf_n_estimators, rf_max_depth,
            rf_min_samples_split, rf_max_features)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_rf_results_global_classifier
    ON experiment_{exp:03d}_classifier_{cls:03d}_rf_results(global_classifier_id);

CREATE INDEX IF NOT EXISTS idx_rf_results_config
    ON experiment_{exp:03d}_classifier_{cls:03d}_rf_results(
        decimation_factor, data_type_id, amplitude_processing_method_id, experiment_feature_set_id
    );

CREATE INDEX IF NOT EXISTS idx_rf_results_test_performance
    ON experiment_{exp:03d}_classifier_{cls:03d}_rf_results(accuracy_test DESC, arc_f1_test DESC);

-- Per-class results table (for detailed class-by-class metrics)
CREATE TABLE IF NOT EXISTS experiment_{exp:03d}_classifier_{cls:03d}_rf_per_class_results (
    per_class_result_id BIGSERIAL PRIMARY KEY,
    result_id BIGINT NOT NULL REFERENCES experiment_{exp:03d}_classifier_{cls:03d}_rf_results(result_id) ON DELETE CASCADE,

    split_type VARCHAR(20) NOT NULL,  -- 'train', 'test', or 'verify'
    class_label_id INTEGER NOT NULL,

    precision DOUBLE PRECISION,
    recall DOUBLE PRECISION,
    f1_score DOUBLE PRECISION,
    support INTEGER,

    UNIQUE (result_id, split_type, class_label_id)
);

CREATE INDEX IF NOT EXISTS idx_rf_per_class_result_id
    ON experiment_{exp:03d}_classifier_{cls:03d}_rf_per_class_results(result_id);

-- Example usage for experiment 41, classifier 1:
-- 1. Replace {exp:03d} with 041 and {cls:03d} with 001
-- 2. Execute the CREATE TABLE statements
-- 3. The trainer will automatically insert results into these tables
