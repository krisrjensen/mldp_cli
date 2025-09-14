-- Database Cleanup Script for Feature Tables
-- Author: Kristophor Jensen
-- Date: 2025-09-14
-- Purpose: Clean up redundant tables and fix channel storage

-- ============================================================
-- PHASE 1: BACKUP (Run these manually first!)
-- ============================================================
-- pg_dump -h localhost -U kjensen -d arc_detection -t experiment_041_feature_parameters > backup_041_feat_params.sql
-- pg_dump -h localhost -U kjensen -d arc_detection -t experiment_041_parameters > backup_041_params.sql

-- ============================================================
-- PHASE 2: VERIFY DATA INTEGRITY
-- ============================================================

-- Check if experiment_041_feature_parameters has any data not in junction tables
SELECT 'Data in experiment_041_feature_parameters not in ml_experiments_feature_sets:' as check_name,
       COUNT(*) as missing_count
FROM experiment_041_feature_parameters efp
WHERE NOT EXISTS (
    SELECT 1 FROM ml_experiments_feature_sets efs
    WHERE efs.experiment_id = 41 
    AND efs.feature_set_id = efp.feature_set_id
);

-- Check N values
SELECT 'N values in experiment_041_feature_parameters not in ml_experiments_feature_n_values:' as check_name,
       COUNT(*) as missing_count
FROM experiment_041_feature_parameters efp
WHERE efp.n_value IS NOT NULL
AND NOT EXISTS (
    SELECT 1 FROM ml_experiments_feature_n_values efn
    WHERE efn.experiment_id = 41 
    AND efn.feature_set_id = efp.feature_set_id
    AND efn.n_value = efp.n_value
);

-- ============================================================
-- PHASE 3: CREATE NEW STRUCTURE (Optional - for feature-level channels)
-- ============================================================

-- Option A: Feature-level channel overrides
-- Uncomment if you want per-feature channel control
/*
CREATE TABLE IF NOT EXISTS ml_experiment_feature_channels (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES ml_experiments(experiment_id),
    feature_set_id BIGINT NOT NULL REFERENCES ml_feature_sets_lut(feature_set_id),
    feature_id BIGINT NOT NULL REFERENCES ml_features_lut(feature_id),
    data_channel VARCHAR(50) NOT NULL CHECK (
        data_channel IN ('source_current', 'load_voltage', 'both')
    ),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_id, feature_set_id, feature_id)
);

CREATE INDEX idx_exp_feat_channels 
ON ml_experiment_feature_channels(experiment_id, feature_set_id);
*/

-- ============================================================
-- PHASE 4: DROP REDUNDANT TABLES
-- ============================================================

-- List tables to be dropped
SELECT 'Tables to be dropped:' as action;
SELECT table_name, 
       pg_size_pretty(pg_total_relation_size('"'||table_name||'"')) as size,
       (SELECT COUNT(*) FROM information_schema.columns 
        WHERE table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_name IN ('experiment_041_feature_parameters', 'experiment_041_parameters')
AND table_schema = 'public';

-- Uncomment to actually drop the tables
-- WARNING: Make sure you have backups first!
/*
DROP TABLE IF EXISTS experiment_041_feature_parameters CASCADE;
DROP TABLE IF EXISTS experiment_041_parameters CASCADE;
*/

-- ============================================================
-- PHASE 5: DOCUMENT CHANNEL LOGIC
-- ============================================================

-- Create a view to show effective channel for each feature
CREATE OR REPLACE VIEW v_experiment_feature_channels AS
SELECT 
    efs.experiment_id,
    efs.feature_set_id,
    fsl.feature_set_name,
    fl.feature_id,
    fl.feature_name,
    CASE 
        -- Fixed channel features
        WHEN fl.feature_name = 'voltage' THEN 'load_voltage'
        WHEN fl.feature_name = 'current' THEN 'source_current'
        WHEN fl.feature_name IN ('impedance', 'power') THEN 'both'
        -- Channel-agnostic features use the set's channel
        WHEN fl.feature_name IN ('raw_data', 'mean', 'variance', 'stddev', 'min', 'max') 
            THEN COALESCE(efs.data_channel, 'load_voltage')
        -- Default
        ELSE COALESCE(efs.data_channel, 'load_voltage')
    END as effective_channel,
    efs.data_channel as set_default_channel
FROM ml_experiments_feature_sets efs
JOIN ml_feature_sets_lut fsl ON efs.feature_set_id = fsl.feature_set_id
JOIN ml_feature_set_features fsf ON fsl.feature_set_id = fsf.feature_set_id
JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
WHERE efs.experiment_id = 41  -- Change for other experiments
ORDER BY efs.feature_set_id, fsf.feature_order;

-- ============================================================
-- PHASE 6: VALIDATION QUERIES
-- ============================================================

-- Show current feature sets with channels for experiment 41
SELECT 'Current feature sets and channels for experiment 41:' as report;
SELECT 
    efs.feature_set_id,
    fsl.feature_set_name,
    efs.data_channel,
    COUNT(DISTINCT fsf.feature_id) as num_features,
    STRING_AGG(fl.feature_name, ', ' ORDER BY fsf.feature_order) as features
FROM ml_experiments_feature_sets efs
JOIN ml_feature_sets_lut fsl ON efs.feature_set_id = fsl.feature_set_id
JOIN ml_feature_set_features fsf ON fsl.feature_set_id = fsf.feature_set_id
JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
WHERE efs.experiment_id = 41
GROUP BY efs.feature_set_id, fsl.feature_set_name, efs.data_channel
ORDER BY efs.feature_set_id;

-- Show feature channel resolution
SELECT 'Feature channel resolution for experiment 41:' as report;
SELECT * FROM v_experiment_feature_channels 
ORDER BY feature_set_id, feature_id;

-- ============================================================
-- ROLLBACK SCRIPT (If needed)
-- ============================================================
-- To restore from backup:
-- psql -h localhost -U kjensen -d arc_detection < backup_041_feat_params.sql
-- psql -h localhost -U kjensen -d arc_detection < backup_041_params.sql