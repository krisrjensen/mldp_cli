-- Fix View Dependencies and Complete Cleanup
-- Author: Kristophor Jensen
-- Date: 20250914
-- Description: Handle view dependencies before dropping num_features column

-- ============================================
-- Step 1: Drop dependent views
-- ============================================
DROP VIEW IF EXISTS v_experiment_041_feature_parameters CASCADE;
DROP VIEW IF EXISTS v_feature_sets_detail CASCADE;

-- ============================================
-- Step 2: Drop the num_features column
-- ============================================
ALTER TABLE ml_feature_sets_lut DROP COLUMN IF EXISTS num_features;

-- ============================================
-- Step 3: Recreate views without num_features dependency
-- ============================================

-- Recreate v_experiment_041_feature_parameters without num_features
CREATE OR REPLACE VIEW v_experiment_041_feature_parameters AS
SELECT 
    efs.experiment_id,
    efs.feature_set_id,
    fs.feature_set_name,
    efs.data_channel,
    efs.priority_order,
    efs.is_active,
    -- Calculate num_features dynamically
    (SELECT COUNT(*) 
     FROM ml_feature_set_features fsf 
     WHERE fsf.feature_set_id = efs.feature_set_id) as num_features,
    efn.n_value
FROM ml_experiments_feature_sets efs
JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
LEFT JOIN ml_experiments_feature_n_values efn 
    ON efs.experiment_id = efn.experiment_id 
    AND efs.feature_set_id = efn.feature_set_id
WHERE efs.experiment_id = 41;

-- Recreate v_feature_sets_detail if it exists (with calculated num_features)
CREATE OR REPLACE VIEW v_feature_sets_detail AS
SELECT 
    fs.feature_set_id,
    fs.feature_set_name,
    -- Calculate num_features dynamically instead of using column
    (SELECT COUNT(*) 
     FROM ml_feature_set_features fsf 
     WHERE fsf.feature_set_id = fs.feature_set_id) as num_features,
    fs.created_date,
    fs.modified_date
FROM ml_feature_sets_lut fs;

-- ============================================
-- Verification
-- ============================================
-- Check that column is dropped
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'ml_feature_sets_lut' 
  AND column_name = 'num_features';

-- Check views are recreated
SELECT viewname 
FROM pg_views 
WHERE viewname IN ('v_experiment_041_feature_parameters', 'v_feature_sets_detail');

-- Show feature sets that were affected
SELECT 
    fs.feature_set_id,
    fs.feature_set_name,
    COUNT(fsf.feature_id) as feature_count
FROM ml_feature_sets_lut fs
LEFT JOIN ml_feature_set_features fsf ON fs.feature_set_id = fsf.feature_set_id
WHERE fs.feature_set_id IN (6, 7, 8, 9, 10)
GROUP BY fs.feature_set_id, fs.feature_set_name
ORDER BY fs.feature_set_id;