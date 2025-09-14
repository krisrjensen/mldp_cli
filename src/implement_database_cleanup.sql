-- Database Cleanup Implementation
-- Author: Kristophor Jensen
-- Date: 20250914
-- Description: Execute approved database cleanup actions

-- ============================================
-- STEP 1: Update Feature Categories (Make Drivers)
-- ============================================
-- Update voltage and current to electrical category (drivers)
UPDATE ml_features_lut SET feature_category = 'electrical' WHERE feature_id = 16; -- voltage
UPDATE ml_features_lut SET feature_category = 'electrical' WHERE feature_id = 18; -- current

-- Update raw_data to compute category (driver)
UPDATE ml_features_lut SET feature_category = 'compute' WHERE feature_name = 'raw_data';

-- Verify updates
SELECT feature_id, feature_name, feature_category 
FROM ml_features_lut 
WHERE feature_id IN (16, 18) OR feature_name = 'raw_data';

-- ============================================
-- STEP 2: Remove Redundant Features from Feature Sets
-- ============================================
-- Check which feature sets use the redundant features
SELECT fs.feature_set_id, fs.feature_set_name, fl.feature_id, fl.feature_name
FROM ml_feature_set_features fsf
JOIN ml_feature_sets_lut fs ON fsf.feature_set_id = fs.feature_set_id
JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
WHERE fsf.feature_id IN (17, 19, 20, 21)
ORDER BY fs.feature_set_id;

-- Remove redundant features from feature sets
DELETE FROM ml_feature_set_features WHERE feature_id IN (17, 19, 20, 21);

-- ============================================
-- STEP 3: Delete Redundant Features from ml_features_lut
-- ============================================
-- Delete the redundant variance features
DELETE FROM ml_features_lut WHERE feature_id IN (17, 19, 20, 21);
-- Removes: variance_voltage, variance_current, variance_impedance, variance_power

-- ============================================
-- STEP 4: Drop Redundant Table
-- ============================================
-- Drop the redundant experiment_041_parameters table
DROP TABLE IF EXISTS experiment_041_parameters;

-- ============================================
-- STEP 5: Rename View to Follow Convention
-- ============================================
-- Rename view to use v_ prefix
ALTER VIEW IF EXISTS experiment_041_feature_parameters 
RENAME TO v_experiment_041_feature_parameters;

-- ============================================
-- STEP 6: Drop Redundant Column
-- ============================================
-- Remove num_features column from ml_feature_sets_lut
ALTER TABLE ml_feature_sets_lut DROP COLUMN IF EXISTS num_features;

-- ============================================
-- VERIFICATION QUERIES
-- ============================================
-- Verify feature categories are updated
SELECT feature_id, feature_name, feature_category 
FROM ml_features_lut 
WHERE feature_id IN (16, 18) OR feature_name = 'raw_data';

-- Verify redundant features are deleted
SELECT feature_id, feature_name FROM ml_features_lut 
WHERE feature_id IN (17, 19, 20, 21);

-- Verify table is dropped
SELECT tablename FROM pg_tables 
WHERE tablename = 'experiment_041_parameters';

-- Verify view is renamed
SELECT viewname FROM pg_views 
WHERE viewname IN ('experiment_041_feature_parameters', 'v_experiment_041_feature_parameters');

-- Verify column is dropped
SELECT column_name FROM information_schema.columns 
WHERE table_name = 'ml_feature_sets_lut' AND column_name = 'num_features';

-- Show affected feature sets that need updating
SELECT feature_set_id, feature_set_name 
FROM ml_feature_sets_lut 
WHERE feature_set_id IN (6, 7, 8, 9, 10);