-- ============================================================================
-- Feature Sets Creation and Experiment Linking Script
-- ============================================================================
-- Filename: create_feature_sets.sql
-- Author: Kristophor Jensen
-- Date Created: 2025-10-29
-- Description: Creates 6 feature sets and links them to experiment 41
--
-- Prerequisites: Run register_new_features.sql first
--
-- Feature Sets:
-- 1. derivative_volatility (6 features)
-- 2. stft_basic (8 STFT scalar features)
-- 3. stft_volatility_composite (4 STFT+volatility features)
-- 4. pink_noise_stft_tmr (12 TMR features)
-- 5. pink_noise_stft_bandpower (8 bandpower features)
-- 6. new_features_comprehensive (64 all features)
-- ============================================================================

-- ============================================================================
-- PART 1: CREATE FEATURE SETS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Feature Set 1: Derivative Volatility (6 features)
-- Feature Set ID: 15
-- ----------------------------------------------------------------------------

INSERT INTO ml_feature_sets_lut
(feature_set_id, feature_set_name, description, category, is_active)
VALUES
(15, 'derivative_volatility', 'Derivative volatility features for arc detection (n=1,2,3 derivatives)', 'temporal', true);

-- Get the feature_set_id we just created
-- For the next statements, replace <FEATURE_SET_ID_1> with actual ID

-- Add features to the set (feature_set_id = 15)
INSERT INTO ml_feature_set_features
(feature_set_id, feature_id, feature_order, data_channel)
SELECT
    15,
    feature_id,
    ROW_NUMBER() OVER (ORDER BY feature_name),
    NULL  -- Will be specified when linking to experiment
FROM ml_features_lut
WHERE feature_name IN (
    'volatility_dxdt_n1_mean',
    'volatility_dxdt_n1_max',
    'volatility_dxdt_n2_mean',
    'volatility_dxdt_n1',
    'volatility_dxdt_n2',
    'volatility_dxdt_n3'
);


-- ----------------------------------------------------------------------------
-- Feature Set 2: STFT Basic (8 scalar STFT features)
-- Feature Set ID: 16
-- ----------------------------------------------------------------------------

INSERT INTO ml_feature_sets_lut
(feature_set_id, feature_set_name, description, category, is_active)
VALUES
(16, 'stft_basic', 'Basic STFT features with scalar aggregates for SVM training', 'spectral', true);

INSERT INTO ml_feature_set_features
(feature_set_id, feature_id, feature_order, data_channel)
SELECT
    16,
    feature_id,
    ROW_NUMBER() OVER (ORDER BY feature_name),
    NULL
FROM ml_features_lut
WHERE feature_name IN (
    'stft_mean_power_n8',
    'stft_max_power_n8',
    'stft_total_energy_n8',
    'stft_low_freq_power_n8',
    'stft_mid_freq_power_n8',
    'stft_high_freq_power_n8',
    'stft_band_ratio_n8',
    'stft_mean_frequency_n8'
);


-- ----------------------------------------------------------------------------
-- Feature Set 3: STFT + Volatility Composite (4 features)
-- Feature Set ID: 17
-- ----------------------------------------------------------------------------

INSERT INTO ml_feature_sets_lut
(feature_set_id, feature_set_name, description, category, is_active)
VALUES
(17, 'stft_volatility_composite', 'STFT applied to volatility signals - time-frequency volatility analysis', 'composite', true);

INSERT INTO ml_feature_set_features
(feature_set_id, feature_id, feature_order, data_channel)
SELECT
    17,
    feature_id,
    ROW_NUMBER() OVER (ORDER BY feature_name),
    NULL
FROM ml_features_lut
WHERE feature_name IN (
    'stft_volatility_n1_8slices',
    'stft_volatility_n1_8slices_o20',
    'stft_volatility_n1_low_freq',
    'stft_volatility_n2_8slices'
);


-- ----------------------------------------------------------------------------
-- Feature Set 4: Pink Noise TMR Method (12 features)
-- Feature Set ID: 18
-- ----------------------------------------------------------------------------

INSERT INTO ml_feature_sets_lut
(feature_set_id, feature_set_name, description, category, is_active)
VALUES
(18, 'pink_noise_stft_tmr', 'TMR paper pink noise analysis - A/f^gamma + c curve fitting per band', 'spectral', true);

INSERT INTO ml_feature_set_features
(feature_set_id, feature_id, feature_order, data_channel)
SELECT
    18,
    feature_id,
    ROW_NUMBER() OVER (ORDER BY feature_name),
    NULL
FROM ml_features_lut
WHERE feature_name LIKE 'pink_noise_tmr_%';


-- ----------------------------------------------------------------------------
-- Feature Set 5: Pink Noise Band Power Method (8 features)
-- Feature Set ID: 19
-- ----------------------------------------------------------------------------

INSERT INTO ml_feature_sets_lut
(feature_set_id, feature_set_name, description, category, is_active)
VALUES
(19, 'pink_noise_stft_bandpower', 'Pink noise band power method - power per frequency band', 'spectral', true);

INSERT INTO ml_feature_set_features
(feature_set_id, feature_id, feature_order, data_channel)
SELECT
    19,
    feature_id,
    ROW_NUMBER() OVER (ORDER BY feature_name),
    NULL
FROM ml_features_lut
WHERE feature_name LIKE 'pink_noise_bandpower_%';


-- ----------------------------------------------------------------------------
-- Feature Set 6: Comprehensive (ALL 64 features)
-- Feature Set ID: 20
-- ----------------------------------------------------------------------------

INSERT INTO ml_feature_sets_lut
(feature_set_id, feature_set_name, description, category, is_active)
VALUES
(20, 'new_features_comprehensive', 'All 64 new feature functions: derivative, MA, STFT, pink noise, composite', 'composite', true);

INSERT INTO ml_feature_set_features
(feature_set_id, feature_id, feature_order, data_channel)
SELECT
    20,
    feature_id,
    ROW_NUMBER() OVER (ORDER BY feature_category, feature_name),
    NULL
FROM ml_features_lut
WHERE feature_name LIKE 'volatility_%'
   OR feature_name LIKE 'moving_average_%'
   OR feature_name LIKE 'stft_%'
   OR feature_name LIKE 'pink_noise_%';


-- ============================================================================
-- VERIFICATION QUERY: Check feature sets created
-- ============================================================================

-- Count feature sets
SELECT
    COUNT(*) as new_feature_sets
FROM ml_feature_sets_lut
WHERE feature_set_name IN (
    'derivative_volatility',
    'stft_basic',
    'stft_volatility_composite',
    'pink_noise_stft_tmr',
    'pink_noise_stft_bandpower',
    'new_features_comprehensive'
);
-- Expected: 6

-- List feature sets with feature counts
SELECT
    fsl.feature_set_id,
    fsl.feature_set_name,
    fsl.description,
    COUNT(fsf.feature_id) as feature_count
FROM ml_feature_sets_lut fsl
LEFT JOIN ml_feature_set_features fsf ON fsl.feature_set_id = fsf.feature_set_id
WHERE fsl.feature_set_name IN (
    'derivative_volatility',
    'stft_basic',
    'stft_volatility_composite',
    'pink_noise_stft_tmr',
    'pink_noise_stft_bandpower',
    'new_features_comprehensive'
)
GROUP BY fsl.feature_set_id, fsl.feature_set_name, fsl.description
ORDER BY fsl.feature_set_name;

-- Expected feature counts:
-- derivative_volatility: 6
-- stft_basic: 8
-- stft_volatility_composite: 4
-- pink_noise_stft_tmr: 12
-- pink_noise_stft_bandpower: 8
-- new_features_comprehensive: 64


-- ============================================================================
-- PART 2: LINK FEATURE SETS TO EXPERIMENT 41
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Link to load_voltage channel (priority_order: 20-25)
-- Experiment Feature Set IDs: 6-11
-- ----------------------------------------------------------------------------

INSERT INTO ml_experiments_feature_sets
(experiment_feature_set_id, experiment_id, feature_set_id, data_channel, n_value, priority_order)
VALUES
(6, 41, 15, 'load_voltage', NULL, 20),  -- derivative_volatility
(7, 41, 16, 'load_voltage', NULL, 21),  -- stft_basic
(8, 41, 17, 'load_voltage', NULL, 22),  -- stft_volatility_composite
(9, 41, 18, 'load_voltage', NULL, 23),  -- pink_noise_stft_tmr
(10, 41, 19, 'load_voltage', NULL, 24), -- pink_noise_stft_bandpower
(11, 41, 20, 'load_voltage', NULL, 25); -- new_features_comprehensive


-- ----------------------------------------------------------------------------
-- Link to source_current channel (priority_order: 30-35)
-- Experiment Feature Set IDs: 12-17
-- ----------------------------------------------------------------------------

INSERT INTO ml_experiments_feature_sets
(experiment_feature_set_id, experiment_id, feature_set_id, data_channel, n_value, priority_order)
VALUES
(12, 41, 15, 'source_current', NULL, 30), -- derivative_volatility
(13, 41, 16, 'source_current', NULL, 31), -- stft_basic
(14, 41, 17, 'source_current', NULL, 32), -- stft_volatility_composite
(15, 41, 18, 'source_current', NULL, 33), -- pink_noise_stft_tmr
(16, 41, 19, 'source_current', NULL, 34), -- pink_noise_stft_bandpower
(17, 41, 20, 'source_current', NULL, 35); -- new_features_comprehensive


-- ============================================================================
-- VERIFICATION QUERY: Check experiment linking
-- ============================================================================

-- Count links for experiment 41
SELECT
    COUNT(*) as feature_set_links
FROM ml_experiments_feature_sets
WHERE experiment_id = 41
  AND feature_set_id IN (
      SELECT feature_set_id
      FROM ml_feature_sets_lut
      WHERE feature_set_name IN (
          'derivative_volatility',
          'stft_basic',
          'stft_volatility_composite',
          'pink_noise_stft_tmr',
          'pink_noise_stft_bandpower',
          'new_features_comprehensive'
      )
  );
-- Expected: 12 (6 feature sets × 2 channels)

-- List all links with details
SELECT
    efs.experiment_id,
    fsl.feature_set_name,
    efs.data_channel,
    efs.priority_order,
    COUNT(fsf.feature_id) as feature_count
FROM ml_experiments_feature_sets efs
JOIN ml_feature_sets_lut fsl ON efs.feature_set_id = fsl.feature_set_id
LEFT JOIN ml_feature_set_features fsf ON fsl.feature_set_id = fsf.feature_set_id
WHERE efs.experiment_id = 41
  AND fsl.feature_set_name IN (
      'derivative_volatility',
      'stft_basic',
      'stft_volatility_composite',
      'pink_noise_stft_tmr',
      'pink_noise_stft_bandpower',
      'new_features_comprehensive'
  )
GROUP BY efs.experiment_id, fsl.feature_set_name, efs.data_channel, efs.priority_order
ORDER BY efs.data_channel, efs.priority_order;


-- ============================================================================
-- SUMMARY QUERY: Overall registration status
-- ============================================================================

SELECT
    'Total new features registered' as metric,
    COUNT(*) as count
FROM ml_features_lut
WHERE feature_name LIKE 'volatility_%'
   OR feature_name LIKE 'moving_average_%'
   OR feature_name LIKE 'stft_%'
   OR feature_name LIKE 'pink_noise_%'

UNION ALL

SELECT
    'Total new feature sets created' as metric,
    COUNT(*) as count
FROM ml_feature_sets_lut
WHERE feature_set_name IN (
    'derivative_volatility',
    'stft_basic',
    'stft_volatility_composite',
    'pink_noise_stft_tmr',
    'pink_noise_stft_bandpower',
    'new_features_comprehensive'
)

UNION ALL

SELECT
    'Experiment 41 feature set links' as metric,
    COUNT(*) as count
FROM ml_experiments_feature_sets
WHERE experiment_id = 41
  AND feature_set_id IN (
      SELECT feature_set_id
      FROM ml_feature_sets_lut
      WHERE feature_set_name IN (
          'derivative_volatility',
          'stft_basic',
          'stft_volatility_composite',
          'pink_noise_stft_tmr',
          'pink_noise_stft_bandpower',
          'new_features_comprehensive'
      )
  );

-- Expected results:
-- Total new features registered: 64
-- Total new feature sets created: 6
-- Experiment 41 feature set links: 12
