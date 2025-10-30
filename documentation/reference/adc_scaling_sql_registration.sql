-- ============================================================================
-- ADC Scaling Scaler - SQL Registration Script
-- ============================================================================
-- Filename: adc_scaling_sql_registration.sql
-- Author: Kristophor Jensen
-- Date Created: 2025-10-29
-- Description: Registers adc_scaling amplitude normalization method
--
-- Method Details:
--   method_id: 7
--   method_name: adc_scaling
--   equation: value / (2^bit_depth - 1)
--   applies_to: Integer ADC data only (ADC6, ADC8, ADC10)
--   skips: Raw float data (returns unchanged)
--
-- Prerequisites:
--   - ml_amplitude_normalization_lut table exists
--   - ml_data_types_lut table has bit_depth column populated
-- ============================================================================

-- ============================================================================
-- PART 1: REGISTER ADC_SCALING METHOD
-- ============================================================================

INSERT INTO ml_amplitude_normalization_lut
(method_id, method_name, display_name, description, function_name,
 function_args, parameters_schema, column_count, is_active)
VALUES
(9, 'adc_scaling', 'ADC Bit-Depth Scaling',
 'Scales data by dividing by (2^bit_depth - 1). Only applies to integer ADC data, returns raw float data unchanged.',
 '_apply_adc_scaling', NULL, NULL, 1, true);


-- ============================================================================
-- VERIFICATION QUERY: Check registration
-- ============================================================================

-- Verify adc_scaling method exists
SELECT
    method_id,
    method_name,
    display_name,
    description,
    function_name,
    is_active
FROM ml_amplitude_normalization_lut
WHERE method_id = 9;

-- Expected result:
-- method_id | method_name  | display_name           | function_name        | is_active
-- ----------|--------------|------------------------|----------------------|----------
-- 9         | adc_scaling  | ADC Bit-Depth Scaling  | _apply_adc_scaling   | true


-- ============================================================================
-- VERIFICATION QUERY: Check bit_depth availability
-- ============================================================================

-- List all data types with their bit depths
SELECT
    data_type_id,
    data_type_name,
    bit_depth,
    CASE
        WHEN bit_depth IS NOT NULL THEN CONCAT('Max ADC: ', (POW(2, bit_depth) - 1)::TEXT)
        ELSE 'Raw data (no bit depth)'
    END as max_adc_value
FROM ml_data_types_lut
ORDER BY data_type_id;

-- Expected results (example):
-- data_type_id | data_type_name | bit_depth | max_adc_value
-- -------------|----------------|-----------|---------------
-- 1            | ADC8           | 8         | Max ADC: 255
-- 2            | ADC10          | 10        | Max ADC: 1023
-- 3            | ADC6           | 6         | Max ADC: 63
-- 4            | RAW            | NULL      | Raw data (no bit depth)


-- ============================================================================
-- PART 2: EXAMPLE LINKING TO EXPERIMENT
-- ============================================================================
-- NOTE: These are EXAMPLE commands only. Do NOT link to existing experiments.
-- Use these as templates when creating new experiments.
-- ============================================================================

-- EXAMPLE: Link adc_scaling to experiment
-- DO NOT EXECUTE - Replace <EXPERIMENT_ID> with your experiment ID
/*
INSERT INTO ml_experiments_amplitude_methods (experiment_id, method_id)
VALUES (<EXPERIMENT_ID>, 9);
*/


-- EXAMPLE: Link multiple scalers to same experiment
-- DO NOT EXECUTE
/*
INSERT INTO ml_experiments_amplitude_methods (experiment_id, method_id)
VALUES
(<EXPERIMENT_ID>, 5),  -- raw
(<EXPERIMENT_ID>, 9);  -- adc_scaling
*/


-- ============================================================================
-- VERIFICATION QUERY: Check experiment linking
-- ============================================================================

-- Check which experiments use adc_scaling
SELECT
    eam.experiment_id,
    e.experiment_name,
    am.method_name,
    am.display_name
FROM ml_experiments_amplitude_methods eam
JOIN ml_experiments e ON eam.experiment_id = e.experiment_id
JOIN ml_amplitude_normalization_lut am ON eam.method_id = am.method_id
WHERE am.method_id = 9
ORDER BY eam.experiment_id;


-- ============================================================================
-- TESTING QUERY: Simulate adc_scaling calculation
-- ============================================================================

-- Test calculation for different bit depths
WITH test_data AS (
    SELECT
        data_type_id,
        data_type_name,
        bit_depth,
        (POW(2, bit_depth) - 1) as max_adc_value
    FROM ml_data_types_lut
    WHERE bit_depth IS NOT NULL
)
SELECT
    data_type_name,
    bit_depth,
    max_adc_value,
    -- Test values
    0 as adc_value_0,
    ROUND((0.0 / max_adc_value)::numeric, 4) as normalized_0,

    FLOOR(max_adc_value / 2) as adc_value_mid,
    ROUND((FLOOR(max_adc_value / 2) / max_adc_value)::numeric, 4) as normalized_mid,

    max_adc_value as adc_value_max,
    ROUND((max_adc_value / max_adc_value)::numeric, 4) as normalized_max
FROM test_data
ORDER BY bit_depth;

-- Expected results (example):
-- data_type_name | bit_depth | max_adc_value | adc_value_0 | normalized_0 | adc_value_mid | normalized_mid | adc_value_max | normalized_max
-- ---------------|-----------|---------------|-------------|--------------|---------------|----------------|---------------|---------------
-- ADC6           | 6         | 63            | 0           | 0.0000       | 31            | 0.4921         | 63            | 1.0000
-- ADC8           | 8         | 255           | 0           | 0.0000       | 127           | 0.4980         | 255           | 1.0000
-- ADC10          | 10        | 1023          | 0           | 0.0000       | 511           | 0.4995         | 1023          | 1.0000


-- ============================================================================
-- UTILITY QUERY: Show all amplitude methods
-- ============================================================================

SELECT
    method_id,
    method_name,
    display_name,
    is_active,
    CASE
        WHEN method_id IN (SELECT method_id FROM ml_experiments_amplitude_methods)
        THEN 'In use'
        ELSE 'Available'
    END as status
FROM ml_amplitude_normalization_lut
ORDER BY method_id;


-- ============================================================================
-- CLEANUP QUERY: Remove adc_scaling registration (if needed)
-- ============================================================================
-- WARNING: Only use this if you need to unregister the method
-- This will cascade delete any experiment links
/*
DELETE FROM ml_amplitude_normalization_lut
WHERE method_id = 9;
*/


-- ============================================================================
-- SUMMARY
-- ============================================================================
-- After running this script:
-- 1. adc_scaling method registered as method_id 9
-- 2. Scaler available for linking to experiments
-- 3. No pre-calculation or initialization required
-- 4. Ready to use with AmplitudeProcessor._apply_adc_scaling()
--
-- To use:
-- 1. Link to experiment: INSERT INTO ml_experiments_amplitude_methods (experiment_id, method_id) VALUES (<EXP_ID>, 9);
-- 2. Extract features - adc_scaling applied automatically
-- ============================================================================
