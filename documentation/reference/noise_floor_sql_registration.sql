-- ============================================================================
-- Noise Floor Scaler - SQL Registration Script
-- ============================================================================
-- Filename: noise_floor_sql_registration.sql
-- Author: Kristophor Jensen
-- Date Created: 2025-10-29
-- Description: Creates experiment_noise_floor table and registers noise_floor
--              amplitude normalization method
--
-- Method Details:
--   method_id: 8
--   method_name: noise_floor
--   equation: (X_i - mean(X)) / noise_floor
--   applies_to: Integer ADC data only (ADC6, ADC8, ADC10)
--   skips: Raw float data (returns unchanged)
--   requires: Pre-calculated noise floor values in experiment_noise_floor table
--
-- Prerequisites:
--   - ml_amplitude_normalization_lut table exists
--   - ml_data_types_lut table exists
-- ============================================================================

-- ============================================================================
-- PART 1: CREATE EXPERIMENT_NOISE_FLOOR TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS experiment_noise_floor (
    data_type_id INTEGER PRIMARY KEY,
    noise_floor_voltage DOUBLE PRECISION NOT NULL,
    noise_floor_current DOUBLE PRECISION NOT NULL,
    calculation_method VARCHAR(50) DEFAULT 'std_dev',
    num_segments_used INTEGER,
    last_calculated TIMESTAMP DEFAULT NOW(),
    notes TEXT,
    FOREIGN KEY (data_type_id)
        REFERENCES ml_data_types_lut(data_type_id)
        ON DELETE CASCADE
);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_experiment_noise_floor_data_type
ON experiment_noise_floor(data_type_id);

-- Add column comments for documentation
COMMENT ON TABLE experiment_noise_floor IS 'Stores pre-calculated noise floor values per data type for amplitude normalization';
COMMENT ON COLUMN experiment_noise_floor.data_type_id IS 'Links to ml_data_types_lut (ADC6, ADC8, ADC10, etc.)';
COMMENT ON COLUMN experiment_noise_floor.noise_floor_voltage IS 'Voltage channel noise floor in ADC counts (RMS)';
COMMENT ON COLUMN experiment_noise_floor.noise_floor_current IS 'Current channel noise floor in ADC counts (RMS)';
COMMENT ON COLUMN experiment_noise_floor.calculation_method IS 'Method used for calculation (default: std_dev)';
COMMENT ON COLUMN experiment_noise_floor.num_segments_used IS 'Number of approved steady-state segments (8192 samples) used in calculation';
COMMENT ON COLUMN experiment_noise_floor.last_calculated IS 'Timestamp of when noise floor was calculated';
COMMENT ON COLUMN experiment_noise_floor.notes IS 'Optional notes about calculation or data quality';


-- ============================================================================
-- PART 2: REGISTER NOISE_FLOOR METHOD
-- ============================================================================

INSERT INTO ml_amplitude_normalization_lut
(method_id, method_name, display_name, description, function_name,
 function_args, parameters_schema, column_count, is_active)
VALUES
(10, 'noise_floor', 'Noise Floor Normalization',
 'Scales data using (X - mean(X)) / noise_floor. Noise floor calculated from approved steady-state segments using spectral methods. Only applies to integer ADC data, returns raw float data unchanged.',
 '_apply_noise_floor', NULL, NULL, 1, true);


-- ============================================================================
-- VERIFICATION QUERY: Check table creation
-- ============================================================================

-- Check if experiment_noise_floor table exists
SELECT
    table_name,
    table_type
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name = 'experiment_noise_floor';

-- Expected result:
-- table_name              | table_type
-- ------------------------|------------
-- experiment_noise_floor  | BASE TABLE


-- Show table structure
\d experiment_noise_floor


-- ============================================================================
-- VERIFICATION QUERY: Check method registration
-- ============================================================================

-- Verify noise_floor method exists
SELECT
    method_id,
    method_name,
    display_name,
    description,
    function_name,
    is_active
FROM ml_amplitude_normalization_lut
WHERE method_id = 10;

-- Expected result:
-- method_id | method_name  | display_name              | function_name        | is_active
-- ----------|--------------|---------------------------|----------------------|----------
-- 10        | noise_floor  | Noise Floor Normalization | _apply_noise_floor   | true


-- ============================================================================
-- PART 3: EXAMPLE DATA (FOR TESTING ONLY)
-- ============================================================================
-- NOTE: These are EXAMPLE entries only. Do NOT execute.
-- Use 'noise-floor-calculate' CLI command to populate with real data.
-- ============================================================================

-- EXAMPLE: Insert test noise floor values
-- DO NOT EXECUTE - Use CLI commands instead
/*
INSERT INTO experiment_noise_floor
(data_type_id, noise_floor, calculation_method, num_segments_used, notes)
VALUES
(1, 0.000156, 'spectral_psd', 245, 'ADC8 test calculation'),
(2, 0.000089, 'spectral_psd', 312, 'ADC10 test calculation'),
(3, 0.000234, 'spectral_psd', 189, 'ADC6 test calculation');
*/


-- ============================================================================
-- VERIFICATION QUERY: Check noise floor entries
-- ============================================================================

-- List all calculated noise floor values
SELECT
    nf.data_type_id,
    dt.data_type_name,
    nf.noise_floor,
    nf.calculation_method,
    nf.num_segments_used,
    nf.last_calculated,
    nf.notes
FROM experiment_noise_floor nf
JOIN ml_data_types_lut dt ON nf.data_type_id = dt.data_type_id
ORDER BY dt.data_type_name;

-- Expected result (after calculation via CLI):
-- data_type_id | data_type_name | noise_floor | calculation_method | num_segments_used | last_calculated      | notes
-- -------------|----------------|-------------|--------------------|--------------------|----------------------|-------
-- 3            | ADC6           | 0.000234    | spectral_psd       | 189                | 2025-10-29 14:23:00  | NULL
-- 1            | ADC8           | 0.000156    | spectral_psd       | 245                | 2025-10-29 14:25:00  | NULL
-- 2            | ADC10          | 0.000089    | spectral_psd       | 312                | 2025-10-29 14:28:00  | NULL


-- ============================================================================
-- PART 4: EXAMPLE LINKING TO EXPERIMENT
-- ============================================================================
-- NOTE: These are EXAMPLE commands only. Do NOT link to existing experiments.
-- Use these as templates when creating new experiments.
-- ============================================================================

-- EXAMPLE: Link noise_floor to experiment
-- DO NOT EXECUTE - Replace <EXPERIMENT_ID> with your experiment ID
/*
INSERT INTO ml_experiments_amplitude_methods (experiment_id, method_id)
VALUES (<EXPERIMENT_ID>, 10);
*/


-- EXAMPLE: Link multiple scalers including noise_floor
-- DO NOT EXECUTE
/*
INSERT INTO ml_experiments_amplitude_methods (experiment_id, method_id)
VALUES
(<EXPERIMENT_ID>, 5),  -- raw
(<EXPERIMENT_ID>, 10);  -- noise_floor
*/


-- ============================================================================
-- VERIFICATION QUERY: Check experiment linking
-- ============================================================================

-- Check which experiments use noise_floor
SELECT
    eam.experiment_id,
    e.experiment_name,
    am.method_name,
    am.display_name
FROM ml_experiments_amplitude_methods eam
JOIN ml_experiments e ON eam.experiment_id = e.experiment_id
JOIN ml_amplitude_normalization_lut am ON eam.method_id = am.method_id
WHERE am.method_id = 10
ORDER BY eam.experiment_id;


-- ============================================================================
-- TESTING QUERY: Simulate noise_floor calculation
-- ============================================================================

-- Test query to show how noise floor would be applied
WITH test_data AS (
    SELECT
        nf.data_type_id,
        dt.data_type_name,
        nf.noise_floor,
        -- Simulate test ADC values
        ARRAY[100, 150, 200, 250] as test_values
    FROM experiment_noise_floor nf
    JOIN ml_data_types_lut dt ON nf.data_type_id = dt.data_type_id
),
calculations AS (
    SELECT
        data_type_name,
        noise_floor,
        test_values,
        -- Calculate mean
        (SELECT AVG(v) FROM UNNEST(test_values) v) as mean_value
    FROM test_data
)
SELECT
    data_type_name,
    noise_floor,
    test_values,
    mean_value,
    -- Show normalized calculation for first value
    ROUND(((test_values[1] - mean_value) / noise_floor)::numeric, 2) as normalized_first,
    -- Show normalized calculation for last value
    ROUND(((test_values[4] - mean_value) / noise_floor)::numeric, 2) as normalized_last
FROM calculations;


-- ============================================================================
-- QUERY: Count approved steady-state segments per data type
-- ============================================================================
-- Use this to see how many segments are available for noise floor calculation

SELECT
    dt.data_type_id,
    dt.data_type_name,
    COUNT(es.segment_id) as approved_steady_state_segments
FROM ml_data_types_lut dt
LEFT JOIN experiment_status es ON dt.data_type_id = es.data_type_id
WHERE es.status = true
  AND es.segment_type = 'steady_state'
  AND EXISTS (
      -- Only count segments from approved files
      SELECT 1
      FROM experiment_status es2
      WHERE es2.file_id = es.file_id
        AND es2.segment_id IS NULL
        AND es2.status = true
  )
GROUP BY dt.data_type_id, dt.data_type_name
ORDER BY dt.data_type_name;

-- Expected result (example):
-- data_type_id | data_type_name | approved_steady_state_segments
-- -------------|----------------|-------------------------------
-- 3            | ADC6           | 189
-- 1            | ADC8           | 245
-- 2            | ADC10          | 312


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
-- MAINTENANCE QUERIES
-- ============================================================================

-- Clear all noise floor entries (requires confirmation in CLI)
-- WARNING: Only use via 'noise-floor-clear --all' command
/*
DELETE FROM experiment_noise_floor;
*/


-- Clear specific data type (requires confirmation in CLI)
-- WARNING: Only use via 'noise-floor-clear --type <id>' command
/*
DELETE FROM experiment_noise_floor
WHERE data_type_id = <DATA_TYPE_ID>;
*/


-- Update noise floor value manually (not recommended - use CLI instead)
/*
UPDATE experiment_noise_floor
SET
    noise_floor = <NEW_VALUE>,
    last_calculated = NOW(),
    notes = 'Manually updated'
WHERE data_type_id = <DATA_TYPE_ID>;
*/


-- ============================================================================
-- CLEANUP QUERY: Remove noise_floor registration (if needed)
-- ============================================================================
-- WARNING: Only use this if you need to unregister the method
-- This will cascade delete any experiment links
-- This will NOT delete the experiment_noise_floor table
/*
DELETE FROM ml_amplitude_normalization_lut
WHERE method_id = 10;
*/


-- Drop experiment_noise_floor table (DESTRUCTIVE)
-- WARNING: This will delete all calculated noise floor values
/*
DROP TABLE IF EXISTS experiment_noise_floor CASCADE;
*/


-- ============================================================================
-- SUMMARY
-- ============================================================================
-- After running this script:
-- 1. experiment_noise_floor table created with proper schema
-- 2. noise_floor method registered as method_id 10
-- 3. Scaler available for linking to experiments
-- 4. Use CLI commands to populate noise floor values:
--    - noise-floor-init (verify table exists)
--    - noise-floor-calculate --all (calculate from approved segments)
--    - noise-floor-show (display values)
--
-- To use:
-- 1. Calculate noise floors: noise-floor-calculate --all
-- 2. Link to experiment: INSERT INTO ml_experiments_amplitude_methods (experiment_id, method_id) VALUES (<EXP_ID>, 10);
-- 3. Extract features - noise_floor applied automatically
--
-- See documentation:
-- - noise_floor_scaler_specification.md (technical details)
-- - noise_floor_cli_commands.md (command reference)
-- ============================================================================
