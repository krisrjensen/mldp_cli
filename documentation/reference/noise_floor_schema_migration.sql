-- ============================================================================
-- Noise Floor Schema Migration Script
-- ============================================================================
-- Filename: noise_floor_schema_migration.sql
-- Author: Kristophor Jensen
-- Date Created: 2025-10-29
-- Description: Migrates experiment_noise_floor table from old schema to new
--
-- OLD SCHEMA:
--   noise_floor DOUBLE PRECISION (single column)
--
-- NEW SCHEMA:
--   noise_floor_voltage DOUBLE PRECISION
--   noise_floor_current DOUBLE PRECISION
--
-- IMPORTANT: This will DROP existing noise floor data!
-- Back up your data before running this migration.
-- ============================================================================

-- ============================================================================
-- OPTION 1: DROP AND RECREATE (DESTRUCTIVE - loses all data)
-- ============================================================================
-- Use this if you don't need to preserve existing noise floor values

/*
DROP TABLE IF EXISTS experiment_noise_floor CASCADE;

CREATE TABLE experiment_noise_floor (
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

CREATE INDEX idx_experiment_noise_floor_data_type
ON experiment_noise_floor(data_type_id);

COMMENT ON TABLE experiment_noise_floor IS 'Stores pre-calculated noise floor values per data type for amplitude normalization';
COMMENT ON COLUMN experiment_noise_floor.data_type_id IS 'Links to ml_data_types_lut (ADC6, ADC8, ADC10, etc.)';
COMMENT ON COLUMN experiment_noise_floor.noise_floor_voltage IS 'Voltage channel noise floor in ADC counts (RMS)';
COMMENT ON COLUMN experiment_noise_floor.noise_floor_current IS 'Current channel noise floor in ADC counts (RMS)';
COMMENT ON COLUMN experiment_noise_floor.calculation_method IS 'Method used for calculation (default: std_dev)';
COMMENT ON COLUMN experiment_noise_floor.num_segments_used IS 'Number of approved steady-state segments (8192 samples) used in calculation';
COMMENT ON COLUMN experiment_noise_floor.last_calculated IS 'Timestamp of when noise floor was calculated';
COMMENT ON COLUMN experiment_noise_floor.notes IS 'Optional notes about calculation or data quality';
*/

-- ============================================================================
-- OPTION 2: ALTER TABLE (preserves metadata, but clears noise floor values)
-- ============================================================================
-- Use this if you want to preserve num_segments_used and last_calculated

-- Step 1: Check current schema
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'experiment_noise_floor'
ORDER BY ordinal_position;

-- Step 2: Backup existing data (IMPORTANT!)
CREATE TABLE IF NOT EXISTS experiment_noise_floor_backup AS
SELECT * FROM experiment_noise_floor;

-- Step 3: Drop the old noise_floor column
ALTER TABLE experiment_noise_floor
DROP COLUMN IF EXISTS noise_floor;

-- Step 4: Add new columns
ALTER TABLE experiment_noise_floor
ADD COLUMN IF NOT EXISTS noise_floor_voltage DOUBLE PRECISION;

ALTER TABLE experiment_noise_floor
ADD COLUMN IF NOT EXISTS noise_floor_current DOUBLE PRECISION;

-- Step 5: Update calculation_method default
ALTER TABLE experiment_noise_floor
ALTER COLUMN calculation_method SET DEFAULT 'std_dev';

-- Step 6: Make new columns NOT NULL (will fail if there's existing data)
-- NOTE: Skip this step if table has existing rows
-- You'll need to recalculate noise floors first
/*
ALTER TABLE experiment_noise_floor
ALTER COLUMN noise_floor_voltage SET NOT NULL;

ALTER TABLE experiment_noise_floor
ALTER COLUMN noise_floor_current SET NOT NULL;
*/

-- Step 7: Update column comments
COMMENT ON COLUMN experiment_noise_floor.noise_floor_voltage IS 'Voltage channel noise floor in ADC counts (RMS)';
COMMENT ON COLUMN experiment_noise_floor.noise_floor_current IS 'Current channel noise floor in ADC counts (RMS)';
COMMENT ON COLUMN experiment_noise_floor.calculation_method IS 'Method used for calculation (default: std_dev)';
COMMENT ON COLUMN experiment_noise_floor.num_segments_used IS 'Number of approved steady-state segments (8192 samples) used in calculation';

-- ============================================================================
-- OPTION 3: In-place ALTER (if table is empty or you want to clear it)
-- ============================================================================
-- Use this if you have an empty table or want to clear it first

/*
-- Clear existing data
DELETE FROM experiment_noise_floor;

-- Drop old column
ALTER TABLE experiment_noise_floor DROP COLUMN IF EXISTS noise_floor;

-- Add new columns
ALTER TABLE experiment_noise_floor ADD COLUMN noise_floor_voltage DOUBLE PRECISION NOT NULL;
ALTER TABLE experiment_noise_floor ADD COLUMN noise_floor_current DOUBLE PRECISION NOT NULL;

-- Update default
ALTER TABLE experiment_noise_floor ALTER COLUMN calculation_method SET DEFAULT 'std_dev';
*/

-- ============================================================================
-- VERIFICATION
-- ============================================================================

-- Check new schema
SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_name = 'experiment_noise_floor'
ORDER BY ordinal_position;

-- Expected output:
-- column_name          | data_type           | is_nullable | column_default
-- ---------------------|---------------------|-------------|----------------
-- data_type_id         | integer             | NO          |
-- noise_floor_voltage  | double precision    | NO          |
-- noise_floor_current  | double precision    | NO          |
-- calculation_method   | character varying   | YES         | 'std_dev'::character varying
-- num_segments_used    | integer             | YES         |
-- last_calculated      | timestamp without   | YES         | now()
-- notes                | text                | YES         |

-- Show table contents (should be empty after migration)
SELECT * FROM experiment_noise_floor;

-- ============================================================================
-- POST-MIGRATION STEPS
-- ============================================================================
-- 1. Run noise-floor-calculate --all to recalculate with new schema
-- 2. Run noise-floor-show to verify values are correct
-- 3. Drop backup table once verified:
--    DROP TABLE IF EXISTS experiment_noise_floor_backup;
-- ============================================================================
