-- Add data channel support to feature sets
-- This allows specifying whether features are calculated from source current or load voltage

-- Step 1: Add data_channel column to ml_experiments_feature_sets junction table
ALTER TABLE ml_experiments_feature_sets 
ADD COLUMN IF NOT EXISTS data_channel VARCHAR(50) DEFAULT 'load_voltage';

-- Step 2: Add constraint to ensure valid values
ALTER TABLE ml_experiments_feature_sets
ADD CONSTRAINT check_data_channel 
CHECK (data_channel IN ('source_current', 'load_voltage'));

-- Step 3: Update existing rows to have a default channel (optional)
-- UPDATE ml_experiments_feature_sets 
-- SET data_channel = 'load_voltage' 
-- WHERE data_channel IS NULL;

-- Step 4: Create index for performance
CREATE INDEX IF NOT EXISTS idx_experiment_feature_channel 
ON ml_experiments_feature_sets(experiment_id, feature_set_id, data_channel);

-- To view the updated table structure:
-- \d ml_experiments_feature_sets