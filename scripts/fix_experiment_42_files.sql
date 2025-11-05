-- Filename: fix_experiment_42_files.sql
-- Author: Kristophor Jensen
-- Date Created: 20251104_000000
-- Description: Clean and repopulate experiment 42 with only desired labels
--
-- This script:
-- 1. Deletes ALL files from experiment 42 training data
-- 2. Adds 50 random files for each of 3 labels: arc, negative_transient, parallel_motor_arc
-- 3. Uses seed 42 for reproducibility

-- Step 1: Clear all existing files for experiment 42
DELETE FROM experiment_042_file_training_data WHERE experiment_id = 42;

-- Step 2: Add 50 'arc' files (random, seed 42)
INSERT INTO experiment_042_file_training_data
    (experiment_id, file_id, file_label_id, file_label_name, selection_order, selection_strategy, random_seed)
SELECT
    42 as experiment_id,
    f.file_id,
    fy.label_id as file_label_id,
    el.experiment_label as file_label_name,
    ROW_NUMBER() OVER (ORDER BY f.file_id) as selection_order,
    'random' as selection_strategy,
    42 as random_seed
FROM files_x f
LEFT JOIN files_y fy ON f.file_id = fy.file_id
LEFT JOIN experiment_labels el ON fy.label_id = el.label_id
WHERE f.total_samples > 0
  AND el.experiment_label = 'arc'
ORDER BY random()  -- PostgreSQL random ordering
LIMIT 50;

-- Step 3: Add 50 'negative_transient' files
INSERT INTO experiment_042_file_training_data
    (experiment_id, file_id, file_label_id, file_label_name, selection_order, selection_strategy, random_seed)
SELECT
    42 as experiment_id,
    f.file_id,
    fy.label_id as file_label_id,
    el.experiment_label as file_label_name,
    ROW_NUMBER() OVER (ORDER BY f.file_id) + 50 as selection_order,
    'random' as selection_strategy,
    42 as random_seed
FROM files_x f
LEFT JOIN files_y fy ON f.file_id = fy.file_id
LEFT JOIN experiment_labels el ON fy.label_id = el.label_id
WHERE f.total_samples > 0
  AND el.experiment_label = 'negative_transient'
ORDER BY random()
LIMIT 50;

-- Step 4: Add 50 'parallel_motor_arc' files
INSERT INTO experiment_042_file_training_data
    (experiment_id, file_id, file_label_id, file_label_name, selection_order, selection_strategy, random_seed)
SELECT
    42 as experiment_id,
    f.file_id,
    fy.label_id as file_label_id,
    el.experiment_label as file_label_name,
    ROW_NUMBER() OVER (ORDER BY f.file_id) + 100 as selection_order,
    'random' as selection_strategy,
    42 as random_seed
FROM files_x f
LEFT JOIN files_y fy ON f.file_id = fy.file_id
LEFT JOIN experiment_labels el ON fy.label_id = el.label_id
WHERE f.total_samples > 0
  AND el.experiment_label = 'parallel_motor_arc'
ORDER BY random()
LIMIT 50;

-- Step 5: Verify the selection
SELECT
    file_label_name,
    COUNT(*) as file_count
FROM experiment_042_file_training_data
WHERE experiment_id = 42
GROUP BY file_label_name
ORDER BY file_label_name;

-- Show total
SELECT COUNT(*) as total_files FROM experiment_042_file_training_data WHERE experiment_id = 42;
