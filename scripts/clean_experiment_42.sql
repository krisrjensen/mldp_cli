-- Clean experiment 42 training data
DELETE FROM experiment_042_segment_training_data WHERE experiment_id = 42;
DELETE FROM experiment_042_file_training_data WHERE experiment_id = 42;

-- Verify cleanup
SELECT 'Files remaining:' as status, COUNT(*) as count FROM experiment_042_file_training_data WHERE experiment_id = 42;
SELECT 'Segments remaining:' as status, COUNT(*) as count FROM experiment_042_segment_training_data WHERE experiment_id = 42;
