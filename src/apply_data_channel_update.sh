#!/bin/bash

# Script to apply data channel support to the database
# Adds data_channel column to ml_experiments_feature_sets table

echo "========================================="
echo "Adding Data Channel Support to Database"
echo "========================================="

cd /Users/kjensen/Documents/GitHub/mldp/mldp_cli/src

echo -e "\n1. Applying database schema changes..."

# Apply the SQL changes
psql -h localhost -d arc_detection -U kjensen < add_data_channel_support.sql

echo -e "\n2. Verifying the update..."

# Check the table structure
echo "\\d ml_experiments_feature_sets" | psql -h localhost -d arc_detection -U kjensen | grep data_channel

if [ $? -eq 0 ]; then
    echo -e "\n✅ Data channel column successfully added!"
else
    echo -e "\n❌ Failed to add data channel column"
    exit 1
fi

echo -e "\n3. Testing the new functionality..."

# Test adding a feature set with data channel
echo "
-- Test adding a feature set with source_current channel
UPDATE ml_experiments_feature_sets 
SET data_channel = 'source_current' 
WHERE experiment_id = 41 
  AND feature_set_id = 7
LIMIT 1;

-- Check the update
SELECT experiment_id, feature_set_id, data_channel 
FROM ml_experiments_feature_sets 
WHERE experiment_id = 41 
LIMIT 5;
" | psql -h localhost -d arc_detection -U kjensen

echo -e "\n========================================="
echo "Database update complete!"
echo ""
echo "You can now use the data channel parameter:"
echo "  mldp> add-feature-set 7 --channel source_current"
echo "  mldp> add-feature-set 8,9,10 --channel load_voltage --n 128"
echo "========================================="