#!/bin/bash

# Database Cleanup Execution Script
# Author: Kristophor Jensen
# Date: 20250914
# Description: Execute approved database cleanup actions

# Database connection parameters
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-arcdb}"
DB_USER="${DB_USER:-kjensen}"

echo "=========================================="
echo "Database Cleanup Implementation"
echo "=========================================="
echo "Database: $DB_NAME"
echo "Host: $DB_HOST:$DB_PORT"
echo "User: $DB_USER"
echo ""

# Execute the SQL script
echo "Executing database cleanup..."
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f implement_database_cleanup.sql -a

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Database cleanup completed successfully!"
    echo ""
    echo "Summary of changes:"
    echo "  - Updated voltage (16) and current (18) to 'electrical' category"
    echo "  - Updated raw_data to 'compute' category"
    echo "  - Removed redundant variance features (17, 19, 20, 21)"
    echo "  - Dropped experiment_041_parameters table"
    echo "  - Renamed view to v_experiment_041_feature_parameters"
    echo "  - Dropped num_features column from ml_feature_sets_lut"
    echo ""
    echo "⚠️  Note: Feature sets 6, 7, 8, 9, 10 may need updating to use generic variance"
else
    echo ""
    echo "❌ Error executing database cleanup!"
    exit 1
fi