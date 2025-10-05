#!/usr/bin/env python3
"""
Check feature set configuration for experiment 041
"""
import psycopg2
from psycopg2.extras import RealDictCursor

db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'arc_detection',
    'user': 'kjensen'
}

conn = psycopg2.connect(**db_config)
cursor = conn.cursor(cursor_factory=RealDictCursor)

# Get feature sets for experiment 041
cursor.execute("""
    SELECT
        efs.feature_set_id,
        fs.feature_set_name,
        efs.data_channel,
        efs.n_value,
        COUNT(fsf.feature_set_feature_id) as num_features
    FROM ml_experiments_feature_sets efs
    JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
    LEFT JOIN ml_feature_set_features fsf ON efs.feature_set_id = fsf.feature_set_id
    WHERE efs.experiment_id = 41
    GROUP BY efs.feature_set_id, fs.feature_set_name, efs.data_channel, efs.n_value
    ORDER BY efs.feature_set_id
""")

feature_sets = cursor.fetchall()

print(f"Feature Sets for Experiment 041:")
print("=" * 80)
print(f"Total feature sets: {len(feature_sets)}")
print()

for fs in feature_sets:
    name = fs['feature_set_name'] or 'Unknown'
    channel = fs['data_channel'] or 'Unknown'
    n_val = fs['n_value'] or 0
    print(f"ID {fs['feature_set_id']:2d}: {name:30s} "
          f"Channel: {channel:20s} N: {n_val:6d} Features: {fs['num_features']}")

print()

# Check for data type restrictions
cursor.execute("""
    SELECT DISTINCT feature_set_id
    FROM ml_experiments_feature_sets
    WHERE experiment_id = 41
""")
configured_sets = [row['feature_set_id'] for row in cursor]

print(f"Feature sets configured: {len(configured_sets)}")
print(f"Expected feature files per segment: {len(configured_sets)}")

cursor.close()
conn.close()
