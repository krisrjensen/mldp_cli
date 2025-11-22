#!/usr/bin/env python3
"""Quick script to get experiment042 data from database and create verification features"""
import psycopg2
import numpy as np
from pathlib import Path

# Query database for all combinations
conn = psycopg2.connect(host='localhost', database='arc_detection', user='kjensen')
cursor = conn.cursor()

cursor.execute("""
SELECT DISTINCT
  dt.data_type_name,
  dt.data_type_id,
  sf.decimation_factor
FROM experiment_042_classifier_001_svm_features sf
JOIN ml_data_types_lut dt ON sf.data_type_id = dt.data_type_id
WHERE sf.amplitude_processing_method_id = (SELECT method_id FROM ml_amplitude_normalization_lut WHERE method_name = 'zscore')
ORDER BY dt.data_type_name, sf.decimation_factor
""")

combinations = cursor.fetchall()
print(f"Found {len(combinations)} data type/decimation combinations:")
for dtype, dtype_id, dec in combinations:
    print(f"  {dtype} (ID={dtype_id}), decimation={dec}")

conn.close()
