#!/usr/bin/env python3
"""
Filename: test_remove_labels.py
Author: Kristophor Jensen
Date Created: 20250916_120000
Date Revised: 20250916_120000
File version: 1.0.0.0
Description: Test remove-file-labels command
"""

import sys
import psycopg2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Test removing file labels"""
    print("\n" + "="*70)
    print("TEST REMOVE-FILE-LABELS COMMAND")
    print("="*70)
    
    # Connect to database
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='arc_detection',
            user='kjensen'
        )
        print("✅ Connected to database")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return False
    
    cursor = conn.cursor()
    experiment_id = 41
    table_name = f"experiment_{experiment_id:03d}_file_training_data"
    
    # Show current state
    print(f"\n📊 Current state of {table_name}:")
    cursor.execute(f"""
        SELECT file_label_name, COUNT(*) as count
        FROM {table_name}
        GROUP BY file_label_name
        ORDER BY file_label_name
    """)
    
    before_labels = {}
    for row in cursor:
        before_labels[row[0]] = row[1]
        print(f"   {row[0]}: {row[1]} files")
    
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    before_total = cursor.fetchone()[0]
    print(f"\nTotal files before: {before_total}")
    
    # Labels to remove
    labels_to_remove = [
        'trash', 'voltage_only', 'arc_short_gap', 
        'arc_extinguish', 'other', 'parallel_motor_continuous'
    ]
    
    print(f"\n🗑️  Removing labels: {', '.join(labels_to_remove)}")
    
    # Calculate expected removals
    expected_removed = 0
    for label in labels_to_remove:
        if label in before_labels:
            expected_removed += before_labels[label]
            print(f"   Will remove {label}: {before_labels[label]} files")
    
    # Perform the deletion
    cursor.execute(f"""
        DELETE FROM {table_name}
        WHERE file_label_name = ANY(%s)
    """, (labels_to_remove,))
    
    actual_removed = cursor.rowcount
    conn.commit()
    
    print(f"\n✅ Removed {actual_removed} files (expected: {expected_removed})")
    
    # Show final state
    print(f"\n📊 Final state of {table_name}:")
    cursor.execute(f"""
        SELECT file_label_name, COUNT(*) as count
        FROM {table_name}
        GROUP BY file_label_name
        ORDER BY file_label_name
    """)
    
    for row in cursor:
        print(f"   {row[0]}: {row[1]} files")
    
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    after_total = cursor.fetchone()[0]
    print(f"\nTotal files after: {after_total}")
    print(f"Difference: {before_total - after_total} files removed")
    
    # Verify none of the removed labels exist
    cursor.execute(f"""
        SELECT file_label_name 
        FROM {table_name}
        WHERE file_label_name = ANY(%s)
    """, (labels_to_remove,))
    
    remaining = cursor.fetchall()
    if remaining:
        print(f"\n❌ ERROR: Some labels were not removed: {remaining}")
        success = False
    else:
        print(f"\n✅ SUCCESS: All specified labels were removed")
        success = True
    
    conn.close()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)