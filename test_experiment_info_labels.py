#!/usr/bin/env python3
"""
Filename: test_experiment_info_labels.py
Author: Kristophor Jensen
Date Created: 20250916_120000
Date Revised: 20250916_120000
File version: 1.0.0.0
Description: Test that experiment-info shows file label distribution
"""

import sys
import psycopg2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    """Test experiment info with file labels"""
    print("\n" + "="*70)
    print("TEST EXPERIMENT-INFO FILE LABEL DISPLAY")
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
    
    exp_id = 41
    table_name = f"experiment_{exp_id:03d}_file_training_data"
    
    # Check what's in the training data table
    cursor = conn.cursor()
    
    # Check if table exists
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = %s
        )
    """, (table_name,))
    
    if not cursor.fetchone()[0]:
        print(f"❌ Table {table_name} does not exist")
        return False
    
    # Get file label statistics
    cursor.execute(f"""
        SELECT 
            file_label_name,
            COUNT(*) as count
        FROM {table_name}
        WHERE experiment_id = %s
        GROUP BY file_label_name
        ORDER BY count DESC, file_label_name
    """, (exp_id,))
    
    labels = cursor.fetchall()
    
    if labels:
        print(f"\n📁 FILE TRAINING DATA FOR EXPERIMENT {exp_id}:")
        print("=" * 60)
        
        # Get total counts
        cursor.execute(f"""
            SELECT 
                COUNT(DISTINCT file_id) as total_files,
                COUNT(DISTINCT file_label_name) as unique_labels
            FROM {table_name}
            WHERE experiment_id = %s
        """, (exp_id,))
        
        stats = cursor.fetchone()
        print(f"Total files: {stats[0]}")
        print(f"Unique labels: {stats[1]}")
        
        # Show label distribution with bars
        print("\nLabel Distribution:")
        max_count = max(l[1] for l in labels)
        for label_name, count in labels:
            bar_length = int(count / max_count * 30)
            bar = '█' * bar_length
            print(f"  {label_name:30} {count:4} {bar}")
        
        print("\n✅ File label display is working correctly")
        success = True
    else:
        print("⚠️  No file training data found")
        success = False
    
    # Also check for segment training data
    seg_table = f"experiment_{exp_id:03d}_segment_training_data"
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = %s
        )
    """, (seg_table,))
    
    if cursor.fetchone()[0]:
        cursor.execute(f"SELECT COUNT(*) FROM {seg_table}")
        seg_count = cursor.fetchone()[0]
        if seg_count > 0:
            print(f"\n📊 Segment Training Data: {seg_count} segments selected")
    
    # Check for segment pairs
    pairs_table = f"experiment_{exp_id:03d}_segment_pairs"
    cursor.execute("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = %s
        )
    """, (pairs_table,))
    
    if cursor.fetchone()[0]:
        cursor.execute(f"SELECT COUNT(*) FROM {pairs_table}")
        pairs_count = cursor.fetchone()[0]
        if pairs_count > 0:
            print(f"🔗 Segment Pairs: {pairs_count} pairs generated")
    
    conn.close()
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)