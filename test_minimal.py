#!/usr/bin/env python3
"""
Filename: test_minimal.py
Author: Kristophor Jensen
Date Created: 20250916_110000
Date Revised: 20250916_110000
File version: 1.0.0.0
Description: Minimal test to verify basic pipeline functionality
"""

import sys
import psycopg2
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from experiment_file_selector import ExperimentFileSelector

def main():
    """Minimal test"""
    print("\n" + "="*70)
    print("MINIMAL PIPELINE TEST")
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
    
    # Check if we have any files in the database
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM files")
    file_count = cursor.fetchone()[0]
    print(f"\n📊 Total files in database: {file_count}")
    
    if file_count == 0:
        print("❌ No files in database - cannot proceed with testing")
        return False
    
    # Check if we have labeled files
    cursor.execute("SELECT COUNT(DISTINCT file_id) FROM files_y WHERE label_text IS NOT NULL")
    labeled_count = cursor.fetchone()[0]
    print(f"📊 Labeled files: {labeled_count}")
    
    if labeled_count == 0:
        print("⚠️  No labeled files - will work with unlabeled data")
    
    # Check label distribution
    cursor.execute("""
        SELECT label_text, COUNT(*) as count 
        FROM files_y 
        WHERE label_text IS NOT NULL
        GROUP BY label_text
        ORDER BY count DESC
        LIMIT 10
    """)
    
    print("\n📊 Label distribution (top 10):")
    for row in cursor:
        print(f"   {row[0]}: {row[1]} files")
    
    # Test file selector with experiment 41
    print("\n" + "="*70)
    print("TESTING FILE SELECTOR")
    print("="*70)
    
    selector = ExperimentFileSelector(41, conn)
    
    # Get available files
    files_by_label = selector.get_available_files()
    print(f"\n📊 Available files by label:")
    for label, files in files_by_label.items():
        print(f"   {label}: {len(files)} files")
    
    if not files_by_label:
        print("❌ No files available for selection")
        return False
    
    # Try to select files
    print("\n🔄 Attempting file selection...")
    result = selector.select_files(
        strategy='random',
        max_files_per_label=10,  # Start small
        seed=42
    )
    
    if result['success']:
        print(f"✅ File selection successful!")
        print(f"   Total selected: {result['total_selected']}")
        if 'statistics' in result:
            stats = result['statistics']
            if 'label_counts' in stats:
                print("\n📊 Selected files by label:")
                for label, count in stats['label_counts'].items():
                    print(f"   {label}: {count}")
    else:
        print(f"❌ File selection failed: {result.get('error', 'Unknown')}")
        return False
    
    # Verify table was created
    cursor.execute("""
        SELECT COUNT(*) 
        FROM experiment_041_file_training_data
    """)
    count = cursor.fetchone()[0]
    print(f"\n✅ Table created with {count} records")
    
    conn.close()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)