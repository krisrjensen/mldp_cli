#!/usr/bin/env python3
"""
Database Cleanup Implementation
Author: Kristophor Jensen
Date: 20250914
Description: Execute approved database cleanup actions
"""

import psycopg2
from psycopg2 import sql
import os
import sys

def get_db_config():
    """Get database configuration from environment or defaults"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'arc_detection'),
        'user': os.getenv('DB_USER', 'kjensen'),
        'password': os.getenv('DB_PASSWORD', '')
    }

def execute_cleanup():
    """Execute the database cleanup operations"""
    config = get_db_config()
    
    print("==========================================")
    print("Database Cleanup Implementation")
    print("==========================================")
    print(f"Database: {config['database']}")
    print(f"Host: {config['host']}:{config['port']}")
    print(f"User: {config['user']}")
    print("")
    
    conn = None
    try:
        # Connect to database
        conn = psycopg2.connect(**config)
        conn.autocommit = False
        cur = conn.cursor()
        
        print("Executing database cleanup...")
        print("")
        
        # Step 1: Update Feature Categories
        print("Step 1: Updating feature categories...")
        cur.execute("UPDATE ml_features_lut SET feature_category = 'electrical' WHERE feature_id = 16")
        print(f"  - Updated voltage (ID 16) to 'electrical' category")
        
        cur.execute("UPDATE ml_features_lut SET feature_category = 'electrical' WHERE feature_id = 18")
        print(f"  - Updated current (ID 18) to 'electrical' category")
        
        cur.execute("UPDATE ml_features_lut SET feature_category = 'compute' WHERE feature_name = 'raw_data'")
        print(f"  - Updated raw_data to 'compute' category")
        
        # Step 2: Check affected feature sets before deletion
        print("\nStep 2: Checking affected feature sets...")
        cur.execute("""
            SELECT fs.feature_set_id, fs.feature_set_name, fl.feature_id, fl.feature_name
            FROM ml_feature_set_features fsf
            JOIN ml_feature_sets_lut fs ON fsf.feature_set_id = fs.feature_set_id
            JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
            WHERE fsf.feature_id IN (17, 19, 20, 21)
            ORDER BY fs.feature_set_id
        """)
        affected_sets = cur.fetchall()
        if affected_sets:
            print("  Affected feature sets:")
            for set_id, set_name, feat_id, feat_name in affected_sets:
                print(f"    - Set {set_id} ({set_name}) uses {feat_name} (ID {feat_id})")
        
        # Step 3: Remove redundant features from feature sets
        print("\nStep 3: Removing redundant features from feature sets...")
        cur.execute("DELETE FROM ml_feature_set_features WHERE feature_id IN (17, 19, 20, 21)")
        print(f"  - Removed {cur.rowcount} feature associations")
        
        # Step 4: Delete redundant features
        print("\nStep 4: Deleting redundant features from ml_features_lut...")
        cur.execute("DELETE FROM ml_features_lut WHERE feature_id IN (17, 19, 20, 21)")
        print(f"  - Deleted {cur.rowcount} redundant features")
        
        # Step 5: Drop redundant table
        print("\nStep 5: Dropping redundant table...")
        cur.execute("DROP TABLE IF EXISTS experiment_041_parameters")
        print("  - Dropped experiment_041_parameters table")
        
        # Step 6: Rename view
        print("\nStep 6: Renaming view to follow convention...")
        cur.execute("ALTER VIEW IF EXISTS experiment_041_feature_parameters RENAME TO v_experiment_041_feature_parameters")
        print("  - Renamed view to v_experiment_041_feature_parameters")
        
        # Step 7: Drop redundant column
        print("\nStep 7: Dropping redundant column...")
        cur.execute("ALTER TABLE ml_feature_sets_lut DROP COLUMN IF EXISTS num_features")
        print("  - Dropped num_features column from ml_feature_sets_lut")
        
        # Verification
        print("\n==========================================")
        print("Verification")
        print("==========================================")
        
        # Verify feature categories
        cur.execute("""
            SELECT feature_id, feature_name, feature_category 
            FROM ml_features_lut 
            WHERE feature_id IN (16, 18) OR feature_name = 'raw_data'
        """)
        updated_features = cur.fetchall()
        print("\nUpdated feature categories:")
        for feat_id, feat_name, feat_cat in updated_features:
            print(f"  - {feat_name} (ID {feat_id}): {feat_cat}")
        
        # Verify deleted features
        cur.execute("SELECT feature_id FROM ml_features_lut WHERE feature_id IN (17, 19, 20, 21)")
        if cur.fetchone():
            print("\n⚠️  Warning: Some redundant features were not deleted!")
        else:
            print("\n✅ All redundant features successfully deleted")
        
        # Commit changes
        conn.commit()
        print("\n==========================================")
        print("✅ Database cleanup completed successfully!")
        print("==========================================")
        
        if affected_sets:
            print("\n⚠️  Note: The following feature sets were modified:")
            seen_sets = set()
            for set_id, set_name, _, _ in affected_sets:
                if set_id not in seen_sets:
                    print(f"  - Set {set_id}: {set_name}")
                    seen_sets.add(set_id)
            print("\nThese sets may need to be updated to use generic 'variance' feature")
        
        cur.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if conn:
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    success = execute_cleanup()
    sys.exit(0 if success else 1)