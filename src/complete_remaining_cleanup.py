#!/usr/bin/env python3
"""
Complete Remaining Database Cleanup
Author: Kristophor Jensen
Date: 20250914
Description: Complete the cleanup tasks that were rolled back
"""

import psycopg2
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

def complete_remaining_cleanup():
    """Complete the remaining cleanup tasks"""
    config = get_db_config()
    
    print("==========================================")
    print("Completing Remaining Database Cleanup")
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
        
        print("Executing remaining cleanup tasks...")
        print("")
        
        # Step 1: Update Feature Categories
        print("Step 1: Updating feature categories...")
        cur.execute("UPDATE ml_features_lut SET feature_category = 'electrical' WHERE feature_id = 16")
        print(f"  ✅ Updated voltage (ID 16) to 'electrical' category")
        
        cur.execute("UPDATE ml_features_lut SET feature_category = 'electrical' WHERE feature_id = 18")
        print(f"  ✅ Updated current (ID 18) to 'electrical' category")
        
        cur.execute("UPDATE ml_features_lut SET feature_category = 'compute' WHERE feature_name = 'raw_data'")
        print(f"  ✅ Updated raw_data to 'compute' category")
        
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
        print(f"  ✅ Removed {cur.rowcount} feature associations")
        
        # Step 4: Add generic variance to affected feature sets
        print("\nStep 4: Adding generic variance to affected feature sets...")
        affected_set_ids = list(set(row[0] for row in affected_sets))
        
        for set_id in affected_set_ids:
            # Check if variance (ID 2) already exists in the set
            cur.execute("""
                SELECT 1 FROM ml_feature_set_features 
                WHERE feature_set_id = %s AND feature_id = 2
            """, (set_id,))
            
            if not cur.fetchone():
                # Get the max feature_order for this set
                cur.execute("""
                    SELECT COALESCE(MAX(feature_order), 0) + 1
                    FROM ml_feature_set_features
                    WHERE feature_set_id = %s
                """, (set_id,))
                next_order = cur.fetchone()[0]
                
                # Get next feature_set_feature_id
                cur.execute("SELECT COALESCE(MAX(feature_set_feature_id), 0) + 1 FROM ml_feature_set_features")
                next_id = cur.fetchone()[0]
                
                # Add generic variance feature
                cur.execute("""
                    INSERT INTO ml_feature_set_features 
                    (feature_set_feature_id, feature_set_id, feature_id, feature_order)
                    VALUES (%s, %s, 2, %s)
                """, (next_id, set_id, next_order))
                print(f"    ✅ Added generic variance to set {set_id}")
            else:
                print(f"    ℹ️  Set {set_id} already has generic variance")
        
        # Step 5: Delete redundant features
        print("\nStep 5: Deleting redundant features from ml_features_lut...")
        cur.execute("DELETE FROM ml_features_lut WHERE feature_id IN (17, 19, 20, 21)")
        print(f"  ✅ Deleted {cur.rowcount} redundant features")
        
        # Step 6: Drop redundant table
        print("\nStep 6: Dropping redundant table...")
        cur.execute("DROP TABLE IF EXISTS experiment_041_parameters")
        print("  ✅ Dropped experiment_041_parameters table")
        
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
        
        # Verify table is dropped
        cur.execute("""
            SELECT tablename FROM pg_tables 
            WHERE tablename = 'experiment_041_parameters'
        """)
        if cur.fetchone():
            print("⚠️  Warning: experiment_041_parameters table still exists!")
        else:
            print("✅ experiment_041_parameters table successfully removed")
        
        # Show updated feature sets
        print("\n==========================================")
        print("Updated Feature Sets")
        print("==========================================")
        for set_id in affected_set_ids:
            cur.execute("""
                SELECT 
                    fs.feature_set_name,
                    array_agg(fl.feature_name ORDER BY fsf.feature_order) as features
                FROM ml_feature_sets_lut fs
                JOIN ml_feature_set_features fsf ON fs.feature_set_id = fsf.feature_set_id
                JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                WHERE fs.feature_set_id = %s
                GROUP BY fs.feature_set_id, fs.feature_set_name
            """, (set_id,))
            result = cur.fetchone()
            if result:
                name, features = result
                print(f"  Set {set_id} ({name}): {', '.join(features)}")
        
        # Commit changes
        conn.commit()
        print("\n==========================================")
        print("✅ Remaining Cleanup Completed Successfully!")
        print("==========================================")
        
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
    success = complete_remaining_cleanup()
    sys.exit(0 if success else 1)