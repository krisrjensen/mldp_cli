#!/usr/bin/env python3
"""
Fix View Dependencies and Complete Database Cleanup
Author: Kristophor Jensen
Date: 20250914
Description: Handle view dependencies and complete the num_features column removal
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

def fix_views_and_complete_cleanup():
    """Fix view dependencies and complete cleanup"""
    config = get_db_config()
    
    print("==========================================")
    print("Fixing View Dependencies")
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
        
        # Step 1: Drop dependent views
        print("Step 1: Dropping dependent views...")
        cur.execute("DROP VIEW IF EXISTS v_experiment_041_feature_parameters CASCADE")
        print("  - Dropped v_experiment_041_feature_parameters")
        
        cur.execute("DROP VIEW IF EXISTS v_feature_sets_detail CASCADE")
        print("  - Dropped v_feature_sets_detail")
        
        # Step 2: Drop the num_features column
        print("\nStep 2: Dropping num_features column...")
        cur.execute("ALTER TABLE ml_feature_sets_lut DROP COLUMN IF EXISTS num_features")
        print("  - Dropped num_features column from ml_feature_sets_lut")
        
        # Step 3: Recreate views
        print("\nStep 3: Recreating views without num_features dependency...")
        
        # Recreate v_experiment_041_feature_parameters
        cur.execute("""
            CREATE OR REPLACE VIEW v_experiment_041_feature_parameters AS
            SELECT 
                efs.experiment_id,
                efs.feature_set_id,
                fs.feature_set_name,
                efs.data_channel,
                efs.priority_order,
                efs.is_active,
                (SELECT COUNT(*) 
                 FROM ml_feature_set_features fsf 
                 WHERE fsf.feature_set_id = efs.feature_set_id) as num_features,
                efn.n_value
            FROM ml_experiments_feature_sets efs
            JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
            LEFT JOIN ml_experiments_feature_n_values efn 
                ON efs.experiment_id = efn.experiment_id 
                AND efs.feature_set_id = efn.feature_set_id
            WHERE efs.experiment_id = 41
        """)
        print("  - Recreated v_experiment_041_feature_parameters")
        
        # Recreate v_feature_sets_detail
        cur.execute("""
            CREATE OR REPLACE VIEW v_feature_sets_detail AS
            SELECT 
                fs.feature_set_id,
                fs.feature_set_name,
                (SELECT COUNT(*) 
                 FROM ml_feature_set_features fsf 
                 WHERE fsf.feature_set_id = fs.feature_set_id) as num_features,
                fs.created_date,
                fs.modified_date
            FROM ml_feature_sets_lut fs
        """)
        print("  - Recreated v_feature_sets_detail")
        
        # Verification
        print("\n==========================================")
        print("Verification")
        print("==========================================")
        
        # Check column is dropped
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'ml_feature_sets_lut' 
              AND column_name = 'num_features'
        """)
        if cur.fetchone():
            print("⚠️  Warning: num_features column still exists!")
        else:
            print("✅ num_features column successfully dropped")
        
        # Check views exist
        cur.execute("""
            SELECT viewname 
            FROM pg_views 
            WHERE viewname IN ('v_experiment_041_feature_parameters', 'v_feature_sets_detail')
        """)
        views = cur.fetchall()
        print(f"\n✅ Recreated {len(views)} views:")
        for view in views:
            print(f"  - {view[0]}")
        
        # Show affected feature sets
        print("\n==========================================")
        print("Affected Feature Sets (need updating)")
        print("==========================================")
        cur.execute("""
            SELECT 
                fs.feature_set_id,
                fs.feature_set_name,
                COUNT(fsf.feature_id) as feature_count
            FROM ml_feature_sets_lut fs
            LEFT JOIN ml_feature_set_features fsf ON fs.feature_set_id = fsf.feature_set_id
            WHERE fs.feature_set_id IN (6, 7, 8, 9, 10)
            GROUP BY fs.feature_set_id, fs.feature_set_name
            ORDER BY fs.feature_set_id
        """)
        affected = cur.fetchall()
        for set_id, set_name, count in affected:
            status = "⚠️ Empty - needs features" if count == 0 else f"Has {count} feature(s)"
            print(f"  - Set {set_id}: {set_name} - {status}")
        
        # Commit changes
        conn.commit()
        print("\n==========================================")
        print("✅ View dependencies fixed successfully!")
        print("==========================================")
        
        print("\n📋 Summary of all cleanup actions completed:")
        print("  ✅ Updated voltage/current to 'electrical' category")
        print("  ✅ Updated raw_data to 'compute' category")
        print("  ✅ Removed redundant variance features from feature sets")
        print("  ✅ Deleted redundant variance features from ml_features_lut")
        print("  ✅ Dropped experiment_041_parameters table")
        print("  ✅ Renamed view to v_experiment_041_feature_parameters")
        print("  ✅ Dropped num_features column from ml_feature_sets_lut")
        print("  ✅ Recreated views with dynamic num_features calculation")
        
        if any(count == 0 for _, _, count in affected):
            print("\n⚠️  Next step: Add generic 'variance' feature to empty feature sets")
        
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
    success = fix_views_and_complete_cleanup()
    sys.exit(0 if success else 1)