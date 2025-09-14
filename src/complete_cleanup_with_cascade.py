#!/usr/bin/env python3
"""
Complete Database Cleanup with CASCADE
Author: Kristophor Jensen
Date: 20250914
Description: Drop num_features column with CASCADE and recreate views
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

def complete_cleanup():
    """Complete cleanup with CASCADE"""
    config = get_db_config()
    
    print("==========================================")
    print("Complete Database Cleanup with CASCADE")
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
        
        # Step 1: Drop the num_features column with CASCADE
        print("Step 1: Dropping num_features column with CASCADE...")
        print("  This will drop dependent views automatically")
        cur.execute("ALTER TABLE ml_feature_sets_lut DROP COLUMN IF EXISTS num_features CASCADE")
        print("  ✅ Dropped num_features column and dependent views")
        
        # Step 2: Recreate the important views
        print("\nStep 2: Recreating views with dynamic num_features...")
        
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
                efn.n_value,
                (SELECT array_agg(fl.feature_name ORDER BY fsf.feature_order) 
                 FROM ml_feature_set_features fsf
                 JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                 WHERE fsf.feature_set_id = efs.feature_set_id) as feature_names,
                (SELECT array_agg(fl.feature_id ORDER BY fsf.feature_order)
                 FROM ml_feature_set_features fsf
                 JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                 WHERE fsf.feature_set_id = efs.feature_set_id) as feature_ids
            FROM ml_experiments_feature_sets efs
            JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
            LEFT JOIN ml_experiments_feature_n_values efn 
                ON efs.experiment_id = efn.experiment_id 
                AND efs.feature_set_id = efn.feature_set_id
            WHERE efs.experiment_id = 41
        """)
        print("  ✅ Created v_experiment_041_feature_parameters")
        
        # Recreate v_feature_sets_detail
        cur.execute("""
            CREATE OR REPLACE VIEW v_feature_sets_detail AS
            SELECT 
                fsl.feature_set_id,
                fsl.feature_set_name,
                fsl.category,
                fsl.description,
                COUNT(fsf.feature_id) as num_features,
                string_agg(fl.feature_name::text, ', '::text ORDER BY fsf.feature_order) AS feature_names,
                array_agg(fl.feature_id ORDER BY fsf.feature_order) AS feature_ids,
                array_agg(fl.behavior_type ORDER BY fsf.feature_order) AS behavior_types
            FROM ml_feature_sets_lut fsl
            LEFT JOIN ml_feature_set_features fsf ON fsl.feature_set_id = fsf.feature_set_id
            LEFT JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
            GROUP BY fsl.feature_set_id, fsl.feature_set_name, fsl.category, fsl.description
        """)
        print("  ✅ Created v_feature_sets_detail")
        
        # Step 3: Verification
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
            print("❌ num_features column still exists!")
        else:
            print("✅ num_features column successfully dropped")
        
        # Check views exist
        cur.execute("""
            SELECT viewname 
            FROM pg_views 
            WHERE viewname IN ('v_experiment_041_feature_parameters', 'v_feature_sets_detail')
            ORDER BY viewname
        """)
        views = cur.fetchall()
        print(f"\n✅ Created {len(views)} views:")
        for view in views:
            print(f"  - {view[0]}")
        
        # Check that old view is gone
        cur.execute("""
            SELECT viewname 
            FROM pg_views 
            WHERE viewname = 'experiment_041_feature_parameters'
        """)
        if cur.fetchone():
            print("\n⚠️  Old view 'experiment_041_feature_parameters' still exists")
        else:
            print("\n✅ Old view 'experiment_041_feature_parameters' removed")
        
        # Step 4: Check feature sets that need updating
        print("\n==========================================")
        print("Feature Sets Status")
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
        
        empty_sets = []
        for set_id, set_name, count in affected:
            if count == 0:
                print(f"  ⚠️  Set {set_id} ({set_name}): EMPTY - needs features added")
                empty_sets.append((set_id, set_name))
            else:
                print(f"  ✅ Set {set_id} ({set_name}): Has {count} feature(s)")
        
        # Commit changes
        conn.commit()
        
        print("\n==========================================")
        print("✅ Database Cleanup COMPLETE!")
        print("==========================================")
        
        print("\n📋 All cleanup actions completed:")
        print("  ✅ Updated voltage/current to 'electrical' category")
        print("  ✅ Updated raw_data to 'compute' category")
        print("  ✅ Removed redundant variance features from feature sets")
        print("  ✅ Deleted redundant variance features from ml_features_lut")
        print("  ✅ Dropped experiment_041_parameters table")
        print("  ✅ Dropped num_features column from ml_feature_sets_lut")
        print("  ✅ Created v_experiment_041_feature_parameters view")
        print("  ✅ Created v_feature_sets_detail view")
        
        if empty_sets:
            print(f"\n⚠️  {len(empty_sets)} feature sets need updating:")
            for set_id, set_name in empty_sets:
                print(f"  - Set {set_id}: {set_name}")
            print("\nThese sets previously used redundant variance features.")
            print("They should be updated to use the generic 'variance' feature.")
        
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
    success = complete_cleanup()
    sys.exit(0 if success else 1)