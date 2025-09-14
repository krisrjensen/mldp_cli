#!/usr/bin/env python3
"""
Test Experiment 41 after Database Cleanup
Author: Kristophor Jensen
Date: 20250914
Description: Verify database changes for experiment 41
"""

import psycopg2
import os

def get_db_config():
    """Get database configuration from environment or defaults"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'arc_detection'),
        'user': os.getenv('DB_USER', 'kjensen'),
        'password': os.getenv('DB_PASSWORD', '')
    }

def test_experiment_41():
    """Test experiment 41 configuration"""
    config = get_db_config()
    
    print("==========================================")
    print("Testing Experiment 41 Configuration")
    print("==========================================\n")
    
    conn = None
    try:
        conn = psycopg2.connect(**config)
        cur = conn.cursor()
        
        # Test 1: Check feature categories
        print("Test 1: Feature Categories")
        print("-" * 40)
        cur.execute("""
            SELECT feature_id, feature_name, feature_category, behavior_type
            FROM ml_features_lut
            WHERE feature_name IN ('voltage', 'current', 'raw_data', 'variance', 'mean', 'stddev')
            ORDER BY feature_id
        """)
        features = cur.fetchall()
        for feat_id, name, category, behavior in features:
            cat_str = category if category else "None"
            driver = " (DRIVER)" if category in ['electrical', 'compute'] else ""
            print(f"  {feat_id:2d}: {name:15s} - {cat_str:12s} ({behavior}){driver}")
        
        # Test 2: Check deleted features are gone
        print("\nTest 2: Verify Redundant Features Deleted")
        print("-" * 40)
        cur.execute("""
            SELECT feature_id, feature_name 
            FROM ml_features_lut 
            WHERE feature_id IN (17, 19, 20, 21)
        """)
        deleted = cur.fetchall()
        if deleted:
            print("  ❌ Some redundant features still exist:")
            for feat_id, name in deleted:
                print(f"    - {feat_id}: {name}")
        else:
            print("  ✅ All redundant features successfully deleted")
        
        # Test 3: Check experiment 41 feature sets
        print("\nTest 3: Experiment 41 Feature Sets")
        print("-" * 40)
        cur.execute("""
            SELECT 
                efs.feature_set_id,
                fs.feature_set_name,
                efs.data_channel,
                efs.priority_order,
                efs.is_active,
                efn.n_value
            FROM ml_experiments_feature_sets efs
            JOIN ml_feature_sets_lut fs ON efs.feature_set_id = fs.feature_set_id
            LEFT JOIN ml_experiments_feature_n_values efn 
                ON efs.experiment_id = efn.experiment_id 
                AND efs.feature_set_id = efn.feature_set_id
            WHERE efs.experiment_id = 41
            ORDER BY efs.priority_order
        """)
        feature_sets = cur.fetchall()
        if feature_sets:
            for set_id, name, channel, priority, active, n_value in feature_sets:
                status = "✅ Active" if active else "⚪ Inactive"
                n_str = f"N={n_value}" if n_value else "N=None"
                print(f"  {status} Set {set_id:2d}: {name:20s} - Channel: {channel:15s} Priority: {priority} {n_str}")
        else:
            print("  No feature sets configured for experiment 41")
        
        # Test 4: Check feature set contents
        print("\nTest 4: Feature Set Contents")
        print("-" * 40)
        for set_id in [1, 6, 7, 8, 9, 10]:  # Check specific sets
            cur.execute("""
                SELECT 
                    fs.feature_set_name,
                    array_agg(fl.feature_name ORDER BY fsf.feature_order) as features
                FROM ml_feature_sets_lut fs
                LEFT JOIN ml_feature_set_features fsf ON fs.feature_set_id = fsf.feature_set_id
                LEFT JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                WHERE fs.feature_set_id = %s
                GROUP BY fs.feature_set_id, fs.feature_set_name
            """, (set_id,))
            result = cur.fetchone()
            if result:
                name, features = result
                if features[0]:  # Check if there are features (not NULL)
                    print(f"  Set {set_id:2d} ({name}): {', '.join(features)}")
                else:
                    print(f"  Set {set_id:2d} ({name}): ⚠️  EMPTY")
        
        # Test 5: Check views exist
        print("\nTest 5: Views")
        print("-" * 40)
        cur.execute("""
            SELECT viewname 
            FROM pg_views 
            WHERE viewname IN ('v_experiment_041_feature_parameters', 'v_feature_sets_detail')
            ORDER BY viewname
        """)
        views = cur.fetchall()
        for view in views:
            print(f"  ✅ {view[0]}")
        
        # Test 6: Check table doesn't exist
        print("\nTest 6: Removed Tables")
        print("-" * 40)
        cur.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE tablename = 'experiment_041_parameters'
        """)
        if cur.fetchone():
            print("  ❌ experiment_041_parameters table still exists")
        else:
            print("  ✅ experiment_041_parameters table successfully removed")
        
        # Test 7: Check column doesn't exist
        print("\nTest 7: Removed Columns")
        print("-" * 40)
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'ml_feature_sets_lut' 
              AND column_name = 'num_features'
        """)
        if cur.fetchone():
            print("  ❌ num_features column still exists")
        else:
            print("  ✅ num_features column successfully removed")
        
        print("\n==========================================")
        print("✅ All Tests Complete!")
        print("==========================================")
        
        cur.close()
        conn.close()
        return True
        
    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        if conn:
            conn.close()
        return False

if __name__ == "__main__":
    test_experiment_41()