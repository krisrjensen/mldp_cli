#!/usr/bin/env python3
"""
Check and Delete Redundant Features
Author: Kristophor Jensen
Date: 20250914
Description: Verify and delete redundant variance features
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

def check_and_delete():
    """Check and delete redundant features"""
    config = get_db_config()
    
    print("==========================================")
    print("Checking Redundant Features")
    print("==========================================\n")
    
    conn = None
    try:
        conn = psycopg2.connect(**config)
        conn.autocommit = True  # Use autocommit to ensure changes persist
        cur = conn.cursor()
        
        # Step 1: Check what's actually in the database
        print("Step 1: Checking current state of ml_features_lut...")
        cur.execute("""
            SELECT feature_id, feature_name, feature_category
            FROM ml_features_lut
            WHERE feature_id IN (17, 19, 20, 21)
            ORDER BY feature_id
        """)
        redundant_features = cur.fetchall()
        
        if redundant_features:
            print("❌ FOUND REDUNDANT FEATURES THAT SHOULD BE DELETED:")
            for feat_id, name, category in redundant_features:
                print(f"  - ID {feat_id}: {name} (category: {category})")
            
            # Step 2: Check if they're used in feature sets
            print("\nStep 2: Checking if these features are used in feature sets...")
            cur.execute("""
                SELECT DISTINCT fsf.feature_id, fl.feature_name, fsf.feature_set_id
                FROM ml_feature_set_features fsf
                JOIN ml_features_lut fl ON fsf.feature_id = fl.feature_id
                WHERE fsf.feature_id IN (17, 19, 20, 21)
                ORDER BY fsf.feature_id
            """)
            usage = cur.fetchall()
            
            if usage:
                print("⚠️  These features are still being used in feature sets:")
                for feat_id, name, set_id in usage:
                    print(f"  - Feature {feat_id} ({name}) used in set {set_id}")
                
                # Step 3: Remove them from feature sets
                print("\nStep 3: Removing from feature sets...")
                cur.execute("DELETE FROM ml_feature_set_features WHERE feature_id IN (17, 19, 20, 21)")
                print(f"  ✅ Removed {cur.rowcount} feature associations")
            else:
                print("  ✅ Not used in any feature sets")
            
            # Step 4: Delete the features
            print("\nStep 4: DELETING REDUNDANT FEATURES...")
            cur.execute("DELETE FROM ml_features_lut WHERE feature_id IN (17, 19, 20, 21)")
            deleted_count = cur.rowcount
            print(f"  ✅ DELETED {deleted_count} redundant features")
            
            # Step 5: Verify they're gone
            print("\nStep 5: Verifying deletion...")
            cur.execute("""
                SELECT feature_id, feature_name
                FROM ml_features_lut
                WHERE feature_id IN (17, 19, 20, 21)
            """)
            still_exist = cur.fetchall()
            
            if still_exist:
                print("❌ ERROR: Features still exist after deletion!")
                for feat_id, name in still_exist:
                    print(f"  - ID {feat_id}: {name}")
            else:
                print("✅ CONFIRMED: All redundant features successfully deleted")
                
        else:
            print("✅ No redundant features found - already clean!")
        
        # Step 6: Show what features DO exist
        print("\n==========================================")
        print("Current Features in ml_features_lut")
        print("==========================================")
        cur.execute("""
            SELECT feature_id, feature_name, feature_category
            FROM ml_features_lut
            WHERE feature_name IN ('variance', 'voltage', 'current', 'impedance', 'power', 'raw_data')
               OR feature_id IN (1, 2, 3, 16, 17, 18, 19, 20, 21, 22)
            ORDER BY feature_id
        """)
        current_features = cur.fetchall()
        
        for feat_id, name, category in current_features:
            cat_str = category if category else "NULL"
            driver = " [DRIVER]" if category in ['electrical', 'compute'] else ""
            print(f"  ID {feat_id:2d}: {name:20s} - {cat_str:12s}{driver}")
        
        cur.close()
        conn.close()
        
        print("\n==========================================")
        if redundant_features:
            print("✅ CLEANUP COMPLETED - Redundant features removed")
        else:
            print("✅ DATABASE CLEAN - No redundant features found")
        print("==========================================")
        
        return True
        
    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        if conn:
            conn.close()
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if conn:
            conn.close()
        return False

if __name__ == "__main__":
    success = check_and_delete()
    sys.exit(0 if success else 1)