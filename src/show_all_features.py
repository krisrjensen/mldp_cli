#!/usr/bin/env python3
"""
Show ALL Features in ml_features_lut
Author: Kristophor Jensen
Date: 20250914
Description: Display complete contents of ml_features_lut table
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

def show_all_features():
    """Show all features in ml_features_lut"""
    config = get_db_config()
    
    print("==========================================")
    print("ALL Features in ml_features_lut")
    print("==========================================\n")
    
    conn = None
    try:
        conn = psycopg2.connect(**config)
        cur = conn.cursor()
        
        # Get ALL features
        cur.execute("""
            SELECT 
                feature_id, 
                feature_name, 
                feature_category,
                behavior_type,
                created_at
            FROM ml_features_lut
            ORDER BY feature_id
        """)
        
        features = cur.fetchall()
        
        print(f"Total features in table: {len(features)}\n")
        print("ID  | Feature Name         | Category     | Behavior        | Created")
        print("-" * 80)
        
        for feat_id, name, category, behavior, created in features:
            cat_str = category if category else "NULL"
            driver = " [DRV]" if category in ['electrical', 'compute'] else ""
            created_str = created.strftime("%Y-%m-%d") if created else "Unknown"
            print(f"{feat_id:3d} | {name:20s} | {cat_str:12s} | {behavior:15s} | {created_str}{driver}")
        
        print("\n" + "=" * 80)
        
        # Specifically check for the problematic IDs
        print("\nChecking for IDs 17, 19, 20, 21:")
        print("-" * 40)
        
        problem_ids = [17, 19, 20, 21]
        found_problems = False
        
        for feat_id, name, category, behavior, created in features:
            if feat_id in problem_ids:
                found_problems = True
                print(f"❌ FOUND: ID {feat_id} = {name}")
        
        if not found_problems:
            print("✅ None of the redundant feature IDs (17, 19, 20, 21) exist")
        
        # Check highest feature ID
        print(f"\n📊 Feature ID Range: 1 to {max(f[0] for f in features)}")
        
        # Check for any variance-related features
        print("\n🔍 Variance-related features:")
        print("-" * 40)
        for feat_id, name, category, behavior, created in features:
            if 'variance' in name.lower():
                print(f"  ID {feat_id}: {name}")
        
        cur.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.close()

if __name__ == "__main__":
    show_all_features()