#!/usr/bin/env python3
"""
Check all views that depend on ml_feature_sets_lut
Author: Kristophor Jensen
Date: 20250914
Description: Find all views that depend on the num_features column
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

def check_views():
    """Check all views that reference ml_feature_sets_lut"""
    config = get_db_config()
    
    conn = None
    try:
        conn = psycopg2.connect(**config)
        cur = conn.cursor()
        
        print("==========================================")
        print("Checking View Dependencies")
        print("==========================================\n")
        
        # Find all views that reference ml_feature_sets_lut
        cur.execute("""
            SELECT DISTINCT 
                v.viewname,
                v.viewowner,
                pg_get_viewdef(c.oid, true) as definition
            FROM pg_views v
            JOIN pg_class c ON c.relname = v.viewname
            WHERE v.schemaname = 'public'
              AND pg_get_viewdef(c.oid, true) LIKE '%ml_feature_sets_lut%'
            ORDER BY v.viewname
        """)
        
        views = cur.fetchall()
        
        if views:
            print(f"Found {len(views)} views referencing ml_feature_sets_lut:\n")
            for viewname, owner, definition in views:
                print(f"View: {viewname}")
                print(f"Owner: {owner}")
                if 'num_features' in definition:
                    print("⚠️  Contains num_features column reference")
                else:
                    print("✅ No num_features reference")
                print("-" * 40)
                print(definition[:500] + "..." if len(definition) > 500 else definition)
                print("=" * 40 + "\n")
        else:
            print("No views found referencing ml_feature_sets_lut")
        
        # Check if experiment_041_feature_parameters exists (without v_ prefix)
        cur.execute("""
            SELECT viewname 
            FROM pg_views 
            WHERE viewname IN ('experiment_041_feature_parameters', 
                             'v_experiment_041_feature_parameters',
                             'v_feature_sets_detail')
        """)
        existing_views = cur.fetchall()
        
        print("\nViews to handle:")
        for view in existing_views:
            print(f"  - {view[0]}")
        
        cur.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.close()

if __name__ == "__main__":
    check_views()