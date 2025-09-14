#!/usr/bin/env python3
"""
Wipe All Feature Sets from Database
Author: Kristophor Jensen
Date: 20250914
Description: Delete all feature set data with CASCADE and ensure clean state
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

def wipe_feature_sets():
    """Wipe all feature set data"""
    config = get_db_config()
    
    print("==========================================")
    print("⚠️  WIPING ALL FEATURE SETS")
    print("==========================================")
    print(f"Database: {config['database']}")
    print(f"Host: {config['host']}:{config['port']}")
    print(f"User: {config['user']}")
    print("")
    
    conn = None
    try:
        conn = psycopg2.connect(**config)
        conn.autocommit = False
        cur = conn.cursor()
        
        # Step 1: Show current state
        print("Step 1: Current Feature Set Data")
        print("-" * 40)
        
        cur.execute("SELECT COUNT(*) FROM ml_feature_sets_lut")
        count_sets = cur.fetchone()[0]
        print(f"  Feature sets: {count_sets}")
        
        cur.execute("SELECT COUNT(*) FROM ml_feature_set_features")
        count_features = cur.fetchone()[0]
        print(f"  Feature set features: {count_features}")
        
        cur.execute("SELECT COUNT(*) FROM ml_experiments_feature_sets")
        count_exp_sets = cur.fetchone()[0]
        print(f"  Experiment feature sets: {count_exp_sets}")
        
        cur.execute("SELECT COUNT(*) FROM ml_experiments_feature_n_values")
        count_n_values = cur.fetchone()[0]
        print(f"  Feature N values: {count_n_values}")
        
        # Step 2: Delete all data with CASCADE
        print("\nStep 2: Deleting All Feature Set Data")
        print("-" * 40)
        
        # Delete from junction tables first
        cur.execute("DELETE FROM ml_experiments_feature_n_values")
        print(f"  ✅ Deleted {cur.rowcount} rows from ml_experiments_feature_n_values")
        
        cur.execute("DELETE FROM ml_experiments_feature_sets")
        print(f"  ✅ Deleted {cur.rowcount} rows from ml_experiments_feature_sets")
        
        cur.execute("DELETE FROM ml_feature_set_features")
        print(f"  ✅ Deleted {cur.rowcount} rows from ml_feature_set_features")
        
        cur.execute("DELETE FROM ml_feature_sets_lut CASCADE")
        print(f"  ✅ Deleted {cur.rowcount} rows from ml_feature_sets_lut (CASCADE)")
        
        # Step 3: Check if num_features column exists
        print("\nStep 3: Checking num_features Column")
        print("-" * 40)
        
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'ml_feature_sets_lut' 
              AND column_name = 'num_features'
        """)
        
        if cur.fetchone():
            print("  ⚠️  num_features column exists - attempting to remove...")
            try:
                cur.execute("ALTER TABLE ml_feature_sets_lut DROP COLUMN num_features CASCADE")
                print("  ✅ Dropped num_features column with CASCADE")
            except psycopg2.Error as e:
                print(f"  ❌ Error dropping column: {e}")
        else:
            print("  ✅ num_features column already removed")
        
        # Step 4: Reset sequences (find actual sequence names)
        print("\nStep 4: Resetting Sequences")
        print("-" * 40)
        
        # Get actual sequence names
        cur.execute("""
            SELECT 
                c.relname as sequence_name,
                t.relname as table_name
            FROM pg_class c
            JOIN pg_depend d ON d.objid = c.oid
            JOIN pg_class t ON d.refobjid = t.oid
            WHERE c.relkind = 'S'
              AND t.relname IN ('ml_feature_sets_lut', 'ml_feature_set_features', 
                                'ml_experiments_feature_sets', 'ml_experiments_feature_n_values')
        """)
        
        sequences = cur.fetchall()
        for seq_name, table_name in sequences:
            cur.execute(f"ALTER SEQUENCE {seq_name} RESTART WITH 1")
            print(f"  ✅ Reset sequence for {table_name}: {seq_name}")
        
        # Step 5: Verify empty state
        print("\nStep 5: Verifying Empty State")
        print("-" * 40)
        
        tables = [
            'ml_feature_sets_lut',
            'ml_feature_set_features',
            'ml_experiments_feature_sets',
            'ml_experiments_feature_n_values'
        ]
        
        all_empty = True
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            if count == 0:
                print(f"  ✅ {table}: EMPTY")
            else:
                print(f"  ❌ {table}: Still has {count} rows")
                all_empty = False
        
        # Step 6: Show table structure
        print("\nStep 6: ml_feature_sets_lut Structure")
        print("-" * 40)
        
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'ml_feature_sets_lut'
            ORDER BY ordinal_position
        """)
        
        columns = cur.fetchall()
        print("  Columns:")
        for col_name, data_type, nullable in columns:
            null_str = "NULL" if nullable == 'YES' else "NOT NULL"
            print(f"    - {col_name}: {data_type} {null_str}")
        
        # Commit changes
        if all_empty:
            conn.commit()
            print("\n==========================================")
            print("✅ ALL FEATURE SETS WIPED SUCCESSFULLY!")
            print("==========================================")
            print("\n📋 Summary:")
            print("  - All feature set data deleted")
            print("  - All sequences reset to 1")
            print("  - num_features column verified removed")
            print("  - Database ready for fresh feature sets")
        else:
            conn.rollback()
            print("\n❌ Some tables not empty - rolling back")
        
        cur.close()
        conn.close()
        return all_empty
        
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
    success = wipe_feature_sets()
    sys.exit(0 if success else 1)