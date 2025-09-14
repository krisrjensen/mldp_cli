#!/usr/bin/env python3
"""
Consolidate N Values into ml_experiments_feature_sets
Author: Kristophor Jensen
Date: 20250914
Description: Drop ml_experiments_feature_n_values table and add n_value to ml_experiments_feature_sets
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

def consolidate_n_values():
    """Consolidate N values into ml_experiments_feature_sets"""
    config = get_db_config()
    
    print("==========================================")
    print("Consolidating N Values")
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
        
        # Step 1: Check current structure
        print("Step 1: Current Structure")
        print("-" * 40)
        
        # Check if n_value column already exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'ml_experiments_feature_sets' 
              AND column_name = 'n_value'
        """)
        
        n_value_exists = cur.fetchone() is not None
        
        if n_value_exists:
            print("  ✅ n_value column already exists in ml_experiments_feature_sets")
        else:
            print("  ⚠️  n_value column does not exist in ml_experiments_feature_sets")
        
        # Check if old table exists
        cur.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE tablename = 'ml_experiments_feature_n_values'
        """)
        
        old_table_exists = cur.fetchone() is not None
        
        if old_table_exists:
            print("  ⚠️  ml_experiments_feature_n_values table still exists")
            
            # Check if it has data
            cur.execute("SELECT COUNT(*) FROM ml_experiments_feature_n_values")
            count = cur.fetchone()[0]
            print(f"      Contains {count} rows")
            
            if count > 0:
                print("\n  📋 Current N values data:")
                cur.execute("""
                    SELECT experiment_id, feature_set_id, n_value
                    FROM ml_experiments_feature_n_values
                    ORDER BY experiment_id, feature_set_id
                """)
                n_values = cur.fetchall()
                for exp_id, fs_id, n_val in n_values:
                    print(f"      Exp {exp_id}, Set {fs_id}: N={n_val}")
        else:
            print("  ✅ ml_experiments_feature_n_values table already dropped")
        
        # Step 2: Add n_value column if needed
        if not n_value_exists:
            print("\nStep 2: Adding n_value Column")
            print("-" * 40)
            
            cur.execute("""
                ALTER TABLE ml_experiments_feature_sets 
                ADD COLUMN n_value INTEGER
            """)
            print("  ✅ Added n_value column to ml_experiments_feature_sets")
            
            # If old table exists with data, migrate it
            if old_table_exists and count > 0:
                print("\n  Migrating existing N values...")
                cur.execute("""
                    UPDATE ml_experiments_feature_sets efs
                    SET n_value = efn.n_value
                    FROM ml_experiments_feature_n_values efn
                    WHERE efs.experiment_id = efn.experiment_id
                      AND efs.feature_set_id = efn.feature_set_id
                """)
                print(f"  ✅ Migrated {cur.rowcount} N values")
        else:
            print("\nStep 2: n_value column already exists, skipping addition")
        
        # Step 3: Drop the old table
        if old_table_exists:
            print("\nStep 3: Dropping Old Table")
            print("-" * 40)
            
            cur.execute("DROP TABLE ml_experiments_feature_n_values CASCADE")
            print("  ✅ Dropped ml_experiments_feature_n_values table")
        else:
            print("\nStep 3: Old table already dropped, skipping")
        
        # Step 4: Show final structure
        print("\nStep 4: Final Structure of ml_experiments_feature_sets")
        print("-" * 40)
        
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'ml_experiments_feature_sets'
            ORDER BY ordinal_position
        """)
        
        columns = cur.fetchall()
        print("  Columns:")
        for col_name, data_type, nullable in columns:
            null_str = "NULL" if nullable == 'YES' else "NOT NULL"
            highlight = " ← NEW" if col_name == 'n_value' else ""
            print(f"    - {col_name}: {data_type} {null_str}{highlight}")
        
        # Step 5: Verify old table is gone
        print("\nStep 5: Verification")
        print("-" * 40)
        
        cur.execute("""
            SELECT tablename 
            FROM pg_tables 
            WHERE tablename = 'ml_experiments_feature_n_values'
        """)
        
        if cur.fetchone():
            print("  ❌ ml_experiments_feature_n_values table still exists!")
        else:
            print("  ✅ ml_experiments_feature_n_values table successfully removed")
        
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'ml_experiments_feature_sets' 
              AND column_name = 'n_value'
        """)
        
        if cur.fetchone():
            print("  ✅ n_value column exists in ml_experiments_feature_sets")
        else:
            print("  ❌ n_value column missing from ml_experiments_feature_sets!")
        
        # Check current data
        cur.execute("SELECT COUNT(*) FROM ml_experiments_feature_sets")
        count = cur.fetchone()[0]
        print(f"\n  Current rows in ml_experiments_feature_sets: {count}")
        
        if count > 0:
            cur.execute("""
                SELECT experiment_id, feature_set_id, data_channel, n_value
                FROM ml_experiments_feature_sets
                ORDER BY experiment_id, feature_set_id
                LIMIT 5
            """)
            rows = cur.fetchall()
            print("\n  Sample data (first 5 rows):")
            for exp_id, fs_id, channel, n_val in rows:
                n_str = f"N={n_val}" if n_val else "N=NULL"
                print(f"    Exp {exp_id}, Set {fs_id}: {channel}, {n_str}")
        
        # Commit changes
        conn.commit()
        print("\n==========================================")
        print("✅ N VALUE CONSOLIDATION COMPLETE!")
        print("==========================================")
        print("\n📋 Summary:")
        print("  - n_value column added to ml_experiments_feature_sets")
        print("  - ml_experiments_feature_n_values table dropped")
        print("  - Data migration complete (if applicable)")
        print("  - Database structure simplified")
        
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
    success = consolidate_n_values()
    sys.exit(0 if success else 1)