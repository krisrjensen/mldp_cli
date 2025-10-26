#!/usr/bin/env python3
"""
Filename: recover_from_sqlite.py
Author: Kristophor Jensen
Date Created: 20251009_120000
Description: Recover experiment_018 distance tables from SQLite backup to PostgreSQL
"""

import sqlite3
import psycopg2
from psycopg2.extras import execute_batch
import sys

# Configuration
SQLITE_BACKUP = "/Users/kjensen/Library/CloudStorage/GoogleDrive-kris.r.jensen@gmail.com/My Drive/V3_database/20251005_131837/experiment_018.db_backup.db"
PG_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'arc_detection',
    'user': 'kjensen'
}

TABLES = [
    'experiment_018_distance_l1',
    'experiment_018_distance_l2',
    'experiment_018_distance_cosine',
    'experiment_018_distance_pearson'
]

BATCH_SIZE = 10000

def recover_table(sqlite_conn, pg_conn, table_name):
    """Recover a single table from SQLite to PostgreSQL"""

    print(f"\n📊 Processing {table_name}...")

    # Get total row count from SQLite (case-insensitive)
    sqlite_cursor = sqlite_conn.cursor()

    # Try both cases for table name
    for try_name in [table_name, table_name.replace('_l1', '_L1').replace('_l2', '_L2')]:
        try:
            sqlite_cursor.execute(f"SELECT COUNT(*) FROM {try_name}")
            total_rows = sqlite_cursor.fetchone()[0]
            sqlite_table_name = try_name
            break
        except sqlite3.OperationalError:
            continue
    else:
        print(f"   ❌ Table not found in SQLite backup")
        return

    if total_rows == 0:
        print(f"   ⏭️  Skipping - no data in backup")
        return

    print(f"   📈 Found {total_rows:,} records in backup")

    # Get column names from SQLite
    sqlite_cursor.execute(f"PRAGMA table_info({sqlite_table_name})")
    columns_info = sqlite_cursor.fetchall()
    column_names = [col[1] for col in columns_info]

    print(f"   📋 Columns: {', '.join(column_names)}")

    # Truncate PostgreSQL table
    print(f"   🗑️  Truncating PostgreSQL table...")
    pg_cursor = pg_conn.cursor()
    pg_cursor.execute(f"TRUNCATE TABLE {table_name} CASCADE")
    pg_conn.commit()

    # Prepare INSERT statement
    placeholders = ', '.join(['%s'] * len(column_names))
    insert_sql = f"INSERT INTO {table_name} ({', '.join(column_names)}) VALUES ({placeholders})"

    # Read and insert in batches
    print(f"   🔄 Transferring data...")
    sqlite_cursor.execute(f"SELECT * FROM {sqlite_table_name}")

    inserted = 0
    batch = []

    while True:
        rows = sqlite_cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break

        batch.extend(rows)

        if len(batch) >= BATCH_SIZE:
            execute_batch(pg_cursor, insert_sql, batch, page_size=BATCH_SIZE)
            pg_conn.commit()
            inserted += len(batch)
            print(f"      Inserted {inserted:,} / {total_rows:,} ({100*inserted/total_rows:.1f}%)")
            batch = []

    # Insert remaining rows
    if batch:
        execute_batch(pg_cursor, insert_sql, batch, page_size=len(batch))
        pg_conn.commit()
        inserted += len(batch)

    print(f"   ✅ Completed: {inserted:,} records transferred")

    # Verify
    pg_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    pg_count = pg_cursor.fetchone()[0]

    if pg_count == total_rows:
        print(f"   ✅ Verification passed: {pg_count:,} records")
    else:
        print(f"   ⚠️  Warning: Expected {total_rows:,} but found {pg_count:,}")

def main():
    print("🔄 Starting recovery from SQLite backup...")
    print(f"📁 Source: {SQLITE_BACKUP}")
    print(f"🎯 Target: PostgreSQL at {PG_CONFIG['host']}:{PG_CONFIG['port']}/{PG_CONFIG['database']}")
    print()

    # Connect to SQLite
    print("📦 Connecting to SQLite backup...")
    try:
        sqlite_conn = sqlite3.connect(SQLITE_BACKUP)
        print("   ✅ Connected to SQLite")
    except Exception as e:
        print(f"   ❌ Failed to connect to SQLite: {e}")
        sys.exit(1)

    # Connect to PostgreSQL
    print("🐘 Connecting to PostgreSQL...")
    try:
        pg_conn = psycopg2.connect(**PG_CONFIG)
        print("   ✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"   ❌ Failed to connect to PostgreSQL: {e}")
        sys.exit(1)

    # Recover each table
    for table in TABLES:
        try:
            recover_table(sqlite_conn, pg_conn, table)
        except Exception as e:
            print(f"   ❌ Error recovering {table}: {e}")
            import traceback
            traceback.print_exc()

    # Cleanup
    sqlite_conn.close()
    pg_conn.close()

    print()
    print("✅ Recovery complete!")
    print()
    print("📊 Verification query:")
    print("""
psql -h localhost -p 5432 -d arc_detection -c "
SELECT 'L1' as metric, COUNT(*) as records FROM experiment_018_distance_l1
UNION ALL SELECT 'L2', COUNT(*) FROM experiment_018_distance_l2
UNION ALL SELECT 'cosine', COUNT(*) FROM experiment_018_distance_cosine
UNION ALL SELECT 'pearson', COUNT(*) FROM experiment_018_distance_pearson"
""")

if __name__ == '__main__':
    main()
