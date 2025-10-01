#!/usr/bin/env python3
"""
Filename: standardize_column_names.py
Author(s): Kristophor Jensen
Date Created: 20250920_110000
Date Revised: 20250920_110000
File version: 0.0.0.1
Description: Standardize column names across all experiment file training data tables
"""

import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def standardize_file_training_tables():
    """
    Standardize column names across all experiment_NNN_file_training_data tables.

    This fixes the inconsistency where some tables use 'assigned_label' and
    others use 'file_label_name' for the same data.

    IMPORTANT: Experiment 18 is EXCLUDED from this migration as it contains
    published data that must remain unchanged for scientific reproducibility.
    """
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )
    cursor = conn.cursor()

    try:
        # Find all experiment file training data tables
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_name LIKE 'experiment_%_file_training_data'
            ORDER BY table_name
        """)

        tables = cursor.fetchall()
        logger.info(f"Found {len(tables)} file training data tables")

        for (table_name,) in tables:
            # SKIP EXPERIMENT 18 - Published data, cannot be modified
            if table_name == 'experiment_018_file_training_data':
                logger.info(f"⚠️ SKIPPING {table_name} - Published data (must remain unchanged)")
                continue
            # Check what columns this table has
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                AND column_name IN ('assigned_label', 'file_label_name')
            """, (table_name,))

            columns = cursor.fetchall()

            if not columns:
                logger.warning(f"Table {table_name} has no label column!")
                continue

            column_name = columns[0][0]

            # If table uses 'assigned_label', rename to 'file_label_name'
            if column_name == 'assigned_label':
                logger.info(f"Renaming column in {table_name}: assigned_label → file_label_name")

                cursor.execute(f"""
                    ALTER TABLE {table_name}
                    RENAME COLUMN assigned_label TO file_label_name
                """)
                conn.commit()
                logger.info(f"  ✅ Successfully renamed column in {table_name}")

            else:
                logger.info(f"  ✓ {table_name} already uses file_label_name")

        # Also check for any file_label_id columns that should exist
        logger.info("\nChecking for file_label_id columns...")
        for (table_name,) in tables:
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s
                AND column_name = 'file_label_id'
            """, (table_name,))

            if cursor.fetchone():
                logger.info(f"  ✓ {table_name} has file_label_id column")
            else:
                logger.info(f"  ⚠ {table_name} missing file_label_id column")

        logger.info("\n✅ Column standardization complete!")

    except Exception as e:
        logger.error(f"Error during migration: {e}")
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()


def verify_standardization():
    """Verify that all tables now use consistent column names"""
    conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user='kjensen'
    )
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT
                table_name,
                column_name
            FROM information_schema.columns
            WHERE table_name LIKE 'experiment_%_file_training_data'
            AND column_name IN ('assigned_label', 'file_label_name')
            ORDER BY table_name, column_name
        """)

        results = cursor.fetchall()

        print("\nVerification Results:")
        print("-" * 50)

        assigned_label_tables = []
        file_label_name_tables = []

        for table, column in results:
            if column == 'assigned_label':
                assigned_label_tables.append(table)
            else:
                file_label_name_tables.append(table)

        # Separate experiment 18 from other assigned_label tables
        exp18_table = None
        other_assigned_tables = []
        for table in assigned_label_tables:
            if table == 'experiment_018_file_training_data':
                exp18_table = table
            else:
                other_assigned_tables.append(table)

        if exp18_table:
            print(f"📌 Protected table (published data):")
            print(f"  - {exp18_table} [uses 'assigned_label' - WILL NOT BE CHANGED]")

        if other_assigned_tables:
            print(f"\n⚠️ Tables still using 'assigned_label' (need migration): {len(other_assigned_tables)}")
            for table in other_assigned_tables:
                print(f"  - {table}")

        if file_label_name_tables:
            print(f"\n✅ Tables using 'file_label_name': {len(file_label_name_tables)}")
            for table in file_label_name_tables:
                print(f"  - {table}")

        if not other_assigned_tables:
            print("\n🎉 All tables (except protected experiment 18) are now standardized!")

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("FILE TRAINING DATA COLUMN STANDARDIZATION")
    print("=" * 60)
    print("\nThis script will rename 'assigned_label' columns to 'file_label_name'")
    print("across all experiment file training data tables for consistency.")
    print("\n⚠️ IMPORTANT: Experiment 18 will be SKIPPED (published data)")
    print("   It will permanently retain 'assigned_label' for reproducibility.\n")

    response = input("Proceed with standardization? (y/n): ")

    if response.lower() == 'y':
        standardize_file_training_tables()
        verify_standardization()
    else:
        print("Standardization cancelled.")