#!/usr/bin/env python3
"""
Execute update_distance_functions_lut.sql

This script executes the SQL update to add pairwise_metric_name column
and update all 17 distance functions.
"""

import psycopg2
import sys
from pathlib import Path

def main():
    # Database connection parameters
    db_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'arc_detection',
        'user': 'kjensen'
    }

    # Read SQL file
    sql_file = Path(__file__).parent / 'update_distance_functions_lut.sql'

    if not sql_file.exists():
        print(f"❌ SQL file not found: {sql_file}")
        sys.exit(1)

    with open(sql_file, 'r') as f:
        sql_content = f.read()

    print(f"📄 Reading SQL from: {sql_file}")
    print(f"🔌 Connecting to PostgreSQL: {db_params['host']}:{db_params['port']}/{db_params['database']}")

    try:
        # Connect to database
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()

        print("✅ Connected to database")
        print("\n📊 Executing SQL update...\n")

        # Execute SQL (split by semicolons to handle multiple statements)
        statements = sql_content.split(';')

        for i, statement in enumerate(statements):
            statement = statement.strip()
            if not statement or statement.startswith('--'):
                continue

            # Execute statement
            cursor.execute(statement)

            # If it's a SELECT, fetch and display results
            if statement.upper().strip().startswith('SELECT'):
                rows = cursor.fetchall()
                colnames = [desc[0] for desc in cursor.description]

                print("=" * 120)
                print("Verification Results:")
                print("=" * 120)
                print(f"{'ID':<5} | {'Function':<20} | {'Pairwise Metric':<20} | {'Library':<30} | {'Import':<25} | Active")
                print("-" * 120)

                for row in rows:
                    # Handle None values
                    row_display = [str(v) if v is not None else 'NULL' for v in row]
                    active = '✅' if row[-1] else '❌'
                    print(f"{row_display[0]:<5} | {row_display[1]:<20} | {row_display[2]:<20} | {row_display[3]:<30} | {row_display[4]:<25} | {active}")

                print("=" * 120)
                print(f"\nTotal functions: {len(rows)}\n")

        # Commit changes
        conn.commit()
        print("✅ SQL update completed successfully")
        print("✅ Changes committed to database")

        cursor.close()
        conn.close()

        print("\n📋 Next steps:")
        print("   1. Verify in MLDP CLI:")
        print("      python -m mldp_cli.main")
        print("      connect")
        print("      show-distance-functions")

    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
