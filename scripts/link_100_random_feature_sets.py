#!/usr/bin/env python3
"""
Filename: link_100_random_feature_sets.py
Author(s): Kristophor Jensen
Date Created: 20251104_000000
Date Revised: 20251104_000000
File version: 1.0.0.1
Description: Randomly select 100 feature sets and link them to experiment 42

This script queries the database for all exp42_separability feature sets,
randomly selects 100, and links them to experiment 42 by inserting into
ml_experiments_feature_sets table.

Random Selection Strategy:
- Ensures unbiased sampling from full combinatorial space
- Reproducible with fixed random seed (42)
- Allows validation of complete feature set creation before selection

Usage:
  python3 link_100_random_feature_sets.py
"""

import psycopg2
import random
import sys

# Configuration
DB_HOST = 'localhost'
DB_NAME = 'arc_detection'
DB_USER = 'kjensen'
CATEGORY = 'exp42_separability'
EXPERIMENT_ID = 42
SAMPLE_SIZE = 100
RANDOM_SEED = 42

def main():
    """Query database, randomly select 100 feature sets, and link to experiment."""

    # Connect to database
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER
        )
        cursor = conn.cursor()
    except Exception as e:
        print(f"ERROR: Database connection failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        # Query all exp42 feature sets
        cursor.execute("""
            SELECT feature_set_id, feature_set_name
            FROM ml_feature_sets_lut
            WHERE category = %s
            ORDER BY feature_set_id
        """, (CATEGORY,))

        all_feature_sets = cursor.fetchall()

        # Validate count
        total_count = len(all_feature_sets)
        print(f"Found {total_count} feature sets with category '{CATEGORY}'")

        if total_count == 0:
            print("ERROR: No feature sets found. Run create_exp42_560_feature_sets.sh first.")
            sys.exit(1)

        if total_count < SAMPLE_SIZE:
            print(f"WARNING: Only {total_count} feature sets available, selecting all.")
            selected = all_feature_sets
        else:
            # Randomly select 100 with fixed seed for reproducibility
            random.seed(RANDOM_SEED)
            selected = random.sample(all_feature_sets, SAMPLE_SIZE)
            selected.sort(key=lambda x: x[0])  # Sort by ID for consistency

        print(f"Randomly selected {len(selected)} feature sets (seed={RANDOM_SEED})")

        # Get next experiment_feature_set_id
        cursor.execute('SELECT COALESCE(MAX(experiment_feature_set_id), 0) FROM ml_experiments_feature_sets')
        next_id = cursor.fetchone()[0]

        # Link feature sets to experiment
        linked_count = 0
        for fs_id, fs_name in selected:
            next_id += 1

            # Insert into junction table
            cursor.execute("""
                INSERT INTO ml_experiments_feature_sets (
                    experiment_feature_set_id,
                    experiment_id,
                    feature_set_id,
                    priority_order,
                    is_active
                ) VALUES (%s, %s, %s, %s, %s)
            """, (next_id, EXPERIMENT_ID, fs_id, linked_count + 1, True))

            linked_count += 1

            if linked_count % 25 == 0:
                print(f"  Progress: {linked_count}/{len(selected)} feature sets linked")

        # Commit transaction
        conn.commit()

        print(f"\n✅ Successfully linked {linked_count} feature sets to experiment {EXPERIMENT_ID}")

        # Verification
        cursor.execute("""
            SELECT COUNT(*)
            FROM ml_experiments_feature_sets
            WHERE experiment_id = %s
        """, (EXPERIMENT_ID,))

        count = cursor.fetchone()[0]
        print(f"✅ Verification: {count} feature sets linked to experiment {EXPERIMENT_ID}")

        # List selected feature set IDs
        print(f"\nSelected feature set IDs:")
        id_list = [str(fs_id) for fs_id, _ in selected]
        print(f"{', '.join(id_list)}")

    except Exception as e:
        conn.rollback()
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()
