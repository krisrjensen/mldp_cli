#!/usr/bin/env python3
"""
Export segment metadata to CSV before regeneration

Author: Kristophor Jensen
Date: 20251005_183500
Version: 1.0.0.0
"""

import sys
import csv
import argparse
import psycopg2
from psycopg2.extras import RealDictCursor
from pathlib import Path
from datetime import datetime

def export_segment_metadata(experiment_id: int, output_dir: str = None):
    """Export segment metadata to CSV"""

    # Database config
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'arc_detection',
        'user': 'kjensen'
    }

    # Output directory
    if output_dir is None:
        output_dir = Path('/Users/kjensen/Documents/GitHub/mldp/mldp_cli/deletion_backups')
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Output file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'experiment_{experiment_id:03d}_segment_metadata_{timestamp}.csv'

    print(f"Exporting segment metadata for experiment {experiment_id:03d}")
    print(f"Output: {output_file}")
    print("=" * 60)
    print()

    # Connect to database
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Get segment metadata
        table_name = f"experiment_{experiment_id:03d}_segment_training_data"

        query = f"""
            SELECT
                st.segment_id,
                st.file_id,
                st.segment_index,
                st.segment_code_type,
                st.segment_code_number,
                st.segment_label_id,
                st.file_label_id,
                ds.beginning_index,
                ds.segment_length,
                f.binary_data_path,
                f.original_path
            FROM {table_name} st
            JOIN data_segments ds ON st.segment_id = ds.segment_id
            JOIN files f ON st.file_id = f.file_id
            WHERE st.experiment_id = %s
            ORDER BY st.segment_id
        """

        print(f"Querying {table_name}...")
        cursor.execute(query, (experiment_id,))
        segments = cursor.fetchall()

        if not segments:
            print(f"ERROR: No segments found in {table_name}")
            return 1

        print(f"Found {len(segments):,} segments")
        print()

        # Write to CSV
        print(f"Writing to CSV...")
        fieldnames = segments[0].keys()

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(segments)

        print(f"✓ Exported {len(segments):,} segments to {output_file}")
        print()

        # Show sample
        print("Sample data (first 5 rows):")
        for i, seg in enumerate(segments[:5], 1):
            print(f"{i}. Segment {seg['segment_id']}, File {seg['file_id']}, "
                  f"Label {seg['segment_label_id']}, Type {seg['segment_code_type']}")

        print()
        print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

        cursor.close()
        conn.close()

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    parser = argparse.ArgumentParser(description='Export segment metadata to CSV')
    parser.add_argument('--experiment', type=int, required=True,
                       help='Experiment ID (e.g., 41)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: mldp_cli/deletion_backups)')

    args = parser.parse_args()

    return export_segment_metadata(args.experiment, args.output_dir)

if __name__ == '__main__':
    sys.exit(main())
