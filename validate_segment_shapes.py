#!/usr/bin/env python3
"""
Validate segment file shapes for an experiment

Checks:
1. All .npy files can be loaded
2. All files have expected shape (N, expected_columns)
3. No corrupt files
4. Distribution of segment lengths

Author: Kristophor Jensen
Date: 20251005_183000
Version: 1.0.0.0
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def get_expected_columns(experiment_id: int) -> int:
    """
    Get expected column count for experiment

    Experiment 018: 8 amplitude methods → 18 columns (2 raw + 16 processed)
    Experiment 041: 2 amplitude methods → 6 columns (2 raw + 4 processed)
    """
    if experiment_id == 18:
        return 18
    elif experiment_id == 41:
        return 6
    else:
        # Unknown experiment, will report but not fail
        return None

def validate_segment_files(experiment_id: int, sample_size: int = None) -> Dict:
    """Validate all segment files for an experiment"""

    base_path = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}/segment_files')

    if not base_path.exists():
        return {
            'error': f'Segment path does not exist: {base_path}',
            'valid': False
        }

    print(f"Validating segment files for experiment {experiment_id:03d}")
    print(f"Path: {base_path}")
    print("=" * 60)
    print()

    # Get expected columns
    expected_cols = get_expected_columns(experiment_id)
    if expected_cols:
        print(f"Expected columns: {expected_cols}")
    else:
        print(f"Expected columns: Unknown for experiment {experiment_id}")
    print()

    # Find all .npy files
    all_files = list(base_path.glob("**/*.npy"))
    total_files = len(all_files)

    if total_files == 0:
        return {
            'error': 'No .npy files found',
            'valid': False
        }

    print(f"Total files found: {total_files:,}")

    # Sample if requested
    if sample_size and sample_size < total_files:
        import random
        files_to_check = random.sample(all_files, sample_size)
        print(f"Sampling {sample_size:,} random files for validation")
    else:
        files_to_check = all_files
        print(f"Validating all {total_files:,} files")

    print()
    print("Validating files...")
    print()

    # Validation statistics
    stats = {
        'total_checked': len(files_to_check),
        'valid': 0,
        'corrupt': 0,
        'wrong_shape': 0,
        'wrong_columns': 0,
        'shapes': defaultdict(int),
        'column_counts': defaultdict(int),
        'errors': [],
        'sample_files': []
    }

    # Validate each file
    progress_interval = max(1, len(files_to_check) // 20)

    for i, file_path in enumerate(files_to_check, 1):
        try:
            # Load file
            data = np.load(file_path)

            # Check shape
            if len(data.shape) != 2:
                stats['wrong_shape'] += 1
                stats['errors'].append(f"Wrong shape: {file_path.name} - shape {data.shape}")
                continue

            rows, cols = data.shape
            stats['shapes'][rows] += 1
            stats['column_counts'][cols] += 1

            # Check column count
            if expected_cols and cols != expected_cols:
                stats['wrong_columns'] += 1
                if len(stats['errors']) < 10:  # Limit error reporting
                    stats['errors'].append(
                        f"Wrong columns: {file_path.name} - has {cols}, expected {expected_cols}"
                    )
            else:
                stats['valid'] += 1

            # Save sample files
            if len(stats['sample_files']) < 5:
                stats['sample_files'].append({
                    'name': file_path.name,
                    'shape': data.shape,
                    'path': str(file_path)
                })

        except Exception as e:
            stats['corrupt'] += 1
            if len(stats['errors']) < 10:
                stats['errors'].append(f"Corrupt file: {file_path.name} - {str(e)}")

        # Progress
        if i % progress_interval == 0:
            print(f"  Processed {i:,}/{len(files_to_check):,} files ({100*i//len(files_to_check)}%)")

    print()
    print("=" * 60)
    print("Validation Results")
    print("=" * 60)
    print()

    # Report results
    print(f"Files checked: {stats['total_checked']:,}")
    print(f"Valid files: {stats['valid']:,} ({100*stats['valid']//stats['total_checked']}%)")
    print(f"Corrupt files: {stats['corrupt']:,}")
    print(f"Wrong shape: {stats['wrong_shape']:,}")
    print(f"Wrong columns: {stats['wrong_columns']:,}")
    print()

    # Column distribution
    print("Column count distribution:")
    for cols in sorted(stats['column_counts'].keys()):
        count = stats['column_counts'][cols]
        pct = 100 * count // stats['total_checked']
        marker = "✓" if cols == expected_cols else "✗"
        print(f"  {marker} {cols} columns: {count:,} files ({pct}%)")
    print()

    # Shape distribution (sample)
    print("Segment length distribution (top 10):")
    top_shapes = sorted(stats['shapes'].items(), key=lambda x: x[1], reverse=True)[:10]
    for rows, count in top_shapes:
        pct = 100 * count // stats['total_checked']
        print(f"  {rows:,} rows: {count:,} files ({pct}%)")
    print()

    # Sample files
    if stats['sample_files']:
        print("Sample files:")
        for sample in stats['sample_files']:
            print(f"  {sample['name']}")
            print(f"    Shape: {sample['shape']}")
        print()

    # Errors
    if stats['errors']:
        print(f"Errors (showing first {len(stats['errors'])}):")
        for error in stats['errors']:
            print(f"  {error}")
        print()

    # Overall validation
    if expected_cols:
        is_valid = (stats['valid'] == stats['total_checked'] and
                   stats['corrupt'] == 0 and
                   stats['wrong_columns'] == 0)
    else:
        is_valid = (stats['corrupt'] == 0 and stats['wrong_shape'] == 0)

    if is_valid:
        print("✓ VALIDATION PASSED")
    else:
        print("✗ VALIDATION FAILED")

    stats['is_valid'] = is_valid
    return stats

def main():
    parser = argparse.ArgumentParser(description='Validate segment file shapes')
    parser.add_argument('--experiment', type=int, required=True,
                       help='Experiment ID (e.g., 41)')
    parser.add_argument('--sample', type=int, default=None,
                       help='Number of files to sample (default: validate all)')

    args = parser.parse_args()

    stats = validate_segment_files(args.experiment, args.sample)

    if 'error' in stats:
        print(f"ERROR: {stats['error']}")
        return 1

    return 0 if stats['is_valid'] else 1

if __name__ == '__main__':
    sys.exit(main())
