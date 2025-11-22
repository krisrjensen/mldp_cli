#!/usr/bin/env python3
"""
Production-scale visualization generation for all verification features
Processes all 24 verification feature matrix files
"""

import psycopg2
from pathlib import Path
import time
from visualize_verification_features import VerificationFeatureVisualizer

# Configuration
MATRIX_DIR = Path("/Volumes/ArcData/V3_database/experiment042/classifier_files/verification_features")
OUTPUT_DIR = Path("/Volumes/ArcData/V3_database/experiment042/visualizations")

print("=" * 80)
print("PRODUCTION VISUALIZATION GENERATION")
print("=" * 80)

# Find all verification feature files (exclude RAW and A01 files, use only A02)
matrix_files = sorted([
    f for f in MATRIX_DIR.glob("features_S008192_*.npy")
    if "_A02.npy" in f.name  # Only A02 (z-score normalized)
])

print(f"\nFound {len(matrix_files)} verification feature files")
print(f"Output directory: {OUTPUT_DIR}")

# Connect to database
print("\nConnecting to database...")
db_conn = psycopg2.connect(
    host='localhost',
    database='arc_detection',
    user='kjensen'
)
print("Connected successfully")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    # Initialize visualizer
    print("\nInitializing visualizer...")
    visualizer = VerificationFeatureVisualizer(
        db_conn=db_conn,
        output_dir=OUTPUT_DIR,
        outlier_method='iqr',
        sigmoid_method='adaptive'
    )
    print(f"Visualizer initialized")
    print(f"  Loaded {len(visualizer.label_map['segment_labels'])} segment labels")

    # Process each matrix file
    total_start = time.time()
    total_3d_plots = 0
    total_pdf_plots = 0

    for file_idx, matrix_file in enumerate(matrix_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing file {file_idx}/{len(matrix_files)}: {matrix_file.name}")
        print(f"{'='*80}")

        file_start = time.time()

        # Load matrix to get feature names
        import numpy as np
        matrix = np.load(matrix_file)
        feature_names = [n for n in matrix.dtype.names if n != 'segment_id']
        n_segments = len(matrix)

        print(f"  Segments: {n_segments}")
        print(f"  Features: {len(feature_names)}")
        print(f"  Features: {', '.join(feature_names)}")

        # Generate 3D scatter plot
        print(f"\n  [1/2] Generating 3D scatter plot...")
        scatter_start = time.time()
        scatter_output = visualizer.plot_3d_scatter_multiversion(matrix_file)
        scatter_time = time.time() - scatter_start
        total_3d_plots += 1
        print(f"  ✓ 3D scatter saved ({scatter_time:.1f}s): {scatter_output.name}")

        # Generate PDF plots for all features at all grouping levels
        print(f"\n  [2/2] Generating PDF plots...")
        grouping_levels = ['population', 'file', 'segment', 'file_segment']

        pdf_start = time.time()
        pdf_count = 0

        for feature_name in feature_names:
            for grouping in grouping_levels:
                try:
                    pdf_output = visualizer.plot_pdf_multiversion(
                        matrix_file=matrix_file,
                        feature_name=feature_name,
                        grouping_level=grouping
                    )
                    pdf_count += 1
                    total_pdf_plots += 1

                    if pdf_count % 9 == 0:  # Report every 9 PDFs (one feature complete)
                        print(f"    Generated {pdf_count}/{len(feature_names)*len(grouping_levels)} PDFs...")

                except Exception as e:
                    print(f"    [WARNING] Failed {feature_name}/{grouping}: {e}")

        pdf_time = time.time() - pdf_start
        file_time = time.time() - file_start

        print(f"  ✓ {pdf_count} PDF plots generated ({pdf_time:.1f}s)")
        print(f"  File processing complete: {file_time:.1f}s total")
        print(f"  Progress: {file_idx}/{len(matrix_files)} files ({100*file_idx/len(matrix_files):.1f}%)")

        # Time estimate
        if file_idx > 0:
            avg_time_per_file = (time.time() - total_start) / file_idx
            remaining_files = len(matrix_files) - file_idx
            eta_seconds = avg_time_per_file * remaining_files
            eta_minutes = eta_seconds / 60
            print(f"  ETA: {eta_minutes:.1f} minutes remaining")

    total_time = time.time() - total_start

    print("\n" + "=" * 80)
    print("PRODUCTION VISUALIZATION COMPLETE!")
    print("=" * 80)

    print(f"\nStatistics:")
    print(f"  Files processed: {len(matrix_files)}")
    print(f"  3D scatter plots: {total_3d_plots}")
    print(f"  PDF plots: {total_pdf_plots}")
    print(f"  Total plots: {total_3d_plots + total_pdf_plots}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Average time per file: {total_time/len(matrix_files):.1f} seconds")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  3D plots: {OUTPUT_DIR}/3d_scatter/")
    print(f"  PDF plots: {OUTPUT_DIR}/pdfs/")

    print(f"\nAll visualizations generated successfully!")

except Exception as e:
    print(f"\n[ERROR] Production visualization failed: {e}")
    import traceback
    traceback.print_exc()

finally:
    db_conn.close()
    print(f"\nDatabase connection closed")
