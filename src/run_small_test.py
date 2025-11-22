#!/usr/bin/env python3
"""
Small-scale test of visualization with real data
Test: 1 file × 1 feature
"""

import psycopg2
from pathlib import Path
from visualize_verification_features import VerificationFeatureVisualizer

# Configuration
MATRIX_FILE = Path("/Volumes/ArcData/V3_database/experiment042/classifier_files/verification_features/features_S008192_D000000_R008192_ADC8_A02.npy")
OUTPUT_DIR = Path("/Volumes/ArcData/V3_database/experiment042/visualizations_test")
TEST_FEATURE = "voltage"  # Test with just one feature

print("=" * 60)
print("SMALL-SCALE VISUALIZATION TEST")
print("=" * 60)
print(f"Matrix file: {MATRIX_FILE.name}")
print(f"Test feature: {TEST_FEATURE}")
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

    # Generate 3D scatter plot
    print("\nGenerating 3D scatter plot...")
    scatter_output = visualizer.plot_3d_scatter_multiversion(MATRIX_FILE)
    print(f"✓ 3D scatter plot saved: {scatter_output.name}")

    # Generate PDF plots for voltage feature at all grouping levels
    print(f"\nGenerating PDF plots for '{TEST_FEATURE}' feature...")
    grouping_levels = ['population', 'file', 'segment', 'file_segment']

    for grouping in grouping_levels:
        pdf_output = visualizer.plot_pdf_multiversion(
            matrix_file=MATRIX_FILE,
            feature_name=TEST_FEATURE,
            grouping_level=grouping
        )
        print(f"✓ {grouping:15s}: {pdf_output.name}")

    print("\n" + "=" * 60)
    print("SMALL-SCALE TEST COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - 1 3D scatter plot (18 subplots)")
    print(f"  - 4 PDF plots (4 grouping levels)")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"\nReview the plots to verify visual quality.")

finally:
    db_conn.close()
    print("\nDatabase connection closed")
