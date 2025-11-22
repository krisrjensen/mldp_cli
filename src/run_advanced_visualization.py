#!/usr/bin/env python3
"""
Advanced dimensionality reduction visualization generation
Generates: t-SNE, LLE, UMAP (2D and 3D) + all C(9,3)=84 feature combinations
"""

import psycopg2
from pathlib import Path
import time
import logging
from visualize_verification_features import VerificationFeatureVisualizer
from visualize_dimreduction import DimensionalityReductionVisualizer, UMAP_AVAILABLE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configuration
MATRIX_DIR = Path("/Volumes/ArcData/V3_database/experiment042/classifier_files/verification_features")
OUTPUT_DIR = Path("/Volumes/ArcData/V3_database/experiment042/visualizations_advanced")

print("=" * 80)
print("ADVANCED DIMENSIONALITY REDUCTION VISUALIZATION")
print("=" * 80)

# Find verification feature files (only A02)
matrix_files = sorted([
    f for f in MATRIX_DIR.glob("features_S008192_*.npy")
    if "_A02.npy" in f.name
])

print(f"\nFound {len(matrix_files)} verification feature files")
print(f"Output directory: {OUTPUT_DIR}")

if not UMAP_AVAILABLE:
    print("\n[WARNING] UMAP not installed. Install with: pip install umap-learn")
    print("[WARNING] UMAP visualizations will be skipped")

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
    # Initialize base visualizer to get label map
    print("\nInitializing visualizers...")
    base_visualizer = VerificationFeatureVisualizer(
        db_conn=db_conn,
        output_dir=OUTPUT_DIR / '_temp',
        outlier_method='iqr',
        sigmoid_method='adaptive'
    )

    # Initialize advanced visualizer
    advanced_visualizer = DimensionalityReductionVisualizer(
        db_conn=db_conn,
        output_dir=OUTPUT_DIR,
        label_map=base_visualizer.label_map
    )

    print(f"Visualizers initialized")
    print(f"  Loaded {len(base_visualizer.label_map['segment_labels'])} segment labels")

    # Process each matrix file
    total_start = time.time()
    total_plots = 0

    for file_idx, matrix_file in enumerate(matrix_files, 1):
        print(f"\n{'='*80}")
        print(f"Processing file {file_idx}/{len(matrix_files)}: {matrix_file.name}")
        print(f"{'='*80}")

        file_start = time.time()

        # 1. t-SNE 2D
        print(f"\n  [1/7] Generating t-SNE 2D...")
        tsne2d_start = time.time()
        advanced_visualizer.plot_tsne(matrix_file, n_components=2, perplexity=30.0)
        print(f"  ✓ t-SNE 2D completed ({time.time() - tsne2d_start:.1f}s)")
        total_plots += 1

        # 2. t-SNE 3D
        print(f"\n  [2/7] Generating t-SNE 3D...")
        tsne3d_start = time.time()
        advanced_visualizer.plot_tsne(matrix_file, n_components=3, perplexity=30.0)
        print(f"  ✓ t-SNE 3D completed ({time.time() - tsne3d_start:.1f}s)")
        total_plots += 1

        # 3. LLE 2D
        print(f"\n  [3/7] Generating LLE 2D...")
        lle2d_start = time.time()
        advanced_visualizer.plot_lle(matrix_file, n_components=2, n_neighbors=30)
        print(f"  ✓ LLE 2D completed ({time.time() - lle2d_start:.1f}s)")
        total_plots += 1

        # 4. LLE 3D
        print(f"\n  [4/7] Generating LLE 3D...")
        lle3d_start = time.time()
        advanced_visualizer.plot_lle(matrix_file, n_components=3, n_neighbors=30)
        print(f"  ✓ LLE 3D completed ({time.time() - lle3d_start:.1f}s)")
        total_plots += 1

        # 5. UMAP 2D (if available)
        if UMAP_AVAILABLE:
            print(f"\n  [5/7] Generating UMAP 2D...")
            umap2d_start = time.time()
            advanced_visualizer.plot_umap(matrix_file, n_components=2, n_neighbors=30, min_dist=0.1)
            print(f"  ✓ UMAP 2D completed ({time.time() - umap2d_start:.1f}s)")
            total_plots += 1
        else:
            print(f"\n  [5/7] Skipping UMAP 2D (not installed)")

        # 6. UMAP 3D (if available)
        if UMAP_AVAILABLE:
            print(f"\n  [6/7] Generating UMAP 3D...")
            umap3d_start = time.time()
            advanced_visualizer.plot_umap(matrix_file, n_components=3, n_neighbors=30, min_dist=0.1)
            print(f"  ✓ UMAP 3D completed ({time.time() - umap3d_start:.1f}s)")
            total_plots += 1
        else:
            print(f"\n  [6/7] Skipping UMAP 3D (not installed)")

        # 7. Feature combinations C(9,3) = 84
        print(f"\n  [7/7] Generating feature combination plots (C(9,3) = 84)...")
        combo_start = time.time()
        combo_paths = advanced_visualizer.plot_feature_combinations(matrix_file)
        print(f"  ✓ {len(combo_paths)} feature combination plots completed ({time.time() - combo_start:.1f}s)")
        total_plots += len(combo_paths)

        file_time = time.time() - file_start
        print(f"\n  File processing complete: {file_time:.1f}s total")
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
    print("ADVANCED VISUALIZATION COMPLETE!")
    print("=" * 80)

    print(f"\nStatistics:")
    print(f"  Files processed: {len(matrix_files)}")
    print(f"  Total plots: {total_plots}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Average time per file: {total_time/len(matrix_files):.1f} seconds")

    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"  t-SNE 2D: {OUTPUT_DIR}/tsne_2d/")
    print(f"  t-SNE 3D: {OUTPUT_DIR}/tsne_3d/")
    print(f"  LLE 2D: {OUTPUT_DIR}/lle_2d/")
    print(f"  LLE 3D: {OUTPUT_DIR}/lle_3d/")
    if UMAP_AVAILABLE:
        print(f"  UMAP 2D: {OUTPUT_DIR}/umap_2d/")
        print(f"  UMAP 3D: {OUTPUT_DIR}/umap_3d/")
    print(f"  Feature combinations: {OUTPUT_DIR}/feature_combinations/")

    print(f"\nAll advanced visualizations generated successfully!")

except Exception as e:
    print(f"\n[ERROR] Advanced visualization failed: {e}")
    import traceback
    traceback.print_exc()

finally:
    db_conn.close()
    print(f"\nDatabase connection closed")
