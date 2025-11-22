#!/usr/bin/env python3
"""
Filename: run_feature_subset_dimreduction.py
Author(s): Kristophor Jensen
Date Created: 20251120_000000
Date Revised: 20251120_000004
File version: 1.0.0.4
Description: Generate dimensionality reduction visualizations for ALL C(16,8) = 12,870
             feature subset combinations from 16 scalar features.
             Applies PCA, LLE, t-SNE, and UMAP to each feature subset.

             Total analyses: 12,870 combinations × 4 methods × 24 files = 1,235,520 plots

             Uses parallel processing for efficiency.

             v1.0.0.1: Added global color mapping for consistent colors across all plots
             v1.0.0.2: Added marker shapes to distinguish arcing (triangles) from non-arcing (circles) segments
             v1.0.0.3: Added parallel_motor_arc_transient to arcing labels
             v1.0.0.4: Added advanced marker styling with fillstyle for steady state (top-filled black) and transient (right-filled white)
"""

import psycopg2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for parallel processing
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from umap import UMAP
from itertools import combinations
import time
import multiprocessing as mp
from functools import partial
import logging

# Configuration
MATRIX_DIR = Path("/Volumes/ArcData/V3_database/experiment042/classifier_files/verification_features")
OUTPUT_DIR = Path("/Volumes/ArcData/V3_database/experiment042/visualizations/feature_subset_analysis")
LOG_FILE = Path("/tmp/feature_subset_dimreduction.log")

# Parallel processing configuration
NUM_WORKERS = 8  # Number of parallel workers

# Feature subset configuration
NUM_FEATURES_PER_SUBSET = 8  # Select 8 features from 16 scalar features

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("FEATURE SUBSET DIMENSIONALITY REDUCTION ANALYSIS")
print("=" * 80)
print(f"Generating visualizations for C(16,{NUM_FEATURES_PER_SUBSET}) = 12,870 feature combinations")
print(f"Methods: PCA, LLE, t-SNE, UMAP (2D and 3D)")
print(f"Workers: {NUM_WORKERS}")
print()

# Find all verification feature files
matrix_files = sorted([
    f for f in MATRIX_DIR.glob("features_S008192_*.npy")
    if "_A02.npy" in f.name
])

print(f"Found {len(matrix_files)} verification feature files")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Log file: {LOG_FILE}")
print()

# Calculate total work
total_combinations = 12870
total_methods = 4  # PCA, LLE, t-SNE, UMAP
total_dimensions = 2  # 2D and 3D
total_files = len(matrix_files)
total_plots = total_combinations * total_methods * total_dimensions * total_files
total_plots_with_projections = total_plots + (total_combinations * total_methods * total_files)  # Add projections for 3D

print(f"Total work:")
print(f"  - Feature combinations: {total_combinations:,}")
print(f"  - Methods: {total_methods} (PCA, LLE, t-SNE, UMAP)")
print(f"  - Dimensions: {total_dimensions} (2D, 3D)")
print(f"  - Files: {total_files}")
print(f"  - Total plots: {total_plots_with_projections:,}")
print()

# Connect to database for labels (shared connection)
logger.info("Connecting to database...")
db_conn = psycopg2.connect(
    host='localhost',
    database='arc_detection',
    user='kjensen'
)
cursor = db_conn.cursor()

# Load segment labels (load once, use for all processing)
logger.info("Loading segment labels...")
cursor.execute("SELECT segment_id, segment_label_id FROM data_segments")
segment_label_ids = dict(cursor.fetchall())

cursor.execute("SELECT label_id, label_name FROM segment_labels")
label_names = dict(cursor.fetchall())

cursor.execute("""
    SELECT ds.segment_id, el.experiment_label
    FROM data_segments ds
    JOIN files_x fx ON ds.experiment_file_id = fx.file_id
    JOIN files_y fy ON fx.file_id = fy.file_id
    JOIN experiment_labels el ON fy.label_id = el.label_id
""")
file_labels = dict(cursor.fetchall())

segment_labels = {sid: label_names.get(lid, 'Unknown') for sid, lid in segment_label_ids.items()}
compound_labels = {sid: f"{file_labels.get(sid, 'Unknown')}.{segment_labels.get(sid, 'Unknown')}"
                  for sid in segment_label_ids.keys()}

logger.info(f"Loaded {len(segment_labels)} segment labels")
db_conn.close()

# Create GLOBAL color mapping for consistent colors across all plots
all_unique_labels = sorted(set(compound_labels.values()))
logger.info(f"Found {len(all_unique_labels)} unique compound labels")
logger.info("Creating global color mapping for consistent colors across all plots")

# Use tab20 colormap (20 colors) and extend if needed
num_colors_needed = len(all_unique_labels)
if num_colors_needed <= 20:
    global_colors = plt.cm.tab20(np.linspace(0, 1, 20))[:num_colors_needed]
else:
    # If more than 20 classes, use tab20 + tab20b + tab20c (60 colors total)
    colors_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))
    colors_tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))
    colors_tab20c = plt.cm.tab20c(np.linspace(0, 1, 20))
    all_colors = np.vstack([colors_tab20, colors_tab20b, colors_tab20c])
    global_colors = all_colors[:num_colors_needed]

# Create the global mapping dictionary
GLOBAL_LABEL_TO_COLOR = {label: global_colors[i] for i, label in enumerate(all_unique_labels)}

logger.info("Global color mapping created")
logger.info(f"Sample mappings:")
for label in list(all_unique_labels)[:3]:
    color = GLOBAL_LABEL_TO_COLOR[label]
    logger.info(f"  {label}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")

# Define arcing labels
ARCING_LABELS = {'arc_continuous', 'arc_initiation', 'parallel_motor_arc_initiation', 'parallel_motor_arc_continuous', 'parallel_motor_arc_transient'}

def is_arcing_label(compound_label):
    """Check if a compound label contains an arcing segment label"""
    # Compound label format: "experiment_label.segment_label"
    if '.' in compound_label:
        segment_label = compound_label.split('.')[1]
        return segment_label in ARCING_LABELS
    return False

# Define steady state labels (top half filled with black)
STEADY_STATE_LABELS = {
    'arc_continuous',
    'steady_state',
    'parallel_motor_arc_continuous',
    'steady_state_parallel_motor'
}

# Define transient labels (right half filled with white)
TRANSIENT_LABELS = {
    'arc_initiation',
    'negative_load_transient',
    'parallel_motor_arc_transient'
}

def get_label_type(compound_label):
    """Get label type (steady_state, transient, or other)"""
    if '.' in compound_label:
        segment_label = compound_label.split('.')[1]
        if segment_label in STEADY_STATE_LABELS:
            return 'steady_state'
        elif segment_label in TRANSIENT_LABELS:
            return 'transient'
    return 'other'

def get_marker_style(is_arcing, label_type, class_color):
    """Get marker, fillstyle, facecolor, edgecolor based on label characteristics"""
    marker = '^' if is_arcing else 'o'

    if label_type == 'steady_state':
        fillstyle = 'top'
        facecolor = 'black'
        edgecolor = class_color
    elif label_type == 'transient':
        fillstyle = 'right'
        facecolor = 'white'
        edgecolor = class_color
    else:  # other
        fillstyle = 'full'
        facecolor = class_color
        edgecolor = class_color

    return marker, fillstyle, facecolor, edgecolor

def plot_with_advanced_styling(ax, coords, labels, label_to_color, is_3d=False):
    """Plot points using advanced marker styling with fillstyle"""
    import numpy as np

    labels_array = np.array(labels)
    label_types = np.array([get_label_type(l) for l in labels])
    arcing_mask = np.array([is_arcing_label(l) for l in labels])

    for is_arcing in [False, True]:
        for label_type in ['other', 'steady_state', 'transient']:
            group_mask = (arcing_mask == is_arcing) & (label_types == label_type)

            if np.any(group_mask):
                coords_group = coords[group_mask]
                labels_group = labels_array[group_mask]

                for unique_label in np.unique(labels_group):
                    label_mask = labels_group == unique_label

                    if is_3d:
                        x = coords_group[label_mask, 0]
                        y = coords_group[label_mask, 1]
                        z = coords_group[label_mask, 2]
                    else:
                        x = coords_group[label_mask, 0]
                        y = coords_group[label_mask, 1]

                    class_color = label_to_color[unique_label]
                    marker, fillstyle, facecolor, edgecolor = get_marker_style(
                        is_arcing, label_type, class_color)

                    if is_3d:
                        ax.plot(x, y, z, marker=marker, fillstyle=fillstyle,
                               markerfacecolor=facecolor, markeredgecolor=edgecolor,
                               linestyle='', markersize=3, markeredgewidth=1, alpha=0.6)
                    else:
                        ax.plot(x, y, marker=marker, fillstyle=fillstyle,
                               markerfacecolor=facecolor, markeredgecolor=edgecolor,
                               linestyle='', markersize=3, markeredgewidth=1.5, alpha=0.7)

# Helper functions
def get_scalar_features(data):
    """Get feature names excluding segment_id, voltage, and current"""
    all_features = [n for n in data.dtype.names if n != 'segment_id']
    scalar_features = [n for n in all_features if n not in ['voltage', 'current']]
    return scalar_features

def prepare_feature_matrix(data, feature_names):
    """Stack features into matrix and handle NaN values"""
    feature_matrix = np.column_stack([data[f] for f in feature_names])

    nan_mask = np.isnan(feature_matrix).any(axis=1)
    valid_mask = ~nan_mask

    return feature_matrix, valid_mask

def get_label_colors_and_markers(segment_ids, compound_labels_dict, global_label_to_color):
    """Get colors and markers using GLOBAL color mapping for consistency

    Returns:
        labels: List of compound labels for each segment
        unique_labels: Sorted unique labels
        global_label_to_color: Global color mapping dictionary
    """
    labels = [compound_labels_dict.get(sid, 'Unknown') for sid in segment_ids]
    unique_labels = sorted(set(labels))
    return labels, unique_labels, global_label_to_color

def generate_2d_plot(coords, labels, label_to_color, unique_labels,
                     xlabel, ylabel, title, output_path):
    """Generate 2D scatter plot with legend using advanced marker styling"""
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot with advanced marker styling
    plot_with_advanced_styling(ax, coords, labels, label_to_color, is_3d=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Create legend with advanced marker styling
    legend_elements = []
    for l in unique_labels:
        if l != 'Unknown':
            is_arcing = is_arcing_label(l)
            label_type = get_label_type(l)
            class_color = label_to_color[l]
            marker, fillstyle, facecolor, edgecolor = get_marker_style(
                is_arcing, label_type, class_color)

            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, fillstyle=fillstyle,
                          color='w', markerfacecolor=facecolor,
                          markeredgecolor=edgecolor, markeredgewidth=1.5,
                          markersize=3, label=l)
            )

    ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_3d_with_projections(coords, labels, label_to_color, unique_labels,
                                  xlabel, ylabel, zlabel, title, output_base_path):
    """Generate 3D scatter plot with 3 viewing angles and xy, yz, xz projections"""
    import numpy as np

    # Create legend elements with advanced marker styling
    legend_elements = []
    for l in unique_labels:
        if l != 'Unknown':
            is_arcing = is_arcing_label(l)
            label_type = get_label_type(l)
            class_color = label_to_color[l]
            marker, fillstyle, facecolor, edgecolor = get_marker_style(
                is_arcing, label_type, class_color)

            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, fillstyle=fillstyle,
                          color='w', markerfacecolor=facecolor,
                          markeredgecolor=edgecolor, markeredgewidth=1.5,
                          markersize=3, label=l)
            )

    # 1. 3D scatter with 3 viewing angles
    fig = plt.figure(figsize=(18, 5))
    for i, (elev, azim, view_title) in enumerate([(30, 45, 'View 1'), (30, 135, 'View 2'), (45, 225, 'View 3')]):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        # Plot with advanced marker styling
        plot_with_advanced_styling(ax, coords, labels, label_to_color, is_3d=True)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(view_title)

        if i == 2:
            ax.legend(handles=legend_elements, loc='upper right', ncol=1, fontsize=6,
                     bbox_to_anchor=(1.15, 1.0))

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    output_path = output_base_path.parent / f'{output_base_path.name}_3d.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. xy, xz, yz projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # XY projection
    coords_xy = np.column_stack([coords[:, 0], coords[:, 1]])
    plot_with_advanced_styling(axes[0], coords_xy, labels, label_to_color, is_3d=False)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('XY Projection')
    axes[0].grid(True, alpha=0.3)

    # XZ projection
    coords_xz = np.column_stack([coords[:, 0], coords[:, 2]])
    plot_with_advanced_styling(axes[1], coords_xz, labels, label_to_color, is_3d=False)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(zlabel)
    axes[1].set_title('XZ Projection')
    axes[1].grid(True, alpha=0.3)

    # YZ projection
    coords_yz = np.column_stack([coords[:, 1], coords[:, 2]])
    plot_with_advanced_styling(axes[2], coords_yz, labels, label_to_color, is_3d=False)
    axes[2].set_xlabel(ylabel)
    axes[2].set_ylabel(zlabel)
    axes[2].set_title('YZ Projection')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(handles=legend_elements, loc='upper right', ncol=1, fontsize=6,
                  bbox_to_anchor=(1.15, 1.0))

    plt.suptitle(f'{title} - Projections', fontsize=14)
    plt.tight_layout()

    output_path_proj = output_base_path.parent / f'{output_base_path.name}_3d_projections.png'
    fig.savefig(output_path_proj, dpi=150, bbox_inches='tight')
    plt.close(fig)

def process_feature_combination(args):
    """
    Process a single feature combination with all methods

    Args:
        args: Tuple of (combo_idx, feature_combo, feature_indices, subset_matrix, segment_ids_clean,
                       labels, label_to_color, unique_labels, file_output_dir, matrix_file_stem)

    Returns:
        Tuple of (combo_idx, success, error_message)
    """
    combo_idx, feature_combo, feature_indices, subset_matrix, segment_ids_clean, \
    labels, label_to_color, unique_labels, file_output_dir, matrix_file_stem = args

    try:
        # Create combo name from feature indices (numeric IDs)
        combo_name = "_".join([str(idx) for idx in feature_indices])
        combo_dir = file_output_dir / f"combo_{combo_idx:05d}_{combo_name}"

        # subset_matrix already contains only the selected features (passed from parent)

        # 1. PCA
        # 2D
        pca_2d = PCA(n_components=2)
        coords_pca_2d = pca_2d.fit_transform(subset_matrix)
        output_path = combo_dir / 'pca_2d' / f'{matrix_file_stem}_pca_2d.png'
        generate_2d_plot(
            coords_pca_2d, labels, label_to_color, unique_labels,
            f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})',
            f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})',
            f'PCA 2D - Combo {combo_idx}: {combo_name}',
            output_path
        )

        # 3D
        if len(subset_matrix) >= 3:  # Need at least 3 samples for 3D
            pca_3d = PCA(n_components=3)
            coords_pca_3d = pca_3d.fit_transform(subset_matrix)
            output_base_path = combo_dir / 'pca_3d' / matrix_file_stem
            generate_3d_with_projections(
                coords_pca_3d, labels, label_to_color, unique_labels,
                f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
                f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
                f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})',
                f'PCA 3D - Combo {combo_idx}: {combo_name}',
                output_base_path
            )

        # 2. LLE
        # 2D
        lle_2d = LocallyLinearEmbedding(n_components=2, n_neighbors=30, random_state=42)
        coords_lle_2d = lle_2d.fit_transform(subset_matrix)
        output_path = combo_dir / 'lle_2d' / f'{matrix_file_stem}_lle_2d.png'
        generate_2d_plot(
            coords_lle_2d, labels, label_to_color, unique_labels,
            'LLE Component 1', 'LLE Component 2',
            f'LLE 2D - Combo {combo_idx}: {combo_name}',
            output_path
        )

        # 3D
        if len(subset_matrix) >= 30:  # LLE needs more neighbors
            lle_3d = LocallyLinearEmbedding(n_components=3, n_neighbors=30, random_state=42)
            coords_lle_3d = lle_3d.fit_transform(subset_matrix)
            output_base_path = combo_dir / 'lle_3d' / matrix_file_stem
            generate_3d_with_projections(
                coords_lle_3d, labels, label_to_color, unique_labels,
                'LLE Component 1', 'LLE Component 2', 'LLE Component 3',
                f'LLE 3D - Combo {combo_idx}: {combo_name}',
                output_base_path
            )

        # 3. t-SNE
        # 2D
        tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
        coords_tsne_2d = tsne_2d.fit_transform(subset_matrix)
        output_path = combo_dir / 'tsne_2d' / f'{matrix_file_stem}_tsne_2d.png'
        generate_2d_plot(
            coords_tsne_2d, labels, label_to_color, unique_labels,
            't-SNE Component 1', 't-SNE Component 2',
            f't-SNE 2D - Combo {combo_idx}: {combo_name}',
            output_path
        )

        # 3D
        tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42, max_iter=1000)
        coords_tsne_3d = tsne_3d.fit_transform(subset_matrix)
        output_base_path = combo_dir / 'tsne_3d' / matrix_file_stem
        generate_3d_with_projections(
            coords_tsne_3d, labels, label_to_color, unique_labels,
            't-SNE Component 1', 't-SNE Component 2', 't-SNE Component 3',
            f't-SNE 3D - Combo {combo_idx}: {combo_name}',
            output_base_path
        )

        # 4. UMAP
        # 2D
        umap_2d = UMAP(n_components=2, random_state=42, n_neighbors=30)
        coords_umap_2d = umap_2d.fit_transform(subset_matrix)
        output_path = combo_dir / 'umap_2d' / f'{matrix_file_stem}_umap_2d.png'
        generate_2d_plot(
            coords_umap_2d, labels, label_to_color, unique_labels,
            'UMAP Component 1', 'UMAP Component 2',
            f'UMAP 2D - Combo {combo_idx}: {combo_name}',
            output_path
        )

        # 3D
        umap_3d = UMAP(n_components=3, random_state=42, n_neighbors=30)
        coords_umap_3d = umap_3d.fit_transform(subset_matrix)
        output_base_path = combo_dir / 'umap_3d' / matrix_file_stem
        generate_3d_with_projections(
            coords_umap_3d, labels, label_to_color, unique_labels,
            'UMAP Component 1', 'UMAP Component 2', 'UMAP Component 3',
            f'UMAP 3D - Combo {combo_idx}: {combo_name}',
            output_base_path
        )

        return (combo_idx, True, None)

    except Exception as e:
        return (combo_idx, False, str(e))

def process_matrix_file(matrix_file, all_scalar_features, compound_labels_dict):
    """
    Process a single matrix file with all feature combinations

    Args:
        matrix_file: Path to the .npy file
        all_scalar_features: List of all 16 scalar feature names
        compound_labels_dict: Dictionary of compound labels for segments

    Returns:
        Number of successful combinations processed
    """
    file_start = time.time()
    logger.info(f"Processing file: {matrix_file.name}")

    # Load matrix
    data = np.load(matrix_file)
    scalar_features = get_scalar_features(data)
    segment_ids = data['segment_id']

    logger.info(f"  Segments: {len(segment_ids)}, Scalar features: {len(scalar_features)}")

    # Prepare feature matrix
    feature_matrix, valid_mask = prepare_feature_matrix(data, scalar_features)
    feature_matrix_clean = feature_matrix[valid_mask]
    segment_ids_clean = segment_ids[valid_mask]

    nan_count = np.sum(~valid_mask)
    if nan_count > 0:
        logger.info(f"  Excluded {nan_count} segments with NaN values ({100*nan_count/len(segment_ids):.1f}%)")

    # Get labels and colors for valid segments using GLOBAL color mapping
    labels, unique_labels, label_to_color = get_label_colors_and_markers(segment_ids_clean, compound_labels_dict, GLOBAL_LABEL_TO_COLOR)

    # Create output directory for this file
    file_output_dir = OUTPUT_DIR / matrix_file.stem
    file_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all C(16, NUM_FEATURES_PER_SUBSET) combinations
    feature_combinations = list(combinations(scalar_features, NUM_FEATURES_PER_SUBSET))
    logger.info(f"  Processing {len(feature_combinations)} feature combinations...")

    # Create arguments for parallel processing
    # NOTE: We need to map feature names to indices in feature_matrix_clean
    process_args = []
    for combo_idx, feature_combo in enumerate(feature_combinations, 1):
        # Get indices of selected features in scalar_features list
        feature_indices = [scalar_features.index(f) for f in feature_combo]
        subset_matrix = feature_matrix_clean[:, feature_indices]

        process_args.append((
            combo_idx,
            feature_combo,
            feature_indices,  # Add feature indices for naming
            subset_matrix,  # Pass pre-extracted subset
            segment_ids_clean,
            labels,
            label_to_color,
            unique_labels,
            file_output_dir,
            matrix_file.stem
        ))

    # Process combinations in parallel
    successful = 0
    failed = 0

    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_feature_combination, process_args)

        for combo_idx, success, error_msg in results:
            if success:
                successful += 1
                if combo_idx % 100 == 0:
                    logger.info(f"    Progress: {combo_idx}/{len(feature_combinations)} ({100*combo_idx/len(feature_combinations):.1f}%)")
            else:
                failed += 1
                logger.error(f"    Combo {combo_idx} FAILED: {error_msg}")

    file_time = time.time() - file_start
    logger.info(f"  File complete: {file_time/60:.1f} minutes")
    logger.info(f"  Successful: {successful}/{len(feature_combinations)}, Failed: {failed}")

    return successful

# Main processing loop
if __name__ == '__main__':
    overall_start = time.time()

    # Get scalar features from first file to establish the feature list
    first_data = np.load(matrix_files[0])
    all_scalar_features = get_scalar_features(first_data)

    logger.info(f"Using {len(all_scalar_features)} scalar features: {all_scalar_features}")
    logger.info(f"Generating C({len(all_scalar_features)}, {NUM_FEATURES_PER_SUBSET}) = {len(list(combinations(all_scalar_features, NUM_FEATURES_PER_SUBSET)))} combinations")
    logger.info("")

    total_successful = 0

    for file_idx, matrix_file in enumerate(matrix_files, 1):
        logger.info("=" * 80)
        logger.info(f"FILE {file_idx}/{len(matrix_files)}: {matrix_file.name}")
        logger.info("=" * 80)

        successful = process_matrix_file(matrix_file, all_scalar_features, compound_labels)
        total_successful += successful

        # Progress estimate
        if file_idx > 0:
            avg_time = (time.time() - overall_start) / file_idx
            remaining = len(matrix_files) - file_idx
            eta_hours = (avg_time * remaining) / 3600
            logger.info(f"Progress: {file_idx}/{len(matrix_files)} ({100*file_idx/len(matrix_files):.1f}%)")
            logger.info(f"ETA: {eta_hours:.1f} hours")
        logger.info("")

    overall_time = time.time() - overall_start

    logger.info("=" * 80)
    logger.info("FEATURE SUBSET DIMENSIONALITY REDUCTION ANALYSIS COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Total time: {overall_time/3600:.1f} hours")
    logger.info(f"Files processed: {len(matrix_files)}")
    logger.info(f"Total feature combinations: {total_combinations}")
    logger.info(f"Total successful analyses: {total_successful:,}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Log file: {LOG_FILE}")
