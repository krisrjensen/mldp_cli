#!/usr/bin/env python3
"""
Filename: run_dimreduction_visualization.py
Author(s): Kristophor Jensen
Date Created: 20251119_000000
Date Revised: 20251120_000004
File version: 1.0.0.5
Description: Generate dimensionality reduction visualizations (PCA, LLE, t-SNE, UMAP) for verification features
             Excludes voltage and current features, using only 16 scalar features
             Includes 3D plots with legends and xy, yz, xz projections

             v1.0.0.2: Added global color mapping for consistent colors across all plots
             v1.0.0.3: Added marker shapes to distinguish arcing (triangles) from non-arcing (circles) segments
             v1.0.0.4: Added parallel_motor_arc_transient to arcing labels
             v1.0.0.5: Added advanced marker styling with fillstyle for steady state (top-filled black) and transient (right-filled white)
"""

import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from umap import UMAP
from itertools import combinations
import time

# Configuration
MATRIX_DIR = Path("/Volumes/ArcData/V3_database/experiment042/classifier_files/verification_features")
OUTPUT_DIR = Path("/Volumes/ArcData/V3_database/experiment042/visualizations")

print("=" * 80)
print("DIMENSIONALITY REDUCTION VISUALIZATION GENERATION")
print("=" * 80)

# Find all verification feature files (exclude RAW, use only A02 - z-score normalized)
matrix_files = sorted([
    f for f in MATRIX_DIR.glob("features_S008192_*.npy")
    if "_A02.npy" in f.name
])

print(f"\nFound {len(matrix_files)} verification feature files")
print(f"Output directory: {OUTPUT_DIR}")

# Connect to database for labels
print("\nConnecting to database...")
db_conn = psycopg2.connect(
    host='localhost',
    database='arc_detection',
    user='kjensen'
)
cursor = db_conn.cursor()

# Load segment labels
print("Loading segment labels...")

# Get segment labels
cursor.execute("SELECT segment_id, segment_label_id FROM data_segments")
segment_label_ids = dict(cursor.fetchall())

# Get label names
cursor.execute("SELECT label_id, label_name FROM segment_labels")
label_names = dict(cursor.fetchall())

# Get file labels
cursor.execute("""
    SELECT ds.segment_id, el.experiment_label
    FROM data_segments ds
    JOIN files_x fx ON ds.experiment_file_id = fx.file_id
    JOIN files_y fy ON fx.file_id = fy.file_id
    JOIN experiment_labels el ON fy.label_id = el.label_id
""")
file_labels = dict(cursor.fetchall())

# Create compound labels
segment_labels = {sid: label_names.get(lid, 'Unknown') for sid, lid in segment_label_ids.items()}
compound_labels = {sid: f"{file_labels.get(sid, 'Unknown')}.{segment_labels.get(sid, 'Unknown')}"
                  for sid in segment_label_ids.keys()}

print(f"Loaded {len(segment_labels)} segment labels")

# Create GLOBAL color mapping for consistent colors across all plots
all_unique_labels = sorted(set(compound_labels.values()))
print(f"Found {len(all_unique_labels)} unique compound labels")
print("Creating global color mapping for consistent colors across all plots")

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

print("Global color mapping created")
print(f"Sample mappings:")
for label in list(all_unique_labels)[:3]:
    color = GLOBAL_LABEL_TO_COLOR[label]
    print(f"  {label}: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")

# Define arcing labels (triangle markers)
ARCING_LABELS = {
    'arc_continuous',
    'arc_initiation',
    'parallel_motor_arc_initiation',
    'parallel_motor_arc_continuous',
    'parallel_motor_arc_transient'
}

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

def is_arcing_label(compound_label):
    """Check if a compound label contains an arcing segment label"""
    # Compound label format: "experiment_label.segment_label"
    if '.' in compound_label:
        segment_label = compound_label.split('.')[1]
        return segment_label in ARCING_LABELS
    return False

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

# Helper function to get scalar features (exclude voltage and current)
def get_scalar_features(data):
    """Get feature names excluding segment_id, voltage, and current"""
    all_features = [n for n in data.dtype.names if n != 'segment_id']
    scalar_features = [n for n in all_features if n not in ['voltage', 'current']]
    return scalar_features

# Helper function to prepare feature matrix
def prepare_feature_matrix(data, feature_names):
    """Stack features into matrix and handle NaN values"""
    feature_matrix = np.column_stack([data[f] for f in feature_names])

    # Identify rows with NaN
    nan_mask = np.isnan(feature_matrix).any(axis=1)
    valid_mask = ~nan_mask

    if np.sum(nan_mask) > 0:
        print(f"  Found {np.sum(nan_mask)} segments with NaN values ({100*np.mean(nan_mask):.1f}%), excluding from analysis")

    return feature_matrix, valid_mask

# Helper function to generate colors for labels
def get_label_colors(segment_ids, global_label_to_color):
    """Get labels and colors using GLOBAL color mapping for consistency"""
    labels = [compound_labels.get(sid, 'Unknown') for sid in segment_ids]
    unique_labels = sorted(set(labels))
    return labels, unique_labels, global_label_to_color

def plot_with_advanced_styling(ax, coords, labels, label_to_color, is_3d=False):
    """
    Plot points using advanced marker styling with fillstyle.

    This uses plot() instead of scatter() to enable fillstyle parameter.
    Points are grouped by (arcing_status, label_type, class) for efficient plotting.

    Args:
        ax: Matplotlib axis object
        coords: Nx2 or Nx3 array of coordinates
        labels: List of labels for each point
        label_to_color: Dict mapping labels to colors
        is_3d: Boolean, True for 3D plots
    """
    import numpy as np

    # Compute label characteristics
    labels_array = np.array(labels)
    label_types = np.array([get_label_type(l) for l in labels])
    arcing_mask = np.array([is_arcing_label(l) for l in labels])

    # Plot each combination of (arcing, label_type, class)
    for is_arcing in [False, True]:
        for label_type in ['other', 'steady_state', 'transient']:
            # Get mask for this group
            group_mask = (arcing_mask == is_arcing) & (label_types == label_type)

            if np.any(group_mask):
                coords_group = coords[group_mask]
                labels_group = labels_array[group_mask]

                # Plot each unique class in this group
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

# Helper function to generate 3D plot with projections
def generate_3d_with_projections(coords, labels, label_to_color, unique_labels,
                                  xlabel, ylabel, zlabel, title, output_base_path):
    """
    Generate 3D scatter plot with 3 viewing angles and xy, yz, xz projections using advanced marker styling

    Args:
        coords: Nx3 array of coordinates
        labels: List of labels for each point
        label_to_color: Dict mapping labels to colors
        unique_labels: List of unique labels
        xlabel, ylabel, zlabel: Axis labels
        title: Plot title
        output_base_path: Path object for output (without extension)
    """
    import numpy as np

    # Create legend elements with advanced styling
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

        # Plot using advanced styling helper
        plot_with_advanced_styling(ax, coords, labels, label_to_color, is_3d=True)

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(view_title)

        # Add legend to the last subplot
        if i == 2:
            ax.legend(handles=legend_elements, loc='upper right', ncol=1, fontsize=6,
                     bbox_to_anchor=(1.15, 1.0))

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    output_path = output_base_path.parent / f'{output_base_path.name}_3d.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. xy, yz, xz projections
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # XY projection
    plot_with_advanced_styling(axes[0], coords[:, [0, 1]], labels, label_to_color, is_3d=False)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('XY Projection')
    axes[0].grid(True, alpha=0.3)

    # XZ projection
    plot_with_advanced_styling(axes[1], coords[:, [0, 2]], labels, label_to_color, is_3d=False)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(zlabel)
    axes[1].set_title('XZ Projection')
    axes[1].grid(True, alpha=0.3)

    # YZ projection
    plot_with_advanced_styling(axes[2], coords[:, [1, 2]], labels, label_to_color, is_3d=False)
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

    return output_path, output_path_proj

# Process each matrix file
total_start = time.time()

for file_idx, matrix_file in enumerate(matrix_files, 1):
    print(f"\n{'='*80}")
    print(f"Processing file {file_idx}/{len(matrix_files)}: {matrix_file.name}")
    print(f"{'='*80}")

    # Load matrix
    data = np.load(matrix_file)
    scalar_features = get_scalar_features(data)
    segment_ids = data['segment_id']

    print(f"  Segments: {len(segment_ids)}")
    print(f"  Scalar features: {len(scalar_features)}")

    # Prepare feature matrix
    feature_matrix, valid_mask = prepare_feature_matrix(data, scalar_features)
    feature_matrix_clean = feature_matrix[valid_mask]
    segment_ids_clean = segment_ids[valid_mask]

    # Get labels and colors for valid segments
    labels, unique_labels, label_to_color = get_label_colors(segment_ids_clean, GLOBAL_LABEL_TO_COLOR)

    # Convert labels to numpy array for marker-aware plotting
    labels_array = np.array(labels)
    arcing_mask = np.array([is_arcing_label(l) for l in labels])
    non_arcing_mask = ~arcing_mask

    # 1. PCA 2D and 3D
    print(f"\n  [1/3] Generating PCA visualizations...")
    pca_start = time.time()

    # PCA 2D
    pca_2d = PCA(n_components=2)
    coords_pca_2d = pca_2d.fit_transform(feature_matrix_clean)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot using advanced styling helper
    plot_with_advanced_styling(ax, coords_pca_2d, labels, label_to_color, is_3d=False)

    ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})')
    ax.set_title(f'PCA 2D: {matrix_file.stem}')
    ax.grid(True, alpha=0.3)

    # Add legend with advanced styling
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

    output_path = OUTPUT_DIR / 'dim_reduction' / 'pca_2d' / f'{matrix_file.stem}_pca_2d.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved PCA 2D: {output_path.name}")

    # PCA 3D
    pca_3d = PCA(n_components=3)
    coords_pca_3d = pca_3d.fit_transform(feature_matrix_clean)

    output_base_path = OUTPUT_DIR / 'dim_reduction' / 'pca_3d' / matrix_file.stem
    generate_3d_with_projections(
        coords_pca_3d, labels, label_to_color, unique_labels,
        f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})',
        f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})',
        f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})',
        f'PCA 3D: {matrix_file.stem}',
        output_base_path
    )

    pca_time = time.time() - pca_start
    print(f"    Saved PCA 3D and projections ({pca_time:.1f}s)")

    # 2. LLE 2D and 3D
    print(f"\n  [2/3] Generating LLE visualizations...")
    lle_start = time.time()

    # LLE 2D
    lle_2d = LocallyLinearEmbedding(n_components=2, n_neighbors=30, random_state=42)
    coords_lle_2d = lle_2d.fit_transform(feature_matrix_clean)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot using advanced styling helper
    plot_with_advanced_styling(ax, coords_lle_2d, labels, label_to_color, is_3d=False)

    ax.set_xlabel('LLE Component 1')
    ax.set_ylabel('LLE Component 2')
    ax.set_title(f'LLE 2D: {matrix_file.stem}')
    ax.grid(True, alpha=0.3)

    # Add legend with advanced styling
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

    output_path = OUTPUT_DIR / 'dim_reduction' / 'lle_2d' / f'{matrix_file.stem}_lle_2d.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved LLE 2D: {output_path.name}")

    # LLE 3D
    lle_3d = LocallyLinearEmbedding(n_components=3, n_neighbors=30, random_state=42)
    coords_lle_3d = lle_3d.fit_transform(feature_matrix_clean)

    output_base_path = OUTPUT_DIR / 'dim_reduction' / 'lle_3d' / matrix_file.stem
    generate_3d_with_projections(
        coords_lle_3d, labels, label_to_color, unique_labels,
        'LLE Component 1', 'LLE Component 2', 'LLE Component 3',
        f'LLE 3D: {matrix_file.stem}',
        output_base_path
    )

    lle_time = time.time() - lle_start
    print(f"    Saved LLE 3D and projections ({lle_time:.1f}s)")

    # 3. t-SNE 2D and 3D
    print(f"\n  [3/3] Generating t-SNE visualizations...")
    tsne_start = time.time()

    # t-SNE 2D
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords_tsne_2d = tsne_2d.fit_transform(feature_matrix_clean)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot using advanced styling helper
    plot_with_advanced_styling(ax, coords_tsne_2d, labels, label_to_color, is_3d=False)

    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title(f't-SNE 2D: {matrix_file.stem}')
    ax.grid(True, alpha=0.3)

    # Add legend with advanced styling
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

    output_path = OUTPUT_DIR / 'dim_reduction' / 'tsne_2d' / f'{matrix_file.stem}_tsne_2d.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved t-SNE 2D: {output_path.name}")

    # t-SNE 3D
    tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42, max_iter=1000)
    coords_tsne_3d = tsne_3d.fit_transform(feature_matrix_clean)

    output_base_path = OUTPUT_DIR / 'dim_reduction' / 'tsne_3d' / matrix_file.stem
    generate_3d_with_projections(
        coords_tsne_3d, labels, label_to_color, unique_labels,
        't-SNE Component 1', 't-SNE Component 2', 't-SNE Component 3',
        f't-SNE 3D: {matrix_file.stem}',
        output_base_path
    )

    tsne_time = time.time() - tsne_start
    print(f"    Saved t-SNE 3D and projections ({tsne_time:.1f}s)")

    # 4. UMAP 2D and 3D
    print(f"\n  [4/4] Generating UMAP visualizations...")
    umap_start = time.time()

    # UMAP 2D
    umap_2d = UMAP(n_components=2, random_state=42, n_neighbors=30)
    coords_umap_2d = umap_2d.fit_transform(feature_matrix_clean)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Plot using advanced styling helper
    plot_with_advanced_styling(ax, coords_umap_2d, labels, label_to_color, is_3d=False)

    ax.set_xlabel('UMAP Component 1')
    ax.set_ylabel('UMAP Component 2')
    ax.set_title(f'UMAP 2D: {matrix_file.stem}')
    ax.grid(True, alpha=0.3)

    # Add legend with advanced styling
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

    output_path = OUTPUT_DIR / 'dim_reduction' / 'umap_2d' / f'{matrix_file.stem}_umap_2d.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved UMAP 2D: {output_path.name}")

    # UMAP 3D
    umap_3d = UMAP(n_components=3, random_state=42, n_neighbors=30)
    coords_umap_3d = umap_3d.fit_transform(feature_matrix_clean)

    output_base_path = OUTPUT_DIR / 'dim_reduction' / 'umap_3d' / matrix_file.stem
    generate_3d_with_projections(
        coords_umap_3d, labels, label_to_color, unique_labels,
        'UMAP Component 1', 'UMAP Component 2', 'UMAP Component 3',
        f'UMAP 3D: {matrix_file.stem}',
        output_base_path
    )

    umap_time = time.time() - umap_start
    print(f"    Saved UMAP 3D and projections ({umap_time:.1f}s)")

    file_time = time.time() - total_start
    print(f"\n  File complete: {file_time:.1f}s total")

    # Progress estimate
    if file_idx > 0:
        avg_time = (time.time() - total_start) / file_idx
        remaining = len(matrix_files) - file_idx
        eta_min = (avg_time * remaining) / 60
        print(f"  Progress: {file_idx}/{len(matrix_files)} ({100*file_idx/len(matrix_files):.1f}%)")
        print(f"  ETA: {eta_min:.1f} minutes")

total_time = time.time() - total_start

print("\n" + "=" * 80)
print("DIMENSIONALITY REDUCTION VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\nTotal time: {total_time/60:.1f} minutes")
print(f"Files processed: {len(matrix_files)}")
print(f"Visualizations per file:")
print(f"  - 2D plots: 4 (PCA, LLE, t-SNE, UMAP)")
print(f"  - 3D plots: 4 (PCA, LLE, t-SNE, UMAP)")
print(f"  - 3D projections: 4 (PCA, LLE, t-SNE, UMAP)")
print(f"  - Total per file: 12 plots")
print(f"Total plots: {len(matrix_files) * 12}")
print(f"\nOutput directory: {OUTPUT_DIR / 'dim_reduction'}")

db_conn.close()
