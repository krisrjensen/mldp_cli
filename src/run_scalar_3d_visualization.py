#!/usr/bin/env python3
"""
Filename: run_scalar_3d_visualization.py
Author(s): Kristophor Jensen
Date Created: 20251119_000000
Date Revised: 20251120_000005
File version: 1.0.0.5
Description: Generate 3D scatter plots for scalar feature combinations
             Uses only 16 scalar features (excluding voltage and current)
             Can generate all C(16,3) = 560 combinations or selected subsets
             Generates both 3D views with legends and xy/xz/yz projections

             v1.0.0.2: Added global color mapping for consistent colors across all plots
             v1.0.0.3: Added marker shapes to distinguish arcing (triangles) from non-arcing (circles) segments
             v1.0.0.4: Added parallel_motor_arc_transient to arcing labels
             v1.0.0.5: Added advanced marker styling with fillstyle for steady state (top-filled black) and transient (right-filled white)
"""

import psycopg2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from itertools import combinations
import time
import sys

# Configuration
MATRIX_DIR = Path("/Volumes/ArcData/V3_database/experiment042/classifier_files/verification_features")
OUTPUT_DIR = Path("/Volumes/ArcData/V3_database/experiment042/visualizations")

# Mode: 'all' generates all 560 combinations, 'subset' generates selected interesting combinations
MODE = 'all'  # Generate all C(16,3) = 560 combinations

print("=" * 80)
print("3D SCALAR FEATURE COMBINATION VISUALIZATION")
print("=" * 80)

# Find all verification feature files (exclude RAW, use only A02 - z-score normalized)
matrix_files = sorted([
    f for f in MATRIX_DIR.glob("features_S008192_*.npy")
    if "_A02.npy" in f.name
])

print(f"\nFound {len(matrix_files)} verification feature files")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Mode: {MODE}")

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

def get_label_colors(segment_ids, global_label_to_color):
    """Get labels and colors using GLOBAL color mapping for consistency"""
    labels = [compound_labels.get(sid, 'Unknown') for sid in segment_ids]
    unique_labels = sorted(set(labels))
    return labels, unique_labels, global_label_to_color

# Define interesting feature combinations for subset mode
INTERESTING_COMBOS = [
    # Spectral combinations
    ('v_ultra_high_snr', 'c_ultra_high_snr', 'v_full_snr'),
    ('v_ultra_high_slope', 'c_ultra_high_slope', 'v_full_slope'),
    ('v_ultra_high_sfm', 'c_ultra_high_sfm', 'v_full_sfm'),

    # Mixed: spectral + statistical
    ('v_ultra_high_snr', 'v_kurtosis', 'v_zcr'),
    ('c_ultra_high_snr', 'c_kurtosis', 'volatility_dxdt_n1'),

    # Statistical + temporal
    ('v_kurtosis', 'c_kurtosis', 'volatility_dxdt_n1'),

    # Slope + SFM across bands
    ('v_ultra_high_slope', 'v_full_slope', 'v_ultra_high_sfm'),
    ('c_ultra_high_slope', 'c_full_slope', 'c_ultra_high_sfm'),

    # SNR variations
    ('v_ultra_high_snr', 'v_full_snr', 'c_full_snr'),
    ('c_ultra_high_snr', 'v_full_snr', 'c_full_snr'),
]

# Process all matrix files
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

    # Handle NaN values
    nan_counts = {}
    for feature in scalar_features:
        nan_count = np.sum(np.isnan(data[feature]))
        if nan_count > 0:
            nan_counts[feature] = nan_count

    if nan_counts:
        print(f"  Features with NaN values:")
        for feature, count in nan_counts.items():
            print(f"    {feature}: {count} ({100*count/len(segment_ids):.1f}%)")

    # Get labels and colors for all segments using GLOBAL color mapping
    labels_all, unique_labels, label_to_color = get_label_colors(segment_ids, GLOBAL_LABEL_TO_COLOR)

    # Convert labels to numpy array for marker-aware plotting
    labels_array_all = np.array(labels_all)

    # Generate combinations
    if MODE == 'all':
        feature_combos = list(combinations(scalar_features, 3))
        print(f"\nGenerating all {len(feature_combos)} feature combinations...")
    else:
        feature_combos = INTERESTING_COMBOS
        print(f"\nGenerating {len(feature_combos)} interesting feature combinations...")

    # Generate 3D plots for each combination
    file_start = time.time()

    for combo_idx, (f1, f2, f3) in enumerate(feature_combos, 1):
        # Extract feature values
        x = data[f1]
        y = data[f2]
        z = data[f3]

        # Find valid (non-NaN) samples for this combination
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        n_valid = np.sum(valid_mask)

        if n_valid < 100:
            print(f"  Skipping {f1}, {f2}, {f3}: only {n_valid} valid samples")
            continue

        # Filter to valid samples
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        segment_ids_valid = segment_ids[valid_mask]

        # Get labels for valid samples
        labels_valid = [labels_array_all[i] for i, v in enumerate(valid_mask) if v]
        labels_array_valid = np.array(labels_valid)

        # Create arcing/non-arcing masks
        arcing_mask = np.array([is_arcing_label(l) for l in labels_valid])
        non_arcing_mask = ~arcing_mask

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

        combo_name = f"{f1}__{f2}__{f3}"

        # 1. Create 3-panel plot with different 3D views
        fig = plt.figure(figsize=(18, 6))

        for view_idx, (elev, azim, title) in enumerate([
            (30, 45, 'View 1 (45°, 30°)'),
            (30, 135, 'View 2 (135°, 30°)'),
            (45, 225, 'View 3 (225°, 45°)')
        ]):
            ax = fig.add_subplot(1, 3, view_idx+1, projection='3d')

            # Create 3D coordinate array for plot_with_advanced_styling
            coords_3d = np.column_stack([x_valid, y_valid, z_valid])

            # Plot with advanced marker styling
            plot_with_advanced_styling(ax, coords_3d, labels_valid, label_to_color, is_3d=True)

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel(f1)
            ax.set_ylabel(f2)
            ax.set_zlabel(f3)
            ax.set_title(title)

            # Add legend to last subplot
            if view_idx == 2:
                ax.legend(handles=legend_elements, loc='upper right', ncol=1, fontsize=6,
                         bbox_to_anchor=(1.15, 1.0))

        plt.suptitle(f'3D Scatter: {f1} vs {f2} vs {f3} ({matrix_file.stem})', fontsize=12)
        plt.tight_layout()

        # Save 3D views
        output_path = OUTPUT_DIR / 'scalar_3d' / matrix_file.stem / f'{combo_name}_3d.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 2. Create xy, xz, yz projections
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # XY projection (f1 vs f2)
        coords_xy = np.column_stack([x_valid, y_valid])
        plot_with_advanced_styling(axes[0], coords_xy, labels_valid, label_to_color, is_3d=False)
        axes[0].set_xlabel(f1)
        axes[0].set_ylabel(f2)
        axes[0].set_title('XY Projection')
        axes[0].grid(True, alpha=0.3)

        # XZ projection (f1 vs f3)
        coords_xz = np.column_stack([x_valid, z_valid])
        plot_with_advanced_styling(axes[1], coords_xz, labels_valid, label_to_color, is_3d=False)
        axes[1].set_xlabel(f1)
        axes[1].set_ylabel(f3)
        axes[1].set_title('XZ Projection')
        axes[1].grid(True, alpha=0.3)

        # YZ projection (f2 vs f3)
        coords_yz = np.column_stack([y_valid, z_valid])
        plot_with_advanced_styling(axes[2], coords_yz, labels_valid, label_to_color, is_3d=False)
        axes[2].set_xlabel(f2)
        axes[2].set_ylabel(f3)
        axes[2].set_title('YZ Projection')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(handles=legend_elements, loc='upper right', ncol=1, fontsize=6,
                      bbox_to_anchor=(1.15, 1.0))

        plt.suptitle(f'Projections: {f1} vs {f2} vs {f3} ({matrix_file.stem})', fontsize=12)
        plt.tight_layout()

        # Save projections
        output_path_proj = OUTPUT_DIR / 'scalar_3d' / matrix_file.stem / f'{combo_name}_projections.png'
        fig.savefig(output_path_proj, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if combo_idx % 10 == 0:
            elapsed = time.time() - file_start
            avg_time = elapsed / combo_idx
            remaining = len(feature_combos) - combo_idx
            eta = (avg_time * remaining) / 60
            print(f"  Progress: {combo_idx}/{len(feature_combos)} ({100*combo_idx/len(feature_combos):.1f}%), ETA: {eta:.1f} min")

    # File complete
    file_time = time.time() - file_start
    print(f"\n  File complete: {file_time:.1f}s total")

    # Overall progress estimate
    if file_idx > 0:
        avg_time_per_file = (time.time() - total_start) / file_idx
        remaining_files = len(matrix_files) - file_idx
        eta_min = (avg_time_per_file * remaining_files) / 60
        print(f"  Overall progress: {file_idx}/{len(matrix_files)} ({100*file_idx/len(matrix_files):.1f}%)")
        print(f"  Overall ETA: {eta_min:.1f} minutes")

total_time = time.time() - total_start
combos_per_file = len(INTERESTING_COMBOS) if MODE == 'subset' else 560
plots_per_combo = 2  # 3D views + projections
total_plots = len(matrix_files) * combos_per_file * plots_per_combo

print("\n" + "=" * 80)
print("3D SCALAR FEATURE VISUALIZATION COMPLETE!")
print("=" * 80)
print(f"\nTotal time: {total_time/60:.1f} minutes")
print(f"Files processed: {len(matrix_files)}")
print(f"Feature combinations per file: {combos_per_file}")
print(f"Plots per combination: {plots_per_combo} (3D views + xy/xz/yz projections)")
print(f"Total plots generated: {total_plots}")
print(f"\nOutput directory: {OUTPUT_DIR / 'scalar_3d'}")

if MODE == 'subset':
    print(f"\nTo generate all {560} combinations per file (total {len(matrix_files)*560*2} plots),")
    print(f"change MODE to 'all' in the script (WARNING: will take significant time)")

db_conn.close()
