# Marker Shape Implementation for Arcing vs Non-Arcing Segments

**Filename:** MARKER_SHAPE_IMPLEMENTATION.md
**Author(s):** Kristophor Jensen
**Date Created:** 20251120_000002
**File version:** 1.0.0.0
**Description:** Implementation guide for using different marker shapes to distinguish arcing from non-arcing segments

---

## Overview

Modify all visualization scripts to use:
- **Triangles ('^')** for arcing segments
- **Circles ('o')** for non-arcing segments

Arcing labels:
- `arc_continuous`
- `arc_initiation`
- `parallel_motor_arc_initiation`
- `parallel_motor_arc_continuous`

---

## Implementation Pattern

### 1. Add Arcing Label Definition (After imports)

```python
# Define arcing labels
ARCING_LABELS = {'arc_continuous', 'arc_initiation', 'parallel_motor_arc_initiation', 'parallel_motor_arc_continuous'}

def is_arcing_label(compound_label):
    """Check if a compound label contains an arcing segment label"""
    # Compound label format: "experiment_label.segment_label"
    if '.' in compound_label:
        segment_label = compound_label.split('.')[1]
        return segment_label in ARCING_LABELS
    return False
```

###

 2. Modified generate_2d_plot() Function

```python
def generate_2d_plot(coords, labels, label_to_color, unique_labels,
                     xlabel, ylabel, title, output_path):
    """Generate 2D scatter plot with legend using different markers for arcing/non-arcing"""
    import numpy as np

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Convert labels to numpy array for boolean indexing
    labels_array = np.array(labels)

    # Separate arcing and non-arcing points
    arcing_mask = np.array([is_arcing_label(l) for l in labels])
    non_arcing_mask = ~arcing_mask

    # Plot non-arcing segments (circles)
    if np.any(non_arcing_mask):
        non_arcing_coords = coords[non_arcing_mask]
        non_arcing_labels = labels_array[non_arcing_mask]
        non_arcing_colors = [label_to_color.get(l, (0.5, 0.5, 0.5)) for l in non_arcing_labels]
        ax.scatter(non_arcing_coords[:, 0], non_arcing_coords[:, 1],
                  c=non_arcing_colors, marker='o', s=5, alpha=0.6, label='Non-arcing')

    # Plot arcing segments (triangles)
    if np.any(arcing_mask):
        arcing_coords = coords[arcing_mask]
        arcing_labels_list = labels_array[arcing_mask]
        arcing_colors = [label_to_color.get(l, (0.5, 0.5, 0.5)) for l in arcing_labels_list]
        ax.scatter(arcing_coords[:, 0], arcing_coords[:, 1],
                  c=arcing_colors, marker='^', s=8, alpha=0.6, label='Arcing')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Create legend with both marker shapes
    legend_elements = []
    for l in unique_labels:
        if l != 'Unknown':
            marker = '^' if is_arcing_label(l) else 'o'
            markersize = 10 if marker == '^' else 8
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w',
                          markerfacecolor=label_to_color[l],
                          markersize=markersize, label=l)
            )

    ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
```

### 3. Modified generate_3d_with_projections() Function

```python
def generate_3d_with_projections(coords, labels, label_to_color, unique_labels,
                                  xlabel, ylabel, zlabel, title, output_base_path):
    """Generate 3D scatter plot with 3 viewing angles and xy, yz, xz projections"""
    import numpy as np

    # Convert labels to numpy array
    labels_array = np.array(labels)
    arcing_mask = np.array([is_arcing_label(l) for l in labels])
    non_arcing_mask = ~arcing_mask

    # Create legend elements
    legend_elements = []
    for l in unique_labels:
        if l != 'Unknown':
            marker = '^' if is_arcing_label(l) else 'o'
            markersize = 10 if marker == '^' else 8
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='w',
                          markerfacecolor=label_to_color[l],
                          markersize=markersize, label=l)
            )

    # 1. 3D scatter with 3 viewing angles
    fig = plt.figure(figsize=(18, 5))
    for i, (elev, azim, view_title) in enumerate([(30, 45, 'View 1'), (30, 135, 'View 2'), (45, 225, 'View 3')]):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')

        # Plot non-arcing (circles)
        if np.any(non_arcing_mask):
            non_arcing_coords = coords[non_arcing_mask]
            non_arcing_labels_list = labels_array[non_arcing_mask]
            non_arcing_colors = [label_to_color.get(l, (0.5, 0.5, 0.5)) for l in non_arcing_labels_list]
            ax.scatter(non_arcing_coords[:, 0], non_arcing_coords[:, 1], non_arcing_coords[:, 2],
                      c=non_arcing_colors, marker='o', s=5, alpha=0.6)

        # Plot arcing (triangles)
        if np.any(arcing_mask):
            arcing_coords = coords[arcing_mask]
            arcing_labels_list = labels_array[arcing_mask]
            arcing_colors = [label_to_color.get(l, (0.5, 0.5, 0.5)) for l in arcing_labels_list]
            ax.scatter(arcing_coords[:, 0], arcing_coords[:, 1], arcing_coords[:, 2],
                      c=arcing_colors, marker='^', s=8, alpha=0.6)

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
    if np.any(non_arcing_mask):
        non_arcing_coords = coords[non_arcing_mask]
        non_arcing_labels_list = labels_array[non_arcing_mask]
        non_arcing_colors = [label_to_color.get(l, (0.5, 0.5, 0.5)) for l in non_arcing_labels_list]
        axes[0].scatter(non_arcing_coords[:, 0], non_arcing_coords[:, 1],
                       c=non_arcing_colors, marker='o', s=5, alpha=0.6)
    if np.any(arcing_mask):
        arcing_coords = coords[arcing_mask]
        arcing_labels_list = labels_array[arcing_mask]
        arcing_colors = [label_to_color.get(l, (0.5, 0.5, 0.5)) for l in arcing_labels_list]
        axes[0].scatter(arcing_coords[:, 0], arcing_coords[:, 1],
                       c=arcing_colors, marker='^', s=8, alpha=0.6)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title('XY Projection')
    axes[0].grid(True, alpha=0.3)

    # XZ projection
    if np.any(non_arcing_mask):
        axes[1].scatter(coords[non_arcing_mask][:, 0], coords[non_arcing_mask][:, 2],
                       c=non_arcing_colors, marker='o', s=5, alpha=0.6)
    if np.any(arcing_mask):
        axes[1].scatter(coords[arcing_mask][:, 0], coords[arcing_mask][:, 2],
                       c=arcing_colors, marker='^', s=8, alpha=0.6)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(zlabel)
    axes[1].set_title('XZ Projection')
    axes[1].grid(True, alpha=0.3)

    # YZ projection
    if np.any(non_arcing_mask):
        axes[2].scatter(coords[non_arcing_mask][:, 1], coords[non_arcing_mask][:, 2],
                       c=non_arcing_colors, marker='o', s=5, alpha=0.6)
    if np.any(arcing_mask):
        axes[2].scatter(coords[arcing_mask][:, 1], coords[arcing_mask][:, 2],
                       c=arcing_colors, marker='^', s=8, alpha=0.6)
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
```

### 4. Update Function Calls

Change from:
```python
point_colors, unique_labels, label_to_color = get_label_colors(segment_ids_clean, compound_labels, GLOBAL_LABEL_TO_COLOR)
generate_2d_plot(coords, point_colors, label_to_color, unique_labels, ...)
```

To:
```python
labels, unique_labels, label_to_color = get_label_colors_and_markers(segment_ids_clean, compound_labels, GLOBAL_LABEL_TO_COLOR)
generate_2d_plot(coords, labels, label_to_color, unique_labels, ...)
```

---

## Files to Update

1. `run_feature_subset_dimreduction.py`
2. `run_dimreduction_visualization.py`
3. `run_scalar_3d_visualization.py`

---

## Visual Result

- **Circles**: Non-arcing segments (normal, no_arc, etc.)
- **Triangles**: Arcing segments (arc_continuous, arc_initiation, etc.)
- **Colors**: Consistent across all plots (per global mapping)
- **Legend**: Shows correct marker shape for each class

---

**Status:** Implementation pattern documented, ready to apply
