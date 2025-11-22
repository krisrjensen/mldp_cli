# Advanced Marker Styling for State Identification

**Filename:** ADVANCED_MARKER_STYLING.md
**Author(s):** Kristophor Jensen
**Date Created:** 20251120_000004
**File version:** 1.0.0.0
**Description:** Implementation guide for advanced marker styling to identify arcing, steady state, and transient conditions

---

## Overview

Implement three-dimensional marker encoding to visually distinguish:
1. **Arcing vs Non-arcing**: Triangle vs Circle markers
2. **Steady State**: Top half filled with black
3. **Transient**: Right half filled with white

This allows identification of all binary conditions within the dataset at a glance.

---

## Label Classifications

### Steady State Labels (Top-filled with black)
- `arc_continuous`
- `steady_state`
- `parallel_motor_arc_continuous`
- `steady_state_parallel_motor`

### Transient Labels (Right-filled with white)
- `arc_initiation`
- `negative_load_transient`
- `parallel_motor_arc_transient`

### Arcing Labels (Triangle markers)
- `arc_continuous`
- `arc_initiation`
- `parallel_motor_arc_initiation`
- `parallel_motor_arc_continuous`
- `parallel_motor_arc_transient`

---

## Visual Encoding Matrix

| Condition | Marker | Fillstyle | Face Color | Edge Color |
|-----------|--------|-----------|------------|------------|
| Arcing + Steady State | Triangle | top | black | class color |
| Arcing + Transient | Triangle | right | white | class color |
| Arcing + Other | Triangle | full | class color | class color |
| Non-arcing + Steady State | Circle | top | black | class color |
| Non-arcing + Transient | Circle | right | white | class color |
| Non-arcing + Other | Circle | full | class color | class color |

---

## Implementation Details

### 1. Add Label Classification Sets

```python
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

# Define arcing labels (triangle markers)
ARCING_LABELS = {
    'arc_continuous',
    'arc_initiation',
    'parallel_motor_arc_initiation',
    'parallel_motor_arc_continuous',
    'parallel_motor_arc_transient'
}
```

### 2. Add Helper Functions

```python
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
```

### 3. Replace scatter() with plot() for 2D plots

```python
# Compute label types for all points
label_types = np.array([get_label_type(l) for l in labels])
arcing_mask = np.array([is_arcing_label(l) for l in labels])

# Plot each combination of (arcing, label_type, class)
for is_arcing in [False, True]:
    for label_type in ['other', 'steady_state', 'transient']:
        # Get mask for this group
        group_mask = (arcing_mask == is_arcing) & (label_types == label_type)

        if np.any(group_mask):
            coords_group = coords[group_mask]
            labels_group = np.array(labels)[group_mask]

            # Plot each unique class in this group
            for unique_label in np.unique(labels_group):
                label_mask = labels_group == unique_label
                x = coords_group[label_mask, 0]
                y = coords_group[label_mask, 1]

                class_color = label_to_color[unique_label]
                marker, fillstyle, facecolor, edgecolor = get_marker_style(
                    is_arcing, label_type, class_color)

                ax.plot(x, y, marker=marker, fillstyle=fillstyle,
                       markerfacecolor=facecolor, markeredgecolor=edgecolor,
                       linestyle='', markersize=6, markeredgewidth=1.5, alpha=0.7)
```

### 4. Update 3D plots similarly

For 3D plots, use `ax.plot()` instead of `ax.scatter()`:

```python
# Same grouping logic
for is_arcing in [False, True]:
    for label_type in ['other', 'steady_state', 'transient']:
        group_mask = (arcing_mask == is_arcing) & (label_types == label_type)

        if np.any(group_mask):
            coords_group = coords[group_mask]
            labels_group = np.array(labels)[group_mask]

            for unique_label in np.unique(labels_group):
                label_mask = labels_group == unique_label
                x = coords_group[label_mask, 0]
                y = coords_group[label_mask, 1]
                z = coords_group[label_mask, 2]

                class_color = label_to_color[unique_label]
                marker, fillstyle, facecolor, edgecolor = get_marker_style(
                    is_arcing, label_type, class_color)

                ax.plot(x, y, z, marker=marker, fillstyle=fillstyle,
                       markerfacecolor=facecolor, markeredgecolor=edgecolor,
                       linestyle='', markersize=5, markeredgewidth=1, alpha=0.6)
```

### 5. Update Legend

```python
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
                      markersize=10, label=l)
        )

ax.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8)
```

---

## Files to Update

1. `run_feature_subset_dimreduction.py`
2. `run_dimreduction_visualization.py`
3. `run_scalar_3d_visualization.py`

---

## Key Technical Points

1. **Use plot() not scatter()**: scatter() doesn't support fillstyle parameter
2. **Matplotlib fillstyle options**: 'full', 'left', 'right', 'top', 'bottom', 'none'
3. **Edge vs Face color**: Edge color shows class, face color/fillstyle shows state
4. **Performance**: More plot() calls, but necessary for advanced styling
5. **3D support**: plot() works in 3D with fillstyle parameter

---

## Visual Result

- **Arcing Steady State** (e.g., arc_continuous): Black-topped triangle, class-colored edge
- **Arcing Transient** (e.g., arc_initiation): White right-filled triangle, class-colored edge
- **Non-arcing Steady State** (e.g., steady_state): Black-topped circle, class-colored edge
- **Non-arcing Transient** (e.g., negative_load_transient): White right-filled circle, class-colored edge
- **Other conditions**: Fully filled with class color

This allows instant visual identification of all three binary conditions: arcing/non-arcing, steady/transient, and class membership.

---

**Status:** Implementation pattern documented, ready to apply
