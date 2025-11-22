# Color Consistency Fix for Visualizations

**Filename:** COLOR_CONSISTENCY_FIX.md
**Author(s):** Kristophor Jensen
**Date Created:** 20251120_000001
**File version:** 1.0.0.0
**Description:** Fix for ensuring consistent class colors across all visualization scripts

---

## Problem

Currently, each visualization script independently assigns colors to compound labels based on the unique labels present in each specific dataset. This causes the same class to have different colors in different plots, making comparison and analysis difficult.

**Example:**
- Plot A: "C2H4.arc" → Blue
- Plot B: "C2H4.arc" → Red  ← INCONSISTENT!

---

## Solution

Create a **GLOBAL color mapping** at the start of each script that maps ALL possible compound labels to consistent colors. Use this mapping throughout all visualizations.

---

## Implementation Steps

### 1. Create Global Color Mapping (After loading segment labels)

```python
# After creating compound_labels dictionary:

# Create GLOBAL color mapping for all unique compound labels
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
```

### 2. Modify get_label_colors() Function

**OLD (Inconsistent):**
```python
def get_label_colors(segment_ids, compound_labels_dict):
    labels = [compound_labels_dict.get(sid, 'Unknown') for sid in segment_ids]
    unique_labels = sorted(set(labels))  # ← PROBLEM: Different for each plot!
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
    point_colors = [label_to_color[l] for l in labels]
    return point_colors, unique_labels, label_to_color
```

**NEW (Consistent):**
```python
def get_label_colors(segment_ids, compound_labels_dict, global_label_to_color):
    """Get colors using GLOBAL color mapping for consistency"""
    labels = [compound_labels_dict.get(sid, 'Unknown') for sid in segment_ids]
    unique_labels = sorted(set(labels))
    point_colors = [global_label_to_color.get(l, (0.5, 0.5, 0.5)) for l in labels]
    return point_colors, unique_labels, global_label_to_color
```

### 3. Update All Function Calls

**OLD:**
```python
point_colors, unique_labels, label_to_color = get_label_colors(segment_ids_clean, compound_labels)
```

**NEW:**
```python
point_colors, unique_labels, label_to_color = get_label_colors(segment_ids_clean, compound_labels, GLOBAL_LABEL_TO_COLOR)
```

---

## Files to Update

1. `run_feature_subset_dimreduction.py` - Feature subset analysis
2. `run_dimreduction_visualization.py` - Standard dimensionality reduction
3. `run_scalar_3d_visualization.py` - Scalar 3D combinations
4. `visualize_verification_features.py` - Production PDFs (optional for scatter plots)

---

## Benefits

1. **Scientific Accuracy**: Same class always has same color
2. **Easy Comparison**: Can compare plots side-by-side
3. **Professional**: Consistent visualization standard
4. **Reproducible**: Color assignment is deterministic (sorted labels)

---

## Testing

After applying fix:
1. Generate plots from different files
2. Verify "C2H4.arc" (or any class) has same color across all plots
3. Check legend consistency

---

## Example Output

```
Found 8 unique compound labels
Creating global color mapping for consistent colors across all plots
Global color mapping created
Sample mappings:
  C2H2.arc: RGB(0.12, 0.47, 0.71)
  C2H4.arc: RGB(1.00, 0.50, 0.05)
  C2H6.arc: RGB(0.17, 0.63, 0.17)
```

Now "C2H4.arc" will ALWAYS be RGB(1.00, 0.50, 0.05) in EVERY plot!

---

**Status:** Ready for implementation
