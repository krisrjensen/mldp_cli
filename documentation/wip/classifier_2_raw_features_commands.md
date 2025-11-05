# Classifier 2: Raw Features (No Distance Calculation)

**Filename:** classifier_2_raw_features_commands.md
**Author(s):** Kristophor Jensen
**Date Created:** 20251027_000000
**Description:** Commands to create classifier 2 using raw features instead of distance-based features

---

## Overview

### Classifier 1 (Existing)
- **Type:** Distance-based SVM
- **Features:** Distances to reference segments (4 distance metrics × 13 classes = 52 features)
- **Feature Builder:**
  - `include_original_feature = false`
  - `compute_baseline_distances_inter = true` (distances to OTHER classes)
  - `compute_baseline_distances_intra = true` (distances to SAME class)

### Classifier 2 (New)
- **Type:** Raw feature-based SVM
- **Features:** Original feature values from feature sets 1, 2, 5
- **Feature Builder:**
  - `include_original_feature = true` ← Use raw features
  - `compute_baseline_distances_inter = false` ← NO distance calculation
  - `compute_baseline_distances_intra = false` ← NO distance calculation

---

## Available Feature Sets

From experiment 41:

| ID | Feature Set Name | Description | Features |
|----|------------------|-------------|----------|
| 1 | current_only | Features: current | Current channel only |
| 2 | voltage_only | Features: voltage | Voltage channel only |
| 5 | all_electrical | Features: current, voltage, impedance, power | All electrical features |

**Note:** These are experiment_feature_set_ids, which map to global feature_set_ids.

---

## Complete Command Sequence

### Step 1: Create Classifier 2

```bash
./mldp
connect
set experiment 41
classifier-new --name raw_features_svm --description "SVM using raw features (no distances)" --type svm
```

**Expected output:**
```
[SUCCESS] Created classifier 'raw_features_svm' (ID: 2) for experiment 41
[INFO] Global classifier ID: 2
```

**What this does:**
- Creates a new classifier entry in `ml_experiment_classifiers`
- Assigns classifier_id = 2 for experiment 41
- Creates global_classifier_id = 2

---

### Step 2: Select Classifier 2

```bash
set classifier 2
```

**Expected output:**
```
[INFO] Selected classifier 2 for experiment 41
```

**What this does:**
- Sets `self.current_classifier_id = 2`
- All subsequent commands will operate on classifier 2

---

### Step 3: Create Configuration

```bash
classifier-config-create --name raw_baseline --active
```

**Expected output:**
```
[SUCCESS] Created configuration 'raw_baseline' (config_id: 2)
[INFO] Configuration is now active
```

**What this does:**
- Creates entry in `ml_classifier_configs` with config_id = 2
- Links to experiment 41, classifier 2
- Sets as active configuration

---

### Step 4: Add Feature Sets to Configuration

```bash
classifier-config-add-feature-sets --config-id 2 --experiment-feature-sets 1 2 5
```

**Expected output:**
```
[SUCCESS] Added feature sets to configuration 2:
  - Experiment feature set 1 (current_only)
  - Experiment feature set 2 (voltage_only)
  - Experiment feature set 5 (all_electrical)
```

**What this does:**
- Links feature sets 1, 2, 5 to config_id 2
- Inserts into `ml_classifier_config_experiment_feature_sets`

---

### Step 5: Set Feature Builder (CRITICAL STEP)

```bash
classifier-config-set-feature-builder --config-id 2 --include-original --no-compute-distances-inter --no-compute-distances-intra
```

**Expected output:**
```
[SUCCESS] Created feature builder for config 'raw_baseline' (ID: 2)

Current Feature Builder Settings:
  Include original features: True  ← USE RAW FEATURES
  Compute inter-class baseline distances: False  ← NO DISTANCES
  Compute intra-class baseline distances: False  ← NO DISTANCES
  Statistical features: False (reserved)
  External function: False (reserved)
```

**What this does:**
- Creates entry in `ml_classifier_feature_builder`
- **Enables raw feature usage**
- **Disables distance calculations**
- This is what makes classifier 2 different from classifier 1!

---

### Step 6: Add Distance Functions (Required Even If Not Computing Distances)

**Note:** Even though we're not computing distances, the distance functions must be configured (system requirement).

```bash
classifier-config-add-distance-functions --config-id 2 --distance-functions 1 2 3 4
```

**Expected output:**
```
[SUCCESS] Added distance functions to configuration 2:
  - Distance function 1 (L1 / Manhattan)
  - Distance function 2 (L2 / Euclidean)
  - Distance function 3 (Cosine)
  - Distance function 4 (Pearson / Correlation)
```

**What this does:**
- Links distance functions to config (even though not used)
- Required for schema consistency

---

### Step 7: Add SVM Hyperparameters

```bash
classifier-config-add-hyperparameters --config-id 2 --kernel linear --C 0.1 1.0 10.0 100.0
```

**Expected output:**
```
[SUCCESS] Added hyperparameters to configuration 2
  SVM Kernel: linear
  C values: 0.1, 1.0, 10.0, 100.0
```

**What this does:**
- Creates entries in `ml_classifier_config_hyperparameters`
- Defines SVM grid search space
- Will train 4 models (one for each C value)

**Optional: Add more kernels**
```bash
# Add RBF kernel with different C and gamma values
classifier-config-add-hyperparameters --config-id 2 --kernel rbf --C 1.0 10.0 --gamma 0.001 0.01 0.1
```

---

### Step 8: Create Data Splits Table

```bash
classifier-create-splits-table
```

**Expected output:**
```
[SUCCESS] Created table: experiment_041_classifier_002_data_splits
```

**What this does:**
- Creates `experiment_041_classifier_002_data_splits` table
- Will store train/test/verification split assignments

---

### Step 9: Assign Data Splits

```bash
classifier-assign-splits --strategy stratified --split-ratios 0.33 0.33 0.34
```

**Expected output:**
```
[INFO] Using stratified split strategy
[INFO] Split ratios: Train=0.33, Test=0.33, Verification=0.34
[SUCCESS] Split assignments completed:
  Training: 11,620 segments
  Test: 11,620 segments
  Verification: 11,620 segments
```

**What this does:**
- Assigns segments to train/test/verification splits
- Uses stratified sampling (balanced across classes)
- Inserts into `experiment_041_classifier_002_data_splits`

**Optional: Check splits**
```bash
classifier-show-splits
```

---

### Step 10: Build Raw Feature Vectors

**Important:** Because we're using `--include-original`, the feature building process will:
1. Load raw features from experiment_041_feature_fileset
2. Concatenate them into feature vectors
3. **NOT compute distances** (because `compute_distances_inter/intra = false`)

```bash
classifier-build-features --amplitude-method 2
```

**Expected output:**
```
[INFO] Building features for classifier 2, config 2
[INFO] Feature builder settings:
  - Include original features: True  ← USING RAW FEATURES
  - Compute inter-class distances: False  ← NOT COMPUTING
  - Compute intra-class distances: False  ← NOT COMPUTING

[INFO] Processing configurations:
  - Decimation factors: 0, 255, 511, 1023, 2047, 4095, 8191
  - Data types: 6, 8, 10, 12
  - Amplitude methods: 2
  - Feature sets: 1, 2, 5

[INFO] Total configurations: 7 × 4 × 1 × 3 = 84

[INFO] Starting feature building with 12 workers...
[PROGRESS] Processed 10000 / 104580 segments...
...
[SUCCESS] Feature building completed
  - Total segments: 104,580
  - Features created: 104,580
  - Time: XX minutes
```

**What this does:**
- Creates `experiment_041_classifier_002_raw_features` table
- Loads raw features and concatenates into vectors
- **Does NOT compute distances** (key difference from classifier 1)
- Feature vector dimensions depend on feature sets used

**Feature Vector Dimensions:**
- Current_only (FS 1): ~N features
- Voltage_only (FS 2): ~N features
- All_electrical (FS 5): ~N features
- Total: Concatenation of all selected features

---

### Step 11: Initialize SVM Training

```bash
classifier-train-svm-init --amplitude-method 2
```

**Expected output:**
```
[INFO] Initializing SVM training for classifier 2
[INFO] Creating SVM models table...
[SUCCESS] Created table: experiment_041_classifier_002_svm_models

[INFO] Hyperparameter grid:
  - Decimation factors: 7
  - Data types: 4
  - Amplitude methods: 1
  - Feature sets: 3
  - Kernels: 1 (linear)
  - C values: 4

[INFO] Total models to train: 7 × 4 × 1 × 3 × 1 × 4 = 336
```

**What this does:**
- Creates `experiment_041_classifier_002_svm_models` table
- Calculates hyperparameter grid
- Prepares for training

---

### Step 12: Train SVM Models

```bash
classifier-train-svm --amplitude-method 2 --workers 12 --kernel linear
```

**Expected output:**
```
[INFO] Training SVM models for classifier 2
[INFO] Using 12 parallel workers
[INFO] Kernel filter: linear only

[INFO] Task grid:
  - Decimation: 7 values
  - Data types: 4 values
  - Feature sets: 3 values
  - Kernels: linear
  - C values: 4

[INFO] Total tasks: 84 (7 × 4 × 3 × 4)

[WORKER] Starting: dec=0, dtype=6, amp=2, efs=1, kernel=linear, C=0.1
[WORKER] DB connected in 0.05s
[WORKER] Loading training features...
[WORKER] Loaded 11620 training samples in 0.12s
...
[TASK 1/84] SUCCESS: dec=0, dtype=6, amp=2, efs=1, kernel=linear, C=0.1, test_acc=0.7234
  Timing: LOAD=0.5s, SVM=15.2s, CV=0.0s, PRED=0.3s, SAVE=1.1s, TOTAL=17.1s
  Data: train=11620, test=11620, verify=11620, features=XXX
...
[SUCCESS] All 84 tasks completed
  - Best model: dec=0, dtype=6, kernel=linear, C=10.0
  - Test accuracy: 0.7856
```

**What this does:**
- Trains 84 SVM models (7×4×3×4 with linear kernel)
- Uses raw feature vectors (not distances)
- Saves models to `/Volumes/ArcData/V3_database/experiment041/classifier_files/svm_models/classifier_002/`
- Inserts results into `experiment_041_classifier_002_svm_models`

**Expected Performance:**
- May be slower than distance-based (more features)
- May be better or worse accuracy (depends on feature quality)

---

## Summary: Key Differences from Classifier 1

| Aspect | Classifier 1 (Distance-based) | Classifier 2 (Raw features) |
|--------|-------------------------------|----------------------------|
| **Feature type** | Distances to references | Raw feature values |
| **Feature dimension** | 52 (13 classes × 4 metrics) | Variable (depends on feature sets) |
| **Reference segments** | Required | Not required |
| **Feature building** | Compute distances | Concatenate raw features |
| **Training speed** | Faster (fewer features) | May be slower (more features) |
| **Interpretability** | Distance-based | Feature-based |

---

## Verification Commands

### Check Classifier 2 Status

```bash
classifier-list
```

**Expected output:**
```
Classifiers for Experiment 41:
  [1] baseline_svm (ACTIVE) - Baseline SVM classifier
  [2] raw_features_svm (ACTIVE) - SVM using raw features (no distances)
```

### Check Configuration

```bash
classifier-config-list
```

**Expected output:**
```
Configurations for Classifier 2:
  [2] raw_baseline (ACTIVE)
    - Feature sets: 1, 2, 5
    - Distance functions: 1, 2, 3, 4
    - Hyperparameters: linear (C: 0.1, 1.0, 10.0, 100.0)
```

### Check Feature Builder Settings

```bash
classifier-config-show --config-id 2
```

**Expected output should show:**
```
Feature Builder:
  Include original features: True  ← KEY SETTING
  Compute inter-class distances: False  ← KEY SETTING
  Compute intra-class distances: False  ← KEY SETTING
```

### Check Feature Table Structure

```sql
-- In PostgreSQL
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'experiment_041_classifier_002_raw_features'
ORDER BY ordinal_position;
```

Should show columns for raw features (not distance features).

---

## Troubleshooting

### Issue: "Feature builder not configured"

**Solution:**
Make sure you ran Step 5 (classifier-config-set-feature-builder) with correct flags.

### Issue: "Reference segments required"

**Solution:**
Even with raw features, the system may require reference segments table to exist (schema constraint). If so:

```bash
classifier-create-references-table
# Then just leave it empty - won't be used
```

### Issue: "Distance functions must be configured"

**Solution:**
Run Step 6 to add distance functions (required even if not computing).

### Issue: Feature dimension mismatch

**Problem:** Raw features have different dimensions than expected.

**Solution:**
Check which features are actually in the feature sets:
```sql
SELECT fs.feature_set_name, f.feature_name
FROM ml_feature_sets_lut fs
JOIN ml_feature_set_features fsf ON fs.feature_set_id = fsf.feature_set_id
JOIN ml_features_lut f ON fsf.feature_id = f.feature_id
WHERE fs.feature_set_id IN (1, 2, 5);
```

---

## Next Steps

1. **Run all commands in sequence** (Steps 1-12)
2. **Compare performance** with classifier 1:
   - Accuracy
   - Training time
   - Model size
3. **Analyze which approach works better** for your arc detection task
4. **Consider hybrid approach:** Both raw features AND distances (set all flags to true)

---

## Hybrid Approach (Optional)

If you want BOTH raw features AND distances:

```bash
classifier-config-set-feature-builder --config-id 2 \
  --include-original \
  --compute-distances-inter \
  --compute-distances-intra
```

This will create feature vectors with:
- Raw features from feature sets 1, 2, 5
- Distance features (4 metrics × 13 classes = 52)
- Total dimension: raw_features + 52

**May give best results but larger feature vectors.**

---

## Complete Command Script

Here's the complete sequence in one script:

```bash
#!/bin/bash
# Create classifier 2 with raw features

./mldp << 'EOF'
connect
set experiment 41

# Step 1: Create classifier
classifier-new --name raw_features_svm --description "SVM using raw features (no distances)" --type svm

# Step 2: Select classifier
set classifier 2

# Step 3: Create configuration
classifier-config-create --name raw_baseline --active

# Step 4: Add feature sets
classifier-config-add-feature-sets --config-id 2 --experiment-feature-sets 1 2 5

# Step 5: Set feature builder (KEY STEP)
classifier-config-set-feature-builder --config-id 2 --include-original --no-compute-distances-inter --no-compute-distances-intra

# Step 6: Add distance functions (required)
classifier-config-add-distance-functions --config-id 2 --distance-functions 1 2 3 4

# Step 7: Add hyperparameters
classifier-config-add-hyperparameters --config-id 2 --kernel linear --C 0.1 1.0 10.0 100.0

# Step 8: Create splits table
classifier-create-splits-table

# Step 9: Assign splits
classifier-assign-splits --strategy stratified --split-ratios 0.33 0.33 0.34

# Step 10: Build features
classifier-build-features --amplitude-method 2

# Step 11: Initialize training
classifier-train-svm-init --amplitude-method 2

# Step 12: Train models
classifier-train-svm --amplitude-method 2 --workers 12 --kernel linear

exit
EOF
```

Save as `create_classifier_2.sh` and run with `bash create_classifier_2.sh`.
