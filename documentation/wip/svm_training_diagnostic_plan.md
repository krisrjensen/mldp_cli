# SVM Training Performance Diagnostic Plan

**Filename:** svm_training_diagnostic_plan.md
**Author(s):** Kristophor Jensen
**Date Created:** 20251019_000000
**Date Revised:** 20251019_000000
**File version:** 2.0.9.16
**Description:** Diagnostic plan to identify SVM training bottleneck in Phase 4

---

## Problem Statement

**Observed Behavior:**
- Feature building: 104,580 vectors in 13.7 minutes (0.01 sec/vector) ✓ FAST
- SVM training: 45+ minutes with ZERO outputs ✗ EXTREMELY SLOW

**Configuration:**
- Command: `classifier-train-svm --amplitude-method 2 --workers 7 --kernel linear`
- Total tasks: 84 (7 decimations × 4 data types × 1 amplitude × 3 feature sets × 1 kernel)
- Training data: 34,860 split assignments
- Feature vectors: 104,580 total

**Expected Performance:**
- Initial estimate: 0.4-0.8 minutes total
- Reality: No output after 45+ minutes

**Hypothesis:**
Something fundamentally wrong with SVM training or feature processing causing extreme slowdown.

---

## Diagnostic Strategy

We will add instrumentation at multiple levels to identify the bottleneck:

### Level 1: Per-Task Timing Breakdown
Add detailed timing logs inside the `train_svm_worker()` function to measure:
- Feature loading time
- Data preparation time
- SVM training time
- Metrics calculation time
- File I/O time
- Database operations time

### Level 2: Worker Health Monitoring
Detect stuck/deadlocked workers:
- Heartbeat signals from each worker
- Timeout detection for stuck tasks
- Exception logging from worker processes

### Level 3: Data Verification
Verify data sanity:
- Feature vector dimensions
- Training data size per model
- Split assignment correctness
- Memory usage per worker

### Level 4: Simplified Test
Create minimal test to isolate problem:
- Single SVM training task
- Verbose logging at each step
- Direct execution (no multiprocessing)
- Verify basic functionality

---

## Implementation Plan

### Phase 1: Add Diagnostic Instrumentation (30 minutes)

**Step 1.1: Enhance train_svm_worker() with timing**

Add timing checkpoints to measure each operation:

```python
def train_svm_worker(work_item):
    """Train single SVM with detailed timing diagnostics"""
    timings = {}
    t_start = time.time()

    try:
        # CHECKPOINT 1: Feature loading
        t_load_start = time.time()
        features = load_features(...)
        timings['feature_loading'] = time.time() - t_load_start

        # CHECKPOINT 2: Data preparation
        t_prep_start = time.time()
        X_train, y_train = prepare_data(...)
        timings['data_preparation'] = time.time() - t_prep_start

        # CHECKPOINT 3: SVM training
        t_train_start = time.time()
        svm.fit(X_train, y_train)
        timings['svm_training'] = time.time() - t_train_start

        # CHECKPOINT 4: Metrics calculation
        t_metrics_start = time.time()
        metrics = calculate_metrics(...)
        timings['metrics_calculation'] = time.time() - t_metrics_start

        # CHECKPOINT 5: File I/O
        t_io_start = time.time()
        save_model(...)
        timings['file_io'] = time.time() - t_io_start

        timings['total'] = time.time() - t_start

        return {
            'success': True,
            'timings': timings,
            'data_size': len(X_train),
            'feature_dim': X_train.shape[1] if len(X_train) > 0 else 0,
            ...
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'timings': timings,
            ...
        }
```

**Step 1.2: Add worker heartbeat logging**

Log when each worker picks up a task:

```python
with Pool(processes=num_workers) as pool:
    for i, result in enumerate(pool.imap_unordered(train_svm_worker, work_items), 1):
        if result['success']:
            timings = result['timings']
            print(f"[TASK {i}/{total_tasks}] SUCCESS:")
            print(f"  Config: dec={dec}, dtype={dtype}, amp={amp}, efs={efs}")
            print(f"  Timing: load={timings['feature_loading']:.2f}s, "
                  f"prep={timings['data_preparation']:.2f}s, "
                  f"train={timings['svm_training']:.2f}s, "
                  f"metrics={timings['metrics_calculation']:.2f}s, "
                  f"io={timings['file_io']:.2f}s")
            print(f"  Data: {result['data_size']} samples, "
                  f"{result['feature_dim']} features")
            print(f"  Total: {timings['total']:.2f}s")
```

**Step 1.3: Add --verbose flag**

Allow user to enable/disable detailed diagnostics:

```bash
classifier-train-svm --amplitude-method 2 --workers 7 --kernel linear --verbose
```

---

### Phase 2: Create Simplified Test Command (20 minutes)

**Step 2.1: Add classifier-test-svm-single command**

Create new command to train ONE SVM with maximum verbosity:

```python
def cmd_classifier_test_svm_single(self, args):
    """Test single SVM training with detailed diagnostics

    Usage: classifier-test-svm-single [OPTIONS]

    Options:
      --decimation-factor <n>   Test with this decimation factor (default: 0)
      --data-type <id>          Test with this data type (default: 6)
      --amplitude-method <id>   Test with this amplitude method (default: 1)
      --feature-set <id>        Test with this feature set (default: 1)
      --kernel <name>           Test with this kernel (default: linear)
      --C <value>               SVM C parameter (default: 1.0)
      --gamma <value>           SVM gamma parameter (default: scale)

    This command trains a SINGLE SVM with the specified configuration
    and displays detailed timing and diagnostic information.
    """
    # Parse arguments
    # Build single work_item
    # Execute train_svm_worker() directly (no Pool)
    # Display detailed results
```

**Step 2.2: Test with minimal configuration**

```bash
# Test single SVM with default settings
classifier-test-svm-single

# Should complete in < 60 seconds if working correctly
# If this hangs, problem is in worker function itself
```

---

### Phase 3: Verify Data Sanity (15 minutes)

**Step 3.1: Add data verification queries**

Check feature vector and split assignment counts:

```sql
-- Verify feature vectors for amplitude method 2
SELECT
    decimation_factor,
    experiment_adc_data_type_id,
    COUNT(*) as feature_count,
    COUNT(CASE WHEN extraction_status_id = 2 THEN 1 END) as extracted_count
FROM experiment_041_classifier_001_svm_features
WHERE amplitude_processing_method_id = 2
GROUP BY decimation_factor, experiment_adc_data_type_id
ORDER BY decimation_factor, experiment_adc_data_type_id;

-- Verify split assignments
SELECT
    split_type_id,
    COUNT(*) as count
FROM experiment_041_split_assignments
GROUP BY split_type_id;
-- Expected: type 1 (train), type 2 (test), type 3 (verification)
```

**Step 3.2: Check feature file sizes**

Verify feature files are reasonable size:

```bash
# Check feature file sizes
find /Volumes/ArcData/V3_database/experiment041/classifier_files/svm_features/classifier_001/ \
  -name "*.npy" -type f -exec ls -lh {} \; | head -20

# Look for anomalies:
# - Files that are too large (> 10 MB)
# - Files that are too small (< 1 KB)
# - Missing files
```

**Step 3.3: Load and inspect sample feature vector**

Verify feature vector format:

```python
import numpy as np

# Load a sample feature vector
sample_file = "/Volumes/ArcData/V3_database/experiment041/classifier_files/svm_features/classifier_001/D000000_TADC6_A2_EFS001/segment_00001_class_02.npy"

features = np.load(sample_file)
print(f"Shape: {features.shape}")
print(f"Dtype: {features.dtype}")
print(f"Min: {features.min()}, Max: {features.max()}")
print(f"Contains NaN: {np.isnan(features).any()}")
print(f"Contains Inf: {np.isinf(features).any()}")
```

---

### Phase 4: Profile Critical Sections (20 minutes)

**Step 4.1: Profile feature loading**

Measure time to load ALL features for one configuration:

```python
import time
import numpy as np

config = (0, 6, 2, 1)  # decimation, data_type, amplitude, feature_set
base_path = "/Volumes/ArcData/V3_database/experiment041/classifier_files/svm_features/classifier_001/"
subdir = f"D{config[0]:06d}_TADC{config[1]}_A{config[2]}_EFS{config[3]:03d}"

# Get all feature files
import glob
feature_files = glob.glob(f"{base_path}/{subdir}/segment_*.npy")

print(f"Loading {len(feature_files)} feature files...")
t_start = time.time()

features_list = []
for f in feature_files:
    features_list.append(np.load(f))

t_elapsed = time.time() - t_start
print(f"Loaded in {t_elapsed:.2f} seconds ({t_elapsed/len(feature_files)*1000:.2f} ms/file)")
```

**Expected:**
- Should load ~35,000 files in < 10 seconds
- If this takes > 30 seconds, feature loading is the bottleneck

**Step 4.2: Profile SVM training**

Measure time to train SVM on loaded data:

```python
from sklearn.svm import SVC
import numpy as np

# Assume features_list loaded from above
X = np.vstack(features_list)
y = np.array([...])  # class labels

print(f"Training SVM on {len(X)} samples with {X.shape[1]} features...")
t_start = time.time()

svm = SVC(kernel='linear', C=1.0)
svm.fit(X, y)

t_elapsed = time.time() - t_start
print(f"Trained in {t_elapsed:.2f} seconds")
```

**Expected:**
- Linear SVM on 30,000 samples should take < 30 seconds
- If this takes > 120 seconds, SVM training itself is slow

---

## Execution Plan

### Recommended Sequence:

1. **First: Simplified Test (Phase 2)**
   - Fastest to implement
   - Isolates problem to single task
   - No multiprocessing complexity
   - **Command:** `classifier-test-svm-single`
   - **Expected outcome:** Identify if problem is in worker function or parallel execution

2. **Second: Data Verification (Phase 3)**
   - Quick sanity checks
   - Verify no corrupted data
   - **Expected outcome:** Rule out data quality issues

3. **Third: Diagnostic Instrumentation (Phase 1)**
   - Add timing to worker function
   - Enable verbose mode
   - **Command:** `classifier-train-svm --amplitude-method 2 --workers 1 --kernel linear --verbose`
   - **Expected outcome:** Identify which operation is slow

4. **Fourth: Profiling (Phase 4)**
   - Deep dive into slow operation
   - Measure feature loading and SVM training separately
   - **Expected outcome:** Quantify bottleneck and guide optimization

---

## Success Criteria

### Minimum Viable Diagnostic:
- [ ] Identify which operation takes > 50% of time per task
- [ ] Measure actual time per task (not estimate)
- [ ] Determine if problem is:
  - Feature loading
  - Data preparation
  - SVM training
  - Metrics calculation
  - File I/O
  - Database operations
  - Worker synchronization

### Complete Diagnostic:
- [ ] Per-operation timing for all checkpoints
- [ ] Memory usage per worker
- [ ] Data sanity verification (no NaN, Inf, corrupted files)
- [ ] Comparison: single-worker vs multi-worker performance
- [ ] Root cause identified with evidence

---

## Known Hypotheses (Ranked by Likelihood)

### Hypothesis 1: Feature Loading Bottleneck (HIGH)
**Evidence for:**
- Loading 104,580 .npy files repeatedly for each SVM
- Each task may load ~35,000 feature files
- File I/O can be slow on network drives

**Test:** Profile feature loading in Phase 4.1

**If confirmed:**
- Optimize: Cache loaded features in memory
- Alternative: Create single consolidated .npz file per configuration

### Hypothesis 2: SVM Training Data Too Large (MEDIUM)
**Evidence for:**
- 34,860 samples is large for SVM
- Feature dimension unknown but potentially high
- Linear kernel should be fast, but maybe features are very high-dimensional

**Test:** Inspect feature dimensions in Phase 3.3, profile SVM training in Phase 4.2

**If confirmed:**
- Reduce training data size
- Use SGDClassifier instead of SVC for large datasets
- Feature selection/dimensionality reduction

### Hypothesis 3: Database Queries in Worker (LOW)
**Evidence for:**
- Workers may be querying database for split assignments
- Database contention with multiple workers

**Test:** Timing instrumentation in Phase 1.1

**If confirmed:**
- Pre-load split assignments before worker pool
- Pass data as arguments instead of querying in worker

### Hypothesis 4: Worker Deadlock/Stuck (LOW)
**Evidence for:**
- No output even with enhanced progress
- Multiple workers may be contending for resources

**Test:** Simplified single-worker test in Phase 2

**If confirmed:**
- Reduce workers to 1
- Investigate resource contention (file locks, database connections)

---

## Timeline

- **Phase 2 (Simplified Test):** 20 minutes implementation + 5 minutes testing = 25 minutes
- **Phase 3 (Data Verification):** 15 minutes queries + 5 minutes analysis = 20 minutes
- **Phase 1 (Instrumentation):** 30 minutes implementation + 10 minutes testing = 40 minutes
- **Phase 4 (Profiling):** 20 minutes profiling + 10 minutes analysis = 30 minutes

**Total Diagnostic Time:** ~2 hours

**Expected Result:** Root cause identified with concrete evidence

---

## Next Steps After Diagnosis

Once bottleneck is identified:

### If Feature Loading:
- Implement feature caching
- Create consolidated feature files
- Use memory-mapped arrays

### If SVM Training:
- Switch to SGDClassifier for large datasets
- Reduce feature dimensions
- Use sparse matrices if applicable

### If Data Preparation:
- Optimize data structures
- Pre-allocate arrays
- Reduce data copying

### If Worker Overhead:
- Reduce number of workers
- Use threading instead of multiprocessing
- Batch multiple tasks per worker

---

## Version History

- v2.0.9.16: Initial diagnostic plan created
