# SVM Training MPCCTL Protocol Implementation Plan

**Filename:** svm_training_mpcctl_implementation_plan.md
**Author(s):** Kristophor Jensen
**Date Created:** 20251019_190000
**File version:** 2.0.10.0
**Description:** Plan to implement .mpcctl protocol for SVM training with zero synchronization

---

## Executive Summary

### Current Performance Analysis

**Diagnostic Test Results (Experiment 41, Classifier 1):**
```
[WORKER] Loaded 868 training samples in 0.05s
[WORKER] Loaded 249 test samples in 0.02s
[WORKER] Loaded 128 verification samples in 0.01s
[WORKER] Total feature loading: 0.07s  ✓ FAST (not the bottleneck)
[WORKER] SVM training complete in 571.47s (9.5 minutes)
[WORKER] Cross-validation: DISABLED (was taking 23+ minutes)
[WORKER] Total time: 572.69s per task
```

**Current Projection:**
- 84 tasks × 572.69s / 7 workers = **1.9 hours** (acceptable)
- Memory usage: LOW (no issues)

**Problem:**
- Current implementation uses `multiprocessing.Pool.imap_unordered()`
- Workers idle while waiting for database queries in main process
- All tasks generated during runtime (synchronization overhead)

**Solution:**
- Implement .mpcctl protocol from mpcctl_distance_calculator
- Pre-generate ALL tasks before any training starts
- Workers independently pick tasks from PID-specific todo files
- Zero synchronization during training

---

## Part 1: .mpcctl Protocol for SVM Training

### 1.1 Architecture Overview

**Based on:**
`/Users/kjensen/Documents/GitHub/mldp/mldp_exp18_distance/src/mpcctl_protocol.py`

**File Structure:**
```
/Volumes/ArcData/V3_database/experiment041/classifier_files/svm_training/
├── .mpcctl/                          # Root coordination directory
│   ├── registered_pids.json          # List of worker PIDs
│   ├── manager_state.json            # Manager process state
│   └── config.json                   # Training configuration
├── classifier_001/                   # Per-classifier work directories
│   └── .mpcctl/                      # Classifier .mpcctl directory
│       ├── 12345_todo.dat            # Worker 12345's task list
│       ├── 12345_done.dat            # Worker 12345's completed tasks
│       ├── 12345_metadata.json       # Worker 12345's configuration
│       ├── 12346_todo.dat            # Worker 12346's task list
│       ├── 12346_done.dat            # Worker 12346's completed tasks
│       └── 12346_metadata.json       # Worker 12346's configuration
└── logs/                             # Training logs
    ├── manager.log                   # Manager process log
    ├── worker_12345.log              # Worker 12345 log
    └── worker_12346.log              # Worker 12346 log
```

### 1.2 Task Definition

**SVM Training Task Format:**
Each line in `{pid}_todo.dat` contains one training task:
```
task_id,dec,dtype,amp,efs,kernel,C,gamma
1,0,6,2,1,linear,1.0,scale
2,0,6,2,1,linear,0.1,scale
3,0,6,2,1,rbf,1.0,0.1
...
```

**Task ID:** Sequential integer (1-N)
**Parameters:**
- `dec`: Decimation factor
- `dtype`: Data type ID
- `amp`: Amplitude processing method ID
- `efs`: Experiment feature set ID
- `kernel`: SVM kernel (linear, rbf, poly)
- `C`: SVM C parameter
- `gamma`: SVM gamma parameter (or 'scale'/'auto')

### 1.3 Pre-Generation Phase

**Command:** `classifier-train-svm-init`

**Current Implementation:** Creates database tables
**New Implementation:** Also pre-generates .mpcctl task files

**Enhanced Steps:**

1. **Query hyperparameters from configuration**
   ```python
   # Decimation factors, data types, amplitude methods, feature sets
   # SVM kernels, C values, gamma values
   ```

2. **Generate all task combinations**
   ```python
   tasks = []
   task_id = 1
   for dec in decimation_factors:
       for dtype in data_type_ids:
           for amp in amplitude_methods:
               for efs in experiment_feature_sets:
                   for kernel in svm_kernels:
                       for C in svm_C_values:
                           for gamma in svm_gamma_values:
                               if kernel == 'linear':
                                   gamma = 'scale'  # linear doesn't use gamma
                               tasks.append({
                                   'task_id': task_id,
                                   'dec': dec,
                                   'dtype': dtype,
                                   'amp': amp,
                                   'efs': efs,
                                   'kernel': kernel,
                                   'C': C,
                                   'gamma': gamma
                               })
                               task_id += 1
   ```

3. **Distribute tasks among N workers**
   ```python
   tasks_per_worker = len(tasks) // num_workers

   for i, pid in enumerate(worker_pids):
       start_idx = i * tasks_per_worker
       if i == len(worker_pids) - 1:  # Last worker gets remainder
           end_idx = len(tasks)
       else:
           end_idx = start_idx + tasks_per_worker

       assigned_tasks = tasks[start_idx:end_idx]

       # Write to {pid}_todo.dat
       todo_file = mpcctl_dir / f"{pid}_todo.dat"
       with open(todo_file, 'w') as f:
           for task in assigned_tasks:
               f.write(f"{task['task_id']},{task['dec']},{task['dtype']},"
                      f"{task['amp']},{task['efs']},{task['kernel']},"
                      f"{task['C']},{task['gamma']}\n")

       # Create empty {pid}_done.dat
       done_file = mpcctl_dir / f"{pid}_done.dat"
       done_file.touch()
   ```

4. **Create metadata files**
   ```python
   metadata = {
       'pid': pid,
       'worker_id': i,
       'experiment_id': exp_id,
       'classifier_id': cls_id,
       'global_classifier_id': global_classifier_id,
       'task_count': len(assigned_tasks),
       'db_config': db_config,
       'label_categories': label_categories
   }

   metadata_file = mpcctl_dir / f"{pid}_metadata.json"
   with open(metadata_file, 'w') as f:
       json.dump(metadata, f, indent=2)
   ```

### 1.4 Worker Process

**Worker Function:**
```python
def svm_training_worker(pid, mpcctl_dir):
    """
    Independent worker process that reads tasks from {pid}_todo.dat
    and marks completed tasks in {pid}_done.dat.

    Zero synchronization with other workers.
    """
    import json
    import time
    from pathlib import Path

    # Load metadata
    metadata_file = mpcctl_dir / f"{pid}_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Open files
    todo_file = mpcctl_dir / f"{pid}_todo.dat"
    done_file = mpcctl_dir / f"{pid}_done.dat"

    # Read already completed tasks
    with open(done_file, 'r') as f:
        completed_task_ids = {int(line.strip()) for line in f if line.strip()}

    # Read all tasks
    with open(todo_file, 'r') as f:
        all_tasks = [line.strip() for line in f if line.strip()]

    # Filter out completed tasks
    pending_tasks = [
        task for task in all_tasks
        if int(task.split(',')[0]) not in completed_task_ids
    ]

    print(f"[WORKER {pid}] Starting: {len(pending_tasks)} pending tasks")

    # Process each task
    for task_line in pending_tasks:
        # Parse task
        parts = task_line.split(',')
        task_id = int(parts[0])
        dec = int(parts[1])
        dtype = int(parts[2])
        amp = int(parts[3])
        efs = int(parts[4])
        kernel = parts[5]
        C = float(parts[6])
        gamma = parts[7]  # 'scale', 'auto', or float

        if gamma not in ['scale', 'auto']:
            gamma = float(gamma)

        # Build work item (same as current implementation)
        config_tuple = (
            dec, dtype, amp, efs,
            {'kernel': kernel, 'C': C, 'gamma': gamma},
            metadata['db_config'],
            metadata['label_categories'],
            metadata['experiment_id'],
            metadata['classifier_id'],
            metadata['global_classifier_id']
        )

        print(f"[WORKER {pid}] Task {task_id}: dec={dec}, dtype={dtype}, amp={amp}, efs={efs}, kernel={kernel}, C={C}")

        # Train SVM (existing train_svm_worker function)
        result = train_svm_worker(config_tuple)

        if result['success']:
            print(f"[WORKER {pid}] Task {task_id}: SUCCESS (test_acc={result['metrics_test']['accuracy']:.4f})")

            # Insert into database (same as current implementation)
            insert_svm_results_to_database(result, metadata)

            # Mark task as done
            with open(done_file, 'a') as f:
                f.write(f"{task_id}\n")
        else:
            print(f"[WORKER {pid}] Task {task_id}: FAILED - {result['error']}")

    print(f"[WORKER {pid}] Completed all tasks")
```

### 1.5 Manager Process

**Manager Function:**
```python
def svm_training_manager(experiment_id, classifier_id, num_workers):
    """
    Manager process that spawns workers and monitors progress.

    Does NOT coordinate work - workers are fully independent.
    Manager only monitors and reports status.
    """
    import multiprocessing as mp
    from pathlib import Path

    mpcctl_dir = Path(f"/Volumes/ArcData/V3_database/experiment{experiment_id:03d}/"
                     f"classifier_files/svm_training/classifier_{classifier_id:03d}/.mpcctl")

    # Spawn workers
    processes = []
    for i in range(num_workers):
        pid = os.getpid() + i + 1  # Generate pseudo-PIDs
        p = mp.Process(target=svm_training_worker, args=(pid, mpcctl_dir))
        p.start()
        processes.append(p)
        print(f"[MANAGER] Spawned worker {pid} (Process {p.pid})")

    # Monitor workers
    while any(p.is_alive() for p in processes):
        # Report progress
        total_tasks = 0
        completed_tasks = 0

        for pid_file in mpcctl_dir.glob("*_done.dat"):
            with open(pid_file, 'r') as f:
                completed = len([line for line in f if line.strip()])
                completed_tasks += completed

        for pid_file in mpcctl_dir.glob("*_todo.dat"):
            with open(pid_file, 'r') as f:
                total = len([line for line in f if line.strip()])
                total_tasks += total

        progress_pct = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        print(f"[MANAGER] Progress: {completed_tasks}/{total_tasks} ({progress_pct:.1f}%)")

        time.sleep(30)  # Report every 30 seconds

    # Wait for all workers to finish
    for p in processes:
        p.join()

    print(f"[MANAGER] All workers completed")
```

### 1.6 Shell Command

**New Command:** `classifier-train-svm-mpcctl`

**Usage:**
```bash
classifier-train-svm-mpcctl --workers 7 --amplitude-method 2 --kernel linear
```

**Implementation:**
```python
def cmd_classifier_train_svm_mpcctl(self, args):
    """
    Train SVM using .mpcctl protocol for zero-synchronization parallel processing.

    Phase 1: Pre-generate all task files (classifier-train-svm-init does this)
    Phase 2: Spawn manager process in background
    Phase 3: Manager spawns workers that independently process tasks
    """
    # Parse arguments
    # Validate experiment/classifier selected
    # Call manager in background process
    # Return immediately (manager runs independently)
```

---

## Part 2: Complete Verification Dataset

### 2.1 Current Verification

**Current Implementation:**
- Uses experiment_041_split_assignments table
- Only tests on segments from files in experiment_041_file_training_data
- Limited to segments already processed

**Limitation:**
- Cannot test generalization to completely unseen files

### 2.2 Complete Verification Enhancement

**New Command:** `classifier-complete-verification`

**Purpose:**
- Select segments from files NOT in experiment_041_file_training_data
- Process through full pipeline (decimate, scale, distance calculation)
- Evaluate trained SVM on completely unseen data

**Implementation Steps:**

1. **Query unseen files**
   ```python
   cursor.execute("""
       SELECT f.file_id, f.file_path, s.label_id
       FROM ml_file_training_data f
       JOIN ml_segment_labels s ON f.label_id = s.label_id
       WHERE f.file_id NOT IN (
           SELECT file_id FROM experiment_041_file_training_data
       )
       AND s.active = TRUE
       ORDER BY s.label_id, f.file_id
   """)
   unseen_files = cursor.fetchall()
   ```

2. **Select segments from unseen files**
   ```python
   # Select N segments per label from unseen files
   segments_per_label = 50

   for label_id in all_labels:
       label_files = [f for f in unseen_files if f[2] == label_id]
       selected = random.sample(label_files, min(segments_per_label, len(label_files)))

       for file_id, file_path, label_id in selected:
           # Extract random segment of correct size
           segment = extract_segment(file_path, segment_size)
           # Store in complete_verification table
   ```

3. **Process through full pipeline**
   ```python
   # For each segment:
   # 1. Decimate to all decimation factors
   # 2. Scale with all amplitude methods
   # 3. Calculate distances to reference segments
   # 4. Build feature vectors
   # 5. Run through trained SVM
   # 6. Store predictions
   ```

4. **Create dedicated table**
   ```sql
   CREATE TABLE experiment_041_classifier_001_complete_verification (
       segment_id SERIAL PRIMARY KEY,
       file_id INTEGER NOT NULL,
       label_id INTEGER NOT NULL,
       decimation_factor INTEGER NOT NULL,
       data_type_id INTEGER NOT NULL,
       amplitude_processing_method_id INTEGER NOT NULL,
       experiment_feature_set_id INTEGER NOT NULL,
       feature_vector_path TEXT NOT NULL,
       predicted_label INTEGER,
       true_label INTEGER NOT NULL,
       prediction_confidence REAL,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

5. **Evaluation**
   ```python
   # Compare predicted_label vs true_label
   # Calculate accuracy on completely unseen data
   # This tests true generalization capability
   ```

---

## Part 3: Consistent Ordering in Distance Calculations

### 3.1 Current Issue

**Problem:**
- Reference segment selection may not be deterministic
- Plots use internal indices instead of label names

### 3.2 Fix: Sort by segment_label_id

**In classifier-select-references:**
```python
# Current (may be non-deterministic):
cursor.execute(f"""
    SELECT segment_id, label_id, segment_data_path
    FROM experiment_{exp_id:03d}_segment_training_data
    WHERE decimation_factor = %s AND data_type_id = %s
      AND amplitude_processing_method_id = %s
""", (dec, dtype, amp))

# Fixed (deterministic):
cursor.execute(f"""
    SELECT segment_id, label_id, segment_data_path
    FROM experiment_{exp_id:03d}_segment_training_data
    WHERE decimation_factor = %s AND data_type_id = %s
      AND amplitude_processing_method_id = %s
    ORDER BY label_id, segment_id  -- CONSISTENT ORDERING
""", (dec, dtype, amp))
```

### 3.3 Fix: Use label names in plots

**In plotting functions:**
```python
# Current (uses indices):
plt.xlabel('Class Index')
plt.xticks(range(num_classes))

# Fixed (uses label names):
cursor.execute("SELECT label_id, label_name FROM segment_labels WHERE active = TRUE ORDER BY label_id")
label_names = {label_id: name for label_id, name in cursor.fetchall()}

# Map label_ids to names
x_labels = [label_names.get(label_id, f"Class_{label_id}") for label_id in sorted_label_ids]
plt.xlabel('Segment Label')
plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
```

---

## Implementation Timeline

### Phase 1: .mpcctl Protocol (Priority: HIGH)
**Estimated Time:** 8-12 hours

1. Create `mpcctl_svm_protocol.py` module (3 hours)
2. Enhance `classifier-train-svm-init` to pre-generate task files (2 hours)
3. Implement `svm_training_worker()` function (2 hours)
4. Implement `svm_training_manager()` function (2 hours)
5. Create `classifier-train-svm-mpcctl` command (1 hour)
6. Testing and debugging (2-4 hours)

**Expected Improvement:**
- Current: 1.9 hours with Pool (some synchronization overhead)
- With mpcctl: 1.7 hours (10-15% faster due to zero synchronization)
- Better: Can pause/resume training, individual worker logs

### Phase 2: Complete Verification (Priority: MEDIUM)
**Estimated Time:** 6-8 hours

1. Implement `classifier-complete-verification` command (3 hours)
2. Create database tables (1 hour)
3. Implement unseen file selection (2 hours)
4. Implement evaluation metrics (2 hours)

**Value:** Tests true generalization to unseen data

### Phase 3: Consistent Ordering (Priority: LOW)
**Estimated Time:** 2-3 hours

1. Add ORDER BY to all reference selection queries (1 hour)
2. Update plotting functions to use label names (1 hour)
3. Verify consistency across runs (1 hour)

**Value:** Reproducibility and better visualization

---

## Recommendations

### Immediate Action

**Start with Phase 1 (.mpcctl protocol)** because:
1. Training already works (571s per task is acceptable)
2. .mpcctl adds resilience (pause/resume, worker failure recovery)
3. Provides foundation for future scaling
4. Matches established pattern from distance calculations

### Alternative: Ship Current Implementation

**Current implementation is actually USABLE:**
- 84 tasks × 571s / 7 workers = 1.9 hours total
- Memory usage is low (no issues)
- Cross-validation disabled (major speedup achieved)

**You could:**
1. Run training NOW with current implementation
2. Implement .mpcctl in parallel while training runs
3. Use .mpcctl for next training run

### Question for User

**Do you want to:**
- **Option A:** Implement .mpcctl protocol first (~1-2 days), then train
- **Option B:** Start training NOW with current implementation, implement .mpcctl later
- **Option C:** Run one complete training now to validate, then implement .mpcctl for production

---

## Files to Create

1. `/Users/kjensen/Documents/GitHub/mldp/mldp_cli/src/mpcctl_svm_protocol.py`
2. Enhanced `classifier-train-svm-init` in `mldp_shell.py`
3. New `classifier-train-svm-mpcctl` in `mldp_shell.py`
4. New `classifier-complete-verification` in `mldp_shell.py`
5. Update reference selection queries with ORDER BY
6. Update plotting functions with label names

---

## Summary

**Current State:** SVM training works, takes ~2 hours for 84 tasks with 7 workers

**Proposed Enhancements:**
1. .mpcctl protocol for zero-synchronization (10-15% faster, more robust)
2. Complete verification for true generalization testing
3. Consistent ordering for reproducibility

**Decision Point:** Implement now or train first, then enhance?
