# Complete Verification Implementation Plan

**Filename:** complete_verification_implementation_plan.md
**Author(s):** Kristophor Jensen
**Date Created:** 20251019_200000
**File version:** 2.0.10.0
**Description:** Complete verification system for testing SVM on unseen segments

---

## Executive Summary

### Purpose

Create a comprehensive verification system that:
1. Tests trained SVM models on segments NOT used in training
2. Handles highly unbalanced datasets intelligently
3. Processes segments through the full pipeline (scale → distance → predict)
4. Generates detailed metrics and visualizations
5. Validates model generalization to truly unseen data

### Key Features

- **Flexible segment selection:** Can use test/verification sets + any other segments
- **Avoids training contamination:** Excludes all training segments
- **Handles class imbalance:** Configurable sampling strategies
- **Full pipeline processing:** Amplitude scaling → feature vectors → predictions
- **Comprehensive evaluation:** Per-class metrics, confusion matrices, ROC curves
- **Stores results:** Dedicated database tables for analysis

---

## Architecture Overview

### Data Flow

```
1. Select Segments
   ↓
   - Query all available segments (experiment_041_segment_training_data)
   - Exclude training segments (split_type_id = 'training')
   - Apply sampling strategy (balanced, stratified, or proportional)
   ↓
2. Load SVM Model
   ↓
   - Load trained model from: /Volumes/ArcData/.../svm_models/classifier_001/
   - Load model metadata (hyperparameters, label mappings)
   ↓
3. Process Each Segment
   ↓
   a. Load segment data from file
   b. Apply amplitude processing (z-score, min-max, etc.)
   c. Calculate distances to 13 reference segments
   d. Build 52-element feature vector (13 classes × 4 distance metrics)
   e. Run through SVM model
   f. Store prediction + confidence
   ↓
4. Generate Metrics
   ↓
   - Overall accuracy
   - Per-class precision/recall/F1
   - Confusion matrix (13-class)
   - Binary arc detection metrics
   - ROC curves, PR curves
   ↓
5. Create Visualizations
   ↓
   - Confusion matrices (13-class and binary)
   - Per-class performance charts
   - ROC/PR curves
   - Class distribution plots
```

---

## Part 1: Database Schema

### Table: experiment_041_classifier_001_complete_verification

```sql
CREATE TABLE experiment_041_classifier_001_complete_verification (
    -- Primary Key
    verification_id SERIAL PRIMARY KEY,

    -- Segment Information
    segment_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    true_label_id INTEGER NOT NULL,
    segment_size INTEGER NOT NULL,
    segment_offset INTEGER NOT NULL,

    -- Processing Configuration
    decimation_factor INTEGER NOT NULL,
    data_type_id INTEGER NOT NULL,
    amplitude_processing_method_id INTEGER NOT NULL,
    experiment_feature_set_id INTEGER NOT NULL,

    -- SVM Model Information
    model_id INTEGER NOT NULL,  -- References trained model
    svm_kernel TEXT NOT NULL,
    svm_C REAL NOT NULL,
    svm_gamma TEXT,

    -- Prediction Results
    predicted_label_id INTEGER NOT NULL,
    prediction_confidence REAL NOT NULL,  -- Max probability
    prediction_probabilities REAL[],      -- All 13 probabilities

    -- Binary Arc Detection
    true_is_arc BOOLEAN NOT NULL,
    predicted_is_arc BOOLEAN NOT NULL,
    arc_confidence REAL NOT NULL,

    -- Feature Vector
    feature_vector_path TEXT NOT NULL,  -- Path to .npy file

    -- Distance Metrics (for analysis)
    distance_l1_min REAL,      -- Min L1 distance to any reference
    distance_l2_min REAL,      -- Min L2 distance
    distance_cosine_min REAL,
    distance_pearson_min REAL,

    -- Metadata
    processing_time REAL,      -- Seconds to process this segment
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    CONSTRAINT fk_segment FOREIGN KEY (segment_id)
        REFERENCES experiment_041_segment_training_data(segment_id),
    CONSTRAINT fk_true_label FOREIGN KEY (true_label_id)
        REFERENCES segment_labels(label_id),
    CONSTRAINT fk_predicted_label FOREIGN KEY (predicted_label_id)
        REFERENCES segment_labels(label_id)
);

-- Indexes for fast queries
CREATE INDEX idx_exp041_cls001_cv_true_label
    ON experiment_041_classifier_001_complete_verification(true_label_id);

CREATE INDEX idx_exp041_cls001_cv_predicted_label
    ON experiment_041_classifier_001_complete_verification(predicted_label_id);

CREATE INDEX idx_exp041_cls001_cv_model
    ON experiment_041_classifier_001_complete_verification(model_id);

CREATE INDEX idx_exp041_cls001_cv_config
    ON experiment_041_classifier_001_complete_verification(
        decimation_factor, data_type_id,
        amplitude_processing_method_id, experiment_feature_set_id
    );
```

### Table: experiment_041_classifier_001_cv_models

```sql
CREATE TABLE experiment_041_classifier_001_cv_models (
    model_id SERIAL PRIMARY KEY,
    decimation_factor INTEGER NOT NULL,
    data_type_id INTEGER NOT NULL,
    amplitude_processing_method_id INTEGER NOT NULL,
    experiment_feature_set_id INTEGER NOT NULL,
    svm_kernel TEXT NOT NULL,
    svm_C REAL NOT NULL,
    svm_gamma TEXT,
    model_path TEXT NOT NULL,
    training_accuracy REAL,
    test_accuracy REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Part 2: Segment Selection Strategy

### 2.1 Available Segments

**Sources:**
1. **Test set:** Already assigned in `experiment_041_split_assignments` (split_type = 'test')
2. **Verification set:** Already assigned (split_type = 'verification')
3. **Other segments:** All segments NOT in training set

**Query:**
```sql
-- Get all non-training segments
SELECT
    s.segment_id,
    s.file_id,
    s.segment_label_id,
    s.segment_size,
    s.segment_offset,
    sa.split_type
FROM experiment_041_segment_training_data s
LEFT JOIN experiment_041_split_assignments sa
    ON s.segment_id = sa.segment_id
WHERE sa.split_type != 'training' OR sa.split_type IS NULL
ORDER BY s.segment_label_id, s.segment_id;
```

### 2.2 Sampling Strategies

**Strategy 1: Balanced Sampling**
- Select equal number of segments per class
- Good for unbalanced datasets
- Ensures all classes represented equally

```python
def balanced_sampling(segments_by_class, n_per_class=100):
    """Select n_per_class segments from each class."""
    selected = []
    for label_id, segments in segments_by_class.items():
        # Sample with replacement if class has fewer than n_per_class
        if len(segments) >= n_per_class:
            selected.extend(random.sample(segments, n_per_class))
        else:
            # Use all available + sample with replacement
            selected.extend(segments)
            remaining = n_per_class - len(segments)
            selected.extend(random.choices(segments, k=remaining))
    return selected
```

**Strategy 2: Stratified Sampling**
- Sample proportional to class distribution
- Maintains natural class balance
- Better reflects real-world distribution

```python
def stratified_sampling(segments_by_class, total_samples=1000):
    """Sample proportional to class sizes."""
    total_available = sum(len(segs) for segs in segments_by_class.values())
    selected = []

    for label_id, segments in segments_by_class.items():
        # Proportional allocation
        class_proportion = len(segments) / total_available
        n_samples = int(total_samples * class_proportion)

        if n_samples > 0:
            if len(segments) >= n_samples:
                selected.extend(random.sample(segments, n_samples))
            else:
                selected.extend(segments)

    return selected
```

**Strategy 3: Minimum Threshold**
- Ensure minimum representation for rare classes
- Cap common classes to avoid overwhelming dataset

```python
def threshold_sampling(segments_by_class, min_per_class=50, max_per_class=500):
    """Sample with min/max constraints per class."""
    selected = []

    for label_id, segments in segments_by_class.items():
        n_available = len(segments)

        if n_available <= min_per_class:
            # Small class: use all segments
            selected.extend(segments)
        elif n_available <= max_per_class:
            # Medium class: sample min_per_class
            selected.extend(random.sample(segments, min_per_class))
        else:
            # Large class: sample max_per_class
            selected.extend(random.sample(segments, max_per_class))

    return selected
```

### 2.3 Default Strategy (Recommended)

**Hybrid Approach:**
1. Use **balanced sampling** for evaluation (fair comparison across classes)
2. Use **stratified sampling** for realistic performance estimation
3. Use **threshold sampling** to handle extreme imbalance

**Default Parameters:**
- Balanced: 100 segments per class
- Minimum: 50 segments per class
- Maximum: 500 segments per class

---

## Part 3: Processing Pipeline

### 3.1 Load SVM Model

```python
def load_svm_model(model_path):
    """
    Load trained SVM model and metadata.

    Args:
        model_path: Path to .pkl model file

    Returns:
        model: Trained SVM model
        metadata: Model configuration
    """
    import joblib
    from pathlib import Path

    # Load model
    model = joblib.load(model_path)

    # Load metadata from parent directory
    model_dir = Path(model_path).parent
    metadata_file = model_dir / 'metadata.json'

    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    return model, metadata
```

### 3.2 Process Single Segment

```python
def process_verification_segment(segment_info, svm_model, reference_segments, config):
    """
    Process one segment through complete pipeline.

    Args:
        segment_info: Dict with segment_id, file_id, label_id, etc.
        svm_model: Trained SVM model
        reference_segments: Dict of reference segment data by label_id
        config: Processing configuration (decimation, amplitude method, etc.)

    Returns:
        result: Dict with predictions and metrics
    """
    import numpy as np
    import time
    from pathlib import Path

    t_start = time.time()

    # Step 1: Load segment data
    segment_data = load_segment_from_file(
        segment_info['file_id'],
        segment_info['segment_offset'],
        segment_info['segment_size']
    )

    # Step 2: Apply amplitude processing
    if config['amplitude_method'] == 1:
        # Raw (no processing)
        processed_data = segment_data
    elif config['amplitude_method'] == 2:
        # Z-score normalization
        processed_data = (segment_data - np.mean(segment_data)) / np.std(segment_data)
    elif config['amplitude_method'] == 3:
        # Min-max scaling
        processed_data = (segment_data - np.min(segment_data)) / (np.max(segment_data) - np.min(segment_data))
    # ... other methods

    # Step 3: Calculate distances to all 13 reference segments
    distances = {}
    for label_id, ref_segment_data in reference_segments.items():
        # Calculate 4 distance metrics
        distances[label_id] = {
            'L1': calculate_l1_distance(processed_data, ref_segment_data),
            'L2': calculate_l2_distance(processed_data, ref_segment_data),
            'cosine': calculate_cosine_distance(processed_data, ref_segment_data),
            'pearson': calculate_pearson_distance(processed_data, ref_segment_data)
        }

    # Step 4: Build feature vector (13 classes × 4 metrics = 52 elements)
    feature_vector = []
    for label_id in sorted(reference_segments.keys()):
        feature_vector.extend([
            distances[label_id]['L1'],
            distances[label_id]['L2'],
            distances[label_id]['cosine'],
            distances[label_id]['pearson']
        ])

    feature_vector = np.array(feature_vector).reshape(1, -1)

    # Step 5: Get SVM prediction
    predicted_label = svm_model.predict(feature_vector)[0]
    prediction_probabilities = svm_model.predict_proba(feature_vector)[0]
    prediction_confidence = np.max(prediction_probabilities)

    # Step 6: Binary arc detection
    true_is_arc = get_label_category(segment_info['label_id']) == 'arc'
    predicted_is_arc = get_label_category(predicted_label) == 'arc'

    # Arc confidence = sum of probabilities for all arc classes
    arc_classes = get_arc_class_indices()
    arc_confidence = np.sum(prediction_probabilities[arc_classes])

    # Step 7: Calculate minimum distances (for analysis)
    min_distances = {
        'L1': min(d['L1'] for d in distances.values()),
        'L2': min(d['L2'] for d in distances.values()),
        'cosine': min(d['cosine'] for d in distances.values()),
        'pearson': min(d['pearson'] for d in distances.values())
    }

    processing_time = time.time() - t_start

    return {
        'segment_id': segment_info['segment_id'],
        'file_id': segment_info['file_id'],
        'true_label_id': segment_info['label_id'],
        'predicted_label_id': predicted_label,
        'prediction_confidence': prediction_confidence,
        'prediction_probabilities': prediction_probabilities.tolist(),
        'true_is_arc': true_is_arc,
        'predicted_is_arc': predicted_is_arc,
        'arc_confidence': arc_confidence,
        'feature_vector': feature_vector[0],
        'min_distances': min_distances,
        'processing_time': processing_time
    }
```

### 3.3 Batch Processing

```python
def process_verification_batch(segments, svm_model, reference_segments, config, workers=12):
    """
    Process multiple segments in parallel.

    Uses multiprocessing.Pool for parallel processing.
    """
    from multiprocessing import Pool
    from functools import partial

    # Create partial function with fixed parameters
    process_func = partial(
        process_verification_segment,
        svm_model=svm_model,
        reference_segments=reference_segments,
        config=config
    )

    results = []
    with Pool(processes=workers) as pool:
        for i, result in enumerate(pool.imap_unordered(process_func, segments), 1):
            results.append(result)
            if i % 100 == 0:
                print(f"[INFO] Processed {i}/{len(segments)} segments...")

    return results
```

---

## Part 4: Command Implementation

### 4.1 Command: classifier-complete-verification

**Usage:**
```bash
classifier-complete-verification [OPTIONS]

Options:
  --model-config <config>       Model configuration to test
                                Format: dec,dtype,amp,efs,kernel,C
                                Example: 0,6,2,1,linear,1.0

  --segments-per-class <n>      Number of segments per class (default: 100)
  --strategy <name>             Sampling strategy: balanced, stratified, threshold
                                (default: balanced)

  --min-per-class <n>           Minimum segments per class (default: 50)
  --max-per-class <n>           Maximum segments per class (default: 500)

  --include-test                Include test set segments (default: true)
  --include-verification        Include verification set segments (default: true)
  --include-other               Include other non-training segments (default: true)

  --workers <n>                 Number of parallel workers (default: 12)
  --batch-size <n>              Segments per batch (default: 1000)

  --output-dir <path>           Output directory for results
  --save-features               Save feature vectors to disk
  --generate-plots              Generate visualization plots (default: true)

  --force                       Overwrite existing results
  --verbose                     Show detailed progress
```

**Examples:**
```bash
# Test best model with balanced sampling
classifier-complete-verification --model-config 0,6,2,1,linear,1.0 --segments-per-class 100

# Test with threshold strategy for imbalanced dataset
classifier-complete-verification --model-config 0,6,2,1,linear,1.0 --strategy threshold --min-per-class 50 --max-per-class 500

# Test only on unseen segments (exclude test/verification)
classifier-complete-verification --model-config 0,6,2,1,linear,1.0 --include-test false --include-verification false

# Quick test with fewer segments
classifier-complete-verification --model-config 0,6,2,1,linear,1.0 --segments-per-class 20
```

### 4.2 Implementation Skeleton

```python
def cmd_classifier_complete_verification(self, args):
    """
    Perform complete verification on unseen segments.

    Tests trained SVM model on segments not used in training.
    """
    if not self.db_conn:
        print("[ERROR] Not connected to database")
        return

    if not self.current_experiment:
        print("[ERROR] No experiment selected")
        return

    if not self.current_classifier_id:
        print("[ERROR] No classifier selected")
        return

    # Parse arguments
    model_config = parse_model_config(args)
    segments_per_class = get_arg(args, '--segments-per-class', 100)
    strategy = get_arg(args, '--strategy', 'balanced')
    workers = get_arg(args, '--workers', 12)
    # ... parse all options

    # Step 1: Load SVM model
    print(f"\n[INFO] Loading SVM model...")
    model_path = get_model_path(model_config)
    svm_model, model_metadata = load_svm_model(model_path)
    print(f"[INFO] Model loaded: {model_path}")

    # Step 2: Load reference segments
    print(f"\n[INFO] Loading reference segments...")
    reference_segments = load_reference_segments(model_config)
    print(f"[INFO] Loaded {len(reference_segments)} reference segments")

    # Step 3: Select verification segments
    print(f"\n[INFO] Selecting verification segments...")
    all_segments = query_available_segments(
        include_test=include_test,
        include_verification=include_verification,
        include_other=include_other
    )

    selected_segments = apply_sampling_strategy(
        all_segments,
        strategy=strategy,
        segments_per_class=segments_per_class,
        min_per_class=min_per_class,
        max_per_class=max_per_class
    )

    print(f"[INFO] Selected {len(selected_segments)} segments")
    print_class_distribution(selected_segments)

    # Step 4: Process segments
    print(f"\n[INFO] Processing segments with {workers} workers...")
    results = process_verification_batch(
        selected_segments,
        svm_model,
        reference_segments,
        model_config,
        workers=workers
    )

    # Step 5: Insert into database
    print(f"\n[INFO] Inserting results into database...")
    insert_verification_results(results, model_config)

    # Step 6: Generate metrics
    print(f"\n[INFO] Calculating metrics...")
    metrics = calculate_verification_metrics(results)
    print_metrics_summary(metrics)

    # Step 7: Generate plots
    if generate_plots:
        print(f"\n[INFO] Generating plots...")
        create_verification_plots(results, metrics, output_dir)

    print(f"\n[SUCCESS] Complete verification finished!")
    print(f"[INFO] Results saved to: experiment_041_classifier_001_complete_verification")
```

---

## Part 5: Metrics and Visualization

### 5.1 Metrics to Calculate

**Overall Metrics:**
- Overall accuracy
- Macro-averaged precision/recall/F1
- Weighted precision/recall/F1
- Binary arc detection accuracy

**Per-Class Metrics:**
```python
for each class:
    - Precision
    - Recall
    - F1-score
    - Support (number of samples)
    - True positives
    - False positives
    - False negatives
```

**Confusion Matrix:**
- 13×13 matrix for all classes
- 2×2 matrix for binary arc detection

**Confidence Analysis:**
- Mean confidence for correct predictions
- Mean confidence for incorrect predictions
- Confidence distribution by class

### 5.2 Visualizations

**1. Confusion Matrix (13-class)**
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, label_names, output_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Complete Verification - Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
```

**2. Per-Class Performance**
```python
def plot_per_class_performance(metrics, label_names, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Precision
    axes[0, 0].bar(label_names, metrics['precision_per_class'])
    axes[0, 0].set_title('Precision by Class')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Recall
    axes[0, 1].bar(label_names, metrics['recall_per_class'])
    axes[0, 1].set_title('Recall by Class')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # F1-Score
    axes[1, 0].bar(label_names, metrics['f1_per_class'])
    axes[1, 0].set_title('F1-Score by Class')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Support
    axes[1, 1].bar(label_names, metrics['support_per_class'])
    axes[1, 1].set_title('Support (Number of Samples) by Class')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
```

**3. ROC Curves (One-vs-Rest)**
```python
def plot_roc_curves(y_true, y_proba, label_names, output_path):
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(label_names)))

    plt.figure(figsize=(12, 10))

    for i, label_name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - One-vs-Rest')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(output_path, dpi=300)
    plt.close()
```

**4. Confidence Distribution**
```python
def plot_confidence_distribution(results, output_path):
    correct_confidences = [r['prediction_confidence']
                          for r in results
                          if r['predicted_label_id'] == r['true_label_id']]

    incorrect_confidences = [r['prediction_confidence']
                            for r in results
                            if r['predicted_label_id'] != r['true_label_id']]

    plt.figure(figsize=(12, 6))
    plt.hist(correct_confidences, bins=50, alpha=0.7, label='Correct', color='green')
    plt.hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300)
    plt.close()
```

---

## Part 6: Implementation Timeline

### Phase 1: Core Infrastructure (8-10 hours)

1. **Database schema** (2 hours)
   - Create tables
   - Add indexes
   - Test insertions

2. **Segment selection** (3 hours)
   - Query available segments
   - Implement sampling strategies
   - Test class distribution

3. **Processing pipeline** (3 hours)
   - Load segment data
   - Apply amplitude processing
   - Calculate distances
   - Build feature vectors

4. **SVM prediction** (2 hours)
   - Load model
   - Get predictions
   - Store results

### Phase 2: Command Implementation (6-8 hours)

1. **Argument parsing** (2 hours)
2. **Main workflow** (3 hours)
3. **Error handling** (1 hour)
4. **Progress reporting** (2 hours)

### Phase 3: Metrics and Visualization (4-6 hours)

1. **Metrics calculation** (2 hours)
2. **Plot generation** (3 hours)
3. **Report generation** (1 hour)

### Total Estimate: 18-24 hours

---

## Part 7: Testing Strategy

### Test 1: Small Scale
```bash
# Test with 10 segments per class
classifier-complete-verification --model-config 0,6,2,1,linear,1.0 --segments-per-class 10
```
**Expected:** Completes in ~5 minutes, produces valid metrics

### Test 2: Single Class
```bash
# Test with only one class
classifier-complete-verification --model-config 0,6,2,1,linear,1.0 --class-filter 2
```
**Expected:** Verifies pipeline for single class

### Test 3: Full Scale
```bash
# Test with 100 segments per class
classifier-complete-verification --model-config 0,6,2,1,linear,1.0 --segments-per-class 100
```
**Expected:** Completes in ~30-60 minutes, comprehensive results

---

## Summary

**Purpose:** Test SVM generalization on truly unseen segments

**Key Features:**
- Flexible segment selection (handles class imbalance)
- Full processing pipeline (scale → distance → predict)
- Comprehensive metrics and visualizations
- Dedicated database storage

**Deliverables:**
1. `classifier-complete-verification` command
2. Database tables for storing results
3. Automated metrics calculation
4. Visualization plots
5. Detailed documentation

**Next Steps:**
1. Review and approve plan
2. Implement database schema
3. Implement segment selection
4. Implement processing pipeline
5. Add command to shell
6. Test and validate
