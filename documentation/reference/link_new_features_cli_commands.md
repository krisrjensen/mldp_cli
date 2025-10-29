# CLI Commands for Linking New Feature Sets to Experiment

**Purpose:** Commands to link the 6 new feature sets (IDs 15-20) to your new experiment
**Date Created:** 2025-10-29
**Status:** Ready for use when new experiment is created

---

## Prerequisites

1. Create your new experiment first
2. Note the experiment ID (e.g., experiment 42, 43, etc.)
3. Ensure you have the feature sets registered (IDs 15-20)

---

## Feature Sets Available

| Feature Set ID | Name | Features | Description |
|---------------|------|----------|-------------|
| 15 | derivative_volatility | 6 | Derivative volatility (n=1,2,3) |
| 16 | stft_basic | 8 | Basic STFT scalar aggregates |
| 17 | stft_volatility_composite | 4 | STFT applied to volatility signals |
| 18 | pink_noise_stft_tmr | 12 | TMR paper pink noise analysis |
| 19 | pink_noise_stft_bandpower | 8 | Pink noise band power method |
| 20 | new_features_comprehensive | 64 | All 64 new features combined |

---

## SQL Commands (Direct Database Access)

Replace `<EXPERIMENT_ID>` with your new experiment ID.

### Option 1: Link All 6 Feature Sets (Multi-Channel)

```sql
-- Link all 6 feature sets to both channels at once
INSERT INTO ml_experiments_feature_sets
(experiment_id, feature_set_id, data_channel, n_value, priority_order)
VALUES
(<EXPERIMENT_ID>, 15, 'source_current,load_voltage', NULL, 20),  -- derivative_volatility
(<EXPERIMENT_ID>, 16, 'source_current,load_voltage', NULL, 21),  -- stft_basic
(<EXPERIMENT_ID>, 17, 'source_current,load_voltage', NULL, 22),  -- stft_volatility_composite
(<EXPERIMENT_ID>, 18, 'source_current,load_voltage', NULL, 23),  -- pink_noise_stft_tmr
(<EXPERIMENT_ID>, 19, 'source_current,load_voltage', NULL, 24),  -- pink_noise_stft_bandpower
(<EXPERIMENT_ID>, 20, 'source_current,load_voltage', NULL, 25);  -- new_features_comprehensive
```

### Option 2: Link Only Comprehensive Set (All 64 Features)

```sql
-- Link only the comprehensive set containing all 64 features
INSERT INTO ml_experiments_feature_sets
(experiment_id, feature_set_id, data_channel, n_value, priority_order)
VALUES
(<EXPERIMENT_ID>, 20, 'source_current,load_voltage', NULL, 20);  -- new_features_comprehensive
```

### Option 3: Link Selective Feature Sets

```sql
-- Example: Link only derivative and STFT features
INSERT INTO ml_experiments_feature_sets
(experiment_id, feature_set_id, data_channel, n_value, priority_order)
VALUES
(<EXPERIMENT_ID>, 15, 'source_current,load_voltage', NULL, 20),  -- derivative_volatility
(<EXPERIMENT_ID>, 16, 'source_current,load_voltage', NULL, 21);  -- stft_basic
```

---

## Execution Examples

### For Experiment 42:

```bash
# Connect to database
/opt/homebrew/opt/postgresql@15/bin/psql -h localhost -U kjensen -d arc_detection

# Link all 6 feature sets
INSERT INTO ml_experiments_feature_sets
(experiment_id, feature_set_id, data_channel, n_value, priority_order)
VALUES
(42, 15, 'source_current,load_voltage', NULL, 20),
(42, 16, 'source_current,load_voltage', NULL, 21),
(42, 17, 'source_current,load_voltage', NULL, 22),
(42, 18, 'source_current,load_voltage', NULL, 23),
(42, 19, 'source_current,load_voltage', NULL, 24),
(42, 20, 'source_current,load_voltage', NULL, 25);
```

---

## Verification Commands

### Check Feature Sets Linked to Your Experiment

```sql
SELECT
    efs.experiment_feature_set_id,
    efs.experiment_id,
    fsl.feature_set_name,
    efs.data_channel,
    efs.priority_order,
    COUNT(fsf.feature_id) as feature_count
FROM ml_experiments_feature_sets efs
JOIN ml_feature_sets_lut fsl ON efs.feature_set_id = fsl.feature_set_id
LEFT JOIN ml_feature_set_features fsf ON fsl.feature_set_id = fsf.feature_set_id
WHERE efs.experiment_id = <EXPERIMENT_ID>
GROUP BY efs.experiment_feature_set_id, efs.experiment_id, fsl.feature_set_name,
         efs.data_channel, efs.priority_order
ORDER BY efs.priority_order;
```

### Count Total Features Available

```sql
-- For experiment ID <EXPERIMENT_ID>
SELECT
    'Total feature sets linked' as metric,
    COUNT(DISTINCT efs.feature_set_id) as count
FROM ml_experiments_feature_sets efs
WHERE efs.experiment_id = <EXPERIMENT_ID>

UNION ALL

SELECT
    'Total features available',
    COUNT(DISTINCT fsf.feature_id)
FROM ml_experiments_feature_sets efs
JOIN ml_feature_set_features fsf ON efs.feature_set_id = fsf.feature_set_id
WHERE efs.experiment_id = <EXPERIMENT_ID>;
```

---

## Important Notes

1. **Multi-Channel Linking**: The database has a unique constraint on `(experiment_id, feature_set_id)`, so you can only link each feature set once per experiment. Use the multi-channel value `'source_current,load_voltage'` to extract features from both channels.

2. **Priority Order**: The priority_order values (20-25) determine the order of feature extraction. These are suggestions - adjust as needed for your experiment.

3. **Feature Count**:
   - Using all 6 sets individually: You'll extract features multiple times (some overlap)
   - Using comprehensive set (ID 20): Single extraction with all 64 features
   - **Recommendation**: Start with the comprehensive set for simplicity

4. **CLI Integration**: Once linked, the feature extraction commands in mldp_cli will automatically use these feature sets when you run feature extraction for your experiment.

---

## After Linking

Once you've linked the feature sets to your experiment, you can:

1. **Extract features** using the CLI feature extraction commands
2. **Verify extraction** by checking the `experiment_<ID>_feature_fileset` table
3. **Train classifiers** using the new features with SVM/RF training commands

---

## Example Workflow

```bash
# 1. Create new experiment (in CLI)
# experiment-create --name "New Features Test" --description "Testing 64 new features"

# 2. Link feature sets (in psql)
# INSERT INTO ml_experiments_feature_sets ... (see above)

# 3. Extract features (in CLI)
# feature-extract --experiment-id <ID> --segment-ids 1-100

# 4. Train classifier (in CLI)
# train-svm --experiment-id <ID> --classifier-id <ID>
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-10-29
