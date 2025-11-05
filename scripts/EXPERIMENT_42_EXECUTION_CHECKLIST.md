# Experiment 42 Execution Checklist

**Author:** Kristophor Jensen
**Date:** 20251104
**Objective:** Feature separability analysis with 100 random feature sets

---

## Quick Reference

**Total Time:** ~12-18 hours
**Storage Required:** ~120-150 GB
**Feature Sets:** Create 560, execute on 100 (randomly selected, seed=42)

---

## Pre-Flight Checks

- [ ] Verify 150+ GB free disk space
- [ ] Database accessible (arc_detection@localhost)
- [ ] MLDP CLI running (v2.0.11.2+)
- [ ] 16 PSD features exist (IDs 95-110)
- [ ] Current directory: `/Users/kjensen/Documents/GitHub/mldp`

---

## Execution Steps

### Step 1: Create All 560 Feature Sets (~10-15 min)
```bash
cd mldp_cli/scripts
python3 generate_exp42_feature_sets.py > create_exp42_560_feature_sets.sh
```

**In MLDP CLI:**
```bash
source mldp_cli/scripts/create_exp42_560_feature_sets.sh
```

**Verify:**
```bash
show-all-feature-sets | grep exp42_separability | wc -l
# Expected: 560
```

- [ ] 560 feature sets created

---

### Step 2: Link 100 Random Feature Sets & Configure Experiment (~1 min)

**Python script to link feature sets (already completed in database):**
```bash
cd mldp_cli/scripts
python3 link_100_random_feature_sets.py
```

**Verify:**
```bash
# In PostgreSQL
psql -h localhost -U kjensen -d arc_detection -c "SELECT COUNT(*) FROM ml_experiments_feature_sets WHERE experiment_id = 42"
# Expected: 100
```

- [x] 100 feature sets randomly selected and linked to experiment 42

---

### Step 3: Complete Experiment Configuration (~5-10 min)

**In MLDP CLI:**
```bash
source mldp_cli/scripts/setup_experiment_42.sh
```

This script will:
- Set current experiment to 42
- Configure segment selection (random seed 42, balanced, position-balanced)
- Select 150 training files (50 per label)
- Select segments with position balancing
- Add 4 data types (adc6, adc8, adc10, adc12)
- Add 6 decimations (0, 7, 15, 31, 64, 128)
- Add 2 distance metrics (L1, Cosine)

**Verify:**
```bash
show
list-data-types
# Expected: adc6, adc8, adc10, adc12
```

- [ ] Experiment configuration complete
- [ ] 150 files selected (50 per label)
- [ ] ~1000 segments selected (position-balanced)
- [ ] 4 data types added
- [ ] 6 decimations added
- [ ] 2 distance metrics added

---

### Step 4: Generate Segment Fileset (~10-15 min)

```bash
generate-segment-fileset
```

**Note:** Z-score amplitude normalization will be applied automatically.

- [ ] ~24,000 segment files generated (1000 segments × 24 combinations)

---

### Step 5: Generate Feature Fileset (~30-45 min)

```bash
generate-feature-fileset --scaling zscore
```

- [ ] Features extracted with z-score normalization

---

### Step 6: Generate Segment Pairs

```bash
generate-segment-pairs
```

- [ ] ~499,500 pairs generated

---

### Step 7: Compute Distances (~8-12 hours)

```bash
mpcctl-distance-function --workers 20 --log --verbose
```

**Monitor:**
- Check log files in `logs/` directory
- 2.4 billion distance calculations
- 4,800 work files (100 feature sets × 24 data combos × 2 metrics)

- [ ] Distance computation complete

---

### Step 8: Insert Distances (~2-4 hours)

```bash
mpcctl-distance-insert --workers 10 --log --verbose
```

**Monitor:**
- Inserting ~2.4 billion rows
- Tables: `experiment_042_distance_l1`, `experiment_042_distance_cosine`

- [ ] Distance insertion complete

---

### Step 9: Generate Heatmaps (~1-2 hours)

```bash
heatmap --output-dir plots/experiment_42
```

- [ ] 4,800 heatmap images generated

---

## Verification

### Final Checks
```bash
# Feature sets
list-feature-sets | wc -l
# Expected: 100

# Files
# (command TBD - check file count)
# Expected: 150

# Segments
# (command TBD - check segment count)
# Expected: ~1000

# Distance rows
# (SQL query TBD)
# Expected: ~2.4 billion total
```

---

## Troubleshooting

### If feature set creation fails:
- Check database connectivity
- Verify features 95-110 exist
- Check for duplicate feature set names

### If random selection returns wrong count:
- Verify all 560 feature sets were created
- Check database query in `select_100_random_feature_sets.py`

### If distance computation stalls:
- Check disk space (need ~24 GB for work files)
- Monitor system resources (CPU, memory)
- Check log files for errors
- Consider reducing worker count if memory constrained

### If distance insertion fails:
- Check database disk space (~120 GB needed)
- Verify work files exist
- Check database connection stability

---

## Timeline

| Step | Duration | Cumulative |
|------|----------|------------|
| 1. Create 560 feature sets | 10-15 min | 0:15 |
| 2. Link 100 random feature sets | 1 min | 0:16 |
| 3. Complete configuration (script) | 5-10 min | 0:26 |
| 4. Generate segments | 10-15 min | 0:41 |
| 5. Generate features | 30-45 min | 1:26 |
| 6. Generate pairs | 5 min | 1:31 |
| 7. Compute distances | 8-12 hrs | 13:31 |
| 8. Insert distances | 2-4 hrs | 17:31 |
| 9. Generate heatmaps | 1-2 hrs | 19:31 |

**Total:** ~12-18 hours (19.5 hours worst case)

---

## Success Criteria

- [x] All 560 feature sets created in database
- [x] 100 feature sets randomly selected and linked
- [x] Experiment 42 created in database
- [x] Z-score amplitude method configured
- [ ] 150 files selected (balanced across 3 labels)
- [ ] ~1000 segments selected (position-balanced)
- [ ] Data types and decimations configured
- [ ] Distance metrics configured
- [ ] 24,000 segment files generated
- [ ] Features extracted with z-score scaling
- [ ] 499,500 segment pairs generated
- [ ] 2.4 billion distances computed
- [ ] 2.4 billion distances inserted to database
- [ ] 4,800 heatmap images generated
- [ ] No errors in log files
- [ ] Database tables populated correctly

---

## Next Steps After Completion

1. **Analyze heatmaps** - Identify best-performing feature sets
2. **Statistical analysis** - Rank features by separability metrics
3. **Feature selection** - Choose top 3-5 features for classifier
4. **Documentation** - Record findings and recommendations
5. **Classifier training** - Use selected features for production model

---

## Notes

- Random seed: 42 (reproducible)
- All 560 feature sets remain in database for future use
- Can expand to additional random samples if needed
- Can target specific feature sets based on results
