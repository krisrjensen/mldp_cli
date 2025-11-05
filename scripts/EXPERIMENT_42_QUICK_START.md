# Experiment 42 Quick Start Guide

**Created:** 20251104
**Status:** Ready for Execution
**Estimated Time:** 12-18 hours total

---

## What's Already Complete

✅ **Database Setup (Completed)**
- Experiment 42 created in database
- 560 feature sets created (all C(16,3) combinations)
- 100 feature sets randomly selected and linked (seed=42)
- Z-score amplitude normalization method configured

---

## What You Need To Do

### Step 1: Run Configuration Script (~5-10 min)

Open MLDP CLI and run:

```bash
source mldp_cli/scripts/setup_experiment_42.sh
```

This will automatically:
- Set current experiment to 42
- Configure segment selection settings
- Select 150 training files (50 per label)
- Select segments with position balancing
- Add 4 data types (adc6, adc8, adc10, adc12)
- Add 6 decimations (0, 7, 15, 31, 64, 128)
- Add 2 distance metrics (L1, Cosine)

### Step 2: Generate Data & Compute Distances (~12-18 hours)

In MLDP CLI, run these commands in sequence:

```bash
# Generate segment files (~10-15 min, ~1.6 GB)
generate-segment-fileset

# Generate feature files (~30-45 min, ~31 MB)
generate-feature-fileset --scaling zscore

# Generate segment pairs (~1 min, ~499,500 pairs)
generate-segment-pairs

# Compute distances (~8-12 hours, 2.4B calculations)
mpcctl-distance-function --workers 20 --log --verbose

# Insert distances to database (~2-4 hours, 2.4B rows)
mpcctl-distance-insert --workers 10 --log --verbose

# Generate heatmaps (~1-2 hours, 4,800 images)
heatmap --output-dir plots/experiment_42
```

---

## Quick Reference

**Experiment Details:**
- ID: 42
- Name: exp42
- Type: separability_analysis
- Feature sets: 100 (randomly selected from 560)
- Files: 150 (50 per label)
- Segments: ~1000 (position-balanced)
- Data combinations: 24 (6 decimations × 4 data types)
- Distance metrics: 2 (L1, Cosine)

**Resource Requirements:**
- Storage: ~120-150 GB
- Time: ~12-18 hours
- CPU: 20 cores recommended
- Memory: 32-64 GB recommended

**Files:**
- Setup script: `mldp_cli/scripts/setup_experiment_42.sh`
- Full checklist: `mldp_cli/scripts/EXPERIMENT_42_EXECUTION_CHECKLIST.md`
- Implementation plan: `documentation/wip/experiment_42_implementation_plan.md`
- Summary: `documentation/wip/experiment_42_summary.md`

**Scripts Used:**
- `generate_exp42_feature_sets.py` - Generated 560 feature sets
- `link_100_random_feature_sets.py` - Randomly selected and linked 100
- `setup_experiment_42.sh` - Complete configuration automation

---

## Verification Commands

After running the setup script:

```bash
# Check current experiment
show

# Verify feature sets
list-feature-sets | wc -l
# Expected: 100

# Check data types
list-data-types
# Expected: adc6, adc8, adc10, adc12

# Check experiment configuration
show-experiment-config
```

---

## Monitoring Long-Running Processes

**Distance computation and insertion:**
- Check log files in `logs/` directory
- Monitor system resources (CPU, memory, disk I/O)
- Expected work files: 4,800 (100 feature sets × 24 data combos × 2 metrics)

**Database tables:**
- `experiment_042_distance_l1` - L1 distance measurements
- `experiment_042_distance_cosine` - Cosine distance measurements
- Each table will contain ~1.2 billion rows

---

## What Happens Next

After all steps complete:
1. Analyze heatmaps to identify best-performing feature sets
2. Statistical analysis to rank features by separability
3. Feature selection for classifier training
4. Document findings and recommendations

---

## Need Help?

- Full checklist: `EXPERIMENT_42_EXECUTION_CHECKLIST.md`
- Implementation details: `documentation/wip/experiment_42_implementation_plan.md`
- Troubleshooting guide in checklist (lines 194-215)
