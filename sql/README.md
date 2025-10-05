# SQL Schema Updates for MLDP CLI

## Update ml_distance_functions_lut for sklearn.metrics.pairwise_distances and POT

**File:** `update_distance_functions_lut.sql`

This script updates the `ml_distance_functions_lut` table to support `sklearn.metrics.pairwise_distances`, Python Optimal Transport (POT), and custom implementations.

---

## What It Does

1. Adds `pairwise_metric_name` column to `ml_distance_functions_lut`
2. Updates all 17 distance functions to use appropriate libraries:
   - **10 functions** → sklearn.metrics.pairwise_distances
   - **1 function** → POT (Python Optimal Transport)
   - **6 functions** → Custom implementations

---

## How To Run

### Option 1: Using Python Script (Recommended)

**If psql is not in your PATH, use the Python script:**

```bash
python /Users/kjensen/Documents/GitHub/mldp/mldp_cli/sql/run_update.py
```

Or from your virtual environment:

```bash
/Users/kjensen/Documents/GitHub/mldp/.venv/bin/python /Users/kjensen/Documents/GitHub/mldp/mldp_cli/sql/run_update.py
```

**This script:**
- Connects to PostgreSQL using psycopg2
- Executes all UPDATE statements
- Displays verification results
- Handles errors gracefully

---

### Option 2: Using psql Command Line

**If psql is in your PATH:**

```bash
psql -h localhost -p 5432 -d arc_detection -f /Users/kjensen/Documents/GitHub/mldp/mldp_cli/sql/update_distance_functions_lut.sql
```

**If you need to provide a username:**
```bash
psql -h localhost -p 5432 -d arc_detection -U kjensen -f /Users/kjensen/Documents/GitHub/mldp/mldp_cli/sql/update_distance_functions_lut.sql
```

---

### Option 3: Using psql Interactive

```bash
# Connect to database
psql -h localhost -p 5432 -d arc_detection

# Run the script
\i /Users/kjensen/Documents/GitHub/mldp/mldp_cli/sql/update_distance_functions_lut.sql

# Exit
\q
```

---

## Verification

After running the script, verify the changes in MLDP CLI:

```bash
# Launch MLDP CLI
python -m mldp_cli.main

# Connect to database
connect

# View distance functions
show-distance-functions
```

**Expected output:**
- No warning about missing pairwise_metric_name column
- manhattan shows pairwise_metric_name: 'manhattan'
- euclidean shows pairwise_metric_name: 'euclidean'
- cosine shows pairwise_metric_name: 'cosine'
- pearson shows pairwise_metric_name: 'correlation'

---

## Troubleshooting

### Error: "permission denied"

**Solution:** Add your username to the psql command:
```bash
psql -h localhost -p 5432 -d arc_detection -U kjensen -f update_distance_functions_lut.sql
```

### Error: "psql: command not found"

**Solution 1 (Recommended):** Use the Python script instead:
```bash
python /Users/kjensen/Documents/GitHub/mldp/mldp_cli/sql/run_update.py
```

**Solution 2:** Find your PostgreSQL installation:
```bash
# Find psql
find /Applications -name psql 2>/dev/null

# Or use full path
/Applications/Postgres.app/Contents/Versions/latest/bin/psql -h localhost -p 5432 -d arc_detection -f update_distance_functions_lut.sql
```

### Error: "relation is locked"

**Solution:** Close other connections to the database:
```bash
# In psql, find blocking processes
SELECT pid, usename, state, query FROM pg_stat_activity WHERE datname = 'arc_detection';

# Terminate if safe
SELECT pg_terminate_backend(pid);
```

### Error: "column already exists"

**Solution:** The script already ran successfully! Verify with:
```bash
# In MLDP CLI
show-distance-functions
```

---

## What Changes

### Before:

All 17 distance functions had inconsistent or missing library configurations.

### After:

**sklearn.metrics.pairwise_distances (10 functions):**

| Function | pairwise_metric_name |
|----------|----------------------|
| manhattan | manhattan |
| euclidean | euclidean |
| cosine | cosine |
| pearson | correlation |
| braycurtis | braycurtis |
| canberra | canberra |
| chebyshev | chebyshev |
| sqeuclidean | sqeuclidean |
| mahalanobis | mahalanobis |
| jensenshannon | jensenshannon |

**POT (Python Optimal Transport) (1 function):**

| Function | library_name | function_import | pairwise_metric_name |
|----------|--------------|-----------------|----------------------|
| wasserstein | pot | ot.wasserstein_1d | wasserstein_1d |

**Custom implementations (6 functions):**

| Function | library_name | function_import |
|----------|--------------|-----------------|
| fidelity | custom | custom_distance_engine |
| kullback_leibler | custom | custom_distance_engine |
| kumar_hassebrook | custom | custom_distance_engine |
| additive_symmetric | custom | custom_distance_engine |
| taneja | custom | custom_distance_engine |
| wavehedges | custom | custom_distance_engine |

---

## Why This Matters

The new distance calculator (`mpcctl_cli_distance_calculator.py`) uses a database-driven approach to distance calculation:

1. **sklearn.metrics.pairwise_distances** - Unified interface for 10 common metrics
2. **POT (Python Optimal Transport)** - Specialized library for Wasserstein distance
3. **Custom implementations** - Existing implementations for specialized metrics

This provides:
- **Unified interface** - Single function call for most metrics
- **Better performance** - Optimized implementations from sklearn/POT
- **Simpler code** - Database-driven routing instead of hardcoded logic
- **Easier maintenance** - Update database, not code
- **Library flexibility** - Can use sklearn, POT, or custom as needed

**Without this update:** The distance calculator won't know which library/metric to use.

**With this update:** The calculator reads `library_name`, `function_import`, and `pairwise_metric_name` from the database and routes to the correct implementation.

---

## Next Steps

After running this script:

1. Verify with `show-distance-functions` in MLDP CLI
2. Ready for Session 3: Creating the actual distance calculator
3. Distance calculator will read pairwise_metric_name from database

---

**If you see the warning in MLDP CLI, run this script!**
