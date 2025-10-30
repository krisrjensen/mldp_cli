# Noise Floor Scaler - CLI Command Reference

**Document Version:** 1.0.0
**Date Created:** 2025-10-29
**Purpose:** User guide for noise floor CLI commands

---

## Quick Start

```bash
# 1. Initialize database table
noise-floor-init

# 2. Calculate noise floors for all data types
noise-floor-calculate --all

# 3. View calculated values
noise-floor-show

# 4. Link to experiment (in psql)
# INSERT INTO ml_experiments_amplitude_methods (experiment_id, method_id)
# VALUES (<YOUR_EXP_ID>, 8);

# 5. Extract features - noise floor normalization applied automatically
```

---

## Commands

### noise-floor-init

**Description:** Initialize the experiment_noise_floor database table.

**Syntax:**
```bash
noise-floor-init
```

**Arguments:** None

**Behavior:**
- Checks if `experiment_noise_floor` table exists
- Creates table if it doesn't exist
- Reports status (created or already exists)
- Shows current entry count if table exists

**Example Output:**

If table doesn't exist:
```bash
> noise-floor-init
✓ Created experiment_noise_floor table
```

If table already exists:
```bash
> noise-floor-init
✓ experiment_noise_floor table already exists
  Current entries: 3
```

**When to Use:**
- First time setup
- After database reset
- To verify table exists before calculation

---

### noise-floor-show

**Description:** Display all calculated noise floor values.

**Syntax:**
```bash
noise-floor-show
```

**Arguments:** None

**Output Format:**
```
Noise Floor Values:
--------------------------------------------------------------------------------
Data Type    ID   Noise Floor     Segments   Calculated           Method
--------------------------------------------------------------------------------
ADC6         3    0.000234        189        2025-10-29 14:23     spectral_psd
ADC8         1    0.000156        245        2025-10-29 14:25     spectral_psd
ADC10        2    0.000089        312        2025-10-29 14:28     spectral_psd
```

**Columns:**
- **Data Type**: Name from ml_data_types_lut (ADC6, ADC8, ADC10, etc.)
- **ID**: data_type_id (primary key)
- **Noise Floor**: Calculated noise floor value in RMS units
- **Segments**: Number of segments used in calculation
- **Calculated**: Timestamp of calculation (YYYY-MM-DD HH:MM)
- **Method**: Calculation method used (spectral_psd)

**Empty Database:**
```bash
> noise-floor-show
No noise floor values calculated yet
Run 'noise-floor-calculate' to compute values
```

**When to Use:**
- After calculation to verify values
- Before feature extraction to check availability
- To see when values were last updated

---

### noise-floor-calculate

**Description:** Calculate noise floor values from approved steady-state segments using spectral PSD method.

**Syntax:**
```bash
noise-floor-calculate              # Calculate for all data types
noise-floor-calculate --all        # Same as above
noise-floor-calculate --type <data_type_id>  # Calculate for specific type
```

**Arguments:**
- **None / --all**: Calculate for all data types with approved segments
- **--type <data_type_id>**: Calculate for specific data type ID only

**Requires:** User confirmation (type 'y' to proceed)

**Process:**
1. Prompts user for confirmation
2. Queries approved steady-state segments from database
3. Filters for segments with status=true from files with status=true
4. Loads segment data from fileset/ or adc_data/ folders
5. Computes PSD using Welch's method
6. Calculates noise floor as 10th percentile of PSD (RMS units)
7. Averages across all segments
8. Stores/updates value in database
9. Displays results with noise-floor-show

**Example - Calculate All:**
```bash
> noise-floor-calculate --all

Calculate noise floors for ALL data types? [y/N]: y

Calculating noise floors for all data types...
  ADC6: Found 189 approved steady-state segments
  ADC6: Processing segments... [████████████████████] 100%
  ADC6: Noise floor = 0.000234 RMS

  ADC8: Found 245 approved steady-state segments
  ADC8: Processing segments... [████████████████████] 100%
  ADC8: Noise floor = 0.000156 RMS

  ADC10: Found 312 approved steady-state segments
  ADC10: Processing segments... [████████████████████] 100%
  ADC10: Noise floor = 0.000089 RMS

✓ Calculated 3 noise floor values

Noise Floor Values:
--------------------------------------------------------------------------------
Data Type    ID   Noise Floor     Segments   Calculated           Method
--------------------------------------------------------------------------------
ADC6         3    0.000234        189        2025-10-29 14:35     spectral_psd
ADC8         1    0.000156        245        2025-10-29 14:35     spectral_psd
ADC10        2    0.000089        312        2025-10-29 14:35     spectral_psd
```

**Example - Calculate Specific:**
```bash
> noise-floor-calculate --type 1

Calculate noise floor for data_type_id 1? [y/N]: y

Calculating noise floor for data_type_id 1...
  Found 245 approved steady-state segments
  Processing segments: [████████████████████] 100%
  Noise floor: 0.000156 RMS
✓ Noise floor: 0.000156

Noise Floor Values:
--------------------------------------------------------------------------------
Data Type    ID   Noise Floor     Segments   Calculated           Method
--------------------------------------------------------------------------------
ADC8         1    0.000156        245        2025-10-29 14:38     spectral_psd
```

**Cancellation:**
```bash
> noise-floor-calculate

Calculate noise floors for ALL data types? [y/N]: n
Cancelled
```

**When to Use:**
- Initial setup after collecting steady-state data
- After adding new data types
- To update values with more segments
- After system changes (new ADC, different wiring, etc.)

**Can Run Multiple Times:** Yes, updates existing entries

---

### noise-floor-clear

**Description:** Clear noise floor entries from database.

**Syntax:**
```bash
noise-floor-clear                           # Show help
noise-floor-clear --all                     # Clear all entries
noise-floor-clear --type <data_type_id>     # Clear specific type
```

**Arguments:**
- **None**: Displays help message (no action)
- **--all**: Clears all noise floor entries (requires confirmation)
- **--type <data_type_id>**: Clears specific data type entry (requires confirmation)

**Requires:** User confirmation for all delete operations

**Example - Show Help:**
```bash
> noise-floor-clear
Usage:
  noise-floor-clear --all              Clear all noise floor entries
  noise-floor-clear --type <id>        Clear specific data type
```

**Example - Clear All:**
```bash
> noise-floor-clear --all

Current noise floor entries:
  ADC6: 0.000234 (189 segments)
  ADC8: 0.000156 (245 segments)
  ADC10: 0.000089 (312 segments)

Delete all 3 entries? [y/N]: y
✓ Cleared 3 entries
```

**Example - Clear Specific:**
```bash
> noise-floor-clear --type 1

Entry to delete:
  ADC8 (ID 1): 0.000156

Delete this entry? [y/N]: y
✓ Cleared entry for ADC8
```

**Example - No Entries:**
```bash
> noise-floor-clear --all
No entries to clear
```

**Example - Entry Not Found:**
```bash
> noise-floor-clear --type 99
No entry found for data_type_id 99
```

**When to Use:**
- Before recalculating with different segment selection
- To remove outdated values after system changes
- To clear invalid calculations
- Before deleting data types

**Warning:** Clearing noise floor values will cause feature extraction to fail if noise_floor method is linked to experiment. Recalculate before extracting features.

---

## Typical Workflows

### Initial Setup

```bash
# 1. Initialize
noise-floor-init

# 2. Calculate all
noise-floor-calculate --all
# Type 'y' to confirm

# 3. Verify values
noise-floor-show

# 4. Link to experiment (in SQL)
# See "Linking to Experiment" section below
```

### Update Single Data Type

```bash
# 1. View current values
noise-floor-show

# 2. Clear specific type
noise-floor-clear --type 1
# Type 'y' to confirm

# 3. Recalculate
noise-floor-calculate --type 1
# Type 'y' to confirm

# 4. Verify
noise-floor-show
```

### Reset All Values

```bash
# 1. Clear all
noise-floor-clear --all
# Type 'y' to confirm

# 2. Recalculate all
noise-floor-calculate --all
# Type 'y' to confirm
```

---

## Linking to Experiment

After calculating noise floors, link the method to your experiment:

```sql
-- Connect to database
psql -h localhost -U kjensen -d arc_detection

-- Link noise_floor method (ID 8) to your experiment
INSERT INTO ml_experiments_amplitude_methods (experiment_id, method_id)
VALUES (<YOUR_EXPERIMENT_ID>, 8);

-- Verify link
SELECT e.experiment_id, e.experiment_name, m.method_name, m.display_name
FROM ml_experiments_amplitude_methods eam
JOIN ml_experiments e ON eam.experiment_id = e.experiment_id
JOIN ml_amplitude_normalization_lut m ON eam.method_id = m.method_id
WHERE e.experiment_id = <YOUR_EXPERIMENT_ID>;
```

---

## Troubleshooting

### Error: No noise floor found for data_type_id X

**Cause:** Noise floor not calculated for this data type

**Solution:**
```bash
noise-floor-calculate --type X
```

### Error: No approved steady-state segments found

**Cause:** No segments with status=true or no steady-state segments

**Solution:**
1. Check experiment_status table for approved segments
2. Ensure segments are marked as steady-state
3. Ensure parent files have status=true

```sql
-- Check approved steady-state segments
SELECT COUNT(*)
FROM experiment_status
WHERE status = true
  AND segment_type = 'steady_state';
```

### Error: Table 'experiment_noise_floor' doesn't exist

**Cause:** Table not initialized

**Solution:**
```bash
noise-floor-init
```

### Unexpected Noise Floor Values

**Check:**
1. Number of segments used (shown in noise-floor-show)
2. Segment quality (review approved segments)
3. Calculation method (should be spectral_psd)

**Recalculate if needed:**
```bash
noise-floor-clear --type <id>
noise-floor-calculate --type <id>
```

---

## Best Practices

1. **Calculate After Data Collection:** Run noise-floor-calculate after collecting representative steady-state data

2. **Review Values:** Always run noise-floor-show after calculation to verify values are reasonable

3. **Document Changes:** If recalculating, note why in the database notes column (manual SQL UPDATE)

4. **Per-System Calibration:** Calculate separate noise floors for different hardware configurations

5. **Regular Updates:** Recalculate if ADC characteristics change (new hardware, different settings)

6. **Backup Values:** Export noise floor values before clearing:
   ```sql
   COPY experiment_noise_floor TO '/path/to/backup.csv' CSV HEADER;
   ```

---

## Command Summary Table

| Command | Arguments | Confirmation | Purpose |
|---------|-----------|--------------|---------|
| noise-floor-init | None | No | Create database table |
| noise-floor-show | None | No | Display values |
| noise-floor-calculate | None/--all/--type <id> | Yes | Calculate noise floors |
| noise-floor-clear | --all/--type <id> | Yes | Delete values |

---

## See Also

- Technical Specification: `noise_floor_scaler_specification.md`
- SQL Registration: `noise_floor_sql_registration.sql`
- Implementation Plan: `../wip/noise_floor_implementation_plan.md`

---

**Document Status:** Complete
**Last Updated:** 2025-10-29
