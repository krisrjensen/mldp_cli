# Noise Floor Normalization Scaler - Technical Specification

**Document Version:** 1.0.0
**Date Created:** 2025-10-29
**Author:** Kristophor Jensen
**Status:** Implementation Ready

---

## Overview

The **noise_floor** amplitude normalization method scales data using pre-calculated noise floor values derived from approved steady-state segments. This method is designed to normalize ADC data by the inherent noise characteristics of the measurement system.

---

## Normalization Equation

```
scaled_value = (X_i - mean(X)) / noise_floor
```

Where:
- `X_i`: Input data value at sample i
- `mean(X)`: Mean of input data array
- `noise_floor`: Pre-calculated noise floor value from database

---

## Key Features

1. **Data Type Selective**: Only applies to integer ADC data (ADC6, ADC8, ADC10)
2. **Raw Data Pass-Through**: Returns raw (float) data unchanged
3. **Database-Driven**: Noise floor values stored in `experiment_noise_floor` table
4. **Spectral Calculation**: Uses Power Spectral Density (PSD) method for noise floor estimation
5. **Approved Segments Only**: Calculates from segments with `experiment_status.status=true`
6. **CLI Management**: Four commands for initialization, calculation, display, and clearing

---

## Database Schema

### experiment_noise_floor Table

```sql
CREATE TABLE experiment_noise_floor (
    data_type_id INTEGER PRIMARY KEY,
    noise_floor DOUBLE PRECISION NOT NULL,
    calculation_method VARCHAR(50) DEFAULT 'spectral_psd',
    num_segments_used INTEGER,
    last_calculated TIMESTAMP DEFAULT NOW(),
    notes TEXT,
    FOREIGN KEY (data_type_id)
        REFERENCES ml_data_types_lut(data_type_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_experiment_noise_floor_data_type
ON experiment_noise_floor(data_type_id);
```

**Columns:**
- `data_type_id` (INTEGER, PRIMARY KEY): Links to ml_data_types_lut
- `noise_floor` (DOUBLE PRECISION, NOT NULL): Calculated noise floor value (RMS units)
- `calculation_method` (VARCHAR(50)): Method used (default: 'spectral_psd')
- `num_segments_used` (INTEGER): Number of segments used in calculation
- `last_calculated` (TIMESTAMP): When calculation was performed
- `notes` (TEXT): Optional notes about calculation

### ml_amplitude_normalization_lut Entry

```sql
INSERT INTO ml_amplitude_normalization_lut
(method_id, method_name, display_name, description, function_name,
 function_args, parameters_schema, column_count, is_active)
VALUES
(8, 'noise_floor', 'Noise Floor Normalization',
 'Scales data using (X - mean(X)) / noise_floor. Noise floor calculated from approved steady-state segments using spectral methods. Only applies to integer ADC data, returns raw float data unchanged.',
 '_apply_noise_floor', NULL, NULL, 1, true);
```

---

## Noise Floor Calculation Algorithm

### Method: Spectral Power Spectral Density (PSD)

**Steps:**

1. **Query Approved Segments**
   - Select segments where `experiment_status.status = true`
   - Filter for steady-state segments only
   - Ensure parent file has `status = true`
   - Filter by data_type_id if specified

2. **Load Segment Data**
   - Load from `fileset/` or `adc_data/` folders
   - Read segment time-series data

3. **Compute PSD per Segment**
   - Apply Welch's method for PSD estimation
   - Parameters:
     - nperseg: min(1024, len(data)//4)
     - Default overlap
     - Hann window
   - Exclude DC component (freq = 0)

4. **Calculate Noise Floor from PSD**
   - Take 10th percentile of PSD values
   - This excludes signal peaks and focuses on noise floor
   - Convert to RMS: `sqrt(mean(psd[psd <= 10th_percentile]))`

5. **Average Across Segments**
   - Calculate noise floor for each segment
   - Take mean across all segments
   - Store in database with metadata

**Rationale:**
- 10th percentile excludes signal content and transients
- Welch's method provides robust PSD estimate
- Multiple segments average out variability
- Spectral approach more robust than time-domain std

---

## CLI Commands

### 1. noise-floor-init

**Purpose:** Initialize the experiment_noise_floor database table.

**Usage:**
```bash
noise-floor-init
```

**Behavior:**
- Checks if `experiment_noise_floor` table exists
- If exists: Reports status and current entry count
- If not exists: Creates table with proper schema and indices
- Always reports final status

**Example Output:**
```
✓ experiment_noise_floor table already exists
  Current entries: 3

# OR

✓ Created experiment_noise_floor table
```

---

### 2. noise-floor-clear

**Purpose:** Clear noise floor entries from database.

**Usage:**
```bash
noise-floor-clear                     # Show help
noise-floor-clear --all               # Clear all entries (requires confirmation)
noise-floor-clear --type <data_type_id>  # Clear specific type (requires confirmation)
```

**Options:**
- **No arguments**: Displays help message
- **--all**: Clears all noise floor entries after user confirmation
- **--type <data_type_id>**: Clears specific data type entry after user confirmation

**Confirmation Required:** Yes, for both --all and --type operations

**Example:**
```bash
> noise-floor-clear --all

Current noise floor entries:
  ADC8: 0.000156 (245 segments)
  ADC10: 0.000089 (312 segments)

Delete all 2 entries? [y/N]: y
✓ Cleared 2 entries
```

---

### 3. noise-floor-show

**Purpose:** Display current noise floor values.

**Usage:**
```bash
noise-floor-show
```

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

**Behavior:**
- Lists all calculated noise floor values
- Shows data type, ID, noise floor, number of segments used
- Displays calculation timestamp and method
- If no entries: Suggests running `noise-floor-calculate`

---

### 4. noise-floor-calculate

**Purpose:** Calculate noise floor values from approved steady-state segments.

**Usage:**
```bash
noise-floor-calculate                      # Calculate all types (requires auth)
noise-floor-calculate --all                # Same as above
noise-floor-calculate --type <data_type_id>  # Calculate specific type (requires auth)
```

**Options:**
- **No arguments / --all**: Calculate for all data types in database
- **--type <data_type_id>**: Calculate for specific data type only

**Confirmation Required:** Yes, user must type 'y' to proceed

**Behavior:**
1. Prompts user for confirmation
2. Queries approved steady-state segments for data type(s)
3. Loads segment data from filesystem
4. Computes noise floor using spectral PSD method
5. Stores/updates values in database
6. Displays updated noise floor values
7. Can be run multiple times (updates existing entries)

**Example:**
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
ADC8         1    0.000156        245        2025-10-29 14:35     spectral_psd
```

---

## Scaler Implementation

### Integer vs Float Detection

```python
# Detect raw (float) data
if np.issubdtype(data.dtype, np.floating):
    # Return unchanged for raw data
    return data.copy()
```

**Rationale:**
- Raw data: float32, float64 (continuous values)
- ADC data: int8, int16, int32 (quantized integer values)
- Uses numpy's issubdtype() for reliable type checking

### Normalization Process

```python
def _apply_noise_floor(data, data_type_id):
    # 1. Check if raw data (float) - return unchanged
    if np.issubdtype(data.dtype, np.floating):
        return data.copy()

    # 2. Query noise floor from database
    noise_floor = query_database(data_type_id)

    # 3. Convert to float for computation
    data_float = data.astype(np.float64)

    # 4. Calculate mean
    data_mean = np.mean(data_float)

    # 5. Apply normalization
    normalized = (data_float - data_mean) / noise_floor

    return normalized
```

---

## Integration with Feature Extraction

### Workflow

1. **Initialization (One-time)**
   ```bash
   noise-floor-init
   ```

2. **Calculate Noise Floors (Per Data Type)**
   ```bash
   noise-floor-calculate --all
   noise-floor-show
   ```

3. **Link to Experiment**
   ```sql
   INSERT INTO ml_experiments_amplitude_methods (experiment_id, method_id)
   VALUES (<EXPERIMENT_ID>, 8);  -- method_id 8 = noise_floor
   ```

4. **Extract Features**
   - Feature extractor automatically loads noise floor from database
   - Applies normalization during feature extraction
   - Skips raw data, normalizes ADC data

### Error Handling

**Missing Noise Floor:**
```
ValueError: No noise floor found for data_type_id 1.
Run 'noise-floor-calculate' first.
```

**Solution:** Run `noise-floor-calculate` before feature extraction.

---

## File Structure

### New Files

```
mldp_cli/
├── src/
│   ├── noise_floor_calculator.py          # NEW: Calculation logic
│   └── mldp_shell.py                      # MODIFIED: Add 4 CLI commands
│
├── documentation/
│   ├── reference/
│   │   ├── noise_floor_scaler_specification.md    # This file
│   │   ├── noise_floor_cli_commands.md            # CLI reference
│   │   └── noise_floor_sql_registration.sql       # SQL scripts
│   └── wip/
│       └── noise_floor_implementation_plan.md     # Implementation tracking
│
└── tests/
    ├── test_noise_floor_calculator.py     # NEW: Calculator tests
    └── test_noise_floor_scaler.py         # NEW: Scaler tests

mldp_feature_processor/
└── src/
    └── scalers/
        └── amplitude_processor.py         # MODIFIED: Add _apply_noise_floor()
```

---

## Testing Strategy

### Unit Tests

1. **Noise Floor Calculator**
   - Test PSD calculation with known noise level
   - Test approved segment filtering
   - Test file status validation
   - Test averaging across segments

2. **Scaler Function**
   - Test normalization equation: (X - mean) / noise_floor
   - Test integer data processing
   - Test float data pass-through
   - Test missing noise floor error

3. **CLI Commands**
   - Test noise-floor-init (create/existing table)
   - Test noise-floor-show (empty/populated)
   - Test noise-floor-clear (all/specific/confirmation)
   - Test noise-floor-calculate (all/specific/authentication)

### Integration Tests

1. **End-to-End Workflow**
   - Initialize → Calculate → Show → Extract Features
   - Verify normalized features in database
   - Compare with manual calculation

2. **Multiple Data Types**
   - Calculate for ADC6, ADC8, ADC10
   - Verify different noise floors
   - Extract features with each type

---

## Performance Considerations

### Calculation Performance

- **Welch's method**: O(N log N) per segment due to FFT
- **Multiple segments**: Parallelizable (future enhancement)
- **Typical time**: ~1-2 seconds per 100 segments

### Storage

- **Database size**: ~100 bytes per data type entry
- **Typical setup**: 3-5 data types = ~500 bytes total
- **Negligible storage impact**

### Feature Extraction Performance

- **Normalization overhead**: ~10% slowdown vs raw
- **Database query**: Cached after first lookup
- **Float conversion**: Minimal overhead

---

## Use Cases

### 1. System Noise Characterization
Measure and normalize by the inherent noise floor of the ADC system.

### 2. Cross-Experiment Comparison
Standardize features across experiments with different noise characteristics.

### 3. Signal-to-Noise Ratio Features
Create SNR-based features by normalizing signal by noise floor.

### 4. Adaptive Thresholding
Use noise floor for dynamic threshold setting in arc detection.

---

## Limitations

1. **Requires Steady-State Segments**: Must have approved steady-state data
2. **Data Type Specific**: Separate noise floor per data type
3. **Static Noise Floor**: Doesn't adapt to time-varying noise
4. **Integer ADC Only**: Doesn't apply to raw floating-point data

---

## Future Enhancements

1. **Time-Varying Noise Floor**: Calculate noise floor per segment
2. **Frequency-Band Specific**: Different noise floors per frequency band
3. **Parallel Calculation**: Use multiprocessing for large datasets
4. **Automatic Recalculation**: Trigger when new segments approved
5. **Noise Floor Visualization**: Plot PSD and noise floor estimates

---

## References

- Welch, P. (1967). "The use of fast Fourier transform for the estimation of power spectra"
- IEEE Standard for Terminology and Test Methods for Analog-to-Digital Converters
- Oppenheim & Schafer. "Discrete-Time Signal Processing"

---

## Changelog

### v1.0.0 - 2025-10-29
- Initial specification
- Defined database schema
- Documented calculation algorithm
- Specified CLI commands
- Integration workflow documented

---

**Document Status:** Ready for Implementation
**Next Steps:** Proceed with Phase 1 (Database Implementation)
