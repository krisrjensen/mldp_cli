# ADC Scaling Normalization - Technical Specification

**Document Version:** 1.0.0
**Date Created:** 2025-10-29
**Author:** Kristophor Jensen
**Status:** Implementation Ready

---

## Overview

The **adc_scaling** amplitude normalization method scales quantized ADC data by dividing by the maximum ADC value based on bit depth. This method normalizes integer ADC values to a [0, 1] range, making features comparable across different ADC resolutions.

---

## Normalization Equation

```
scaled_value = value / (2^bit_depth - 1)
```

Where:
- `value`: Input ADC data value (integer)
- `bit_depth`: ADC resolution in bits (from ml_data_types_lut table)
- `2^bit_depth - 1`: Maximum ADC value (e.g., 255 for 8-bit, 1023 for 10-bit)

---

## Key Features

1. **Data Type Selective**: Only applies to integer ADC data (ADC6, ADC8, ADC10)
2. **Raw Data Pass-Through**: Returns raw (float) data unchanged
3. **Database-Driven**: Bit depth values from `ml_data_types_lut` table
4. **Resolution Independent**: Normalizes to [0, 1] regardless of ADC resolution
5. **No Setup Required**: No pre-calculation or initialization needed

---

## Examples

### 8-bit ADC (ADC8)
```
bit_depth = 8
max_value = 2^8 - 1 = 255

Input: [0, 127, 255]
Output: [0.0, 0.498, 1.0]
```

### 10-bit ADC (ADC10)
```
bit_depth = 10
max_value = 2^10 - 1 = 1023

Input: [0, 512, 1023]
Output: [0.0, 0.5005, 1.0]
```

### 6-bit ADC (ADC6)
```
bit_depth = 6
max_value = 2^6 - 1 = 63

Input: [0, 32, 63]
Output: [0.0, 0.508, 1.0]
```

---

## Database Schema

### ml_data_types_lut (Existing Table)

The scaler queries bit depth from the existing data types table:

```sql
SELECT bit_depth
FROM ml_data_types_lut
WHERE data_type_id = <data_type_id>;
```

**Relevant Columns:**
- `data_type_id` (INTEGER, PRIMARY KEY): Unique identifier
- `data_type_name` (VARCHAR): e.g., 'ADC8', 'ADC10', 'RAW'
- `bit_depth` (INTEGER): ADC resolution in bits (6, 8, 10, etc.)

**Example Data:**
```sql
data_type_id | data_type_name | bit_depth
-------------|----------------|----------
1            | ADC8           | 8
2            | ADC10          | 10
3            | ADC6           | 6
4            | RAW            | NULL
```

### ml_amplitude_normalization_lut Entry

```sql
INSERT INTO ml_amplitude_normalization_lut
(method_id, method_name, display_name, description, function_name,
 function_args, parameters_schema, column_count, is_active)
VALUES
(7, 'adc_scaling', 'ADC Bit-Depth Scaling',
 'Scales data by dividing by (2^bit_depth - 1). Only applies to integer ADC data, returns raw float data unchanged.',
 '_apply_adc_scaling', NULL, NULL, 1, true);
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
- Raw data: float32, float64 (continuous values, already normalized)
- ADC data: int8, int16, int32 (quantized integer values)
- Uses numpy's issubdtype() for reliable type checking

### Normalization Process

```python
def _apply_adc_scaling(data, data_type_id):
    """
    Scale ADC data by bit depth maximum value

    Args:
        data: Input array (integer ADC or float raw)
        data_type_id: Data type ID for querying bit depth

    Returns:
        Normalized array in [0, 1] range for ADC data,
        unchanged copy for raw data
    """
    # 1. Check if raw data (float) - return unchanged
    if np.issubdtype(data.dtype, np.floating):
        return data.copy()

    # 2. Query bit depth from database
    bit_depth = query_bit_depth(data_type_id)

    # 3. Check for invalid/missing bit depth
    if bit_depth is None:
        raise ValueError(f"No bit_depth found for data_type_id {data_type_id}")

    # 4. Calculate max ADC value
    max_adc_value = (2 ** bit_depth) - 1

    # 5. Convert to float for division
    data_float = data.astype(np.float64)

    # 6. Apply normalization
    normalized = data_float / max_adc_value

    return normalized
```

---

## Integration with Feature Extraction

### Workflow

1. **Link to Experiment**
   ```sql
   INSERT INTO ml_experiments_amplitude_methods (experiment_id, method_id)
   VALUES (<EXPERIMENT_ID>, 7);  -- method_id 7 = adc_scaling
   ```

2. **Extract Features**
   - Feature extractor automatically loads bit_depth from database
   - Applies normalization during feature extraction
   - Skips raw data, normalizes ADC data

### Error Handling

**Missing Bit Depth:**
```
ValueError: No bit_depth found for data_type_id 1.
Check ml_data_types_lut table.
```

**Solution:** Ensure data types have valid bit_depth values in database.

---

## File Structure

### Modified Files

```
mldp_feature_processor/
└── src/
    └── scalers/
        └── amplitude_processor.py         # MODIFIED: Add _apply_adc_scaling()
```

### Documentation Files

```
mldp_cli/
├── documentation/
│   ├── reference/
│   │   ├── adc_scaling_scaler_specification.md    # This file
│   │   └── adc_scaling_sql_registration.sql       # SQL registration
│   └── wip/
│       └── adc_scaling_implementation_plan.md     # Implementation tracking
│
└── tests/
    └── test_adc_scaling_scaler.py                 # NEW: Scaler tests
```

---

## Testing Strategy

### Unit Tests

1. **Normalization Equation**
   - Test 6-bit ADC: 63 → 1.0
   - Test 8-bit ADC: 255 → 1.0
   - Test 10-bit ADC: 1023 → 1.0
   - Test mid-range values: 127/255 ≈ 0.498
   - Test zero: 0 → 0.0

2. **Data Type Handling**
   - Test integer data processing (int8, int16, int32)
   - Test float data pass-through (float32, float64)
   - Test type preservation after normalization

3. **Database Integration**
   - Test bit_depth query for each data type
   - Test missing bit_depth error handling
   - Test invalid data_type_id error

4. **Edge Cases**
   - Test empty arrays
   - Test single-value arrays
   - Test negative values (should not occur in ADC data)
   - Test out-of-range values

### Integration Tests

1. **End-to-End Workflow**
   - Link adc_scaling to experiment
   - Extract features with ADC data
   - Verify normalized values in [0, 1] range
   - Compare with manual calculation

2. **Multiple Data Types**
   - Extract features for ADC6, ADC8, ADC10
   - Verify correct bit_depth used for each
   - Verify raw data unchanged

---

## Performance Considerations

### Computation Performance

- **Division overhead**: O(N) single division operation
- **Type conversion**: O(N) integer to float64
- **Database query**: One-time lookup per data type (cached)
- **Overall impact**: ~5% slowdown vs raw (minimal)

### Memory

- **Temporary float64 array**: 8 bytes × N samples
- **Original data preserved**: No in-place modification
- **Typical segment**: 2000 samples = 16 KB temporary memory

---

## Use Cases

### 1. Cross-Resolution Comparison
Normalize features extracted from different ADC resolutions (6-bit, 8-bit, 10-bit) to comparable scales.

### 2. Fraction of Full Scale
Convert ADC counts to fraction of full-scale range for interpretability.

### 3. Pre-Processing for ML
Normalize ADC data to [0, 1] before applying additional feature extraction methods.

### 4. Standardized Feature Sets
Create feature sets that work consistently across different hardware configurations.

---

## Comparison with Other Scalers

### vs raw (method_id 5)
- **raw**: Returns data unchanged
- **adc_scaling**: Normalizes to [0, 1] based on bit depth

### vs minmax (method_id 1)
- **minmax**: Normalizes based on observed min/max in segment
- **adc_scaling**: Normalizes based on theoretical ADC range (0 to 2^N-1)

### vs zscore (method_id 2)
- **zscore**: Centers around mean, scales by std dev (range unbounded)
- **adc_scaling**: Preserves zero, scales to [0, 1] (range bounded)

### vs noise_floor (method_id 8)
- **noise_floor**: Scales by measured noise characteristics
- **adc_scaling**: Scales by ADC resolution characteristics

---

## Limitations

1. **Assumes Full ADC Range**: Uses theoretical max (2^N-1), not observed max
2. **No Centering**: Doesn't remove DC offset or mean
3. **Integer ADC Only**: Doesn't modify raw floating-point data
4. **Static Scaling**: Doesn't adapt to signal characteristics

---

## When to Use

**Use adc_scaling when:**
- Comparing features across different ADC resolutions
- Want features as fraction of full-scale range
- Need simple, interpretable normalization
- Have well-calibrated ADC with known bit depth

**Don't use adc_scaling when:**
- Data already normalized (raw format)
- Need zero-mean features (use zscore)
- Need segment-adaptive scaling (use minmax)
- Need noise-based normalization (use noise_floor)

---

## References

- IEEE Standard for Terminology and Test Methods for Analog-to-Digital Converters
- Microchip AN1259: "Matching the Resolution of ADCs and DACs to the Application"

---

## Changelog

### v1.0.0 - 2025-10-29
- Initial specification
- Defined normalization equation
- Documented database integration
- Integration workflow documented

---

**Document Status:** Ready for Implementation
**Next Steps:** Proceed with implementation in AmplitudeProcessor class
