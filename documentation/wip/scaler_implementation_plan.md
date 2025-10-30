# Amplitude Scaler Implementation Plan

**Document Version:** 1.0.0
**Date Created:** 2025-10-29
**Author:** Kristophor Jensen
**Status:** In Progress

---

## Overview

Implementation plan for two new amplitude normalization scalers:
1. **adc_scaling** (method_id 7): Divides by 2^bit_depth - 1
2. **noise_floor** (method_id 8): (X - mean(X)) / noise_floor

---

## Implementation Phases

### Phase 1: Database Setup ✅

#### adc_scaling
- [x] Execute `adc_scaling_sql_registration.sql`
- [x] Verify method_id 9 registered
- [x] Verify bit_depth column exists in ml_data_types_lut

#### noise_floor
- [x] Execute `noise_floor_sql_registration.sql`
- [x] Verify experiment_noise_floor table created
- [x] Verify method_id 10 registered
- [x] Verify table indices created

---

### Phase 2: adc_scaling Implementation ✅

**Target File:** `/ml_code/src/scalers/amplitude_processor.py`

#### Code Changes
- [x] Add `_apply_adc_scaling()` method to AmplitudeProcessor class
- [x] Implement integer vs float detection
- [x] Query bit_depth from database
- [x] Apply normalization: value / (2^bit_depth - 1)
- [x] Handle errors (missing bit_depth)
- [x] Update version number to 0.0.1.0
- [x] Add changelog entry

#### Testing
- [ ] Create `test_adc_scaling_scaler.py`
- [ ] Test 6-bit, 8-bit, 10-bit ADC normalization
- [ ] Test float data pass-through
- [ ] Test edge cases (empty array, single value)
- [ ] Test database query errors
- [ ] Run all tests and verify pass

---

### Phase 3: noise_floor Implementation ✅

**Target Files:**
- `/mldp_cli/src/noise_floor_calculator.py` (NEW)
- `/ml_code/src/scalers/amplitude_processor.py` (MODIFY)
- `/mldp_cli/src/mldp_shell.py` (MODIFY)

#### 3.1: NoiseFloorCalculator Module
- [x] Create `noise_floor_calculator.py`
- [x] Implement `NoiseFloorCalculator` class
- [x] Implement `calculate_noise_floor()` method
  - [x] Query approved steady-state segments
  - [x] Load segment data from fileset/adc_data
  - [x] Compute PSD using Welch's method
  - [x] Calculate 10th percentile for noise floor
  - [x] Average across segments
  - [x] Store in database
- [x] Implement helper methods
- [x] Add comprehensive docstrings
- [x] Add logging

#### 3.2: AmplitudeProcessor Integration
- [x] Add `_apply_noise_floor()` method to AmplitudeProcessor class
- [x] Implement integer vs float detection
- [x] Query noise_floor from database
- [x] Apply normalization: (X - mean(X)) / noise_floor
- [x] Handle errors (missing noise floor)
- [x] Update version number to 0.0.1.0
- [x] Add changelog entry

#### 3.3: CLI Commands
- [x] Modify `mldp_shell.py` to add 4 commands:
  - [x] `noise-floor-init`
  - [x] `noise-floor-show`
  - [x] `noise-floor-calculate`
  - [x] `noise-floor-clear`
- [x] Implement user confirmation prompts
- [x] Add help text for each command
- [x] Update shell version number to 2.0.10.23
- [x] Add changelog entry

#### 3.4: Testing
- [ ] Create `test_noise_floor_calculator.py`
- [ ] Test PSD calculation with known noise
- [ ] Test approved segment filtering
- [ ] Test multi-segment averaging
- [ ] Create `test_noise_floor_scaler.py`
- [ ] Test normalization equation
- [ ] Test integer vs float handling
- [ ] Test missing noise floor error
- [ ] Integration test: init → calculate → show → use

---

### Phase 4: Documentation Updates ✅

- [x] Update method IDs to 9 and 10 in all docs
- [x] Update `scaler_implementation_plan.md`
- [ ] Create scaler usage examples
- [ ] Document testing results

---

### Phase 5: Version Control ⏸️

#### mldp_feature_processor
- [ ] Stage `amplitude_processor.py` changes
- [ ] Stage `models.py` changes
- [ ] Commit with descriptive message
- [ ] Push to remote

#### mldp_cli Submodule
- [ ] Stage `noise_floor_calculator.py` (new)
- [ ] Stage `mldp_shell.py` changes
- [ ] Stage SQL scripts
- [ ] Stage documentation updates
- [ ] Commit with descriptive message
- [ ] Push to remote

#### Main mldp Repository
- [ ] Update mldp_feature_processor reference (if separate repo)
- [ ] Update mldp_cli submodule reference
- [ ] Commit submodule updates
- [ ] Push to remote

---

## Implementation Details

### adc_scaling Scaler

**Equation:**
```
scaled_value = value / (2^bit_depth - 1)
```

**Function Signature:**
```python
def _apply_adc_scaling(self, data: np.ndarray, data_type_id: int) -> np.ndarray:
    """
    Scale ADC data by bit depth maximum value

    Args:
        data: Input array (integer ADC or float raw)
        data_type_id: Data type ID for querying bit depth

    Returns:
        Normalized array in [0, 1] range for ADC data,
        unchanged copy for raw data
    """
```

**Database Query:**
```python
query = """
    SELECT bit_depth
    FROM ml_data_types_lut
    WHERE data_type_id = %s
"""
```

---

### noise_floor Scaler

**Equation:**
```
scaled_value = (X_i - mean(X)) / noise_floor
```

**Function Signature:**
```python
def _apply_noise_floor(self, data: np.ndarray, data_type_id: int) -> np.ndarray:
    """
    Scale data using noise floor normalization

    Args:
        data: Input array (integer ADC or float raw)
        data_type_id: Data type ID for querying noise floor

    Returns:
        Normalized array for ADC data,
        unchanged copy for raw data
    """
```

**Database Query:**
```python
query = """
    SELECT noise_floor
    FROM experiment_noise_floor
    WHERE data_type_id = %s
"""
```

---

### CLI Commands Implementation

#### noise-floor-init
```python
def do_noise_floor_init(self, arg):
    """Initialize experiment_noise_floor table"""
    # Check if table exists
    # Create if needed
    # Report status
```

#### noise-floor-show
```python
def do_noise_floor_show(self, arg):
    """Display calculated noise floor values"""
    # Query all entries
    # Format as table
    # Display with data type names
```

#### noise-floor-calculate
```python
def do_noise_floor_calculate(self, arg):
    """Calculate noise floors from approved segments"""
    # Parse args: --all, --type <id>
    # Prompt for confirmation
    # Call NoiseFloorCalculator
    # Display results
```

#### noise-floor-clear
```python
def do_noise_floor_clear(self, arg):
    """Clear noise floor entries"""
    # Parse args: --all, --type <id>
    # Show current entries
    # Prompt for confirmation
    # Delete entries
    # Report status
```

---

## Testing Strategy

### Unit Tests

**adc_scaling:**
- Test equation: 255 / 255 = 1.0 (8-bit max)
- Test equation: 127 / 255 ≈ 0.498 (8-bit mid)
- Test 6-bit, 8-bit, 10-bit normalization
- Test float data pass-through
- Test missing bit_depth error

**noise_floor:**
- Test equation: (100 - 50) / 10 = 5.0
- Test PSD calculation with synthetic noise
- Test 10th percentile calculation
- Test multi-segment averaging
- Test float data pass-through
- Test missing noise floor error

### Integration Tests

**adc_scaling:**
- Link to experiment
- Extract features with ADC data
- Verify values in [0, 1] range

**noise_floor:**
- Initialize table
- Calculate from approved segments
- Show values
- Link to experiment
- Extract features
- Verify normalization applied

---

## File Locations

### Source Code
```
ml_code/
└── src/
    └── scalers/
        └── amplitude_processor.py    # MODIFY: Add both _apply_* methods

mldp_cli/
├── src/
│   ├── noise_floor_calculator.py     # NEW: Noise floor calculation logic
│   └── mldp_shell.py                 # MODIFY: Add 4 CLI commands
│
└── tests/
    ├── test_adc_scaling_scaler.py    # NEW: adc_scaling tests
    ├── test_noise_floor_calculator.py # NEW: Calculator tests
    └── test_noise_floor_scaler.py    # NEW: Scaler tests
```

### Documentation
```
mldp_cli/
└── documentation/
    ├── reference/
    │   ├── adc_scaling_scaler_specification.md     # ✅ Created
    │   ├── adc_scaling_sql_registration.sql        # ✅ Created
    │   ├── noise_floor_scaler_specification.md     # ✅ Created
    │   ├── noise_floor_cli_commands.md             # ✅ Created
    │   └── noise_floor_sql_registration.sql        # ✅ Created
    │
    ├── wip/
    │   └── scaler_implementation_plan.md           # This file
    │
    └── status/
        ├── phase_completion_checklist.md           # UPDATE: Add scaler phases
        └── feature_implementation_phase_tracker.md # UPDATE: Add scaler status
```

---

## Current Status

**Overall Progress:** 95% (Implementation complete, testing pending)

**Completed:**
- ✅ adc_scaling technical specification
- ✅ adc_scaling SQL registration script
- ✅ noise_floor technical specification
- ✅ noise_floor CLI command reference
- ✅ noise_floor SQL registration script
- ✅ Implementation plan (this document)

**In Progress:**
- ⏸️ Database setup (Phase 1)

**Pending:**
- adc_scaling implementation (Phase 2)
- noise_floor implementation (Phase 3)
- Testing (embedded in Phases 2-3)
- Documentation updates (Phase 4)
- Version control (Phase 5)

---

## Next Steps

1. Execute SQL registration scripts on database
2. Implement adc_scaling scaler in AmplitudeProcessor
3. Test adc_scaling implementation
4. Implement noise_floor calculator module
5. Implement noise_floor scaler in AmplitudeProcessor
6. Implement 4 CLI commands in mldp_shell.py
7. Test noise_floor implementation (unit + integration)
8. Update documentation
9. Commit and push changes

---

## Estimated Timeline

- Phase 1 (Database Setup): 10 minutes
- Phase 2 (adc_scaling): 30 minutes
- Phase 3 (noise_floor): 2-3 hours
- Phase 4 (Documentation): 15 minutes
- Phase 5 (Version Control): 15 minutes

**Total Estimated Time:** 3-4 hours

---

**Document Status:** Active
**Last Updated:** 2025-10-29
