# Modular Visualization Architecture Plan

**Filename:** MODULAR_VISUALIZATION_ARCHITECTURE.md
**Author(s):** Kristophor Jensen
**Date Created:** 20251120_000000
**Date Revised:** 20251120_000000
**File version:** 1.0.0.0
**Description:** Long-term plan for refactoring visualization system into modular architecture with <2500 lines per file

---

## 1. Executive Summary

The current visualization system is approaching monolithic scale with single files potentially exceeding 2000-3000 lines. This document outlines a plan to refactor the system into a modular architecture where no file exceeds 2500 lines, improving maintainability, testability, and extensibility.

**Timeline:** Implement after current urgent visualization deliverables (1.5 hour deadline) are met.

**Constraint:** Maximum 2500 lines per file

---

## 2. Current State Analysis

### 2.1 Current File Structure

```
mldp_cli/src/
├── visualize_verification_features.py  (~1200+ lines, growing)
├── run_production_visualization.py     (~100 lines)
├── run_dimreduction_visualization.py   (~350 lines)
├── run_scalar_3d_visualization.py      (~300 lines)
└── quick_fix_get_exp042_data.py        (~30 lines)
```

### 2.2 Problems with Current Architecture

1. **Growing Monolith Risk**: `visualize_verification_features.py` will exceed 2500 lines as features are added
2. **Mixed Concerns**: Single file handles:
   - Data loading
   - Outlier detection (4 methods)
   - Sigmoid squashing (4 methods)
   - PDF plotting
   - 3D scatter plotting
   - Database queries
   - File I/O
   - Metadata generation
3. **Testing Difficulty**: Hard to unit test individual components
4. **Code Reuse**: Difficult to reuse components across different visualization types
5. **Parallel Development**: Hard for multiple developers to work simultaneously

### 2.3 Technical Debt

- Duplicate code between visualization scripts
- Hard-coded paths and configuration
- Tightly coupled database access
- No configuration management system

---

## 3. Target Modular Architecture

### 3.1 Proposed Directory Structure

```
mldp_cli/
├── src/
│   ├── mldp_viz/                          # New visualization package
│   │   ├── __init__.py                    # Package initialization
│   │   ├── core/                          # Core components
│   │   │   ├── __init__.py
│   │   │   ├── data_loader.py            # Data loading from database/files (~300 lines)
│   │   │   ├── database.py               # Database queries and connections (~200 lines)
│   │   │   └── config.py                 # Configuration management (~150 lines)
│   │   ├── preprocessing/                # Data preprocessing
│   │   │   ├── __init__.py
│   │   │   ├── outlier_detectors.py      # Outlier detection methods (~400 lines)
│   │   │   └── sigmoid_squashers.py      # Sigmoid squashing methods (~300 lines)
│   │   ├── plotters/                     # Plotting engines
│   │   │   ├── __init__.py
│   │   │   ├── base_plotter.py           # Base plotter class (~200 lines)
│   │   │   ├── pdf_plotter.py            # PDF/histogram plotting (~500 lines)
│   │   │   ├── scatter_3d_plotter.py     # 3D scatter plotting (~400 lines)
│   │   │   ├── dimreduction_plotter.py   # Dimensionality reduction (~500 lines)
│   │   │   └── projection_plotter.py     # XY/XZ/YZ projections (~300 lines)
│   │   ├── utils/                        # Utilities
│   │   │   ├── __init__.py
│   │   │   ├── color_utils.py            # Color mapping and palettes (~150 lines)
│   │   │   ├── legend_utils.py           # Legend generation (~150 lines)
│   │   │   ├── file_utils.py             # File I/O helpers (~200 lines)
│   │   │   └── metadata_utils.py         # Metadata generation (~150 lines)
│   │   └── runners/                      # High-level orchestration scripts
│   │       ├── __init__.py
│   │       ├── run_production.py         # Production visualization runner (~300 lines)
│   │       ├── run_dimreduction.py       # Dimensionality reduction runner (~200 lines)
│   │       └── run_scalar_3d.py          # Scalar 3D runner (~200 lines)
│   └── (legacy scripts remain for backward compatibility during migration)
├── tests/                                # New test directory
│   ├── test_outlier_detectors.py
│   ├── test_sigmoid_squashers.py
│   ├── test_pdf_plotter.py
│   └── ...
└── config/                               # Configuration files
    ├── visualization_config.yaml
    └── database_config.yaml
```

### 3.2 Module Responsibilities

#### **Core Modules**

- **`data_loader.py`**: Load verification feature matrices from .npy files, handle NaN values, extract metadata
- **`database.py`**: Database connections, segment label queries, file label queries, experiment configuration
- **`config.py`**: Load/save configuration, path management, default parameters

#### **Preprocessing Modules**

- **`outlier_detectors.py`**:
  - `IQROutlierDetector`
  - `ZScoreOutlierDetector`
  - `ModifiedZScoreOutlierDetector`
  - `IsolationForestOutlierDetector`

- **`sigmoid_squashers.py`**:
  - `StandardSigmoidSquasher`
  - `TanhSquasher`
  - `AdaptiveSigmoidSquasher`
  - `SoftClipSquasher`

#### **Plotter Modules**

- **`base_plotter.py`**: Abstract base class defining plotter interface
- **`pdf_plotter.py`**: PDF/histogram plotting with smoothing, multi-version support, grouping levels
- **`scatter_3d_plotter.py`**: 3D scatter plots with multiple viewing angles
- **`dimreduction_plotter.py`**: PCA, LLE, t-SNE, UMAP dimensionality reduction visualizations
- **`projection_plotter.py`**: XY, XZ, YZ projection plots

#### **Utility Modules**

- **`color_utils.py`**: Consistent color mapping across plots, color palette generation
- **`legend_utils.py`**: Legend creation with bbox positioning, label filtering
- **`file_utils.py`**: File path validation, directory creation, safe file writing
- **`metadata_utils.py`**: JSON metadata generation, outlier count tracking, version info

#### **Runner Modules**

- **`run_production.py`**: High-level orchestration for production visualizations
- **`run_dimreduction.py`**: Dimensionality reduction visualization orchestration
- **`run_scalar_3d.py`**: Scalar 3D combination visualization orchestration

---

## 4. Migration Strategy

### 4.1 Phase 1: Extract Core and Utils (Week 1)

**Goal:** Extract reusable core components with no visualization logic

**Steps:**
1. Create package structure and `__init__.py` files
2. Extract `database.py` from existing code
3. Extract `data_loader.py` from existing code
4. Extract `config.py` and create YAML config files
5. Extract utility modules (color, legend, file, metadata)
6. Write unit tests for each module
7. Update existing scripts to import from new modules

**Success Criteria:**
- All extracted modules < 300 lines
- 100% backward compatibility with existing scripts
- All unit tests pass

### 4.2 Phase 2: Extract Preprocessing (Week 2)

**Goal:** Modularize outlier detection and sigmoid squashing

**Steps:**
1. Create base class for outlier detectors
2. Extract 4 outlier detector classes into `outlier_detectors.py`
3. Create base class for sigmoid squashers
4. Extract 4 sigmoid squasher classes into `sigmoid_squashers.py`
5. Write unit tests with sample data
6. Update existing scripts to use new classes

**Success Criteria:**
- `outlier_detectors.py` < 400 lines
- `sigmoid_squashers.py` < 300 lines
- All existing visualizations produce identical output

### 4.3 Phase 3: Extract Plotters (Week 3-4)

**Goal:** Modularize plotting logic into separate classes

**Steps:**
1. Create `base_plotter.py` with abstract interface
2. Extract `pdf_plotter.py` with smoothing logic
3. Extract `scatter_3d_plotter.py`
4. Extract `dimreduction_plotter.py`
5. Extract `projection_plotter.py`
6. Write integration tests for each plotter
7. Update runner scripts to use new plotters

**Success Criteria:**
- Each plotter < 500 lines
- All plots visually identical to current output
- Integration tests pass

### 4.4 Phase 4: Create New Runners (Week 5)

**Goal:** Replace legacy scripts with modular runners

**Steps:**
1. Implement `run_production.py` using new modules
2. Implement `run_dimreduction.py` using new modules
3. Implement `run_scalar_3d.py` using new modules
4. Run parallel testing: old vs. new runners
5. Deprecate old scripts after validation

**Success Criteria:**
- Each runner < 300 lines
- 100% feature parity with legacy scripts
- Performance within 10% of legacy scripts

### 4.5 Phase 5: Cleanup and Documentation (Week 6)

**Goal:** Remove technical debt and document system

**Steps:**
1. Remove deprecated legacy scripts
2. Write comprehensive README for `mldp_viz` package
3. Generate API documentation
4. Create usage examples and tutorials
5. Update main project documentation

**Success Criteria:**
- No files exceed 2500 lines
- Documentation coverage > 90%
- All examples work correctly

---

## 5. Design Principles

### 5.1 SOLID Principles

- **Single Responsibility**: Each module has one clear purpose
- **Open/Closed**: Extendable without modifying existing code
- **Liskov Substitution**: Derived classes are substitutable for base classes
- **Interface Segregation**: Clients don't depend on unused interfaces
- **Dependency Inversion**: Depend on abstractions, not implementations

### 5.2 Code Organization

- Maximum 2500 lines per file (target: <500 lines)
- Maximum 40 lines per function
- Classes in separate files when possible
- Descriptive variable names
- Type hints for all function signatures

### 5.3 Configuration Management

- YAML configuration files for paths and parameters
- Environment variable support
- Override hierarchy: CLI args > env vars > config file > defaults

---

## 6. Benefits

### 6.1 Maintainability

- Easier to locate and fix bugs
- Smaller files are easier to understand
- Clear separation of concerns

### 6.2 Testability

- Unit tests for individual components
- Integration tests for workflows
- Easier to mock dependencies

### 6.3 Reusability

- Components can be used in different contexts
- Easier to build new visualizations
- Consistent interfaces across modules

### 6.4 Extensibility

- Add new outlier detectors without touching other code
- Add new plot types without modifying existing plotters
- Add new data sources without changing plotters

### 6.5 Collaboration

- Multiple developers can work on different modules
- Reduced merge conflicts
- Clear ownership of components

---

## 7. Risks and Mitigation

### 7.1 Risk: Breaking Changes

**Mitigation:**
- Maintain backward compatibility during migration
- Run parallel testing (old vs. new)
- Deprecate old scripts only after validation
- Comprehensive integration tests

### 7.2 Risk: Performance Degradation

**Mitigation:**
- Profile code before and after refactoring
- Optimize hot paths if needed
- Use caching where appropriate
- Benchmark against legacy system

### 7.3 Risk: Over-Engineering

**Mitigation:**
- Start with simplest design that works
- Refactor incrementally based on actual needs
- Avoid premature optimization
- Regular code reviews

### 7.4 Risk: Incomplete Migration

**Mitigation:**
- Clear milestones and success criteria
- Dedicated time for migration work
- Prioritize based on most-used features
- Document migration progress

---

## 8. Success Metrics

### 8.1 Code Quality

- [ ] No files exceed 2500 lines
- [ ] No functions exceed 40 lines
- [ ] All functions have type hints
- [ ] Code coverage > 80%

### 8.2 Performance

- [ ] Visualization generation time within 10% of legacy system
- [ ] Memory usage within 10% of legacy system

### 8.3 Maintainability

- [ ] Time to fix bugs reduced by 30%
- [ ] Time to add new features reduced by 40%
- [ ] Onboarding time for new developers reduced by 50%

### 8.4 Reliability

- [ ] Zero regressions in existing functionality
- [ ] All integration tests pass
- [ ] 100% backward compatibility during transition

---

## 9. Next Steps

### 9.1 Immediate (Post-Deadline)

1. Complete current urgent visualizations (1.5 hour deadline)
2. Review this plan with stakeholders
3. Get approval to proceed with migration

### 9.2 Short-Term (Week 1)

1. Begin Phase 1: Extract Core and Utils
2. Set up test framework
3. Create initial configuration files

### 9.3 Long-Term (Weeks 2-6)

1. Execute Phases 2-5 according to timeline
2. Regular progress reviews
3. Adjust plan based on feedback

---

## 10. Appendix

### 10.1 Related Documents

- `DATABASE_SCHEMA_FEATURE_FUNCTIONS_PLAN.md` - Database schema planning
- `INTERACTIVE_SHELL_DEMO.md` - CLI interface documentation

### 10.2 References

- PEP 8: Python Style Guide
- Clean Code by Robert C. Martin
- Design Patterns by Gang of Four
- Refactoring by Martin Fowler

### 10.3 Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0.0 | 2025-11-20 | Kristophor Jensen | Initial plan document |

---

**Document Status:** DRAFT - Pending stakeholder review
