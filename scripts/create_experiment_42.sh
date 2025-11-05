# Experiment 42 Creation Script
# Filename: create_experiment_42.sh
# Author(s): Kristophor Jensen
# Date Created: 20251104_000000
# Date Revised: 20251104_000000
# File version: 1.0.0.1
# Description: Create experiment 42 for feature separability analysis
#
# Objective: Search for separability of PSD features across:
# - Segment size: 8192
# - Decimations: 0, 7, 15, 31, 64, 128
# - Data types: adc6, adc8, adc10, adc12
# - Feature grid: {V|C} × {ultra_high|full} × {mean_psd|sfm|slope|snr}
# - Distances: L1, cosine
#
# File selection:
# - 50 arc_transient
# - 50 negative_load_transient
# - 50 parallel_motor_arc
# - At least 1 segment per position label
#
# Scaling: Z-score (remove DC component)

# ============================================================================
# Step 1: Create Experiment 42
# ============================================================================

# Create experiment with configuration
# Note: Replace with actual command based on CLI review

# ============================================================================
# Step 2: Create Features (if needed)
# ============================================================================

# All required features already exist (IDs 95-110):
# - v_ultra_high: snr(95), mean_psd(96), slope(97), sfm(98)
# - c_ultra_high: snr(99), mean_psd(100), slope(101), sfm(102)
# - v_full: snr(103), mean_psd(104), slope(105), sfm(106)
# - c_full: snr(107), mean_psd(108), slope(109), sfm(110)

# No new features needed - skip to step 3

# ============================================================================
# Step 3: Create Feature Sets
# ============================================================================

# Create 16 individual feature sets for separability analysis
# Format: {channel}_{band}_{metric}_fs42

# Voltage Ultra-High Band
create-global-feature-set --name "v_ultra_high_snr_fs42" --category "exp42_separability" --description "Voltage ultra-high band SNR for experiment 42 separability analysis"
add-features-to-set --set-name "v_ultra_high_snr_fs42" --features 95

create-global-feature-set --name "v_ultra_high_mean_psd_fs42" --category "exp42_separability" --description "Voltage ultra-high band mean PSD for experiment 42"
add-features-to-set --set-name "v_ultra_high_mean_psd_fs42" --features 96

create-global-feature-set --name "v_ultra_high_slope_fs42" --category "exp42_separability" --description "Voltage ultra-high band slope for experiment 42"
add-features-to-set --set-name "v_ultra_high_slope_fs42" --features 97

create-global-feature-set --name "v_ultra_high_sfm_fs42" --category "exp42_separability" --description "Voltage ultra-high band SFM for experiment 42"
add-features-to-set --set-name "v_ultra_high_sfm_fs42" --features 98

# Current Ultra-High Band
create-global-feature-set --name "c_ultra_high_snr_fs42" --category "exp42_separability" --description "Current ultra-high band SNR for experiment 42"
add-features-to-set --set-name "c_ultra_high_snr_fs42" --features 99

create-global-feature-set --name "c_ultra_high_mean_psd_fs42" --category "exp42_separability" --description "Current ultra-high band mean PSD for experiment 42"
add-features-to-set --set-name "c_ultra_high_mean_psd_fs42" --features 100

create-global-feature-set --name "c_ultra_high_slope_fs42" --category "exp42_separability" --description "Current ultra-high band slope for experiment 42"
add-features-to-set --set-name "c_ultra_high_slope_fs42" --features 101

create-global-feature-set --name "c_ultra_high_sfm_fs42" --category "exp42_separability" --description "Current ultra-high band SFM for experiment 42"
add-features-to-set --set-name "c_ultra_high_sfm_fs42" --features 102

# Voltage Full Spectrum
create-global-feature-set --name "v_full_snr_fs42" --category "exp42_separability" --description "Voltage full spectrum SNR for experiment 42"
add-features-to-set --set-name "v_full_snr_fs42" --features 103

create-global-feature-set --name "v_full_mean_psd_fs42" --category "exp42_separability" --description "Voltage full spectrum mean PSD for experiment 42"
add-features-to-set --set-name "v_full_mean_psd_fs42" --features 104

create-global-feature-set --name "v_full_slope_fs42" --category "exp42_separability" --description "Voltage full spectrum slope for experiment 42"
add-features-to-set --set-name "v_full_slope_fs42" --features 105

create-global-feature-set --name "v_full_sfm_fs42" --category "exp42_separability" --description "Voltage full spectrum SFM for experiment 42"
add-features-to-set --set-name "v_full_sfm_fs42" --features 106

# Current Full Spectrum
create-global-feature-set --name "c_full_snr_fs42" --category "exp42_separability" --description "Current full spectrum SNR for experiment 42"
add-features-to-set --set-name "c_full_snr_fs42" --features 107

create-global-feature-set --name "c_full_mean_psd_fs42" --category "exp42_separability" --description "Current full spectrum mean PSD for experiment 42"
add-features-to-set --set-name "c_full_mean_psd_fs42" --features 108

create-global-feature-set --name "c_full_slope_fs42" --category "exp42_separability" --description "Current full spectrum slope for experiment 42"
add-features-to-set --set-name "c_full_slope_fs42" --features 109

create-global-feature-set --name "c_full_sfm_fs42" --category "exp42_separability" --description "Current full spectrum SFM for experiment 42"
add-features-to-set --set-name "c_full_sfm_fs42" --features 110

# ============================================================================
# Step 4: Link Feature Sets to Experiment 42
# ============================================================================

# Link all 16 feature sets to experiment
# Note: Use actual feature set IDs after creation (will be 25-40 if sequential)
# Format: link-feature-set <id>

# ============================================================================
# Step 5: Configure Experiment
# ============================================================================

# Set segment size
# update-selection-config --segment-size 8192

# Configure segment selection strategy
# update-selection-config --strategy position_balanced_per_file --balanced --random-seed 42

# ============================================================================
# Step 6: Select Training Files
# ============================================================================

# Select 50 files from each label
select-files arc_transient --count 50 --strategy random --seed 42
select-files negative_load_transient --count 50 --strategy random --seed 42
select-files parallel_motor_arc --count 50 --strategy random --seed 42

# ============================================================================
# Step 7: Select Training Segments
# ============================================================================

# Select segments with position balancing (at least 1 per position)
select-segments --balanced --position-balance at_least_one

# ============================================================================
# Step 8: Generate Segment Pairs
# ============================================================================

# Generate segment pairs for distance calculations
generate-segment-pairs

# ============================================================================
# Step 9: Add Data Types and Decimations
# ============================================================================

# Add data types: adc6, adc8, adc10, adc12
add-data-type adc6
add-data-type adc8
add-data-type adc10
add-data-type adc12

# Add decimations: 0, 7, 15, 31, 64, 128
add-decimation 0
add-decimation 7
add-decimation 15
add-decimation 31
add-decimation 64
add-decimation 128

# ============================================================================
# Step 10: Generate Segment Fileset
# ============================================================================

# Generate segment files for all data types and decimations
generate-segment-fileset

# ============================================================================
# Step 11: Generate Feature Fileset
# ============================================================================

# Generate features using z-score scaling
generate-feature-fileset --scaling zscore

# ============================================================================
# Step 12: Add Distance Metrics
# ============================================================================

# Add L1 and cosine distance metrics
add-distance-metric l1
add-distance-metric cosine

# ============================================================================
# Step 13: Compute Distances (MPCCTL)
# ============================================================================

# Compute distances using multiprocessing
# mpcctl-distance-function --workers 20 --log --verbose

# ============================================================================
# Step 14: Insert Distances (MPCCTL)
# ============================================================================

# Insert computed distances to database
# mpcctl-distance-insert --workers 10 --log --verbose

# ============================================================================
# Step 15: Generate Heatmap Plots
# ============================================================================

# Generate heatmaps for visualization
# heatmap --output-dir plots/experiment_42

# ============================================================================
# End of Experiment 42 Setup
# ============================================================================
