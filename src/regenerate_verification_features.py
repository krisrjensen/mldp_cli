#!/usr/bin/env python3
"""
Regenerate verification feature matrices with correct 90-segment, 8-class dataset
Uses experiment_042_classifier_001_data_splits table instead of file_training_data exclusion
"""

import subprocess
import sys
from pathlib import Path

# Parameters based on previous run
EXPERIMENT_ID = 42
DATA_TYPES = "2,3,4,6"  # ADC8 (ID=2), ADC10 (ID=3), ADC12 (ID=4), ADC6 (ID=6)
DECIMATIONS = "0,7,15,31,63,127"
SEGMENT_SIZE = 8192
AMPLITUDE_METHOD = 2  # Z-score normalization
FEATURE_IDS = "16,18,31,95,97,98,99,101,102,103,105,106,107,109,110,135,136,137"  # 18 features:
# Electrical: voltage(16), current(18)
# Temporal: volatility_dxdt_n1(31)
# Spectral: v_ultra_high_snr(95), v_ultra_high_slope(97), v_ultra_high_sfm(98),
#           c_ultra_high_snr(99), c_ultra_high_slope(101), c_ultra_high_sfm(102),
#           v_full_snr(103), v_full_slope(105), v_full_sfm(106),
#           c_full_snr(107), c_full_slope(109), c_full_sfm(110)
# Statistical: v_kurtosis(135), v_zcr(136), c_kurtosis(137)
WORKERS = 10
BATCH_SIZE_MB = 100

OUTPUT_FOLDER = "/Volumes/ArcData/V3_database/experiment042/classifier_files/verification_features"
RAW_DATA_FOLDER = "/Volumes/ArcData/V3_database/fileset"
ADC_DATA_FOLDER = "/Volumes/ArcData/V3_database/adc_data"
MPCCTL_DIR = "/Volumes/ArcData/V3_database/experiment042/.mpcctl"

print("=" * 80)
print("REGENERATING VERIFICATION FEATURE MATRICES")
print("=" * 80)
print(f"Experiment: {EXPERIMENT_ID}")
print(f"Data types: {DATA_TYPES}")
print(f"Decimations: {DECIMATIONS}")
print(f"Segment size: {SEGMENT_SIZE}")
print(f"Amplitude method: {AMPLITUDE_METHOD}")
print(f"Feature IDs: {FEATURE_IDS}")
print(f"Workers: {WORKERS}")
print(f"Output: {OUTPUT_FOLDER}")
print()
print("This will generate:")
print("  - 7,500 segments (ALL segments from training files)")
print("  - ALL 8 compound classes represented")
print("  - 18 features (complete feature set)")
print("  - 24 feature matrix files (4 ADC types × 6 decimations)")
print()
print("Features included:")
print("  - Electrical: voltage, current")
print("  - Temporal: volatility_dxdt_n1")
print("  - Spectral: v_ultra_high_snr, v_ultra_high_slope, v_ultra_high_sfm")
print("             c_ultra_high_snr, c_ultra_high_slope, c_ultra_high_sfm")
print("             v_full_snr, v_full_slope, v_full_sfm")
print("             c_full_snr, c_full_slope, c_full_sfm")
print("  - Statistical: v_kurtosis, v_zcr, c_kurtosis")
print()

# Run the mpcctl_verification_feature_matrix.py directly
script_path = Path(__file__).parent / "mpcctl_verification_feature_matrix.py"

cmd = [
    "python3",
    str(script_path),
    "--experiment-id", str(EXPERIMENT_ID),
    "--data-type", DATA_TYPES,
    "--decimation", DECIMATIONS,
    "--segment-size", str(SEGMENT_SIZE),
    "--amplitude-method", str(AMPLITUDE_METHOD),
    "--feature-id", FEATURE_IDS,
    "--mpcctl-dir", MPCCTL_DIR,
    "--output-folder", OUTPUT_FOLDER,
    "--input-raw-data-folder", RAW_DATA_FOLDER,
    "--input-adc-data-folder", ADC_DATA_FOLDER,
    "--workers", str(WORKERS),
    "--batch-size-mb", str(BATCH_SIZE_MB)
]

print("Running command:")
print(" ".join(cmd))
print()

try:
    result = subprocess.run(cmd, check=True)
    print("\n" + "=" * 80)
    print("VERIFICATION FEATURE REGENERATION COMPLETE!")
    print("=" * 80)
    sys.exit(0)
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] Command failed with exit code {e.returncode}")
    sys.exit(1)
