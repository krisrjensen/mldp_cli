#!/usr/bin/env python3
"""
Filename: test_visualizer.py
Author(s): Kristophor Jensen
Date Created: 20251119_000000
Date Revised: 20251119_000000
File version: 1.0.0.1
Description: Test script for VerificationFeatureVisualizer

This script validates the visualization module with synthetic data before
running on real verification feature matrices.
"""

import numpy as np
import psycopg2
from pathlib import Path
import tempfile
import shutil
import sys

def create_synthetic_matrix(n_segments=100, n_features=9):
    """
    Create a synthetic feature matrix for testing

    Args:
        n_segments: Number of segments to generate
        n_features: Number of features to generate

    Returns:
        Structured numpy array with segment_id and feature columns
    """
    print(f"\n[TEST] Creating synthetic matrix: {n_segments} segments × {n_features} features")

    # Define feature names (matching experiment 42 features)
    feature_names = [
        'voltage', 'current', 'volatility_dxdt_n1',
        'v_ultra_high_snr', 'v_ultra_high_slope', 'c_ultra_high_snr',
        'v_kurtosis', 'v_zcr', 'c_kurtosis'
    ][:n_features]

    # Create dtype with segment_id + features
    dtype = [('segment_id', np.int32)] + [(name, np.float32) for name in feature_names]

    # Generate synthetic data
    matrix = np.zeros(n_segments, dtype=dtype)
    matrix['segment_id'] = np.arange(1000, 1000 + n_segments)  # Segment IDs starting at 1000

    # Generate feature data with different distributions
    for i, name in enumerate(feature_names):
        if i % 3 == 0:
            # Normal distribution with some outliers
            data = np.random.normal(loc=10.0, scale=2.0, size=n_segments)
            # Add 5% outliers
            n_outliers = max(1, n_segments // 20)
            outlier_indices = np.random.choice(n_segments, n_outliers, replace=False)
            data[outlier_indices] += np.random.choice([-1, 1], n_outliers) * np.random.uniform(10, 20, n_outliers)
        elif i % 3 == 1:
            # Exponential distribution
            data = np.random.exponential(scale=5.0, size=n_segments)
        else:
            # Uniform distribution
            data = np.random.uniform(low=0.0, high=100.0, size=n_segments)

        matrix[name] = data.astype(np.float32)

    print(f"[TEST] Synthetic matrix created successfully")
    print(f"       Segment IDs: {matrix['segment_id'].min()} to {matrix['segment_id'].max()}")
    print(f"       Features: {feature_names}")

    return matrix, feature_names


def test_outlier_detection():
    """Test all outlier detection methods"""
    print("\n" + "="*60)
    print("[TEST] Testing Outlier Detection Methods")
    print("="*60)

    from visualize_verification_features import (
        detect_outliers_iqr,
        detect_outliers_zscore,
        detect_outliers_modified_zscore,
        detect_outliers_isolation_forest
    )

    # Create test data with known outliers
    data = np.concatenate([
        np.random.normal(0, 1, 95),  # Normal data
        np.array([10, -10, 15, -15, 20])  # Clear outliers
    ])

    methods = [
        ('IQR', detect_outliers_iqr),
        ('Z-score', detect_outliers_zscore),
        ('Modified Z-score', detect_outliers_modified_zscore),
        ('Isolation Forest', detect_outliers_isolation_forest)
    ]

    for name, method in methods:
        try:
            mask, metadata = method(data)
            n_outliers = np.sum(mask)
            print(f"\n[TEST] {name:20s}: Detected {n_outliers:3d} outliers ({100*n_outliers/len(data):.1f}%)")
            print(f"       Metadata: {metadata}")
            assert isinstance(mask, np.ndarray), f"{name} did not return mask array"
            assert isinstance(metadata, dict), f"{name} did not return metadata dict"
            print(f"[PASS] {name} method working correctly")
        except Exception as e:
            print(f"[FAIL] {name} method failed: {e}")
            return False

    return True


def test_sigmoid_squashing():
    """Test all sigmoid squashing methods"""
    print("\n" + "="*60)
    print("[TEST] Testing Sigmoid Squashing Methods")
    print("="*60)

    from visualize_verification_features import (
        sigmoid_squash_standard,
        sigmoid_squash_tanh,
        sigmoid_squash_adaptive,
        sigmoid_squash_soft_clip
    )

    # Create test data with outliers
    data = np.concatenate([
        np.random.normal(0, 1, 95),
        np.array([10, -10, 15, -15, 20])
    ])

    methods = [
        ('Standard', sigmoid_squash_standard),
        ('Tanh', sigmoid_squash_tanh),
        ('Adaptive', sigmoid_squash_adaptive),
        ('Soft Clip', sigmoid_squash_soft_clip)
    ]

    for name, method in methods:
        try:
            squashed, metadata = method(data)
            print(f"\n[TEST] {name:20s}:")
            print(f"       Input range:  [{data.min():.2f}, {data.max():.2f}]")
            print(f"       Output range: [{squashed.min():.2f}, {squashed.max():.2f}]")
            print(f"       Metadata: {metadata}")
            assert isinstance(squashed, np.ndarray), f"{name} did not return squashed array"
            assert isinstance(metadata, dict), f"{name} did not return metadata dict"
            assert len(squashed) == len(data), f"{name} changed array length"
            print(f"[PASS] {name} method working correctly")
        except Exception as e:
            print(f"[FAIL] {name} method failed: {e}")
            return False

    return True


def test_visualizer_initialization(db_conn, temp_dir):
    """Test VerificationFeatureVisualizer initialization"""
    print("\n" + "="*60)
    print("[TEST] Testing Visualizer Initialization")
    print("="*60)

    from visualize_verification_features import VerificationFeatureVisualizer

    try:
        visualizer = VerificationFeatureVisualizer(
            db_conn=db_conn,
            output_dir=temp_dir,
            outlier_method='iqr',
            sigmoid_method='adaptive'
        )

        print(f"[TEST] Visualizer initialized successfully")
        print(f"       Output directory: {visualizer.output_dir}")
        print(f"       Outlier method: {visualizer.outlier_method}")
        print(f"       Sigmoid method: {visualizer.sigmoid_method}")
        print(f"       Loaded {len(visualizer.label_map)} segment labels")

        assert visualizer.output_dir.exists(), "Output directory not created"
        assert visualizer.outlier_method == 'iqr', "Outlier method not set correctly"
        assert visualizer.sigmoid_method == 'adaptive', "Sigmoid method not set correctly"

        print(f"[PASS] Visualizer initialization working correctly")
        return visualizer
    except Exception as e:
        print(f"[FAIL] Visualizer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_preparation(visualizer, matrix, feature_names):
    """Test data preparation and version creation"""
    print("\n" + "="*60)
    print("[TEST] Testing Data Preparation")
    print("="*60)

    try:
        versions = visualizer.prepare_data_versions(matrix, feature_names)

        print(f"[TEST] Data versions prepared successfully")
        print(f"       Versions created: {list(versions.keys())}")

        # Check versions
        expected_versions = ['original', 'no_outliers', 'sigmoid']
        for version in expected_versions:
            assert version in versions, f"Missing version: {version}"
            assert isinstance(versions[version], dict), f"Version {version} not a dict"
            n_features = len(versions[version])
            print(f"       {version:12s}: {n_features} features × {len(matrix)} segments")

        # Check metadata structure
        assert 'metadata' in versions, "Missing metadata"
        assert 'segment_ids' in versions, "Missing segment_ids"
        assert 'outlier_masks' in versions, "Missing outlier_masks"

        # Count total outliers across all features
        total_outliers = sum(np.sum(mask) for mask in versions['outlier_masks'].values())
        n_datapoints = len(matrix) * len(feature_names)
        print(f"       Outliers detected: {total_outliers}/{n_datapoints} ({100*total_outliers/n_datapoints:.1f}%)")

        print(f"[PASS] Data preparation working correctly")
        return versions
    except Exception as e:
        print(f"[FAIL] Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_3d_scatter_plot(visualizer, matrix_file):
    """Test 3D scatter plot generation"""
    print("\n" + "="*60)
    print("[TEST] Testing 3D Scatter Plot Generation")
    print("="*60)

    try:
        output_file = visualizer.plot_3d_scatter_multiversion(matrix_file)

        print(f"[TEST] 3D scatter plot generated successfully")
        print(f"       Output file: {output_file.name}")

        assert output_file.exists(), "Output file not created"
        assert output_file.suffix == '.png', "Wrong file format"

        # Check file size (should be reasonable for a plot)
        file_size_kb = output_file.stat().st_size / 1024
        print(f"       File size: {file_size_kb:.1f} KB")
        assert file_size_kb > 10, "File too small (plot may be empty)"
        assert file_size_kb < 10000, "File too large (possible error)"

        print(f"[PASS] 3D scatter plot generation working correctly")
        return True
    except Exception as e:
        print(f"[FAIL] 3D scatter plot generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pdf_plot(visualizer, matrix_file, feature_name):
    """Test PDF plot generation"""
    print("\n" + "="*60)
    print(f"[TEST] Testing PDF Plot Generation: {feature_name}")
    print("="*60)

    grouping_levels = ['population', 'file', 'segment', 'file_segment']

    for grouping in grouping_levels:
        try:
            output_file = visualizer.plot_pdf_multiversion(
                matrix_file=matrix_file,
                feature_name=feature_name,
                grouping_level=grouping
            )

            print(f"[TEST] PDF plot generated: {grouping:15s} -> {output_file.name}")

            assert output_file.exists(), f"Output file not created for {grouping}"
            assert output_file.suffix == '.png', f"Wrong file format for {grouping}"

            # Check file size
            file_size_kb = output_file.stat().st_size / 1024
            assert file_size_kb > 10, f"File too small for {grouping}"
            assert file_size_kb < 10000, f"File too large for {grouping}"

        except Exception as e:
            print(f"[FAIL] PDF plot generation failed for {grouping}: {e}")
            return False

    print(f"[PASS] PDF plot generation working correctly")
    return True


def run_all_tests():
    """Run all validation tests"""
    print("\n" + "#"*60)
    print("# VERIFICATION FEATURE VISUALIZER - TEST SUITE")
    print("#"*60)

    # Connect to database
    print("\n[TEST] Connecting to database...")
    try:
        db_conn = psycopg2.connect(
            host='localhost',
            database='arc_detection',
            user='kjensen'
        )
        print("[TEST] Database connection successful")
    except Exception as e:
        print(f"[FAIL] Database connection failed: {e}")
        return False

    # Create temporary directory for test outputs
    temp_dir = Path(tempfile.mkdtemp(prefix='visualizer_test_'))
    print(f"[TEST] Created temporary directory: {temp_dir}")

    try:
        # Test 1: Outlier detection methods
        if not test_outlier_detection():
            return False

        # Test 2: Sigmoid squashing methods
        if not test_sigmoid_squashing():
            return False

        # Test 3: Visualizer initialization
        visualizer = test_visualizer_initialization(db_conn, temp_dir)
        if visualizer is None:
            return False

        # Create synthetic matrix
        matrix, feature_names = create_synthetic_matrix(n_segments=100, n_features=9)

        # Save matrix to temporary file
        matrix_file = temp_dir / 'test_features_S008192_D000000_R008192_ADC8_A02.npy'
        np.save(matrix_file, matrix)
        print(f"\n[TEST] Saved synthetic matrix to: {matrix_file.name}")

        # Test 4: Data preparation
        versions = test_data_preparation(visualizer, matrix, feature_names)
        if versions is None:
            return False

        # Test 5: 3D scatter plot generation
        if not test_3d_scatter_plot(visualizer, matrix_file):
            return False

        # Test 6: PDF plot generation
        if not test_pdf_plot(visualizer, matrix_file, feature_names[0]):
            return False

        # Success!
        print("\n" + "#"*60)
        print("# ALL TESTS PASSED!")
        print("#"*60)
        print(f"\n[SUCCESS] Test outputs saved to: {temp_dir}")
        print(f"[INFO] Review the generated plots to verify visual quality")
        print(f"[INFO] Run 'rm -rf {temp_dir}' to clean up test files")

        return True

    except Exception as e:
        print(f"\n[FAIL] Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Close database connection
        db_conn.close()
        print(f"\n[TEST] Database connection closed")


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
