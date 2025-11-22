#!/usr/bin/env python3
"""
Filename: visualize_verification_features.py
Author(s): Kristophor Jensen
Date Created: 20251116_000000
Date Revised: 20251116_000000
File version: 1.0.0.0
Description: Comprehensive visualization suite for verification feature matrices
             with outlier detection, sigmoid squashing, and multi-version plotting

Features:
- 3D scatter plots with 6 views (XY, YZ, XZ + 3 perspectives)
- PDF plots with 4 grouping levels (file_label.segment_label, segment_label, file_label, population)
- 3 versions per plot: Original, Outliers Removed, Sigmoid Squashed
- 4 outlier detection methods: IQR, Z-score, Modified Z-score, Isolation Forest
- 4 sigmoid squashing methods: Standard, Tanh, Adaptive, Soft Clip
- Metadata export (JSON) with outlier counts and scaling parameters
"""

import os
import sys
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import psycopg2
from scipy.ndimage import gaussian_filter1d
import psycopg2.extras
from scipy import stats
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import LocallyLinearEmbedding, TSNE
from itertools import combinations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Outlier Detection Functions
# ============================================================================

def detect_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, Dict]:
    """
    Interquartile range method for outlier detection.

    Outliers: values < Q1 - factor*IQR or > Q3 + factor*IQR

    Args:
        data: Input array
        factor: IQR multiplier (1.5=standard, 3.0=extreme)

    Returns:
        outlier_mask: Boolean array (True = outlier)
        metadata: Dict with Q1, Q3, IQR, bounds
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    outlier_mask = (data < lower_bound) | (data > upper_bound)

    metadata = {
        'method': 'iqr',
        'factor': factor,
        'Q1': float(Q1),
        'Q3': float(Q3),
        'IQR': float(IQR),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'outlier_count': int(np.sum(outlier_mask)),
        'outlier_percentage': float(np.mean(outlier_mask) * 100)
    }

    return outlier_mask, metadata


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, Dict]:
    """
    Z-score method for outlier detection.

    Outliers: |z-score| > threshold

    Args:
        data: Input array
        threshold: Z-score threshold (3.0=standard, 2.5=aggressive)

    Returns:
        outlier_mask: Boolean array
        metadata: Dict with mean, std, threshold
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std) if std > 0 else np.zeros_like(data)

    outlier_mask = z_scores > threshold

    metadata = {
        'method': 'zscore',
        'threshold': threshold,
        'mean': float(mean),
        'std': float(std),
        'outlier_count': int(np.sum(outlier_mask)),
        'outlier_percentage': float(np.mean(outlier_mask) * 100)
    }

    return outlier_mask, metadata


def detect_outliers_modified_zscore(data: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, Dict]:
    """
    Modified Z-score using median absolute deviation (MAD).
    More robust to outliers than standard Z-score.

    Args:
        data: Input array
        threshold: Modified Z-score threshold

    Returns:
        outlier_mask: Boolean array
        metadata: Dict with median, MAD, threshold
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    if mad > 0:
        modified_z_scores = 0.6745 * np.abs(data - median) / mad
    else:
        modified_z_scores = np.zeros_like(data)

    outlier_mask = modified_z_scores > threshold

    metadata = {
        'method': 'modified_zscore',
        'threshold': threshold,
        'median': float(median),
        'mad': float(mad),
        'outlier_count': int(np.sum(outlier_mask)),
        'outlier_percentage': float(np.mean(outlier_mask) * 100)
    }

    return outlier_mask, metadata


def detect_outliers_isolation_forest(data: np.ndarray, contamination: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    Isolation Forest ML-based outlier detection.

    Args:
        data: Input array
        contamination: Expected proportion of outliers

    Returns:
        outlier_mask: Boolean array
        metadata: Dict with contamination, outlier count
    """
    data_2d = data.reshape(-1, 1)

    clf = IsolationForest(contamination=contamination, random_state=42)
    outlier_labels = clf.fit_predict(data_2d)

    outlier_mask = outlier_labels == -1

    metadata = {
        'method': 'isolation_forest',
        'contamination': contamination,
        'outlier_count': int(np.sum(outlier_mask)),
        'outlier_percentage': float(np.mean(outlier_mask) * 100)
    }

    return outlier_mask, metadata


# ============================================================================
# Sigmoid Squashing Functions
# ============================================================================

def sigmoid_squash_standard(data: np.ndarray, k: float = 1.0, x0: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """
    Standard logistic sigmoid: 1 / (1 + exp(-k*(x - x0)))

    Args:
        data: Input array
        k: Steepness parameter
        x0: Midpoint (default: median)

    Returns:
        squashed: Squashed array (range 0-1)
        metadata: Dict with k, x0
    """
    if x0 is None:
        x0 = np.median(data)

    squashed = 1.0 / (1.0 + np.exp(-k * (data - x0)))

    metadata = {
        'method': 'standard',
        'k': float(k),
        'x0': float(x0),
        'output_range': [float(np.min(squashed)), float(np.max(squashed))]
    }

    return squashed, metadata


def sigmoid_squash_tanh(data: np.ndarray, k: float = 1.0, x0: Optional[float] = None) -> Tuple[np.ndarray, Dict]:
    """
    Hyperbolic tangent sigmoid: tanh(k*(x - x0))

    Args:
        data: Input array
        k: Steepness parameter
        x0: Midpoint (default: median)

    Returns:
        squashed: Squashed array (range -1 to 1)
        metadata: Dict with k, x0
    """
    if x0 is None:
        x0 = np.median(data)

    squashed = np.tanh(k * (data - x0))

    metadata = {
        'method': 'tanh',
        'k': float(k),
        'x0': float(x0),
        'output_range': [float(np.min(squashed)), float(np.max(squashed))]
    }

    return squashed, metadata


def sigmoid_squash_adaptive(data: np.ndarray, iqr_factor: float = 1.5) -> Tuple[np.ndarray, Dict]:
    """
    Adaptive sigmoid based on IQR.
    Automatically scales k based on data distribution.

    Maps:
    - Median → 0.5
    - Q1 - iqr_factor*IQR → ~0.05
    - Q3 + iqr_factor*IQR → ~0.95

    Args:
        data: Input array
        iqr_factor: IQR multiplier for scaling

    Returns:
        squashed: Squashed array (range 0-1)
        metadata: Dict with parameters
    """
    median = np.median(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Calculate k to map ±iqr_factor*IQR to sigmoid(±3) ≈ 0.05/0.95
    if IQR > 0:
        k = 3.0 / (IQR * iqr_factor)
    else:
        k = 1.0

    squashed = 1.0 / (1.0 + np.exp(-k * (data - median)))

    metadata = {
        'method': 'adaptive',
        'iqr_factor': float(iqr_factor),
        'median': float(median),
        'IQR': float(IQR),
        'k': float(k),
        'output_range': [float(np.min(squashed)), float(np.max(squashed))]
    }

    return squashed, metadata


def sigmoid_squash_soft_clip(data: np.ndarray, lower_percentile: float = 5,
                              upper_percentile: float = 95) -> Tuple[np.ndarray, Dict]:
    """
    Soft clipping using sigmoid at percentile boundaries.
    Preserves central data, compresses tails.

    Args:
        data: Input array
        lower_percentile: Lower boundary percentile
        upper_percentile: Upper boundary percentile

    Returns:
        squashed: Soft-clipped array
        metadata: Dict with parameters
    """
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)

    result = np.copy(data)

    # Lower tail compression
    lower_mask = data < lower
    if np.any(lower_mask) and np.min(data[lower_mask]) < lower:
        data_min = np.min(data[lower_mask])
        k = 1.0 / (lower - data_min) if (lower - data_min) > 0 else 1.0
        sigmoid_vals = 1.0 / (1.0 + np.exp(-k * (data[lower_mask] - lower)))
        result[lower_mask] = lower - (lower - data_min) * (1 - sigmoid_vals)

    # Upper tail compression
    upper_mask = data > upper
    if np.any(upper_mask) and np.max(data[upper_mask]) > upper:
        data_max = np.max(data[upper_mask])
        k = 1.0 / (data_max - upper) if (data_max - upper) > 0 else 1.0
        sigmoid_vals = 1.0 / (1.0 + np.exp(-k * (data[upper_mask] - upper)))
        result[upper_mask] = upper + (data_max - upper) * sigmoid_vals

    metadata = {
        'method': 'soft_clip',
        'lower_percentile': lower_percentile,
        'upper_percentile': upper_percentile,
        'lower_bound': float(lower),
        'upper_bound': float(upper),
        'output_range': [float(np.min(result)), float(np.max(result))]
    }

    return result, metadata


# ============================================================================
# Main Visualizer Class
# ============================================================================

class VerificationFeatureVisualizer:
    """
    Comprehensive visualization suite for verification feature matrices.
    """

    def __init__(self, db_conn, output_dir: Path,
                 outlier_method: str = 'iqr',
                 outlier_params: Optional[Dict] = None,
                 sigmoid_method: str = 'adaptive',
                 sigmoid_params: Optional[Dict] = None):
        """
        Initialize visualizer.

        Args:
            db_conn: Database connection
            output_dir: Output directory path
            outlier_method: 'iqr', 'zscore', 'modified_zscore', 'isolation_forest'
            outlier_params: Parameters for outlier detection
            sigmoid_method: 'standard', 'tanh', 'adaptive', 'soft_clip'
            sigmoid_params: Parameters for sigmoid squashing
        """
        self.db_conn = db_conn
        self.output_dir = Path(output_dir)
        self.outlier_method = outlier_method
        self.outlier_params = outlier_params or {}
        self.sigmoid_method = sigmoid_method
        self.sigmoid_params = sigmoid_params or {}

        # Load labels from database
        self.label_map = self._load_labels()

        # Create output directories
        (self.output_dir / '3d_scatter').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'pdfs').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'summary').mkdir(parents=True, exist_ok=True)

        logger.info(f"Visualizer initialized: outlier={outlier_method}, sigmoid={sigmoid_method}")

    def _load_labels(self) -> Dict:
        """
        Load segment and file labels from database.

        Returns:
            Dict with 'segment_labels', 'label_names', 'file_labels', 'compound_labels'
        """
        cursor = self.db_conn.cursor()

        # Get segment labels
        cursor.execute("SELECT segment_id, segment_label_id FROM data_segments")
        segment_labels = dict(cursor.fetchall())

        # Get label names
        cursor.execute("SELECT label_id, label_name FROM segment_labels")
        label_names = dict(cursor.fetchall())

        # Get file labels
        cursor.execute("""
            SELECT ds.segment_id, el.experiment_label
            FROM data_segments ds
            JOIN files_x fx ON ds.experiment_file_id = fx.file_id
            JOIN files_y fy ON fx.file_id = fy.file_id
            JOIN experiment_labels el ON fy.label_id = el.label_id
        """)
        file_labels = dict(cursor.fetchall())

        # Get compound labels (file_label.segment_label)
        cursor.execute("""
            SELECT
                ds.segment_id,
                el.experiment_label || '.' || sl.label_name as compound_label
            FROM data_segments ds
            JOIN segment_labels sl ON ds.segment_label_id = sl.label_id
            JOIN files_x fx ON ds.experiment_file_id = fx.file_id
            JOIN files_y fy ON fx.file_id = fy.file_id
            JOIN experiment_labels el ON fy.label_id = el.label_id
        """)
        compound_labels = dict(cursor.fetchall())

        logger.info(f"Loaded labels: {len(segment_labels)} segments, {len(label_names)} label types, {len(file_labels)} file mappings, {len(compound_labels)} compound labels")

        return {
            'segment_labels': segment_labels,
            'label_names': label_names,
            'file_labels': file_labels,
            'compound_labels': compound_labels
        }

    def detect_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Detect outliers using configured method.

        Args:
            data: Input array

        Returns:
            outlier_mask: Boolean array
            metadata: Dict with detection parameters
        """
        if self.outlier_method == 'iqr':
            return detect_outliers_iqr(data, **self.outlier_params)
        elif self.outlier_method == 'zscore':
            return detect_outliers_zscore(data, **self.outlier_params)
        elif self.outlier_method == 'modified_zscore':
            return detect_outliers_modified_zscore(data, **self.outlier_params)
        elif self.outlier_method == 'isolation_forest':
            return detect_outliers_isolation_forest(data, **self.outlier_params)
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")

    def apply_sigmoid(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Apply sigmoid squashing using configured method.

        Args:
            data: Input array

        Returns:
            squashed: Squashed array
            metadata: Dict with squashing parameters
        """
        if self.sigmoid_method == 'standard':
            return sigmoid_squash_standard(data, **self.sigmoid_params)
        elif self.sigmoid_method == 'tanh':
            return sigmoid_squash_tanh(data, **self.sigmoid_params)
        elif self.sigmoid_method == 'adaptive':
            return sigmoid_squash_adaptive(data, **self.sigmoid_params)
        elif self.sigmoid_method == 'soft_clip':
            return sigmoid_squash_soft_clip(data, **self.sigmoid_params)
        else:
            raise ValueError(f"Unknown sigmoid method: {self.sigmoid_method}")

    def prepare_data_versions(self, data: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Prepare 3 versions of data: original, outliers removed, sigmoid squashed.

        CRITICAL: Outlier detection is performed WITHIN each compound label group
        (file_label.segment_label), not across the entire population.

        Args:
            data: Structured numpy array with named fields
            feature_names: List of feature names (excluding segment_id)

        Returns:
            Dict with keys:
            - 'original': Dict[feature_name] -> values
            - 'no_outliers': Dict[feature_name] -> values (NaN for outliers)
            - 'sigmoid': Dict[feature_name] -> sigmoid-squashed values
            - 'outlier_masks': Dict[feature_name] -> boolean mask
            - 'metadata': Dict with outlier/sigmoid metadata per feature
        """
        segment_ids = data['segment_id']
        n_segments = len(segment_ids)

        # Get compound labels for all segments
        compound_labels = np.array([
            self.label_map['compound_labels'].get(sid, 'Unknown')
            for sid in segment_ids
        ])
        unique_labels = np.unique(compound_labels)

        original = {}
        no_outliers = {}
        sigmoid = {}
        outlier_masks = {}
        metadata = {}

        for feature_name in feature_names:
            # Extract feature values
            feature_values = data[feature_name].astype(float)

            # Original version
            original[feature_name] = feature_values

            # Detect outliers WITHIN each compound label group
            outlier_mask = np.zeros(n_segments, dtype=bool)
            group_metadata = {}

            for label in unique_labels:
                # Find segments belonging to this label
                label_mask = compound_labels == label
                label_indices = np.where(label_mask)[0]

                # Skip if too few samples in this group
                if len(label_indices) < 4:
                    continue

                # Extract feature values for this label group only
                label_feature_values = feature_values[label_indices]

                # Detect outliers within this label group
                label_outlier_mask, label_outlier_meta = self.detect_outliers(label_feature_values)

                # Map back to full array
                outlier_mask[label_indices] = label_outlier_mask

                # Store per-group metadata
                group_metadata[label] = label_outlier_meta

            outlier_masks[feature_name] = outlier_mask

            # No outliers version (set outliers to NaN)
            no_outlier_values = feature_values.copy()
            no_outlier_values[outlier_mask] = np.nan
            no_outliers[feature_name] = no_outlier_values

            # Sigmoid squashed version
            sigmoid_values, sigmoid_meta = self.apply_sigmoid(feature_values)
            sigmoid[feature_name] = sigmoid_values

            # Store metadata
            metadata[feature_name] = {
                'outlier_detection': {
                    'method': self.outlier_method,
                    'total_outliers': int(np.sum(outlier_mask)),
                    'outlier_percentage': float(np.mean(outlier_mask) * 100),
                    'per_group': group_metadata
                },
                'sigmoid_squashing': sigmoid_meta,
                'statistics': {
                    'mean': float(np.mean(feature_values)),
                    'median': float(np.median(feature_values)),
                    'std': float(np.std(feature_values)),
                    'min': float(np.min(feature_values)),
                    'max': float(np.max(feature_values)),
                    'Q1': float(np.percentile(feature_values, 25)),
                    'Q3': float(np.percentile(feature_values, 75))
                }
            }

        return {
            'segment_ids': segment_ids,
            'original': original,
            'no_outliers': no_outliers,
            'sigmoid': sigmoid,
            'outlier_masks': outlier_masks,
            'metadata': metadata
        }

    def plot_3d_scatter_multiversion(self, matrix_file: Path):
        """
        Generate 3-version × 6-panel 3D scatter plot (18 subplots).

        Layout:
        Row 1: Original (XY, YZ, XZ, Persp1, Persp2, Persp3)
        Row 2: No outliers (XY, YZ, XZ, Persp1, Persp2, Persp3)
        Row 3: Sigmoid (XY, YZ, XZ, Persp1, Persp2, Persp3)

        Args:
            matrix_file: Path to .npy matrix file
        """
        logger.info(f"Generating 3D scatter plot for {matrix_file.name}")

        # Load data
        data = np.load(matrix_file)
        feature_names = [n for n in data.dtype.names if n != 'segment_id']

        # Filter to scalar features (exclude voltage and current) for dimensionality reduction
        scalar_features = [n for n in feature_names if n not in ['voltage', 'current']]
        logger.info(f"Using {len(scalar_features)} scalar features for PCA (excluding voltage and current)")

        # Prepare 3 versions
        data_versions = self.prepare_data_versions(data, feature_names)
        segment_ids = data_versions['segment_ids']

        # Apply PCA for dimensionality reduction to 3D
        # Use original data for PCA fit - use scalar features only
        feature_matrix_orig = np.column_stack([data_versions['original'][f] for f in scalar_features])

        # Handle NaN values in original data
        # Identify rows with any NaN values
        nan_mask = np.isnan(feature_matrix_orig).any(axis=1)
        n_nan = np.sum(nan_mask)
        if n_nan > 0:
            logger.warning(f"Found {n_nan} segments with NaN values ({100*n_nan/len(feature_matrix_orig):.1f}%), excluding from PCA")

        # Filter out NaN rows for PCA fitting
        valid_mask = ~nan_mask
        feature_matrix_orig_clean = feature_matrix_orig[valid_mask]

        if len(scalar_features) > 3:
            pca = PCA(n_components=3)
            coords_orig_valid = pca.fit_transform(feature_matrix_orig_clean)
            axis_labels = [f'PC{i+1} ({pca.explained_variance_ratio_[i]:.1%})' for i in range(3)]

            # Create full coordinate array with NaN for invalid samples
            coords_orig = np.full((len(feature_matrix_orig), 3), np.nan)
            coords_orig[valid_mask] = coords_orig_valid

            # Transform other versions using same PCA - use scalar features only
            # For no_outliers: replace NaN with 0 for visualization
            feature_matrix_clean = np.column_stack([
                np.nan_to_num(data_versions['no_outliers'][f], nan=0.0) for f in scalar_features
            ])
            coords_clean_valid = pca.transform(feature_matrix_clean[valid_mask])
            coords_clean = np.full((len(feature_matrix_clean), 3), np.nan)
            coords_clean[valid_mask] = coords_clean_valid

            # For sigmoid: replace NaN with 0 and filter to valid samples
            # Sigmoid preserves NaN from original + may introduce new NaN from squashing
            feature_matrix_sigmoid = np.column_stack([
                np.nan_to_num(data_versions['sigmoid'][f], nan=0.0) for f in scalar_features
            ])
            coords_sigmoid_valid = pca.transform(feature_matrix_sigmoid[valid_mask])
            coords_sigmoid = np.full((len(feature_matrix_sigmoid), 3), np.nan)
            coords_sigmoid[valid_mask] = coords_sigmoid_valid
        else:
            coords_orig = feature_matrix_orig[:, :3]
            coords_clean = np.column_stack([
                np.nan_to_num(data_versions['no_outliers'][f], nan=0.0) for f in scalar_features[:3]
            ])
            coords_sigmoid = np.column_stack([data_versions['sigmoid'][f] for f in scalar_features[:3]])
            axis_labels = scalar_features[:3]

        # Get label colors using compound labels (file_label.segment_label)
        labels = [self.label_map['compound_labels'].get(sid, 'Unknown') for sid in segment_ids]
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
        point_colors = [label_to_color[l] for l in labels]

        # Create figure with 3 rows × 6 columns
        fig = plt.figure(figsize=(30, 15))

        versions = [
            ('Original', coords_orig),
            ('No Outliers', coords_clean),
            ('Sigmoid Squashed', coords_sigmoid)
        ]

        perspectives = [
            (30, 45, 'Perspective 1 (45°, 30°)'),
            (30, 135, 'Perspective 2 (135°, 30°)'),
            (45, 225, 'Perspective 3 (225°, 45°)')
        ]

        for row, (version_name, coords) in enumerate(versions):
            # XY projection
            ax = fig.add_subplot(3, 6, row*6 + 1)
            ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=5, alpha=0.6)
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.set_title(f'{version_name}: XY Projection')
            ax.grid(True, alpha=0.3)

            # YZ projection
            ax = fig.add_subplot(3, 6, row*6 + 2)
            ax.scatter(coords[:, 1], coords[:, 2], c=point_colors, s=5, alpha=0.6)
            ax.set_xlabel(axis_labels[1])
            ax.set_ylabel(axis_labels[2])
            ax.set_title(f'{version_name}: YZ Projection')
            ax.grid(True, alpha=0.3)

            # XZ projection
            ax = fig.add_subplot(3, 6, row*6 + 3)
            ax.scatter(coords[:, 0], coords[:, 2], c=point_colors, s=5, alpha=0.6)
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[2])
            ax.set_title(f'{version_name}: XZ Projection')
            ax.grid(True, alpha=0.3)

            # 3D perspectives
            for col, (elev, azim, title) in enumerate(perspectives):
                ax = fig.add_subplot(3, 6, row*6 + 4 + col, projection='3d')
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                          c=point_colors, s=5, alpha=0.6)
                ax.view_init(elev=elev, azim=azim)
                ax.set_xlabel(axis_labels[0])
                ax.set_ylabel(axis_labels[1])
                ax.set_zlabel(axis_labels[2])
                ax.set_title(f'{version_name}: {title}')

        # Add legend with compound labels
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=label_to_color[l],
                      markersize=8,
                      label=l)  # l is already the compound label string
            for l in unique_labels if l != 'Unknown'
        ]
        fig.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=8)

        plt.suptitle(f'3D Scatter Plots: {matrix_file.stem}', fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save
        output_path = self.output_dir / '3d_scatter' / f'{matrix_file.stem}_3d_multiversion.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved 3D scatter plot: {output_path}")
        return output_path

    def plot_pdf_multiversion(self, matrix_file: Path, feature_name: str, grouping_level: str):
        """
        Generate 3-panel PDF plot (original, no outliers, sigmoid).

        Args:
            matrix_file: Path to .npy matrix file
            feature_name: Feature to plot
            grouping_level: 'file_label.segment_label', 'segment_label', 'file_label', 'entire_population'
        """
        # Load data
        data = np.load(matrix_file)
        feature_names = [n for n in data.dtype.names if n != 'segment_id']

        if feature_name not in feature_names:
            logger.warning(f"Feature {feature_name} not in {matrix_file.name}, skipping")
            return

        # Prepare 3 versions
        data_versions = self.prepare_data_versions(data, feature_names)
        segment_ids = data_versions['segment_ids']

        # Get grouping labels using compound labels
        if grouping_level == 'file_segment':
            # Use compound labels directly (file_label.segment_label)
            groups = [self.label_map['compound_labels'].get(sid, 'Unknown') for sid in segment_ids]
        elif grouping_level == 'segment':
            # Extract segment label from compound label
            groups = [
                self.label_map['compound_labels'].get(sid, 'Unknown').split('.')[-1]
                if '.' in self.label_map['compound_labels'].get(sid, '')
                else self.label_map['compound_labels'].get(sid, 'Unknown')
                for sid in segment_ids
            ]
        elif grouping_level == 'file':
            # Extract file label from compound label
            groups = [
                self.label_map['compound_labels'].get(sid, 'Unknown').split('.')[0]
                if '.' in self.label_map['compound_labels'].get(sid, '')
                else 'Unknown'
                for sid in segment_ids
            ]
        else:  # population
            groups = ['all'] * len(segment_ids)

        # Group data for all 3 versions
        unique_groups = sorted(set(groups))
        if len(unique_groups) > 50:
            logger.warning(f"Too many groups ({len(unique_groups)}), limiting to top 20")
            # Take top 20 by count
            group_counts = {g: groups.count(g) for g in unique_groups}
            unique_groups = sorted(group_counts.keys(), key=lambda x: group_counts[x], reverse=True)[:20]

        # Create 2-panel figure (Original and No Outliers only)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        versions_data = [
            ('Original', data_versions['original'][feature_name]),
            ('No Outliers', data_versions['no_outliers'][feature_name])
        ]

        # Plot normalized histogram - Y-axis shows PROBABILITY (0 to 1)

        for ax, (version_name, feature_values) in zip(axes, versions_data):
            # Get min/max for THIS VERSION to set up x-axis mapping
            all_valid_values = feature_values[~np.isnan(feature_values)]
            if len(all_valid_values) > 0:
                version_min = np.min(all_valid_values)
                version_max = np.max(all_valid_values)
                version_range = version_max - version_min
            else:
                version_min = 0.0
                version_max = 1.0
                version_range = 1.0

            for i, group in enumerate(unique_groups):
                group_mask = np.array([g == group for g in groups])
                group_values = feature_values[group_mask]

                # Remove NaNs
                group_values = group_values[~np.isnan(group_values)]

                if len(group_values) > 1 and version_range > 0:
                    try:
                        # Create normalized histogram with 100 bins for better resolution
                        counts, bin_edges = np.histogram(group_values, bins=100,
                                                        range=(version_min, version_max))

                        # Normalize: divide by total samples to get PROBABILITY
                        probabilities = counts / len(group_values)

                        # Apply Gaussian smoothing to probability distribution (sigma=2.0 for smooth curves)
                        probabilities_smooth = gaussian_filter1d(probabilities, sigma=2.0)

                        # Bin centers for plotting
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                        # Map x-axis to [0,1] for visualization
                        bin_centers_norm = (bin_centers - version_min) / version_range

                        # Plot smoothed probability distribution curve
                        ax.plot(bin_centers_norm, probabilities_smooth,
                               label=str(group),
                               color=plt.cm.tab20(i / max(len(unique_groups), 1)),
                               linewidth=1.5, alpha=0.8)
                    except Exception as e:
                        # Fallback to step plot if smoothing fails
                        try:
                            ax.step(bin_centers_norm, probabilities, where='mid',
                                   label=str(group),
                                   color=plt.cm.tab20(i / max(len(unique_groups), 1)),
                                   linewidth=1.5)
                        except:
                            pass

            # Set x-axis tick labels to show actual feature values for this version
            tick_positions = [0, 0.25, 0.5, 0.75, 1.0]
            tick_labels = [f'{version_min + p * version_range:.3f}' for p in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)

            ax.set_xlabel(feature_name)
            ax.set_ylabel('Probability')
            ax.set_title(f'{version_name}')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(0, None)  # Auto-scale y

            if len(unique_groups) <= 10 and ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc='best')

        plt.suptitle(f'{feature_name} PDFs - Grouped by {grouping_level}\n{matrix_file.stem}',
                     fontsize=12)
        plt.tight_layout()

        # Save
        output_dir = self.output_dir / 'pdfs' / feature_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{matrix_file.stem}_{grouping_level}_pdf_multiversion.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved PDF plot: {output_path}")
        return output_path

    def generate_summary_report(self, matrix_file: Path, data_versions: Dict):
        """
        Generate summary metadata JSON for a matrix file.

        Args:
            matrix_file: Path to .npy matrix file
            data_versions: Output from prepare_data_versions()
        """
        metadata = data_versions['metadata']

        summary = {
            'file': matrix_file.name,
            'n_segments': len(data_versions['segment_ids']),
            'n_features': len(metadata),
            'outlier_detection_method': self.outlier_method,
            'sigmoid_method': self.sigmoid_method,
            'features': metadata
        }

        # Save to JSON
        output_path = self.output_dir / 'metadata' / f'{matrix_file.stem}_summary.json'
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved metadata: {output_path}")

    def generate_all_visualizations(self, matrix_files: List[Path],
                                    specific_feature: Optional[str] = None,
                                    specific_grouping: Optional[str] = None):
        """
        Generate all visualizations for all matrix files.

        Args:
            matrix_files: List of .npy matrix file paths
            specific_feature: If set, only process this feature
            specific_grouping: If set, only process this grouping level
        """
        grouping_levels = ['file_label.segment_label', 'segment_label', 'file_label', 'entire_population']
        if specific_grouping:
            grouping_levels = [specific_grouping]

        for i, matrix_file in enumerate(matrix_files):
            logger.info(f"Processing {i+1}/{len(matrix_files)}: {matrix_file.name}")

            try:
                # Load data and prepare versions once
                data = np.load(matrix_file)
                feature_names = [n for n in data.dtype.names if n != 'segment_id']
                data_versions = self.prepare_data_versions(data, feature_names)

                # 3D scatter plot
                self.plot_3d_scatter_multiversion(matrix_file)

                # Generate summary metadata
                self.generate_summary_report(matrix_file, data_versions)

                # PDF plots for each feature and grouping level
                features_to_plot = [specific_feature] if specific_feature else feature_names

                for feature_name in features_to_plot:
                    if feature_name not in feature_names:
                        continue

                    for grouping in grouping_levels:
                        self.plot_pdf_multiversion(matrix_file, feature_name, grouping)

            except Exception as e:
                logger.error(f"Failed to process {matrix_file.name}: {e}", exc_info=True)

        logger.info(f"Completed processing {len(matrix_files)} matrix files")


def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize verification feature matrices')
    parser.add_argument('--matrix-dir', type=str,
                       default='/Volumes/ArcData/V3_database/experiment042/classifier_files/verification_features',
                       help='Directory containing .npy matrix files')
    parser.add_argument('--output-dir', type=str,
                       default='/Volumes/ArcData/V3_database/experiment042/classifier_files/verification_plots',
                       help='Output directory for plots')
    parser.add_argument('--outlier-method', type=str, default='iqr',
                       choices=['iqr', 'zscore', 'modified_zscore', 'isolation_forest'],
                       help='Outlier detection method')
    parser.add_argument('--iqr-factor', type=float, default=1.5,
                       help='IQR factor for outlier detection')
    parser.add_argument('--sigmoid-method', type=str, default='adaptive',
                       choices=['standard', 'tanh', 'adaptive', 'soft_clip'],
                       help='Sigmoid squashing method')
    parser.add_argument('--sigmoid-k', type=float, default=1.0,
                       help='Sigmoid steepness parameter')
    parser.add_argument('--matrix-file', type=str, default=None,
                       help='Process specific matrix file only')
    parser.add_argument('--feature', type=str, default=None,
                       help='Process specific feature only')
    parser.add_argument('--grouping', type=str, default=None,
                       choices=['file_label.segment_label', 'segment_label', 'file_label', 'entire_population'],
                       help='Process specific grouping level only')

    args = parser.parse_args()

    # Database connection
    db_conn = psycopg2.connect(
        host='localhost',
        port=5432,
        database='arc_detection',
        user=os.environ.get('USER', 'kjensen')
    )

    # Find matrix files
    matrix_dir = Path(args.matrix_dir)
    if args.matrix_file:
        matrix_files = [matrix_dir / args.matrix_file]
    else:
        matrix_files = sorted(matrix_dir.glob('features_*.npy'))

    if not matrix_files:
        logger.error(f"No matrix files found in {matrix_dir}")
        return 1

    logger.info(f"Found {len(matrix_files)} matrix files to process")

    # Set up outlier and sigmoid parameters
    outlier_params = {}
    if args.outlier_method == 'iqr':
        outlier_params['factor'] = args.iqr_factor

    sigmoid_params = {}
    if args.sigmoid_method in ['standard', 'tanh']:
        sigmoid_params['k'] = args.sigmoid_k

    # Create visualizer
    visualizer = VerificationFeatureVisualizer(
        db_conn=db_conn,
        output_dir=args.output_dir,
        outlier_method=args.outlier_method,
        outlier_params=outlier_params,
        sigmoid_method=args.sigmoid_method,
        sigmoid_params=sigmoid_params
    )

    # Generate all visualizations
    visualizer.generate_all_visualizations(
        matrix_files=matrix_files,
        specific_feature=args.feature,
        specific_grouping=args.grouping
    )

    logger.info("All visualizations complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
