#!/usr/bin/env python3
"""
Filename: visualize_dimreduction.py
Author(s): Kristophor Jensen
Date Created: 20251119_000000
Date Revised: 20251119_000000
File version: 1.0.0.1
Description: Advanced dimensionality reduction visualizations for verification features
             Includes: t-SNE, LLE, UMAP, and all C(n,3) feature combinations

Dimensionality Reduction Methods:
- t-SNE: Preserves local structure, good for visualizing clusters
- LLE: Local Linear Embedding, preserves local geometry
- UMAP: Uniform Manifold Approximation, balances local and global structure
- Feature Combinations: Direct visualization of all 3-feature subsets
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations
import logging

# Dimensionality reduction imports
from sklearn.manifold import TSNE, LocallyLinearEmbedding
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available. Install with: pip install umap-learn")

logger = logging.getLogger(__name__)

class DimensionalityReductionVisualizer:
    """
    Advanced dimensionality reduction visualizations
    """

    def __init__(self, db_conn, output_dir: Path, label_map: Dict):
        """
        Initialize visualizer

        Args:
            db_conn: Database connection
            output_dir: Output directory for plots
            label_map: Label mapping from main visualizer
        """
        self.db_conn = db_conn
        self.output_dir = Path(output_dir)
        self.label_map = label_map

        # Create subdirectories
        (self.output_dir / 'tsne_2d').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'tsne_3d').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'lle_2d').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'lle_3d').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'umap_2d').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'umap_3d').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'feature_combinations').mkdir(parents=True, exist_ok=True)

        logger.info("DimensionalityReductionVisualizer initialized")

    def plot_tsne(self, matrix_file: Path, n_components: int = 3, perplexity: float = 30.0) -> Path:
        """
        Generate t-SNE visualization (2D or 3D)

        Args:
            matrix_file: Path to verification feature matrix
            n_components: 2 or 3
            perplexity: t-SNE perplexity parameter (default 30)

        Returns:
            Path to saved plot
        """
        logger.info(f"Generating {n_components}D t-SNE for {matrix_file.name}")

        # Load data
        data = np.load(matrix_file)
        feature_names = [n for n in data.dtype.names if n != 'segment_id']
        segment_ids = data['segment_id']

        # Build feature matrix, handling NaN
        feature_matrix = np.column_stack([data[f] for f in feature_names])
        nan_mask = np.isnan(feature_matrix).any(axis=1)
        valid_mask = ~nan_mask

        if np.sum(nan_mask) > 0:
            logger.warning(f"Excluding {np.sum(nan_mask)} segments with NaN values")

        feature_matrix_clean = feature_matrix[valid_mask]
        segment_ids_clean = segment_ids[valid_mask]

        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, n_jobs=-1)
        coords = tsne.fit_transform(feature_matrix_clean)

        # Get colors
        labels = [self.label_map['compound_labels'].get(sid, 'Unknown') for sid in segment_ids_clean]
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
        point_colors = [label_to_color[l] for l in labels]

        # Plot
        if n_components == 2:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=5, alpha=0.6)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title(f't-SNE 2D: {matrix_file.stem}\\nPerplexity={perplexity}')
            ax.grid(True, alpha=0.3)

            output_path = self.output_dir / 'tsne_2d' / f'{matrix_file.stem}_tsne2d.png'
        else:
            fig = plt.figure(figsize=(15, 12))

            # 4 views: XY, YZ, XZ, 3D perspective
            views = [
                (1, 2, 'XY Projection'),
                (2, 3, 'YZ Projection'),
                (1, 3, 'XZ Projection'),
                (None, None, '3D Perspective')
            ]

            for idx, (dim1, dim2, title) in enumerate(views, 1):
                if dim1 is None:
                    ax = fig.add_subplot(2, 2, idx, projection='3d')
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                              c=point_colors, s=5, alpha=0.6)
                    ax.view_init(elev=30, azim=45)
                    ax.set_xlabel('t-SNE 1')
                    ax.set_ylabel('t-SNE 2')
                    ax.set_zlabel('t-SNE 3')
                else:
                    ax = fig.add_subplot(2, 2, idx)
                    ax.scatter(coords[:, dim1-1], coords[:, dim2-1], c=point_colors, s=5, alpha=0.6)
                    ax.set_xlabel(f't-SNE {dim1}')
                    ax.set_ylabel(f't-SNE {dim2}')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

            plt.suptitle(f't-SNE 3D: {matrix_file.stem}\\nPerplexity={perplexity}', fontsize=14)
            output_path = self.output_dir / 'tsne_3d' / f'{matrix_file.stem}_tsne3d.png'

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved t-SNE {n_components}D: {output_path.name}")
        return output_path

    def plot_lle(self, matrix_file: Path, n_components: int = 3, n_neighbors: int = 30) -> Path:
        """
        Generate LLE (Local Linear Embedding) visualization (2D or 3D)

        Args:
            matrix_file: Path to verification feature matrix
            n_components: 2 or 3
            n_neighbors: Number of neighbors for LLE (default 30)

        Returns:
            Path to saved plot
        """
        logger.info(f"Generating {n_components}D LLE for {matrix_file.name}")

        # Load data (same as t-SNE)
        data = np.load(matrix_file)
        feature_names = [n for n in data.dtype.names if n != 'segment_id']
        segment_ids = data['segment_id']

        feature_matrix = np.column_stack([data[f] for f in feature_names])
        nan_mask = np.isnan(feature_matrix).any(axis=1)
        valid_mask = ~nan_mask

        if np.sum(nan_mask) > 0:
            logger.warning(f"Excluding {np.sum(nan_mask)} segments with NaN values")

        feature_matrix_clean = feature_matrix[valid_mask]
        segment_ids_clean = segment_ids[valid_mask]

        # Apply LLE
        lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=42, n_jobs=-1)
        coords = lle.fit_transform(feature_matrix_clean)

        # Get colors
        labels = [self.label_map['compound_labels'].get(sid, 'Unknown') for sid in segment_ids_clean]
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
        point_colors = [label_to_color[l] for l in labels]

        # Plot (similar structure to t-SNE)
        if n_components == 2:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=5, alpha=0.6)
            ax.set_xlabel('LLE 1')
            ax.set_ylabel('LLE 2')
            ax.set_title(f'LLE 2D: {matrix_file.stem}\\nNeighbors={n_neighbors}')
            ax.grid(True, alpha=0.3)

            output_path = self.output_dir / 'lle_2d' / f'{matrix_file.stem}_lle2d.png'
        else:
            fig = plt.figure(figsize=(15, 12))

            views = [
                (1, 2, 'XY Projection'),
                (2, 3, 'YZ Projection'),
                (1, 3, 'XZ Projection'),
                (None, None, '3D Perspective')
            ]

            for idx, (dim1, dim2, title) in enumerate(views, 1):
                if dim1 is None:
                    ax = fig.add_subplot(2, 2, idx, projection='3d')
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                              c=point_colors, s=5, alpha=0.6)
                    ax.view_init(elev=30, azim=45)
                    ax.set_xlabel('LLE 1')
                    ax.set_ylabel('LLE 2')
                    ax.set_zlabel('LLE 3')
                else:
                    ax = fig.add_subplot(2, 2, idx)
                    ax.scatter(coords[:, dim1-1], coords[:, dim2-1], c=point_colors, s=5, alpha=0.6)
                    ax.set_xlabel(f'LLE {dim1}')
                    ax.set_ylabel(f'LLE {dim2}')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

            plt.suptitle(f'LLE 3D: {matrix_file.stem}\\nNeighbors={n_neighbors}', fontsize=14)
            output_path = self.output_dir / 'lle_3d' / f'{matrix_file.stem}_lle3d.png'

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved LLE {n_components}D: {output_path.name}")
        return output_path

    def plot_umap(self, matrix_file: Path, n_components: int = 3, n_neighbors: int = 30, min_dist: float = 0.1) -> Path:
        """
        Generate UMAP visualization (2D or 3D)

        Args:
            matrix_file: Path to verification feature matrix
            n_components: 2 or 3
            n_neighbors: Number of neighbors for UMAP (default 30)
            min_dist: Minimum distance for UMAP (default 0.1)

        Returns:
            Path to saved plot
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")

        logger.info(f"Generating {n_components}D UMAP for {matrix_file.name}")

        # Load data (same as others)
        data = np.load(matrix_file)
        feature_names = [n for n in data.dtype.names if n != 'segment_id']
        segment_ids = data['segment_id']

        feature_matrix = np.column_stack([data[f] for f in feature_names])
        nan_mask = np.isnan(feature_matrix).any(axis=1)
        valid_mask = ~nan_mask

        if np.sum(nan_mask) > 0:
            logger.warning(f"Excluding {np.sum(nan_mask)} segments with NaN values")

        feature_matrix_clean = feature_matrix[valid_mask]
        segment_ids_clean = segment_ids[valid_mask]

        # Apply UMAP
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        coords = reducer.fit_transform(feature_matrix_clean)

        # Get colors
        labels = [self.label_map['compound_labels'].get(sid, 'Unknown') for sid in segment_ids_clean]
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
        point_colors = [label_to_color[l] for l in labels]

        # Plot
        if n_components == 2:
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.scatter(coords[:, 0], coords[:, 1], c=point_colors, s=5, alpha=0.6)
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title(f'UMAP 2D: {matrix_file.stem}\\nNeighbors={n_neighbors}, MinDist={min_dist}')
            ax.grid(True, alpha=0.3)

            output_path = self.output_dir / 'umap_2d' / f'{matrix_file.stem}_umap2d.png'
        else:
            fig = plt.figure(figsize=(15, 12))

            views = [
                (1, 2, 'XY Projection'),
                (2, 3, 'YZ Projection'),
                (1, 3, 'XZ Projection'),
                (None, None, '3D Perspective')
            ]

            for idx, (dim1, dim2, title) in enumerate(views, 1):
                if dim1 is None:
                    ax = fig.add_subplot(2, 2, idx, projection='3d')
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                              c=point_colors, s=5, alpha=0.6)
                    ax.view_init(elev=30, azim=45)
                    ax.set_xlabel('UMAP 1')
                    ax.set_ylabel('UMAP 2')
                    ax.set_zlabel('UMAP 3')
                else:
                    ax = fig.add_subplot(2, 2, idx)
                    ax.scatter(coords[:, dim1-1], coords[:, dim2-1], c=point_colors, s=5, alpha=0.6)
                    ax.set_xlabel(f'UMAP {dim1}')
                    ax.set_ylabel(f'UMAP {dim2}')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)

            plt.suptitle(f'UMAP 3D: {matrix_file.stem}\\nNeighbors={n_neighbors}, MinDist={min_dist}', fontsize=14)
            output_path = self.output_dir / 'umap_3d' / f'{matrix_file.stem}_umap3d.png'

        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Saved UMAP {n_components}D: {output_path.name}")
        return output_path

    def plot_feature_combinations(self, matrix_file: Path, max_combinations: int = None) -> List[Path]:
        """
        Generate 3D scatter plots for all C(n,3) combinations of features

        For 9 features: C(9,3) = 84 combinations

        Args:
            matrix_file: Path to verification feature matrix
            max_combinations: Maximum number of combinations to plot (None = all)

        Returns:
            List of paths to saved plots
        """
        logger.info(f"Generating feature combination plots for {matrix_file.name}")

        # Load data
        data = np.load(matrix_file)
        feature_names = [n for n in data.dtype.names if n != 'segment_id']
        segment_ids = data['segment_id']
        n_features = len(feature_names)

        # Calculate number of combinations
        from math import comb
        n_combinations = comb(n_features, 3)
        logger.info(f"Total combinations: C({n_features},3) = {n_combinations}")

        if max_combinations and n_combinations > max_combinations:
            logger.warning(f"Limiting to {max_combinations} combinations (out of {n_combinations})")

        # Get all 3-feature combinations
        feature_combos = list(combinations(range(n_features), 3))
        if max_combinations:
            feature_combos = feature_combos[:max_combinations]

        # Get colors
        labels = [self.label_map['compound_labels'].get(sid, 'Unknown') for sid in segment_ids]
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        label_to_color = {l: colors[i] for i, l in enumerate(unique_labels)}
        point_colors = [label_to_color[l] for l in labels]

        output_paths = []

        # Generate plot for each combination
        for combo_idx, (f1_idx, f2_idx, f3_idx) in enumerate(feature_combos, 1):
            f1_name = feature_names[f1_idx]
            f2_name = feature_names[f2_idx]
            f3_name = feature_names[f3_idx]

            # Extract feature data
            f1_data = data[f1_name]
            f2_data = data[f2_name]
            f3_data = data[f3_name]

            # Handle NaN
            valid_mask = ~(np.isnan(f1_data) | np.isnan(f2_data) | np.isnan(f3_data))

            if np.sum(~valid_mask) > 0:
                logger.debug(f"Combo {combo_idx}: Excluding {np.sum(~valid_mask)} NaN segments")

            f1_clean = f1_data[valid_mask]
            f2_clean = f2_data[valid_mask]
            f3_clean = f3_data[valid_mask]
            colors_clean = [point_colors[i] for i, v in enumerate(valid_mask) if v]

            # Plot 4 views
            fig = plt.figure(figsize=(15, 12))

            # XY projection
            ax = fig.add_subplot(2, 2, 1)
            ax.scatter(f1_clean, f2_clean, c=colors_clean, s=5, alpha=0.6)
            ax.set_xlabel(f1_name)
            ax.set_ylabel(f2_name)
            ax.set_title('XY Projection')
            ax.grid(True, alpha=0.3)

            # YZ projection
            ax = fig.add_subplot(2, 2, 2)
            ax.scatter(f2_clean, f3_clean, c=colors_clean, s=5, alpha=0.6)
            ax.set_xlabel(f2_name)
            ax.set_ylabel(f3_name)
            ax.set_title('YZ Projection')
            ax.grid(True, alpha=0.3)

            # XZ projection
            ax = fig.add_subplot(2, 2, 3)
            ax.scatter(f1_clean, f3_clean, c=colors_clean, s=5, alpha=0.6)
            ax.set_xlabel(f1_name)
            ax.set_ylabel(f3_name)
            ax.set_title('XZ Projection')
            ax.grid(True, alpha=0.3)

            # 3D perspective
            ax = fig.add_subplot(2, 2, 4, projection='3d')
            ax.scatter(f1_clean, f2_clean, f3_clean, c=colors_clean, s=5, alpha=0.6)
            ax.view_init(elev=30, azim=45)
            ax.set_xlabel(f1_name)
            ax.set_ylabel(f2_name)
            ax.set_zlabel(f3_name)
            ax.set_title('3D Perspective')

            plt.suptitle(f'Feature Combination {combo_idx}/{len(feature_combos)}: {f1_name}, {f2_name}, {f3_name}\\n{matrix_file.stem}', fontsize=12)
            plt.tight_layout()

            # Save
            output_path = self.output_dir / 'feature_combinations' / f'{matrix_file.stem}_combo{combo_idx:03d}_{f1_name}_{f2_name}_{f3_name}.png'
            fig.savefig(output_path, dpi=100, bbox_inches='tight')  # Lower DPI for file size
            plt.close(fig)

            output_paths.append(output_path)

            if combo_idx % 10 == 0:
                logger.info(f"Generated {combo_idx}/{len(feature_combos)} feature combination plots")

        logger.info(f"Completed {len(output_paths)} feature combination plots")
        return output_paths
