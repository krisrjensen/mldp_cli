#!/usr/bin/env python3
"""
Quick plot utility for segment and feature files

Usage:
  python plot_files.py segment /path/to/file.npy
  python plot_files.py feature /path/to/file.npy
  python plot_files.py segment /path/to/file.npy --columns 2,3
  python plot_files.py segment /path/to/file.npy --save plot.png

Author: Kristophor Jensen
Date: 20251005_141500
Version: 1.0.0.0
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_segment(file_path, columns='0,1', save=None):
    """Plot segment file data"""
    data = np.load(file_path)
    filename = Path(file_path).name

    print(f"📊 Segment File: {filename}")
    print(f"   Shape: {data.shape}")
    print(f"   Samples: {data.shape[0]:,}")
    print(f"   Channels: {data.shape[1]}")

    # Determine columns and labels
    if columns == 'all':
        cols = list(range(data.shape[1]))
        labels = []
        if data.shape[1] >= 2:
            labels.extend(['Raw Voltage', 'Raw Current'])
        if data.shape[1] >= 4:
            labels.extend(['Minmax Voltage', 'Minmax Current'])
        if data.shape[1] >= 6:
            labels.extend(['Zscore Voltage', 'Zscore Current'])
        # Pad with generic labels if needed
        while len(labels) < len(cols):
            labels.append(f'Column {len(labels)}')
    else:
        cols = [int(c.strip()) for c in columns.split(',')]
        labels = []
        for c in cols:
            if c == 0:
                labels.append('Raw Voltage')
            elif c == 1:
                labels.append('Raw Current')
            elif c == 2:
                labels.append('Minmax Voltage')
            elif c == 3:
                labels.append('Minmax Current')
            elif c == 4:
                labels.append('Zscore Voltage')
            elif c == 5:
                labels.append('Zscore Current')
            else:
                labels.append(f'Column {c}')

    print(f"   Plotting columns: {cols}")
    print()

    # Create plot
    fig, axes = plt.subplots(len(cols), 1, figsize=(14, 3*len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for i, (col, label) in enumerate(zip(cols, labels)):
        axes[i].plot(data[:, col], linewidth=0.5, color='#2E86AB')
        axes[i].set_ylabel(label, fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3, linestyle='--')
        axes[i].set_xlim(0, len(data))

        # Add statistics
        mean = np.mean(data[:, col])
        std = np.std(data[:, col])
        axes[i].text(0.02, 0.95, f'μ={mean:.2f}, σ={std:.2f}',
                    transform=axes[i].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9)

    axes[-1].set_xlabel('Sample', fontsize=12, fontweight='bold')
    plt.suptitle(f'Segment File: {filename}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"💾 Saved plot to: {save}")
    else:
        plt.show()

def plot_feature(file_path, save=None):
    """Plot feature file data"""
    data = np.load(file_path)
    filename = Path(file_path).name

    print(f"📊 Feature File: {filename}")
    print(f"   Shape: {data.shape}")
    print(f"   Windows: {data.shape[0]:,}")
    print(f"   Features: {data.shape[1]}")
    print()

    # Create plot
    fig, axes = plt.subplots(data.shape[1], 1, figsize=(14, 2.5*data.shape[1]), sharex=True)
    if data.shape[1] == 1:
        axes = [axes]

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']

    for i in range(data.shape[1]):
        color = colors[i % len(colors)]
        axes[i].plot(data[:, i], linewidth=1, color=color)
        axes[i].set_ylabel(f'Feature {i}', fontsize=10, fontweight='bold')
        axes[i].grid(True, alpha=0.3, linestyle='--')

        # Add statistics
        mean = np.mean(data[:, i])
        std = np.std(data[:, i])
        min_val = np.min(data[:, i])
        max_val = np.max(data[:, i])
        axes[i].text(0.02, 0.95,
                    f'μ={mean:.2f}, σ={std:.2f}, min={min_val:.2f}, max={max_val:.2f}',
                    transform=axes[i].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=8)

    axes[-1].set_xlabel('Window', fontsize=12, fontweight='bold')
    plt.suptitle(f'Feature File: {filename}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"💾 Saved plot to: {save}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Plot segment or feature files from MLDP experiment 041',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot raw voltage/current
  python plot_files.py segment /path/to/segment.npy

  # Plot minmax normalized voltage/current
  python plot_files.py segment /path/to/segment.npy --columns 2,3

  # Plot all columns
  python plot_files.py segment /path/to/segment.npy --columns all

  # Save plot to file
  python plot_files.py segment /path/to/segment.npy --save output.png

  # Plot feature file
  python plot_files.py feature /path/to/feature.npy
        """
    )

    parser.add_argument('type', choices=['segment', 'feature'],
                       help='File type to plot')
    parser.add_argument('file', help='Path to .npy file')
    parser.add_argument('--columns', default='0,1',
                       help='Columns to plot for segment files (e.g., "0,1" or "2,3" or "all")')
    parser.add_argument('--save', help='Save plot to file instead of displaying')

    args = parser.parse_args()

    # Validate file exists
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ Error: File not found: {args.file}")
        sys.exit(1)

    # Plot
    try:
        if args.type == 'segment':
            plot_segment(args.file, args.columns, args.save)
        else:
            plot_feature(args.file, args.save)
    except Exception as e:
        print(f"❌ Error plotting file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
