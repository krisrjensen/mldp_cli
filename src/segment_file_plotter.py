#!/usr/bin/env python3
"""
Filename: segment_file_plotter.py
Author(s): Kristophor Jensen
Date Created: 20250902_100000
Date Revised: 20250902_100000
File version: 0.0.0.1
Description: Enhanced segment file plotter with statistical analysis for MLDP CLI
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'common'))

try:
    from publication_settings import (
        data_cleaning_tool_apply_publication_settings,
        segment_visualizer_apply_publication_settings,
        transient_viewer_apply_publication_settings
    )
except ImportError:
    logger.warning("Could not import publication settings, using defaults")
    def data_cleaning_tool_apply_publication_settings(): pass
    def segment_visualizer_apply_publication_settings(): pass
    def transient_viewer_apply_publication_settings(): pass


class EnhancedSegmentPlotter:
    """Enhanced segment file plotter with statistical analysis capabilities"""
    
    PLOT_STYLES = {
        'cleaning': data_cleaning_tool_apply_publication_settings,
        'segment': segment_visualizer_apply_publication_settings,
        'transient': transient_viewer_apply_publication_settings
    }
    
    def __init__(self, experiment_id: int = 18):
        """Initialize the plotter with experiment ID"""
        self.experiment_id = experiment_id
        self.base_path = Path(f'/Volumes/ArcData/V3_database/experiment{experiment_id:03d}/segment_files')
        
        if not self.base_path.exists():
            logger.warning(f"Base path {self.base_path} does not exist")
    
    def sample_data(self, data: np.ndarray, num_points: int = 1000, 
                   peak_detect: bool = False) -> Dict[str, np.ndarray]:
        """
        Sample data to specified number of points with optional peak detection
        
        Args:
            data: Input data array
            num_points: Target number of points
            peak_detect: If True, preserve peaks in each interval
            
        Returns:
            Dictionary containing sampled data and optional peak information
        """
        if len(data) <= num_points:
            return {
                'sampled': data,
                'indices': np.arange(len(data))
            }
        
        # Calculate interval size
        interval_size = len(data) / num_points
        result = {
            'sampled': [],
            'indices': [],
            'minimums': [],
            'maximums': [],
            'average': []
        }
        
        for i in range(num_points):
            start_idx = int(i * interval_size)
            end_idx = int((i + 1) * interval_size)
            interval = data[start_idx:end_idx]
            
            if len(interval) == 0:
                continue
            
            if peak_detect:
                # Find both min and max in interval
                min_val = np.min(interval)
                max_val = np.max(interval)
                avg_val = np.mean(interval)
                
                # Choose representative value based on which is furthest from mean
                if abs(max_val - avg_val) > abs(min_val - avg_val):
                    result['sampled'].append(max_val)
                else:
                    result['sampled'].append(min_val)
                
                result['minimums'].append(min_val)
                result['maximums'].append(max_val)
                result['average'].append(avg_val)
            else:
                # Simple decimation - take middle point
                mid_idx = len(interval) // 2
                result['sampled'].append(interval[mid_idx])
            
            # Store the index for time axis
            result['indices'].append(start_idx + len(interval) // 2)
        
        # Convert to numpy arrays
        for key in result:
            if result[key]:
                result[key] = np.array(result[key])
        
        return result
    
    def calculate_statistics(self, data: np.ndarray, num_points: int = 1000) -> Dict[str, np.ndarray]:
        """
        Calculate statistical measures for data
        
        Returns dict with minimums, maximums, average, variance, stddev
        """
        if len(data) <= num_points:
            # Return single values for each statistic
            return {
                'minimums': np.array([np.min(data)]),
                'maximums': np.array([np.max(data)]),
                'average': np.array([np.mean(data)]),
                'variance': np.array([np.var(data)]),
                'stddev': np.array([np.std(data)])
            }
        
        interval_size = len(data) / num_points
        stats = {
            'minimums': [],
            'maximums': [],
            'average': [],
            'variance': [],
            'stddev': []
        }
        
        for i in range(num_points):
            start_idx = int(i * interval_size)
            end_idx = int((i + 1) * interval_size)
            interval = data[start_idx:end_idx]
            
            if len(interval) == 0:
                continue
            
            stats['minimums'].append(np.min(interval))
            stats['maximums'].append(np.max(interval))
            stats['average'].append(np.mean(interval))
            stats['variance'].append(np.var(interval))
            stats['stddev'].append(np.std(interval))
        
        # Convert to numpy arrays
        for key in stats:
            stats[key] = np.array(stats[key])
        
        return stats
    
    def plot_with_statistics(self, ax: plt.Axes, time: np.ndarray, data: np.ndarray, 
                            plot_options: Dict[str, Any], label_prefix: str = ""):
        """
        Plot data with multiple statistical overlays
        
        Args:
            ax: Matplotlib axis
            time: Time array
            data: Raw data (voltage or current)
            plot_options: Dictionary of plotting options
            label_prefix: Prefix for plot labels (e.g., "Voltage", "Current")
        """
        # Sample data if needed
        num_points = plot_options.get('num_points', 1000)
        peak_detect = plot_options.get('peak_detect', False)
        
        if len(data) > num_points:
            sampled = self.sample_data(data, num_points, peak_detect)
            sampled_time = time[sampled['indices']]
            sampled_data = sampled['sampled']
        else:
            sampled_time = time
            sampled_data = data
        
        # Calculate statistics
        stats = self.calculate_statistics(data, num_points)
        stats_time = np.linspace(time[0], time[-1], len(stats['average']))
        
        # Plot actual data if requested
        if plot_options.get('plot_actual', True):
            ax.plot(sampled_time, sampled_data, 'b-', alpha=0.5, 
                   linewidth=1, label='Actual')
        
        # Plot statistical measures
        if plot_options.get('plot_average'):
            if plot_options.get('average_line', True):
                ax.plot(stats_time, stats['average'], 'g-', 
                       linewidth=2, label='Average')
            else:
                ax.plot(stats_time, stats['average'], 'go', 
                       markersize=3, label='Average')
        
        if plot_options.get('plot_minimums'):
            if plot_options.get('minimums_line', False):
                ax.plot(stats_time, stats['minimums'], 'r-',
                       linewidth=1.5, label='Minimums')
            else:
                ax.plot(stats_time, stats['minimums'], 'rv',
                       markersize=4, label='Minimums')
        
        if plot_options.get('plot_maximums'):
            if plot_options.get('maximums_line', False):
                ax.plot(stats_time, stats['maximums'], 'r-',
                       linewidth=1.5, label='Maximums', alpha=0.7)
            else:
                ax.plot(stats_time, stats['maximums'], 'r^',
                       markersize=4, label='Maximums')
        
        # Handle variance and stddev (potentially on second y-axis)
        if plot_options.get('plot_variance') or plot_options.get('plot_stddev'):
            ax2 = ax.twinx()
            
            if plot_options.get('plot_variance'):
                if plot_options.get('variance_line', True):
                    ax2.plot(stats_time, stats['variance'], 'm-',
                            linewidth=1.5, label='Variance', alpha=0.7)
                else:
                    ax2.plot(stats_time, stats['variance'], 'mo',
                            markersize=3, label='Variance')
            
            if plot_options.get('plot_stddev'):
                if plot_options.get('stddev_line', True):
                    ax2.plot(stats_time, stats['stddev'], 'c-',
                            linewidth=1.5, label='Std Dev', alpha=0.7)
                else:
                    ax2.plot(stats_time, stats['stddev'], 'c*',
                            markersize=4, label='Std Dev')
            
            ax2.set_ylabel('Variance/StdDev', color='m', fontsize=11)
            ax2.tick_params(axis='y', labelcolor='m', labelsize=10)
            
            # Combine legends with better positioning
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
                     fontsize=9, framealpha=0.9, ncol=2)
        else:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Signal', fontsize=11)
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    def create_data_cleaning_style_plot(self, time: np.ndarray, voltage: np.ndarray, 
                                       current: np.ndarray, plot_options: Dict[str, Any],
                                       title: str = None) -> plt.Figure:
        """
        Create plot in data_cleaning_tool style with separate subplots for voltage and current
        """
        data_cleaning_tool_apply_publication_settings()
        
        # Check if we have actual current data
        has_current = not np.allclose(current, 0)
        
        if has_current:
            # Create two subplots for voltage and current
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                          constrained_layout=True)
            
            # Plot voltage
            ax1.set_ylabel('Voltage (V)', fontsize=11)
            self.plot_with_statistics(ax1, time, voltage, plot_options)
            ax1.grid(True, alpha=0.3)
            
            # Plot current
            ax2.set_ylabel('Current (A)', fontsize=11)
            # Use statistical plotting for current if enabled
            if any([plot_options.get(k) for k in ['plot_minimums', 'plot_maximums', 
                                                   'plot_average', 'plot_variance', 'plot_stddev']]):
                self.plot_with_statistics(ax2, time, current, plot_options)
            else:
                # Simple plot for current
                num_points = plot_options.get('num_points', 1000)
                if len(current) > num_points:
                    sampled = self.sample_data(current, num_points, plot_options.get('peak_detect', False))
                    sampled_time = time[sampled['indices']]
                    sampled_current = sampled['sampled']
                else:
                    sampled_time = time
                    sampled_current = current
                ax2.plot(sampled_time, sampled_current, 'r-', linewidth=1)
            
            ax2.set_xlabel('Time (s)', fontsize=11)
            ax2.grid(True, alpha=0.3)
        else:
            # Single plot for voltage only
            fig, ax1 = plt.subplots(figsize=(14, 8), constrained_layout=True)
            
            # Plot voltage
            ax1.set_ylabel('Voltage (V)', fontsize=11)
            ax1.set_xlabel('Time (s)', fontsize=11)
            self.plot_with_statistics(ax1, time, voltage, plot_options)
            ax1.grid(True, alpha=0.3)
        
        if title:
            fig.suptitle(title, fontsize=14, y=0.99)
        
        return fig
    
    def find_segment_files(self, original_segment: Optional[int] = None,
                          result_segment_size: Optional[int] = None,
                          segment_labels: Optional[List[int]] = None,
                          file_labels: Optional[List[int]] = None,
                          decimations: Optional[List[int]] = None,
                          types: Optional[List[str]] = None) -> List[Path]:
        """
        Find segment files based on filtering criteria
        
        Returns list of file paths matching the criteria
        """
        matching_files = []
        
        # Set defaults
        if decimations is None:
            decimations = [0]  # Default to no decimation
        if types is None:
            types = ['RAW']    # Default to RAW type
        
        # Search through directory structure
        for size_dir in self.base_path.glob('S*'):
            if not size_dir.is_dir():
                continue
            
            # Check result segment size filter
            dir_size = int(size_dir.name[1:])
            if result_segment_size and dir_size != result_segment_size:
                continue
            
            for type_dir in size_dir.glob('T*'):
                if not type_dir.is_dir():
                    continue
                
                # Check type filter
                type_name = type_dir.name[1:]
                if type_name not in types:
                    continue
                
                for dec_dir in type_dir.glob('D*'):
                    if not dec_dir.is_dir():
                        continue
                    
                    # Check decimation filter
                    dec_value = int(dec_dir.name[1:])
                    if dec_value not in decimations:
                        continue
                    
                    # Search for matching files
                    for file_path in dec_dir.glob('*.npy'):
                        # Parse filename: {segment_id}_{file_id}_D{dec}_T{type}_S{size}.npy
                        parts = file_path.stem.split('_')
                        if len(parts) < 5:
                            continue
                        
                        try:
                            seg_id = int(parts[0])
                            file_id = int(parts[1])
                        except ValueError:
                            continue
                        
                        # Apply filters
                        if original_segment and seg_id != original_segment:
                            continue
                        if segment_labels and seg_id not in segment_labels:
                            continue
                        if file_labels and file_id not in file_labels:
                            continue
                        
                        matching_files.append(file_path)
        
        return sorted(matching_files)
    
    def organize_subplots(self, file_paths: List[Path], grouping: str = 'file',
                         max_subplot: Tuple[int, int] = (3, 3)) -> List[Dict]:
        """
        Organize files into subplot groups
        
        Args:
            file_paths: List of file paths to plot
            grouping: How to group ('file', 'statistics', 'decimations', 'type')
            max_subplot: Maximum grid size (rows, cols)
            
        Returns:
            List of page configurations
        """
        grouped = {}
        
        for file_path in file_paths:
            # Parse filename to get metadata
            parts = file_path.stem.split('_')
            seg_id = int(parts[0])
            file_id = int(parts[1])
            dec_value = int(parts[2][1:])  # Remove 'D' prefix
            type_name = parts[3][1:]       # Remove 'T' prefix
            
            # Determine grouping key
            if grouping == 'file':
                key = f"File_{file_id}"
            elif grouping == 'decimations':
                key = f"Dec_{dec_value}"
            elif grouping == 'type':
                key = type_name
            else:
                key = 'all'
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append({
                'path': file_path,
                'segment_id': seg_id,
                'file_id': file_id,
                'decimation': dec_value,
                'type': type_name
            })
        
        # Split into pages if exceeds max_subplot
        max_plots_per_page = max_subplot[0] * max_subplot[1]
        pages = []
        
        for key, configs in grouped.items():
            for i in range(0, len(configs), max_plots_per_page):
                page_configs = configs[i:i+max_plots_per_page]
                n_plots = len(page_configs)
                
                # Calculate optimal grid
                if n_plots <= max_subplot[1]:
                    grid = (1, n_plots)
                else:
                    cols = min(n_plots, max_subplot[1])
                    rows = min((n_plots + cols - 1) // cols, max_subplot[0])
                    grid = (rows, cols)
                
                pages.append({
                    'key': key,
                    'configs': page_configs,
                    'grid': grid
                })
        
        return pages
    
    def load_segment_data(self, file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load segment data from file
        
        Returns:
            time, voltage, current arrays
        """
        data = np.load(file_path)
        
        # Parse filename to get decimation factor
        # Format: {segment_id}_{file_id}_D{dec}_T{type}_S{size}.npy
        filename = file_path.stem
        parts = filename.split('_')
        decimation = int(parts[2][1:])  # Remove 'D' prefix
        
        # Assume data format: [voltage, current] or single channel
        if data.ndim == 2 and data.shape[1] == 2:
            voltage = data[:, 0]
            current = data[:, 1]
        else:
            voltage = data.flatten()
            current = np.zeros_like(voltage)  # No current data
        
        # Calculate effective sample rate after decimation
        # Original sample rate is 40MHz, decimation reduces this
        # Decimation factor is 2^n - 1, so we keep every (decimation+1)th sample
        if decimation == 0:
            effective_sample_rate = 40e6
        else:
            effective_sample_rate = 40e6 / (decimation + 1)
        
        # Generate time array with correct sample rate
        # This preserves the total time span of the original segment
        time = np.arange(len(voltage)) / effective_sample_rate
        
        return time, voltage, current
    
    def plot_segments(self, output_folder: Path, plot_options: Dict[str, Any]):
        """
        Main plotting function that processes all options and creates plots
        
        Args:
            output_folder: Directory to save plots
            plot_options: Dictionary containing all plotting options
        """
        # Create output folder if needed
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find matching files
        files = self.find_segment_files(
            original_segment=plot_options.get('original_segment'),
            result_segment_size=plot_options.get('result_segment_size'),
            segment_labels=plot_options.get('segment_labels'),
            file_labels=plot_options.get('file_labels'),
            decimations=plot_options.get('decimations', [0]),
            types=plot_options.get('types', ['RAW'])
        )
        
        if not files:
            logger.warning("No matching files found")
            return
        
        logger.info(f"Found {len(files)} matching files")
        
        # Apply plot style
        style = plot_options.get('plot_style', 'cleaning')
        if style in self.PLOT_STYLES:
            self.PLOT_STYLES[style]()
        
        # Handle subplot organization
        if plot_options.get('no_subplots', False):
            # Create separate file for each plot
            for file_path in files:
                time, voltage, current = self.load_segment_data(file_path)
                
                # Create plot based on style
                if style == 'cleaning':
                    fig = self.create_data_cleaning_style_plot(
                        time, voltage, current, plot_options,
                        title=plot_options.get('title', file_path.stem)
                    )
                else:
                    # Generic plot for other styles
                    fig, ax = plt.subplots(figsize=(12, 8))
                    self.plot_with_statistics(ax, time, voltage, plot_options)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Signal')
                    ax.set_title(plot_options.get('title', file_path.stem))
                
                # Save plot
                output_file = output_folder / f"{file_path.stem}.{plot_options.get('format', 'png')}"
                fig.savefig(output_file, dpi=plot_options.get('dpi', 300))
                plt.close(fig)
                logger.info(f"Saved: {output_file}")
        else:
            # Create subplots
            grouping = plot_options.get('subplots', 'file')
            max_subplot = plot_options.get('max_subplot', (3, 3))
            
            pages = self.organize_subplots(files, grouping, max_subplot)
            
            for page_idx, page in enumerate(pages):
                # Calculate better figure size based on grid
                fig_width = min(8 * page['grid'][1], 24)  # Max 24 inches wide
                fig_height = min(6 * page['grid'][0], 18)  # Max 18 inches tall
                
                fig, axes = plt.subplots(page['grid'][0], page['grid'][1],
                                       figsize=(fig_width, fig_height), 
                                       constrained_layout=True)
                
                # Flatten axes for easier iteration
                if page['grid'][0] == 1 and page['grid'][1] == 1:
                    axes = [axes]
                elif page['grid'][0] == 1 or page['grid'][1] == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()
                
                for idx, config in enumerate(page['configs']):
                    ax = axes[idx]
                    time, voltage, current = self.load_segment_data(config['path'])
                    
                    # Plot based on grouping type
                    self.plot_with_statistics(ax, time, voltage, plot_options)
                    # Title will be set in plot_with_statistics for consistency
                    ax.set_title(f"Seg {config['segment_id']} File {config['file_id']} Dec {config['decimation']}", 
                                fontsize=11, pad=10)
                
                # Hide unused subplots
                for idx in range(len(page['configs']), len(axes)):
                    axes[idx].set_visible(False)
                
                # Set overall title with better spacing
                if plot_options.get('title'):
                    fig.suptitle(plot_options.get('title'), fontsize=14, y=0.98)
                else:
                    fig.suptitle(f"{page['key']} - Page {page_idx + 1}", fontsize=14, y=0.98)
                
                # Save plot
                output_file = output_folder / f"{page['key']}_page{page_idx+1}.{plot_options.get('format', 'png')}"
                fig.savefig(output_file, dpi=plot_options.get('dpi', 300))
                plt.close(fig)
                logger.info(f"Saved: {output_file}")


def plot_segment_files(experiment_id: int = 18, **kwargs):
    """
    Main entry point for segment file plotting
    
    Args:
        experiment_id: Experiment ID
        **kwargs: All plotting options
    """
    plotter = EnhancedSegmentPlotter(experiment_id)
    
    # Prepare plot options
    plot_options = {
        'original_segment': kwargs.get('original_segment'),
        'result_segment_size': kwargs.get('result_segment_size'),
        'segment_labels': kwargs.get('segment_labels'),
        'file_labels': kwargs.get('file_labels'),
        'decimations': kwargs.get('decimations', [0]),
        'types': kwargs.get('types', ['RAW']),
        'num_points': kwargs.get('num_points', 1000),
        'peak_detect': kwargs.get('peak_detect', False),
        'plot_actual': kwargs.get('plot_actual', True),
        'plot_minimums': kwargs.get('plot_minimums', False),
        'plot_maximums': kwargs.get('plot_maximums', False),
        'plot_average': kwargs.get('plot_average', False),
        'plot_variance': kwargs.get('plot_variance', False),
        'plot_stddev': kwargs.get('plot_stddev', False),
        'minimums_line': kwargs.get('minimums_line', False),
        'minimums_point': kwargs.get('minimums_point', False),
        'maximums_line': kwargs.get('maximums_line', False),
        'maximums_point': kwargs.get('maximums_point', False),
        'average_line': kwargs.get('average_line', True),
        'average_point': kwargs.get('average_point', False),
        'variance_line': kwargs.get('variance_line', True),
        'variance_point': kwargs.get('variance_point', False),
        'stddev_line': kwargs.get('stddev_line', True),
        'stddev_point': kwargs.get('stddev_point', False),
        'no_subplots': kwargs.get('no_subplots', False),
        'subplots': kwargs.get('subplots', 'file'),
        'max_subplot': kwargs.get('max_subplot', (3, 3)),
        'dpi': kwargs.get('dpi', 300),
        'format': kwargs.get('format', 'png'),
        'title': kwargs.get('title'),
        'plot_style': kwargs.get('plot_style', 'cleaning')
    }
    
    # Handle line/point modifiers
    if plot_options['minimums_point']:
        plot_options['minimums_line'] = False
    if plot_options['maximums_point']:
        plot_options['maximums_line'] = False
    if plot_options['average_point']:
        plot_options['average_line'] = False
    if plot_options['variance_point']:
        plot_options['variance_line'] = False
    if plot_options['stddev_point']:
        plot_options['stddev_line'] = False
    
    # Get output folder
    output_folder = kwargs.get('output_folder')
    if not output_folder:
        raise ValueError("--output-folder is required")
    
    # Run plotting
    plotter.plot_segments(Path(output_folder), plot_options)


if __name__ == "__main__":
    # Test the plotter
    plot_segment_files(
        experiment_id=18,
        original_segment=104075,
        decimations=[0],
        types=['RAW'],
        num_points=500,
        peak_detect=True,
        plot_minimums=True,
        minimums_point=True,
        plot_maximums=True,
        maximums_point=True,
        plot_average=True,
        average_line=True,
        output_folder='/tmp/test_plots',
        dpi=150,
        title="Test Statistical Plot"
    )