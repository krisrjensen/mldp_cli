#!/usr/bin/env python3
"""
Extended Experiment Generation Configuration

Filename: experiment_generation_config_extended.py
Author(s): Kristophor Jensen  
Date Created: 20250907_181500
Date Revised: 20250907_181500
File version: 0.0.0.1
Description: Extended configuration for complex experiment generation with file selection
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import logging
import random
import psycopg2

logger = logging.getLogger(__name__)


@dataclass
class FileSelectionConfig:
    """Configuration for file selection strategy"""
    strategy: str = 'random'  # random, all, specific
    max_files: int = 50
    random_seed: Optional[int] = 42
    min_examples_per_class: int = 25
    exclude_labels: List[str] = field(default_factory=lambda: ['trash', 'current_only', 'voltage_only', 'other'])


@dataclass
class ExtendedExperimentConfig:
    """Extended configuration for complex experiment generation"""
    
    # Basic settings
    experiment_name: str
    experiment_type: str = 'classification'
    experiment_version: str = '1.0.0'
    experiment_description: str = ''
    
    # File selection
    file_selection: FileSelectionConfig = field(default_factory=FileSelectionConfig)
    
    # Data selection
    target_labels: List[int] = field(default_factory=list)
    instances_per_label: Optional[int] = None  # None means use all available
    
    # Segment configuration  
    segment_sizes: List[int] = field(default_factory=lambda: [128, 1024, 8192])
    decimation_factors: List[int] = field(default_factory=lambda: [0, 7, 15])
    data_types: List[str] = field(default_factory=lambda: ['raw', 'adc6', 'adc8', 'adc10', 'adc12', 'adc14'])
    amplitude_methods: List[str] = field(default_factory=lambda: [
        'minmax', 'zscore', 'robust', 'unit_vector', 'power',
        'chunk_4096_standardize', 'chunk_8192_standardize', 'chunk_16384_standardize'
    ])
    distance_functions: List[str] = field(default_factory=lambda: [
        'manhattan', 'euclidean', 'cosine', 'pearson', 'additive_symmetric',
        'braycurtis', 'canberra', 'chebyshev', 'jensenshannon', 'kullback_leibler',
        'kumar_hassebrook', 'sqeuclidean', 'taneja', 'wasserstein', 'wavehedges', 'fidelity'
    ])
    
    # Segment selection strategy
    selection_strategy: str = 'position_balanced_per_file'
    min_segments_per_position: int = 1
    min_segments_per_file: int = 3
    random_seed: Optional[int] = 42
    balanced_segments: bool = True
    position_balance_mode: str = 'at_least_one'
    
    # Quality filters
    min_quality_score: float = 1.0
    validation_status_filter: str = 'all'
    exclude_edge_segments: bool = False
    require_full_transient: bool = False
    
    # Pipeline configuration
    pipeline_id: Optional[int] = None
    skip_existing: bool = True
    parallel_processing: bool = True
    max_workers: int = 16
    dry_run: bool = False
    
    def select_files(self, conn) -> List[int]:
        """Select files based on configuration"""
        cursor = conn.cursor()
        
        # Get valid labels
        if not self.target_labels:
            # Auto-detect valid labels
            cursor.execute('''
                SELECT el.label_id
                FROM experiment_labels el
                LEFT JOIN files_Y fy ON el.label_id = fy.label_id
                WHERE el.experiment_label NOT IN %s
                AND el.active = true
                GROUP BY el.label_id
                HAVING COUNT(DISTINCT fy.file_id) >= %s
                ORDER BY el.label_id
            ''', (tuple(self.file_selection.exclude_labels), self.file_selection.min_examples_per_class))
            
            self.target_labels = [row[0] for row in cursor.fetchall()]
            logger.info(f"Auto-detected {len(self.target_labels)} valid labels")
        
        # Get files for valid labels
        cursor.execute('''
            SELECT DISTINCT f.file_id, f.original_path, fy.label_id
            FROM files f
            JOIN files_Y fy ON f.file_id = fy.file_id
            JOIN experiment_status es ON f.file_id = es.file_id
            WHERE fy.label_id = ANY(%s)
            AND es.status = true
            ORDER BY f.file_id
        ''', (self.target_labels,))
        
        all_files = cursor.fetchall()
        logger.info(f"Found {len(all_files)} total files for selected labels")
        
        # Apply selection strategy
        if self.file_selection.strategy == 'random':
            rng = random.Random(self.file_selection.random_seed)
            selected = rng.sample(all_files, min(self.file_selection.max_files, len(all_files)))
            file_ids = [f[0] for f in selected]
        elif self.file_selection.strategy == 'all':
            file_ids = [f[0] for f in all_files]
        else:
            file_ids = [f[0] for f in all_files[:self.file_selection.max_files]]
        
        cursor.close()
        logger.info(f"Selected {len(file_ids)} files using {self.file_selection.strategy} strategy")
        return file_ids
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Check required fields
        if not self.experiment_name:
            errors.append("experiment_name is required")
        
        if not self.segment_sizes:
            errors.append("segment_sizes cannot be empty")
        
        # Check segment sizes are valid
        valid_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 
                      32768, 65536, 131072, 262144, 524288]
        for size in self.segment_sizes:
            if size not in valid_sizes:
                errors.append(f"Invalid segment size: {size}")
        
        # Check decimation factors
        valid_decimations = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023]
        for dec in self.decimation_factors:
            if dec not in valid_decimations:
                errors.append(f"Invalid decimation factor: {dec}")
        
        # Check selection strategy
        if self.min_segments_per_position < 1:
            errors.append("min_segments_per_position must be at least 1")
        
        if self.min_segments_per_file < self.min_segments_per_position:
            errors.append("min_segments_per_file must be >= min_segments_per_position")
        
        # Check quality score
        if not 0.0 <= self.min_quality_score <= 1.0:
            errors.append("min_quality_score must be between 0.0 and 1.0")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        return True
    
    def calculate_combinations(self) -> Dict[str, Any]:
        """Calculate the number of parameter combinations"""
        n_files = self.file_selection.max_files
        n_labels = len(self.target_labels) if self.target_labels else 8
        n_sizes = len(self.segment_sizes)
        n_decimations = len(self.decimation_factors)
        n_data_types = len(self.data_types)
        n_amplitude = len(self.amplitude_methods)
        n_distance = len(self.distance_functions)
        
        # Total parameter combinations (not counting segments)
        total_params = n_sizes * n_decimations * n_data_types * n_amplitude
        
        # Minimum segments
        min_segments = n_files * self.min_segments_per_file
        
        return {
            'files': n_files,
            'labels': n_labels,
            'segment_sizes': n_sizes,
            'decimations': n_decimations,
            'data_types': n_data_types,
            'amplitude_methods': n_amplitude,
            'distance_functions': n_distance,
            'parameter_combinations': total_params,
            'min_total_segments': min_segments,
            'estimated_distance_calculations': min_segments * n_distance
        }
    
    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        config_dict = {
            'experiment_name': self.experiment_name,
            'experiment_type': self.experiment_type,
            'experiment_version': self.experiment_version,
            'experiment_description': self.experiment_description,
            'file_selection': {
                'strategy': self.file_selection.strategy,
                'max_files': self.file_selection.max_files,
                'random_seed': self.file_selection.random_seed,
                'min_examples_per_class': self.file_selection.min_examples_per_class,
                'exclude_labels': self.file_selection.exclude_labels
            },
            'target_labels': self.target_labels,
            'segment_sizes': self.segment_sizes,
            'decimation_factors': self.decimation_factors,
            'data_types': self.data_types,
            'amplitude_methods': self.amplitude_methods,
            'distance_functions': self.distance_functions,
            'selection_strategy': self.selection_strategy,
            'min_segments_per_position': self.min_segments_per_position,
            'min_segments_per_file': self.min_segments_per_file,
            'random_seed': self.random_seed,
            'balanced_segments': self.balanced_segments,
            'position_balance_mode': self.position_balance_mode,
            'min_quality_score': self.min_quality_score,
            'validation_status_filter': self.validation_status_filter,
            'calculations': self.calculate_combinations()
        }
        return json.dumps(config_dict, indent=2)
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        calcs = self.calculate_combinations()
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Description: {self.experiment_description}",
            f"",
            f"File Selection:",
            f"  Strategy: {self.file_selection.strategy}",
            f"  Max files: {self.file_selection.max_files}",
            f"  Labels: {len(self.target_labels)} classes",
            f"",
            f"Parameters:",
            f"  Segment sizes: {self.segment_sizes}",
            f"  Decimations: {self.decimation_factors}",
            f"  Data types: {len(self.data_types)} types",
            f"  Amplitude methods: {len(self.amplitude_methods)} methods",
            f"  Distance functions: {len(self.distance_functions)} functions",
            f"",
            f"Segment Selection:",
            f"  Min per position: {self.min_segments_per_position}",
            f"  Min per file: {self.min_segments_per_file}",
            f"  Balance mode: {self.position_balance_mode}",
            f"",
            f"Estimated Scale:",
            f"  Parameter combinations: {calcs['parameter_combinations']:,}",
            f"  Minimum segments: {calcs['min_total_segments']:,}",
            f"  Distance calculations: {calcs['estimated_distance_calculations']:,}"
        ]
        return "\n".join(lines)


# Predefined configuration for random 50 files
RANDOM_50FILES_CONFIG = ExtendedExperimentConfig(
    experiment_name="random_50files_all_positions",
    experiment_description="Random selection of up to 50 files with minimum 1 segment per position type per file",
    file_selection=FileSelectionConfig(
        strategy='random',
        max_files=50,
        random_seed=42,
        min_examples_per_class=25,
        exclude_labels=['trash', 'current_only', 'voltage_only', 'other']
    ),
    target_labels=[1, 2, 3, 4, 5, 6, 7, 16],  # 8 valid labels
    segment_sizes=[128, 1024, 8192],
    decimation_factors=[0, 7, 15],
    data_types=['raw', 'adc6', 'adc8', 'adc10', 'adc12', 'adc14'],
    amplitude_methods=[
        'minmax', 'zscore', 'robust', 'unit_vector', 'power',
        'chunk_4096_standardize', 'chunk_8192_standardize', 'chunk_16384_standardize'
    ],
    distance_functions=[
        'manhattan', 'euclidean', 'cosine', 'pearson', 'additive_symmetric',
        'braycurtis', 'canberra', 'chebyshev', 'jensenshannon', 'kullback_leibler',
        'kumar_hassebrook', 'sqeuclidean', 'taneja', 'wasserstein', 'wavehedges', 'fidelity'
    ],
    selection_strategy='position_balanced_per_file',
    min_segments_per_position=1,
    min_segments_per_file=3,
    random_seed=42,
    balanced_segments=True,
    position_balance_mode='at_least_one'
)


if __name__ == "__main__":
    # Test the configuration
    print("Testing Extended Experiment Configuration")
    print("=" * 80)
    
    config = RANDOM_50FILES_CONFIG
    
    # Validate
    if config.validate():
        print("✓ Configuration is valid\n")
    else:
        print("✗ Configuration is invalid\n")
    
    # Display summary
    print(config.summary())
    
    # Test file selection (requires database connection)
    try:
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            database='arc_detection',
            user='kjensen'
        )
        
        file_ids = config.select_files(conn)
        print(f"\n✓ Selected {len(file_ids)} files")
        print(f"  First 5 file IDs: {file_ids[:5]}")
        
        conn.close()
    except Exception as e:
        print(f"\n✗ Could not test file selection: {e}")