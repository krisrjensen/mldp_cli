#!/usr/bin/env python3
"""
Experiment Generation Configuration Manager

Filename: experiment_generation_config.py
Author(s): Kristophor Jensen  
Date Created: 20250907_174000
Date Revised: 20250907_174000
File version: 0.0.0.1
Description: Configuration management for experiment generation with segment selection
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentGenerationConfig:
    """Configuration for experiment generation with segment selection"""
    
    # Basic settings
    experiment_name: str
    experiment_type: str = 'classification'
    experiment_version: str = '1.0.0'
    experiment_description: str = ''
    
    # Data selection
    target_labels: List[int] = field(default_factory=lambda: [1, 4, 5])
    instances_per_label: int = 750
    
    # Segment configuration  
    segment_sizes: List[int] = field(default_factory=lambda: [128, 1024])
    decimation_factors: List[int] = field(default_factory=lambda: [0, 1, 3])
    data_types: List[str] = field(default_factory=lambda: ['raw'])
    amplitude_methods: List[str] = field(default_factory=lambda: ['none'])
    
    # Selection strategy
    selection_strategy: str = 'random'
    random_seed: Optional[int] = 42
    balanced_segments: bool = True
    position_balance_mode: str = 'equal'
    
    # Quality filters
    min_quality_score: float = 1.0
    validation_status_filter: str = 'all'  # 'expert', 'auto', 'all'
    exclude_edge_segments: bool = False
    require_full_transient: bool = False
    
    # Pipeline configuration
    pipeline_id: Optional[int] = None
    skip_existing: bool = True
    dry_run: bool = False
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Check required fields
        if not self.experiment_name:
            errors.append("experiment_name is required")
        
        if self.instances_per_label <= 0:
            errors.append("instances_per_label must be positive")
        
        if not self.target_labels:
            errors.append("target_labels cannot be empty")
        
        if not self.segment_sizes:
            errors.append("segment_sizes cannot be empty")
        
        # Check segment sizes are valid
        valid_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 
                      32768, 65536, 131072, 262144, 524288]
        for size in self.segment_sizes:
            if size not in valid_sizes:
                errors.append(f"Invalid segment size: {size}")
        
        # Check selection strategy
        valid_strategies = ['random', 'quality_first', 'distributed', 'balanced']
        if self.selection_strategy not in valid_strategies:
            errors.append(f"Invalid selection_strategy: {self.selection_strategy}")
        
        # Check position balance mode
        valid_balance_modes = ['equal', 'proportional', 'at_least_one']
        if self.position_balance_mode not in valid_balance_modes:
            errors.append(f"Invalid position_balance_mode: {self.position_balance_mode}")
        
        # Check decimation factors
        for factor in self.decimation_factors:
            if factor < 0:
                errors.append(f"Invalid decimation factor: {factor}")
        
        # Check quality score
        if not 0.0 <= self.min_quality_score <= 1.0:
            errors.append(f"min_quality_score must be between 0.0 and 1.0")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        return True
    
    def to_json(self) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.__dict__, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'experiment_name': self.experiment_name,
            'experiment_type': self.experiment_type,
            'experiment_version': self.experiment_version,
            'experiment_description': self.experiment_description,
            'target_labels': self.target_labels,
            'instances_per_label': self.instances_per_label,
            'segment_sizes': self.segment_sizes,
            'decimation_factors': self.decimation_factors,
            'data_types': self.data_types,
            'amplitude_methods': self.amplitude_methods,
            'selection_strategy': self.selection_strategy,
            'random_seed': self.random_seed,
            'balanced_segments': self.balanced_segments,
            'position_balance_mode': self.position_balance_mode,
            'min_quality_score': self.min_quality_score,
            'validation_status_filter': self.validation_status_filter,
            'exclude_edge_segments': self.exclude_edge_segments,
            'require_full_transient': self.require_full_transient,
            'pipeline_id': self.pipeline_id,
            'skip_existing': self.skip_existing,
            'dry_run': self.dry_run
        }
    
    def get_segment_selection_config(self) -> Dict[str, Any]:
        """Get segment selection configuration for database storage"""
        return {
            'selection_strategy': self.selection_strategy,
            'random_seed': self.random_seed,
            'balanced_segments': self.balanced_segments,
            'position_balance': {
                'mode': self.position_balance_mode,
                'minimum_per_position': 1
            },
            'label_balance': {
                'target_per_label': self.instances_per_label,
                'enforce_exact': False
            },
            'filters': {
                'exclude_edge_segments': self.exclude_edge_segments,
                'require_full_transient': self.require_full_transient,
                'min_quality_score': self.min_quality_score,
                'validation_status_filter': self.validation_status_filter
            }
        }
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ExperimentGenerationConfig':
        """Create configuration from JSON string"""
        data = json.loads(json_str)
        return cls(**data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentGenerationConfig':
        """Create configuration from dictionary"""
        return cls(**data)
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = [
            f"Experiment: {self.experiment_name}",
            f"Type: {self.experiment_type} v{self.experiment_version}",
            f"Labels: {len(self.target_labels)} labels × {self.instances_per_label} instances = {len(self.target_labels) * self.instances_per_label} total",
            f"Segment sizes: {self.segment_sizes}",
            f"Selection: {self.selection_strategy} (seed={self.random_seed})",
            f"Position balance: {self.balanced_segments} (mode={self.position_balance_mode})",
            f"Quality threshold: {self.min_quality_score}"
        ]
        return "\n".join(lines)


# Predefined configurations
BALANCED_18CLASS_CONFIG = ExperimentGenerationConfig(
    experiment_name="balanced_18class_750each",
    experiment_description="Balanced dataset with 18 classes and 750 instances each",
    target_labels=list(range(1, 19)),  # Classes 1-18
    instances_per_label=750,
    segment_sizes=[128, 1024],
    selection_strategy="random",
    random_seed=42,
    balanced_segments=True,
    position_balance_mode="equal"
)

SMALL_TEST_CONFIG = ExperimentGenerationConfig(
    experiment_name="test_3class_100each",
    experiment_description="Small test dataset with 3 classes and 100 instances each",
    target_labels=[1, 4, 5],
    instances_per_label=100,
    segment_sizes=[128],
    selection_strategy="random",
    random_seed=42,
    balanced_segments=True,
    position_balance_mode="at_least_one"
)

LARGE_UNBALANCED_CONFIG = ExperimentGenerationConfig(
    experiment_name="large_18class_unbalanced",
    experiment_description="Large unbalanced dataset with all segment sizes",
    target_labels=list(range(1, 19)),
    instances_per_label=1000,
    segment_sizes=[128, 256, 512, 1024, 2048],
    selection_strategy="quality_first",
    random_seed=None,  # No seed for variation
    balanced_segments=False,
    position_balance_mode="proportional"
)


def load_config_from_file(filepath: str) -> ExperimentGenerationConfig:
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return ExperimentGenerationConfig.from_dict(data)


def save_config_to_file(config: ExperimentGenerationConfig, filepath: str):
    """Save configuration to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


if __name__ == "__main__":
    # Test the configuration
    print("Testing Experiment Generation Configuration")
    print("=" * 60)
    
    # Test balanced configuration
    config = BALANCED_18CLASS_CONFIG
    print("\nBalanced 18-class configuration:")
    print(config.summary())
    
    # Validate
    if config.validate():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration is invalid")
    
    # Test JSON serialization
    json_str = config.to_json()
    print(f"\nJSON length: {len(json_str)} characters")
    
    # Test deserialization
    config2 = ExperimentGenerationConfig.from_json(json_str)
    print(f"Deserialized: {config2.experiment_name}")
    
    # Test segment selection config
    sel_config = config.get_segment_selection_config()
    print(f"\nSegment selection config keys: {list(sel_config.keys())}")
    
    # Test small config
    print("\n" + "=" * 60)
    print("Small test configuration:")
    print(SMALL_TEST_CONFIG.summary())
    
    # Test large config
    print("\n" + "=" * 60)
    print("Large unbalanced configuration:")
    print(LARGE_UNBALANCED_CONFIG.summary())