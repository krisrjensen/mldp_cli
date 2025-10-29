#!/usr/bin/env python3
"""
Filename: classifier_configs.py
Author(s): Kristophor Jensen
Date Created: 20251028_153000
Date Revised: 20251028_153000
File version: 1.0.0.1
Description: Classifier configuration registry pattern for extensible classifier support

This module provides a registration-based architecture for managing different classifier types
(SVM, Random Forest, XGBoost, etc.) without requiring modifications to core plotting logic.

Usage:
    from classifier_configs import ClassifierRegistry

    registry = ClassifierRegistry()
    config = registry.get('svm')
    table_name = config.get_table_name(exp_id, cls_id)
"""

from typing import Dict, Callable, List, Optional


class ClassifierConfig:
    """Base configuration class for classifier types

    Each classifier type (SVM, RF, XGBoost, etc.) registers its configuration
    including table naming, hyperparameter mapping, and directory formatting.
    """

    def __init__(self, name: str, display_name: str):
        """
        Initialize classifier configuration

        Args:
            name: Short name for CLI (e.g., 'svm', 'rf', 'xgboost')
            display_name: Human-readable name (e.g., 'SVM', 'Random Forest')
        """
        self.name = name
        self.display_name = display_name
        self.table_suffix = None  # e.g., '_svm_results', '_rf_results'
        self.hyperparameters: Dict[str, str] = {}  # generic_name -> column_name
        self.group_by_formatters: Dict[str, Callable] = {}  # column_name -> formatter

    def get_table_name(self, exp_id: int, cls_id: int) -> str:
        """Get results table name for this classifier

        Args:
            exp_id: Experiment ID
            cls_id: Classifier ID

        Returns:
            Table name (e.g., 'experiment_041_classifier_002_svm_results')
        """
        return f"experiment_{exp_id:03d}_classifier_{cls_id:03d}{self.table_suffix}"

    def get_hyperparameter_column(self, param_name: str) -> str:
        """Map generic parameter name to actual database column

        Args:
            param_name: Generic parameter name (e.g., 'c_parameter')

        Returns:
            Actual column name (e.g., 'svm_c_parameter')
        """
        return self.hyperparameters.get(param_name, param_name)

    def format_group_value(self, group_column: str, value) -> str:
        """Format group-by value for directory naming

        Args:
            group_column: Column being grouped by
            value: Value to format

        Returns:
            Formatted directory name (e.g., 'C1_0', 'N100')
        """
        formatter = self.group_by_formatters.get(group_column)
        if formatter:
            return formatter(value)
        # Default: replace periods and spaces with underscores
        return str(value).replace('.', '_').replace(' ', '_')

    def get_hyperparameter_columns(self) -> List[str]:
        """Get list of hyperparameter column names

        Returns:
            List of database column names for hyperparameters
        """
        return list(self.hyperparameters.values())


class ClassifierRegistry:
    """Singleton registry for classifier configurations

    Maintains a registry of all available classifier types and their configurations.
    New classifiers can be added by creating a configuration and registering it.
    """

    _instance = None

    def __new__(cls):
        """Ensure singleton pattern - only one registry exists"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._classifiers = {}
            cls._instance._initialized = False
        return cls._instance

    def register(self, config: ClassifierConfig):
        """Register a classifier configuration

        Args:
            config: ClassifierConfig instance to register
        """
        self._classifiers[config.name] = config

    def get(self, name: str) -> ClassifierConfig:
        """Get classifier configuration by name

        Args:
            name: Classifier name (e.g., 'svm', 'rf')

        Returns:
            ClassifierConfig instance

        Raises:
            ValueError: If classifier name not found in registry
        """
        if name not in self._classifiers:
            available = ', '.join(self.list_classifiers())
            raise ValueError(f"Unknown classifier type: '{name}'. Available: {available}")
        return self._classifiers[name]

    def list_classifiers(self) -> List[str]:
        """List all registered classifier names

        Returns:
            List of registered classifier names
        """
        return sorted(self._classifiers.keys())

    def has_classifier(self, name: str) -> bool:
        """Check if classifier is registered

        Args:
            name: Classifier name

        Returns:
            True if registered, False otherwise
        """
        return name in self._classifiers


class SharedColumnConfig:
    """Configuration for columns shared across all classifiers

    Defines standard configuration columns (decimation_factor, data_type, etc.)
    and metric columns that are common to all classifier types.
    """

    # Standard configuration columns
    CONFIG_COLUMNS = {
        'decimation_factor': {
            'display_name': 'Decimation Factor',
            'format': lambda v: f"D{int(v)}",
            'data_type': 'integer'
        },
        'data_type_id': {
            'display_name': 'Data Type',
            'format': lambda v: f"TADC{int(v)}",
            'lookup_table': 'ml_data_types_lut',
            'lookup_column': 'data_type_name',
            'data_type': 'integer'
        },
        'amplitude_processing_method_id': {
            'display_name': 'Amplitude Method',
            'format': lambda v: f"A{int(v)}",
            'lookup_table': 'ml_amplitude_normalization_lut',
            'lookup_column': 'method_name',
            'data_type': 'integer'
        },
        'experiment_feature_set_id': {
            'display_name': 'Feature Set',
            'format': lambda v: f"EFS{int(v):03d}",
            'lookup_table': 'ml_experiments_feature_sets',
            'lookup_join': 'ml_feature_set_lut',
            'lookup_column': 'feature_set_name',
            'data_type': 'integer'
        }
    }

    # Metric columns (same for all classifiers)
    METRIC_COLUMNS = [
        'arc_roc_auc_train', 'arc_f1_train', 'arc_pr_auc_train',
        'arc_roc_auc_test', 'arc_f1_test', 'arc_pr_auc_test',
        'arc_roc_auc_verify', 'arc_f1_verify', 'arc_pr_auc_verify',
        'accuracy_train', 'accuracy_test', 'accuracy_verify'
    ]

    @classmethod
    def get_format_function(cls, column_name: str) -> Optional[Callable]:
        """Get formatting function for a column

        Args:
            column_name: Column name

        Returns:
            Formatting function or None
        """
        col_config = cls.CONFIG_COLUMNS.get(column_name)
        return col_config['format'] if col_config else None


def create_svm_config() -> ClassifierConfig:
    """Create SVM classifier configuration

    Returns:
        ClassifierConfig for SVM
    """
    config = ClassifierConfig('svm', 'SVM')
    config.table_suffix = '_svm_results'

    # Hyperparameter mapping: generic name -> database column
    config.hyperparameters = {
        'kernel': 'svm_kernel',
        'c_parameter': 'svm_c_parameter',
        'gamma': 'svm_gamma'
    }

    # Group-by directory name formatters
    config.group_by_formatters = {
        'svm_c_parameter': lambda v: f"C{str(v).replace('.', '_')}",
        'svm_kernel': lambda v: f"kernel_{v}",
        'svm_gamma': lambda v: f"gamma_{str(v).replace('.', '_') if v else 'auto'}",
        # Common columns
        'decimation_factor': lambda v: f"D{int(v)}",
        'data_type_id': lambda v: f"TADC{int(v)}",
        'amplitude_processing_method_id': lambda v: f"A{int(v)}",
        'experiment_feature_set_id': lambda v: f"EFS{int(v):03d}"
    }

    return config


def create_rf_config() -> ClassifierConfig:
    """Create Random Forest classifier configuration

    Returns:
        ClassifierConfig for Random Forest
    """
    config = ClassifierConfig('rf', 'Random Forest')
    config.table_suffix = '_rf_results'

    # Hyperparameter mapping
    config.hyperparameters = {
        'n_estimators': 'rf_n_estimators',
        'max_depth': 'rf_max_depth',
        'min_samples_split': 'rf_min_samples_split',
        'max_features': 'rf_max_features'
    }

    # Group-by directory name formatters
    config.group_by_formatters = {
        'rf_n_estimators': lambda v: f"N{int(v)}",
        'rf_max_depth': lambda v: f"depth_{int(v) if v is not None else 'None'}",
        'rf_min_samples_split': lambda v: f"split_{int(v)}",
        'rf_max_features': lambda v: f"features_{v if v else 'all'}",
        # Common columns
        'decimation_factor': lambda v: f"D{int(v)}",
        'data_type_id': lambda v: f"TADC{int(v)}",
        'amplitude_processing_method_id': lambda v: f"A{int(v)}",
        'experiment_feature_set_id': lambda v: f"EFS{int(v):03d}"
    }

    return config


# Auto-register built-in classifiers at module import
_registry = ClassifierRegistry()
_registry.register(create_svm_config())
_registry.register(create_rf_config())


# Example: Future classifier configuration (commented out)
"""
def create_xgboost_config() -> ClassifierConfig:
    '''Create XGBoost classifier configuration - FUTURE'''
    config = ClassifierConfig('xgboost', 'XGBoost')
    config.table_suffix = '_xgboost_results'

    config.hyperparameters = {
        'n_estimators': 'xgb_n_estimators',
        'max_depth': 'xgb_max_depth',
        'learning_rate': 'xgb_learning_rate',
        'subsample': 'xgb_subsample'
    }

    config.group_by_formatters = {
        'xgb_learning_rate': lambda v: f"lr_{str(v).replace('.', '_')}",
        'xgb_n_estimators': lambda v: f"N{int(v)}",
        'xgb_max_depth': lambda v: f"depth_{int(v)}",
        'decimation_factor': lambda v: f"D{int(v)}",
        'data_type_id': lambda v: f"TADC{int(v)}",
        'amplitude_processing_method_id': lambda v: f"A{int(v)}",
        'experiment_feature_set_id': lambda v: f"EFS{int(v):03d}"
    }

    return config

# To enable XGBoost support, uncomment these lines:
# _registry.register(create_xgboost_config())
"""
