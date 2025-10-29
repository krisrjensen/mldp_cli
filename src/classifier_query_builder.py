#!/usr/bin/env python3
"""
Filename: classifier_query_builder.py
Author(s): Kristophor Jensen
Date Created: 20251028_153500
Date Revised: 20251028_153500
File version: 1.0.0.1
Description: SQL query builder for classifier results (classifier-agnostic)

This module provides SQL query building logic that works with any classifier type
registered in the ClassifierRegistry. It handles JOINs for lookup tables and
builds WHERE clauses based on filters.

Usage:
    from classifier_configs import ClassifierRegistry
    from classifier_query_builder import ClassifierQueryBuilder

    registry = ClassifierRegistry()
    config = registry.get('svm')
    builder = ClassifierQueryBuilder(config)
    query = builder.build_query(exp_id=41, cls_id=2)
"""

from typing import Dict, List, Optional, Any
from classifier_configs import ClassifierConfig, SharedColumnConfig


class ClassifierQueryBuilder:
    """Build database queries for any classifier type

    This class constructs SQL queries that are classifier-agnostic by using
    the ClassifierConfig to determine table names and hyperparameter columns.
    """

    def __init__(self, classifier_config: ClassifierConfig):
        """
        Initialize query builder with classifier configuration

        Args:
            classifier_config: Configuration for the classifier type
        """
        self.config = classifier_config

    def build_query(self, exp_id: int, cls_id: int, filters: Optional[Dict[str, Any]] = None) -> str:
        """Build SQL query for classifier results

        Args:
            exp_id: Experiment ID
            cls_id: Classifier ID
            filters: Optional dictionary of column -> value filters

        Returns:
            Complete SQL query string
        """
        table_name = self.config.get_table_name(exp_id, cls_id)

        query = f"""
            SELECT
                r.decimation_factor,
                dt.data_type_id,
                dt.data_type_name,
                am.method_id as amplitude_method_id,
                am.method_name as amplitude_method_name,
                efs.experiment_feature_set_id,
                fsl.feature_set_name,
                {self._get_hyperparameter_columns()}{self._get_metric_columns()}
            FROM {table_name} r
            JOIN ml_data_types_lut dt ON r.data_type_id = dt.data_type_id
            JOIN ml_amplitude_normalization_lut am
                ON r.amplitude_processing_method_id = am.method_id
            JOIN ml_experiments_feature_sets efs
                ON r.experiment_feature_set_id = efs.experiment_feature_set_id
            JOIN ml_feature_set_lut fsl
                ON efs.feature_set_id = fsl.feature_set_id
            WHERE 1=1
                {self._build_filter_clause(filters)}
            ORDER BY {self._get_order_by_clause()}
        """

        return query

    def _get_hyperparameter_columns(self) -> str:
        """Get classifier-specific hyperparameter columns for SELECT

        Returns:
            Comma-separated list of hyperparameter columns
        """
        columns = []
        for column_name in self.config.get_hyperparameter_columns():
            columns.append(f"r.{column_name}")

        if columns:
            return ",\n                ".join(columns) + ",\n                "
        return ""

    def _get_metric_columns(self) -> str:
        """Get standard metric columns for SELECT

        Returns:
            Comma-separated list of metric columns
        """
        columns = [f"r.{col}" for col in SharedColumnConfig.METRIC_COLUMNS]
        return ",\n                ".join(columns)

    def _build_filter_clause(self, filters: Optional[Dict[str, Any]]) -> str:
        """Build WHERE clause from filters dictionary

        Args:
            filters: Dictionary of column -> value or column -> list of values

        Returns:
            WHERE clause string (e.g., "AND r.decimation_factor = 0")
        """
        if not filters:
            return ""

        clauses = []
        for column, value in filters.items():
            if value is None:
                continue

            if isinstance(value, (list, tuple)):
                # Multiple values: IN clause
                if len(value) == 1:
                    clauses.append(f"AND r.{column} = {self._format_value(value[0])}")
                else:
                    values_str = ", ".join([self._format_value(v) for v in value])
                    clauses.append(f"AND r.{column} IN ({values_str})")
            else:
                # Single value: equality
                clauses.append(f"AND r.{column} = {self._format_value(value)}")

        return "\n                ".join(clauses)

    def _format_value(self, value: Any) -> str:
        """Format value for SQL query

        Args:
            value: Value to format

        Returns:
            Formatted value as string (with quotes if needed)
        """
        if isinstance(value, str):
            # Escape single quotes
            escaped = value.replace("'", "''")
            return f"'{escaped}'"
        elif value is None:
            return "NULL"
        else:
            return str(value)

    def _get_order_by_clause(self) -> str:
        """Get ORDER BY clause

        Returns:
            ORDER BY columns string
        """
        # Order by: hyperparameters, then feature_set, amplitude, decimation, data_type
        order_columns = []

        # Add hyperparameter columns
        for column_name in self.config.get_hyperparameter_columns():
            order_columns.append(f"r.{column_name}")

        # Add standard configuration columns
        order_columns.extend([
            "efs.experiment_feature_set_id",
            "am.method_id",
            "r.decimation_factor",
            "dt.data_type_id"
        ])

        return ", ".join(order_columns)

    def check_table_exists(self, cursor, exp_id: int, cls_id: int) -> bool:
        """Check if results table exists in database

        Args:
            cursor: Database cursor
            exp_id: Experiment ID
            cls_id: Classifier ID

        Returns:
            True if table exists, False otherwise
        """
        table_name = self.config.get_table_name(exp_id, cls_id)

        query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_name = %s
            )
        """

        cursor.execute(query, (table_name,))
        return cursor.fetchone()[0]

    def get_available_values(self, cursor, exp_id: int, cls_id: int, column: str) -> List[Any]:
        """Get distinct values for a column from results table

        Args:
            cursor: Database cursor
            exp_id: Experiment ID
            cls_id: Classifier ID
            column: Column name

        Returns:
            List of distinct values (sorted)
        """
        table_name = self.config.get_table_name(exp_id, cls_id)

        query = f"""
            SELECT DISTINCT {column}
            FROM {table_name}
            ORDER BY {column}
        """

        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]
