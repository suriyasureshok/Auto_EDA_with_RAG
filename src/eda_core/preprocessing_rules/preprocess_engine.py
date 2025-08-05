"""
Preprocessing Rules Engine.

This module orchestrates all preprocessing rule evaluations and returns
a consolidated decision map for transformations, missing values,
outlier handling, and encoding strategies.
"""

#Import libraries
import pandas as pd
from typing import Dict

#Import util modules
from src.utils.logging import get_logger
from src.utils.exceptions import RuleProcessingError
from src.utils.models import ColumnSchema

#Import rules
from src.eda_core.preprocessing_rules.transformations import transformation_rules
from src.eda_core.preprocessing_rules.missing_values import missing_value_rules
from src.eda_core.preprocessing_rules.outliers import outlier_rules
from src.eda_core.preprocessing_rules.encodings import encoding_rules

logger = get_logger(__name__)

def run_preprocessing_rules(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> Dict[str, Dict[str, str]]:
    """
    Run all preprocessing rule modules and return consolidated results.

    Args:
        df (pd.DataFrame): The dataset to evaluate.
        column_stats (Dict[str, ColumnStats]): Profiling statistics for each column.

    Returns:
        Dict[str, Dict[str, str]]: Dictionary with rule category as keys and 
                                   {column_name: decision} as values.
    """
    logger.info("Running preprocessing rules engine...")

    try:
        results = {
            "transformations": transformation_rules(df, column_stats),
            "missing_values": missing_value_rules(df, column_stats),
            "outliers": outlier_rules(df, column_stats),
            "encodings": encoding_rules(df, column_stats)
        }
        logger.info("Preprocessing rules engine completed successfully.")
        return results

    except Exception as e:
        logger.error(f"Error while running preprocessing rules engine: {e}")
        raise RuleProcessingError(f"Preprocessing rules engine failed: {e}") from e
