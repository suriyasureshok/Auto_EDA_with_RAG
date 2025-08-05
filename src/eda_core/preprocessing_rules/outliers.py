"""
Outlier Detection and Handling Rules.

This module applies statistical methods (IQR, Z-score) to detect and
recommend handling strategies for outliers in numeric columns.
"""

#Import libraries
import pandas as pd
import numpy as np
from typing import Dict

#Import util modules
from src.utils.logging import get_logger
from src.utils.exceptions import RuleProcessingError
from src.utils.models import ColumnSchema, ColType

logger = get_logger(__name__)

def outlier_rules(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> Dict[str, str]:
    """
    Determine outlier handling strategy for each numeric column.

    Args:
        df (pd.DataFrame): The dataset to evaluate.
        column_stats (Dict[str, ColumnStats]): Profiling statistics for each column.

    Returns:
        Dict[str, str]: Mapping of column names to recommended outlier handling strategies.

    Raises:
        RuleProcessingError: If rule evaluation fails.
    """
    logger.info("Starting outlier detection rule evaluation...")
    strategies = {}

    try:
        for col, stats in column_stats.items():
            if stats.type != ColType.NUMERIC:
                continue

            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
                continue

            col_data = df[col].dropna()
            if col_data.empty:
                strategies[col] = "no-action"
                logger.warning(f"Column '{col}' has no data after NA removal.")
                continue

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

            logger.debug(f"Column '{col}': IQR={IQR:.3f}, Outliers={outlier_count}")

            if outlier_count == 0:
                strategies[col] = "no-action"
                logger.info(f"Column '{col}' has no detected outliers.")
            elif outlier_count / len(col_data) > 0.2:
                strategies[col] = "cap-at-percentiles"
                logger.info(f"Column '{col}' has >20% outliers. Recommended capping at percentiles.")
            else:
                strategies[col] = "remove-outliers"
                logger.info(f"Column '{col}' has moderate outliers. Recommended removal.")

        logger.info("Outlier detection rule evaluation completed successfully.")
        return strategies

    except Exception as e:
        logger.error(f"Error while applying outlier detection rules: {e}")
        raise RuleProcessingError(f"Failed to process outlier rules: {e}") from e
