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

def handle_outliers(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> pd.DataFrame:
    """
    Apply outlier handling to a dataset based on determined rules.

    Args:
        df (pd.DataFrame): Input dataset.
        column_stats (Dict[str, ColumnSchema]): Metadata for each column.

    Returns:
        pd.DataFrame: Dataset with outliers handled.

    Raises:
        PreprocessingError: If processing fails.
    """
    from src.utils.exceptions import PreprocessingError

    logger.info("Starting outlier handling...")
    try:
        strategies = outlier_rules(df, column_stats)

        for col, action in strategies.items():
            if action == "no-action":
                continue

            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame during outlier handling. Skipping.")
                continue

            col_data = df[col].dropna()

            if action == "remove-outliers":
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                initial_shape = df.shape
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                logger.info(f"Removed outliers from '{col}'. Rows before: {initial_shape[0]}, after: {df.shape[0]}.")

            elif action == "cap-at-percentiles":
                lower_cap = col_data.quantile(0.05)
                upper_cap = col_data.quantile(0.95)
                df[col] = np.clip(df[col], lower_cap, upper_cap)
                logger.info(f"Capped values in '{col}' between {lower_cap:.3f} and {upper_cap:.3f}.")

            else:
                logger.warning(f"Unknown outlier handling strategy '{action}' for column '{col}'. Skipping.")

        logger.info("Outlier handling completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during outlier handling: {e}")
        raise PreprocessingError(f"Failed to handle outliers: {e}") from e
