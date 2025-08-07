"""
Missing Value Handling Rules & Processing.

This module determines the optimal strategy for handling missing values
based on column statistics and data distribution, and applies the chosen
strategies to the dataset.
"""

# Import libraries
import pandas as pd
from typing import Dict

# Import util modules
from src.utils.logging import get_logger
from src.utils.exceptions import RuleProcessingError, PreprocessingError
from src.utils.models import ColumnSchema, ColType

logger = get_logger(__name__)

def missing_value_rules(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> Dict[str, str]:
    """
    Determine missing value handling strategy for each column.

    Args:
        df (pd.DataFrame): The dataset to evaluate.
        column_stats (Dict[str, ColumnStats]): Profiling statistics for each column.

    Returns:
        Dict[str, str]: Mapping of column names to recommended missing value handling strategies.

    Raises:
        RuleProcessingError: If rule evaluation fails.
    """
    logger.info("Starting missing value rule evaluation...")
    strategies = {}

    try:
        for col, stats in column_stats.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
                continue

            missing_pct = stats.missing_pct or 0.0
            logger.debug(f"Column '{col}' missing %: {missing_pct:.2f}")

            if missing_pct == 0:
                strategies[col] = "no-action"
                continue

            if missing_pct > 50:
                strategies[col] = "drop-column"
                continue

            if stats.type == ColType.NUMERIC:
                skewness = df[col].dropna().skew()
                strategies[col] = "impute-median" if skewness > 1 else "impute-mean"

            elif stats.type == ColType.CATEGORICAL:
                strategies[col] = "impute-mode"

            elif stats.type == ColType.DATETIME:
                strategies[col] = "impute-most-frequent-date"

            elif stats.type == ColType.BOOLEAN:
                strategies[col] = "impute-mode"

            else:
                strategies[col] = "no-action"

        logger.info("Missing value rule evaluation completed successfully.")
        return strategies

    except Exception as e:
        logger.error(f"Error while applying missing value rules: {e}")
        raise RuleProcessingError(f"Failed to process missing value rules: {e}") from e


def handle_missing_values(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> pd.DataFrame:
    """
    Apply missing value handling to a dataset based on determined rules.

    Args:
        df (pd.DataFrame): Input dataset.
        column_stats (Dict[str, ColumnSchema]): Metadata for each column.

    Returns:
        pd.DataFrame: Dataset with missing values handled.

    Raises:
        PreprocessingError: If processing fails.
    """
    logger.info("Starting missing value handling...")
    try:
        strategies = missing_value_rules(df, column_stats)

        for col, action in strategies.items():
            if action == "no-action":
                continue
            elif action == "drop-column":
                logger.info(f"Dropping column '{col}' due to high missing percentage.")
                df = df.drop(columns=[col])
            elif action == "impute-mean":
                df[col] = df[col].fillna(df[col].mean())
                logger.info(f"Applied mean imputation for column '{col}'.")
            elif action == "impute-median":
                df[col] = df[col].fillna(df[col].median())
                logger.info(f"Applied median imputation for column '{col}'.")
            elif action == "impute-mode":
                df[col] = df[col].fillna(df[col].mode()[0])
                logger.info(f"Applied mode imputation for column '{col}'.")
            elif action == "impute-most-frequent-date":
                most_freq_date = df[col].dropna().mode()[0]
                df[col] = df[col].fillna(most_freq_date)
                logger.info(f"Applied most frequent date imputation for column '{col}'.")
            else:
                logger.warning(f"No recognized action for column '{col}'. Skipping.")

        logger.info("Missing value handling completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during missing value handling: {e}")
        raise PreprocessingError(f"Failed to handle missing values: {e}") from e
