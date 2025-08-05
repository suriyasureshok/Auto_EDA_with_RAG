"""
Missing Value Handling Rules.

This module determines the optimal strategy for handling missing values
based on column statistics and data distribution. The rules here are
deterministic and aim to minimize bias introduced by imputation.
"""

#Import libraries
import pandas as pd
from typing import Dict

#Import util modules
from src.utils.logging import get_logger
from src.utils.exceptions import RuleProcessingError
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
                logger.info(f"Column '{col}' has no missing values. No action needed.")
                continue

            if missing_pct > 50:
                strategies[col] = "drop-column"
                logger.warning(f"Column '{col}' has >50% missing values. Recommended drop.")
                continue

            if stats.type == ColType.NUMERIC:
                skewness = df[col].dropna().skew()
                if skewness > 1:
                    strategies[col] = "impute-median"
                    logger.info(f"Column '{col}' is skewed. Recommended median imputation.")
                else:
                    strategies[col] = "impute-mean"
                    logger.info(f"Column '{col}' is symmetric. Recommended mean imputation.")

            elif stats.type == ColType.CATEGORICAL:
                strategies[col] = "impute-mode"
                logger.info(f"Column '{col}' is categorical. Recommended mode imputation.")

            elif stats.type == ColType.DATETIME:
                strategies[col] = "impute-most-frequent-date"
                logger.info(f"Column '{col}' is datetime. Recommended most frequent date imputation.")

            elif stats.type == ColType.BOOLEAN:
                strategies[col] = "impute-mode"
                logger.info(f"Column '{col}' is boolean. Recommended mode imputation.")

            else:
                strategies[col] = "no-action"
                logger.warning(f"Column '{col}' type '{stats.type}' not recognized. No imputation applied.")

        logger.info("Missing value rule evaluation completed successfully.")
        return strategies

    except Exception as e:
        logger.error(f"Error while applying missing value rules: {e}")
        raise RuleProcessingError(f"Failed to process missing value rules: {e}") from e
