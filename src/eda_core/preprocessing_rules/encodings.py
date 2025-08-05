"""
Categorical Encoding Rules.

This module determines the appropriate encoding technique for categorical
features based on cardinality and dataset size.
"""

#Import libraries
import pandas as pd
from typing import Dict

#Import utils modules
from src.utils.logging import get_logger
from src.utils.exceptions import RuleProcessingError
from src.utils.models import ColumnSchema, ColType

logger = get_logger(__name__)

def encoding_rules(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> Dict[str, str]:
    """
    Determine encoding strategy for categorical features.

    Args:
        df (pd.DataFrame): The dataset to evaluate.
        column_stats (Dict[str, ColumnStats]): Profiling statistics for each column.

    Returns:
        Dict[str, str]: Mapping of column names to recommended encoding strategies.

    Raises:
        RuleProcessingError: If rule evaluation fails.
    """
    logger.info("Starting encoding rule evaluation...")
    strategies = {}

    try:
        for col, stats in column_stats.items():
            if stats.type != ColType.CATEGORICAL:
                continue

            cardinality = stats.unique or 0
            logger.debug(f"Column '{col}' cardinality: {cardinality}")

            if cardinality <= 10:
                strategies[col] = "one-hot-encode"
                logger.info(f"Column '{col}' is low cardinality. Recommended one-hot encoding.")
            elif cardinality <= 50:
                strategies[col] = "target-encode"
                logger.info(f"Column '{col}' is medium cardinality. Recommended target encoding.")
            else:
                strategies[col] = "embedding-encode"
                logger.warning(f"Column '{col}' is high cardinality. Recommended embeddings.")

        logger.info("Encoding rule evaluation completed successfully.")
        return strategies

    except Exception as e:
        logger.error(f"Error while applying encoding rules: {e}")
        raise RuleProcessingError(f"Failed to process encoding rules: {e}") from e
