"""
Preprocessing rules for Transformations.

This module decides scaling/transformation techniques for dataset columns
based on statistical metrics such as skewness, range, and cardinality.
"""

# Import libraries
import pandas as pd
from typing import Dict
from scipy.stats import skew

# Import util modules
from src.utils.logging import get_logger
from src.utils.exceptions import RuleProcessingError
from src.utils.models import ColumnSchema, ColType

logger = get_logger(__name__)

def transformation_rules(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> Dict[str, str]:
    """
    Decide scaling/transformation techniques based on column statistics.

    Args:
        df: DataFrame in which the preprocessing occurs.
        column_stats: Stats of the dataset.
    
    Returns:
        dict: Returns column name and recommended transformation.
    
    Raises:
        RuleProcessingError: If transformation logic fails unexpectedly.
    """
    logger.info("Starting transformation rule evaluation...")
    transformations = {}

    try:
        for col, stats in column_stats.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame. Skipping.")
                continue

            logger.debug(f"Processing column '{col}' of type '{stats.type}'.")

            # For Numeric Columns
            if stats.type == ColType.NUMERIC:
                col_data = df[col].dropna()

                if col_data.nunique() == 1:
                    transformations[col] = "drop (constant feature)"
                    logger.warning(f"Column '{col}' is constant. Marked for drop.")
                    continue

                col_skew = skew(col_data)
                logger.debug(f"Column '{col}' skewness: {col_skew:.3f}")

                if abs(col_skew) > 1:
                    transformations[col] = "log-transform or Box-Cox"
                    logger.info(f"Column '{col}' is highly skewed. Suggested log/Box-Cox transform.")
                elif col_data.max() - col_data.min() > 1000:
                    transformations[col] = "standard-scaling"
                    logger.info(f"Column '{col}' has large range. Suggested standard scaling.")
                else:
                    transformations[col] = "min-max-scaling"
                    logger.info(f"Column '{col}' has moderate range. Suggested min-max scaling.")

            # For Categorical Columns
            elif stats.type == ColType.CATEGORICAL:
                cardinality = stats.unique or 0
                logger.debug(f"Column '{col}' cardinality: {cardinality}")

                if cardinality <= 10:
                    transformations[col] = "one-hot-encode"
                    logger.info(f"Column '{col}' is low cardinality. Suggested one-hot encoding.")
                elif cardinality <= 50:
                    transformations[col] = "target/frequency-encode"
                    logger.info(f"Column '{col}' is medium cardinality. Suggested target/frequency encoding.")
                else:
                    transformations[col] = "embedding-encode"
                    logger.warning(f"Column '{col}' is high cardinality. Suggested embedding encoding.")

            # For Datetime Columns
            elif stats.type == ColType.DATETIME:
                transformations[col] = "extract date parts (year, month, day, weekday)"
                logger.info(f"Column '{col}' is datetime. Suggested extracting date parts.")

            # For Boolean Columns
            elif stats.type == ColType.BOOLEAN:
                transformations[col] = "convert to int (0/1)"
                logger.info(f"Column '{col}' is boolean. Suggested converting to int (0/1).")

            else:
                transformations[col] = "no-action"
                logger.warning(f"Column '{col}' type '{stats.type}' not recognized. No transformation applied.")

        logger.info("Transformation rule evaluation completed successfully.")
        return transformations

    except Exception as e:
        logger.error(f"Error while applying transformation rules: {e}")
        raise RuleProcessingError(f"Failed to process transformation rules: {e}") from e