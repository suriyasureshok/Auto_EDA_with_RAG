"""
Categorical Encoding Rules.

This module determines the appropriate encoding technique for categorical
features based on cardinality and dataset size.
Also provides an execution function to apply the encodings to a DataFrame.
"""

# Import libraries
import pandas as pd
from typing import Dict
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import TargetEncoder

# Import utils modules
from src.utils.logging import get_logger
from src.utils.exceptions import RuleProcessingError, PreprocessingError
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


def encode_and_scale(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema], target_column: str = None) -> pd.DataFrame:
    """
    Apply encoding and scaling transformations to the dataset.

    Args:
        df (pd.DataFrame): The dataset to transform.
        column_stats (Dict[str, ColumnSchema]): Profiling statistics for each column.
        target_column (str, optional): Target variable, required for target encoding.

    Returns:
        pd.DataFrame: Transformed dataset with encoded and scaled features.

    Raises:
        PreprocessingError: If transformation fails.
    """
    logger.info("Starting encoding & scaling process...")
    try:
        strategies = encoding_rules(df, column_stats)

        for col, strategy in strategies.items():
            logger.debug(f"Applying {strategy} to column '{col}'...")

            if strategy == "one-hot-encode":
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
                encoded = encoder.fit_transform(df[[col]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
                logger.info(f"One-hot encoding applied to '{col}'.")

            elif strategy == "target-encode":
                if target_column is None or target_column not in df.columns:
                    logger.error(f"Target column required for target encoding '{col}' but not provided.")
                    raise PreprocessingError(f"Target column required for target encoding '{col}'")
                encoder = TargetEncoder()
                df[col] = encoder.fit_transform(df[col], df[target_column])
                logger.info(f"Target encoding applied to '{col}'.")

            elif strategy == "embedding-encode":
                # Placeholder for embedding-based encoding
                logger.warning(f"Embedding encoding not implemented. Column '{col}' left unchanged.")

        # Apply scaling to numeric columns
        numeric_cols = [col for col, stats in column_stats.items() if stats.type == ColType.NUMERIC and col in df.columns]
        if numeric_cols:
            logger.debug(f"Scaling numeric columns: {numeric_cols}")
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            logger.info("Numeric feature scaling applied.")

        logger.info("Encoding & scaling process completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Encoding & scaling process failed: {e}")
        raise PreprocessingError(f"Encoding & scaling process failed: {e}") from e
