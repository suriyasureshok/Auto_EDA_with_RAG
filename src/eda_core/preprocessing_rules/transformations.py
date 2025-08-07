"""
Advanced Preprocessing Rules for Transformations.

This module decides scaling/transformation techniques based on:
    - Data semantics (IDs, age, percentages, money, etc.)
    - Statistical metrics (skewness, range, cardinality)
    - Interpretability requirements
    - Robust handling of edge cases
"""

import pandas as pd
import numpy as np
from typing import Dict
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.logging import get_logger
from src.utils.exceptions import PreprocessingError, RuleProcessingError
from src.utils.models import ColumnSchema, ColType

logger = get_logger(__name__)

# -------- Utility semantic checks --------
def is_id_like(series: pd.Series) -> bool:
    """Detect if a numeric column is likely an identifier."""
    return (
        pd.api.types.is_integer_dtype(series) and
        series.nunique() / len(series) > 0.98 and
        series.nunique() > 50
    )

def is_count_like(series: pd.Series) -> bool:
    """Detect count-like small integer features."""
    return pd.api.types.is_integer_dtype(series) and series.nunique() < 20

def is_age_like(name: str, series: pd.Series) -> bool:
    """Detect if column is age-like based on name and value distribution."""
    lower_name = name.lower()

    # Keywords that indicate age
    keywords = ["age", "yrs", "years", "yr", "age_in_years"]
    if any(kw in lower_name for kw in keywords):
        return True

    # If name doesn't indicate, check numeric range
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if non_null.between(0, 120).mean() > 0.8:  # 80% of values between 0 and 120
            return True

    return False


def is_percentage_like(series: pd.Series) -> bool:
    """Detect if column is percentage or ratio."""
    return series.between(0, 1).all() or series.between(0, 100).all()

def is_money_like(name: str) -> bool:
    """Detect if column represents monetary values."""
    return any(keyword in name.lower() for keyword in ["price", "cost", "revenue", "income", "amount", "salary"])

# -------- Transformation Rule Engine --------
def transformation_rules(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> Dict[str, str]:
    """
    Decide scaling/transformation techniques based on advanced semantic & statistical rules.
    """
    logger.info("Starting advanced transformation rule evaluation...")
    transformations = {}

    try:
        for col, stats in column_stats.items():
            if col not in df.columns:
                continue

            col_data = df[col].dropna()

            # Empty column
            if col_data.empty:
                transformations[col] = "drop (empty column)"
                continue

            # Drop constant
            if col_data.nunique() == 1:
                transformations[col] = "drop (constant feature)"
                continue

            # ----- NUMERIC -----
            if stats.type == ColType.NUMERIC:
                # Drop ID-like
                if is_id_like(col_data):
                    transformations[col] = "drop (identifier-like)"
                    continue

                # Leave counts as-is
                if is_count_like(col_data):
                    transformations[col] = "no-transform (count-like)"
                    continue

                # Age-like: keep raw
                if is_age_like(col, col_data):
                    transformations[col] = "no-transform (age-like)"
                    continue

                # Percentages: keep raw
                if is_percentage_like(col_data):
                    transformations[col] = "no-transform (percentage/ratio)"
                    continue

                # Monetary: scale but avoid log unless skewed
                if is_money_like(col):
                    col_skew = skew(col_data)
                    if abs(col_skew) > 1 and (col_data > 0).all():
                        transformations[col] = "log-transform"
                    else:
                        transformations[col] = "standard-scaling"
                    continue

                # Skewness-driven transform
                col_skew = skew(col_data)
                if abs(col_skew) > 1 and (col_data > 0).all():
                    transformations[col] = "log-transform"
                elif col_data.max() - col_data.min() > 1000:
                    transformations[col] = "standard-scaling"
                else:
                    transformations[col] = "min-max-scaling"

            # ----- CATEGORICAL -----
            elif stats.type == ColType.CATEGORICAL:
                cardinality = stats.unique or col_data.nunique()
                if cardinality <= 10:
                    transformations[col] = "one-hot-encode"
                elif cardinality <= 50:
                    transformations[col] = "target/frequency-encode"
                else:
                    transformations[col] = "embedding-encode"

            # ----- DATETIME -----
            elif stats.type == ColType.DATETIME:
                transformations[col] = "extract date parts (year, month, day, weekday, hour)"

            # ----- BOOLEAN -----
            elif stats.type == ColType.BOOLEAN:
                transformations[col] = "convert to int (0/1)"

            # ----- UNKNOWN -----
            else:
                transformations[col] = "no-action"

        logger.info("Advanced transformation rule evaluation completed.")
        return transformations

    except Exception as e:
        logger.error(f"Error while applying transformation rules: {e}")
        raise RuleProcessingError(f"Failed to process transformation rules: {e}") from e


# -------- Apply Transformations --------
def apply_transformations(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> pd.DataFrame:
    """
    Apply transformations to the dataset based on transformation rules.
    """
    logger.info("Starting transformation application...")
    try:
        rules = transformation_rules(df, column_stats)

        for col, action in rules.items():
            if col not in df.columns:
                continue

            # 1️⃣ Drop columns
            if action.startswith("drop"):
                logger.info(f"Dropping column {col} ({action})")
                df = df.drop(columns=[col])
                continue

            # 2️⃣ Skip explicitly marked no-transform or no-action
            if action.startswith("no-transform") or action.startswith("no-action"):
                logger.info(f"Skipping transformation for {col} ({action})")
                continue

            # 3️⃣ Log-transform
            if action.startswith("log-transform"):
                if (df[col] > 0).all():
                    df[col] = np.log1p(df[col])
                    logger.info(f"Applied log-transform to {col}")
                else:
                    logger.warning(f"Skipped log-transform for {col} due to non-positive values.")
                continue

            # 4️⃣ Standard scaling
            if action.startswith("standard-scaling"):
                df[col] = StandardScaler().fit_transform(df[[col]])
                logger.info(f"Applied StandardScaler to {col}")
                continue

            # 5️⃣ Min-Max scaling
            if action.startswith("min-max-scaling"):
                df[col] = MinMaxScaler().fit_transform(df[[col]])
                logger.info(f"Applied MinMaxScaler to {col}")
                continue

            # 6️⃣ Convert boolean to int
            if action.startswith("convert to int"):
                df[col] = df[col].astype(int)
                logger.info(f"Converted {col} to int")
                continue

            # 7️⃣ Extract datetime parts
            if action.startswith("extract date parts"):
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_weekday"] = df[col].dt.weekday
                    df[f"{col}_hour"] = df[col].dt.hour
                    logger.info(f"Extracted date parts from {col}")
                else:
                    logger.warning(f"Column {col} not datetime. Skipped extraction.")
                continue

            # 8️⃣ Encoding (handled elsewhere, so skip here)
            if "encode" in action:
                logger.debug(f"Skipping encoding for {col} ({action}) - handled in encoding step")
                continue

            # 9️⃣ Box-Cox or other future transforms (placeholder)
            if "Box-Cox" in action:
                logger.warning(f"Box-Cox transform suggested for {col} but not implemented.")
                continue

        logger.info("Transformation application completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during transformation application: {e}")
        raise PreprocessingError(f"Failed to apply transformations: {e}") from e
