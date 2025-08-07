"""
Summarizer Module

Generates a dataset summary using either:
    1. Rule-based approach (no LLM, deterministic)
    2. Mistral LLM for natural language summarization

Designed for business and ML use cases (non-NLP).
"""

from typing import Dict
import pandas as pd

# Import util modules
from src.utils.logging import get_logger
from src.utils.exceptions import SummarizationError, LLMError
from src.utils.models import ColumnSchema
from src.utils.llm_clients import GeminiClient

logger = get_logger(__name__)

# ---------------------------------------------------------------------
# Rule-based Summarization
# ---------------------------------------------------------------------
def generate_summary_rule_based(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> Dict[str, str]:
    """
    Generate a structured textual summary of the dataset (rule-based).

    Args:
        df (pd.DataFrame): The dataset.
        column_stats (Dict[str, ColumnSchema]): Column-level metadata and stats.

    Returns:
        Dict[str, str]: Structured summary.
    """
    logger.info("Starting rule-based dataset summarization...")
    num_rows, num_cols = df.shape

    # Feature type counts
    type_counts = {}
    for col, stats in column_stats.items():
        type_counts[stats.type.value] = type_counts.get(stats.type.value, 0) + 1

    # Missing data report
    missing_report = {
        col: f"{stats.missing_pct:.2f}%" for col, stats in column_stats.items() if stats.missing_pct > 0
    }

    # Potential issues
    potential_issues = []
    for col, stats in column_stats.items():
        if stats.unique == 1:
            potential_issues.append(f"Column '{col}' is constant.")
        if stats.missing_pct > 50:
            potential_issues.append(f"Column '{col}' has over 50% missing values.")

    summary = {
        "dataset_overview": f"The dataset contains {num_rows} rows and {num_cols} columns.",
        "feature_types": f"Feature composition: {type_counts}",
        "missing_data": f"Columns with missing values: {missing_report}" if missing_report else "No missing values detected.",
        "size_info": f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        "potential_issues": potential_issues if potential_issues else ["No major issues detected."]
    }

    logger.info("Rule-based dataset summarization completed.")
    return summary


# ---------------------------------------------------------------------
# LLM-based Summarization
# ---------------------------------------------------------------------
def generate_summary_llm(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema]) -> str:
    """
    Generate a natural language dataset summary using Mistral LLM.

    Args:
        df (pd.DataFrame): The dataset.
        column_stats (Dict[str, ColumnSchema]): Column-level metadata and stats.

    Returns:
        str: Natural language summary.

    Raises:
        LLMError: If the LLM request fails.
    """
    try:
        logger.info("Starting LLM-based dataset summarization...")
        client = GeminiClient(model="gemini-1.5-flash")
        prompt = f"""
        You are a data analysis expert. 
        Dataset statistics: {column_stats}
        
        Summarize the dataset in clear, business-friendly terms, 
        and suggest preprocessing steps for model training.
        """
        response = client.generate(prompt)
        logger.info("LLM-based dataset summarization completed.")
        return response
    except Exception as e:
        logger.error(f"LLM summarization failed: {e}")
        raise LLMError(f"LLM summarization failed: {e}")


# ---------------------------------------------------------------------
# Wrapper function
# ---------------------------------------------------------------------
def generate_summary(df: pd.DataFrame, column_stats: Dict[str, ColumnSchema], use_llm: bool = True):
    """
    Generate a dataset summary using the selected method.

    Args:
        df (pd.DataFrame): The dataset.
        column_stats (Dict[str, ColumnSchema]): Column-level metadata and stats.
        use_llm (bool): If True, uses Mistral LLM. Else, uses rule-based summarization.

    Returns:
        str | Dict[str, str]: Summary text (LLM) or structured dict (rule-based).

    Raises:
        SummarizationError: If summarization fails.
    """
    try:
        if use_llm:
            return generate_summary_llm(df, column_stats)
        return generate_summary_rule_based(df, column_stats)
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise SummarizationError(f"Failed to summarize dataset: {e}")
