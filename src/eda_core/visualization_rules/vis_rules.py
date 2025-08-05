"""
Visualization Rules Engine (Tier 3)

Determines suitable visualization types for tabular business/ML datasets,
including univariate, bivariate, multivariate, and time-series plots.
Supports task-aware recommendations for regression, classification,
and time-series datasets.

Each recommendation is classified as:
- basic      : Quick & essential plots
- diagnostic : Data quality or deeper analysis plots
- advanced   : Computationally heavier or multi-dimensional plots
"""

#Import libraries
from typing import Dict, List
import pandas as pd

#Import util modules
from src.utils.logging import get_logger
from src.utils.exceptions import RuleProcessingError
from src.utils.models import ColumnSchema, ColType, VisualizationLevel

logger = get_logger(__name__)

def visualization_rules(
    df: pd.DataFrame,
    column_stats: Dict[str, ColumnSchema],
    target_column: str = None,
    task_type: str = None  # 'regression', 'classification', 'time-series'
) -> Dict[str, List[Dict[str, str]]]:
    """
    Determine visualization strategies for the dataset.

    Args:
        df (pd.DataFrame): The dataset to analyze.
        column_stats (Dict[str, ColumnStats]): Column-level statistics.
        target_column (str, optional): Target variable name.
        task_type (str, optional): ML task type ('regression', 'classification', 'time-series').

    Returns:
        Dict[str, List[Dict[str, str]]]: Mapping of column names or column pairs to a list
                                         of recommended visualizations with their levels.

    Raises:
        RuleProcessingError: If rule evaluation fails.
    """
    logger.info("Starting Tier 3 visualization rule evaluation...")
    visualizations: Dict[str, List[Dict[str, str]]] = {}

    try:
        # --- Univariate Rules ---
        for col, stats in column_stats.items():
            if col == target_column:
                continue

            visualizations[col] = []

            # For Numeric
            if stats.type == ColType.NUMERIC:
                visualizations[col].append({"chart": "histogram", "level": VisualizationLevel.BASIC})
                visualizations[col].append({"chart": "histogram_kde", "level": VisualizationLevel.DIAGNOSTIC})

                skewness = df[col].dropna().skew()
                if abs(skewness) > 1:
                    visualizations[col].append({"chart": "histogram_log_scale", "level": VisualizationLevel.ADVANCED})

                visualizations[col].append({"chart": "boxplot", "level": VisualizationLevel.BASIC})

            # For Categorical
            elif stats.type == ColType.CATEGORICAL:
                visualizations[col].append({"chart": "barplot", "level": VisualizationLevel.BASIC})
                visualizations[col].append({"chart": "pareto_chart", "level": VisualizationLevel.DIAGNOSTIC})
                visualizations[col].append({"chart": "proportion_chart", "level": VisualizationLevel.ADVANCED})

            # For Datetime
            elif stats.type == ColType.DATETIME:
                visualizations[col].append({"chart": "lineplot", "level": VisualizationLevel.BASIC})
                visualizations[col].append({"chart": "seasonal_decompose", "level": VisualizationLevel.DIAGNOSTIC})
                visualizations[col].append({"chart": "rolling_avg_plot", "level": VisualizationLevel.ADVANCED})

            # For Boolean
            elif stats.type == ColType.BOOLEAN:
                visualizations[col].append({"chart": "barplot", "level": VisualizationLevel.BASIC})
                visualizations[col].append({"chart": "class_balance_plot", "level": VisualizationLevel.DIAGNOSTIC})

        # --- Bivariate Rules ---
        if target_column and target_column in column_stats:
            target_type = column_stats[target_column].type

            for col, stats in column_stats.items():
                if col == target_column:
                    continue

                key = f"{col}_vs_{target_column}"
                visualizations[key] = []

                if task_type == "regression":
                    if stats.type == ColType.NUMERIC:
                        visualizations[key].append({"chart": "scatter", "level": VisualizationLevel.BASIC})
                        visualizations[key].append({"chart": "scatter_trendline", "level": VisualizationLevel.ADVANCED})
                        visualizations[key].append({"chart": "partial_dependence_plot", "level": VisualizationLevel.ADVANCED})
                    elif stats.type == ColType.CATEGORICAL:
                        visualizations[key].append({"chart": "boxplot", "level": VisualizationLevel.BASIC})
                        visualizations[key].append({"chart": "violin_plot", "level": VisualizationLevel.DIAGNOSTIC})

                elif task_type == "classification":
                    if stats.type == ColType.NUMERIC:
                        visualizations[key].append({"chart": "boxplot", "level": VisualizationLevel.BASIC})
                        visualizations[key].append({"chart": "violin_plot", "level": VisualizationLevel.DIAGNOSTIC})
                    elif stats.type == ColType.CATEGORICAL:
                        visualizations[key].append({"chart": "countplot", "level": VisualizationLevel.BASIC})
                        visualizations[key].append({"chart": "stacked_bar", "level": VisualizationLevel.DIAGNOSTIC})

                elif task_type == "time-series":
                    if stats.type == ColType.NUMERIC:
                        visualizations[key].append({"chart": "lineplot", "level": VisualizationLevel.BASIC})
                        visualizations[key].append({"chart": "lag_plot", "level": VisualizationLevel.DIAGNOSTIC})
                        visualizations[key].append({"chart": "acf_pacf", "level": VisualizationLevel.ADVANCED})

        # --- Multivariate Rules ---
        num_cols = [c for c, s in column_stats.items() if s.type == ColType.NUMERIC]
        cat_cols = [c for c, s in column_stats.items() if s.type == ColType.CATEGORICAL]

        if len(num_cols) > 1:
            visualizations["numeric_correlation"] = [{"chart": "heatmap", "level": VisualizationLevel.BASIC}]
            visualizations["clustered_correlation"] = [{"chart": "cluster_heatmap", "level": VisualizationLevel.ADVANCED}]
            if len(df) < 1000:
                visualizations["pairplot"] = [{"chart": "pairplot", "level": VisualizationLevel.ADVANCED}]

        for cat_col in cat_cols:
            for num_col in num_cols:
                visualizations[f"{cat_col}_vs_{num_col}"] = [{"chart": "barplot", "level": VisualizationLevel.BASIC}]

        for i, cat1 in enumerate(cat_cols):
            for cat2 in cat_cols[i+1:]:
                visualizations[f"{cat1}_vs_{cat2}"] = [{"chart": "stacked_bar", "level": VisualizationLevel.DIAGNOSTIC}]

        logger.info("Tier 3 visualization rule evaluation completed successfully.")
        return visualizations

    except Exception as e:
        logger.error(f"Error while applying visualization rules: {e}")
        raise RuleProcessingError(f"Failed to process visualization rules: {e}") from e
