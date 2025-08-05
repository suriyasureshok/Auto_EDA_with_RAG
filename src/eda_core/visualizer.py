"""
Visualizer Module

Generates visualizations based on the recommendations provided by
the Visualization Rules Engine. Designed for tabular datasets in
business and ML tasks.
"""

#Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from pathlib import Path

#Import util modules
from src.utils.logging import get_logger
from src.utils.exceptions import VisualizationError
from src.utils.models import ColumnSchema

logger = get_logger(__name__)

class Visualizer:
    """
    Visualizer class to render plots based on visualization rules.
    Uses Matplotlib for static rendering.
    """

    #Initializes directory for storing visuals.
    def __init__(self, output_dir: str = "artifacts/visuals"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizer output directory set to: {self.output_dir}")

    def generate_plots(self, df: pd.DataFrame, rules: Dict[str, List[Dict[str, str]]], column_stats: Dict[str, ColumnSchema]) -> Dict[str, List[str]]:
        """
        Generate and save visualizations as per rules.

        Args:
            df (pd.DataFrame): The dataset.
            rules (Dict[str, List[Dict[str, str]]]): Mapping of col/col_pair to chart types and levels.
            column_stats (Dict[str, ColumnStats]): Column statistics from MetadataExtractor.

        Returns:
            Dict[str, List[str]]: Mapping of column/col_pair to generated plot file paths.

        Raises:
            VisualizationError: If plot generation fails.
        """
        logger.info("Starting visualization generation...")
        generated_paths: Dict[str, List[str]] = {}

        try:
            for feature, chart_list in rules.items():
                generated_paths[feature] = []
                for chart_info in chart_list:
                    chart_type = chart_info["chart"]
                    try:
                        fig_path = self._render_chart(df, feature, chart_type, column_stats)
                        if fig_path:
                            generated_paths[feature].append(str(fig_path))
                    except Exception as e:
                        logger.warning(f"Failed to generate chart '{chart_type}' for '{feature}': {e}")

            logger.info(f"Visualization generation completed. Total features plotted: {len(generated_paths)}")
            return generated_paths

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            raise VisualizationError(f"Visualization generation failed: {e}") from e

    def _render_chart(self, df: pd.DataFrame, feature: str, chart_type: str, column_stats: Dict[str, ColumnSchema]
    ) -> str:
        """
        Render an individual chart.

        Args:
            df (pd.DataFrame): The dataset.
            feature (str): Column name or column pair (e.g., 'col1_vs_target').
            chart_type (str): Type of chart to generate.
            column_stats (Dict[str, ColumnStats]): Column statistics.

        Returns:
            str: Path to saved chart image file.
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        output_file = self.output_dir / f"{feature}_{chart_type}.png"

        # Feature might be a pair
        if "_vs_" in feature:
            col1, col2 = feature.split("_vs_")
            if col1 not in df.columns or col2 not in df.columns:
                logger.warning(f"Skipping plot {chart_type} for {feature}: columns not found")
                plt.close(fig)
                return None
        else:
            col1, col2 = feature, None
            if col1 not in df.columns:
                logger.warning(f"Skipping plot {chart_type} for {feature}: column not found")
                plt.close(fig)
                return None

        # Chart implementations
        if chart_type == "histogram":
            df[col1].dropna().hist(ax=ax, bins=20)
            ax.set_title(f"Histogram of {col1}")
        elif chart_type == "histogram_kde":
            df[col1].dropna().plot(kind="hist", density=True, alpha=0.5, ax=ax)
            df[col1].dropna().plot(kind="kde", ax=ax)
            ax.set_title(f"Histogram + KDE of {col1}")
        elif chart_type == "histogram_log_scale":
            data = np.log1p(df[col1].dropna())
            data.plot(kind="hist", bins=20, ax=ax)
            ax.set_title(f"Log-Scale Histogram of {col1}")
        elif chart_type == "boxplot":
            df.boxplot(column=[col1], ax=ax)
            ax.set_title(f"Boxplot of {col1}")
        elif chart_type == "scatter":
            ax.scatter(df[col1], df[col2], alpha=0.5)
            ax.set_title(f"Scatter plot: {col1} vs {col2}")
        elif chart_type == "barplot":
            df[col1].value_counts().plot(kind="bar", ax=ax)
            ax.set_title(f"Barplot of {col1}")
        elif chart_type == "heatmap":
            corr = df.corr(numeric_only=True)
            im = ax.imshow(corr, cmap="coolwarm", aspect="auto")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
            ax.set_yticks(range(len(corr.columns)), corr.columns)
            ax.set_title("Correlation Heatmap")
        elif chart_type == "feature_importance":
            if feature not in column_stats:  # feature here would be special key like '__feature_importance__'
                logger.warning("Feature importance data not found in rules.")
                plt.close(fig)
                return None
            importance_data = column_stats[feature]  # This will hold the dict from compute_feature_importance
            features = list(importance_data.keys())
            scores = list(importance_data.values())
            ax.barh(features, scores)
            ax.invert_yaxis()
            ax.set_title("Feature Importance")
        else:
            logger.debug(f"Chart type '{chart_type}' not implemented yet.")
            plt.close(fig)
            return None

        plt.tight_layout()
        fig.savefig(output_file, dpi=120)
        plt.close(fig)
        logger.debug(f"Saved {chart_type} for {feature} at {output_file}")
        return output_file
