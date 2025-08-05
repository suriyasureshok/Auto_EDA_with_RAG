"""
Feature Importance Calculation

Calculates feature importances using SHAP values for tree-based models or
other fallback methods for non-tree models.
"""

#Import libraries
import pandas as pd
from typing import Dict, Any
import shap

#Import util modules
from src.utils.logging import get_logger
from src.utils.exceptions import FeatureImportanceError

logger = get_logger(__name__)

def compute_feature_importance(model: Any, X: pd.DataFrame) -> Dict[str, float]:
    """
    Compute feature importance for the given model and dataset.

    Args:
        model: Trained model object (tree-based preferred for SHAP).
        X (pd.DataFrame): Feature set.

    Returns:
        Dict[str, float]: Mapping of feature name to importance score.
    """
    try:
        logger.info("Computing feature importances...")
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        mean_abs_shap = shap_values.abs.mean(0).values
        importances = dict(zip(X.columns, mean_abs_shap))
        logger.info("Feature importances computed successfully.")
        return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    except Exception as e:
        logger.error(f"Feature importance computation failed: {e}")
        raise FeatureImportanceError(f"Failed to compute feature importance: {e}")
