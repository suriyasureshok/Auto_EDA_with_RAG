"""
EDA Engine Orchestrator

Runs the complete EDA process: metadata extraction, summarization,
visualization, and optional feature importance computation.
"""

#Import libraries
from pathlib import Path

from src.eda_core.metadata_extractor import MetadataExtractor
from src.eda_core.visualization_rules.vis_rules import visualization_rules
from src.eda_core.visualizer import Visualizer
from src.eda_core.summarizer import generate_summary
from src.eda_core.feature_importance import compute_feature_importance
from src.utils.logging import get_logger

logger = get_logger(__name__)

class EDAEngine:
    def __init__(self, output_dir: str = "artifacts", use_llm_summary: bool = True):
        logger.info(f"Initializing EDAEngine with output_dir='{output_dir}', use_llm_summary={use_llm_summary}")
        self.output_dir = output_dir
        self.use_llm_summary = use_llm_summary
        self.output_dir = Path(output_dir)
        self.visualizer = Visualizer(output_dir=self.output_dir / "visuals")
        logger.info(f"EDA Engine initialized. LLM Summary: {self.use_llm_summary}")

    def run(self, df, profile_json_path, target_column=None, task_type=None, model=None):
        logger.info("Starting EDA Engine...")
        logger.debug(f"Parameters received - profile_json_path: {profile_json_path}, target_column: {target_column}, task_type: {task_type}, model: {type(model).__name__ if model else None}")
        logger.debug(f"DataFrame shape: {df.shape}")

        metadata_extractor = MetadataExtractor()
        logger.info("Extracting column metadata...")
        column_stats = metadata_extractor.extract_col_data(profile_json_path)
        logger.debug(f"Extracted metadata for {len(column_stats)} columns")

        logger.info("Generating dataset summary...")
        summary = generate_summary(df, column_stats)
        if isinstance(summary, dict):
            logger.debug(f"Generated summary keys: {list(summary.keys())}")
        else:
            logger.debug(f"Generated summary text: {summary[:100]}...")

        logger.info("Determining visualization rules...")
        rules = visualization_rules(df, column_stats, target_column=target_column, task_type=task_type)
        logger.debug(f"Visualization rules generated for {len(rules)} features/feature pairs")

        logger.info("Rendering visualizations...")
        visualizer = Visualizer(output_dir=f"{self.output_dir}/visuals")
        plots = visualizer.generate_plots(df, rules, column_stats)
        logger.debug(f"Generated {sum(len(v) for v in plots.values())} plot files")

        feature_importances = None
        if model is not None and target_column:
            logger.info("Computing feature importance...")
            X = df.drop(columns=[target_column])
            feature_importances = compute_feature_importance(model, X)
            logger.debug(f"Feature importance computed for {len(feature_importances)} features")
            plots["feature_importance"] = ["artifacts/visuals/feature_importance.png"]
            logger.info("Feature importance plot added to visualization outputs")

        logger.info("EDA Engine completed successfully.")
        return {
            "summary": summary,
            "plots": plots,
            "feature_importance": feature_importances
        }
