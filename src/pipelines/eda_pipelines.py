"""
EDA Pipeline

Orchestrates the ingestion, profiling, and exploratory data analysis process.
"""

from pathlib import Path
from src.parsers.ingestor import IngestionFactory
from src.eda_core.profiler import Profiler
from src.eda_core.eda_engine import EDAEngine
from src.utils.logging import get_logger
from src.utils.exceptions import FileLoadError, SummarizationError, HTMLProfilingError, JSONProfilingError

logger = get_logger(__name__)

class EDAPipeline:
    def __init__(self, artifacts_dir: str = "artifacts", use_llm_summary: bool = False):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.use_llm_summary = use_llm_summary

        self.profiler = Profiler(output_dir=self.artifacts_dir / "profiles")
        self.eda_engine = EDAEngine(output_dir=self.artifacts_dir, use_llm_summary=self.use_llm_summary)

        logger.info(f"Initialized EDAPipeline. Artifacts dir: {self.artifacts_dir}, LLM Summary: {self.use_llm_summary}")

    def run(self, file_path: str, target_column: str = None, task_type: str = None, model=None):
        """
        Execute the full EDA workflow.

        Args:
            file_path (str): Path to dataset file.
            target_column (str): Target variable (optional).
            task_type (str): 'classification' or 'regression' (optional).
            model: Optional trained model for feature importance.
        
        Returns:
            dict: EDA results containing summary, plots, feature importance.
        """
        try:
            # Step 1: Ingest data
            logger.info(f"Ingesting dataset from: {file_path}")
            parser = IngestionFactory.get_parser(file_path)
            df, metadata = parser.load_from_path(file_path)
            logger.info(f"✅ Data loaded: shape={df.shape}, filename={metadata.filename}")

            # Step 2: Generate profile report
            logger.info("Running profiler...")
            profile_json_path = self.profiler.generate_profile(df, report_name=metadata.filename)

            # Step 3: Run EDA Engine
            logger.info("Running EDA Engine...")
            results = self.eda_engine.run(
                df=df,
                profile_json_path=profile_json_path,
                target_column=target_column,
                task_type=task_type,
                model=model
            )

            logger.info("✅ EDA Pipeline completed successfully.")
            return results

        except (FileLoadError, HTMLProfilingError, JSONProfilingError, SummarizationError) as e:
            logger.error(f"EDA Pipeline failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in EDA Pipeline: {e}")
            raise
