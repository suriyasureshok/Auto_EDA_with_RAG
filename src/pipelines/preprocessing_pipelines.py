"""
Preprocessing Pipeline

Orchestrates the dataset preprocessing steps:
    - Missing value handling
    - Outlier detection/handling
    - Encoding/scaling
    - Transformations
"""

from pathlib import Path
import pandas as pd
from src.eda_core.preprocessing_rules.missing_values import handle_missing_values
from src.eda_core.preprocessing_rules.outliers import handle_outliers
from src.eda_core.preprocessing_rules.encodings import encode_and_scale
from src.eda_core.preprocessing_rules.transformations import apply_transformations
from src.utils.logging import get_logger
from src.utils.exceptions import PreprocessingError

logger = get_logger(__name__)

class PreprocessingPipeline:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized PreprocessingPipeline. Artifacts dir: {self.artifacts_dir}")

    def run(self, df: pd.DataFrame, column_stats: dict, save_output: bool = True) -> pd.DataFrame:
        """
        Execute the preprocessing pipeline.

        Args:
            df (pd.DataFrame): Input dataset.
            column_stats (dict): Metadata from MetadataExtractor (ColumnSchema).
            save_output (bool): Whether to save the processed dataset.

        Returns:
            pd.DataFrame: Preprocessed dataset.
        """
        try:
            logger.info("🚀 Starting preprocessing pipeline...")

            # Step 1: Missing value handling
            logger.info("Handling missing values...")
            df = handle_missing_values(df, column_stats)

            # Step 2: Outlier detection & handling
            logger.info("Detecting and handling outliers...")
            df = handle_outliers(df, column_stats)

            # Step 3: Encoding & Scaling
            logger.info("Applying encoding and scaling...")
            df = encode_and_scale(df, column_stats)

            # Step 4: Additional transformations
            logger.info("Applying transformations...")
            df = apply_transformations(df, column_stats)

            # Save processed dataset
            if save_output:
                output_path = self.artifacts_dir / "preprocessed_dataset.csv"
                df.to_csv(output_path, index=False)
                logger.info(f"✅ Preprocessed dataset saved to: {output_path}")

            logger.info("✅ Preprocessing pipeline completed successfully.")
            return df

        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {e}")
            raise PreprocessingError(f"Preprocessing failed: {e}")
