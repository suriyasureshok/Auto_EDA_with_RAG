"""
Parquet Parser

This module loads Parquet files and parses them into a usable format.
"""

# Import the abstract base class
from src.parsers.base_parser import BaseParser

#Import required libraries
import pandas as pd
from typing import Tuple
from fastapi import UploadFile
from datetime import datetime, timezone
import io

#Import the util modules
from src.utils.exceptions import FileLoadError, FileEmptyError
from src.utils.logging import get_logger
from src.utils.models import DatasetMetaData, DocType

logger = get_logger(__name__)

class ParquetParser(BaseParser):
    """Parquet Parser class."""

    def load(self, file: UploadFile) -> Tuple[pd.DataFrame, DatasetMetaData]:
        """
        Loads and parses a Parquet file into a DataFrame and generates metadata.

        Args:
            file (UploadFile): The Parquet file to be loaded.

        Returns:
            Tuple[pd.DataFrame, DatasetMetaData]: The parsed DataFrame and dataset metadata.

        Raises:
            FileEmptyError: If the file is empty or has no columns.
            FileLoadError: If the file cannot be read as a Parquet file.
        """
        try:
            content = file.file.read()
            buffer = io.BytesIO(content)

            df = pd.read_parquet(buffer)

            if df.empty or df.shape[1] == 0:
                logger.warning("Parquet file is empty or has no columns.")
                raise FileEmptyError("The Parquet file is empty or invalid.")

            metadata = DatasetMetaData(
                filename = file.filename,
                file_type=DocType.PARQUET,
                upload_time=datetime.now(timezone.utc),
                num_rows=df.shape[0],
                num_columns=df.shape[1],
                column_names=list(df.columns),
            )

            return df, metadata

        except Exception as e:
            logger.error(f"Error loading Parquet file: {e}")
            raise FileLoadError("Unable to load the Parquet file.") from e
