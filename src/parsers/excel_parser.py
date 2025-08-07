"""
Excel Parser 

This module loads the Excel files and parses them into a usable format.
"""

#Import the abstract class
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

class XLSXParser(BaseParser):
    """XLSX Parser class."""
    def load(self, file) -> Tuple[pd.DataFrame, DatasetMetaData]:
        """
        Load an Excel file into a DataFrame with metadata.

        Args:
            file: Uploaded file object (FastAPI UploadFile or Streamlit UploadedFile).

        Returns:
            Tuple of DataFrame and DatasetMetaData.
        """
        try:
            # Read file content
            content = file.file.read() if hasattr(file, "file") else file.read()
            df = pd.read_excel(io.BytesIO(content), engine='openpyxl')

            if df.empty:
                logger.warning('Excel File is Empty')
                raise FileEmptyError("The Excel file is empty")

            metadata = DatasetMetaData(
                filename=file.filename,
                file_type=DocType.XLSX,
                upload_time=datetime.now(timezone.utc),
                num_rows=df.shape[0],
                num_columns=df.shape[1],
                column_names=list(df.columns),
            )
            return df, metadata

        except Exception as e:
            logger.error(f"Error loading Excel File: {e}")
            raise FileLoadError('Unable to load the Excel File') from e
