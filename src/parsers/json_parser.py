"""
JSON Parser 

This module loads the JSON files and parses them into a usable format.
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

class JSONParser(BaseParser):
    """JSON Parser class."""
    def load(self, file: UploadFile) -> Tuple[pd.DataFrame, DatasetMetaData]:
        """
        This function overrides the load function from the base parser class.
        It stores the metadata and parses the csv into a pandas DataFrame.

        Args:
            file: The JSON file to be loaded.

        Returns:
            A tuple containing the parsed DataFrame and the metadata of the dataset.

        Raises:
            FileLoadError: If the file is unable to load.
        """
        try:
            content = file.file.read()
            try:
                df = pd.read_json(io.BytesIO(content))
            except ValueError:
                # If the json file is nested
                df = pd.read_json(io.BytesIO(content), lines=True)

            # If there is no columns
            if df.shape[1] == 0:
                logger.warning("JSON file has no columns.")
                raise FileEmptyError("JSON file has no columns.")

            metadata = DatasetMetaData(
                filename = file.filename,
                file_type = DocType.JSON,
                upload_time = datetime.now(timezone.utc),
                num_rows = df.shape[0],
                num_columns = df.shape[1],
                column_names = list(df.columns),
            )
            
            return df, metadata
        
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise FileLoadError('Unable to load the file') from e