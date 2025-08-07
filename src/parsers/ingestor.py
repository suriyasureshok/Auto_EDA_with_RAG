"""
Ingestion Factory for the data parsers

This module is like the hub for all the data parsers
and chooses the appropriate parser based on the data type.
"""

from fastapi import UploadFile

# Import Parsers and utils
from src.utils.models import DocType
from src.parsers.csv_parser import CSVParser
from src.parsers.excel_parser import XLSXParser
from src.parsers.json_parser import JSONParser
from src.parsers.parquet_parser import ParquetParser

class IngestionFactory:
    """This class is the factory for all ingestion parsers."""
    @staticmethod
    def get_parser(file_type: DocType):
        """
        This method returns the appropriate parser based on the file type.

        Args:
            file_type: The type of document being uploaded.

        Returns:
            Parser : The parser for the given file type.
        """
        match file_type:
            case DocType.CSV:
                return CSVParser()
            case DocType.XLSX:
                return XLSXParser()
            case DocType.JSON:
                return JSONParser()
            case DocType.PARQUET:
                return ParquetParser()
            case _:
                raise NotImplementedError(f"No parser available for {file_type}")
            
"""
Example usage:

from ingestion.factory import IngestionFactory
from ingestion.models import FileType

def handle_upload(file: UploadFile):
    ext = file.filename.split(".")[-1].lower()
    file_type = FileType(ext)
    parser = IngestionFactory.get_parser(file_type)
    df, metadata = parser.load(file)
    return df, metadata
"""