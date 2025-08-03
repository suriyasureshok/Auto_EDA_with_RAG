"""
Base Parser

This module is the base class for all parsers. 
This acts as Factory class to create different parsers based on the input file type.
"""

from abc import ABC, abstractmethod
from pandas import DataFrame
from src.utils.models import DatasetMetaData
from fastapi import UploadFile

class BaseParser(ABC):
    """
    Base class for all parsers. This class acts as a factory to create different parsers.
    """
    @abstractmethod
    def load(self, file: UploadFile) -> tuple[DataFrame, DatasetMetaData]:
        pass