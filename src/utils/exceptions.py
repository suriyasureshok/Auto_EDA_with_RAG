"""
Custom Exceptions

This module defines the Exceptions that should be raised for each Exception.
"""

class FileError(Exception):
    """Base class for File loader errors."""
    pass

class FileLoadError(FileError):
    """Raised when the File is unable to load."""
    pass

class FileEmptyError(FileError):
    """Raised when the File is Empty."""
    pass

class DataProfilerError(Exception):
    """Base class for Data Profiler errors."""
    pass

class HTMLProfilingError(DataProfilerError):
    """Raised when HTML report is not generated"""
    pass
class JSONProfilingError(DataProfilerError):
    """Raised when JSON report is not generated"""
    pass

class ColumnSchemaError(Exception):
    """Base class for Column schema errors."""
    pass

class RuleProcessingError(Exception):
    """Base Class for errors in defining rules for preprocessing."""
    pass

class VisualizationError(Exception):
    """Base class for visuals generation errors."""
    pass

class SummarizationError(Exception):
    """Base class for summarization errors."""
    pass

class FeatureImportanceError(Exception):
    """Base class for feature importance calculation errors."""
    pass

class LLMError(Exception):
    """Base class for LLM Errors errors."""
    pass

class FileNotFoundError(Exception):
    pass

class PreprocessingError(Exception):
    pass