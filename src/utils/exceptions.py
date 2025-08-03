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