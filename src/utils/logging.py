"""
Custom logging module.

This module provides a custom logging.
"""
import logging
import os

def get_logger(name: str, log_file: str = "logs.csv") -> logging.Logger:
    """
    Configures and returns a logger instance that writes to a CSV file.

    Args:
        name (str): Name of the logger, typically __name__ of the module.
        log_file (str): Path to the CSV log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # CSV Format: timestamp, level, name, message
        csv_formatter = logging.Formatter('%(asctime)s,%(levelname)s,%(name)s,"%(message)s"')

        # StreamHandler for console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(csv_formatter)
        logger.addHandler(stream_handler)

        # FileHandler for CSV file
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(csv_formatter)
        logger.addHandler(file_handler)

    return logger
