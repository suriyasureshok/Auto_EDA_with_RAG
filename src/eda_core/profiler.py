"""
Profiler Module

Wraps ydata-profiling to generate dataset reports (HTML + JSON).
Used by the EDA Engine to extract metadata and generate visual reports.
"""

from pathlib import Path
import pandas as pd
from typing import Tuple

# Local imports
from src.utils.logging import get_logger
from src.utils.exceptions import HTMLProfilingError, JSONProfilingError
from ydata_profiling import ProfileReport

logger = get_logger(__name__)

class Profiler:
    """
    Profiler class to generate and persist profiling reports.
    """

    def __init__(self, output_dir: str = "artifacts/profiles"):
        """
        Args:
            output_dir (str): Directory to store generated reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Profiler output directory set to: {self.output_dir}")

    def generate_profile(self, df: pd.DataFrame, report_name: str = "report") -> str:
        """
        Generate profiling reports (HTML + JSON) and save to disk.

        Args:
            df (pd.DataFrame): Dataset to be profiled.
            report_name (str): Base name for output files.

        Returns:
            str: Path to the generated JSON report.

        Raises:
            HTMLProfilingError, JSONProfilingError: On profiling failure.
        """
        try:
            logger.info(f"Generating profile report for '{report_name}'...")

            profile = ProfileReport(df, title="AutoEDA Report", minimal=True)

            html_path = self.output_dir / f"{report_name}.html"
            json_path = self.output_dir / f"{report_name}.json"

            try:
                profile.to_file(html_path)
                logger.debug(f"HTML report saved at {html_path}")
            except Exception as e:
                logger.error(f"Failed to generate HTML report: {e}")
                raise HTMLProfilingError("Failed to generate HTML report") from e

            try:
                json_data = profile.to_json()
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(json_data)
                logger.debug(f"JSON report saved at {json_path}")
            except Exception as e:
                logger.error(f"Failed to generate JSON report: {e}")
                raise JSONProfilingError("Failed to generate JSON report") from e

            logger.info("Profile report generation completed successfully.")
            return str(json_path)

        except Exception as e:
            logger.error(f"Failed to generate profile report: {e}")
            raise
