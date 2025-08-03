from ydata_profiling import ProfileReport
import pandas as pd
from typing import Tuple
from pathlib import Path

def generate_profile_report(df: pd.DataFrame, output_dir: Path, report_name: str = "report") -> Tuple[str, str]:
    """
    Generates a profile report and saves it to disk.

    Args:
        df (pd.DataFrame): The DataFrame to profile.
        output_dir (Path): Path to save reports.
        report_name (str): Base name for output files.

    Returns:
        Tuple[str, str]: Paths to the HTML and JSON reports.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report = ProfileReport(df, title="AutoEDA Report", minimal=True)

    html_path = output_dir / f"{report_name}.html"
    json_path = output_dir / f"{report_name}.json"

    report.to_file(html_path)
    report.to_file(json_path)

    return str(html_path), str(json_path)