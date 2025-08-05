from src.parsers.ingestor import IngestionFactory
from src.eda_core.profiler import Profiler
from src.eda_core.eda_engine import EDAEngine
import pandas as pd

# --- 1. Load dataset ---
dataset_path = "src/unit_test/iris.csv"  # put your dataset here
file_type = "csv"

parser = IngestionFactory.get_parser(file_type)
from fastapi import UploadFile
from pathlib import Path
file_path = Path("src/unit_test/iris.csv")
with open(file_path, "rb") as f:
    upload_file = UploadFile(filename=file_path.name, file=f)
    df, metadata = parser.load(upload_file)

print(df.head())
print(metadata)

print("✅ Data loaded:", df.shape, metadata)

# --- 2. Profile dataset ---
profiler = Profiler(output_dir="artifacts/profiles")
profile_path = profiler.generate_profile(df, metadata.filename)
print("✅ Profile saved at:", profile_path)

# --- 3. Run EDA ---
engine = EDAEngine(output_dir="artifacts", use_llm_summary=True)
results = engine.run(df, profile_path, target_column="species", task_type="classification")

print("\n--- SUMMARY ---")
print(results["summary"])
print("\n--- PLOTS ---")
print(results["plots"])
print("\n--- FEATURE IMPORTANCE ---")
print(results["feature_importance"])
