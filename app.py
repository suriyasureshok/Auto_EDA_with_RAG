import streamlit as st
import os
from types import SimpleNamespace
from src.parsers.ingestor import IngestionFactory
from src.eda_core.metadata_extractor import MetadataExtractor
from src.eda_core.profiler import Profiler
from src.eda_core.eda_engine import EDAEngine
from src.utils.models import DocType
from src.pipelines.preprocessing_pipelines import PreprocessingPipeline

st.set_page_config(page_title="Auto EDA with RAG", layout="wide")
st.title("📊 Auto EDA & Preprocessing Tool")

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "json", "parquet"]
)

if uploaded_file:
    # 1️⃣ Load data dynamically via ingestion factory
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext == ".csv":
            file_type = DocType.CSV
        elif ext in [".xls", ".xlsx"]:
            file_type = DocType.XLSX
        elif ext == ".json":
            file_type = DocType.JSON
        elif ext == ".parquet":
            file_type = DocType.PARQUET
        else:
            st.error(f"Unsupported file type: {ext}")
            st.stop()

        fake_upload_file = SimpleNamespace(
            filename=uploaded_file.name,
            file=uploaded_file  # Streamlit's file-like object
        )

        parser = IngestionFactory.get_parser(file_type)
        df, metadata = parser.load(fake_upload_file)

        st.success(f"✅ Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Failed to load file: {e}")
        st.stop()

    # 2️⃣ Target column selector
    target_col = st.selectbox("🎯 Select Target Column", [""] + list(df.columns))

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Generate EDA Report"):
            if not target_col:
                st.error("Please select a target column first!")
            else:
                profiler = Profiler(output_dir="src/artifacts/profiles")
                json_path = profiler.generate_profile(df, report_name=uploaded_file.name)

                eda_engine = EDAEngine(output_dir="src/artifacts", use_llm_summary=False)
                results = eda_engine.run(df, json_path, target_column=target_col, task_type="classification")

                st.subheader("📄 Dataset Summary")
                summary = results["summary"]
                if isinstance(summary, dict):
                    st.json(summary)
                else:
                    st.write(summary)

                st.subheader("📊 Generated Plots")
                for plot_list in results["plots"].values():
                    for plot in plot_list:
                        st.image(plot)

    with col2:
        if st.button("Preprocess Data"):
            if not target_col:
                st.error("Please select a target column first!")
            else:
               # 1️⃣ Generate profiling report
                profiler = Profiler(output_dir="src/artifacts/profiles")
                json_path = profiler.generate_profile(df, report_name=uploaded_file.name)

                # 2️⃣ Extract column stats
                metadata_extractor = MetadataExtractor()
                column_stats = metadata_extractor.extract_col_data(json_path)

            # 3️⃣ Run preprocessing
                pipeline = PreprocessingPipeline(artifacts_dir="src/artifacts")
                processed_df = pipeline.run(df, column_stats)
            
            # 4️⃣ Show & download
                st.subheader("⚙️ Preprocessed Data")
                st.dataframe(processed_df.head())
                st.download_button(
                    label="Download Processed Data",
                    data=processed_df.to_csv(index=False).encode(),
                    file_name="processed_dataset.csv",
                    mime="text/csv"
                )
