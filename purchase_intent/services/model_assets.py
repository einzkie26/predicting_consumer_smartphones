from pathlib import Path
import traceback

import joblib
import streamlit as st


ARTIFACTS_DIRECTORY = Path(__file__).resolve().parents[2] / "artifacts"


class ModelAssets:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    @staticmethod
    @st.cache_resource
    def load(
        model_path=ARTIFACTS_DIRECTORY / "fair_electronics_model.pkl",
        scaler_path=ARTIFACTS_DIRECTORY / "refined_scaler.pkl",
    ):
        model = ModelAssets._load_file(Path(model_path), "Model")
        if model is None:
            return None

        scaler = ModelAssets._load_file(Path(scaler_path), "Scaler")
        if scaler is None:
            return None

        return ModelAssets(model, scaler)

    @staticmethod
    def _load_file(path, label):
        if not path.exists():
            st.error(f"❌ **File Not Found:** `{path.name}`")
            st.info(f"📁 Please ensure the {label.lower()} file exists in the expected directory.")
            st.code(f"Current directory: {Path.cwd()}")
            st.write("**Expected file location:**")
            st.code(str(path.resolve()))
            return None

        try:
            asset = joblib.load(path)
            st.success(f"✅ {label} loaded successfully: `{path.name}`")
            return asset
        except Exception as error:
            st.error(f"❌ **Error Loading File:** `{path.name}`")
            st.error(f"**Error Type:** `{type(error).__name__}`")
            st.error(f"**Error Message:** {error}")

            with st.expander("🔍 Detailed Error Information"):
                st.write("**Full Error Traceback:**")
                st.code(traceback.format_exc())
                st.write("\n**Possible Causes:**")
                dependency = "scikit-learn/xgboost" if label == "Model" else "scikit-learn"
                st.write(f"- File was created with incompatible {dependency} version")
                st.write("- File is corrupted or incomplete")
                st.write("- Wrong file format")
                if label == "Model":
                    st.write("- Missing dependencies")

                st.write("\n**Solutions:**")
                st.write("1. Check your scikit-learn version: `pip show scikit-learn`")
                if label == "Model":
                    st.write("2. Check your xgboost version: `pip show xgboost`")
                    st.write("3. Recreate the model file with your current environment")
                    st.write("4. Ensure all dependencies are installed")
                else:
                    st.write("2. Recreate the scaler file with your current environment")
                    st.write("3. Ensure scikit-learn is properly installed")

            return None
