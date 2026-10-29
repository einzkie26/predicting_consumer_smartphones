import hashlib
import json
from pathlib import Path

import joblib
import streamlit as st

from purchase_intent.models.prediction_input import PredictionInput


ARTIFACTS_DIRECTORY = Path(__file__).resolve().parents[2] / "artifacts"


class ModelAssets:
    def __init__(self, model, report=None):
        self.model = model
        self.report = report or {}

    @staticmethod
    def load(
        model_path=ARTIFACTS_DIRECTORY / "purchase_intent_model.joblib",
        report_path=ARTIFACTS_DIRECTORY / "model_report.json",
    ):
        model_path, report_path = Path(model_path), Path(report_path)
        try:
            return ModelAssets._load_cached(
                model_path, report_path, model_path.stat().st_mtime_ns, report_path.stat().st_mtime_ns
            )
        except Exception as error:
            st.error("The prediction model could not be loaded.")
            with st.expander("Model loading details"):
                st.code(f"{type(error).__name__}: {error}")
            return None

    @staticmethod
    @st.cache_resource
    def _load_cached(model_path, report_path, model_revision, report_revision):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if checksum != report["model_sha256"]:
            raise ValueError("The model does not match its evaluation report. Retrain both artifacts together.")
        model = joblib.load(model_path)
        if list(model.feature_names_in_) != report["features"] or list(model.classes_) != [0, 1]:
            raise ValueError("The model's feature or class schema does not match its report.")
        if report["gender_encoding"] != {"Female": 0, "Male": 1}:
            raise ValueError("The model's gender encoding does not match the input form.")
        for column, limits in (
            ("CustomerAge", PredictionInput.AGE_RANGE),
            ("ProductPrice", PredictionInput.PRICE_RANGE),
            ("PurchaseFrequency", PredictionInput.FREQUENCY_RANGE),
            ("CustomerSatisfaction", PredictionInput.SATISFACTION_RANGE),
        ):
            supported = report["supported_ranges"][column]
            if limits[0] < supported["min"] or limits[1] > supported["max"]:
                raise ValueError(f"The form permits unsupported values for {column}.")
        for column, options in (
            ("ProductCategory", PredictionInput.CATEGORIES),
            ("ProductBrand", PredictionInput.BRANDS),
        ):
            if set(options) != set(report["supported_categories"][column]):
                raise ValueError(f"The form's choices do not match the model for {column}.")
        return ModelAssets(model, report)
