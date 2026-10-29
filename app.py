from dataclasses import dataclass
from pathlib import Path
import traceback

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import shap
except ImportError:
    shap = None


@dataclass(frozen=True)
class PredictionInput:
    age: int | None
    gender: str | None
    category: str | None
    brand: str | None
    price: float | None
    frequency: int | None

    @property
    def is_complete(self):
        return all(
            value is not None
            for value in (
                self.age,
                self.gender,
                self.category,
                self.brand,
                self.price,
                self.frequency,
            )
        )


class ModelAssets:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    @staticmethod
    @st.cache_resource
    def load(model_path="fair_electronics_model.pkl", scaler_path="refined_scaler.pkl"):
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
            st.error(f"❌ **File Not Found:** `{path}`")
            st.info(f"📁 Please ensure the {label.lower()} file exists in the current directory.")
            st.code(f"Current directory: {Path.cwd()}")
            st.write("**Expected file location:**")
            st.code(str(path.resolve()))
            return None

        try:
            asset = joblib.load(path)
            st.success(f"✅ {label} loaded successfully: `{path}`")
            return asset
        except Exception as error:
            st.error(f"❌ **Error Loading File:** `{path}`")
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


class PurchasePredictor:
    SCALER_COLUMNS = (
        "ProductPrice",
        "CustomerAge",
        "CustomerGender",
        "PurchaseFrequency",
        "ProductCategory_Headphones",
        "ProductCategory_Laptops",
        "ProductCategory_Smart Watches",
        "ProductCategory_Smartphones",
        "ProductCategory_Tablets",
        "ProductBrand_Apple",
        "ProductBrand_HP",
        "ProductBrand_Other Brands",
        "ProductBrand_Samsung",
        "ProductBrand_Sony",
        "IsMostSoldBrand",
        "Price Per Frequency",
    )
    MODEL_COLUMNS = (
        "CustomerAge",
        "ProductCategory",
        "ProductPrice",
        "PurchaseFrequency",
        "ProductBrand",
        "CustomerSatisfaction",
    )

    def __init__(self, assets):
        self.assets = assets

    def scale_input(self, inputs):
        row = dict.fromkeys(self.SCALER_COLUMNS, 0)
        row.update(
            ProductPrice=inputs.price,
            CustomerAge=inputs.age,
            CustomerGender=int(inputs.gender == "Male"),
            PurchaseFrequency=inputs.frequency,
            IsMostSoldBrand=int(inputs.brand in ("Apple", "Samsung")),
        )
        row["Price Per Frequency"] = inputs.price / inputs.frequency if inputs.frequency > 0 else 0
        row[f"ProductCategory_{inputs.category}"] = 1
        row[f"ProductBrand_{inputs.brand}"] = 1

        frame = pd.DataFrame([row], columns=self.SCALER_COLUMNS)
        frame[list(self.SCALER_COLUMNS)] = self.assets.scaler.transform(frame)
        return frame

    def prepare_model_input(self, inputs):
        row = {
            "CustomerAge": inputs.age,
            "ProductCategory": inputs.category,
            "ProductPrice": inputs.price,
            "PurchaseFrequency": inputs.frequency,
            "ProductBrand": inputs.brand,
            "CustomerSatisfaction": 3,
        }
        return pd.DataFrame([row], columns=self.MODEL_COLUMNS)

    def predict(self, model_input):
        return self.assets.model.predict_proba(model_input)[0][1]

    def feature_contributions(self, model_input):
        if shap is None:
            return pd.DataFrame(columns=["Feature", "Contribution"])

        explainer = shap.TreeExplainer(self.assets.model)
        shap_values = explainer.shap_values(model_input)
        values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        contributions = pd.DataFrame({
            "Feature": list(model_input.columns),
            "Contribution": values,
        })
        contributions = contributions[contributions["Contribution"] != 0].copy()
        contributions["AbsContribution"] = contributions["Contribution"].abs()
        return contributions.sort_values("AbsContribution", ascending=True).tail(10)


class PredictionCharts:
    def __init__(self):
        try:
            is_dark = st.get_option("theme.base") == "dark"
        except Exception:
            is_dark = False

        self.theme = {
            "line": "rgba(255,255,255,0.25)" if is_dark else "rgba(0,0,0,0.25)",
            "plot_bg": "rgba(0,0,0,0)",
            "gauge_red": "rgba(239,68,68,0.20)" if is_dark else "#ffcccc",
            "gauge_yellow": "rgba(245,158,11,0.20)" if is_dark else "#fff4cc",
            "gauge_green": "rgba(34,197,94,0.20)" if is_dark else "#ccffcc",
            "placeholder_bar": "#98A2B3" if is_dark else "#d0d5dd",
            "positive": "#2ca02c",
            "negative": "#d62728",
            "neutral_bar": "#9CA3AF" if is_dark else "#6B7280",
        }

    def create_gauge(self, probability=None):
        display_value = 0 if probability is None else probability * 100
        bar_color = self.theme["neutral_bar"]
        if probability is not None:
            bar_color = "#22C55E" if probability > 0.5 else "#EF4444"

        figure = go.Figure(go.Indicator(
            mode="gauge+number",
            value=display_value,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Purchase Probability", "font": {"size": 22}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": bar_color},
                "bgcolor": self.theme["plot_bg"],
                "borderwidth": 1,
                "bordercolor": self.theme["line"],
                "steps": [
                    {"range": [0, 40], "color": self.theme["gauge_red"]},
                    {"range": [40, 70], "color": self.theme["gauge_yellow"]},
                    {"range": [70, 100], "color": self.theme["gauge_green"]},
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.75,
                    "value": display_value,
                },
            },
        ))
        figure.update_layout(height=400, margin=dict(l=30, r=30, t=50, b=30))
        return figure

    def create_feature_chart(self, contributions=None):
        if contributions is None or contributions.empty:
            contributions = pd.DataFrame({"Feature": ["No data yet"], "Contribution": [0]})
            colors = self.theme["placeholder_bar"]
        else:
            colors = [
                self.theme["positive"] if value > 0 else self.theme["negative"]
                for value in contributions["Contribution"]
            ]

        figure = go.Figure(go.Bar(
            x=contributions["Contribution"],
            y=contributions["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{value:.4f}" for value in contributions["Contribution"]],
            textposition="auto",
        ))
        figure.update_layout(
            title={"text": "Feature Contribution to This Prediction", "font": {"size": 20}},
            xaxis_title="Contribution",
            yaxis_title=None,
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(zeroline=True, showline=True),
            yaxis=dict(showline=False),
        )
        return figure


class PurchaseIntentApp:
    STYLES = """
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
    }

    [data-testid="InputInstructions"],
    button[title="Clear value"],
    div[data-testid="InputEndAdornment"],
    .stNumberInput div[data-testid="InputEndAdornment"] {
        display: none !important;
    }

    [data-testid="stContainer"] {
        background-color: var(--secondary-background-color);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.2);
    }

    .status-box {
        padding: 12px 16px;
        border-radius: 12px;
        font-weight: 600;
        margin-top: 8px;
        margin-bottom: 8px;
    }

    .status-success {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.35);
    }

    .status-danger {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .status-neutral {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.35);
    }
</style>
"""

    def __init__(self):
        self.predictor = None
        self.charts = None

    def run(self):
        st.set_page_config(page_title="Purchase Intent Prediction", layout="wide")
        st.markdown(self.STYLES, unsafe_allow_html=True)

        assets = ModelAssets.load()
        if assets is None:
            st.stop()

        self.predictor = PurchasePredictor(assets)
        self.charts = PredictionCharts()
        st.title("Purchase Intent Prediction")
        inputs, calculate_clicked = self.render_inputs()

        probability = None
        feature_figure = self.charts.create_feature_chart()
        if calculate_clicked:
            probability, feature_figure = self.calculate_prediction(inputs)

        self.render_results(probability, feature_figure)

    def render_inputs(self):
        with st.container(border=True):
            user_column, product_column, pricing_column, behavior_column = st.columns(4)

            with user_column:
                st.markdown("#### User Info")
                age = st.number_input(
                    "Customer Age", min_value=10, max_value=100, placeholder="e.g. 30", value=None
                )
                gender = st.selectbox(
                    "Customer Gender", ["Female", "Male"], placeholder="Select a gender", index=None
                )

            with product_column:
                st.markdown("#### Product")
                category = st.selectbox(
                    "Category",
                    ["Headphones", "Laptops", "Smart Watches", "Smartphones", "Tablets"],
                    placeholder="Select a category",
                    index=None,
                )
                brand = st.selectbox(
                    "Brand",
                    ["Apple", "HP", "Other Brands", "Samsung", "Sony"],
                    placeholder="Select a brand",
                    index=None,
                )

            with pricing_column:
                st.markdown("#### Pricing")
                price = st.number_input(
                    "Product Price ($)", min_value=0.0, placeholder="$ 00.00", value=None
                )

            with behavior_column:
                st.markdown("#### Behavior")
                frequency = st.number_input(
                    "Purchase Frequency", min_value=1, placeholder="e.g. 5", value=None
                )
                st.write("")
                calculate_clicked = st.button(
                    "Calculate Prediction", type="primary", use_container_width=True
                )

        return PredictionInput(age, gender, category, brand, price, frequency), calculate_clicked

    def calculate_prediction(self, inputs):
        feature_figure = self.charts.create_feature_chart()
        if not inputs.is_complete:
            st.warning("Please complete all fields before calculating the prediction.")
            return None, feature_figure

        try:
            self.predictor.scale_input(inputs)
        except Exception as error:
            st.error(f"❌ Scaler Error: {type(error).__name__}")
            st.error(f"Details: {error}")
            st.stop()

        model_input = self.predictor.prepare_model_input(inputs)
        try:
            probability = self.predictor.predict(model_input)
        except Exception as error:
            self.render_prediction_error(error, model_input)
            return None, feature_figure

        try:
            contributions = self.predictor.feature_contributions(model_input)
            feature_figure = self.charts.create_feature_chart(contributions)
        except Exception:
            feature_figure = self.charts.create_feature_chart()

        return probability, feature_figure

    def render_prediction_error(self, error, model_input):
        st.error(f"❌ Prediction Error: {type(error).__name__}")
        st.error(f"Details: {error}")

        with st.expander("🔍 Troubleshooting Information"):
            st.write("**Input DataFrame Info:**")
            st.write(f"- Shape: {model_input.shape}")
            st.write(f"- Columns: {list(model_input.columns)}")

            for label, asset in (
                ("Model", self.predictor.assets.model),
                ("Scaler", self.predictor.assets.scaler),
            ):
                st.write(f"\n**{label} Info:**")
                if hasattr(asset, "n_features_in_"):
                    st.write(f"- Expected features: {asset.n_features_in_}")
                if hasattr(asset, "feature_names_in_"):
                    st.write(f"- Expected feature names: {list(asset.feature_names_in_)}")

            st.write("\n**Solution:**")
            st.write("Check your input features and model training pipeline for consistency.")

    def render_results(self, probability, feature_figure):
        st.divider()

        if probability is not None:
            st.write(f"### Purchase Probability: {probability:.2%}")
            if probability > 0.5:
                status = "success"
                message = f"🟢 Likely to Buy ({probability:.1%})"
            else:
                status = "danger"
                message = f"🔴 Unlikely to Buy ({probability:.1%})"
        else:
            st.write("### Purchase Probability:")
            status = "neutral"
            message = "⚪ No Prediction Yet"

        st.markdown(
            f'<div class="status-box status-{status}">{message}</div>',
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("### Prediction Analytics")
            gauge_column, feature_column = st.columns(2)

            with gauge_column:
                st.plotly_chart(
                    self.charts.create_gauge(probability),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

            with feature_column:
                st.plotly_chart(
                    feature_figure,
                    use_container_width=True,
                    config={"displayModeBar": False},
                )


if __name__ == "__main__":
    PurchaseIntentApp().run()
