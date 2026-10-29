from pathlib import Path

import streamlit as st

from purchase_intent.models.prediction_input import PredictionInput
from purchase_intent.services.model_assets import ModelAssets
from purchase_intent.services.purchase_predictor import PurchasePredictor
from purchase_intent.ui.prediction_charts import PredictionCharts


class PurchaseIntentApp:
    STYLES_PATH = Path(__file__).with_name("styles.css")

    def __init__(self):
        self.predictor = None
        self.charts = None

    def run(self):
        st.set_page_config(page_title="Purchase Intent Prediction", layout="wide")
        st.markdown(
            f"<style>{self.STYLES_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )

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
