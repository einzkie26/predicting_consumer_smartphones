from pathlib import Path

import streamlit as st

from purchase_intent.models.prediction_input import PredictionInput
from purchase_intent.models.prediction_result import PredictionResult
from purchase_intent.services.model_assets import ModelAssets
from purchase_intent.services.purchase_predictor import PurchasePredictor
from purchase_intent.ui.components import InterfaceComponents
from purchase_intent.ui.prediction_charts import PredictionCharts


class PurchaseIntentApp:
    STYLES_PATH = Path(__file__).with_name("styles.css")
    RESULT_KEY = "purchase_prediction"
    INPUT_KEYS = (
        "customer_age",
        "customer_gender",
        "product_category",
        "product_brand",
        "product_price",
        "purchase_frequency",
    )

    def __init__(self):
        self.predictor = None
        self.charts = PredictionCharts()

    def run(self):
        st.set_page_config(
            page_title="Purchase Intent Prediction",
            layout="wide",
            initial_sidebar_state="collapsed",
        )
        st.markdown(
            f"<style>{self.STYLES_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )

        with st.spinner("Preparing your workspace..."):
            assets = ModelAssets.load()
        if assets is None:
            st.stop()

        self.predictor = PurchasePredictor(assets)
        InterfaceComponents.render_header()

        with st.container(key="workspace"):
            input_column, result_column = st.columns([1, 1.32], gap="medium")

            with input_column:
                inputs, submitted = self.render_inputs()
                if submitted:
                    with st.spinner("Calculating purchase likelihood..."):
                        self.calculate_prediction(inputs)

            with result_column:
                self.render_results(st.session_state.get(self.RESULT_KEY))

        InterfaceComponents.render_footer()

    def render_inputs(self):
        with st.container(border=False, key="customer_profile"):
            InterfaceComponents.render_panel_heading(
                "Customer profile",
                "Add the details for the purchase you want to explore.",
                "THE INPUT",
            )

            with st.form("customer_form", border=False):
                InterfaceComponents.render_form_section("01", "Customer")
                age_column, gender_column = st.columns(2, gap="medium")
                with age_column:
                    age = st.number_input(
                        "Customer age",
                        min_value=10,
                        max_value=100,
                        placeholder="e.g. 30",
                        value=None,
                        key="customer_age",
                    )
                with gender_column:
                    gender = st.selectbox(
                        "Gender",
                        ["Female", "Male"],
                        placeholder="Select gender",
                        index=None,
                        key="customer_gender",
                    )

                InterfaceComponents.render_form_section("02", "Product")
                category_column, brand_column = st.columns(2, gap="medium")
                with category_column:
                    category = st.selectbox(
                        "Category",
                        ["Headphones", "Laptops", "Smart Watches", "Smartphones", "Tablets"],
                        placeholder="Select category",
                        index=None,
                        key="product_category",
                    )
                with brand_column:
                    brand = st.selectbox(
                        "Brand",
                        ["Apple", "HP", "Other Brands", "Samsung", "Sony"],
                        placeholder="Select brand",
                        index=None,
                        key="product_brand",
                    )

                InterfaceComponents.render_form_section("03", "Purchase behavior")
                price_column, frequency_column = st.columns(2, gap="medium")
                with price_column:
                    price = st.number_input(
                        "Product price ($)",
                        min_value=0.0,
                        placeholder="e.g. 999.00",
                        format="%.2f",
                        value=None,
                        key="product_price",
                    )
                with frequency_column:
                    frequency = st.number_input(
                        "Purchase frequency",
                        min_value=1,
                        placeholder="e.g. 5",
                        value=None,
                        key="purchase_frequency",
                        help="Number of previous purchases by this customer.",
                    )

                InterfaceComponents.render(
                    '<p class="form-note">All fields are required. Prices are in USD.</p>'
                )
                submit_column, reset_column = st.columns([2.8, 1], gap="small")
                with submit_column:
                    submitted = st.form_submit_button(
                        "Run prediction",
                        type="primary",
                        icon=":material/arrow_forward:",
                        icon_position="right",
                        width="stretch",
                        key="run_prediction",
                    )
                with reset_column:
                    st.form_submit_button(
                        "Reset",
                        on_click=self.reset_workspace,
                        width="stretch",
                        key="reset_prediction",
                    )

        return PredictionInput(age, gender, category, brand, price, frequency), submitted

    def reset_workspace(self):
        for key in self.INPUT_KEYS:
            st.session_state[key] = None
        st.session_state.pop(self.RESULT_KEY, None)

    def calculate_prediction(self, inputs):
        if not inputs.is_complete:
            st.warning(
                "Complete all six fields before running a prediction.",
                icon=":material/info:",
            )
            return

        model_input = None
        try:
            self.predictor.scale_input(inputs)
            model_input = self.predictor.prepare_model_input(inputs)
            probability = float(self.predictor.predict(model_input))
        except Exception as error:
            self.render_prediction_error(error, model_input)
            return

        try:
            contributions = self.predictor.feature_contributions(model_input)
            if contributions.empty:
                contributions = None
        except Exception:
            contributions = None

        st.session_state[self.RESULT_KEY] = PredictionResult(
            inputs=inputs,
            probability=probability,
            contributions=contributions,
        )

    def render_prediction_error(self, error, model_input):
        st.error("We couldn't complete this prediction. Please review the profile and try again.")
        with st.expander("Error details"):
            st.code(f"{type(error).__name__}: {error}")
            if model_input is not None:
                st.write("Submitted profile")
                st.dataframe(model_input, hide_index=True)

    def render_results(self, result):
        probability = result.probability if result is not None else None

        with st.container(border=False, key="prediction_overview"):
            InterfaceComponents.render_panel_heading(
                "Prediction overview",
                "An estimate of this customer's likelihood to purchase.",
                "THE OUTPUT",
            )
            with st.container(key="score_layout"):
                gauge_column, verdict_column = st.columns(
                    [1, 1.15], gap="medium", vertical_alignment="center"
                )
                with gauge_column:
                    st.plotly_chart(
                        self.charts.create_gauge(probability),
                        width="stretch",
                        theme=None,
                        config={"displayModeBar": False, "responsive": True},
                        key="purchase_gauge",
                        alt="Estimated purchase probability" if result else "Awaiting a purchase prediction",
                    )
                with verdict_column:
                    InterfaceComponents.render_verdict(result)

        with st.container(border=False, key="profile_snapshot"):
            InterfaceComponents.render_panel_heading(
                "Profile snapshot",
                "The details behind your latest prediction.",
            )
            InterfaceComponents.render_profile(result)

        if result is not None and result.contributions is not None:
            with st.container(border=False, key="feature_insights"):
                InterfaceComponents.render_panel_heading(
                    "Feature insights",
                    "How individual inputs influenced this prediction.",
                )
                InterfaceComponents.render_feature_legend()
                st.plotly_chart(
                    self.charts.create_feature_chart(result.contributions),
                    width="stretch",
                    theme=None,
                    config={"displayModeBar": False, "responsive": True},
                    key="feature_contributions",
                    alt="Feature contributions to the purchase prediction",
                )
