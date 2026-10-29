from pathlib import Path

import streamlit as st

from purchase_intent.models.prediction_input import PredictionInput
from purchase_intent.models.prediction_result import PredictionResult
from purchase_intent.services.model_assets import ModelAssets
from purchase_intent.services.purchase_predictor import PurchasePredictor
from purchase_intent.ui.components import InterfaceComponents
from purchase_intent.ui.prediction_charts import PredictionCharts
from purchase_intent.ui.profile_chat import ProfileChatPanel


class PurchaseIntentApp:
    STYLES_PATH = Path(__file__).with_name("styles.css")
    RESULT_KEY = "purchase_prediction"
    MODEL_REVISION_KEY = "purchase_model_revision"
    INPUT_MODE_KEY = "profile_input_mode"
    DROPDOWN_MODE = "Dropdown selection"
    CHAT_MODE = "Chat prompt"
    INPUT_KEYS = (
        "customer_age",
        "customer_gender",
        "product_category",
        "product_brand",
        "product_price",
        "purchase_frequency",
        "customer_satisfaction",
    )

    def __init__(self):
        self.predictor = None
        self.charts = PredictionCharts()
        self.chat = ProfileChatPanel()

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
        revision = assets.report["model_sha256"]
        if st.session_state.get(self.MODEL_REVISION_KEY) != revision:
            st.session_state.pop(self.RESULT_KEY, None)
            self.chat.reset()
            st.session_state[self.MODEL_REVISION_KEY] = revision
        for name, key in zip(PredictionInput.FIELD_LABELS, self.INPUT_KEYS):
            if PredictionInput.field_error(name, st.session_state.get(key)):
                st.session_state[key] = None
        InterfaceComponents.render_header()

        with st.container(key="workspace"):
            input_column, result_column = st.columns([1, 1.32], gap="medium")

            with input_column:
                inputs, submitted = self.render_inputs()
                if submitted:
                    with st.spinner("Calculating purchase likelihood..."):
                        result = self.calculate_prediction(inputs)
                    if st.session_state[self.INPUT_MODE_KEY] == self.CHAT_MODE:
                        self.chat.finish(result)
                        st.rerun()
                    elif result is not None:
                        self.chat.sync_profile(inputs)

            with result_column:
                if st.session_state[self.INPUT_MODE_KEY] == self.CHAT_MODE and self.chat.has_invalid_fields():
                    st.session_state.pop(self.RESULT_KEY, None)
                    st.info("Prediction withheld. Correct the unsupported or invalid profile details in chat.")
                self.render_results(st.session_state.get(self.RESULT_KEY))

        InterfaceComponents.render_footer()

    def render_inputs(self):
        with st.container(border=False, key="customer_profile"):
            InterfaceComponents.render_panel_heading(
                "Customer profile",
                "Enter the profile using dropdowns or a chat prompt.",
                "THE INPUT",
            )
            with st.container(key="input_method"):
                mode = st.segmented_control(
                    "Input method",
                    [self.DROPDOWN_MODE, self.CHAT_MODE],
                    default=self.DROPDOWN_MODE,
                    required=True,
                    key=self.INPUT_MODE_KEY,
                    on_change=self.on_input_mode_change,
                    label_visibility="collapsed",
                    width="stretch",
                )
            if mode == self.CHAT_MODE:
                result = st.session_state.get(self.RESULT_KEY)
                seed = result.inputs if result is not None else PredictionInput(
                    *(st.session_state.get(key) for key in self.INPUT_KEYS)
                )
                inputs = self.chat.render(seed=seed, on_reset=self.reset_workspace)
                return inputs, inputs is not None
            return self.render_dropdown_inputs()

    def render_dropdown_inputs(self):
        with st.form("customer_form", border=False):
            InterfaceComponents.render_form_section("01", "Customer")
            age_column, gender_column = st.columns(2, gap="medium")
            with age_column:
                age = st.number_input(
                    "Customer age",
                    min_value=PredictionInput.AGE_RANGE[0],
                    max_value=PredictionInput.AGE_RANGE[1],
                    placeholder="e.g. 30",
                    value=None,
                    key="customer_age",
                    help="The training data covers ages 18–69. Other ages cannot be estimated reliably.",
                )
            with gender_column:
                gender = st.selectbox(
                    "Gender",
                    PredictionInput.GENDERS,
                    placeholder="Select gender",
                    index=None,
                    key="customer_gender",
                )

            InterfaceComponents.render_form_section("02", "Product")
            category_column, brand_column = st.columns(2, gap="medium")
            with category_column:
                category = st.selectbox(
                    "Category",
                    PredictionInput.CATEGORIES,
                    placeholder="Select category",
                    index=None,
                    key="product_category",
                )
            with brand_column:
                brand = st.selectbox(
                    "Brand",
                    PredictionInput.BRANDS,
                    placeholder="Select brand",
                    index=None,
                    key="product_brand",
                )

            InterfaceComponents.render_form_section("03", "Purchase behavior")
            price_column, frequency_column = st.columns(2, gap="medium")
            with price_column:
                price = st.number_input(
                    "Product price ($)",
                    min_value=PredictionInput.PRICE_RANGE[0],
                    max_value=PredictionInput.PRICE_RANGE[1],
                    placeholder="e.g. 999.00",
                    format="%.2f",
                    value=None,
                    key="product_price",
                )
            with frequency_column:
                frequency = st.number_input(
                    "Purchase frequency",
                    min_value=1,
                    max_value=PredictionInput.FREQUENCY_RANGE[1],
                    placeholder="e.g. 5",
                    value=None,
                    key="purchase_frequency",
                    help="Number of previous purchases by this customer. Supported range: 1–19.",
                )

            satisfaction = st.selectbox(
                "Customer satisfaction (1–5)",
                range(1, 6),
                index=None,
                placeholder="Select the recorded rating",
                key="customer_satisfaction",
                help="Use the customer's actual recorded rating. No rating is assumed when this field is empty.",
            )

            InterfaceComponents.render(
                '<p class="form-note">All fields are required. Ages 18–69. Prices $100.38–$2,999.85.</p>'
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

        return PredictionInput(age, gender, category, brand, price, frequency, satisfaction), submitted

    def on_input_mode_change(self):
        if st.session_state.get(self.INPUT_MODE_KEY) != self.DROPDOWN_MODE:
            return
        profile = self.chat.draft_profile()
        if profile is None:
            result = st.session_state.get(self.RESULT_KEY)
            profile = result.inputs if result is not None else None
        if profile is not None:
            for name, key in zip(PredictionInput.FIELD_LABELS, self.INPUT_KEYS):
                st.session_state[key] = getattr(profile, name)

    def reset_workspace(self):
        for key in self.INPUT_KEYS:
            st.session_state[key] = None
        st.session_state.pop(self.RESULT_KEY, None)
        self.chat.reset()

    def calculate_prediction(self, inputs):
        st.session_state.pop(self.RESULT_KEY, None)
        if not inputs.is_complete:
            st.warning(
                "Complete all seven fields, including the recorded satisfaction rating, before running a prediction.",
                icon=":material/info:",
            )
            return

        if inputs.validation_errors:
            st.warning(" ".join(inputs.validation_errors), icon=":material/info:")
            return

        model_input = None
        try:
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

        result = PredictionResult(
            inputs=inputs,
            probability=probability,
            contributions=contributions,
        )
        st.session_state[self.RESULT_KEY] = result
        return result

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

            if self.predictor is not None:
                report = self.predictor.assets.report
                accuracy = report["test_metrics"]["accuracy"]
                rows = report["split"]["test_rows"]
                st.caption(
                    f"Held-out accuracy: {accuracy:.1%} on {rows:,} records from the supplied dataset. "
                    "Individual outcomes may differ."
                )

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
