import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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
