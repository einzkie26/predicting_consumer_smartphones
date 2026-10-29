import pandas as pd
import plotly.graph_objects as go


class PredictionCharts:
    FEATURE_NAMES = {
        "CustomerAge": "Customer age",
        "ProductCategory": "Product category",
        "ProductPrice": "Product price",
        "PurchaseFrequency": "Purchase frequency",
        "ProductBrand": "Product brand",
        "CustomerSatisfaction": "Customer satisfaction",
    }

    def __init__(self):
        self.theme = {
            "text": "#E7EDFA",
            "muted": "#A3B1CB",
            "accent": "#A6BDFF",
            "track": "#2C3A55",
            "positive": "#9FB8FF",
            "negative": "#8795AF",
            "line": "#495975",
            "grid": "rgba(163,177,203,0.12)",
        }

    def create_gauge(self, probability=None):
        values = [100] if probability is None else [probability * 100, (1 - probability) * 100]
        colors = [self.theme["track"]] if probability is None else [
            self.theme["accent"], self.theme["track"]
        ]
        figure = go.Figure(go.Pie(
            values=values,
            labels=["Awaiting prediction"] if probability is None else ["Purchase", "No purchase"],
            hole=0.82,
            sort=False,
            direction="clockwise",
            rotation=90,
            marker={"colors": colors, "line": {"width": 0}},
            textinfo="none",
            hoverinfo="skip",
            showlegend=False,
        ))
        figure.add_annotation(
            text="—" if probability is None else f"{probability:.1%}",
            x=0.5,
            y=0.55,
            showarrow=False,
            font={
                "family": "Inter, Segoe UI, sans-serif",
                "size": 35,
                "color": self.theme["muted"] if probability is None else self.theme["text"],
            },
        )
        figure.add_annotation(
            text="PURCHASE PROBABILITY",
            x=0.5,
            y=0.37,
            showarrow=False,
            font={"family": "Inter, Segoe UI, sans-serif", "size": 9, "color": self.theme["muted"]},
        )
        figure.update_layout(
            template="plotly_dark",
            height=238,
            margin=dict(l=12, r=12, t=16, b=16),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, Segoe UI, sans-serif", "color": self.theme["text"]},
            dragmode=False,
        )
        return figure

    def create_feature_chart(self, contributions=None):
        if contributions is None or contributions.empty:
            contributions = pd.DataFrame(columns=["Feature", "Contribution"])

        labels = contributions["Feature"].map(
            lambda name: self.FEATURE_NAMES.get(name, name.replace("_", " "))
        )
        figure = go.Figure(go.Bar(
            x=contributions["Contribution"],
            y=labels,
            orientation="h",
            marker_color=[
                self.theme["positive"] if value > 0 else self.theme["negative"]
                for value in contributions["Contribution"]
            ],
            text=[f"{value:+.3f}" for value in contributions["Contribution"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>Contribution: %{x:+.4f}<extra></extra>",
        ))
        figure.update_layout(
            template="plotly_dark",
            height=max(245, len(contributions) * 34 + 55),
            margin=dict(l=0, r=45, t=12, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, Segoe UI, sans-serif", "color": self.theme["text"], "size": 11},
            xaxis={
                "title": None,
                "zeroline": True,
                "zerolinecolor": self.theme["line"],
                "gridcolor": self.theme["grid"],
                "tickfont": {"color": self.theme["muted"], "size": 10},
                "fixedrange": True,
            },
            yaxis={
                "title": None,
                "showgrid": False,
                "automargin": True,
                "tickfont": {"color": self.theme["muted"], "size": 11},
                "fixedrange": True,
            },
            bargap=0.4,
            showlegend=False,
            dragmode=False,
        )
        return figure
