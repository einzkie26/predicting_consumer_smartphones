import pandas as pd

try:
    import shap
except ImportError:
    shap = None


class PurchasePredictor:
    MODEL_COLUMNS = (
        "CustomerAge",
        "ProductCategory",
        "ProductPrice",
        "PurchaseFrequency",
        "ProductBrand",
        "CustomerSatisfaction",
        "CustomerGender",
    )

    def __init__(self, assets):
        self.assets = assets

    def prepare_model_input(self, inputs):
        if not inputs.is_complete:
            raise ValueError("Complete every profile field, including the recorded satisfaction rating.")
        if inputs.validation_errors:
            raise ValueError(" ".join(inputs.validation_errors))
        row = {
            "CustomerAge": inputs.age,
            "ProductCategory": inputs.category,
            "ProductPrice": inputs.price,
            "PurchaseFrequency": inputs.frequency,
            "ProductBrand": inputs.brand,
            "CustomerSatisfaction": inputs.satisfaction,
            "CustomerGender": int(inputs.gender == "Male"),
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
