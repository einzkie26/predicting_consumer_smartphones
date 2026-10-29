import pandas as pd

try:
    import shap
except ImportError:
    shap = None


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
