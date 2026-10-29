import argparse
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost
from xgboost import XGBClassifier


class ModelTrainer:
    FEATURES = (
        "CustomerAge", "ProductCategory", "ProductPrice", "PurchaseFrequency",
        "ProductBrand", "CustomerSatisfaction", "CustomerGender",
    )
    NUMERIC = ("CustomerAge", "ProductPrice", "PurchaseFrequency", "CustomerSatisfaction", "CustomerGender")
    CATEGORICAL = ("ProductCategory", "ProductBrand")
    TARGET = "PurchaseIntent"
    SEED = 42
    SOURCE = "https://drive.google.com/file/d/1piFr6qLWeWRMq5j1WaLTPcYNy8SCb-z5/view"

    def __init__(self, dataset_path, output_directory):
        self.dataset_path = Path(dataset_path)
        self.output_directory = Path(output_directory)

    def load_data(self):
        frame = pd.read_csv(self.dataset_path)
        required = {*self.FEATURES, self.TARGET, "ProductID"}
        if not required.issubset(frame.columns):
            raise ValueError("The dataset is missing required labeled columns.")
        if frame[list(required)].isna().any().any():
            raise ValueError("The dataset contains missing values.")
        if frame["ProductID"].duplicated().any():
            raise ValueError("Product IDs must be unique before splitting the dataset.")
        if set(frame[self.TARGET].unique()) != {0, 1}:
            raise ValueError("PurchaseIntent must contain both binary labels 0 and 1.")
        return frame

    def pipeline(self, classifier):
        preprocessing = ColumnTransformer([
            ("numeric", StandardScaler(), list(self.NUMERIC)),
            ("categorical", OneHotEncoder(handle_unknown="error", sparse_output=False), list(self.CATEGORICAL)),
        ])
        return Pipeline([("preprocessor", preprocessing), ("classifier", classifier)])

    def candidates(self):
        models = {"logistic_regression": LogisticRegression(max_iter=2000, random_state=self.SEED)}
        for depth in (2, 3, 5):
            models[f"xgboost_depth_{depth}"] = XGBClassifier(
                n_estimators=400,
                max_depth=depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=1.0,
                eval_metric="logloss",
                random_state=self.SEED,
                n_jobs=4,
            )
        for leaves in (7, 15):
            models[f"hist_gradient_boosting_leaves_{leaves}"] = HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_leaf_nodes=leaves,
                l2_regularization=1.0,
                random_state=self.SEED,
            )
        return {name: self.pipeline(classifier) for name, classifier in models.items()}

    def calibrated(self, estimator, method):
        return CalibratedClassifierCV(
            estimator=clone(estimator),
            method=method,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.SEED),
            ensemble=False,
            n_jobs=1,
        )

    @staticmethod
    def metrics(labels, probabilities):
        predictions = (probabilities > 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, probabilities)),
            "log_loss": float(log_loss(labels, probabilities)),
            "brier_score": float(brier_score_loss(labels, probabilities)),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        }

    def train(self):
        frame = self.load_data()
        development, test = train_test_split(
            frame, test_size=0.2, stratify=frame[self.TARGET], random_state=self.SEED
        )
        training, validation = train_test_split(
            development, test_size=0.25, stratify=development[self.TARGET], random_state=self.SEED
        )
        candidates = self.candidates()
        leaderboard = []
        for name, estimator in candidates.items():
            estimator.fit(training[list(self.FEATURES)], training[self.TARGET])
            metrics = self.metrics(
                validation[self.TARGET], estimator.predict_proba(validation[list(self.FEATURES)])[:, 1]
            )
            leaderboard.append({"model": name, "calibration": "none", **metrics})
            print(json.dumps(leaderboard[-1]), flush=True)

        best_base = min(leaderboard, key=lambda result: result["log_loss"])["model"]
        for method in ("sigmoid", "isotonic"):
            estimator = self.calibrated(candidates[best_base], method)
            estimator.fit(training[list(self.FEATURES)], training[self.TARGET])
            metrics = self.metrics(
                validation[self.TARGET], estimator.predict_proba(validation[list(self.FEATURES)])[:, 1]
            )
            leaderboard.append({"model": best_base, "calibration": method, **metrics})
            print(json.dumps(leaderboard[-1]), flush=True)

        selected = min(leaderboard, key=lambda result: result["log_loss"])
        base = candidates[selected["model"]]
        final = clone(base) if selected["calibration"] == "none" else self.calibrated(base, selected["calibration"])
        final.fit(development[list(self.FEATURES)], development[self.TARGET])
        probabilities = final.predict_proba(test[list(self.FEATURES)])[:, 1]
        metrics = self.metrics(test[self.TARGET], probabilities)
        observed, predicted = calibration_curve(test[self.TARGET], probabilities, n_bins=10, strategy="quantile")
        dummy = DummyClassifier(strategy="prior").fit(development[list(self.FEATURES)], development[self.TARGET])
        baseline = self.metrics(test[self.TARGET], dummy.predict_proba(test[list(self.FEATURES)])[:, 1])
        self.output_directory.mkdir(parents=True, exist_ok=True)
        model_path = self.output_directory / "purchase_intent_model.joblib"
        joblib.dump(final, model_path)
        restored = joblib.load(model_path)
        np.testing.assert_allclose(restored.predict_proba(test[list(self.FEATURES)])[:, 1], probabilities)
        report = {
            "schema_version": 1,
            "dataset": {
                "file": self.dataset_path.name,
                "source": self.SOURCE,
                "sha256": hashlib.sha256(self.dataset_path.read_bytes()).hexdigest(),
                "rows": len(frame),
                "labels_modified": False,
                "positive_rate": float(frame[self.TARGET].mean()),
            },
            "split": {
                "seed": self.SEED,
                "stratified": True,
                "training_rows": len(training),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "final_fit_rows": len(development),
                "test_used_for_selection": False,
                "test_product_ids": sorted(int(value) for value in test["ProductID"]),
            },
            "features": list(self.FEATURES),
            "excluded_features": ["ProductID"],
            "gender_encoding": {"Female": 0, "Male": 1},
            "supported_ranges": {
                name: {"min": float(frame[name].min()), "max": float(frame[name].max())}
                for name in self.NUMERIC
            },
            "supported_categories": {
                name: sorted(frame[name].unique().tolist()) for name in self.CATEGORICAL
            },
            "selection_metric": "validation log_loss",
            "validation_results": leaderboard,
            "selected_model": selected["model"],
            "calibration": selected["calibration"],
            "calibration_folds": 0 if selected["calibration"] == "none" else 5,
            "decision_threshold": 0.5,
            "test_metrics": metrics,
            "test_majority_baseline": baseline,
            "test_calibration": {"mean_predicted_probability": predicted.tolist(), "observed_purchase_rate": observed.tolist()},
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
            "limitations": [
                "A random holdout from this CSV measures performance on similar rows, not future real-world customers.",
                "Dataset collection and real-world representativeness have not been independently verified.",
                "An individual probability is an estimate, not a guaranteed outcome.",
                "Gender associations in this dataset may not generalize to other populations.",
                "A preliminary six-feature diagnostic was evaluated on this split before correcting the schema to include gender. The holdout was never used for fitting, calibration, threshold or hyperparameter selection; a separate dataset is needed for fully independent confirmation.",
            ],
        }
        report_path = self.output_directory / "model_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"model_path": str(model_path), "report_path": str(report_path), "selected_model": selected["model"], "calibration": selected["calibration"], "test_metrics": metrics}), flush=True)
        return report


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=project / "data" / "consumer_electronics_sales_data.csv")
    parser.add_argument("--output", type=Path, default=project / "artifacts")
    arguments = parser.parse_args()
    ModelTrainer(arguments.data, arguments.output).train()


if __name__ == "__main__":
    main()
