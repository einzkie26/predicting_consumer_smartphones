from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from purchase_intent.models.prediction_input import PredictionInput
from purchase_intent.services.profile_prompt_parser import ProfilePromptParser
from purchase_intent.services.purchase_predictor import PurchasePredictor
from purchase_intent.training.train_model import ModelTrainer


class PurchasePredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = Path(__file__).resolve().parents[1]
        cls.report = json.loads((cls.project / "artifacts/model_report.json").read_text(encoding="utf-8"))
        cls.model = joblib.load(cls.project / "artifacts/purchase_intent_model.joblib")

    def setUp(self):
        self.profile = PredictionInput(30, "Male", "Smartphones", "Apple", 999.99, 5, 4)
        self.predictor = PurchasePredictor(SimpleNamespace(model=self.model))

    def test_form_and_chat_use_identical_values_and_predictions(self):
        parsed = ProfilePromptParser().parse(
            "A 30-year-old male buying an Apple smartphone for $999.99 with 5 purchases and satisfaction 4/5"
        )
        chat = self.predictor.prepare_model_input(parsed.inputs)
        form = self.predictor.prepare_model_input(self.profile)
        pd.testing.assert_frame_equal(chat, form)
        self.assertEqual(self.predictor.predict(chat), self.predictor.predict(form))

    def test_age_100_and_unsupported_profiles_never_reach_the_model(self):
        model = Mock()
        predictor = PurchasePredictor(SimpleNamespace(model=model))
        for updates in (
            {"age": 100}, {"age": 17}, {"age": 70}, {"age": 30.5},
            {"price": 0}, {"price": 3000}, {"price": float("nan")},
            {"frequency": 20}, {"satisfaction": 6}, {"satisfaction": None},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                frame = predictor.prepare_model_input(replace(self.profile, **updates))
                predictor.predict(frame)
        model.predict_proba.assert_not_called()

    def test_actual_satisfaction_is_passed_without_substitution(self):
        probabilities = []
        for rating in (1, 4):
            frame = self.predictor.prepare_model_input(replace(self.profile, satisfaction=rating))
            self.assertEqual(frame.iloc[0]["CustomerSatisfaction"], rating)
            probabilities.append(self.predictor.predict(frame))
        self.assertNotAlmostEqual(*probabilities, places=3)

    def test_gender_encoding_matches_the_notebook(self):
        for gender, value in (("Female", 0), ("Male", 1)):
            frame = self.predictor.prepare_model_input(replace(self.profile, gender=gender))
            self.assertEqual(frame.iloc[0]["CustomerGender"], value)

    def test_boundary_profiles_are_supported_without_clamping(self):
        for age, price, frequency, satisfaction in ((18, 100.38, 1, 1), (69, 2999.85, 19, 5)):
            profile = replace(self.profile, age=age, price=price, frequency=frequency, satisfaction=satisfaction)
            frame = self.predictor.prepare_model_input(profile)
            self.assertEqual(frame.iloc[0]["CustomerAge"], age)
            probability = self.predictor.predict(frame)
            self.assertTrue(np.isfinite(probability) and 0 <= probability <= 1)

    def test_saved_report_matches_original_data_and_held_out_predictions(self):
        dataset = self.project / "data/consumer_electronics_sales_data.csv"
        self.assertEqual(hashlib.sha256(dataset.read_bytes()).hexdigest(), self.report["dataset"]["sha256"])
        self.assertFalse(self.report["dataset"]["labels_modified"])
        frame = pd.read_csv(dataset)
        development, test = train_test_split(
            frame, test_size=0.2, stratify=frame["PurchaseIntent"], random_state=self.report["split"]["seed"]
        )
        self.assertFalse(set(development["ProductID"]) & set(test["ProductID"]))
        self.assertEqual(sorted(test["ProductID"]), self.report["split"]["test_product_ids"])
        self.assertEqual(len(test), 1800)
        self.assertEqual(len(development), self.report["split"]["final_fit_rows"])
        probabilities = self.model.predict_proba(test[self.report["features"]])[:, 1]
        metrics = ModelTrainer.metrics(test["PurchaseIntent"], probabilities)
        self.assertEqual(metrics, self.report["test_metrics"])
        self.assertFalse(self.report["split"]["test_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
