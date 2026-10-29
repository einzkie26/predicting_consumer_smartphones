import unittest

from purchase_intent.models.prediction_input import PredictionInput
from purchase_intent.services.profile_prompt_parser import ProfilePromptParser


class ProfilePromptParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = ProfilePromptParser()
        self.profile = PredictionInput(30, "Male", "Smartphones", "Apple", 999.99, 5, 4)

    def test_complete_descriptions_match_the_form_profile(self):
        prompts = (
            "A 30-year-old male buying an Apple smartphone for $999.99 with 5 previous purchases and satisfaction 4/5.",
            "Age: 30, gender: Male, category: Smartphones, brand: Apple, price: 999.99, frequency: 5, rating: 4",
            "A thirty year old man wants an iPhone for USD 999.99, has made five purchases, and satisfaction four.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                parsed = self.parser.parse(prompt)
                self.assertEqual(parsed.inputs, self.profile)
                self.assertFalse(parsed.errors)

    def test_supported_options_can_be_entered_in_chat(self):
        for category in PredictionInput.CATEGORIES:
            for brand in PredictionInput.BRANDS:
                with self.subTest(category=category, brand=brand):
                    parsed = self.parser.parse(
                        f"Age 40, female, category {category}, brand {brand}, price $200, frequency 3, satisfaction 4"
                    )
                    self.assertFalse(parsed.errors)
                    self.assertEqual(parsed.inputs.category, category)
                    self.assertEqual(parsed.inputs.brand, brand)
                    self.assertTrue(parsed.inputs.is_complete)

    def test_followup_updates_only_the_requested_details(self):
        parsed = self.parser.parse("Change the price to $500 and frequency to 8", self.profile)
        self.assertEqual(parsed.inputs, PredictionInput(30, "Male", "Smartphones", "Apple", 500, 8, 4))
        self.assertFalse(parsed.errors)

    def test_invalid_numbers_do_not_replace_a_valid_profile(self):
        cases = (
            ("age 9", "age"),
            ("age 101", "age"),
            ("100 years old", "age"),
            ("age 17", "age"),
            ("age 70", "age"),
            ("age 30.5", "age"),
            ("-30-year-old", "age"),
            ("price -10", "price"),
            ("-$10", "price"),
            ("price NaN", "price"),
            ("price 1e309", "price"),
            ("price $99.99", "price"),
            ("price $3000", "price"),
            ("frequency 0", "frequency"),
            ("-1 previous purchases", "frequency"),
            ("2.5 previous purchases", "frequency"),
            ("frequency 20", "frequency"),
            ("satisfaction 0", "satisfaction"),
            ("rating 6", "satisfaction"),
            ("satisfaction 3.5", "satisfaction"),
            ("satisfaction 4/10", "satisfaction"),
            ("rating four out of ten", "satisfaction"),
        )
        for prompt, field in cases:
            with self.subTest(prompt=prompt):
                parsed = self.parser.parse(prompt, self.profile)
                self.assertIn(field, parsed.invalid_fields)
                self.assertEqual(getattr(parsed.inputs, field), getattr(self.profile, field))

    def test_conflicting_values_require_clarification(self):
        for prompt, field in (("Apple or Samsung", "brand"), ("male or female", "gender")):
            with self.subTest(prompt=prompt):
                parsed = self.parser.parse(prompt, self.profile)
                self.assertIn(field, parsed.invalid_fields)
                self.assertEqual(parsed.inputs, self.profile)

    def test_missing_details_are_not_guessed(self):
        parsed = self.parser.parse("A customer wants a camera for $300")
        self.assertEqual(parsed.inputs.price, 300)
        self.assertEqual(len(parsed.inputs.missing_fields), 6)
        self.assertIsNone(parsed.inputs.brand)
        self.assertIsNone(parsed.inputs.category)

    def test_unlabeled_numbers_only_answer_the_expected_question(self):
        self.assertEqual(self.parser.parse("5", expected_field="frequency").inputs.frequency, 5)
        self.assertEqual(self.parser.parse("30", expected_field="age").inputs.age, 30)
        self.assertEqual(self.parser.parse("500", expected_field="price").inputs.price, 500)
        self.assertEqual(self.parser.parse("4", expected_field="satisfaction").inputs.satisfaction, 4)
        self.assertFalse(self.parser.parse("500").updated_fields)

    def test_supported_price_formats(self):
        for prompt, expected in (
            ("price $100.38", 100.38),
            ("USD 1,299.50", 1299.5),
            ("price $1.5k", 1500),
            ("price $2,999.85", 2999.85),
            ("one hundred and fifty dollars", 150),
        ):
            with self.subTest(prompt=prompt):
                parsed = self.parser.parse(prompt)
                self.assertEqual(parsed.inputs.price, expected)
                self.assertFalse(parsed.errors)

    def test_satisfaction_is_required_and_never_assumed(self):
        parsed = self.parser.parse("Age 30, male, Apple smartphone, price $999.99, frequency 5")
        self.assertEqual(parsed.inputs.missing_fields, ("satisfaction",))
        self.assertIsNone(parsed.inputs.satisfaction)

    def test_satisfaction_quick_replies_use_the_recorded_five_point_scale(self):
        for value in range(1, 6):
            parsed = self.parser.parse(f"{value} / 5", expected_field="satisfaction")
            self.assertEqual(parsed.inputs.satisfaction, value)
            self.assertFalse(parsed.errors)

    def test_followup_rating_on_another_scale_requires_clarification(self):
        parsed = self.parser.parse("4 / 10", self.profile, expected_field="satisfaction")
        self.assertEqual(parsed.invalid_fields, ("satisfaction",))
        self.assertEqual(parsed.inputs.satisfaction, 4)


if __name__ == "__main__":
    unittest.main()
