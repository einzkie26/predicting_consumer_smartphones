import unittest

from purchase_intent.models.prediction_input import PredictionInput
from purchase_intent.services.profile_prompt_parser import ProfilePromptParser


class ProfilePromptParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = ProfilePromptParser()
        self.profile = PredictionInput(30, "Male", "Smartphones", "Apple", 999.99, 5)

    def test_complete_descriptions_match_the_form_profile(self):
        prompts = (
            "A 30-year-old male buying an Apple smartphone for $999.99 with 5 previous purchases.",
            "Age: 30, gender: Male, category: Smartphones, brand: Apple, price: 999.99, frequency: 5",
            "A thirty year old man wants an iPhone for USD 999.99 and has made five purchases.",
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
                        f"Age 40, female, category {category}, brand {brand}, price $200, frequency 3"
                    )
                    self.assertFalse(parsed.errors)
                    self.assertEqual(parsed.inputs.category, category)
                    self.assertEqual(parsed.inputs.brand, brand)
                    self.assertTrue(parsed.inputs.is_complete)

    def test_followup_updates_only_the_requested_details(self):
        parsed = self.parser.parse("Change the price to $500 and frequency to 8", self.profile)
        self.assertEqual(parsed.inputs, PredictionInput(30, "Male", "Smartphones", "Apple", 500, 8))
        self.assertFalse(parsed.errors)

    def test_invalid_numbers_do_not_replace_a_valid_profile(self):
        cases = (
            ("age 9", "age"),
            ("age 101", "age"),
            ("age 30.5", "age"),
            ("-30-year-old", "age"),
            ("price -10", "price"),
            ("-$10", "price"),
            ("price NaN", "price"),
            ("price 1e309", "price"),
            ("frequency 0", "frequency"),
            ("-1 previous purchases", "frequency"),
            ("2.5 previous purchases", "frequency"),
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
        self.assertEqual(len(parsed.inputs.missing_fields), 5)
        self.assertIsNone(parsed.inputs.brand)
        self.assertIsNone(parsed.inputs.category)

    def test_unlabeled_numbers_only_answer_the_expected_question(self):
        self.assertEqual(self.parser.parse("5", expected_field="frequency").inputs.frequency, 5)
        self.assertEqual(self.parser.parse("30", expected_field="age").inputs.age, 30)
        self.assertEqual(self.parser.parse("500", expected_field="price").inputs.price, 500)
        self.assertFalse(self.parser.parse("500").updated_fields)

    def test_price_formats_and_zero_price(self):
        for prompt, expected in (
            ("price $0", 0),
            ("USD 1,299.50", 1299.5),
            ("price $1.5k", 1500),
            ("price $.50", 0.5),
            ("one hundred and fifty dollars", 150),
        ):
            with self.subTest(prompt=prompt):
                parsed = self.parser.parse(prompt)
                self.assertEqual(parsed.inputs.price, expected)
                self.assertFalse(parsed.errors)


if __name__ == "__main__":
    unittest.main()
