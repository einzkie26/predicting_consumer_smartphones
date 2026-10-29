from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class PurchaseIntentAppTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30
        ).run()
        self.assertFalse(self.app.exception)

    def fill_form(self, satisfaction=4):
        self.app.number_input(key="customer_age").set_value(30)
        self.app.selectbox(key="customer_gender").select("Male")
        self.app.selectbox(key="product_category").select("Smartphones")
        self.app.selectbox(key="product_brand").select("Apple")
        self.app.number_input(key="product_price").set_value(999.99)
        self.app.number_input(key="purchase_frequency").set_value(5)
        self.app.selectbox(key="customer_satisfaction").set_value(satisfaction)
        self.app.button(key="run_prediction").click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)

    def chat(self, prompt):
        self.app.chat_input[0].set_value(prompt).run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.error)

    def test_missing_satisfaction_withholds_prediction_and_clears_an_old_result(self):
        self.fill_form()
        self.assertIn("purchase_prediction", self.app.session_state)
        self.fill_form(satisfaction=None)
        self.assertNotIn("purchase_prediction", self.app.session_state)
        self.assertTrue(any("satisfaction" in warning.value for warning in self.app.warning))

    def test_invalid_chat_age_clears_the_score_and_correction_recalculates(self):
        self.fill_form()
        self.app.get("segmented_control")[0].set_value("Chat prompt").run()
        self.chat("100 years old")
        self.assertNotIn("purchase_prediction", self.app.session_state)
        self.assertTrue(any("withheld" in notice.value for notice in self.app.info))
        self.assertEqual(self.app.session_state["profile_chat_session"].invalid_fields, ("age",))
        self.chat("69")
        self.assertEqual(self.app.session_state["purchase_prediction"].inputs.age, 69)
        self.assertFalse(self.app.info)

    def test_both_modes_show_identical_predictions_and_all_seven_profile_fields(self):
        self.fill_form()
        form = self.app.session_state["purchase_prediction"]
        self.app.button(key="reset_prediction").click().run()
        self.app.get("segmented_control")[0].set_value("Chat prompt").run()
        self.chat(
            "Age 30, male, Apple smartphone, price $999.99, frequency 5, satisfaction 4/5"
        )
        chat = self.app.session_state["purchase_prediction"]
        self.assertEqual(chat.inputs, form.inputs)
        self.assertEqual(chat.probability, form.probability)
        self.assertTrue(any("Customer satisfaction" in item.value for item in self.app.markdown))
        self.assertTrue(any("94.0%" in item.value and "1,800" in item.value for item in self.app.caption))
        self.app.get("segmented_control")[0].set_value("Dropdown selection").run()
        self.assertEqual(self.app.selectbox(key="customer_satisfaction").value, 4)

    def test_seven_quick_reply_turns_produce_a_complete_prediction(self):
        self.app.get("segmented_control")[0].set_value("Chat prompt").run()
        for _ in range(7):
            replies = [button for button in self.app.button if button.key.startswith("profile_quick_reply_")]
            self.assertTrue(replies)
            replies[0].click().run()
            self.assertFalse(self.app.exception)
        result = self.app.session_state["purchase_prediction"]
        self.assertTrue(result.inputs.is_complete)
        self.assertFalse(result.inputs.validation_errors)
        self.assertEqual(self.app.session_state["profile_chat_session"].provided_count, 7)
        self.assertFalse([button for button in self.app.button if button.key.startswith("profile_quick_reply_")])


if __name__ == "__main__":
    unittest.main()
