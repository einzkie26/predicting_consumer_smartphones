import unittest

from purchase_intent.models.prediction_input import PredictionInput
from purchase_intent.services.profile_chat_intake import ProfileChatIntake


class ProfileChatIntakeTests(unittest.TestCase):
    def setUp(self):
        self.intake = ProfileChatIntake()
        self.profile = PredictionInput(30, "Male", "Smartphones", "Apple", 999.99, 5, 4)

    def test_profile_can_be_collected_over_seven_turns(self):
        session = self.intake.start()
        for prompt in ("30", "male", "smartphone", "Apple", "$999.99", "5"):
            self.assertIsNone(self.intake.receive(session, prompt))
        ready = self.intake.receive(session, "4 / 5")
        self.assertEqual(ready, self.profile)
        self.assertIsNone(session.next_field)
        self.assertEqual(session.provided_count, 7)

    def test_all_invalid_details_must_be_fixed_before_predicting(self):
        session = self.intake.start(self.profile)
        self.assertIsNone(self.intake.receive(session, "age 5 and price -1"))
        self.assertEqual(session.provided_count, 5)
        self.assertEqual(session.next_field, "age")
        self.assertIsNone(self.intake.receive(session, "40"))
        self.assertEqual(session.next_field, "price")
        ready = self.intake.receive(session, "500")
        self.assertEqual(ready, PredictionInput(40, "Male", "Smartphones", "Apple", 500, 5, 4))
        self.assertFalse(session.invalid_fields)

    def test_unrelated_messages_do_not_trigger_predictions(self):
        session = self.intake.start(self.profile)
        self.assertIsNone(self.intake.receive(session, "hello"))
        self.assertEqual(session.inputs, self.profile)
        self.assertIn("which profile detail", session.messages[-1]["content"])

    def test_form_updates_sync_with_an_existing_conversation(self):
        session = self.intake.start()
        self.intake.receive(session, "age 5")
        self.intake.sync(session, self.profile)
        self.assertEqual(session.inputs, self.profile)
        self.assertFalse(session.invalid_fields)
        self.assertIn("Dropdown selection", session.messages[-1]["content"])

    def test_failed_predictions_get_a_persistent_chat_response(self):
        session = self.intake.start(self.profile)
        self.intake.finish(session, None)
        self.assertIn("couldn't complete", session.messages[-1]["content"])

    def test_unsupported_age_blocks_prediction_until_corrected(self):
        session = self.intake.start(self.profile)
        self.assertIsNone(self.intake.receive(session, "100 years old"))
        self.assertIn("18–69", session.messages[-1]["content"])
        self.assertEqual(session.invalid_fields, ("age",))
        self.assertIsNone(self.intake.receive(session, "price $800"))
        self.assertEqual(session.next_field, "age")
        ready = self.intake.receive(session, "69")
        self.assertEqual(ready.age, 69)
        self.assertEqual(ready.price, 800)
        self.assertFalse(session.invalid_fields)


if __name__ == "__main__":
    unittest.main()
