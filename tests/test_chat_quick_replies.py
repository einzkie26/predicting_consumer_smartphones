from dataclasses import replace
from random import Random
import unittest

from purchase_intent.models.chat_session import ChatSession
from purchase_intent.models.prediction_input import PredictionInput
from purchase_intent.services.chat_quick_replies import ChatQuickReplies
from purchase_intent.services.profile_chat_intake import ProfileChatIntake
from purchase_intent.services.profile_prompt_parser import ProfilePromptParser


class ChatQuickRepliesTests(unittest.TestCase):
    def setUp(self):
        self.suggestions = ChatQuickReplies(Random(42))
        self.profile = PredictionInput(30, "Male", "Smartphones", "Apple", 999.99, 5, 4)

    def test_each_reply_answers_the_current_question(self):
        parser = ProfilePromptParser()
        for name in PredictionInput.FIELD_LABELS:
            with self.subTest(field=name):
                session = ChatSession(inputs=replace(self.profile, **{name: None}))
                replies = self.suggestions.for_session(session)
                self.assertEqual(len(replies), 2 if name == "gender" else 3)
                self.assertEqual(len(replies), len(set(replies)))
                for reply in replies:
                    parsed = parser.parse(reply, session.inputs, session.next_field)
                    self.assertFalse(parsed.errors)
                    self.assertTrue(parsed.inputs.is_complete)
                    self.assertIn(name, parsed.updated_fields)

    def test_suggestions_stay_fixed_until_the_conversation_changes(self):
        session = ChatSession(inputs=replace(self.profile, price=None))
        original = self.suggestions.for_session(session)
        another_render = ChatQuickReplies(Random(99)).for_session(session)
        self.assertEqual(original, another_render)

    def test_reply_advances_the_chat_to_the_next_missing_detail(self):
        intake = ProfileChatIntake()
        session = intake.start(replace(self.profile, price=None, frequency=None))
        reply = self.suggestions.for_session(session)[0]
        self.assertIsNone(intake.receive(session, reply))
        self.assertEqual(session.next_field, "frequency")
        self.assertEqual(session.messages[-2], {"role": "user", "content": reply})
        next_replies = self.suggestions.for_session(session)
        self.assertEqual(len(next_replies), 3)
        self.assertTrue(all("purchase" in value for value in next_replies))

    def test_complete_profiles_do_not_show_reply_buttons(self):
        session = ChatSession(inputs=self.profile)
        self.assertEqual(self.suggestions.for_session(session), ())

    def test_every_available_reply_stays_inside_the_supported_domain(self):
        parser = ProfilePromptParser()
        for name, replies in self.suggestions.OPTIONS.items():
            for reply in replies:
                with self.subTest(field=name, reply=reply):
                    parsed = parser.parse(reply, expected_field=name)
                    self.assertFalse(parsed.errors)
                    self.assertIn(name, parsed.updated_fields)

    def test_invalid_details_receive_valid_correction_suggestions(self):
        intake = ProfileChatIntake()
        session = intake.start(self.profile)
        self.assertIsNone(intake.receive(session, "price -10"))
        replies = self.suggestions.for_session(session)
        self.assertEqual(len(replies), 3)
        ready = intake.receive(session, replies[0])
        self.assertIsNotNone(ready)
        self.assertGreaterEqual(ready.price, 0)
        self.assertFalse(session.invalid_fields)
        self.assertEqual(self.suggestions.for_session(session), ())


if __name__ == "__main__":
    unittest.main()
