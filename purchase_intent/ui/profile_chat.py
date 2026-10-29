from dataclasses import replace

import streamlit as st

from purchase_intent.models.prediction_input import PredictionInput
from purchase_intent.services.chat_quick_replies import ChatQuickReplies
from purchase_intent.services.profile_chat_intake import ProfileChatIntake


class ProfileChatPanel:
    SESSION_KEY = "profile_chat_session"
    PENDING_KEY = "profile_chat_submission"
    VERSION_KEY = "profile_chat_version"
    EXAMPLE = "A 30-year-old male buying an Apple smartphone for $999.99 with 5 previous purchases and satisfaction 4/5."

    def __init__(self):
        self.intake = ProfileChatIntake()
        self.quick_replies = ChatQuickReplies()

    def get_session(self, seed=None):
        if self.SESSION_KEY not in st.session_state:
            st.session_state[self.SESSION_KEY] = self.intake.start(seed)
        return st.session_state[self.SESSION_KEY]

    def render(self, seed=None, on_reset=None):
        session = self.get_session(seed)

        with st.expander("Example prompt and supported options"):
            st.write(self.EXAMPLE)
            st.caption("Categories: " + ", ".join(PredictionInput.CATEGORIES))
            st.caption("Brands: " + ", ".join(PredictionInput.BRANDS))
            st.caption("Gender: Female or Male. Ages: 18–69. Prices: $100.38–$2,999.85.")
            st.caption("Previous purchases: 1–19. Recorded satisfaction: 1–5.")

        with st.container(height=340, border=False, key="profile_conversation", autoscroll=True):
            for message in session.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        self.render_quick_replies(session)
        total = len(PredictionInput.FIELD_LABELS)
        st.progress(session.provided_count / total, text=f"{session.provided_count} of {total} details provided")
        composer_key = f"profile_chat_message_{st.session_state.get(self.VERSION_KEY, 0)}"
        st.chat_input(
            "Describe a profile or change a detail...",
            key=composer_key,
            max_chars=2000,
            on_submit=self.submit_message,
            args=(composer_key,),
        )
        st.button("Reset", key="reset_chat", on_click=on_reset, width="stretch")
        return st.session_state.pop(self.PENDING_KEY, None)

    def render_quick_replies(self, session):
        replies = self.quick_replies.for_session(session)
        if not replies:
            return

        with st.container(key="chat_quick_replies", horizontal=True, wrap=True, gap="small"):
            for index, reply in enumerate(replies):
                st.button(
                    reply,
                    key=f"profile_quick_reply_{session.next_field}_{len(session.messages)}_{index}",
                    on_click=self.submit_quick_reply,
                    args=(reply,),
                )

    def submit_quick_reply(self, prompt):
        self.submit_reply(prompt)
        st.session_state[self.VERSION_KEY] = st.session_state.get(self.VERSION_KEY, 0) + 1

    def submit_message(self, composer_key):
        self.submit_reply(st.session_state.get(composer_key))

    def submit_reply(self, prompt):
        if not prompt or not prompt.strip():
            return
        inputs = self.intake.receive(self.get_session(), prompt.strip())
        if inputs is not None:
            st.session_state[self.PENDING_KEY] = inputs

    def finish(self, result):
        self.intake.finish(self.get_session(), result)

    def sync_profile(self, inputs):
        if self.SESSION_KEY in st.session_state:
            self.intake.sync(st.session_state[self.SESSION_KEY], inputs)

    def draft_profile(self):
        if self.SESSION_KEY not in st.session_state:
            return None
        session = st.session_state[self.SESSION_KEY]
        return replace(session.inputs, **dict.fromkeys(session.invalid_fields))

    def has_invalid_fields(self):
        session = st.session_state.get(self.SESSION_KEY)
        return session is not None and bool(session.invalid_fields)

    def reset(self):
        st.session_state.pop(self.SESSION_KEY, None)
        st.session_state.pop(self.PENDING_KEY, None)
        st.session_state[self.VERSION_KEY] = st.session_state.get(self.VERSION_KEY, 0) + 1
