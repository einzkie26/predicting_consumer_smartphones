from purchase_intent.models.chat_session import ChatSession
from purchase_intent.services.profile_prompt_parser import ProfilePromptParser


class ProfileChatIntake:
    QUESTIONS = {
        "age": "How old is the customer? The model supports ages 18–69.",
        "gender": "What is the customer's gender? Choose Female or Male.",
        "category": "Which category: Headphones, Laptops, Smart Watches, Smartphones, or Tablets?",
        "brand": "Which brand: Apple, HP, Other Brands, Samsung, or Sony?",
        "price": "What is the product price in USD? Enter $100.38–$2,999.85, for example $999.99.",
        "frequency": "How many previous purchases has the customer made? Enter a whole number from 1 to 19.",
        "satisfaction": "What is the customer's recorded satisfaction rating, from 1 to 5? Use the actual rating; I won't assume one.",
    }

    def __init__(self, parser=None):
        self.parser = parser or ProfilePromptParser()

    def start(self, inputs=None):
        session = ChatSession() if inputs is None else ChatSession(inputs=inputs)
        if session.inputs.is_complete:
            message = (
                "I've loaded your latest profile. Tell me what to change, such as "
                "'change the price to $500', or describe a new profile with all seven details."
            )
        else:
            message = (
                "Tell me the customer's age and gender, product category and brand, price in USD, "
                "number of previous purchases, and recorded satisfaction rating (1–5). "
                "You can send all seven details at once or add them one at a time."
            )
        session.messages.append({"role": "assistant", "content": message})
        return session

    def receive(self, session, prompt):
        session.messages.append({"role": "user", "content": prompt})
        parsed = self.parser.parse(prompt, session.inputs, session.next_field)
        session.inputs = parsed.inputs
        unresolved = [name for name in session.invalid_fields if name not in parsed.updated_fields]
        session.invalid_fields = tuple(dict.fromkeys([*unresolved, *parsed.invalid_fields]))

        if session.next_field is not None:
            question = self.QUESTIONS[session.next_field]
            if parsed.errors:
                reply = " ".join(parsed.errors) + "\n\n" + question
            elif parsed.updated_fields:
                reply = "Got it. " + question
            else:
                reply = "Let's add the profile details. " + question
            session.messages.append({"role": "assistant", "content": reply})
            return None

        if not parsed.updated_fields:
            session.messages.append({
                "role": "assistant",
                "content": "Tell me which profile detail to change, such as 'price $500' or 'frequency 3'.",
            })
            return None

        return session.inputs

    def finish(self, session, result):
        if result is None:
            message = "I couldn't complete that prediction. Please check the profile details and try again."
        else:
            verdict = "likely to purchase" if result.is_likely else "less likely to purchase"
            message = (
                f"Purchase probability: **{result.probability:.2%}** — {verdict}. "
                "Prediction overview and Profile snapshot now show this result. "
                "You can change any detail with another message."
            )
        session.messages.append({"role": "assistant", "content": message})

    def sync(self, session, inputs):
        session.inputs = inputs
        session.invalid_fields = ()
        session.messages.append({
            "role": "assistant",
            "content": "The profile was updated using Dropdown selection. Tell me which details to change.",
        })
