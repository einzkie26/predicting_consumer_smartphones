from random import Random

from purchase_intent.models.prediction_input import PredictionInput


class ChatQuickReplies:
    OPTIONS = {
        "age": ("18", "21", "25", "30", "35", "40", "45", "50", "60", "65"),
        "gender": PredictionInput.GENDERS,
        "category": PredictionInput.CATEGORIES,
        "brand": PredictionInput.BRANDS,
        "price": (
            "$149.99", "$199.99", "$299.99", "$499.99",
            "$599.99", "$799.99", "$999.99", "$1,299.99", "$1,499.99",
        ),
        "frequency": (
            "1 purchase", "2 purchases", "3 purchases", "5 purchases",
            "8 purchases", "10 purchases", "15 purchases", "19 purchases",
        ),
        "satisfaction": ("1 / 5", "2 / 5", "3 / 5", "4 / 5", "5 / 5"),
    }

    def __init__(self, random_source=None):
        self.random = random_source if random_source is not None else Random()

    def for_session(self, session):
        context = (session.next_field, len(session.messages))
        if getattr(session, "quick_reply_context", None) != context:
            options = self.OPTIONS.get(session.next_field, ())
            session.quick_replies = tuple(self.random.sample(options, min(3, len(options))))
            session.quick_reply_context = context
        return session.quick_replies
