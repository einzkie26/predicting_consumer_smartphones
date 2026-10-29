from dataclasses import dataclass, field

from purchase_intent.models.prediction_input import PredictionInput


@dataclass
class ChatSession:
    inputs: PredictionInput = field(default_factory=PredictionInput.empty)
    messages: list[dict[str, str]] = field(default_factory=list)
    invalid_fields: tuple[str, ...] = ()
    quick_reply_context: tuple[str | None, int] | None = None
    quick_replies: tuple[str, ...] = ()

    @property
    def next_field(self):
        fields = self.invalid_fields or self.inputs.missing_fields
        return fields[0] if fields else None

    @property
    def provided_count(self):
        return sum(
            getattr(self.inputs, name) is not None and name not in self.invalid_fields
            for name in PredictionInput.FIELD_LABELS
        )
