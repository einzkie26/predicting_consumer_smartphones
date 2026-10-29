from dataclasses import dataclass

from purchase_intent.models.prediction_input import PredictionInput


@dataclass(frozen=True)
class PromptParseResult:
    inputs: PredictionInput
    updated_fields: tuple[str, ...] = ()
    invalid_fields: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
