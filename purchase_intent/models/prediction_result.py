from dataclasses import dataclass

from pandas import DataFrame

from purchase_intent.models.prediction_input import PredictionInput


@dataclass(frozen=True)
class PredictionResult:
    inputs: PredictionInput
    probability: float
    contributions: DataFrame | None = None

    @property
    def is_likely(self):
        return self.probability > 0.5
