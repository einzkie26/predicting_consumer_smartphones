from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionInput:
    age: int | None
    gender: str | None
    category: str | None
    brand: str | None
    price: float | None
    frequency: int | None

    @property
    def is_complete(self):
        return all(
            value is not None
            for value in (
                self.age,
                self.gender,
                self.category,
                self.brand,
                self.price,
                self.frequency,
            )
        )
