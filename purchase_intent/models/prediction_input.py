from dataclasses import dataclass
from math import isfinite
from typing import ClassVar


@dataclass(frozen=True)
class PredictionInput:
    AGE_RANGE: ClassVar[tuple[int, int]] = (18, 69)
    PRICE_RANGE: ClassVar[tuple[float, float]] = (100.38, 2999.85)
    FREQUENCY_RANGE: ClassVar[tuple[int, int]] = (1, 19)
    SATISFACTION_RANGE: ClassVar[tuple[int, int]] = (1, 5)
    CATEGORIES: ClassVar[tuple[str, ...]] = (
        "Headphones", "Laptops", "Smart Watches", "Smartphones", "Tablets"
    )
    BRANDS: ClassVar[tuple[str, ...]] = ("Apple", "HP", "Other Brands", "Samsung", "Sony")
    GENDERS: ClassVar[tuple[str, ...]] = ("Female", "Male")
    FIELD_LABELS: ClassVar[dict[str, str]] = {
        "age": "customer age",
        "gender": "gender",
        "category": "product category",
        "brand": "brand",
        "price": "product price",
        "frequency": "purchase frequency",
        "satisfaction": "customer satisfaction",
    }

    age: int | None
    gender: str | None
    category: str | None
    brand: str | None
    price: float | None
    frequency: int | None
    satisfaction: int | None = None

    @classmethod
    def empty(cls):
        return cls(None, None, None, None, None, None, None)

    @classmethod
    def field_error(cls, name, value):
        if value is None:
            return None
        if name == "age" and (type(value) is not int or not cls.AGE_RANGE[0] <= value <= cls.AGE_RANGE[1]):
            return "No prediction is available for this age. The training data covers whole-number ages 18–69."
        if name == "frequency" and (type(value) is not int or not cls.FREQUENCY_RANGE[0] <= value <= cls.FREQUENCY_RANGE[1]):
            return "No prediction is available for this purchase frequency. The training data covers 1–19 previous purchases."
        if name == "satisfaction" and (type(value) is not int or not cls.SATISFACTION_RANGE[0] <= value <= cls.SATISFACTION_RANGE[1]):
            return "Customer satisfaction must be the customer's recorded whole-number rating from 1 to 5."
        if name == "price" and (
            type(value) not in (int, float) or not isfinite(value)
            or not cls.PRICE_RANGE[0] <= value <= cls.PRICE_RANGE[1]
        ):
            return "No prediction is available for this price. Enter an amount from $100.38 to $2,999.85, within the training data's price range."
        options = {"gender": cls.GENDERS, "category": cls.CATEGORIES, "brand": cls.BRANDS}
        if name in options and value not in options[name]:
            return f"Choose a {cls.FIELD_LABELS[name]} from: {', '.join(options[name])}."
        return None

    @property
    def missing_fields(self):
        return tuple(name for name in self.FIELD_LABELS if getattr(self, name) is None)

    @property
    def validation_errors(self):
        return tuple(
            error
            for name in self.FIELD_LABELS
            if (error := self.field_error(name, getattr(self, name))) is not None
        )

    @property
    def is_complete(self):
        return not self.missing_fields
