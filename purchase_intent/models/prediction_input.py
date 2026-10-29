from dataclasses import dataclass
from math import isfinite
from typing import ClassVar


@dataclass(frozen=True)
class PredictionInput:
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
    }

    age: int | None
    gender: str | None
    category: str | None
    brand: str | None
    price: float | None
    frequency: int | None

    @classmethod
    def empty(cls):
        return cls(None, None, None, None, None, None)

    @classmethod
    def field_error(cls, name, value):
        if value is None:
            return None
        if name == "age" and (not isinstance(value, int) or not 10 <= value <= 100):
            return "Customer age must be a whole number from 10 to 100."
        if name == "frequency" and (not isinstance(value, int) or value < 1):
            return "Purchase frequency must be a whole number of at least 1."
        if name == "price" and (
            not isinstance(value, (int, float)) or not isfinite(value) or value < 0
        ):
            return "Product price must be a non-negative amount in USD."
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
