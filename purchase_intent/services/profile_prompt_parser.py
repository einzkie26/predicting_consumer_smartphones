from dataclasses import replace
import re

from purchase_intent.models.prediction_input import PredictionInput
from purchase_intent.models.prompt_parse_result import PromptParseResult


class ProfilePromptParser:
    WORD_NUMBERS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    }
    WORD = "(?:" + "|".join(WORD_NUMBERS) + "|hundred|thousand)"
    DIGITS = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+"
    NUMBER = rf"(?:[+-]?(?:{DIGITS})|(?:(?:minus|negative)\s+)?{WORD}(?:[\s-]+(?:and\s+)?{WORD})*)"
    VALUE = rf"(?P<value>{NUMBER})(?!\w|\.\d|,\d)"
    AMOUNT = rf"(?P<value>{NUMBER})(?:\s*(?P<scale>k))?(?!\w|\.\d|,\d)"
    CHOICES = {
        "gender": {
            "Female": r"\b(?:female|woman|girl)\b",
            "Male": r"\b(?:male|man|boy)\b",
        },
        "category": {
            "Headphones": r"\b(?:headphones?|earphones?|earbuds?|headsets?|airpods?)\b",
            "Laptops": r"\b(?:laptops?|notebooks?|macbooks?)\b",
            "Smart Watches": r"\b(?:smart[\s-]*watch(?:es)?|apple\s+watch(?:es)?)\b",
            "Smartphones": r"\b(?:smart[\s-]*phones?|mobile(?:\s+phones?)?|phones?|iphones?)\b",
            "Tablets": r"\b(?:tablets?|ipads?)\b",
        },
        "brand": {
            "Apple": r"\b(?:apple|iphones?|ipads?|macbooks?|airpods?)\b",
            "HP": r"\b(?:hp|hewlett[\s-]+packard)\b",
            "Other Brands": r"\bother(?:\s+brands?)?\b",
            "Samsung": r"\bsamsung\b",
            "Sony": r"\bsony\b",
        },
    }
    FIELD_MARKERS = {
        "age": r"\b(?:age|aged)\b",
        "gender": r"\b(?:gender|sex)\s*(?:(?:is|to)\s+|[:=]\s*)?\S+",
        "category": r"\bcategory\s*(?:(?:is|to)\s+|[:=]\s*)?\S+",
        "brand": r"\bbrand\s*(?:(?:is|to)\s+|[:=]\s*)?\S+",
        "price": r"\b(?:price|cost|costs|priced)\b|\$|\busd\b",
        "frequency": r"\b(?:frequency|previous\s+purchases|past\s+purchases)\b",
        "satisfaction": r"\b(?:satisfaction|rating)\b",
    }

    def parse(self, prompt, current=None, expected_field=None):
        current = current or PredictionInput.empty()
        text = prompt.casefold().replace("–", "-").replace("‑", "-").replace("—", "-")
        text = re.sub(r"([+-])\s*(\$|usd\s*)", r"\2\1", text)
        updates = {}
        invalid_fields = []
        errors = []

        for name in PredictionInput.FIELD_LABELS:
            if name == "satisfaction" and self._has_unsupported_rating_scale(text, expected_field):
                invalid_fields.append(name)
                errors.append("Use the customer's recorded satisfaction on a 1–5 scale. Ratings on other scales are unsupported.")
                continue
            values = self._extract_values(name, text)
            if not values and name == expected_field:
                bare_value = self._bare_value(name, text.strip().rstrip(".!"))
                if bare_value is not None:
                    values = [bare_value]

            values = list(dict.fromkeys(values))
            if len(values) > 1:
                invalid_fields.append(name)
                errors.append(f"Please provide just one {PredictionInput.FIELD_LABELS[name]}.")
                continue
            if not values:
                if re.search(self.FIELD_MARKERS[name], text):
                    invalid_fields.append(name)
                    errors.append(self._missing_value_message(name))
                continue

            value = values[0]
            if name in ("age", "frequency", "satisfaction") and isinstance(value, float) and value.is_integer():
                value = int(value)
            error = PredictionInput.field_error(name, value)
            if error:
                invalid_fields.append(name)
                errors.append(error)
            else:
                updates[name] = value

        return PromptParseResult(
            inputs=replace(current, **updates),
            updated_fields=tuple(updates),
            invalid_fields=tuple(invalid_fields),
            errors=tuple(errors),
        )

    def _has_unsupported_rating_scale(self, text, expected_field):
        prefix = r"\b(?:customer\s+)?(?:satisfaction|rating)\s*(?:(?:rating|score)\s*)?(?:(?:is|of|to)\s*)?[:=]?\s*"
        fraction = rf"{self.NUMBER}\s*(?:/|out\s+of)\s*(?P<denominator>{self.NUMBER})"
        matches = list(re.finditer(prefix + fraction, text))
        if expected_field == "satisfaction" and (match := re.fullmatch(fraction, text.strip().rstrip(".!"))):
            matches.append(match)
        return any(self._to_number(match.group("denominator")) != 5 for match in matches)

    def _extract_values(self, name, text):
        if name in self.CHOICES:
            return [value for value, pattern in self.CHOICES[name].items() if re.search(pattern, text)]

        value = self.VALUE
        amount = self.AMOUNT
        patterns = {
            "age": (
                rf"\b(?:age|aged)\s*(?:(?:is|of|to)\s*)?[:=]?\s*{value}",
                rf"(?<![\w.+-]){value}\s*[-\s]?(?:years?\s*[-\s]?old|y/?o|yrs?\s*old)\b",
                rf"(?<![\w.+-]){value}\s+(?:male|female|man|woman)\b",
            ),
            "price": (
                rf"\b(?:product\s+)?price\s*(?:(?:is|of|at|to)\s*)?[:=]?\s*(?:\$|usd\s*)?{amount}",
                rf"(?:\$|\busd\s*)\s*{amount}",
                rf"(?<![\w.+-]){amount}\s*(?:usd|dollars?|bucks)\b",
                rf"\b(?:costs?|priced(?:\s+at)?)\s*(?:(?:is|of|at|to)\s*)?[:=]?\s*(?:\$|usd\s*)?{amount}",
            ),
            "frequency": (
                rf"\b(?:purchase\s+frequency|frequency|previous\s+purchases|past\s+purchases)\s*(?:(?:is|of|to)\s*)?[:=]?\s*{value}",
                rf"(?<![\w.+-]){value}\s*(?:(?:previous|past|prior|earlier)\s+)?(?:purchases?|orders?|times)\b",
                rf"\b(?:bought|purchased|ordered)\s+{value}\s+times\b",
            ),
            "satisfaction": (
                rf"\b(?:customer\s+)?(?:satisfaction|rating)\s*(?:(?:rating|score)\s*)?(?:(?:is|of|to)\s*)?[:=]?\s*{value}",
                rf"(?<![\w.+-]){value}\s*(?:/\s*5|out\s+of\s+five|out\s+of\s+5|stars?)\b",
            ),
        }
        values = []
        for pattern in patterns[name]:
            for match in re.finditer(pattern, text):
                number = self._to_number(match.group("value"))
                if match.groupdict().get("scale"):
                    number *= 1000
                values.append(number)
        if name == "frequency":
            if re.search(r"\b(?:bought|purchased|ordered)\s+(?:it\s+)?once\b", text):
                values.append(1)
            if re.search(r"\b(?:bought|purchased|ordered)\s+(?:it\s+)?twice\b", text):
                values.append(2)
        return values

    def _bare_value(self, name, text):
        if name in ("age", "price", "frequency", "satisfaction") and re.fullmatch(self.NUMBER, text):
            return self._to_number(text)
        if name == "gender" and text in ("f", "m"):
            return "Female" if text == "f" else "Male"
        return None

    def _to_number(self, value):
        value = value.replace(",", "")
        try:
            return float(value)
        except ValueError:
            sign = -1 if value.startswith(("minus ", "negative ")) else 1
            words = re.split(r"[\s-]+", value)
            total = 0
            current = 0
            for word in words:
                if word in self.WORD_NUMBERS:
                    current += self.WORD_NUMBERS[word]
                elif word == "hundred":
                    current = (current or 1) * 100
                elif word == "thousand":
                    total += (current or 1) * 1000
                    current = 0
            return float(sign * (total + current))

    @staticmethod
    def _missing_value_message(name):
        choices = {
            "gender": PredictionInput.GENDERS,
            "category": PredictionInput.CATEGORIES,
            "brand": PredictionInput.BRANDS,
        }
        label = PredictionInput.FIELD_LABELS[name]
        if name in choices:
            return f"Choose a {label} from: {', '.join(choices[name])}."
        return f"Please enter a valid {label} as a number."
