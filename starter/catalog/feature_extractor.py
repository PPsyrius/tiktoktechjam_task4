"""Deterministic catalog normalization and conservative attribute extraction."""
from __future__ import annotations

import copy
import math
import re
from typing import Any, Iterable, Mapping, Optional

from .product import ATTRIBUTE_FIELDS, TEXT_FIELDS, Product


DETAIL_ATTRIBUTE_KEYS = {
    "brand": "brand",
    "brand name": "brand",
    "color": "color",
    "color name": "color",
    "stone color": "color",
    "band color": "color",
    "lens color": "color",
    "material": "material",
    "material type": "material",
    "outer material": "material",
    "inner material": "material",
    "frame material": "material",
    "handle material": "material",
    "shaft material": "material",
    "size": "size",
    "band size": "size",
    "ring size": "size",
    "style": "style",
    "occasion": "use_case",
    "sport type": "use_case",
    "special feature": "feature",
}

COLOR_ALIASES = (
    ("black", "black"), ("white", "white"), ("red", "red"),
    ("blue", "blue"), ("green", "green"), ("yellow", "yellow"),
    ("orange", "orange"), ("purple", "purple"), ("pink", "pink"),
    ("brown", "brown"), ("beige", "beige"), ("grey", "gray"),
    ("gray", "gray"), ("navy", "navy"), ("gold", "gold"),
    ("silver", "silver"),
)
MATERIAL_ALIASES = (
    ("stainless steel", "stainless steel"), ("cotton", "cotton"),
    ("polyester", "polyester"), ("nylon", "nylon"),
    ("spandex", "spandex"), ("leather", "leather"), ("wool", "wool"),
    ("silk", "silk"), ("rayon", "rayon"), ("rubber", "rubber"),
    ("suede", "suede"), ("denim", "denim"), ("linen", "linen"),
    ("fleece", "fleece"), ("mesh", "mesh"), ("acrylic", "acrylic"),
    ("alloy", "alloy"),
)
FEATURE_ALIASES = (
    ("waterproof", "waterproof"), ("water resistant", "water resistant"),
    ("breathable", "breathable"), ("lightweight", "lightweight"),
    ("machine wash", "machine wash"), ("hand wash", "hand wash"),
    ("pockets", "pockets"), ("pocket", "pockets"),
    ("cushioned", "cushioned"), ("adjustable", "adjustable"),
    ("moisture wicking", "moisture wicking"), ("stretchy", "stretch"),
    ("stretch", "stretch"),
)
USE_CASE_ALIASES = tuple((value, value) for value in (
    "running", "hiking", "workout", "gym", "yoga", "walking", "travel",
    "outdoor", "casual", "formal", "wedding", "sports",
))


def value_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def normalize_text(value: object) -> str:
    return " ".join(str(value).casefold().split())


def parse_price(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        cleaned = value.strip().casefold().replace(",", "")
        match = re.fullmatch(r"(?:from\s+)?\$?\s*(\d+(?:\.\d+)?)", cleaned)
        if match is None:
            return None
        value = match.group(1)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _display_text(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _items(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values: Iterable[object]
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = (value,)
    return tuple(item for raw in values if (item := _display_text(raw)))


def _normalized_values(value: object) -> tuple[str, ...]:
    return tuple(sorted({
        normalized for raw in _items(value) if (normalized := normalize_text(raw))
    }))


def _word_text(value: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _lexicon_matches(text: str, aliases: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    padded = f" {_word_text(text)} "
    return tuple(sorted({
        canonical
        for phrase, canonical in aliases
        if f" {_word_text(phrase)} " in padded
    }))


def _canonical_brand(value: object) -> Optional[str]:
    brand = _display_text(value)
    if not brand:
        return None
    return brand.title() if brand.isupper() else brand


class FeatureExtractor:
    def extract(self, record: Mapping[str, object]) -> Product:
        if not isinstance(record, Mapping):
            raise TypeError("catalog record must be a mapping")
        asin = record.get("parent_asin")
        if not isinstance(asin, str) or not asin.strip():
            raise ValueError("parent_asin must be a nonempty string")

        raw_categories = record.get("categories")
        if isinstance(raw_categories, str):
            category_path = tuple(
                part for part in (_display_text(item) for item in raw_categories.split(">"))
                if part
            )
        else:
            category_path = _items(raw_categories)
        category = category_path[-1] if category_path else None

        raw_details = record.get("details")
        if raw_details is None:
            raw_details = {}
        if not isinstance(raw_details, Mapping):
            raise TypeError("details must be a mapping")
        details = {str(key): copy.deepcopy(value) for key, value in raw_details.items()}
        features = _items(record.get("features"))
        descriptions = _items(record.get("description"))
        description = " ".join(descriptions)
        title = _display_text(record.get("title"))
        store = _display_text(record.get("store")) or None

        explicit: dict[str, set[str]] = {field: set() for field in ATTRIBUTE_FIELDS}
        explicit["category"].update(
            normalized for item in category_path if (normalized := normalize_text(item))
        )
        enriched = {field: set(values) for field, values in explicit.items()}

        brand_candidates: list[object] = []
        if record.get("brand") is not None:
            brand_candidates.append(record.get("brand"))
            explicit["brand"].update(_normalized_values(record.get("brand")))
            enriched["brand"].update(_normalized_values(record.get("brand")))
        for key, value in details.items():
            normalized_key = normalize_text(key)
            if normalized_key in ATTRIBUTE_FIELDS and normalized_key != "category":
                explicit[normalized_key].update(_normalized_values(value))
            field = DETAIL_ATTRIBUTE_KEYS.get(normalized_key)
            if field is None:
                continue
            enriched[field].update(_normalized_values(value))
            if field == "brand":
                brand_candidates.extend(_items(value))

        explicit_attributes = record.get("attributes")
        if explicit_attributes is None:
            explicit_attributes = {}
        if not isinstance(explicit_attributes, Mapping):
            raise TypeError("attributes must be a mapping")
        for raw_field, value in explicit_attributes.items():
            field = normalize_text(raw_field)
            if field in ATTRIBUTE_FIELDS:
                explicit[field].update(_normalized_values(value))
                enriched[field].update(_normalized_values(value))

        brand = next(
            (candidate for raw in brand_candidates if (candidate := _canonical_brand(raw))),
            None,
        )
        if brand:
            enriched["brand"].add(normalize_text(brand))

        rule_text = " ".join((title, " ".join(features), description, value_to_text(details)))
        enriched["color"].update(_lexicon_matches(rule_text, COLOR_ALIASES))
        enriched["material"].update(_lexicon_matches(rule_text, MATERIAL_ALIASES))
        enriched["feature"].update(_lexicon_matches(rule_text, FEATURE_ALIASES))
        enriched["use_case"].update(_lexicon_matches(rule_text, USE_CASE_ALIASES))

        attributes = {
            field: tuple(sorted(values))
            for field, values in explicit.items()
            if values
        }
        enriched_attributes = {
            field: tuple(sorted(values))
            for field, values in enriched.items()
            if values
        }
        fields = tuple(value_to_text(record.get(field)) for field in TEXT_FIELDS)
        structured_terms = tuple(dict.fromkeys(filter(None, (
            brand,
            *enriched_attributes.get("color", ()),
            *enriched_attributes.get("material", ()),
            *enriched_attributes.get("size", ()),
            *enriched_attributes.get("style", ()),
            *enriched_attributes.get("feature", ()),
            *enriched_attributes.get("use_case", ()),
        ))))
        search_text = "\n".join(filter(None, (
            title,
            " > ".join(category_path),
            " ".join(features),
            value_to_text(details),
            store,
            description,
            " ".join(structured_terms),
        )))
        rating = record.get("average_rating")
        average_rating = parse_price(rating)
        rating_number = record.get("rating_number")
        try:
            numeric_rating_number = float(rating_number or 0)
            parsed_rating_number = (
                max(0, int(numeric_rating_number))
                if math.isfinite(numeric_rating_number)
                else 0
            )
        except (TypeError, ValueError, OverflowError):
            parsed_rating_number = 0

        return Product(
            parent_asin=asin.strip(),
            title=title,
            category=category,
            category_path=category_path,
            brand=brand,
            price=parse_price(record.get("price")),
            color=enriched_attributes.get("color", ()),
            size=enriched_attributes.get("size", ()),
            material=enriched_attributes.get("material", ()),
            style=enriched_attributes.get("style", ()),
            use_case=enriched_attributes.get("use_case", ()),
            feature=enriched_attributes.get("feature", ()),
            features=features,
            description=description,
            details=details,
            store=store,
            average_rating=average_rating,
            rating_number=parsed_rating_number,
            search_text=search_text,
            fields=fields,
            attributes=attributes,
            raw=copy.deepcopy(dict(record)),
        )
