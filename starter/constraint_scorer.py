from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ATTRIBUTE_SLOTS = frozenset({
    "category",
    "brand",
    "material",
    "color",
    "size",
    "style",
    "use_case",
    "feature",
})
NUMERIC_SLOTS = frozenset({"price_min", "price_max", "rating_min"})

BUYING_WEIGHTS = {
    "hard_pass": 3.0,
    "hard_fail": -8.0,
    "soft_match": 1.25,
    "exclude_fail": -6.0,
}
BROWSING_WEIGHTS = {
    "hard_pass": 1.5,
    "hard_fail": -2.5,
    "soft_match": 2.0,
    "exclude_fail": -3.0,
}
UNKNOWN_WEIGHTS = {
    "hard_pass": 2.0,
    "hard_fail": -5.0,
    "soft_match": 1.5,
    "exclude_fail": -4.0,
}


@dataclass(frozen=True)
class ConstraintScore:
    score: float
    reason_codes: list[str]
    matched_preferences: list[str]
    hard_failures: list[str]


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _product_text(product: dict[str, Any]) -> str:
    parts = [
        _flatten_text(product.get("title")),
        _flatten_text(product.get("categories")),
        _flatten_text(product.get("features")),
        _flatten_text(product.get("details")),
        _flatten_text(product.get("description")),
        _flatten_text(product.get("store")),
    ]
    return " ".join(part for part in parts if part).lower()


def _normalize_constraints(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []
    return [str(value).strip().lower()] if str(value).strip() else []


def _normalize_constraint_map(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for slot, raw_value in value.items():
        key = str(slot).strip().lower()
        if not key:
            continue
        values = _normalize_constraints(raw_value)
        if values:
            normalized[key] = values
    return normalized


def _get_hard_constraints(search_context: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    hard = search_context.get("hard_constraints")
    if isinstance(hard, dict):
        return _normalize_constraint_map(hard), []
    if hard:
        return {}, _normalize_constraints(hard)

    constraints = search_context.get("constraints")
    if isinstance(constraints, dict):
        return {}, _normalize_constraints(constraints.get("hard"))

    return {}, []


def _get_soft_preferences(search_context: dict[str, Any]) -> tuple[dict[str, list[str]], list[str]]:
    soft = search_context.get("soft_preferences")
    if isinstance(soft, dict):
        return _normalize_constraint_map(soft), []
    if soft:
        return {}, _normalize_constraints(soft)
    return {}, []


def _get_excluded(search_context: dict[str, Any]) -> dict[str, list[str]]:
    excluded = search_context.get("excluded")
    if not isinstance(excluded, dict):
        return {}
    return _normalize_constraint_map(excluded)


def _normalize_observed_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        observed: list[str] = []
        for key, item in value.items():
            observed.extend(_normalize_constraints([key, item]))
        return observed
    return _normalize_constraints(value)


def _product_attributes(product: dict[str, Any]) -> dict[str, list[str]]:
    attributes = product.get("attributes")
    if not isinstance(attributes, dict):
        return {}
    return {
        str(slot).strip().lower(): _normalize_observed_values(values)
        for slot, values in attributes.items()
        if _normalize_observed_values(values)
    }


def _attribute_status(
    product: dict[str, Any],
    product_text: str,
    slot: str,
    expected_values: list[str],
) -> str:
    attributes = _product_attributes(product)
    observed_values = attributes.get(slot, [])
    expected = {value for value in expected_values if value}
    if observed_values:
        return "pass" if expected & set(observed_values) else "fail"
    if slot in product:
        direct_values = _normalize_observed_values(product.get(slot))
        if direct_values:
            return "pass" if expected & set(direct_values) else "fail"
    return "pass" if any(value in product_text for value in expected) else "fail"


def _numeric_value(product: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = product.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _price_status(product: dict[str, Any], minimum: float | None, maximum: float | None) -> str:
    price = _numeric_value(product, "price")
    if price is None:
        return "unknown"
    if minimum is not None and price < minimum:
        return "fail"
    if maximum is not None and price > maximum:
        return "fail"
    return "pass"


def _rating_status(product: dict[str, Any], minimum: float | None) -> str:
    rating = _numeric_value(product, "rating", "average_rating")
    if rating is None or minimum is None:
        return "unknown"
    return "pass" if rating >= minimum else "fail"


def _weights(intent: str | None) -> dict[str, float]:
    if intent == "buying":
        return BUYING_WEIGHTS
    if intent == "browsing":
        return BROWSING_WEIGHTS
    return UNKNOWN_WEIGHTS


def _float_value(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    if isinstance(value, (list, tuple)) and value:
        return _float_value(value[0])
    return None


def score_constraints(
    product: dict[str, Any],
    search_context: dict[str, Any],
) -> ConstraintScore:
    product_text = _product_text(product)
    weights = _weights(str(search_context.get("intent") or "").lower() or None)
    hard_constraints, hard_terms = _get_hard_constraints(search_context)
    soft_preferences, soft_terms = _get_soft_preferences(search_context)
    excluded = _get_excluded(search_context)

    score = 0.0
    reason_codes: list[str] = []
    matched_preferences: list[str] = []
    hard_failures: list[str] = []

    price_min = _float_value(hard_constraints.get("price_min"))
    price_max = _float_value(hard_constraints.get("price_max"))
    if price_min is not None or price_max is not None:
        status = _price_status(product, price_min, price_max)
        if status == "pass":
            score += weights["hard_pass"]
            reason_codes.append("hard_pass_price")
            matched_preferences.append("price")
        elif status == "fail":
            score += weights["hard_fail"]
            reason_codes.append("hard_fail_price")
            hard_failures.append("price")

    rating_min = _float_value(hard_constraints.get("rating_min"))
    if rating_min is not None:
        status = _rating_status(product, rating_min)
        if status == "pass":
            score += weights["hard_pass"]
            reason_codes.append("hard_pass_rating")
            matched_preferences.append("rating")
        elif status == "fail":
            score += weights["hard_fail"]
            reason_codes.append("hard_fail_rating")
            hard_failures.append("rating")

    for slot, values in hard_constraints.items():
        if slot in NUMERIC_SLOTS or slot not in ATTRIBUTE_SLOTS:
            continue
        status = _attribute_status(product, product_text, slot, values)
        label = f"{slot}:{'/'.join(values)}"
        if status == "pass":
            score += weights["hard_pass"]
            reason_codes.append(f"hard_pass_{slot}")
            matched_preferences.append(label)
        else:
            score += weights["hard_fail"]
            reason_codes.append(f"hard_fail_{slot}")
            hard_failures.append(label)

    for term in hard_terms:
        if term in product_text:
            score += weights["hard_pass"]
            reason_codes.append("hard_pass_text")
            matched_preferences.append(term)
        else:
            score += weights["hard_fail"]
            reason_codes.append("hard_fail_text")
            hard_failures.append(term)

    for slot, values in soft_preferences.items():
        if slot not in ATTRIBUTE_SLOTS:
            continue
        if _attribute_status(product, product_text, slot, values) == "pass":
            score += weights["soft_match"]
            reason_codes.append(f"soft_match_{slot}")
            matched_preferences.append(f"{slot}:{'/'.join(values)}")

    for term in soft_terms:
        if term in product_text:
            score += weights["soft_match"]
            reason_codes.append("soft_match_text")
            matched_preferences.append(term)

    for slot, values in excluded.items():
        if slot == "other":
            if any(value in product_text for value in values):
                score += weights["exclude_fail"]
                reason_codes.append("exclude_fail_text")
                hard_failures.extend(values)
            continue
        if slot not in ATTRIBUTE_SLOTS:
            continue
        if _attribute_status(product, product_text, slot, values) == "pass":
            score += weights["exclude_fail"]
            reason_codes.append(f"exclude_fail_{slot}")
            hard_failures.append(f"exclude:{slot}:{'/'.join(values)}")

    return ConstraintScore(
        score=score,
        reason_codes=reason_codes,
        matched_preferences=matched_preferences,
        hard_failures=hard_failures,
    )
