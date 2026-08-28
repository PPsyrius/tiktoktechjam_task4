from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []
    return []


def _get_hard_constraints(search_context: dict[str, Any]) -> list[str]:
    hard = search_context.get("hard_constraints")
    if hard:
        return _normalize_constraints(hard)

    constraints = search_context.get("constraints")
    if isinstance(constraints, dict):
        return _normalize_constraints(constraints.get("hard"))

    return []


def score_constraints(
    product: dict[str, Any],
    search_context: dict[str, Any],
) -> ConstraintScore:
    product_text = _product_text(product)
    hard_constraints = _get_hard_constraints(search_context)

    score = 0.0
    reason_codes: list[str] = []
    matched_preferences: list[str] = []
    hard_failures: list[str] = []

    for constraint in hard_constraints:
        if constraint in product_text:
            score += 2.0
            reason_codes.append("hard_pass")
            matched_preferences.append(constraint)
        else:
            score -= 5.0
            reason_codes.append("hard_fail")
            hard_failures.append(constraint)

    return ConstraintScore(
        score=score,
        reason_codes=reason_codes,
        matched_preferences=matched_preferences,
        hard_failures=hard_failures,
    )