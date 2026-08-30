from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from starter.memory.models import NUMERIC_SLOTS
from starter.snippet_index import flatten_phrases, fold_text, fold_variants


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


@dataclass(frozen=True)
class ConstraintScore:
    score: float
    reason_codes: list[str]
    matched_preferences: list[str]
    hard_failures: list[str]


def _product_text(product: dict[str, Any]) -> str:
    parts = [
        fold_text(product.get("title")),
        fold_text(product.get("categories")),
        fold_text(product.get("features")),
        fold_text(product.get("details")),
        fold_text(product.get("description")),
        fold_text(product.get("store")),
    ]
    return " ".join(part for part in parts if part)


def _constraint_phrases(search_context: dict[str, Any]) -> list[str]:
    def textual_phrases(value: object) -> list[str]:
        if not isinstance(value, dict):
            return flatten_phrases(value)
        phrases: list[str] = []
        for field, field_value in value.items():
            if field not in NUMERIC_SLOTS:
                phrases.extend(flatten_phrases(field_value))
        return phrases

    phrases = textual_phrases(search_context.get("hard_constraints"))
    phrases.extend(textual_phrases(search_context.get("soft_preferences")))
    constraints = search_context.get("constraints")
    if isinstance(constraints, dict):
        phrases.extend(textual_phrases(constraints.get("hard")))
        phrases.extend(textual_phrases(constraints.get("soft")))
    category = search_context.get("category")
    if isinstance(category, str) and category.strip():
        phrases.append(category.strip())
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        key = fold_text(phrase)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(phrase)
    return unique


def _tokens(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(text):
        term = token.lower()
        if len(term) <= 1 or term in STOPWORDS or term in seen:
            continue
        seen.add(term)
        tokens.append(term)
    return tokens


def score_constraints(
    product: dict[str, Any],
    search_context: dict[str, Any],
) -> ConstraintScore:
    product_text = _product_text(product)
    phrases = _constraint_phrases(search_context)
    excluded = flatten_phrases(search_context.get("excluded"))

    score = 0.0
    reason_codes: list[str] = []
    matched_preferences: list[str] = []
    hard_failures: list[str] = []

    for phrase in phrases:
        folded_options = fold_variants(phrase)
        if not folded_options:
            continue
        matched = next((item for item in folded_options if item in product_text), None)
        if matched:
            bonus = 4.0 + min(len(matched), 180) / 15.0
            score += bonus
            reason_codes.append("hard_pass")
            matched_preferences.append(phrase)
            continue
        tokens = _tokens(folded_options[0])
        if tokens:
            hits = sum(1 for token in tokens if token in product_text)
            if hits:
                score += 1.5 * hits / len(tokens)
                reason_codes.append("partial_pass")
                continue
        penalty = 5.0 if len(folded_options[0]) < 16 else 2.0
        score -= penalty
        reason_codes.append("hard_fail")
        hard_failures.append(phrase)

    for phrase in excluded:
        folded = fold_text(phrase)
        if folded and folded in product_text:
            score -= 8.0
            reason_codes.append("excluded")
            hard_failures.append(phrase)

    price = product.get("price")
    hard = search_context.get("hard_constraints")
    if isinstance(hard, dict) and price is not None:
        maximum = hard.get("price_max")
        minimum = hard.get("price_min")
        try:
            price_value = float(price)
            if maximum is not None and price_value > float(maximum):
                score -= 4.0
                reason_codes.append("price_fail")
            elif minimum is not None and price_value < float(minimum):
                score -= 2.0
                reason_codes.append("price_fail")
            elif maximum is not None or minimum is not None:
                score += 1.0
                reason_codes.append("price_pass")
        except (TypeError, ValueError):
            pass

    return ConstraintScore(
        score=score,
        reason_codes=reason_codes,
        matched_preferences=matched_preferences,
        hard_failures=hard_failures,
    )
