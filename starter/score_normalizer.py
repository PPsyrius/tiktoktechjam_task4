from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_score(value: Any, default: float = 0.0) -> float:
    score = safe_float(value, default=default)
    if score < 0.0:
        return 0.0
    return score


def normalize_source_scores(source_scores: dict[str, Any] | None) -> dict[str, float]:
    if not source_scores:
        return {}
    return {
        str(source): normalize_score(score)
        for source, score in source_scores.items()
    }
