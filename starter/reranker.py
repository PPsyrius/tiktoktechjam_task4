from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .candidate_fusion import fuse_candidates
from .constraint_scorer import score_constraints


@dataclass(frozen=True)
class RankedItem:
    parent_asin: str
    final_score: float
    reason_codes: list[str]
    matched_preferences: list[str]
    hard_failures: list[str]


@dataclass(frozen=True)
class RankingResult:
    items: list[RankedItem]
    clarification_slot: str | None = None


def _base_candidate_score(candidate: dict[str, Any]) -> float:
    source_scores = candidate.get("source_scores") or {}
    return sum(float(score) for score in source_scores.values())


def rank_candidates(
    search_context: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    top_k: int = 10,
) -> RankingResult:
    fused_candidates = fuse_candidates(candidate_pool)
    ranked_items: list[RankedItem] = []

    for candidate in fused_candidates:
        constraint_score = score_constraints(candidate, search_context)
        final_score = _base_candidate_score(candidate) + constraint_score.score

        ranked_items.append(
            RankedItem(
                parent_asin=str(candidate["parent_asin"]),
                final_score=final_score,
                reason_codes=list(constraint_score.reason_codes),
                matched_preferences=list(constraint_score.matched_preferences),
                hard_failures=list(constraint_score.hard_failures),
            )
        )

    ranked_items.sort(
        key=lambda item: (-item.final_score, item.parent_asin)
    )

    return RankingResult(items=ranked_items[:top_k], clarification_slot=None)
