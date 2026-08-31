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


def _base_candidate_score(candidate: dict[str, Any], semantic_weight: float = 0.6) -> float:
    ranks = candidate.get("source_ranks") or {}
    scores = candidate.get("source_scores") or {}
    total = 0.0
    snippet_rank = ranks.get("snippet")
    if snippet_rank:
        total += 1.2 / float(snippet_rank)
    elif scores.get("snippet"):
        total += min(float(scores["snippet"]), 80.0) / 40.0
    bm25_rank = ranks.get("bm25")
    if bm25_rank:
        total += 2.0 / float(bm25_rank)
    elif scores.get("bm25"):
        total += min(float(scores["bm25"]), 20.0) / 20.0
    structured_rank = ranks.get("structured")
    if structured_rank:
        total += 0.4 / float(structured_rank)
    elif scores.get("structured"):
        total += min(float(scores["structured"]), 5.0) / 10.0
    semantic_rank = ranks.get("semantic")
    if semantic_rank:
        total += semantic_weight / float(semantic_rank)
    return total


def rank_candidates(
    search_context: dict[str, Any],
    candidate_pool: list[dict[str, Any]],
    top_k: int = 10,
    semantic_weight: float = 0.6,
) -> RankingResult:
    fused_candidates = fuse_candidates(candidate_pool)
    ranked_items: list[RankedItem] = []

    for candidate in fused_candidates:
        constraint_score = score_constraints(candidate, search_context)
        final_score = _base_candidate_score(candidate, semantic_weight) + constraint_score.score

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
