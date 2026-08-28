from __future__ import annotations

from typing import Any

from .score_normalizer import normalize_source_scores


def fuse_candidates(candidate_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}

    for candidate in candidate_pool:
        parent_asin = str(candidate.get("parent_asin", "")).strip()
        if not parent_asin:
            continue

        source_scores = normalize_source_scores(candidate.get("source_scores"))
        source_ranks = dict(candidate.get("source_ranks") or {})

        if parent_asin not in fused:
            fused[parent_asin] = {
                **candidate,
                "parent_asin": parent_asin,
                "source_scores": source_scores,
                "source_ranks": source_ranks,
            }
            continue

        existing = fused[parent_asin]
        existing_scores = dict(existing.get("source_scores") or {})
        existing_ranks = dict(existing.get("source_ranks") or {})

        for source, score in source_scores.items():
            existing_scores[source] = max(existing_scores.get(source, 0.0), score)

        for source, rank in source_ranks.items():
            if source not in existing_ranks:
                existing_ranks[source] = rank
            else:
                existing_ranks[source] = min(existing_ranks[source], rank)

        existing["source_scores"] = existing_scores
        existing["source_ranks"] = existing_ranks

    return list(fused.values())
