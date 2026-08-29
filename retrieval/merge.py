from __future__ import annotations

from itertools import zip_longest
from .types import Candidate


def interleave(routes, limit):
    """Fair candidate admission, NOT score fusion. Keep evidence from every route."""
    if limit <= 0:
        return []
    evidence = {}
    for route in routes:
        for candidate in route:
            hits = evidence.setdefault(candidate.parent_asin, [])
            for hit in candidate.hits:
                if hit not in hits:
                    hits.append(hit)
    selected = []
    seen = set()
    for row in zip_longest(*routes):
        for candidate in row:
            if candidate is not None and candidate.parent_asin not in seen:
                selected.append(Candidate(candidate.parent_asin, tuple(evidence[candidate.parent_asin])))
                seen.add(candidate.parent_asin)
                if len(selected) >= limit:
                    return selected
    return selected
