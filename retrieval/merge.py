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


def interleave_with_route_cap(named_routes, limit, capped_route, cap):
    """Admit candidates fairly while limiting one route's unique additions.

    A candidate already admitted by an uncapped route does not consume the
    capped route's budget. Evidence is still retained from every route.
    """
    if limit <= 0:
        return []
    if cap < 0:
        raise ValueError("route cap must be non-negative")

    evidence = {}
    for route in named_routes.values():
        for candidate in route:
            hits = evidence.setdefault(candidate.parent_asin, [])
            for hit in candidate.hits:
                if hit not in hits:
                    hits.append(hit)

    selected_ids = []
    seen = set()

    def admit(routes, budget):
        admitted = 0
        if budget <= 0 or not routes:
            return
        for row in zip_longest(*routes):
            for candidate in row:
                if candidate is None or candidate.parent_asin in seen:
                    continue
                selected_ids.append(candidate.parent_asin)
                seen.add(candidate.parent_asin)
                admitted += 1
                if admitted >= budget or len(selected_ids) >= limit:
                    return

    capped_candidates = named_routes.get(capped_route, [])
    capped_budget = min(cap, limit, len(capped_candidates))
    uncapped_routes = [
        route for name, route in named_routes.items() if name != capped_route
    ]
    admit(uncapped_routes, limit - capped_budget)
    admit([capped_candidates], capped_budget)
    # Fill only from uncapped routes. An undersized pool is preferable to
    # silently exceeding the semantic admission ceiling.
    admit(uncapped_routes, limit - len(selected_ids))

    return [Candidate(asin, tuple(evidence[asin])) for asin in selected_ids]
