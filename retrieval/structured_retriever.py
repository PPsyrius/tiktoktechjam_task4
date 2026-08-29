"""Exact-attribute postings and numeric price retrieval; unknown is not false."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from .product_store import constraint_status, normalized
from .types import Candidate, SourceHit


class StructuredRetriever:
    def __init__(self, store):
        self.store = store
        self.postings = defaultdict(set)
        self.known = defaultdict(set)
        prices = []
        for asin, product in store.products.items():
            for field, values in product.attributes.items():
                for value in values:
                    self.postings[(field, value)].add(asin)
                    self.known[field].add(asin)
            if product.price is not None:
                prices.append((product.price, asin))
        prices.sort()
        self.prices = [p for p, _ in prices]
        self.price_ids = [asin for _, asin in prices]

    def search(self, context, limit):
        if limit <= 0 or not context.constraints:
            return []
        matches = defaultdict(int)
        for constraint in context.constraints:
            if constraint.field == "price":
                lo = bisect_left(self.prices, constraint.minimum) if constraint.minimum is not None else 0
                hi = bisect_right(self.prices, constraint.maximum) if constraint.maximum is not None else len(self.prices)
                matching = self.price_ids[lo:hi]
            else:
                matching = set().union(*(self.postings.get((constraint.field, normalized(v)), set())
                                         for v in constraint.values))
                if constraint.negative:
                    matching = self.known[constraint.field] - matching
            for asin in matching:
                matches[asin] += 1
        # Unknown facts can survive, but at least one known match starts this route.
        eligible = [asin for asin in matches if not any(
            c.hard and constraint_status(self.store[asin], c) == "fail" for c in context.constraints)]
        eligible.sort(key=lambda asin: (-matches[asin], asin))
        return [Candidate(asin, (SourceHit("structured", rank, float(matches[asin])),))
                for rank, asin in enumerate(eligible[:limit], 1)]
