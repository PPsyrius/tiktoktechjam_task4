from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .bm25_retriever import DEFAULT_WEIGHTS, BM25Retriever
from .merge import interleave
from .product_store import constraint_status
from .structured_retriever import StructuredRetriever
from .types import Candidate, CandidatePool, RetrievalDiagnostics, SearchContext, SourceHit

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalConfig:
    enable_bm25: bool = True
    enable_structured: bool = True
    enable_semantic: bool = False
    filter_known_hard_failures: bool = False
    catalog_fallback: bool = True
    max_candidates: int = 200
    bm25_weights: tuple = DEFAULT_WEIGHTS

    def __post_init__(self):
        object.__setattr__(self, "bm25_weights", tuple(self.bm25_weights))
        if not isinstance(self.max_candidates, int) or isinstance(self.max_candidates, bool) or self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")


class HybridRetriever:
    def __init__(self, store, config=None, cache_dir=None, semantic=None, semantic_error=None):
        self.store = store
        self.config = config or RetrievalConfig()
        self.routes = {}
        self.startup_errors = {}
        self.bm25 = None
        if self.config.enable_bm25:
            self.bm25 = BM25Retriever(store, cache_dir=cache_dir, weights=self.config.bm25_weights)
            self.routes["bm25"] = self.bm25
        if self.config.enable_structured:
            self.routes["structured"] = StructuredRetriever(store)
        if self.config.enable_semantic:
            if semantic is not None:
                self.routes["semantic"] = semantic
            else:
                self.startup_errors["semantic"] = semantic_error or "Semantic route requested without an encoder/index"
                LOGGER.warning(self.startup_errors["semantic"])
        self._fallback_ids = sorted(store.products)

    def retrieve(self, context: SearchContext, limit=100) -> CandidatePool:
        if not isinstance(context, SearchContext):
            raise TypeError("Adapt upstream state to retrieval.SearchContext first")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 <= limit <= self.config.max_candidates:
            raise ValueError("limit must be between 0 and max_candidates")
        started = time.perf_counter()
        diagnostics = RetrievalDiagnostics(errors=dict(self.startup_errors))
        if limit == 0:
            return CandidatePool((), diagnostics)
        multiplier = 3 if context.mode in {"browsing", "unknown"} else 2
        depth = min(len(self.store), limit * multiplier)
        routes = []
        for name, retriever in self.routes.items():
            route_start = time.perf_counter()
            try:
                candidates = []
                for candidate in retriever.search(context, depth):
                    if candidate.parent_asin not in self.store:
                        continue
                    if not self._allowed(candidate.parent_asin, context):
                        diagnostics.filtered_count += 1
                        continue
                    candidates.append(candidate)
                routes.append(candidates)
                diagnostics.route_counts[name] = len(candidates)
            except Exception as exc:
                diagnostics.errors[name] = f"{type(exc).__name__}: {exc}"
                diagnostics.route_counts[name] = 0
                LOGGER.warning("Retrieval route %s failed: %s", name, exc)
            diagnostics.route_ms[name] = (time.perf_counter() - route_start) * 1000
        selected = interleave(routes, limit)
        if not selected and self.config.catalog_fallback:
            diagnostics.fallback_used = True
            for asin in self._fallback_ids:
                if self._allowed(asin, context):
                    selected.append(Candidate(asin, (SourceHit("fallback", len(selected) + 1, 0.0),)))
                if len(selected) >= limit:
                    break
        diagnostics.total_ms = (time.perf_counter() - started) * 1000
        return CandidatePool(tuple(selected), diagnostics)

    def _allowed(self, asin, context):
        return not self.config.filter_known_hard_failures or not any(
            c.hard and constraint_status(self.store[asin], c) == "fail" for c in context.constraints)

    def close(self):
        if self.bm25 is not None:
            self.bm25.close()
