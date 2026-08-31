from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .bm25_retriever import DEFAULT_WEIGHTS, BM25Retriever, lexical_queries, terms
from .merge import interleave, interleave_with_route_cap
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
    semantic_candidate_limit: int = 40
    dynamic_semantic_gate: bool = False
    semantic_min_lexical_fill: float = 0.75
    semantic_shadow_min_lexical_overlap: int = 2
    semantic_shadow_lexical_window: int = 160
    bm25_weights: tuple = DEFAULT_WEIGHTS

    def __post_init__(self):
        object.__setattr__(self, "bm25_weights", tuple(self.bm25_weights))
        if not isinstance(self.max_candidates, int) or isinstance(self.max_candidates, bool) or self.max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if (not isinstance(self.semantic_candidate_limit, int)
                or isinstance(self.semantic_candidate_limit, bool)
                or self.semantic_candidate_limit < 0):
            raise ValueError("semantic_candidate_limit must be non-negative")
        if not isinstance(self.dynamic_semantic_gate, bool):
            raise TypeError("dynamic_semantic_gate must be a boolean")
        if not 0 < self.semantic_min_lexical_fill <= 1:
            raise ValueError("semantic_min_lexical_fill must be in (0, 1]")
        if (not isinstance(self.semantic_shadow_min_lexical_overlap, int)
                or isinstance(self.semantic_shadow_min_lexical_overlap, bool)
                or self.semantic_shadow_min_lexical_overlap < 0):
            raise ValueError("semantic_shadow_min_lexical_overlap must be non-negative")
        if (not isinstance(self.semantic_shadow_lexical_window, int)
                or isinstance(self.semantic_shadow_lexical_window, bool)
                or self.semantic_shadow_lexical_window < 1):
            raise ValueError("semantic_shadow_lexical_window must be positive")


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
        routes = {}
        deferred_candidates = ()
        for name, retriever in self.routes.items():
            route_start = time.perf_counter()
            if name == "semantic" and self.config.dynamic_semantic_gate:
                diagnostics.semantic_gate = self._semantic_gate(context, routes, depth)
                if diagnostics.semantic_gate <= 0:
                    if context.initial_turn and self._has_informative_query(context):
                        try:
                            shadow = retriever.search(
                                context,
                                self.config.semantic_candidate_limit,
                            )
                            shadow_ids = tuple(
                                candidate.parent_asin
                                for candidate in shadow
                                if candidate.parent_asin in self.store
                                and self._allowed(candidate.parent_asin, context)
                            )
                            admit_shadow = context.mode != "buying"
                            admit_direct = False
                            if context.mode == "buying":
                                lexical_ids = {
                                    candidate.parent_asin
                                    for candidate in routes.get("bm25", [])[
                                        :self.config.semantic_shadow_lexical_window
                                    ]
                                }
                                diagnostics.semantic_shadow_overlap = len(
                                    lexical_ids & set(shadow_ids)
                                )
                                admit_direct = (
                                    diagnostics.semantic_shadow_overlap
                                    >= self.config.semantic_shadow_min_lexical_overlap
                                )
                            if admit_direct:
                                routes[name] = list(shadow)
                                diagnostics.semantic_gate = 1.0
                                diagnostics.route_counts[name] = len(shadow)
                            elif admit_shadow:
                                deferred_candidates = shadow_ids
                            diagnostics.route_counts["semantic_shadow"] = len(
                                deferred_candidates
                            )
                        except Exception as exc:
                            diagnostics.errors["semantic_shadow"] = (
                                f"{type(exc).__name__}: {exc}"
                            )
                    diagnostics.route_counts.setdefault(name, 0)
                    diagnostics.route_ms[name] = (time.perf_counter() - route_start) * 1000
                    continue
            try:
                candidates = []
                for candidate in retriever.search(context, depth):
                    if candidate.parent_asin not in self.store:
                        continue
                    if not self._allowed(candidate.parent_asin, context):
                        diagnostics.filtered_count += 1
                        continue
                    candidates.append(candidate)
                routes[name] = candidates
                diagnostics.route_counts[name] = len(candidates)
            except Exception as exc:
                diagnostics.errors[name] = f"{type(exc).__name__}: {exc}"
                diagnostics.route_counts[name] = 0
                LOGGER.warning("Retrieval route %s failed: %s", name, exc)
            diagnostics.route_ms[name] = (time.perf_counter() - route_start) * 1000
        if "semantic" in routes and not self.config.dynamic_semantic_gate:
            diagnostics.semantic_gate = 1.0
        if "semantic" in routes:
            selected = interleave_with_route_cap(
                routes,
                limit,
                capped_route="semantic",
                cap=round(
                    self.config.semantic_candidate_limit * diagnostics.semantic_gate
                ),
            )
        else:
            selected = interleave(list(routes.values()), limit)
        if not selected and self.config.catalog_fallback:
            diagnostics.fallback_used = True
            for asin in self._fallback_ids:
                if self._allowed(asin, context):
                    selected.append(Candidate(asin, (SourceHit("fallback", len(selected) + 1, 0.0),)))
                if len(selected) >= limit:
                    break
        diagnostics.total_ms = (time.perf_counter() - started) * 1000
        return CandidatePool(
            tuple(selected),
            diagnostics,
            deferred_candidates=deferred_candidates,
        )

    @staticmethod
    def _has_informative_query(context):
        informative_terms = {
            token
            for query in lexical_queries(context)
            for token in terms(query)
        }
        informative_terms.update(
            token
            for constraint in context.constraints
            for value in constraint.values
            for token in terms(value)
        )
        return len(informative_terms) >= 2

    def _semantic_gate(self, context, completed_routes, depth):
        """Return a runtime gate using only observable retrieval signals."""
        if not lexical_queries(context):
            return 0.0
        if context.preference_override:
            return 1.0

        if not self._has_informative_query(context):
            return 0.0

        lexical = completed_routes.get("bm25")
        if lexical is None:
            return 1.0
        lexical_fill = len(lexical) / max(depth, 1)
        threshold = self.config.semantic_min_lexical_fill
        if lexical_fill >= threshold:
            return 0.0
        return min(1.0, max(0.0, 1.0 - lexical_fill / threshold))

    def _allowed(self, asin, context):
        return not self.config.filter_known_hard_failures or not any(
            c.hard and constraint_status(self.store[asin], c) == "fail" for c in context.constraints)

    def close(self):
        if self.bm25 is not None:
            self.bm25.close()
