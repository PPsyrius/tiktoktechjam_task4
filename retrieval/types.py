"""Task 4 boundary objects. No session state or evaluation labels belong here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math

from starter.catalog.product import ATTRIBUTE_FIELDS


@dataclass(frozen=True)
class Constraint:
    field: str
    values: tuple[str, ...] = ()  # OR within a field; AND between constraints.
    minimum: float | None = None
    maximum: float | None = None
    hard: bool = False
    negative: bool = False

    def __post_init__(self) -> None:
        if self.field not in ATTRIBUTE_FIELDS | {"price"}:
            raise ValueError("Unsupported constraint field: " + self.field)
        if isinstance(self.values, str):
            raise TypeError("values must be a sequence, not a string")
        object.__setattr__(self, "values", tuple(self.values))
        if self.field == "price":
            if self.values or self.negative:
                raise ValueError("price uses minimum/maximum, not values or negative")
            if self.minimum is None and self.maximum is None:
                raise ValueError("price requires a bound")
            for bound in (self.minimum, self.maximum):
                if bound is not None and (not math.isfinite(bound) or bound < 0):
                    raise ValueError("price bounds must be finite and non-negative")
            if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
                raise ValueError("minimum exceeds maximum")
        elif not self.values or any(not isinstance(v, str) or not v.strip() for v in self.values):
            raise ValueError("attribute constraints require nonempty string values")
        elif self.minimum is not None or self.maximum is not None:
            raise ValueError("numeric bounds are only valid for price")


@dataclass(frozen=True)
class SearchContext:
    queries: tuple[str, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    mode: str = "unknown"
    semantic_query: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.queries, str):
            raise TypeError("queries must be a sequence, not a string")
        object.__setattr__(self, "queries", tuple(self.queries))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        if self.mode not in {"buying", "browsing", "unknown"}:
            raise ValueError("mode must be buying, browsing or unknown")
        if any(not isinstance(q, str) for q in self.queries):
            raise TypeError("queries must contain strings")
        if any(not isinstance(c, Constraint) for c in self.constraints):
            raise TypeError("constraints must contain Constraint objects")
        if not isinstance(self.semantic_query, str):
            raise TypeError("semantic_query must be a string")


@dataclass(frozen=True)
class SourceHit:
    source: str
    rank: int
    score: float
    query: str = ""
    higher_is_better: bool = True


@dataclass(frozen=True)
class Candidate:
    parent_asin: str
    hits: tuple[SourceHit, ...] = ()


@dataclass
class RetrievalDiagnostics:
    route_counts: dict = field(default_factory=dict)
    route_ms: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    fallback_used: bool = False
    filtered_count: int = 0
    total_ms: float = 0.0


@dataclass(frozen=True)
class CandidatePool:
    candidates: tuple[Candidate, ...]
    diagnostics: RetrievalDiagnostics

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self):
        return len(self.candidates)
