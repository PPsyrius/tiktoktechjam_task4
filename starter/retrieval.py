from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from starter.memory.models import RetrievalState


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
BM25_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
DEFAULT_POOL_SIZE = 50
MAX_QUERY_TERMS = 40


def flatten_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def query_terms(*texts: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for token in TOKEN_RE.findall(text):
            term = token.lower()
            if len(term) <= 1 or term in STOPWORDS or term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) >= MAX_QUERY_TERMS:
                return terms
    return terms


def _string_phrases(mapping: dict[str, Any] | None) -> list[str]:
    phrases: list[str] = []
    for value in (mapping or {}).values():
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str) and item.strip():
                phrases.append(item.strip())
    return phrases


def search_context_from_state(state: RetrievalState | dict[str, Any]) -> dict[str, Any]:
    """Adapt Memory RetrievalState into the ranking SearchContext shape."""
    payload = state.to_dict() if isinstance(state, RetrievalState) else dict(state)
    hard_constraints = payload.get("hard_constraints") or {}
    soft_preferences = payload.get("soft_preferences") or {}
    if not isinstance(hard_constraints, dict):
        hard_constraints = {}
    if not isinstance(soft_preferences, dict):
        soft_preferences = {}
    return {
        "intent": payload.get("intent"),
        "category": payload.get("category"),
        "product_type": payload.get("product_type"),
        "hard_constraints": _string_phrases(hard_constraints) + _string_phrases(soft_preferences),
        "soft_preferences": soft_preferences,
        "excluded": payload.get("excluded") or {},
    }


def _positive_bm25(raw_rank: object) -> float:
    # FTS5 bm25() is negative, with more negative meaning a better match.
    # Ranking expects non-negative source scores where higher is better.
    try:
        return max(0.0, -float(raw_rank))
    except (TypeError, ValueError):
        return 0.0


class Retriever:
    """Keyword BM25 retrieval that emits a Candidate Pool for rank_candidates()."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        flatten_text(product.get("title")),
                        flatten_text(product.get("categories")),
                        flatten_text(product.get("features")),
                        flatten_text(product.get("details")),
                        flatten_text(product.get("store")),
                        flatten_text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def retrieve(
        self,
        query: str,
        search_context: RetrievalState | dict[str, Any] | None = None,
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> list[dict[str, Any]]:
        extra_texts = [query]
        if isinstance(search_context, RetrievalState):
            extra_texts.append(search_context.retrieval_text())
        elif isinstance(search_context, dict):
            extra_texts.append(str(search_context.get("category") or ""))
            extra_texts.append(str(search_context.get("product_type") or ""))
            hard = search_context.get("hard_constraints")
            if isinstance(hard, list):
                extra_texts.extend(str(item) for item in hard)
            elif isinstance(hard, dict):
                extra_texts.extend(_string_phrases(hard))
            extra_texts.extend(_string_phrases(search_context.get("soft_preferences") or {}))

        terms = query_terms(*extra_texts)
        if not terms:
            return []

        expression = " OR ".join(f'"{term}"' for term in terms)
        weight_list = ", ".join(str(weight) for weight in BM25_WEIGHTS)
        rows = self.connection.execute(
            "SELECT parent_asin, title, categories, features, details, store, description, "
            f"bm25(products, {weight_list}) AS rank "
            "FROM products WHERE products MATCH ? "
            "ORDER BY rank LIMIT ?",
            (expression, max(1, pool_size)),
        ).fetchall()

        candidates: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            parent_asin, title, categories, features, details, store, description, rank = row
            candidates.append({
                "parent_asin": str(parent_asin),
                "title": title,
                "categories": categories,
                "features": features,
                "details": details,
                "store": store,
                "description": description,
                "source_scores": {"bm25": _positive_bm25(rank)},
                "source_ranks": {"bm25": index},
            })
        return candidates
