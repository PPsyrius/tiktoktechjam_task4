from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from retrieval import Constraint, HybridRetriever, ProductStore, SearchContext
from starter.memory import (
    AttributeStatus,
    CurrentState,
    MemoryService,
    ParseUpdate,
    StateNotFoundError,
    StateUpdate,
    UpdateOperation,
)
from starter.reranker import rank_candidates
from starter.snippet_index import SnippetIndex, flatten_phrases
from starter.understanding import parse_requirement


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _terms(text: str) -> list[str]:
    """Compatibility helper for the local experiments' legacy retriever."""
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


CLARIFY_ORDER = (
    "color",
    "material",
    "brand",
    "size",
    "style",
    "feature",
    "use_case",
    "budget",
)
RETRIEVAL_POOL_SIZE = 50
SNIPPET_POOL_SIZE = 400
PRODUCT_TEXT_FIELDS = ("title", "categories", "features", "details", "store", "description")
ATTRIBUTE_FIELDS = frozenset({
    "category", "brand", "material", "color", "size", "style", "use_case", "feature",
})


def _values(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _string_values(value: object) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in (str(raw) for raw in _values(value))
        if item.strip()
    )


def _constraint_phrases(state: Any) -> list[str]:
    phrases = flatten_phrases(state.hard_constraints)
    phrases.extend(flatten_phrases(state.soft_preferences))
    return list(dict.fromkeys(phrase for phrase in phrases if isinstance(phrase, str) and phrase.strip()))


def _retrieval_context(message: str, state: Any) -> SearchContext:
    queries = list(dict.fromkeys(
        text.strip()
        for text in (message, state.category, state.product_type, state.retrieval_text())
        if isinstance(text, str) and text.strip()
    ))
    constraints: list[Constraint] = []

    hard = state.hard_constraints
    price_min = hard.get("price_min")
    price_max = hard.get("price_max")
    if price_min is not None or price_max is not None:
        constraints.append(Constraint(
            "price",
            minimum=float(price_min) if price_min is not None else None,
            maximum=float(price_max) if price_max is not None else None,
            hard=True,
        ))

    for field, raw_value in hard.items():
        if field in ATTRIBUTE_FIELDS:
            values = _string_values(raw_value)
            if values:
                constraints.append(Constraint(field, values, hard=True))

    for field, raw_value in state.soft_preferences.items():
        if field in ATTRIBUTE_FIELDS:
            values = _string_values(raw_value)
            if values:
                constraints.append(Constraint(field, values))

    for field, raw_value in state.excluded.items():
        if field in ATTRIBUTE_FIELDS:
            values = _string_values(raw_value)
            if values:
                constraints.append(Constraint(field, values, hard=True, negative=True))

    return SearchContext(
        queries=tuple(queries),
        constraints=tuple(constraints),
        mode=state.intent.value,
    )


class Agent:
    """Session memory, hybrid retrieval, snippet matching, and constraint-aware reranking."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        retriever: HybridRetriever | None = None,
        cache_dir: str | Path | None = None,
        candidate_limit: int = 100,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.memory = MemoryService()
        self.retriever = retriever or HybridRetriever(
            ProductStore.from_jsonl(self.catalog_path), cache_dir=cache_dir,
        )
        self.store = self.retriever.store
        self.snippets = SnippetIndex(
            self.store,
            catalog_path=self.catalog_path if retriever is None else None,
        )
        # Legacy local experiments reuse the BM25 connection for diagnostics.
        self.connection = self.retriever.bm25.connection if self.retriever.bm25 else None
        max_candidates = self.retriever.config.max_candidates
        if (
            not isinstance(candidate_limit, int)
            or isinstance(candidate_limit, bool)
            or not 1 <= candidate_limit <= max_candidates
        ):
            raise ValueError("candidate_limit exceeds retriever capacity")
        self.candidate_limit = candidate_limit

    def reset(self, session_id: str, user_profile: dict) -> None:
        # CurrentState stores the active shopping task, not catalog records or
        # raw user history. Replacing it is the session-reset invariant.
        self.memory.reset_state(session_id)

    def apply_update(self, parsed: ParseUpdate | dict) -> CurrentState:
        """Apply a structured Dialogue-module update to session memory."""
        return self.memory.apply_update(parsed)

    def get_state(self, session_id: str) -> CurrentState:
        return self.memory.get_state(session_id)

    def _product_payload(self, parent_asin: str) -> dict[str, Any]:
        product = self.store[parent_asin]
        payload = {
            "parent_asin": parent_asin,
            "attributes": dict(product.attributes),
            "price": product.price,
        }
        payload.update(dict(zip(PRODUCT_TEXT_FIELDS, product.fields)))
        return payload

    def _candidate_payloads(self, pool: object) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for candidate in pool:
            payload = self._product_payload(candidate.parent_asin)
            source_scores: dict[str, float] = {}
            source_ranks: dict[str, int] = {}
            for hit in candidate.hits:
                score = float(hit.score)
                if not hit.higher_is_better:
                    score = max(0.0, -score)
                source_scores[hit.source] = max(source_scores.get(hit.source, 0.0), score)
                source_ranks[hit.source] = min(source_ranks.get(hit.source, hit.rank), hit.rank)
            payload["source_scores"] = source_scores
            payload["source_ranks"] = source_ranks
            candidates.append(payload)
        return candidates

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        try:
            self.memory.get_state(session_id)
        except StateNotFoundError as error:
            raise RuntimeError("reset must be called before respond") from error

        retrieval_state = self.memory.get_retrieval_state(session_id)
        parse_result = parse_requirement(
            session_id,
            user_message,
            turn,
            search_context=retrieval_state,
        )
        parsed = parse_result.parsed
        if parsed is not None:
            if parsed.reset_task:
                current = self.memory.get_state(session_id)
                if current.category and not any(update.slot == "category" for update in parsed.updates):
                    parsed = ParseUpdate(
                        session_id=parsed.session_id,
                        intent=parsed.intent,
                        updates=(
                            StateUpdate("category", UpdateOperation.SET, current.category),
                            *parsed.updates,
                        ),
                        reset_task=True,
                        source_turn=parsed.source_turn,
                    )
            try:
                self.memory.apply_update(parsed)
            except (TypeError, ValueError):
                pass
            else:
                retrieval_state = self.memory.get_retrieval_state(session_id)

        retrieval_context = _retrieval_context(user_message, retrieval_state)
        pool = self.retriever.retrieve(
            retrieval_context,
            limit=max(RETRIEVAL_POOL_SIZE, top_k),
        )
        ranking_context = {
            "intent": retrieval_state.intent.value,
            "hard_constraints": retrieval_state.hard_constraints,
            "soft_preferences": retrieval_state.soft_preferences,
            "excluded": retrieval_state.excluded,
        }
        payloads = self._candidate_payloads(pool)
        seen = {item["parent_asin"] for item in payloads}
        snippet_hits = self.snippets.search(
            _constraint_phrases(retrieval_state),
            message=user_message,
            limit=SNIPPET_POOL_SIZE,
        )
        for hit in snippet_hits:
            asin = hit["parent_asin"]
            if asin in seen:
                existing = next(item for item in payloads if item["parent_asin"] == asin)
                source_scores = dict(existing.get("source_scores") or {})
                source_ranks = dict(existing.get("source_ranks") or {})
                for source, score in (hit.get("source_scores") or {}).items():
                    source_scores[source] = max(source_scores.get(source, 0.0), float(score))
                for source, rank in (hit.get("source_ranks") or {}).items():
                    rank_value = int(rank)
                    source_ranks[source] = min(source_ranks.get(source, rank_value), rank_value)
                existing["source_scores"] = source_scores
                existing["source_ranks"] = source_ranks
                continue
            payload = self._product_payload(asin)
            payload["source_scores"] = dict(hit.get("source_scores") or {})
            payload["source_ranks"] = dict(hit.get("source_ranks") or {})
            payloads.append(payload)
            seen.add(asin)
        ranking = rank_candidates(
            ranking_context,
            payloads,
            top_k=top_k,
        )
        recommendations = [
            {"parent_asin": item.parent_asin, "score": item.final_score}
            for item in ranking.items
        ]

        state = self.memory.get_state(session_id)
        clarify_candidates = [
            attribute
            for attribute in CLARIFY_ORDER
            if state.status_for(attribute) is AttributeStatus.UNKNOWN
        ]
        ask_attribute = self.memory.get_or_record_asked_attribute(
            session_id,
            turn,
            clarify_candidates,
        )
        if ask_attribute:
            message = f"Do you have a {ask_attribute} preference?"
        else:
            message = "Here are the closest matches I found."
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": parse_result.prompt_tokens,
                "completion_tokens": parse_result.completion_tokens,
            },
        }

    def close(self) -> None:
        self.retriever.close()
