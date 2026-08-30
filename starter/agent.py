from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from retrieval import Constraint, HybridRetriever, SearchContext
from starter.catalog import ProductStore
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
from starter.retrieval import search_context_from_state
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
    "feature",
    "material",
    "color",
    "other",
    "style",
    "use_case",
    "size",
    "brand",
    "budget",
)
RETRIEVAL_POOL_SIZE = 200
SNIPPET_POOL_SIZE = 600
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
    return list(dict.fromkeys(phrase for phrase in phrases if phrase.strip()))


def _retrieval_context(message: str, state: Any) -> SearchContext:
    phrases = _constraint_phrases(state)
    queries: list[str] = []
    for phrase in phrases:
        if len(phrase) >= 8:
            queries.append(phrase.strip())
    retrieval_text = state.retrieval_text()
    if isinstance(retrieval_text, str) and retrieval_text.strip():
        queries.append(retrieval_text.strip())
    if isinstance(message, str) and message.strip():
        queries.append(message.strip())
    for extra in (state.category, state.product_type):
        if isinstance(extra, str) and extra.strip():
            queries.append(extra.strip())
    queries = list(dict.fromkeys(text for text in queries if text))[:8]
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


def _same_value(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    return " ".join(str(left).split()).casefold() == " ".join(str(right).split()).casefold()


def _scope_override_update(parsed: ParseUpdate, current: CurrentState) -> ParseUpdate:
    """Keep a preference override local unless it explicitly starts a new task.

    The parser uses ``reset_task`` for both a same-task phrase such as
    "ignore my earlier preference" and a genuine category replacement. Memory's
    reset operation intentionally clears the whole shopping task, so applying it
    to the first case discards useful constraints gathered on earlier turns.
    """
    if not parsed.reset_task:
        return parsed

    direct_values = {
        "category": current.category,
        "product_type": current.product_type,
    }
    starts_new_task = any(
        update.slot in direct_values
        and update.op is UpdateOperation.SET
        and not _same_value(update.value, direct_values[update.slot])
        for update in parsed.updates
    )
    if starts_new_task:
        return parsed

    return ParseUpdate(
        session_id=parsed.session_id,
        intent=parsed.intent,
        # Scalar SET operations already replace the named slot. ADD operations
        # are intentionally preserved: the generic feature slot can contain
        # several independent requirements, and the utterance does not identify
        # which earlier item should be removed.
        updates=parsed.updates,
        reset_task=False,
        source_turn=parsed.source_turn,
    )


def _default_cache_dir() -> Path | None:
    cache_dir = Path(".cache/retrieval")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return cache_dir


class Agent:
    """Session memory, hybrid retrieval, snippet match, and constraint-aware reranking."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        retriever: HybridRetriever | None = None,
        cache_dir: str | Path | None = None,
        candidate_limit: int = 200,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.memory = MemoryService()
        resolved_cache = cache_dir if cache_dir is not None else _default_cache_dir()
        self.retriever = retriever or HybridRetriever(
            ProductStore.from_jsonl(
                self.catalog_path,
                cache_dir=(
                    Path(resolved_cache) / "catalog"
                    if resolved_cache is not None
                    else None
                ),
            ),
            cache_dir=resolved_cache,
        )
        self.store = self.retriever.store
        self.snippets = SnippetIndex(self.store)
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
        # Keep a bounded set of candidates already admitted for the active
        # shopping task.  Only identifiers are retained: old retrieval scores
        # must not influence ranking after the dialogue state changes.
        self._candidate_history: dict[str, tuple[int, tuple[str, ...]]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        # CurrentState stores the active shopping task, not catalog records or
        # raw user history. Replacing it is the session-reset invariant.
        self.memory.reset_state(session_id)
        self._candidate_history.pop(session_id, None)

    def apply_update(self, parsed: ParseUpdate | dict) -> CurrentState:
        """Apply a structured Dialogue-module update to session memory."""
        return self.memory.apply_update(parsed)

    def get_state(self, session_id: str) -> CurrentState:
        return self.memory.get_state(session_id)

    def _product_payload(self, parent_asin: str) -> dict[str, Any]:
        product = self.store[parent_asin]
        payload = {
            "parent_asin": parent_asin,
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

    def _admit_candidate_history(
        self,
        session_id: str,
        task_version: int,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Preserve candidate recall across turns of the same shopping task.

        A later clarification may produce more lexical queries than the fixed
        retrieval budget can admit.  Products seen on an earlier turn remain
        eligible for current-state reranking, but receive no inherited source
        score.  A real task reset starts a fresh history.
        """
        previous_version, previous_ids = self._candidate_history.get(
            session_id,
            (task_version, ()),
        )
        if previous_version != task_version:
            previous_ids = ()

        seen = {
            str(candidate.get("parent_asin", "")).strip()
            for candidate in candidates
            if str(candidate.get("parent_asin", "")).strip()
        }
        for parent_asin in previous_ids:
            if parent_asin in seen or parent_asin not in self.store:
                continue
            payload = self._product_payload(parent_asin)
            payload["source_scores"] = {}
            payload["source_ranks"] = {}
            candidates.append(payload)
            seen.add(parent_asin)

        current_ids = tuple(
            str(candidate["parent_asin"])
            for candidate in candidates
            if candidate.get("parent_asin")
        )
        history_limit = self.candidate_limit * 2
        retained_ids = tuple(dict.fromkeys((*previous_ids, *current_ids)))[:history_limit]
        self._candidate_history[session_id] = (task_version, retained_ids)
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
        scoped_override = False
        if parsed is not None:
            parser_requested_reset = parsed.reset_task
            parsed = _scope_override_update(
                parsed,
                self.memory.get_state(session_id),
            )
            scoped_override = parser_requested_reset and not parsed.reset_task
            try:
                self.memory.apply_update(parsed)
            except (TypeError, ValueError):
                pass
            else:
                retrieval_state = self.memory.get_retrieval_state(session_id)

        retrieval_context = _retrieval_context(user_message, retrieval_state)
        pool = self.retriever.retrieve(
            retrieval_context,
            limit=max(min(self.candidate_limit, RETRIEVAL_POOL_SIZE), top_k),
        )
        ranking_context = search_context_from_state(retrieval_state)
        snippet_hits = self.snippets.search(
            ranking_context.get("hard_constraints") or [],
            message=user_message,
            limit=SNIPPET_POOL_SIZE,
        )
        payloads = self._candidate_payloads(pool)
        seen = {item["parent_asin"] for item in payloads}
        for hit in snippet_hits:
            asin = hit["parent_asin"]
            payload = self._product_payload(asin)
            if asin in seen:
                existing = next(item for item in payloads if item["parent_asin"] == asin)
                scores = dict(existing.get("source_scores") or {})
                ranks = dict(existing.get("source_ranks") or {})
                scores.update(hit.get("source_scores") or {})
                ranks.update(hit.get("source_ranks") or {})
                existing["source_scores"] = scores
                existing["source_ranks"] = ranks
                continue
            payload["source_scores"] = dict(hit.get("source_scores") or {})
            payload["source_ranks"] = dict(hit.get("source_ranks") or {})
            payloads.append(payload)
            seen.add(asin)

        payloads = self._admit_candidate_history(
            session_id,
            self.memory.get_state(session_id).task_version,
            payloads,
        )

        ranking = rank_candidates(ranking_context, payloads, top_k=top_k)
        recommendations = [
            {"parent_asin": item.parent_asin, "score": item.final_score}
            for item in ranking.items
        ]

        state = self.memory.get_state(session_id)
        has_specified = any(
            state.status_for(attribute) is AttributeStatus.SPECIFIED
            for attribute in ("material", "color", "feature", "style")
        )
        clarify_order = (
            ("other", "feature", "style", "use_case", "size", "brand", "budget", "material", "color")
            if has_specified
            else CLARIFY_ORDER
        )
        clarify_candidates = [
            attribute
            for attribute in clarify_order
            if attribute == "other" or state.status_for(attribute) is AttributeStatus.UNKNOWN
        ]
        # A same-task override can invalidate the previous dialogue path. Ask a
        # broad follow-up once so undisclosed constraints are not skipped merely
        # because "other" was asked before the preference changed.
        ask_attribute = (
            "other"
            if scoped_override
            else self.memory.get_or_record_asked_attribute(
                session_id,
                turn,
                clarify_candidates,
            )
        )
        if ask_attribute == "other":
            message = "Is there anything else that matters for this product?"
        elif ask_attribute:
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
