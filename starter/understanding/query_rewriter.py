from __future__ import annotations

import re
from typing import Any

from starter.memory.models import ParseUpdate, RetrievalState, UpdateOperation


WRAPPER_RE = re.compile(
    r"(?:i(?:'m| am) looking for|i(?:'m| am) after|i(?:'m| am) shopping for|"
    r"please find|show me|i want|i need(?!\s+is\b)|"
    r"a key requirement is|what matters is|what i(?: really)? need is|"
    r"for that,?\s+what matters is|i really need|"
    r"please (?:prioritise|prioritize|focus on)|"
    r"must be|has to be|have to be|"
    r"but i(?:'m| am) still exploring|i(?:'m| am) still exploring|"
    r"actually,|ignore my earlier preference|"
    r"forget (?:what i said|my earlier(?: preference)?|that)|"
    r"please use your judgment)",
    re.IGNORECASE,
)
SPACE_RE = re.compile(r"\s+")
MAX_QUERIES = 8


def _clean_query(text: str, limit: int = 180) -> str:
    stripped = WRAPPER_RE.sub(" ", text)
    stripped = SPACE_RE.sub(" ", stripped).strip(" -;,.\t\n:")
    return stripped[:limit].rstrip()


def _values(mapping: dict[str, Any] | None) -> list[str]:
    phrases: list[str] = []
    for value in (mapping or {}).values():
        items = value if isinstance(value, (list, tuple)) else [value]
        for item in items:
            if isinstance(item, str) and item.strip():
                phrases.append(item.strip())
    return phrases


def _context_payload(search_context: RetrievalState | dict[str, Any] | None) -> dict[str, Any]:
    if search_context is None:
        return {}
    if isinstance(search_context, RetrievalState):
        return search_context.to_dict()
    return dict(search_context)


def rewrite_queries(
    user_message: str,
    parsed: ParseUpdate | None = None,
    search_context: RetrievalState | dict[str, Any] | None = None,
    limit: int = MAX_QUERIES,
) -> tuple[str, ...]:
    """Build retrieval queries from this turn plus the active SearchContext summary."""
    queries: list[str] = []
    payload = _context_payload(search_context)

    category = None
    if parsed is not None:
        for update in parsed.updates:
            if update.slot == "category" and isinstance(update.value, str) and update.value.strip():
                category = update.value.strip()
                break
    if not category and isinstance(payload.get("category"), str):
        category = payload["category"].strip() or None

    tokens: list[str] = []
    features: list[str] = []
    if parsed is not None:
        for update in parsed.updates:
            if update.op in {UpdateOperation.EXCLUDE, UpdateOperation.DECLINE, UpdateOperation.CLEAR}:
                continue
            value = update.value
            if not isinstance(value, str) or not value.strip():
                continue
            text = value.strip()
            if update.slot == "feature" or len(text) >= 8:
                features.append(text)
            elif update.slot in {"color", "material", "brand", "size", "style", "use_case"}:
                tokens.append(text)

    tokens.extend(
        item for item in (
            *_values(payload.get("hard_constraints") if isinstance(payload.get("hard_constraints"), dict) else None),
            *_values(payload.get("soft_preferences") if isinstance(payload.get("soft_preferences"), dict) else None),
        )
        if len(item.split()) <= 3
    )

    if category:
        queries.append(category)
        unique_tokens = list(dict.fromkeys(tokens))
        if unique_tokens:
            queries.append(" ".join([category, *unique_tokens[:4]]))

    for feature in features:
        if len(feature) >= 8:
            queries.append(feature[:180])

    cleaned = _clean_query(user_message)
    if cleaned:
        queries.append(cleaned)

    retrieval_text = payload.get("product_type")
    if isinstance(retrieval_text, str) and retrieval_text.strip():
        queries.append(retrieval_text.strip())

    if isinstance(payload.get("category"), str) and payload["category"].strip():
        queries.append(payload["category"].strip())

    unique = list(dict.fromkeys(item for item in queries if item and item.strip()))
    return tuple(unique[: max(1, min(limit, MAX_QUERIES))])
