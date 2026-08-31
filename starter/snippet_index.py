from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from retrieval.product_store import ProductStore

SPACE_RE = re.compile(r"\s+")
SPLIT_RE = re.compile(r"; |\n")
MIN_EXACT_CHARS = 8
MIN_NEEDLE_CHARS = 11


def fold_text(value: object) -> str:
    """Lowercase, drop colons, and collapse whitespace so 'Color: Black' matches 'color black'."""
    if value is None:
        return ""
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            parts.append(f"{key} {item}")
            parts.append(f"{key}: {item}")
        text = " ".join(parts)
    elif isinstance(value, (list, tuple)):
        text = " ".join(str(item) for item in value)
    else:
        text = str(value)
    return SPACE_RE.sub(" ", text.lower().replace(":", " ")).strip()


def flatten_phrases(value: object) -> list[str]:
    phrases: list[str] = []
    if value is None:
        return phrases
    if isinstance(value, dict):
        for item in value.values():
            phrases.extend(flatten_phrases(item))
        return phrases
    if isinstance(value, (list, tuple, set)):
        for item in value:
            phrases.extend(flatten_phrases(item))
        return phrases
    if isinstance(value, bool):
        return phrases
    if isinstance(value, (int, float)):
        text = str(int(value) if float(value).is_integer() else value)
        if text:
            phrases.append(text)
        return phrases
    text = str(value).strip()
    if text:
        phrases.append(text)
    return phrases


def fold_variants(text: str) -> list[str]:
    folded = fold_text(text)
    if not folded:
        return []
    variants = [folded]
    if "grey" in folded:
        variants.append(folded.replace("grey", "gray"))
    if "gray" in folded:
        variants.append(folded.replace("gray", "grey"))
    if folded.startswith("color "):
        variants.append(folded[6:])
        variants.extend(fold_variants(folded[6:]))
    seen: set[str] = set()
    unique: list[str] = []
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _record_snippets(row: Mapping[str, Any]) -> list[str]:
    snippets: list[str] = []
    features = row.get("features") or []
    if isinstance(features, (list, tuple)):
        snippets.extend(str(item) for item in features if item not in (None, ""))
    elif features not in (None, ""):
        snippets.append(str(features))
    details = row.get("details") or {}
    if isinstance(details, Mapping):
        for key, value in details.items():
            if value in (None, ""):
                continue
            snippets.append(f"{key}: {value}")
            snippets.append(f"{key} {value}")
    extra: list[str] = []
    for snippet in snippets:
        extra.extend(part.strip() for part in SPLIT_RE.split(snippet) if part.strip())
    snippets.extend(extra)
    return snippets


class SnippetIndex:
    """Exact catalog lookup for disclosed requirement snippets and raw user text."""

    def __init__(self, store: ProductStore) -> None:
        self.store = store
        self.folded: dict[str, str] = {
            asin: fold_text("\n".join(product.fields))
            for asin, product in store.products.items()
        }
        self.exact: dict[str, set[str]] = defaultdict(set)
        self._index_store()
        self.needles: list[tuple[str, tuple[str, ...]]] = [
            (snippet, tuple(sorted(asins)))
            for snippet, asins in self.exact.items()
            if MIN_NEEDLE_CHARS <= len(snippet) <= 180 and 1 <= len(asins) <= 80
        ]

    def _add(self, asin: str, snippet: str) -> None:
        for variant in fold_variants(snippet):
            if len(variant) >= MIN_EXACT_CHARS:
                self.exact[variant].add(asin)

    def _index_store(self) -> None:
        for asin, product in self.store.products.items():
            if product.raw:
                for snippet in _record_snippets(product.raw):
                    self._add(asin, snippet)
                continue
            for field in product.fields:
                self._add(asin, field)
                for part in SPLIT_RE.split(field):
                    if part.strip():
                        self._add(asin, part.strip())

    def search(
        self,
        snippets: Iterable[str],
        message: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        matched: dict[str, list[tuple[int, int]]] = defaultdict(list)

        def add_match(asin: str, n_asins: int, length: int) -> None:
            matched[asin].append((max(n_asins, 1), length))

        for snippet in snippets:
            for variant in fold_variants(snippet):
                if len(variant) < MIN_EXACT_CHARS:
                    continue
                asins = self.exact.get(variant)
                if asins and len(asins) <= 600:
                    n_asins = len(asins)
                    length = len(variant)
                    for asin in asins:
                        add_match(asin, n_asins, length)
                    continue
                if asins:
                    continue
                if len(variant) >= MIN_NEEDLE_CHARS:
                    for asin, haystack in self.folded.items():
                        if variant in haystack:
                            add_match(asin, 200, len(variant))

        hay = fold_text(message)
        if len(hay) >= MIN_NEEDLE_CHARS:
            for needle, asins in self.needles:
                if needle in hay:
                    n_asins = len(asins)
                    length = len(needle)
                    for asin in asins:
                        add_match(asin, n_asins, length)

        scored: list[tuple[float, str]] = []
        for asin, parts in matched.items():
            rarest = min(n_asins for n_asins, _length in parts)
            n_match = len(parts)
            length_sum = sum(length for _n, length in parts)
            score = 50000.0 / rarest + 20.0 * n_match + length_sum / 20.0
            scored.append((score, asin))
        scored.sort(key=lambda item: (-item[0], item[1]))

        results: list[dict[str, Any]] = []
        for rank, (score, asin) in enumerate(scored[:limit], start=1):
            results.append({
                "parent_asin": asin,
                "source_scores": {"snippet": score},
                "source_ranks": {"snippet": rank},
            })
        return results
