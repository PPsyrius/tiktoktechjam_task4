"""Catalog-aligned tokens for Assignment 2. Base color/material/use-case lists
stay identical to the tuples Assignment 1 imports so Product fingerprints do not
change. Parser-only extras cover more catalog language without touching extraction.
"""
from __future__ import annotations

import re


# Shared with starter.catalog.feature_extractor. Do not expand these three tuples.
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown",
    "gray", "grey", "purple", "yellow", "orange", "navy", "beige",
    "silver", "gold",
)
MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex",
    "silk", "rayon", "fabric",
)
USE_CASES = (
    "hiking", "running", "gym", "winter", "outdoor", "work",
    "wedding", "walking", "travel",
)

SIZES = (
    "xxs", "xs", "xl", "xxl", "xxxl", "one size",
    "wide", "narrow", "plus size",
)
STYLES = (
    "athletic", "casual", "formal", "vintage", "slim", "regular",
    "loose", "crew", "v-neck", "polo", "hoodie", "classic",
)
PARSER_COLORS = COLORS + (
    "maroon", "khaki", "ivory", "teal", "burgundy", "charcoal",
    "cream", "olive", "coral", "turquoise", "tan",
)
PARSER_MATERIALS = MATERIALS + (
    "canvas", "denim", "suede", "mesh", "rubber", "vinyl",
    "linen", "fleece", "cashmere", "satin", "knit",
)
PARSER_USE_CASES = USE_CASES + (
    "casual", "athletic", "training", "office", "beach",
)
FEATURE_TOKENS = (
    "moisture wicking", "machine wash", "hand wash", "water resistant",
    "waterproof", "breathable", "lightweight", "cushioned", "adjustable",
    "stretchy", "stretch", "pockets", "pocket",
)
BRANDS = (
    "under armour", "new balance", "north face", "calvin klein",
    "ralph lauren", "tommy hilfiger", "timberland", "skechers",
    "champion", "columbia", "converse", "adidas", "reebok",
    "levis", "levi's", "asics", "nike", "puma", "vans",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def word_text(value: object) -> str:
    return " ".join(_TOKEN_RE.findall(str(value).casefold()))


def _sorted_phrases(phrases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(phrases, key=lambda item: (-len(item.split()), -len(item), item)))


def _match_phrases(text: str, phrases: tuple[str, ...]) -> list[str]:
    padded = f" {word_text(text)} "
    found: list[str] = []
    seen: set[str] = set()
    for phrase in _sorted_phrases(phrases):
        needle = f" {word_text(phrase)} "
        if needle in padded and phrase not in seen:
            seen.add(phrase)
            found.append(phrase)
    return found


def typed_catalog_matches(text: str) -> list[tuple[str, str]]:
    """Map a snippet onto catalog slots. Longer phrases win within a slot."""
    matches: list[tuple[str, str]] = []
    for brand in _match_phrases(text, BRANDS):
        matches.append(("brand", brand.replace("levi's", "levis")))
    for material in _match_phrases(text, PARSER_MATERIALS):
        matches.append(("material", material))
    for color in _match_phrases(text, PARSER_COLORS):
        canonical = "gray" if color == "grey" else color
        matches.append(("color", canonical))
    for size in _match_phrases(text, SIZES):
        matches.append(("size", size))
    for use_case in _match_phrases(text, PARSER_USE_CASES):
        matches.append(("use_case", use_case))
    for style in _match_phrases(text, STYLES):
        matches.append(("style", style))
    for feature in _match_phrases(text, FEATURE_TOKENS):
        canonical = "pockets" if feature == "pocket" else "stretch" if feature == "stretchy" else feature
        matches.append(("feature", canonical))
    return matches
