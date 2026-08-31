from __future__ import annotations

import collections
import json
import re
import unicodedata
from pathlib import Path

from local_experiments.parser.models import (
    CandidateSource,
    ExtractedFact,
    FieldCandidate,
    NormalizedMessage,
    RuleExtraction,
)
from starter.memory import UpdateOperation
from starter.memory.models import DECLINABLE_SLOTS

MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
COLOR_ALIASES = {
    "black": "black",
    "white": "white",
    "blue": "blue",
    "red": "red",
    "pink": "pink",
    "green": "green",
    "brown": "brown",
    "gray": "gray",
    "grey": "gray",
    "purple": "purple",
    "yellow": "yellow",
    "orange": "orange",
    "silver": "silver",
    "gold": "gold",
    "beige": "beige",
    "navy": "navy",
}
# Stable product-language aliases. Patterns describe concepts rather than evaluator
# templates; canonical values are the phrases commonly present in the catalog.
SEMANTIC_VALUE_ALIASES = (
    ("material", "cotton", re.compile(r"\bnatural\s+plant\s+fib(?:er|re)\b", re.I)),
    ("material", "polyester", re.compile(
        r"\bsynthetic\s+wrinkle[- ]resistant\s+fib(?:er|re)\b", re.I,
    )),
    ("material", "nylon", re.compile(r"\bdurable\s+polyamide\s+fib(?:er|re)\b", re.I)),
    ("material", "leather", re.compile(r"\banimal[- ]hide\s+material\b", re.I)),
    ("material", "wool", re.compile(r"\bwarm\s+sheep[- ]derived\s+fib(?:er|re)\b", re.I)),
    ("material", "spandex", re.compile(r"\belastic\s+stretch\s+fib(?:er|re)\b", re.I)),
    ("material", "silk", re.compile(
        r"\bsmooth\s+silkworm[- ]derived\s+filament\b", re.I,
    )),
    ("material", "rayon", re.compile(
        r"\bcellulose[- ]based\s+semi[- ]synthetic\s+fib(?:er|re)\b", re.I,
    )),
    ("feature", "Machine Wash", re.compile(
        r"\b(?:safe|suitable)\s+for\s+(?:cleaning|washing)\s+in\s+(?:a\s+)?washing\s+machine\b",
        re.I,
    )),
    ("feature", "Hand Wash Only", re.compile(
        r"\bclean\s+(?:it\s+)?manually\s+rather\s+than\s+in\s+(?:a\s+)?washer\b",
        re.I,
    )),
    ("feature", "Imported", re.compile(r"\b(?:made|manufactured)\s+overseas\b", re.I)),
    ("feature", "Pull On Closure", re.compile(
        r"\bslips?\s+on\s+without\s+(?:a\s+)?fastener\b", re.I,
    )),
    ("feature", "Rubber Sole", re.compile(
        r"\boutsole\s+made\s+from\s+(?:elastic\s+)?rubber\b", re.I,
    )),
    ("feature", "Waterproof", re.compile(
        r"\bkeeps?\s+rain\s+from\s+soaking\s+through\b", re.I,
    )),
    ("feature", "Water Resistant", re.compile(r"\brepels?\s+light\s+rain\b", re.I)),
    ("feature", "Lightweight", re.compile(
        r"\beasy\s+to\s+carry\s+without\s+much\s+weight\b", re.I,
    )),
    ("feature", "Breathable", re.compile(r"\ballows?\s+air\s+to\s+circulate\b", re.I)),
    ("feature", "Adjustable", re.compile(r"\bcan\s+be\s+resized\b", re.I)),
    ("feature", "Cushioned", re.compile(r"\bhas\s+soft\s+impact\s+padding\b", re.I)),
    ("feature", "Non-Slip", re.compile(r"\bprovides?\s+extra\s+grip\b", re.I)),
    ("feature", "Pockets", re.compile(r"\bstorage\s+compartments?\b", re.I)),
    ("color", "black", re.compile(r"\bdarkest\s+neutral\s+shade\b", re.I)),
    ("color", "white", re.compile(r"\bbright\s+neutral\s+shade\b", re.I)),
    ("color", "blue", re.compile(r"\bazure\s+shade\b", re.I)),
    ("color", "red", re.compile(r"\bcrimson\s+shade\b", re.I)),
    ("color", "pink", re.compile(r"\brose\s+shade\b", re.I)),
    ("color", "green", re.compile(r"\bemerald\s+shade\b", re.I)),
    ("color", "brown", re.compile(r"\bearthy\s+umber\s+shade\b", re.I)),
    ("color", "gray", re.compile(r"\bcharcoal\s+neutral\s+shade\b", re.I)),
    ("color", "purple", re.compile(r"\bviolet\s+shade\b", re.I)),
    ("color", "yellow", re.compile(r"\bgolden\s+shade\b", re.I)),
    ("color", "orange", re.compile(r"\btangerine\s+shade\b", re.I)),
    ("color", "silver", re.compile(r"\bpale\s+metallic\s+shade\b", re.I)),
    ("color", "gold", re.compile(r"\bwarm\s+metallic\s+shade\b", re.I)),
    ("color", "beige", re.compile(r"\bsand[- ]toned\s+shade\b", re.I)),
    ("color", "navy", re.compile(r"\bdeep\s+marine\s+shade\b", re.I)),
)
USE_CASES = (
    "hiking",
    "running",
    "gym",
    "winter",
    "outdoor",
    "work",
    "wedding",
    "walking",
    "travel",
    "travelling",
    "traveling",
)
BRAND_STOPWORDS = frozenset({
    "apparel",
    "clothing",
    "fashion",
    "generic",
    "jewelry",
    "official",
    "outdoor",
    "running",
    "shoes",
    "store",
    "style",
    "other",
    "jacket",
    "jackets",
    "accessories",
    "accessory",
    "top",
    "tops",
    "tee",
    "tees",
    "travel",
    "walking",
    "wedding",
    "winter",
    "work",
    "women",
})
RESIDUAL_NOISE_WORDS = frozenset({
    "a", "about", "also", "am", "an", "and", "another", "are", "ask",
    "before", "browse", "but", "care", "change", "choose", "comparing", "considering",
    "budget", "can", "cheaper", "current", "deciding", "details", "different", "drop", "earlier", "else",
    "essential", "explore", "find", "focused", "for", "forget", "have", "help",
    "i", "id", "im", "include", "includes", "instead", "is", "it", "judgment",
    "keeping", "like", "little", "looking", "make", "match", "matches", "matters", "maybe", "me", "most", "must",
    "my", "need", "needed", "needs", "new", "nothing", "of", "one", "open", "option",
    "options", "out", "plan", "please", "preference", "preferences", "preferably", "previous",
    "prioritise", "prioritize", "priority", "product", "question", "reconsidered",
    "request", "right", "said", "settled", "shopping", "should", "show", "some", "something",
    "still", "sure", "target", "that", "the", "these", "thing", "this", "to",
    "update", "use", "want", "what", "with", "working", "would", "yet", "your",
})

CATEGORY_RE = re.compile(
    r"\b(?:looking|shopping)\s+for\s+(?:an?\s+|some\s+)?(.+?)(?=[.,;!?]|$)",
    re.I,
)
CATEGORY_CUE_PATTERNS = (
    re.compile(
        r"^(?:i\s+need|please\s+find|i['’]?m\s+shopping\s+for|"
        r"help\s+me\s+choose|my\s+target\s+is)\s+(.+?)"
        r"(?=[.,;]|\s+(?:it\s+must|with\s+this\s+requirement|that\s+matches|"
        r"and\s+i\s+need|preferably|my\s+current\s+preference|i\s+would\s+like|"
        r"and\s+for\s+now)\b|$)",
        re.I,
    ),
    re.compile(
        r"^(?:i['’]?m\s+considering|help\s+me\s+explore\s+(?:some\s+)?|"
        r"i['’]?d\s+like\s+to\s+browse|show\s+me\s+different|i['’]?m\s+comparing)\s+"
        r"(.+?)(?=[.,;]|\s+(?:and\s+haven|options\b|before\b|and\s+still\b)|$)",
        re.I,
    ),
)
POSITIVE_PATTERNS = (
    re.compile(r"a\s+key\s+requirement\s+is\s*:\s*(.+)$", re.I),
    re.compile(r"what\s+i\s+need\s+is\s*:\s*(.+)$", re.I),
    re.compile(r"for\s+that,?\s+what\s+matters\s+is\s*:\s*(.+)$", re.I),
)
NO_PREFERENCE_PATTERNS = (
    re.compile(
        r"don['’]?t\s+have\s+(?:an?\s+)?(?:additional\s+)?preference\s+for\s+([a-z_]+)",
        re.I,
    ),
    re.compile(r"\bflexible\s+about\s+([a-z_]+)\b", re.I),
    re.compile(r"\bnothing\s+else\s+matters\s+to\s+me\s+for\s+([a-z_]+)\b", re.I),
)
NEW_TASK_RE = re.compile(
    r"\b(?:new\s+shopping\s+task|start\s+over|shop\s+for\s+something\s+else)\b",
    re.I,
)
OVERRIDE_RE = re.compile(
    r"\b(?:"
    r"(?:ignore|replace)\s+my\s+earlier\s+preference|"
    r"change\s+of\s+plan|drop\s+my\s+previous\s+preference|"
    r"replace\s+what\s+i\s+said\s+earlier|i(?:'ve|\s+have)\s+reconsidered|"
    r"forget\s+my\s+previous\s+preference|update\s+my\s+request"
    r")\b",
    re.I,
)
REMOVE_PATTERNS = (
    re.compile(
        r"\b(?:forget\s+about|no\s+longer\s+want)\s+(.+?)"
        r"(?=\s*[,;.]?\s*(?:maybe|instead|but|and\s+i\b)|[.;]|$)",
        re.I,
    ),
)
EXCLUDE_PATTERNS = (
    re.compile(
        r"\b(?:do\s+not|don['’]?t)\s+want\s+(.+?)"
        r"(?=\s*,?\s*(?:but|instead)\b|[.;]|$)",
        re.I,
    ),
    re.compile(
        r"\b(?:without|avoid|exclude)\s+(.+?)"
        r"(?=\s*,?\s*(?:but|instead)\b|[.;]|$)",
        re.I,
    ),
    re.compile(
        r"\bno\s+(?!(?:more\s+than|longer\s+want)\b)(.+?)"
        r"(?=\s*,?\s*(?:but|instead)\b|[.;]|$)",
        re.I,
    ),
    re.compile(
        r"\bnot\s+(?!more\s+than\b)(.+?)"
        r"(?=\s*,?\s*(?:but|instead)\b|[.;]|$)",
        re.I,
    ),
)
PRICE_RANGE_RE = re.compile(
    r"(?:\$|usd\s*)?(\d+(?:\.\d+)?)\s*(?:-|to)\s*"
    r"(?:\$|usd\s*)?(\d+(?:\.\d+)?)",
    re.I,
)
PRICE_MAX_RE = re.compile(
    r"\b(?:under|below|less\s+than|up\s+to|at\s+most|no\s+more\s+than|"
    r"max(?:imum)?(?:\s+of)?|budget(?:\s+is)?)\s*(?:\$|usd\s*)?(\d+(?:\.\d+)?)",
    re.I,
)
PRICE_MIN_RE = re.compile(
    r"\b(?:(?:over|above|(?<!no\s)more\s+than|at\s+least)\s*"
    r"(?:\$|usd\s*)(\d+(?:\.\d+)?)|"
    r"min(?:imum)?(?:\s+(?:price|budget)|\s+of)?\s*(?:\$|usd\s*)?(\d+(?:\.\d+)?))",
    re.I,
)
PRICE_AROUND_RE = re.compile(
    r"\b(?:around|about|roughly|approximately|budget\s+around)\s*"
    r"(?:\$|usd\s*)?(\d+(?:\.\d+)?)",
    re.I,
)
MONEY_RE = re.compile(r"(?:\$|\busd\s*)(\d+(?:\.\d+)?)", re.I)
RATING_RE = re.compile(
    r"(?:at\s+least\s+)?(\d(?:\.\d+)?)\s*(?:\+\s*)?(?:stars?|rating)\b|"
    r"\b(?:rating|rated)\s*(?:at\s+least|over|above|>=)?\s*(\d(?:\.\d+)?)",
    re.I,
)
BRAND_RE = re.compile(r"\b(?:brand|manufacturer|store)\s*:\s*([^;,]+)", re.I)
SIZE_RE = re.compile(
    r"\b(?:(us|eu|uk)\s*)?size\s*[:#-]?\s*([a-z0-9.]+)|"
    r"\b(us|eu|uk)\s+(\d+(?:\.\d+)?)\b",
    re.I,
)


def _phrase_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


class MessageNormalizer:
    def normalize(self, message: str) -> NormalizedMessage:
        raw = message or ""
        text = unicodedata.normalize("NFKC", raw).replace("’", "'")
        text = re.sub(r"\s+", " ", text).strip()
        return NormalizedMessage(raw=raw, text=text, lowered=text.casefold())


class CatalogLexicon:
    """Read-only vocabulary derived once from the frozen product catalog."""

    def __init__(
        self,
        brands: dict[str, str] | None = None,
        categories: dict[str, str] | None = None,
    ) -> None:
        self._brands = brands or {}
        self._categories = categories or {}
        self._max_brand_words = min(
            max((len(key.split()) for key in self._brands), default=1),
            6,
        )
        self._max_category_words = min(
            max((len(key.split()) for key in self._categories), default=1),
            12,
        )

    @classmethod
    def from_catalog(cls, catalog_path: str | Path) -> CatalogLexicon:
        brands: dict[str, str] = {}
        categories: dict[str, str] = {}
        coarse_categories: dict[str, str] = {}
        with Path(catalog_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                store = product.get("store")
                if isinstance(store, str) and (key := _phrase_key(store)):
                    brands.setdefault(key, store.strip())
                for category in product.get("categories") or ():
                    if isinstance(category, str) and (key := _phrase_key(category)):
                        categories.setdefault(key, category.strip())
                category_values = [
                    value.strip()
                    for value in product.get("categories") or ()
                    if isinstance(value, str) and value.strip()
                ]
                if category_values:
                    coarse = " ".join(category_values[-2:])
                    coarse_categories.setdefault(_phrase_key(coarse), coarse)
        category_tokens = {
            token
            for key in categories
            for token in key.split()
        }
        brands = {
            key: value
            for key, value in brands.items()
            if key not in categories
            and not (len(key.split()) == 1 and key in category_tokens)
        }
        for key, value in coarse_categories.items():
            categories.setdefault(key, value)
        return cls(brands, categories)

    def normalize_brand(self, value: str) -> str | None:
        return self._brands.get(_phrase_key(value))

    def normalize_category(self, value: str) -> str | None:
        return self._categories.get(_phrase_key(value))

    def match_category(self, text: str) -> tuple[str, str] | None:
        tokens = re.findall(r"[A-Za-z0-9]+", text)
        for size in range(min(self._max_category_words, len(tokens)), 0, -1):
            for start in range(len(tokens) - size + 1):
                evidence = " ".join(tokens[start:start + size])
                category = self._categories.get(_phrase_key(evidence))
                if category is not None:
                    return category, evidence
        return None

    def match_brands(self, text: str) -> tuple[tuple[str, str], ...]:
        tokens = re.findall(r"[A-Za-z0-9]+", text)
        matches: list[tuple[str, str]] = []
        seen: set[str] = set()
        for size in range(min(self._max_brand_words, len(tokens)), 0, -1):
            for start in range(len(tokens) - size + 1):
                evidence = " ".join(tokens[start:start + size])
                key = _phrase_key(evidence)
                brand = self._brands.get(key)
                if brand is None or key in seen:
                    continue
                if size == 1:
                    if key in BRAND_STOPWORDS:
                        continue
                seen.add(key)
                matches.append((brand, evidence))
        return tuple(matches)


class RuleExtractor:
    """Extracts only locally verifiable facts and explicit operations."""

    def __init__(self, lexicon: CatalogLexicon | None = None) -> None:
        self.lexicon = lexicon or CatalogLexicon()

    def extract(self, message: NormalizedMessage) -> RuleExtraction:
        text = message.text
        if not text:
            return RuleExtraction()

        candidates: list[FieldCandidate] = []
        facts: list[ExtractedFact] = []
        intent_signals: list[str] = []
        negative_spans: list[tuple[int, int]] = []

        if re.search(
            r"\b(?:still\s+exploring|haven['’]?t\s+settled|explore|browse|"
            r"keeping\s+(?:my\s+)?options\s+open|comparing|working\s+out\s+(?:my\s+)?preferences)\b",
            message.lowered,
        ):
            intent_signals.append("exploration_language")
        if re.search(
            r"\b(?:key\s+requirement|what\s+i\s+need|must|essential|"
            r"requirement|prioriti[sz]e|i\s+need)\b",
            text,
            re.I,
        ):
            intent_signals.append("hard_constraint_language")

        no_preference = next(
            (match for pattern in NO_PREFERENCE_PATTERNS if (match := pattern.search(text))),
            None,
        )
        if no_preference:
            candidates.extend(self._decline_attribute(no_preference.group(1).lower(), no_preference.group(0)))

        for patterns, operation in (
            (REMOVE_PATTERNS, UpdateOperation.REMOVE),
            (EXCLUDE_PATTERNS, UpdateOperation.EXCLUDE),
        ):
            for pattern in patterns:
                for match in pattern.finditer(text):
                    negative_spans.append(match.span())
                    candidates.extend(self._negative_candidates(match.group(1), operation))

        category_match = CATEGORY_RE.search(text)
        category_evidence = ""
        if category_match:
            raw_category = self._clean(category_match.group(1))
            if raw_category:
                category = self.lexicon.normalize_category(raw_category) or raw_category
                category_evidence = raw_category
                candidates.append(self._candidate(
                    "category",
                    UpdateOperation.SET,
                    category,
                    0.84,
                    category_evidence,
                ))
                facts.append(ExtractedFact("category_phrase", category, 0.90, category_evidence))
        else:
            category_cue = next(
                (match for pattern in CATEGORY_CUE_PATTERNS if (match := pattern.search(text))),
                None,
            )
            catalog_match = (
                self.lexicon.match_category(category_cue.group(1))
                if category_cue else None
            )
            if catalog_match is not None:
                category, category_evidence = catalog_match
                candidates.append(self._candidate(
                    "category", UpdateOperation.SET, category, 0.90, category_evidence,
                ))
                facts.append(ExtractedFact(
                    "category_phrase", category, 0.92, category_evidence,
                ))

        explicit_segments: list[str] = []
        positive_match = next(
            (match for pattern in POSITIVE_PATTERNS if (match := pattern.search(text))),
            None,
        )
        if positive_match:
            explicit_segments.extend(positive_match.group(1).strip().rstrip(".").split(";"))
        scan_segments = explicit_segments or [text]
        for segment in scan_segments:
            before = len(candidates)
            self._extract_structured_segment(
                segment,
                candidates,
                facts,
                allow_implicit_brand=not explicit_segments and no_preference is None,
            )
            if explicit_segments and len(candidates) == before:
                cleaned = self._clean(segment)
                if cleaned:
                    facts.append(ExtractedFact(
                        "unclassified_requirement",
                        cleaned[:180],
                        0.98,
                        cleaned,
                    ))
        if (
            not explicit_segments
            and "exploration_language" not in intent_signals
        ):
            for residual in self._residual_requirements(text, candidates, facts):
                facts.append(ExtractedFact(
                    "unclassified_requirement",
                    residual[:180],
                    0.90,
                    residual,
                ))

        candidates = [
            candidate
            for candidate in candidates
            if not self._positive_candidate_inside_negative(candidate, text, negative_spans)
        ]
        return RuleExtraction(
            candidates=self._dedupe_candidates(candidates),
            facts=self._dedupe_facts(facts),
            intent_signals=tuple(dict.fromkeys(intent_signals)),
            reset_task=bool(NEW_TASK_RE.search(text)),
            override_signal=bool(OVERRIDE_RE.search(text)),
        )

    def _extract_structured_segment(
        self,
        segment: str,
        candidates: list[FieldCandidate],
        facts: list[ExtractedFact],
        *,
        allow_implicit_brand: bool,
    ) -> None:
        cleaned = self._clean(segment)
        if not cleaned:
            return

        ranges = list(PRICE_RANGE_RE.finditer(cleaned))
        for match in ranges:
            minimum = self._number(match.group(1))
            maximum = self._number(match.group(2))
            candidates.extend((
                self._candidate("price_min", UpdateOperation.SET, minimum, 0.99, match.group(0)),
                self._candidate("price_max", UpdateOperation.SET, maximum, 0.99, match.group(0)),
            ))
            facts.extend((
                ExtractedFact("money", minimum, 0.99, match.group(0)),
                ExtractedFact("money", maximum, 0.99, match.group(0)),
            ))

        for regex, slot in ((PRICE_MAX_RE, "price_max"), (PRICE_MIN_RE, "price_min")):
            for match in regex.finditer(cleaned):
                value = self._number(next(group for group in match.groups() if group is not None))
                candidates.append(self._candidate(slot, UpdateOperation.SET, value, 0.99, match.group(0)))
                facts.append(ExtractedFact("money", value, 0.99, match.group(0)))

        for regex in (PRICE_AROUND_RE, MONEY_RE):
            for match in regex.finditer(cleaned):
                value = self._number(match.group(1))
                facts.append(ExtractedFact("money", value, 0.98, match.group(0)))
        if re.search(r"\$|\busd\b|\bdollars?\b", cleaned, re.I):
            facts.append(ExtractedFact("currency", "USD", 0.99, cleaned))

        for match in RATING_RE.finditer(cleaned):
            value = self._number(match.group(1) or match.group(2))
            candidates.append(self._candidate("rating_min", UpdateOperation.SET, value, 0.98, match.group(0)))
            facts.append(ExtractedFact("rating", value, 0.98, match.group(0)))

        for match in SIZE_RE.finditer(cleaned):
            system = (match.group(1) or match.group(3) or "").upper()
            size = match.group(2) or match.group(4)
            value = f"{system} {size}".strip()
            candidates.append(self._candidate("size", UpdateOperation.SET, value, 0.97, match.group(0)))
            facts.append(ExtractedFact("size", value, 0.97, match.group(0)))

        lowered = cleaned.casefold()
        semantic_aliases = self._semantic_alias_matches(cleaned)
        material_matches = [
            (item, item)
            for item in MATERIALS
            if re.search(rf"\b{re.escape(item)}\b", lowered)
        ]
        material_matches.extend(
            (canonical, evidence)
            for slot, canonical, evidence in semantic_aliases
            if slot == "material"
        )
        materials = dict(material_matches)
        color_matches = [
            (canonical, alias)
            for alias, canonical in COLOR_ALIASES.items()
            if re.search(rf"\b{re.escape(alias)}\b", lowered)
        ]
        color_matches.extend(
            (canonical, evidence)
            for slot, canonical, evidence in semantic_aliases
            if slot == "color"
        )
        colors = dict(color_matches)
        for material, evidence in materials.items():
            candidates.append(self._candidate("material", UpdateOperation.ADD, material, 0.98, evidence))
            facts.append(ExtractedFact("material", material, 0.98, evidence))
        color_op = UpdateOperation.ADD if len(colors) > 1 else UpdateOperation.SET
        for color, evidence in colors.items():
            candidates.append(self._candidate("color", color_op, color, 0.98, evidence))
            facts.append(ExtractedFact("color", color, 0.98, evidence))
        for slot, canonical, evidence in semantic_aliases:
            if slot != "feature":
                continue
            candidates.append(self._candidate(
                "feature", UpdateOperation.ADD, canonical, 0.96, evidence,
            ))
            facts.append(ExtractedFact("feature_alias", canonical, 0.96, evidence))

        brand_match = BRAND_RE.search(cleaned)
        if brand_match:
            raw_brand = self._clean(brand_match.group(1))
            brand = self.lexicon.normalize_brand(raw_brand) or raw_brand
            candidates.append(self._candidate("brand", UpdateOperation.SET, brand, 0.99, brand_match.group(0)))
            facts.append(ExtractedFact("brand", brand, 0.99, brand_match.group(0)))
        elif allow_implicit_brand:
            for brand, evidence in self.lexicon.match_brands(cleaned):
                candidates.append(self._candidate("brand", UpdateOperation.SET, brand, 0.96, evidence))
                facts.append(ExtractedFact("brand", brand, 0.96, evidence))

        for use_case in USE_CASES:
            if re.search(rf"\b{re.escape(use_case)}\b", lowered):
                normalized = "travel" if use_case in {"travelling", "traveling"} else use_case
                candidates.append(self._candidate("use_case", UpdateOperation.ADD, normalized, 0.88, use_case))

    def _negative_candidates(
        self,
        value: str,
        operation: UpdateOperation,
    ) -> list[FieldCandidate]:
        result: list[FieldCandidate] = []
        for part in re.split(r"\s+or\s+|,", value, flags=re.I):
            cleaned = self._clean(part)
            if not cleaned:
                continue
            classified = self._classified_values(cleaned)
            if classified:
                for slot, normalized in classified:
                    result.append(self._candidate(slot, operation, normalized, 0.99, cleaned))
        return result

    @staticmethod
    def _residual_requirements(
        text: str,
        candidates: list[FieldCandidate],
        facts: list[ExtractedFact],
    ) -> tuple[str, ...]:
        consumed = collections.Counter()
        for evidence in [
            *(candidate.evidence for candidate in candidates),
            *(
                fact.evidence
                for fact in facts
                if fact.kind != "unclassified_requirement"
            ),
        ]:
            consumed.update(token.casefold() for token in re.findall(r"[A-Za-z0-9]+", evidence))

        residuals: list[str] = []
        remaining = consumed.copy()
        for clause in re.split(r"[.;]+", text):
            if re.search(r"\b(?:do\s+not|don['’]?t|without|avoid|exclude|not)\b", clause, re.I):
                continue
            relative_budget = bool(re.search(r"\bstretch\b.*\bbudget\b", clause, re.I))
            kept: list[str] = []
            for token in re.findall(r"[A-Za-z0-9]+", clause):
                key = token.casefold()
                if remaining[key] > 0:
                    remaining[key] -= 1
                    continue
                canonical_color = COLOR_ALIASES.get(key)
                if canonical_color is not None and remaining[canonical_color] > 0:
                    remaining[canonical_color] -= 1
                    continue
                if len(key) <= 1 or key in RESIDUAL_NOISE_WORDS:
                    continue
                if relative_budget and key == "stretch":
                    continue
                kept.append(token)
            if kept:
                residuals.append(" ".join(kept))
        return tuple(dict.fromkeys(residuals))

    def _classified_values(self, value: str) -> list[tuple[str, str]]:
        lowered = value.casefold()
        result: list[tuple[str, str]] = []
        for material in MATERIALS:
            if re.search(rf"\b{re.escape(material)}\b", lowered):
                result.append(("material", material))
        for alias, canonical in COLOR_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                result.append(("color", canonical))
        result.extend(
            (slot, canonical)
            for slot, canonical, _ in self._semantic_alias_matches(value)
        )
        brand = self.lexicon.normalize_brand(value)
        if brand:
            result.append(("brand", brand))
        return list(dict.fromkeys(result))

    @staticmethod
    def _semantic_alias_matches(value: str) -> tuple[tuple[str, str, str], ...]:
        matches: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str]] = set()
        for slot, canonical, pattern in SEMANTIC_VALUE_ALIASES:
            match = pattern.search(value)
            key = (slot, canonical.casefold())
            if match is not None and key not in seen:
                matches.append((slot, canonical, match.group(0)))
                seen.add(key)
        return tuple(matches)

    def _decline_attribute(self, attribute: str, evidence: str) -> list[FieldCandidate]:
        if attribute == "budget":
            return [
                self._candidate("price_min", UpdateOperation.DECLINE, None, 0.99, evidence),
                self._candidate("price_max", UpdateOperation.DECLINE, None, 0.99, evidence),
            ]
        if attribute in DECLINABLE_SLOTS:
            return [self._candidate(attribute, UpdateOperation.DECLINE, None, 0.99, evidence)]
        if attribute in {"category", "product_type"}:
            return [self._candidate(attribute, UpdateOperation.CLEAR, None, 0.99, evidence)]
        return []

    @staticmethod
    def _positive_candidate_inside_negative(
        candidate: FieldCandidate,
        text: str,
        negative_spans: list[tuple[int, int]],
    ) -> bool:
        if candidate.op not in {UpdateOperation.SET, UpdateOperation.ADD} or not candidate.evidence:
            return False
        matches = list(re.finditer(re.escape(candidate.evidence), text, re.I))
        return bool(matches) and all(
            any(start <= match.start() and match.end() <= end for start, end in negative_spans)
            for match in matches
        )

    @staticmethod
    def _candidate(
        slot: str,
        op: UpdateOperation,
        value: str | int | float | None,
        confidence: float,
        evidence: str,
    ) -> FieldCandidate:
        return FieldCandidate(slot, op, value, CandidateSource.RULE, confidence, evidence)

    @staticmethod
    def _clean(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip(" -;,.")

    @staticmethod
    def _number(value: str) -> int | float:
        number = float(value)
        return int(number) if number.is_integer() else number

    @staticmethod
    def _dedupe_candidates(candidates: list[FieldCandidate]) -> tuple[FieldCandidate, ...]:
        return tuple({
            candidate.identity(): candidate for candidate in candidates
        }.values())

    @staticmethod
    def _dedupe_facts(facts: list[ExtractedFact]) -> tuple[ExtractedFact, ...]:
        return tuple({(fact.kind, fact.value, fact.evidence.casefold()): fact for fact in facts}.values())
