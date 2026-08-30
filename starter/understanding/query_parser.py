from __future__ import annotations

import re

from starter.memory.models import (
    CONSTRAINT_SLOTS,
    EXCLUDABLE_SLOTS,
    PREFERENCE_SLOTS,
    Intent,
    ParseUpdate,
    StateUpdate,
    UpdateOperation,
)


HARD_SLOTS = frozenset({
    "brand", "color", "size", "price_min", "price_max", "rating_min",
})
SOFT_SLOTS = frozenset({"material", "style", "feature", "use_case"})
MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex",
    "silk", "rayon", "fabric",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown",
    "gray", "grey", "purple", "yellow", "orange", "navy", "beige",
    "silver", "gold",
)
SIZES = (
    "xxs", "xs", "xl", "xxl", "xxxl", "one size",
    "wide", "narrow", "plus size",
)
USE_CASES = (
    "hiking", "running", "gym", "winter", "outdoor", "work",
    "wedding", "walking", "travel",
)
LOOKING_FOR_RE = re.compile(
    r"(?:i(?:'m| am) looking for|i(?:'m| am) after)\s+(.+?)(?:\.|, but\b| but i\b|$)",
    re.IGNORECASE | re.DOTALL,
)
CATEGORY_CUE_RE = re.compile(
    r"(?:i(?:'m| am) shopping for|please find|show me|i want|i need(?!\s+is\b))\s+"
    r"(?:an?\s+|some\s+)?(.+?)(?:\.|, but\b|,| but i\b|$)",
    re.IGNORECASE | re.DOTALL,
)
REQUIREMENT_RE = re.compile(
    r"(?:a key requirement is|what matters is|what i(?: really)? need is|"
    r"i really need|please prioritise|please prioritize|"
    r"must be|has to be|have to be|please focus on)\s*:?\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
DECLINE_RE = re.compile(
    r"(?:i don't have (?:a |an additional )?preference for|"
    r"no preference for|flexible about|"
    r"(?:it )?(?:doesn't|does not) matter(?: to me)? for|"
    r"\bany)\s+([a-z_ ]+?)(?:\s+(?:is|are)\s+fine)?(?:\b|$)",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"(?:ignore my earlier preference|forget (?:what i said|my earlier(?: preference)?|that)|"
    r"never mind (?:that|my earlier)|change of plan|i(?:'ve| have) reconsidered)",
    re.IGNORECASE,
)
REMOVE_RE = re.compile(
    r"\b(?:forget about|no longer want|remove|drop|clear)\s+"
    r"(?!what i said|my earlier|that\b)(?:the\s+)?(.+?)"
    r"(?:\s+(?:constraint|preference|requirement))?(?:[.;]|$)",
    re.IGNORECASE,
)
EXCLUDE_RE = re.compile(
    r"\b(?:(?:do not|don't)\s+want|without|avoid|except|exclude)\s+"
    r"(?!a preference|an additional)(.+?)(?:[.;]|, but\b|$)",
    re.IGNORECASE,
)
NOT_VALUE_RE = re.compile(
    r"\bnot\s+(?!quite|sure|really|more\s+than|right)([a-z0-9][\w %.-]{0,40})",
    re.IGNORECASE,
)
NO_VALUE_RE = re.compile(
    r"\bno\s+(?!more\s+than|longer|preference|additional|matter)([a-z0-9][\w %.-]{0,40})",
    re.IGNORECASE,
)
NEGATION_PREFIX_RE = re.compile(
    r"^\s*(?:not|no|without|except|avoid|exclude)\s+(?!more\s+than|longer|preference)",
    re.IGNORECASE,
)
BUDGET_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
PRICE_MAX_RE = re.compile(
    r"\b(?:under|below|less\s+than|up\s+to|at\s+most|no\s+more\s+than|"
    r"max(?:imum)?(?:\s+of)?)\s*(?:\$)?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PRICE_MIN_RE = re.compile(
    r"\b(?:over|above|(?<!no\s)more\s+than|at\s+least)\s*(?:\$)?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
COLOR_LABEL_RE = re.compile(r"\bcolor\s*:\s*([a-z]+)", re.IGNORECASE)
BRAND_LABEL_RE = re.compile(
    r"\b(?:brand|manufacturer)\s*:\s*([A-Za-z0-9][\w.&'’ -]{0,32})",
    re.IGNORECASE,
)
BRAND_SUFFIX_RE = re.compile(r"\b([A-Za-z0-9][\w.&'’-]{1,24})\s+brand\b", re.IGNORECASE)
SIZE_LABEL_RE = re.compile(
    r"\b(?:(?:us|eu|uk)\s+)?size\s*[:#-]?\s*([a-z0-9./-]{1,8})\b",
    re.IGNORECASE,
)
BROWSING_RE = re.compile(
    r"\b(?:still exploring|keeping(?: my)? options open|haven't settled|"
    r"browsing|comparing|just looking)\b",
    re.IGNORECASE,
)
BUYING_RE = re.compile(
    r"(?:a key requirement is|what matters is|what i(?: really)? need is|"
    r"must be|has to be|have to be|please (?:prioritise|prioritize|focus on)|"
    r"i really need)",
    re.IGNORECASE,
)
DECLINE_SLOT = {
    "category": "category",
    "material": "material",
    "color": "color",
    "size": "size",
    "style": "style",
    "brand": "brand",
    "budget": "price_max",
    "price": "price_max",
    "feature": "feature",
    "use_case": "use_case",
    "use case": "use_case",
}


def _clean(value: str, limit: int = 180) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _number(raw: str) -> int | float:
    number = float(raw)
    return int(number) if number.is_integer() else number


def constraint_kind(slot: str, op: UpdateOperation | str) -> str:
    """Map a Memory update onto hard / soft / negative / decline / clear / task."""
    operation = op if isinstance(op, UpdateOperation) else UpdateOperation(op)
    if operation is UpdateOperation.EXCLUDE:
        return "negative"
    if operation is UpdateOperation.DECLINE:
        return "decline"
    if operation is UpdateOperation.CLEAR:
        return "clear"
    if operation is UpdateOperation.REMOVE:
        return "clear"
    if slot in {"category", "product_type", "current_product"}:
        return "task"
    if slot in HARD_SLOTS:
        return "hard"
    if slot in SOFT_SLOTS:
        return "soft"
    return "task"


def classify_intent(
    message: str,
    parsed: ParseUpdate | None = None,
) -> tuple[Intent, float]:
    """Return (intent, confidence) for this turn. Does not read evaluation labels."""
    if parsed is not None and parsed.intent is not None:
        if parsed.reset_task:
            return parsed.intent, 0.90
        if parsed.intent is Intent.BUYING:
            return parsed.intent, 0.95
        if parsed.intent is Intent.BROWSING:
            return parsed.intent, 0.92
        return parsed.intent, 0.55

    text = message.strip()
    if not text:
        return Intent.UNKNOWN, 0.0
    browsing = bool(BROWSING_RE.search(text))
    buying = bool(BUYING_RE.search(text) or OVERRIDE_RE.search(text))
    if browsing and not buying:
        return Intent.BROWSING, 0.92
    if buying:
        return Intent.BUYING, 0.90 if OVERRIDE_RE.search(text) else 0.95
    if LOOKING_FOR_RE.search(text) or CATEGORY_CUE_RE.search(text):
        return Intent.UNKNOWN, 0.45
    return Intent.UNKNOWN, 0.30


def _classify(text: str) -> tuple[str, str | float] | None:
    cleaned = _clean(text)
    if not cleaned:
        return None
    body = NEGATION_PREFIX_RE.sub("", cleaned, count=1).strip() or cleaned
    lowered = body.lower()

    color_label = COLOR_LABEL_RE.search(lowered)
    if color_label:
        return "color", color_label.group(1)

    brand_label = BRAND_LABEL_RE.search(body)
    if brand_label:
        brand = _clean(brand_label.group(1), limit=40)
        if brand:
            return "brand", brand

    brand_suffix = BRAND_SUFFIX_RE.search(body)
    if brand_suffix:
        return "brand", _clean(brand_suffix.group(1), limit=40)

    size_label = SIZE_LABEL_RE.search(lowered)
    if size_label:
        return "size", size_label.group(1).lower()

    max_price = PRICE_MAX_RE.search(lowered)
    if max_price:
        return "price_max", _number(max_price.group(1))

    min_price = PRICE_MIN_RE.search(lowered)
    if min_price:
        return "price_min", _number(min_price.group(1))

    budget = BUDGET_RE.search(lowered)
    if budget and ("budget" in lowered or "$" in cleaned):
        return "price_max", _number(budget.group(1))

    for material in MATERIALS:
        if re.search(rf"\b{material}\b", lowered):
            return "material", material

    for color in COLORS:
        if re.search(rf"\b{color}\b", lowered):
            return "color", color

    for size in SIZES:
        if re.search(rf"\b{re.escape(size)}\b", lowered):
            return "size", size

    if any(word in lowered for word in ("size", "sizing", "width")):
        return "size", cleaned
    if any(word in lowered for word in USE_CASES):
        return "use_case", cleaned
    if any(word in lowered for word in ("style", "fit", "sleeve", "neck", "department")):
        return "style", cleaned
    if "brand" in lowered:
        return "brand", cleaned
    return "feature", cleaned


def _split_constraints(text: str) -> list[str]:
    """Keep the joined catalog line, then each '; '-separated part."""
    cleaned_full = _clean(text, limit=180)
    parts = [_clean(part, limit=180) for part in re.split(r";\s+", text)]
    parts = [part for part in parts if part]
    if not cleaned_full:
        return []
    if len(parts) <= 1:
        return [cleaned_full]
    return [cleaned_full, *parts]


def _has_update(
    updates: list[StateUpdate],
    slot: str,
    op: UpdateOperation | None = None,
    value: object | None = None,
) -> bool:
    for update in updates:
        if update.slot != slot:
            continue
        if op is not None and update.op is not op:
            continue
        if value is not None and update.value != value:
            continue
        return True
    return False


def _append_constraint(
    updates: list[StateUpdate],
    chunk: str,
    *,
    keep_short_snippet: bool,
    force_negative: bool = False,
) -> None:
    cleaned = _clean(chunk, limit=180)
    if not cleaned:
        return
    classified = _classify(chunk)
    if classified is None:
        return
    slot, value = classified
    negative = force_negative or bool(NEGATION_PREFIX_RE.search(cleaned))
    if negative and slot in EXCLUDABLE_SLOTS:
        if not _has_update(updates, slot, UpdateOperation.EXCLUDE, value):
            updates.append(StateUpdate(slot, UpdateOperation.EXCLUDE, value))
        return
    if negative and slot in {"price_min", "price_max"}:
        return
    op = (
        UpdateOperation.ADD
        if slot in {"feature", "use_case"}
        else UpdateOperation.SET
    )
    if not _has_update(updates, slot, op, value):
        updates.append(StateUpdate(slot, op, value))
    if slot == "feature":
        return
    if keep_short_snippet or cleaned.lower() != str(value).strip().lower():
        if not _has_update(updates, "feature", UpdateOperation.ADD, cleaned):
            updates.append(StateUpdate("feature", UpdateOperation.ADD, cleaned))


def _typed_tokens(text: str) -> list[tuple[str, str]]:
    lowered = text.lower()
    found: list[tuple[str, str]] = []
    for material in MATERIALS:
        if re.search(rf"\b{material}\b", lowered):
            found.append(("material", material))
    for color in COLORS:
        if re.search(rf"\b{color}\b", lowered):
            found.append(("color", color))
    return found


def _append_decline(updates: list[StateUpdate], attribute: str) -> None:
    key = attribute.strip().lower().replace("-", " ")
    key = re.sub(r"\s+", " ", key).strip()
    slot = DECLINE_SLOT.get(key) or DECLINE_SLOT.get(key.replace(" ", "_"))
    if slot in CONSTRAINT_SLOTS | PREFERENCE_SLOTS and not _has_update(
        updates, slot, UpdateOperation.DECLINE,
    ):
        updates.append(StateUpdate(slot, UpdateOperation.DECLINE))


def _append_remove(updates: list[StateUpdate], raw: str) -> None:
    cleaned = _clean(raw, limit=80)
    if not cleaned:
        return
    attribute = cleaned.lower().replace(" ", "_")
    slot = DECLINE_SLOT.get(attribute) or DECLINE_SLOT.get(cleaned.lower())
    if slot in CONSTRAINT_SLOTS | PREFERENCE_SLOTS:
        if not _has_update(updates, slot, UpdateOperation.CLEAR):
            updates.append(StateUpdate(slot, UpdateOperation.CLEAR))
        return
    classified = _classify(cleaned)
    if classified is None:
        return
    slot, value = classified
    if slot in CONSTRAINT_SLOTS | PREFERENCE_SLOTS and slot not in {
        "price_min", "price_max", "rating_min",
    }:
        if not _has_update(updates, slot, UpdateOperation.REMOVE, value):
            updates.append(StateUpdate(slot, UpdateOperation.REMOVE, value))


def _extract_labeled_slots(message: str, updates: list[StateUpdate]) -> None:
    if not any(item.slot == "brand" for item in updates):
        brand = BRAND_LABEL_RE.search(message) or BRAND_SUFFIX_RE.search(message)
        if brand:
            value = _clean(brand.group(1), limit=40)
            if value:
                updates.append(StateUpdate("brand", UpdateOperation.SET, value))
    if not any(item.slot == "size" for item in updates):
        size = SIZE_LABEL_RE.search(message)
        if size:
            updates.append(StateUpdate("size", UpdateOperation.SET, size.group(1).lower()))
        else:
            lowered = message.lower()
            for token in SIZES:
                if re.search(rf"\b{re.escape(token)}\b", lowered):
                    updates.append(StateUpdate("size", UpdateOperation.SET, token))
                    break
    if not any(item.slot in {"price_min", "price_max"} for item in updates):
        max_price = PRICE_MAX_RE.search(message)
        min_price = PRICE_MIN_RE.search(message)
        budget = BUDGET_RE.search(message)
        if max_price:
            updates.append(StateUpdate("price_max", UpdateOperation.SET, _number(max_price.group(1))))
        elif budget and re.search(r"budget|\$", message, re.IGNORECASE):
            updates.append(StateUpdate("price_max", UpdateOperation.SET, _number(budget.group(1))))
        if min_price:
            updates.append(StateUpdate("price_min", UpdateOperation.SET, _number(min_price.group(1))))


def _extract_negatives(message: str, updates: list[StateUpdate]) -> None:
    if DECLINE_RE.search(message) and not EXCLUDE_RE.search(message):
        if "don't want" not in message.lower() and "do not want" not in message.lower():
            if not NOT_VALUE_RE.search(message) and not NO_VALUE_RE.search(message):
                return
    for match in EXCLUDE_RE.finditer(message):
        _append_constraint(updates, match.group(1), keep_short_snippet=False, force_negative=True)
    for pattern in (NOT_VALUE_RE, NO_VALUE_RE):
        for match in pattern.finditer(message):
            _append_constraint(updates, match.group(1), keep_short_snippet=False, force_negative=True)


def parse_user_message(
    session_id: str,
    user_message: str,
    turn: int,
) -> ParseUpdate | None:
    """Turn a customer utterance into a Memory ParseUpdate."""
    message = user_message.strip()
    if not message:
        return None

    reset_task = bool(OVERRIDE_RE.search(message))
    intent: Intent | None = None
    updates: list[StateUpdate] = []

    looking = LOOKING_FOR_RE.search(message)
    if looking is None:
        looking = CATEGORY_CUE_RE.search(message)
    if looking:
        category = _clean(looking.group(1), limit=120)
        if category:
            updates.append(StateUpdate("category", UpdateOperation.SET, category))
            for slot, value in _typed_tokens(category):
                if not _has_update(updates, slot):
                    updates.append(StateUpdate(slot, UpdateOperation.SET, value))
        if BROWSING_RE.search(message):
            intent = Intent.BROWSING
        elif reset_task or REQUIREMENT_RE.search(message):
            intent = Intent.BUYING

    requirement = REQUIREMENT_RE.search(message)
    constraint_chunks: list[str] = []
    if requirement:
        constraint_chunks.extend(_split_constraints(requirement.group(1)))
        intent = intent or Intent.BUYING
    elif looking is None:
        pass
    else:
        remainder = message[looking.end():].strip(" .")
        if remainder and not BROWSING_RE.search(remainder):
            if not REQUIREMENT_RE.search(remainder):
                constraint_chunks.extend(_split_constraints(remainder))

    seen_chunks: set[str] = set()
    for chunk in constraint_chunks:
        key = _clean(chunk).lower()
        if not key or key in seen_chunks:
            continue
        seen_chunks.add(key)
        _append_constraint(updates, chunk, keep_short_snippet=reset_task)

    for decline in DECLINE_RE.finditer(message):
        _append_decline(updates, decline.group(1))

    if not reset_task:
        for match in REMOVE_RE.finditer(message):
            _append_remove(updates, match.group(1))

    _extract_labeled_slots(message, updates)
    _extract_negatives(message, updates)

    hard_ops = any(
        constraint_kind(update.slot, update.op) in {"hard", "negative"}
        for update in updates
        if update.slot != "category"
    )
    if intent is None and hard_ops:
        intent = Intent.BUYING
    if intent is None and BROWSING_RE.search(message):
        intent = Intent.BROWSING

    if intent is None and not updates and not reset_task:
        return None

    return ParseUpdate(
        session_id=session_id,
        intent=intent,
        updates=tuple(updates),
        reset_task=reset_task,
        source_turn=turn,
    )
