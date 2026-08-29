from __future__ import annotations

import re

from starter.memory.models import (
    CONSTRAINT_SLOTS,
    PREFERENCE_SLOTS,
    Intent,
    ParseUpdate,
    StateUpdate,
    UpdateOperation,
)


MATERIALS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex",
    "silk", "rayon", "fabric",
)
COLORS = (
    "black", "white", "blue", "red", "pink", "green", "brown",
    "gray", "grey", "purple", "yellow", "orange",
)
LOOKING_FOR_RE = re.compile(
    r"i(?:'m| am) looking for\s+(.+?)(?:\.|, but\b| but i\b|$)",
    re.IGNORECASE | re.DOTALL,
)
REQUIREMENT_RE = re.compile(
    r"(?:a key requirement is|what matters is|what i need is)\s*:?\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
DECLINE_RE = re.compile(
    r"i don't have (?:a |an additional )?preference for\s+([a-z_ ]+)",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(r"ignore my earlier preference", re.IGNORECASE)
BUDGET_RE = re.compile(r"\$\s*(\d+(?:\.\d+)?)")
COLOR_LABEL_RE = re.compile(r"\bcolor\s*:\s*([a-z]+)", re.IGNORECASE)
DECLINE_SLOT = {
    "category": "category",
    "material": "material",
    "color": "color",
    "size": "size",
    "style": "style",
    "brand": "brand",
    "budget": "price_max",
    "feature": "feature",
    "use_case": "use_case",
}


def _clean(value: str, limit: int = 80) -> str:
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def _classify(text: str) -> tuple[str, str | float] | None:
    cleaned = _clean(text)
    if not cleaned:
        return None
    lowered = cleaned.lower()

    color_label = COLOR_LABEL_RE.search(lowered)
    if color_label:
        return "color", color_label.group(1)

    budget = BUDGET_RE.search(lowered)
    if budget and ("budget" in lowered or "$" in cleaned):
        number = float(budget.group(1))
        return "price_max", int(number) if number.is_integer() else number

    for material in MATERIALS:
        if re.search(rf"\b{material}\b", lowered):
            return "material", material

    for color in COLORS:
        if re.search(rf"\b{color}\b", lowered):
            return "color", color

    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size", cleaned
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case", cleaned
    if any(word in lowered for word in ("style", "fit", "sleeve", "neck", "department")):
        return "style", cleaned
    if "brand" in lowered:
        return "brand", cleaned
    return "feature", cleaned


def _split_constraints(text: str) -> list[str]:
    return [part.strip() for part in text.split(";") if _clean(part)]


def parse_user_message(
    session_id: str,
    user_message: str,
    turn: int,
) -> ParseUpdate | None:
    """Turn a simulator-style utterance into a Memory ParseUpdate."""
    message = user_message.strip()
    if not message:
        return None

    reset_task = bool(OVERRIDE_RE.search(message))
    intent: Intent | None = None
    updates: list[StateUpdate] = []

    looking = LOOKING_FOR_RE.search(message)
    if looking:
        category = _clean(looking.group(1), limit=120)
        if category:
            updates.append(StateUpdate("category", UpdateOperation.SET, category))
        if re.search(r"still exploring", message, re.IGNORECASE):
            intent = Intent.BROWSING
        elif reset_task or re.search(r"key requirement", message, re.IGNORECASE):
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
        if remainder and not re.search(r"still exploring", remainder, re.IGNORECASE):
            if not REQUIREMENT_RE.search(remainder):
                constraint_chunks.append(remainder)

    for chunk in constraint_chunks:
        classified = _classify(chunk)
        if classified is None:
            continue
        slot, value = classified
        op = (
            UpdateOperation.ADD
            if slot in {"feature", "use_case"}
            else UpdateOperation.SET
        )
        updates.append(StateUpdate(slot, op, value))

    decline = DECLINE_RE.search(message)
    if decline:
        attribute = decline.group(1).strip().lower().replace(" ", "_")
        slot = DECLINE_SLOT.get(attribute)
        if slot in CONSTRAINT_SLOTS | PREFERENCE_SLOTS:
            updates.append(StateUpdate(slot, UpdateOperation.DECLINE))

    if intent is None and not updates and not reset_task:
        return None

    return ParseUpdate(
        session_id=session_id,
        intent=intent,
        updates=tuple(updates),
        reset_task=reset_task,
        source_turn=turn,
    )
