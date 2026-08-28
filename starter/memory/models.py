from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias


JsonScalar: TypeAlias = str | int | float | bool
StateValue: TypeAlias = JsonScalar | tuple[JsonScalar, ...]
MEMORY_SCHEMA_VERSION = "4.0"
HISTORY_LIMIT = 20


class Intent(str, Enum):
    BUYING = "buying"
    BROWSING = "browsing"
    UNKNOWN = "unknown"


class UpdateOperation(str, Enum):
    SET = "set"
    ADD = "add"
    REMOVE = "remove"
    CLEAR = "clear"
    DECLINE = "decline"
    EXCLUDE = "exclude"


class AttributeStatus(str, Enum):
    UNKNOWN = "unknown"
    SPECIFIED = "specified"
    NO_PREFERENCE = "no_preference"


DIRECT_SLOTS = frozenset({"category", "product_type", "current_product"})
CONSTRAINT_SLOTS = frozenset({
    "price_min",
    "price_max",
    "brand",
    "color",
    "size",
    "rating_min",
})
PREFERENCE_SLOTS = frozenset({"material", "style", "feature", "use_case"})
EXCLUDED_SLOT = "excluded"
ALLOWED_SLOTS = DIRECT_SLOTS | CONSTRAINT_SLOTS | PREFERENCE_SLOTS | {EXCLUDED_SLOT}
NUMERIC_SLOTS = frozenset({"price_min", "price_max", "rating_min"})
STRING_SLOTS = ALLOWED_SLOTS - NUMERIC_SLOTS
DECLINABLE_SLOTS = CONSTRAINT_SLOTS | PREFERENCE_SLOTS
EXCLUDABLE_SLOTS = (CONSTRAINT_SLOTS | PREFERENCE_SLOTS) - NUMERIC_SLOTS
TRACKED_ATTRIBUTE_SLOTS = DECLINABLE_SLOTS | {"category", "product_type"}
DIALOGUE_ATTRIBUTES = frozenset({
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
})
ATTRIBUTE_SLOTS = {
    "budget": ("price_min", "price_max"),
}


def _validate_session_id(session_id: object) -> str:
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    return session_id


def _validate_turn(turn: object, field_name: str) -> int:
    if isinstance(turn, bool) or not isinstance(turn, int) or turn < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return turn


def _normalize_value(slot: str, value: object) -> JsonScalar:
    if value is None:
        raise ValueError(f"{slot} requires a value; use clear to remove it")
    if slot in NUMERIC_SLOTS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{slot} must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{slot} must be a non-negative finite number")
        if slot == "rating_min" and number > 5:
            raise ValueError("rating_min cannot exceed 5")
        return int(number) if number.is_integer() else number
    if slot in STRING_SLOTS:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{slot} must be a non-empty string")
        return value.strip()
    raise ValueError(f"unsupported slot: {slot}")


def _json_value(value: StateValue) -> JsonScalar | list[JsonScalar]:
    return list(value) if isinstance(value, tuple) else value


@dataclass(frozen=True, slots=True)
class StateUpdate:
    slot: str
    op: UpdateOperation
    value: JsonScalar | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.slot, str) or self.slot not in ALLOWED_SLOTS:
            raise ValueError(f"unsupported slot: {self.slot!r}")
        object.__setattr__(self, "op", UpdateOperation(self.op))
        if self.op in {UpdateOperation.CLEAR, UpdateOperation.DECLINE}:
            if self.value is not None:
                raise ValueError(f"{self.op.value} does not accept a value")
            if self.op is UpdateOperation.DECLINE and self.slot not in DECLINABLE_SLOTS:
                raise ValueError(f"decline is not supported for {self.slot}")
            return
        if self.op is UpdateOperation.EXCLUDE and self.slot not in EXCLUDABLE_SLOTS:
            raise ValueError(f"exclude is not supported for {self.slot}")
        if self.op in {UpdateOperation.ADD, UpdateOperation.REMOVE} and self.slot in DIRECT_SLOTS:
            raise ValueError(f"{self.op.value} is not supported for {self.slot}")
        if self.op in {UpdateOperation.ADD, UpdateOperation.REMOVE} and self.slot in NUMERIC_SLOTS:
            raise ValueError(f"{self.op.value} is not supported for {self.slot}; use set or clear")
        object.__setattr__(self, "value", _normalize_value(self.slot, self.value))

    @classmethod
    def from_dict(cls, payload: dict) -> StateUpdate:
        if not isinstance(payload, dict):
            raise TypeError("each update must be an object")
        unknown = set(payload) - {"slot", "op", "value"}
        if unknown:
            raise ValueError(f"unknown update fields: {sorted(unknown)}")
        if "slot" not in payload or "op" not in payload:
            raise ValueError("each update requires slot and op")
        return cls(
            slot=payload["slot"],
            op=payload["op"],
            value=payload.get("value"),
        )

    def to_dict(self) -> dict:
        result = {"slot": self.slot, "op": self.op.value}
        if self.op not in {UpdateOperation.CLEAR, UpdateOperation.DECLINE}:
            result["value"] = self.value
        return result


@dataclass(frozen=True, slots=True)
class ParseUpdate:
    session_id: str
    intent: Intent | None = None
    updates: tuple[StateUpdate, ...] = ()
    reset_task: bool = False
    source_turn: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _validate_session_id(self.session_id))
        if self.intent is not None:
            object.__setattr__(self, "intent", Intent(self.intent))
        if not isinstance(self.reset_task, bool):
            raise TypeError("reset_task must be a boolean")
        if self.source_turn is not None:
            object.__setattr__(
                self,
                "source_turn",
                _validate_turn(self.source_turn, "source_turn"),
            )
        object.__setattr__(
            self,
            "updates",
            tuple(
                update if isinstance(update, StateUpdate) else StateUpdate.from_dict(update)
                for update in self.updates
            ),
        )

    @classmethod
    def from_dict(cls, payload: dict) -> ParseUpdate:
        if not isinstance(payload, dict):
            raise TypeError("ParseUpdate must be an object")
        unknown = set(payload) - {
            "session_id",
            "intent",
            "updates",
            "reset_task",
            "source_turn",
        }
        if unknown:
            raise ValueError(f"unknown ParseUpdate fields: {sorted(unknown)}")
        if "session_id" not in payload:
            raise ValueError("ParseUpdate requires session_id")
        updates = payload.get("updates", [])
        if not isinstance(updates, (list, tuple)):
            raise TypeError("updates must be an array")
        return cls(
            session_id=payload["session_id"],
            intent=payload.get("intent"),
            updates=tuple(StateUpdate.from_dict(update) for update in updates),
            reset_task=payload.get("reset_task", False),
            source_turn=payload.get("source_turn"),
        )

    def to_dict(self) -> dict:
        result = {
            "session_id": self.session_id,
            "intent": self.intent.value if self.intent is not None else None,
            "updates": [update.to_dict() for update in self.updates],
        }
        if self.reset_task:
            result["reset_task"] = True
        if self.source_turn is not None:
            result["source_turn"] = self.source_turn
        return result

    def signature(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class RetrievalState:
    schema_version: str
    intent: Intent
    category: str | None
    product_type: str | None
    hard_constraints: dict[str, JsonScalar | list[JsonScalar]]
    soft_preferences: dict[str, JsonScalar | list[JsonScalar]]
    excluded: dict[str, list[JsonScalar]]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "intent": self.intent.value,
            "category": self.category,
            "product_type": self.product_type,
            "hard_constraints": copy.deepcopy(self.hard_constraints),
            "soft_preferences": copy.deepcopy(self.soft_preferences),
            "excluded": copy.deepcopy(self.excluded),
        }

    def retrieval_text(self) -> str:
        values: list[str] = []
        for value in (self.category, self.product_type):
            if value:
                values.append(value)
        for mapping in (self.hard_constraints, self.soft_preferences):
            for value in mapping.values():
                items = value if isinstance(value, list) else [value]
                values.extend(str(item) for item in items if isinstance(item, str))
        return " ".join(values)


@dataclass(frozen=True, slots=True)
class StateDelta:
    changed_slots: tuple[str, ...] = ()
    removed_slots: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.changed_slots and not self.removed_slots

    def to_dict(self) -> dict:
        return {
            "changed_slots": list(self.changed_slots),
            "removed_slots": list(self.removed_slots),
        }


@dataclass(frozen=True, slots=True)
class StateChange:
    turn: int | None
    slot: str
    old: object
    new: object
    op: str

    def __post_init__(self) -> None:
        if self.turn is not None:
            _validate_turn(self.turn, "history turn")
        if not isinstance(self.slot, str) or not self.slot:
            raise ValueError("history slot must be a non-empty string")
        if not isinstance(self.op, str) or not self.op:
            raise ValueError("history op must be a non-empty string")
        try:
            json.dumps([self.old, self.new])
        except (TypeError, ValueError) as error:
            raise TypeError("history old/new values must be JSON serializable") from error

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "slot": self.slot,
            "old": copy.deepcopy(self.old),
            "new": copy.deepcopy(self.new),
            "op": self.op,
        }


@dataclass(slots=True)
class CurrentState:
    session_id: str
    schema_version: str = MEMORY_SCHEMA_VERSION
    intent: Intent = Intent.UNKNOWN
    category: str | None = None
    product_type: str | None = None
    constraints: dict[str, StateValue] = field(default_factory=dict)
    preferences: dict[str, StateValue] = field(default_factory=dict)
    excluded: list[JsonScalar] = field(default_factory=list)
    excluded_by_slot: dict[str, StateValue] = field(default_factory=dict)
    current_product: str | None = None
    attribute_status: dict[str, AttributeStatus] = field(default_factory=dict)
    asked_attributes: list[str] = field(default_factory=list)
    asked_attribute_by_turn: dict[int, str | None] = field(default_factory=dict)
    applied_turn_signatures: dict[int, str] = field(default_factory=dict)
    change_history: list[StateChange] = field(default_factory=list)
    task_version: int = 0
    updated_at: int = 0

    def __post_init__(self) -> None:
        self.session_id = _validate_session_id(self.session_id)
        if self.schema_version != MEMORY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version: {self.schema_version!r}; "
                f"expected {MEMORY_SCHEMA_VERSION!r}"
            )
        self.intent = Intent(self.intent)
        for slot in DIRECT_SLOTS:
            value = getattr(self, slot)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{slot} must be a non-empty string or null")
            if isinstance(value, str):
                setattr(self, slot, value.strip())
        unknown_constraints = set(self.constraints) - CONSTRAINT_SLOTS
        if unknown_constraints:
            raise ValueError(f"unsupported constraint slots: {sorted(unknown_constraints)}")
        unknown_preferences = set(self.preferences) - PREFERENCE_SLOTS
        if unknown_preferences:
            raise ValueError(f"unsupported preference slots: {sorted(unknown_preferences)}")
        for mapping in (self.constraints, self.preferences):
            for slot, raw_value in mapping.items():
                raw_items = raw_value if isinstance(raw_value, tuple) else (raw_value,)
                if not raw_items:
                    raise ValueError(f"{slot} cannot contain an empty value tuple")
                for item in raw_items:
                    _normalize_value(slot, item)
        self.excluded = [
            _normalize_value(EXCLUDED_SLOT, value) for value in self.excluded
        ]
        normalized_exclusions: dict[str, StateValue] = {}
        for slot, raw_value in self.excluded_by_slot.items():
            if slot not in EXCLUDABLE_SLOTS:
                raise ValueError(f"unsupported structured exclusion slot: {slot}")
            raw_items = raw_value if isinstance(raw_value, (tuple, list)) else (raw_value,)
            values = tuple(dict.fromkeys(_normalize_value(slot, item) for item in raw_items))
            if not values:
                raise ValueError(f"structured exclusion for {slot} cannot be empty")
            normalized_exclusions[slot] = values[0] if len(values) == 1 else values
        self.excluded_by_slot = normalized_exclusions
        self.attribute_status = {
            slot: AttributeStatus(status) for slot, status in self.attribute_status.items()
        }
        unknown_status_slots = set(self.attribute_status) - TRACKED_ATTRIBUTE_SLOTS
        if unknown_status_slots:
            raise ValueError(f"unsupported attribute status slots: {sorted(unknown_status_slots)}")
        for attribute in self.asked_attributes:
            if attribute not in DIALOGUE_ATTRIBUTES:
                raise ValueError(f"unsupported dialogue attribute: {attribute}")
        self.asked_attributes = list(dict.fromkeys(self.asked_attributes))
        normalized_questions: dict[int, str | None] = {}
        for turn, attribute in self.asked_attribute_by_turn.items():
            normalized_turn = _validate_turn(turn, "asked_attribute_by_turn key")
            if attribute is not None and attribute not in DIALOGUE_ATTRIBUTES:
                raise ValueError(f"unsupported dialogue attribute: {attribute}")
            normalized_questions[normalized_turn] = attribute
        self.asked_attribute_by_turn = normalized_questions
        normalized_signatures: dict[int, str] = {}
        for turn, signature in self.applied_turn_signatures.items():
            normalized_turn = _validate_turn(turn, "applied_turn_signatures key")
            if not isinstance(signature, str) or not signature:
                raise ValueError("applied turn signature must be a non-empty string")
            normalized_signatures[normalized_turn] = signature
        self.applied_turn_signatures = normalized_signatures
        self.change_history = [
            change if isinstance(change, StateChange) else StateChange(**change)
            for change in self.change_history[-HISTORY_LIMIT:]
        ]
        for slot in TRACKED_ATTRIBUTE_SLOTS:
            if self._has_value(slot):
                status = self.attribute_status.setdefault(slot, AttributeStatus.SPECIFIED)
                if status is not AttributeStatus.SPECIFIED:
                    raise ValueError(f"{slot} cannot have a value and status {status.value}")
            elif self.attribute_status.get(slot) is AttributeStatus.SPECIFIED:
                raise ValueError(f"{slot} requires a value when status is specified")
        for slot, excluded in self.excluded_by_slot.items():
            if slot in CONSTRAINT_SLOTS:
                positive = self.constraints.get(slot)
            else:
                positive = self.preferences.get(slot)
            positive_values = set(
                positive if isinstance(positive, tuple) else (() if positive is None else (positive,))
            )
            excluded_values = set(
                excluded if isinstance(excluded, tuple) else (excluded,)
            )
            if positive_values & excluded_values:
                raise ValueError(f"{slot} cannot contain the same positive and excluded value")
            if self.attribute_status.get(slot) is AttributeStatus.NO_PREFERENCE:
                raise ValueError(f"{slot} cannot be declined and excluded at the same time")
        if isinstance(self.task_version, bool) or not isinstance(self.task_version, int):
            raise TypeError("task_version must be an integer")
        if self.task_version < 0:
            raise ValueError("task_version must be non-negative")
        if isinstance(self.updated_at, bool) or not isinstance(self.updated_at, int):
            raise TypeError("updated_at must be an integer")
        if self.updated_at < 0:
            raise ValueError("updated_at must be a non-negative logical version")

    def _has_value(self, slot: str) -> bool:
        if slot in DIRECT_SLOTS:
            return getattr(self, slot) is not None
        if slot in CONSTRAINT_SLOTS:
            return slot in self.constraints
        if slot in PREFERENCE_SLOTS:
            return slot in self.preferences
        return False

    def status_for(self, attribute: str) -> AttributeStatus:
        if attribute not in DIALOGUE_ATTRIBUTES:
            raise ValueError(f"unsupported dialogue attribute: {attribute}")
        slots = ATTRIBUTE_SLOTS.get(attribute, (attribute,))
        statuses = [self.attribute_status.get(slot, AttributeStatus.UNKNOWN) for slot in slots]
        if AttributeStatus.SPECIFIED in statuses:
            return AttributeStatus.SPECIFIED
        if AttributeStatus.NO_PREFERENCE in statuses:
            return AttributeStatus.NO_PREFERENCE
        return AttributeStatus.UNKNOWN

    def was_asked(self, attribute: str) -> bool:
        if attribute not in DIALOGUE_ATTRIBUTES:
            raise ValueError(f"unsupported dialogue attribute: {attribute}")
        return attribute in self.asked_attributes

    def clone(self) -> CurrentState:
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "schema_version": self.schema_version,
            "intent": self.intent.value,
            "category": self.category,
            "product_type": self.product_type,
            "constraints": {
                key: _json_value(value) for key, value in sorted(self.constraints.items())
            },
            "preferences": {
                key: _json_value(value) for key, value in sorted(self.preferences.items())
            },
            "excluded": list(self.excluded),
            "excluded_by_slot": {
                key: _json_value(value) for key, value in sorted(self.excluded_by_slot.items())
            },
            "current_product": self.current_product,
            "attribute_status": {
                slot: status.value for slot, status in sorted(self.attribute_status.items())
            },
            "asked_attributes": list(self.asked_attributes),
            "asked_attribute_by_turn": dict(sorted(self.asked_attribute_by_turn.items())),
            "applied_turn_signatures": dict(sorted(self.applied_turn_signatures.items())),
            "change_history": [change.to_dict() for change in self.change_history],
            "task_version": self.task_version,
            "updated_at": self.updated_at,
        }

    @staticmethod
    def _retrieval_value(value: StateValue) -> JsonScalar | list[JsonScalar]:
        json_value = _json_value(value)
        if isinstance(json_value, str):
            return [json_value]
        return json_value

    def to_retrieval_state(self) -> RetrievalState:
        excluded = {
            key: list(value) if isinstance(value, tuple) else [value]
            for key, value in sorted(self.excluded_by_slot.items())
        }
        if self.excluded:
            excluded["other"] = list(self.excluded)
        return RetrievalState(
            schema_version=self.schema_version,
            intent=self.intent,
            category=self.category,
            product_type=self.product_type,
            hard_constraints={
                key: self._retrieval_value(value)
                for key, value in sorted(self.constraints.items())
            },
            soft_preferences={
                key: self._retrieval_value(value)
                for key, value in sorted(self.preferences.items())
            },
            excluded=excluded,
        )

    def to_retrieval_payload(self) -> dict:
        return self.to_retrieval_state().to_dict()

    def retrieval_text(self) -> str:
        return self.to_retrieval_state().retrieval_text()
