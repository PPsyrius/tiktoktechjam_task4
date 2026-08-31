from __future__ import annotations

from starter.memory.models import (
    CONSTRAINT_SLOTS,
    DIRECT_SLOTS,
    EXCLUDED_SLOT,
    HISTORY_LIMIT,
    PREFERENCE_SLOTS,
    AttributeStatus,
    CurrentState,
    ParseUpdate,
    StateChange,
    StateDelta,
    StateUpdate,
    StateValue,
    UpdateOperation,
)


class TurnConflictError(ValueError):
    pass


class OutOfOrderTurnError(ValueError):
    pass


class StateManager:
    """Applies validated updates atomically to a detached state snapshot."""

    def apply_update(self, state: CurrentState, parsed: ParseUpdate) -> CurrentState:
        updated, _ = self.apply_update_with_delta(state, parsed)
        return updated

    def validate_state(self, state: CurrentState) -> None:
        self._validate_cross_field_constraints(state)

    def apply_update_with_delta(
        self,
        state: CurrentState,
        parsed: ParseUpdate,
    ) -> tuple[CurrentState, StateDelta]:
        if state.session_id != parsed.session_id:
            raise ValueError("ParseUpdate session_id does not match CurrentState")

        signature = parsed.signature() if parsed.source_turn is not None else None
        if parsed.source_turn is not None:
            existing = state.applied_turn_signatures.get(parsed.source_turn)
            if existing is not None:
                if existing == signature:
                    return state.clone(), StateDelta()
                raise TurnConflictError(
                    f"source_turn {parsed.source_turn} was already applied with different content"
                )
            if (
                state.applied_turn_signatures
                and parsed.source_turn < max(state.applied_turn_signatures)
            ):
                raise OutOfOrderTurnError(
                    f"source_turn {parsed.source_turn} is older than the latest applied turn"
                )

        before = state.to_dict()
        before_demand = self._demand_snapshot(state)
        if parsed.reset_task:
            candidate = CurrentState(
                session_id=state.session_id,
                applied_turn_signatures=dict(state.applied_turn_signatures),
                change_history=list(state.change_history),
                task_version=state.task_version + 1,
                updated_at=state.updated_at,
            )
        else:
            candidate = state.clone()
        if parsed.intent is not None:
            candidate.intent = parsed.intent
        for update in parsed.updates:
            self._apply_one(candidate, update)
        self._validate_cross_field_constraints(candidate)
        after_demand = self._demand_snapshot(candidate)
        delta = self._build_delta(before_demand, after_demand)
        self._append_history(candidate, parsed, before_demand, after_demand, delta)
        if parsed.source_turn is not None and signature is not None:
            candidate.applied_turn_signatures[parsed.source_turn] = signature
        if candidate.to_dict() != before:
            candidate.updated_at = state.updated_at + 1
        return candidate, delta

    def get_or_record_asked_attribute(
        self,
        state: CurrentState,
        source_turn: int,
        candidates: tuple[str, ...] | list[str],
    ) -> tuple[CurrentState, str | None]:
        if isinstance(source_turn, bool) or not isinstance(source_turn, int) or source_turn < 1:
            raise ValueError("source_turn must be a positive integer")
        if source_turn in state.asked_attribute_by_turn:
            return state.clone(), state.asked_attribute_by_turn[source_turn]
        if (
            state.asked_attribute_by_turn
            and source_turn < max(state.asked_attribute_by_turn)
        ):
            raise OutOfOrderTurnError(
                f"source_turn {source_turn} is older than the latest question turn"
            )

        candidate = state.clone()
        selected = next(
            (attribute for attribute in candidates if not candidate.was_asked(attribute)),
            None,
        )
        candidate.asked_attribute_by_turn[source_turn] = selected
        if selected is not None:
            candidate.asked_attributes.append(selected)
        candidate.updated_at += 1
        return candidate, selected

    def mark_attribute_asked(self, state: CurrentState, attribute: str) -> CurrentState:
        candidate = state.clone()
        if candidate.was_asked(attribute):
            return candidate
        candidate.asked_attributes.append(attribute)
        candidate.updated_at += 1
        return candidate

    def _apply_one(self, state: CurrentState, update: StateUpdate) -> None:
        if update.slot in DIRECT_SLOTS:
            attribute = update.slot
            setattr(
                state,
                attribute,
                None if update.op is UpdateOperation.CLEAR else str(update.value),
            )
            if update.slot in {"category", "product_type"}:
                if update.op is UpdateOperation.CLEAR:
                    state.attribute_status.pop(update.slot, None)
                else:
                    state.attribute_status[update.slot] = AttributeStatus.SPECIFIED
            return
        if update.slot == EXCLUDED_SLOT:
            self._apply_excluded(state, update)
            return
        if update.slot in CONSTRAINT_SLOTS:
            target = state.constraints
        elif update.slot in PREFERENCE_SLOTS:
            target = state.preferences
        else:
            raise ValueError(f"unsupported slot: {update.slot}")
        if update.op is UpdateOperation.DECLINE:
            target.pop(update.slot, None)
            state.excluded_by_slot.pop(update.slot, None)
            state.attribute_status[update.slot] = AttributeStatus.NO_PREFERENCE
            return
        if update.op is UpdateOperation.EXCLUDE:
            value = update.value
            if value is None:
                raise ValueError("exclude requires a value")
            self._remove_mapping_value(target, update.slot, value)
            self._add_structured_exclusion(state, update.slot, value)
            if update.slot in target:
                state.attribute_status[update.slot] = AttributeStatus.SPECIFIED
            else:
                state.attribute_status.pop(update.slot, None)
            return
        existed_before = update.slot in target
        self._apply_mapping(target, update)
        if update.op is UpdateOperation.CLEAR:
            state.excluded_by_slot.pop(update.slot, None)
            state.attribute_status.pop(update.slot, None)
        elif update.slot in target:
            value = update.value
            if value is not None:
                self._remove_structured_exclusion(state, update.slot, value)
                state.excluded[:] = [item for item in state.excluded if item != value]
            state.attribute_status[update.slot] = AttributeStatus.SPECIFIED
        elif existed_before:
            state.attribute_status.pop(update.slot, None)

    @staticmethod
    def _remove_mapping_value(
        target: dict[str, StateValue],
        slot: str,
        value: object,
    ) -> None:
        current = target.get(slot)
        if current is None:
            return
        values = list(current) if isinstance(current, tuple) else [current]
        remaining = [item for item in values if item != value]
        if not remaining:
            target.pop(slot, None)
        elif len(remaining) == 1:
            target[slot] = remaining[0]
        else:
            target[slot] = tuple(remaining)

    @staticmethod
    def _add_structured_exclusion(
        state: CurrentState,
        slot: str,
        value: object,
    ) -> None:
        current = state.excluded_by_slot.get(slot)
        values = list(current) if isinstance(current, tuple) else ([] if current is None else [current])
        if value not in values:
            values.append(value)
        state.excluded_by_slot[slot] = values[0] if len(values) == 1 else tuple(values)

    @staticmethod
    def _remove_structured_exclusion(
        state: CurrentState,
        slot: str,
        value: object,
    ) -> None:
        current = state.excluded_by_slot.get(slot)
        if current is None:
            return
        values = list(current) if isinstance(current, tuple) else [current]
        remaining = [item for item in values if item != value]
        if not remaining:
            state.excluded_by_slot.pop(slot, None)
        elif len(remaining) == 1:
            state.excluded_by_slot[slot] = remaining[0]
        else:
            state.excluded_by_slot[slot] = tuple(remaining)

    @staticmethod
    def _apply_mapping(target: dict[str, StateValue], update: StateUpdate) -> None:
        if update.op is UpdateOperation.CLEAR:
            target.pop(update.slot, None)
            return
        value = update.value
        if value is None:
            raise ValueError(f"{update.op.value} requires a value")
        if update.op is UpdateOperation.SET:
            target[update.slot] = value
            return
        if update.op is UpdateOperation.ADD:
            current = target.get(update.slot)
            if current is None:
                target[update.slot] = (value,)
                return
            values = list(current) if isinstance(current, tuple) else [current]
            if value not in values:
                values.append(value)
            target[update.slot] = tuple(values)
            return
        current = target.get(update.slot)
        if current is None:
            return
        values = list(current) if isinstance(current, tuple) else [current]
        remaining = [item for item in values if item != value]
        if not remaining:
            target.pop(update.slot, None)
        elif len(remaining) == 1:
            target[update.slot] = remaining[0]
        else:
            target[update.slot] = tuple(remaining)

    @staticmethod
    def _apply_excluded(state: CurrentState, update: StateUpdate) -> None:
        if update.op is UpdateOperation.CLEAR:
            state.excluded.clear()
            return
        value = update.value
        if value is None:
            raise ValueError(f"{update.op.value} requires a value")
        if update.op is UpdateOperation.SET:
            state.excluded[:] = [value]
        elif update.op is UpdateOperation.ADD:
            if value not in state.excluded:
                state.excluded.append(value)
        elif update.op is UpdateOperation.REMOVE:
            state.excluded[:] = [item for item in state.excluded if item != value]

    @staticmethod
    def _validate_cross_field_constraints(state: CurrentState) -> None:
        minimum = state.constraints.get("price_min")
        maximum = state.constraints.get("price_max")
        if isinstance(minimum, tuple) or isinstance(maximum, tuple):
            raise ValueError("price bounds cannot contain multiple values")
        if minimum is not None and maximum is not None and float(minimum) > float(maximum):
            raise ValueError("price_min cannot exceed price_max")
        for slot, excluded in state.excluded_by_slot.items():
            target = state.constraints if slot in CONSTRAINT_SLOTS else state.preferences
            current = target.get(slot)
            positive = set(current if isinstance(current, tuple) else (() if current is None else (current,)))
            negative = set(excluded if isinstance(excluded, tuple) else (excluded,))
            overlap = positive & negative
            if overlap:
                raise ValueError(f"{slot} cannot contain the same positive and excluded value")
            if state.attribute_status.get(slot) is AttributeStatus.NO_PREFERENCE:
                raise ValueError(f"{slot} cannot be declined and excluded at the same time")

    @staticmethod
    def _slot_snapshot(state: CurrentState, slot: str) -> object:
        if slot == "intent":
            return state.intent.value
        if slot == "excluded":
            return list(state.excluded) or None
        if slot in DIRECT_SLOTS:
            return getattr(state, slot)
        if slot in CONSTRAINT_SLOTS:
            value = state.constraints.get(slot)
        elif slot in PREFERENCE_SLOTS:
            value = state.preferences.get(slot)
        else:
            value = None
        status = state.attribute_status.get(slot, AttributeStatus.UNKNOWN)
        excluded = state.excluded_by_slot.get(slot)
        if value is None and status is AttributeStatus.UNKNOWN and excluded is None:
            return None
        return {
            "value": list(value) if isinstance(value, tuple) else value,
            "status": status.value,
            "excluded": (
                list(excluded) if isinstance(excluded, tuple) else ([] if excluded is None else [excluded])
            ),
        }

    @classmethod
    def _demand_snapshot(cls, state: CurrentState) -> dict[str, object]:
        slots = {
            "intent",
            "excluded",
            *DIRECT_SLOTS,
            *CONSTRAINT_SLOTS,
            *PREFERENCE_SLOTS,
            *state.attribute_status,
            *state.excluded_by_slot,
        }
        return {slot: cls._slot_snapshot(state, slot) for slot in slots}

    @staticmethod
    def _build_delta(
        before: dict[str, object],
        after: dict[str, object],
    ) -> StateDelta:
        changed: list[str] = []
        removed: list[str] = []
        for slot in sorted(set(before) | set(after)):
            old = before.get(slot)
            new = after.get(slot)
            if old == new:
                continue
            if new is None:
                removed.append(slot)
            else:
                changed.append(slot)
        return StateDelta(tuple(changed), tuple(removed))

    @staticmethod
    def _append_history(
        state: CurrentState,
        parsed: ParseUpdate,
        before: dict[str, object],
        after: dict[str, object],
        delta: StateDelta,
    ) -> None:
        operations = {update.slot: update.op.value for update in parsed.updates}
        if parsed.intent is not None:
            operations["intent"] = "set"
        default_operation = "reset_task" if parsed.reset_task else "derived"
        for slot in (*delta.changed_slots, *delta.removed_slots):
            state.change_history.append(StateChange(
                turn=parsed.source_turn,
                slot=slot,
                old=before.get(slot),
                new=after.get(slot),
                op=operations.get(slot, default_operation),
            ))
        state.change_history[:] = state.change_history[-HISTORY_LIMIT:]
