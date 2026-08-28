from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace

from starter.memory.memory_store import InMemoryStore, MemoryStore
from starter.memory.models import CurrentState, Intent, ParseUpdate, RetrievalState, StateDelta
from starter.memory.state_manager import (
    OutOfOrderTurnError,
    StateManager,
    TurnConflictError,
)


class StateNotFoundError(KeyError):
    pass


@dataclass(slots=True)
class MemoryMetrics:
    update_count: int = 0
    applied_update_count: int = 0
    noop_count: int = 0
    rollback_count: int = 0
    turn_conflict_count: int = 0
    out_of_order_count: int = 0
    reset_count: int = 0
    total_update_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        average = self.total_update_latency_ms / self.update_count if self.update_count else 0.0
        return {
            "update_count": self.update_count,
            "applied_update_count": self.applied_update_count,
            "noop_count": self.noop_count,
            "rollback_count": self.rollback_count,
            "turn_conflict_count": self.turn_conflict_count,
            "out_of_order_count": self.out_of_order_count,
            "reset_count": self.reset_count,
            "memory_update_latency_ms": round(average, 6),
        }


class MemoryService:
    """Single public entry point for state CRUD and deterministic merge behavior."""

    def __init__(
        self,
        store: MemoryStore | None = None,
        state_manager: StateManager | None = None,
    ) -> None:
        self.store = store if store is not None else InMemoryStore()
        self.state_manager = state_manager if state_manager is not None else StateManager()
        self._metrics = MemoryMetrics()

    def get_state(self, session_id: str) -> CurrentState:
        state = self.store.get(session_id)
        if state is None:
            raise StateNotFoundError(session_id)
        return state

    def apply_update(self, parsed: ParseUpdate | dict) -> CurrentState:
        updated, _ = self.apply_update_with_delta(parsed)
        return updated

    def apply_update_with_delta(
        self,
        parsed: ParseUpdate | dict,
    ) -> tuple[CurrentState, StateDelta]:
        started = time.perf_counter()
        self._metrics.update_count += 1
        try:
            update = parsed if isinstance(parsed, ParseUpdate) else ParseUpdate.from_dict(parsed)
            current = self.get_state(update.session_id)
            updated, delta = self.state_manager.apply_update_with_delta(current, update)
            self.store.save(updated)
        except TurnConflictError:
            self._metrics.turn_conflict_count += 1
            self._metrics.rollback_count += 1
            raise
        except OutOfOrderTurnError:
            self._metrics.out_of_order_count += 1
            self._metrics.rollback_count += 1
            raise
        except Exception:
            self._metrics.rollback_count += 1
            raise
        else:
            if updated.updated_at == current.updated_at:
                self._metrics.noop_count += 1
            else:
                self._metrics.applied_update_count += 1
                if update.reset_task:
                    self._metrics.reset_count += 1
            return updated.clone(), delta
        finally:
            self._metrics.total_update_latency_ms += (
                time.perf_counter() - started
            ) * 1000

    def get_retrieval_state(self, session_id: str) -> RetrievalState:
        return self.get_state(session_id).to_retrieval_state()

    def get_metrics(self, session_id: str | None = None) -> dict:
        result = self._metrics.to_dict()
        if session_id is None:
            return result
        state = self.get_state(session_id)
        retrieval = state.to_retrieval_state()
        result.update({
            "state_size_bytes": len(json.dumps(state.to_dict(), separators=(",", ":"))),
            "active_constraint_count": (
                int(retrieval.category is not None)
                + int(retrieval.product_type is not None)
                + len(retrieval.hard_constraints)
                + len(retrieval.soft_preferences)
                + len(retrieval.excluded)
            ),
        })
        return result

    def mark_attribute_asked(self, session_id: str, attribute: str) -> CurrentState:
        current = self.get_state(session_id)
        updated = self.state_manager.mark_attribute_asked(current, attribute)
        self.store.save(updated)
        return updated.clone()

    def get_or_record_asked_attribute(
        self,
        session_id: str,
        source_turn: int,
        candidates: tuple[str, ...] | list[str],
    ) -> str | None:
        current = self.get_state(session_id)
        updated, attribute = self.state_manager.get_or_record_asked_attribute(
            current,
            source_turn,
            candidates,
        )
        self.store.save(updated)
        return attribute

    def next_unasked_attribute(
        self,
        session_id: str,
        candidates: tuple[str, ...] | list[str],
    ) -> str | None:
        state = self.get_state(session_id)
        for attribute in candidates:
            if not state.was_asked(attribute):
                return attribute
        return None

    def start_new_task(
        self,
        session_id: str,
        intent: Intent | str | None = None,
    ) -> CurrentState:
        return self.apply_update(ParseUpdate(
            session_id=session_id,
            intent=None if intent is None else Intent(intent),
            reset_task=True,
        ))

    def save_state(self, state: CurrentState) -> CurrentState:
        if not isinstance(state, CurrentState):
            raise TypeError("state must be a CurrentState")
        validated = replace(state)
        self.state_manager.validate_state(validated)
        current = self.store.get(validated.session_id)
        if current is not None:
            if validated.updated_at < current.updated_at:
                raise ValueError("updated_at cannot move backwards")
            if validated.task_version < current.task_version:
                raise ValueError("task_version cannot move backwards")
            for turn, signature in current.applied_turn_signatures.items():
                if validated.applied_turn_signatures.get(turn) != signature:
                    raise ValueError("applied turn signatures cannot be removed or changed")
            for turn, attribute in current.asked_attribute_by_turn.items():
                if validated.asked_attribute_by_turn.get(turn) != attribute:
                    raise ValueError("asked attributes by turn cannot be removed or changed")
            if (
                validated.updated_at == current.updated_at
                and validated.to_dict() != current.to_dict()
            ):
                raise ValueError("changed state requires a newer updated_at")
        self.store.save(validated)
        return self.get_state(state.session_id)

    def reset_state(
        self,
        session_id: str,
        intent: Intent | str = Intent.UNKNOWN,
    ) -> CurrentState:
        state = CurrentState(session_id=session_id, intent=Intent(intent))
        self.store.save(state)
        self._metrics.reset_count += 1
        return state.clone()

    def delete_state(self, session_id: str) -> None:
        self.store.delete(session_id)
