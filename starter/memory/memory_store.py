from __future__ import annotations

from typing import Protocol

from starter.memory.models import CurrentState


class MemoryStore(Protocol):
    def get(self, session_id: str) -> CurrentState | None:
        ...

    def save(self, state: CurrentState) -> None:
        ...

    def delete(self, session_id: str) -> None:
        ...


class InMemoryStore:
    """Session-scoped store that never exposes its mutable internal objects."""

    def __init__(self) -> None:
        self._states: dict[str, CurrentState] = {}

    def get(self, session_id: str) -> CurrentState | None:
        state = self._states.get(session_id)
        return state.clone() if state is not None else None

    def save(self, state: CurrentState) -> None:
        self._states[state.session_id] = state.clone()

    def delete(self, session_id: str) -> None:
        self._states.pop(session_id, None)
