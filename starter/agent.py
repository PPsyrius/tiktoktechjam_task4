from __future__ import annotations

from pathlib import Path

from starter.memory import (
    AttributeStatus,
    CurrentState,
    MemoryService,
    ParseUpdate,
    StateNotFoundError,
    StateUpdate,
    UpdateOperation,
)
from starter.understanding import parse_requirement
from starter.reranker import rank_candidates
from starter.retrieval import Retriever, search_context_from_state


CLARIFY_ORDER = (
    "color",
    "material",
    "brand",
    "size",
    "style",
    "feature",
    "use_case",
    "budget",
)
RETRIEVAL_POOL_SIZE = 50


class Agent:
    """Session memory, BM25 retrieval, and constraint-aware reranking."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.retriever = Retriever(self.catalog_path)
        self.memory = MemoryService()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # CurrentState stores the active shopping task, not catalog records or
        # raw user history. Replacing it is the session-reset invariant.
        self.memory.reset_state(session_id)

    def apply_update(self, parsed: ParseUpdate | dict) -> CurrentState:
        """Apply a structured Dialogue-module update to session memory."""
        return self.memory.apply_update(parsed)

    def get_state(self, session_id: str) -> CurrentState:
        return self.memory.get_state(session_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        try:
            self.memory.get_state(session_id)
        except StateNotFoundError as error:
            raise RuntimeError("reset must be called before respond") from error

        retrieval_state = self.memory.get_retrieval_state(session_id)
        parse_result = parse_requirement(
            session_id,
            user_message,
            turn,
            search_context=retrieval_state,
        )
        parsed = parse_result.parsed
        if parsed is not None:
            if parsed.reset_task:
                current = self.memory.get_state(session_id)
                if current.category and not any(update.slot == "category" for update in parsed.updates):
                    parsed = ParseUpdate(
                        session_id=parsed.session_id,
                        intent=parsed.intent,
                        updates=(
                            StateUpdate("category", UpdateOperation.SET, current.category),
                            *parsed.updates,
                        ),
                        reset_task=True,
                        source_turn=parsed.source_turn,
                    )
            try:
                self.memory.apply_update(parsed)
            except (TypeError, ValueError):
                pass
            else:
                retrieval_state = self.memory.get_retrieval_state(session_id)

        search_context = search_context_from_state(retrieval_state)
        candidate_pool = self.retriever.retrieve(
            user_message,
            search_context=retrieval_state,
            pool_size=max(RETRIEVAL_POOL_SIZE, top_k),
        )
        ranking = rank_candidates(search_context, candidate_pool, top_k=top_k)
        recommendations = [
            {"parent_asin": item.parent_asin, "score": item.final_score}
            for item in ranking.items
        ]

        state = self.memory.get_state(session_id)
        clarify_candidates = [
            attribute
            for attribute in CLARIFY_ORDER
            if state.status_for(attribute) is AttributeStatus.UNKNOWN
        ]
        ask_attribute = self.memory.get_or_record_asked_attribute(
            session_id,
            turn,
            clarify_candidates,
        )
        if ask_attribute:
            message = f"Do you have a {ask_attribute} preference?"
        else:
            message = "Here are the closest matches I found."
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {
                "prompt_tokens": parse_result.prompt_tokens,
                "completion_tokens": parse_result.completion_tokens,
            },
        }
