from __future__ import annotations

from pathlib import Path
from retrieval import HybridRetriever, ProductStore, SearchContext


class Agent:
    """Official interface adapter. Parsing, Memory and final reranking are still stubs.

    Teammates should replace context construction and the Top-10 slice, not retrieval.
    A ready retriever can be injected to share task 1's ProductStore.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl", *,
                 retriever=None, cache_dir=None, candidate_limit=100) -> None:
        self.catalog_path = Path(catalog_path)
        self.retriever = retriever or HybridRetriever(
            ProductStore.from_jsonl(self.catalog_path), cache_dir=cache_dir)
        if (not isinstance(candidate_limit, int) or isinstance(candidate_limit, bool)
                or not 1 <= candidate_limit <= self.retriever.config.max_candidates):
            raise ValueError("candidate_limit exceeds retriever capacity")
        self.candidate_limit = candidate_limit
        self._sessions: set[str] = set()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # Task 3 will own actual session state and profile handling.
        self._sessions.add(session_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        # Integration placeholder: replace with parser + memory's effective snapshot.
        context = SearchContext(queries=(user_message,))
        pool = self.retriever.retrieve(context, limit=self.candidate_limit)
        # Integration placeholder: task 5 ranks the full pool before producing Top 10.
        recommendations = [{"parent_asin": c.parent_asin}
                           for c in pool.candidates[:max(0, min(top_k, 10))]]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def close(self):
        self.retriever.close()
