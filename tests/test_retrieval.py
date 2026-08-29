from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.reranker import rank_candidates
from starter.retrieval import Retriever, search_context_from_state


def write_catalog(path: Path) -> None:
    products = [
        {
            "parent_asin": "A_BASIC",
            "title": "Basic shoe",
            "features": [],
            "description": [],
            "categories": ["Shoes"],
            "details": {},
            "store": "Example",
        },
        {
            "parent_asin": "B_BLACK",
            "title": "Black running shoe",
            "features": ["cotton", "black"],
            "description": [],
            "categories": ["Shoes"],
            "details": {},
            "store": "Example",
        },
        {
            "parent_asin": "C_RED",
            "title": "Red polyester jacket",
            "features": ["polyester"],
            "description": [],
            "categories": ["Jackets"],
            "details": {},
            "store": "Example",
        },
    ]
    path.write_text("".join(json.dumps(product) + "\n" for product in products), encoding="utf-8")


class RetrievalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        write_catalog(self.catalog_path)
        self.retriever = Retriever(self.catalog_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_candidate_pool_matches_ranking_contract(self) -> None:
        pool = self.retriever.retrieve("running shoe")
        self.assertGreaterEqual(len(pool), 1)
        candidate = pool[0]
        self.assertIn("parent_asin", candidate)
        self.assertIn("title", candidate)
        self.assertIn("source_scores", candidate)
        self.assertGreaterEqual(candidate["source_scores"]["bm25"], 0.0)
        self.assertEqual(candidate["source_ranks"]["bm25"], 1)

    def test_retrieval_plus_ranking_promotes_constraint_match(self) -> None:
        search_context = {"hard_constraints": ["cotton", "black"]}
        pool = self.retriever.retrieve("shoe", search_context=search_context, pool_size=10)
        result = rank_candidates(search_context, pool, top_k=10)
        self.assertEqual(result.items[0].parent_asin, "B_BLACK")

    def test_search_context_flattens_memory_state(self) -> None:
        context = search_context_from_state({
            "intent": "buying",
            "category": "running shoes",
            "product_type": None,
            "hard_constraints": {"color": ["black"], "price_max": 120},
            "soft_preferences": {"material": ["cotton"]},
            "excluded": {},
        })
        self.assertEqual(context["hard_constraints"], ["black", "cotton"])
        self.assertEqual(context["intent"], "buying")


class AgentRetrievalPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["LLM_PARSER"] = "0"
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        write_catalog(self.catalog_path)
        self.agent = Agent(self.catalog_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_agent_parses_message_then_ranks_with_section5(self) -> None:
        self.agent.reset("s001", {})
        response = self.agent.respond(
            "s001",
            "I'm looking for running shoes. A key requirement is: black.",
            1,
            10,
        )
        state = self.agent.get_state("s001")
        self.assertEqual(state.category, "running shoes")
        self.assertEqual(state.constraints["color"], "black")
        self.assertEqual(response["recommendations"][0]["parent_asin"], "B_BLACK")
        self.assertIn(response["ask_attribute"], {
            "color", "material", "brand", "size", "style", "feature", "use_case", "budget", None,
        })
        # color was specified, so the first clarification should not be color
        self.assertNotEqual(response["ask_attribute"], "color")


if __name__ == "__main__":
    unittest.main()
