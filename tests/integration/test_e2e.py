from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.catalog import ProductStore
from starter.memory.models import AttributeStatus


def write_small_catalog(path: Path) -> None:
    products = [
        {
            "parent_asin": "A_BASIC",
            "title": "Basic shoe",
            "features": [],
            "description": [],
            "categories": ["Shoes"],
            "details": {},
            "store": "Example",
            "price": 50,
        },
        {
            "parent_asin": "B_BLACK",
            "title": "Black running shoe cotton",
            "features": ["cotton", "black"],
            "description": [],
            "categories": ["Shoes", "Running"],
            "details": {"Color": "Black", "Material": "Cotton"},
            "store": "Example",
            "price": 60,
        },
        {
            "parent_asin": "C_RED",
            "title": "Red polyester jacket",
            "features": ["polyester"],
            "description": [],
            "categories": ["Jackets"],
            "details": {"Color": "Red", "Material": "Polyester"},
            "store": "Example",
            "price": 120,
        },
        {
            "parent_asin": "D_NOPRICE",
            "title": "Blue hiking jacket no price",
            "features": ["hiking"],
            "description": [],
            "categories": ["Jackets"],
            "details": {"Color": "Blue"},
            "store": "Example",
            "price": None,
        },
        {
            "parent_asin": "E_BROWSE",
            "title": "B browsing canvas sneaker",
            "features": [],
            "description": [],
            "categories": ["Shoes"],
            "details": {},
            "store": "Example",
            "price": 30,
        },
    ]
    import json

    path.write_text("".join(json.dumps(p) + "\n" for p in products), encoding="utf-8")


class IntegrationPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["LLM_PARSER"] = "0"
        self.tmp = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.tmp.name) / "catalog.jsonl"
        write_small_catalog(self.catalog_path)
        self.agent = Agent(self.catalog_path)

    def tearDown(self) -> None:
        self.agent.close()
        self.tmp.cleanup()

    def test_e2e_buying_pipeline(self) -> None:
        self.agent.reset("s1", {})
        response = self.agent.respond(
            "s1",
            "I'm looking for running shoes. A key requirement is: black.",
            1,
            10,
        )
        state = self.agent.get_state("s1")
        self.assertEqual(state.category, "running shoes")
        self.assertEqual(state.constraints["color"], "black")
        self.assertGreaterEqual(len(response["recommendations"]), 1)
        self.assertEqual(response["recommendations"][0]["parent_asin"], "B_BLACK")
        self.assertNotEqual(response["ask_attribute"], "color")
        self.assertIn(response["ask_attribute"], {"other", "material", "feature", "style", "use_case", "size", "brand", "budget", None})

    def test_e2e_browsing_pipeline(self) -> None:
        self.agent.reset("s2", {})
        response = self.agent.respond(
            "s2",
            "I'm still exploring running shoes, keeping my options open.",
            1,
            10,
        )
        state = self.agent.get_state("s2")
        self.assertEqual(state.intent.value, "browsing")
        self.assertGreaterEqual(len(response["recommendations"]), 1)

    def test_e2e_boundary_no_preference(self) -> None:
        self.agent.reset("s3", {})
        self.agent.respond("s3", "I'm looking for jackets.", 1, 10)
        self.agent.respond("s3", "No preference for color is fine.", 2, 10)
        state = self.agent.get_state("s3")
        self.assertEqual(state.status_for("color"), AttributeStatus.NO_PREFERENCE)
        # not re-asked immediately
        response = self.agent.respond("s3", "Anything else?", 3, 10)
        self.assertNotEqual(response.get("ask_attribute"), "color")

    def test_session_isolation(self) -> None:
        self.agent.reset("a", {})
        self.agent.reset("b", {})
        self.agent.respond("a", "I'm looking for running shoes. A key requirement is: black.", 1, 10)
        self.agent.respond("b", "I'm looking for jackets. A key requirement is: red.", 1, 10)
        state_a = self.agent.get_state("a")
        state_b = self.agent.get_state("b")
        self.assertEqual(state_a.constraints["color"], "black")
        self.assertEqual(state_b.constraints["color"], "red")
        self.agent.reset("a", {})
        self.assertNotIn("color", self.agent.get_state("a").constraints)

    def test_override_invalidation_starts_new_task(self) -> None:
        self.agent.reset("ov", {})
        self.agent.respond("ov", "I'm looking for running shoes. A key requirement is: black.", 1, 10)
        self.agent.respond(
            "ov",
            "Actually, ignore my earlier preference. I'm looking for jackets. A key requirement is: polyester.",
            2,
            10,
        )
        state = self.agent.get_state("ov")
        self.assertEqual(state.task_version, 1)
        self.assertEqual(state.category, "jackets")
        self.assertNotIn("color", state.constraints)
        self.assertEqual(state.preferences["material"], "polyester")

    def test_same_task_override_preserves_category(self) -> None:
        self.agent.reset("ov2", {})
        self.agent.respond("ov2", "I'm looking for shirts. Button closure.", 1, 10)
        self.agent.respond("ov2", "For that, what matters is: cotton.", 2, 10)
        self.agent.respond("ov2", "Actually, ignore my earlier preference. What I need is: cotton.", 3, 10)
        state = self.agent.get_state("ov2")
        self.assertEqual(state.task_version, 0)
        self.assertEqual(state.category, "shirts")

    def test_determinism(self) -> None:
        self.agent.reset("det", {})
        r1 = self.agent.respond("det", "I'm looking for running shoes. A key requirement is: black.", 1, 10)
        # second agent same catalog should give same ranking for same message
        tmp2 = tempfile.TemporaryDirectory()
        path2 = Path(tmp2.name) / "catalog.jsonl"
        write_small_catalog(path2)
        agent2 = Agent(path2)
        try:
            agent2.reset("det", {})
            r2 = agent2.respond("det", "I'm looking for running shoes. A key requirement is: black.", 1, 10)
            self.assertEqual(r1["recommendations"], r2["recommendations"])
        finally:
            agent2.close()
            tmp2.cleanup()

    def test_missing_price_not_penalized(self) -> None:
        self.agent.reset("price", {})
        self.agent.respond("price", "I'm looking for jackets.", 1, 10)
        response = self.agent.respond("price", "Must be under $80.", 2, 10)
        rec_ids = [r["parent_asin"] for r in response["recommendations"]]
        # D_NOPRICE price None -> unknown should not be filtered; A_BASIC price 50 -> pass
        self.assertIn("D_NOPRICE", rec_ids)
        self.assertIn("A_BASIC", rec_ids)
        self.assertEqual(len(rec_ids), len(set(rec_ids)))
        self.assertLessEqual(len(rec_ids), 10)

    def test_candidate_history_admission(self) -> None:
        self.agent.reset("hist", {})
        r1 = self.agent.respond("hist", "I'm looking for running shoes.", 1, 10)
        first_top = r1["recommendations"][0]["parent_asin"]
        r2 = self.agent.respond("hist", "A key requirement is: black.", 2, 10)
        second_ids = {r["parent_asin"] for r in r2["recommendations"]}
        # history admits prior candidate even when turn2 narrows query
        self.assertIn(first_top, second_ids)

    def test_productstore_not_reparsed(self) -> None:
        cache_dir = Path(self.tmp.name) / "cache"
        store1 = ProductStore.from_jsonl(self.catalog_path, cache_dir=cache_dir)
        self.assertFalse(store1.cache_hit)
        fp1 = store1.fingerprint
        store2 = ProductStore.from_jsonl(self.catalog_path, cache_dir=cache_dir)
        self.assertTrue(store2.cache_hit)
        self.assertEqual(fp1, store2.fingerprint)
        self.assertEqual(store1.fingerprint, self.agent.store.fingerprint)


if __name__ == "__main__":
    unittest.main()
