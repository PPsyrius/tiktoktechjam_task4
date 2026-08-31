from __future__ import annotations

import json
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.catalog import CatalogLoader, ProductStore


class SystemSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog_path = Path("ParticipationKit/catalog.jsonl.gz")
        cls.dataset_path = Path("data/public_set.jsonl")
        if not cls.catalog_path.exists() or not cls.dataset_path.exists():
            raise unittest.SkipTest("real catalog/dataset not found")

    def test_catalog_fingerprint_stable(self) -> None:
        fp1 = CatalogLoader(self.catalog_path).source_fingerprint()
        fp2 = CatalogLoader(self.catalog_path).source_fingerprint()
        self.assertEqual(fp1, fp2)
        self.assertEqual(len(fp1), 64)

    def test_productstore_loads_50000(self) -> None:
        store = ProductStore.from_jsonl(self.catalog_path)
        self.assertEqual(len(store), 50000)
        self.assertEqual(len(store.fingerprint), 64)

    def test_agent_smoke_3_cases(self) -> None:
        with self.dataset_path.open("rt", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()][:3]
        agent = Agent(self.catalog_path)
        self.addCleanup(agent.close)
        for sample in samples:
            sid = sample["sample_id"]
            agent.reset(sid, sample.get("user_profile", {}))
            resp = agent.respond(sid, "I'm looking for running shoes. A key requirement is: black.", 1, 10)
            self.assertIn("recommendations", resp)
            self.assertIn("message", resp)
            self.assertLessEqual(len(resp["recommendations"]), 10)
            for rec in resp["recommendations"]:
                self.assertIn(rec["parent_asin"], agent.store)
                self.assertIsInstance(rec["score"], float)

    def test_no_absolute_paths_in_source(self) -> None:
        roots = [Path("starter"), Path("retrieval"), Path("scripts")]
        bad = []
        for root in roots:
            if not root.exists():
                continue
            for p in root.rglob("*.py"):
                text = p.read_text(encoding="utf-8", errors="ignore")
                if "C:\\" in text or "/home/" in text:
                    for i, line in enumerate(text.splitlines(), 1):
                        if ("C:\\" in line or "/home/" in line) and "example" not in line.lower():
                            bad.append(f"{p}:{i}:{line.strip()}")
        self.assertEqual(bad, [], f"absolute paths found: {bad[:3]}")

    def test_determinism_across_resets(self) -> None:
        agent = Agent(self.catalog_path)
        self.addCleanup(agent.close)
        agent.reset("sys_det", {})
        r1 = agent.respond("sys_det", "I'm looking for jackets.", 1, 10)
        agent2 = Agent(self.catalog_path)
        self.addCleanup(agent2.close)
        agent2.reset("sys_det", {})
        r2 = agent2.respond("sys_det", "I'm looking for jackets.", 1, 10)
        self.assertEqual(r1["recommendations"], r2["recommendations"])
