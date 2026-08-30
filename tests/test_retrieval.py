from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.catalog import ProductStore
from starter.reranker import rank_candidates
from starter.retrieval import Retriever, search_context_from_state
from starter.snippet_index import SnippetIndex


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
        self.retriever.close()
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

    def test_retriever_can_consume_product_store_without_reading_catalog(self) -> None:
        store = ProductStore.from_jsonl(self.catalog_path)
        retriever = Retriever(
            self.catalog_path.with_name("does-not-exist.jsonl"),
            store=store,
        )
        self.addCleanup(retriever.close)

        self.assertIs(retriever.store, store)
        self.assertEqual(retriever.retrieve("running shoe")[0]["parent_asin"], "B_BLACK")

    def test_snippet_index_reads_frozen_product_raw_fields(self) -> None:
        store = ProductStore.from_records(({
            "parent_asin": "A",
            "features": ["Rare Requirement Phrase"],
            "details": {"Closure Type": "Button"},
        },))
        index = SnippetIndex(store)

        self.assertEqual(index.search(("Rare Requirement Phrase",))[0]["parent_asin"], "A")
        self.assertEqual(index.search(("Closure Type: Button",))[0]["parent_asin"], "A")


class AgentRetrievalPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["LLM_PARSER"] = "0"
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        write_catalog(self.catalog_path)
        self.agent = Agent(self.catalog_path)

    def tearDown(self) -> None:
        self.agent.close()
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
            "other", "color", "material", "brand", "size", "style", "feature", "use_case", "budget", None,
        })
        # color was specified, so the first clarification should not be color
        self.assertNotEqual(response["ask_attribute"], "color")

    def test_same_task_override_preserves_other_disclosed_constraints(self) -> None:
        self.agent.reset("override", {})
        self.agent.respond(
            "override",
            "I'm looking for shirts. Button closure.",
            1,
            10,
        )
        self.agent.respond(
            "override",
            "For that, what matters is: cotton; 60% Cotton, 40% Polyester.",
            2,
            10,
        )

        response = self.agent.respond(
            "override",
            "Actually, ignore my earlier preference. What I need is: cotton.",
            3,
            10,
        )
        state = self.agent.get_state("override")

        self.assertEqual(state.task_version, 0)
        self.assertEqual(state.category, "shirts")
        self.assertEqual(state.preferences["material"], "cotton")
        self.assertIn("60% Cotton, 40% Polyester", state.preferences["feature"])
        self.assertEqual(response["ask_attribute"], "other")

    def test_feature_override_does_not_clear_independent_features(self) -> None:
        self.agent.reset("feature-override", {})
        self.agent.respond(
            "feature-override",
            "I'm looking for watches. Stainless Steel Band.",
            1,
            10,
        )
        self.agent.respond(
            "feature-override",
            "For that, what matters is: Water Resistant; 3 Year Battery.",
            2,
            10,
        )

        self.agent.respond(
            "feature-override",
            "Actually, ignore my earlier preference. What I need is: Water Resistant.",
            3,
            10,
        )
        features = self.agent.get_state("feature-override").preferences["feature"]

        self.assertIn("Water Resistant", features)
        self.assertIn("3 Year Battery", features)

    def test_override_with_new_category_starts_a_new_task(self) -> None:
        self.agent.reset("new-task", {})
        self.agent.respond(
            "new-task",
            "I'm looking for running shoes. A key requirement is: black.",
            1,
            10,
        )

        self.agent.respond(
            "new-task",
            "Actually, ignore my earlier preference. I'm looking for jackets. "
            "A key requirement is: polyester.",
            2,
            10,
        )
        state = self.agent.get_state("new-task")

        self.assertEqual(state.task_version, 1)
        self.assertEqual(state.category, "jackets")
        self.assertNotIn("color", state.constraints)
        self.assertEqual(state.preferences["material"], "polyester")

    def test_candidate_history_admits_prior_candidate_without_old_score(self) -> None:
        current = [{
            "parent_asin": "C_RED",
            "source_scores": {"bm25": 9.0},
            "source_ranks": {"bm25": 1},
        }]
        self.agent._candidate_history["stable"] = (0, ("B_BLACK",))

        admitted = self.agent._admit_candidate_history("stable", 0, current)
        historical = next(item for item in admitted if item["parent_asin"] == "B_BLACK")

        self.assertEqual(historical["source_scores"], {})
        self.assertEqual(historical["source_ranks"], {})

    def test_candidate_history_is_discarded_for_a_new_task(self) -> None:
        current = [{
            "parent_asin": "C_RED",
            "source_scores": {"bm25": 9.0},
            "source_ranks": {"bm25": 1},
        }]
        self.agent._candidate_history["changed"] = (0, ("B_BLACK",))

        admitted = self.agent._admit_candidate_history("changed", 1, current)

        self.assertEqual([item["parent_asin"] for item in admitted], ["C_RED"])

    def test_reset_discards_candidate_history(self) -> None:
        self.agent._candidate_history["reset-history"] = (0, ("B_BLACK",))

        self.agent.reset("reset-history", {})

        self.assertNotIn("reset-history", self.agent._candidate_history)
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from retrieval import (Candidate, Constraint, HybridRetriever, ProductStore,
                       RetrievalConfig, SearchContext, SourceHit)
from retrieval.bm25_retriever import BM25Retriever, terms
from retrieval.bm25_retriever import BM25CacheError
from retrieval.merge import interleave
from retrieval.product_store import constraint_status
from retrieval.structured_retriever import StructuredRetriever
from starter.agent import Agent


def records():
    return [
        {"parent_asin": "A", "title": "black running shoes", "categories": ["Shoes"],
         "price": 50, "details": {"Color": "Black", "Material": "leather"}},
        {"parent_asin": "B", "title": "red running shoes", "categories": ["Shoes"],
         "price": 120, "details": {"Color": "Red", "Material": "cotton"}},
        {"parent_asin": "C", "title": "black cushioned sneakers", "categories": ["Shoes"],
         "price": None, "details": {"Color": "Black"}},
        {"parent_asin": "D", "title": "blue hiking jacket", "categories": ["Jackets"],
         "price": "$40.00", "attributes": {"color": ["blue"], "material": ["nylon"]}},
    ]


def ids(candidates):
    return [c.parent_asin for c in candidates]


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = ProductStore.from_records(records())

    def test_explicit_normalization_and_unknown(self):
        self.assertEqual(self.store["A"].attributes["color"], ("black",))
        self.assertEqual(self.store["D"].price, 40)
        self.assertEqual(constraint_status(self.store["C"], Constraint("price", maximum=80)), "unknown")
        self.assertEqual(constraint_status(self.store["B"], Constraint("price", maximum=80)), "fail")
        self.assertEqual(constraint_status(self.store["C"], Constraint("material", ("cotton",))), "unknown")

    def test_inferred_material_is_not_promoted_to_a_structured_fact(self):
        store = ProductStore.from_records([{"parent_asin": "X", "title": "leather shoe", "store": "shop"}])
        self.assertNotIn("brand", store["X"].attributes)
        self.assertNotIn("material", store["X"].attributes)
        self.assertEqual(store["X"].material, ("leather",))

    def test_store_is_read_only_after_fingerprinting(self):
        with self.assertRaises(TypeError):
            self.store.products["A"] = self.store["B"]
        with self.assertRaises(TypeError):
            self.store["A"].attributes["color"] = ("red",)

    def test_bad_prices_are_unknown(self):
        for value in (None, True, "unknown", "10-20", float("nan"), float("inf"), -2):
            with self.subTest(value=value):
                self.assertIsNone(ProductStore.from_records([{"parent_asin": "X", "price": value}])["X"].price)

    def test_reject_invalid_ids(self):
        for data in ([{"parent_asin": None}], [{"parent_asin": " "}], records() + records()):
            with self.assertRaises(ValueError):
                ProductStore.from_records(data)

    def test_gzip_and_jsonl_have_same_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "catalog.jsonl"
            zipped = Path(directory) / "catalog.jsonl.gz"
            content = "".join(json.dumps(r) + "\n" for r in records())
            plain.write_text(content)
            with gzip.open(zipped, "wt") as handle:
                handle.write(content)
            self.assertEqual(ProductStore.from_jsonl(plain).fingerprint,
                             ProductStore.from_jsonl(zipped).fingerprint)

    def test_boundary_validation(self):
        for make in (lambda: SearchContext(queries="shoe"), lambda: SearchContext(mode="invalid"),
                     lambda: SearchContext(constraints=({},)),
                     lambda: Constraint("ground_truth", ("A",)),
                     lambda: Constraint("color", "black"),
                     lambda: Constraint("price", maximum=-1),
                     lambda: Constraint("price", minimum=20, maximum=10),
                     lambda: Constraint("price", maximum=float("nan"))):
            with self.assertRaises((ValueError, TypeError)):
                make()


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.store = ProductStore.from_records(records())
        self.retriever = HybridRetriever(self.store)
        self.addCleanup(self.retriever.close)

    def test_bm25_matches_original_sql(self):
        query = "I'm looking for black running shoes."
        bm25 = self.retriever.bm25
        expression = " OR ".join('"' + t + '"' for t in terms(query))
        expected = bm25.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT 3",
            (expression,)).fetchall()
        actual = bm25.search(SearchContext(queries=(query,)), 3)
        self.assertEqual(ids(actual), [row[0] for row in expected])
        self.assertFalse(actual[0].hits[0].higher_is_better)

    def test_queries_are_data_not_fts_syntax(self):
        for query in ('" OR * NOT title: shoes', "'); DROP TABLE products; --", "", "的"):
            self.retriever.retrieve(SearchContext(queries=(query,)), 4)
        self.assertEqual(self.retriever.bm25.connection.execute("SELECT count(*) FROM products").fetchone()[0], 4)

    def test_multiple_queries_preserve_all_evidence(self):
        pool = self.retriever.retrieve(SearchContext(queries=("running shoes", "jacket", "black")), 4)
        self.assertIn("D", ids(pool))
        self.assertEqual(len(ids(pool)), len(set(ids(pool))))
        self.assertTrue(any(len(c.hits) > 1 for c in pool))

    def test_fair_merge_keeps_other_route_evidence(self):
        hit = lambda source, rank: (SourceHit(source, rank, float(rank)),)
        first = [Candidate("A", hit("bm25", 1)), Candidate("B", hit("bm25", 2))]
        second = [Candidate("C", hit("semantic", 1)), Candidate("A", hit("semantic", 2))]
        result = interleave([first, second], 2)
        self.assertEqual(ids(result), ["A", "C"])
        self.assertEqual({h.source for h in result[0].hits}, {"bm25", "semantic"})

    def test_structured_retrieval_and_budget_unknown(self):
        context = SearchContext(constraints=(Constraint("category", ("Shoes",), hard=True),
                                            Constraint("price", maximum=80, hard=True)))
        result = StructuredRetriever(self.store).search(context, 4)
        self.assertEqual(ids(result), ["A", "C"])

    def test_negative_attribute(self):
        context = SearchContext(constraints=(Constraint("material", ("leather",), negative=True, hard=True),))
        result = StructuredRetriever(self.store).search(context, 4)
        self.assertEqual(set(ids(result)), {"B", "D"})

    def test_opt_in_hard_filter_retains_unknowns(self):
        retriever = HybridRetriever(self.store, RetrievalConfig(filter_known_hard_failures=True))
        self.addCleanup(retriever.close)
        context = SearchContext(queries=("shoes sneakers",),
                                constraints=(Constraint("price", maximum=80, hard=True),))
        self.assertEqual(set(ids(retriever.retrieve(context, 4))), {"A", "C", "D"})

    def test_default_does_not_globally_discard_conflicts(self):
        context = SearchContext(queries=("red running",), constraints=(Constraint("price", maximum=80, hard=True),))
        self.assertIn("B", ids(self.retriever.retrieve(context, 4)))

    def test_no_cross_call_constraint_memory(self):
        first = SearchContext(constraints=(Constraint("color", ("red",), hard=True),))
        second = SearchContext(constraints=(Constraint("color", ("black",), hard=True),))
        structured = StructuredRetriever(self.store)
        self.assertEqual(ids(structured.search(first, 4)), ["B"])
        self.assertEqual(ids(structured.search(second, 4)), ["A", "C"])

    def test_fallback_is_deterministic_and_explicit(self):
        pool = self.retriever.retrieve(SearchContext(), 2)
        self.assertEqual(ids(pool), ["A", "B"])
        self.assertTrue(pool.diagnostics.fallback_used)
        self.assertEqual(pool.candidates[0].hits[0].source, "fallback")

    def test_disable_routes_and_fallback(self):
        retriever = HybridRetriever(self.store, RetrievalConfig(enable_bm25=False,
                                    enable_structured=False, catalog_fallback=False))
        self.addCleanup(retriever.close)
        self.assertEqual(len(retriever.retrieve(SearchContext(queries=("shoes",)), 4)), 0)

    def test_route_failure_is_visible_and_other_routes_continue(self):
        class Broken:
            def search(self, context, limit):
                raise RuntimeError("test failure")
        retriever = HybridRetriever(self.store, RetrievalConfig(enable_semantic=True), semantic=Broken())
        self.addCleanup(retriever.close)
        pool = retriever.retrieve(SearchContext(queries=("running",)), 4)
        self.assertIn("A", ids(pool))
        self.assertIn("test failure", pool.diagnostics.errors["semantic"])

    def test_missing_semantic_reports_degradation(self):
        retriever = HybridRetriever(self.store, RetrievalConfig(enable_semantic=True))
        self.addCleanup(retriever.close)
        self.assertIn("semantic", retriever.retrieve(SearchContext(), 4).diagnostics.errors)

    def test_invalid_ids_from_route_are_dropped(self):
        class Foreign:
            def search(self, context, limit):
                return [Candidate("NOT_IN_CATALOG"), Candidate("A"), Candidate("A")]
        self.retriever.routes["foreign"] = Foreign()
        pool = self.retriever.retrieve(SearchContext(), 4)
        self.assertEqual(ids(pool), ["A"])

    def test_limits_and_empty_catalog(self):
        self.assertEqual(len(self.retriever.retrieve(SearchContext(), 0)), 0)
        for limit in (-1, 201, True, 1.5):
            with self.assertRaises(ValueError):
                self.retriever.retrieve(SearchContext(), limit)
        empty = HybridRetriever(ProductStore.from_records([]))
        self.addCleanup(empty.close)
        self.assertEqual(len(empty.retrieve(SearchContext(), 100)), 0)

    def test_candidate_budget_and_mode_depth(self):
        store = ProductStore.from_records({"parent_asin": str(i), "title": "shoe"} for i in range(700))
        retriever = HybridRetriever(store)
        self.addCleanup(retriever.close)
        depths = []
        class Spy:
            def search(self, context, limit):
                depths.append(limit)
                return []
        retriever.routes["spy"] = Spy()
        for mode, expected_depth in (("buying", 200), ("browsing", 300)):
            pool = retriever.retrieve(SearchContext(queries=("shoe",), mode=mode), 100)
            self.assertEqual(len(pool), 100)
            self.assertEqual(len(set(ids(pool))), 100)
            self.assertEqual(depths[-1], expected_depth)

    def test_fallback_never_relaxes_opt_in_hard_filters(self):
        retriever = HybridRetriever(self.store, RetrievalConfig(filter_known_hard_failures=True))
        self.addCleanup(retriever.close)
        context = SearchContext(constraints=(Constraint("category", ("Unknown category",), hard=True),))
        self.assertEqual(len(retriever.retrieve(context, 4)), 0)

    def test_frozen_context_benchmark(self):
        from scripts.evaluate_retrieval import run_benchmark
        cases = [{"target_asin": "A", "context": {"queries": ["black running shoes"]}}]
        result = run_benchmark(self.retriever, cases, [1, 4])
        self.assertEqual(result["4"]["recall"], 1.0)
        self.assertEqual(result["4"]["route_error_cases"], {})
        with self.assertRaises(ValueError):
            run_benchmark(self.retriever, [], [4])

    def test_cache_reload_and_content_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            first = BM25Retriever(self.store, cache_dir=directory)
            self.assertFalse(first.cache_hit)
            path = first.cache_path
            expected = ids(first.search(SearchContext(queries=("running",)), 4))
            first.close()
            second = BM25Retriever(self.store, cache_dir=directory)
            self.assertTrue(second.cache_hit)
            self.assertEqual(ids(second.search(SearchContext(queries=("running",)), 4)), expected)
            second.close()
            changed = records()
            changed[0]["title"] = "changed title"
            third = BM25Retriever(ProductStore.from_records(changed), cache_dir=directory)
            self.assertNotEqual(third.cache_path, path)
            third.close()

    def test_corrupt_bm25_cache_fails_loudly_and_can_be_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            first = BM25Retriever(self.store, cache_dir=directory)
            path = first.cache_path
            first.close()
            path.write_bytes(b"not a sqlite database")

            with self.assertRaises(BM25CacheError):
                BM25Retriever(self.store, cache_dir=directory)

            rebuilt = BM25Retriever(self.store, cache_dir=directory, rebuild_cache=True)
            self.assertFalse(rebuilt.cache_hit)
            self.assertGreater(len(rebuilt.search(SearchContext(queries=("running",)), 4)), 0)
            rebuilt.close()

    def test_product_store_compatibility_export_is_canonical(self):
        from retrieval import ProductStore as CompatibilityStore
        from starter.catalog import ProductStore as CanonicalStore

        self.assertIs(CompatibilityStore, CanonicalStore)

    def test_agent_official_contract_and_sessions(self):
        agent = Agent(retriever=self.retriever)
        with self.assertRaises(RuntimeError):
            agent.respond("a", "running shoes", 1, 10)
        agent.reset("a", {})
        agent.reset("b", {})
        result = agent.respond("a", "running shoes", 1, 10)
        self.assertIsInstance(result["message"], str)
        self.assertLessEqual(len(result["recommendations"]), 10)
        self.assertEqual(result["usage"]["prompt_tokens"], 0)
        agent.reset("a", {})
        self.assertEqual(result, agent.respond("a", "running shoes", 1, 10))


@unittest.skipUnless(importlib.util.find_spec("numpy"), "optional numpy not installed")
class SemanticTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        class Encoder:
            key = "test-fixture-v1"
            def encode_documents(self, texts):
                return np.asarray([[1., 0.] if "running" in t else [0., 1.] for t in texts])
            def encode_queries(self, texts):
                return np.asarray([[1., 0.] for _ in texts])
        self.encoder = Encoder()
        self.store = ProductStore.from_records(records())

    def test_build_reload_cosine_and_stable_ties(self):
        from retrieval.semantic_retriever import SemanticRetriever
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.npz"
            built = SemanticRetriever.build(self.store, self.encoder, path, batch_size=2)
            loaded = SemanticRetriever.load(self.store, self.encoder, path)
            context = SearchContext(semantic_query="exercise footwear")
            self.assertEqual(ids(built.search(context, 2)), ["A", "B"])
            self.assertEqual(built.search(context, 4), loaded.search(context, 4))
            self.assertEqual(built.search(SearchContext(), 4), [])
            hybrid = HybridRetriever(self.store, RetrievalConfig(enable_semantic=True), semantic=loaded)
            self.addCleanup(hybrid.close)
            self.assertIn("semantic", {h.source for c in hybrid.retrieve(context, 4) for h in c.hits})

    def test_reject_stale_assets(self):
        from retrieval.semantic_retriever import SemanticRetriever
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dense.npz"
            SemanticRetriever.build(self.store, self.encoder, path)
            changed = records()
            changed[0]["title"] = "new title"
            with self.assertRaises(ValueError):
                SemanticRetriever.load(ProductStore.from_records(changed), self.encoder, path)
            self.encoder.key = "other-model"
            with self.assertRaises(ValueError):
                SemanticRetriever.load(self.store, self.encoder, path)

    def test_reject_invalid_vectors_and_ids(self):
        import numpy as np
        from retrieval.semantic_retriever import SemanticRetriever
        for ids_, vectors in ((["A"] * 4, np.ones((4, 2))), (list("ABCD"), np.ones((3, 2))),
                              (list("ABCD"), np.zeros((4, 2))), (list("ABCD"), np.full((4, 2), np.nan))):
            with self.assertRaises(ValueError):
                SemanticRetriever(self.store, self.encoder, ids_, vectors)

    def test_invalid_query_vector_falls_back(self):
        import numpy as np
        from retrieval.semantic_retriever import SemanticRetriever
        semantic = SemanticRetriever(self.store, self.encoder, list("ABCD"), np.ones((4, 2)))
        self.encoder.encode_queries = lambda texts: np.ones((1, 3))
        hybrid = HybridRetriever(self.store, RetrievalConfig(enable_semantic=True), semantic=semantic)
        self.addCleanup(hybrid.close)
        pool = hybrid.retrieve(SearchContext(queries=("running",)), 4)
        self.assertIn("semantic", pool.diagnostics.errors)
        self.assertIn("A", ids(pool))


if __name__ == "__main__":
    unittest.main()
