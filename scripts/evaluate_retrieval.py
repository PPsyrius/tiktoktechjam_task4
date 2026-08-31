"""Fixed-input candidate recall benchmark, optionally followed by official evaluation."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from evaluator.local_evaluator import coarse_category, evaluate, initial_message, materialize_hidden_fields
from retrieval import Constraint, HybridRetriever, ProductStore, RetrievalConfig, SearchContext
from starter.agent import Agent


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def public_cases(samples, products):
    """Labels are used by the official simulator, never passed to the retriever."""
    cases = []
    for sample in samples:
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        simulated = {**sample, "intent_card": card, "behavior": behavior}
        message = initial_message(simulated, coarse_category(products[target].get("categories") or []), set())
        cases.append({"case_id": sample["sample_id"], "target_asin": target,
                      "scenario": sample["scenario_type"], "context": {"queries": [message]}})
    return cases


def context_from_case(case):
    payload = dict(case["context"])
    payload["constraints"] = tuple(Constraint(**c) for c in payload.get("constraints", ()))
    return SearchContext(**payload)


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def run_benchmark(retriever, cases, ks):
    if not cases:
        raise ValueError("Benchmark requires at least one case")
    for case in cases:
        if case["target_asin"] not in retriever.store:
            raise ValueError("Target is not in the catalog: " + case["target_asin"])
    result = {}
    for k in ks:
        latencies, sizes, hits = [], [], []
        errors, counts, source_hits = Counter(), Counter(), Counter()
        scenarios = defaultdict(list)
        fallback = 0
        for case in cases:
            pool = retriever.retrieve(context_from_case(case), limit=k)
            target = case["target_asin"]
            hit = target in {c.parent_asin for c in pool}
            hits.append(hit)
            sizes.append(len(pool))
            latencies.append(pool.diagnostics.total_ms)
            scenarios[case.get("scenario", "unspecified")].append(hit)
            errors.update(pool.diagnostics.errors.keys())
            counts.update(pool.diagnostics.route_counts)
            fallback += int(pool.diagnostics.fallback_used)
            for candidate in pool:
                if candidate.parent_asin == target:
                    source_hits.update({h.source for h in candidate.hits})
        result[str(k)] = {
            "recall": statistics.fmean(hits), "hits": sum(hits),
            "mean_pool_size": statistics.fmean(sizes),
            "latency_ms_p50": statistics.median(latencies),
            "latency_ms_p95": percentile(latencies, .95),
            "fallback_cases": fallback, "route_error_cases": dict(errors),
            "mean_route_candidate_count": {name: count / len(cases) for name, count in counts.items()},
            "target_source_hits_in_admitted_pool": dict(source_hits),
            "scenario_recall": {name: statistics.fmean(values) for name, values in scenarios.items()},
        }
    return result


def write_result(path, value):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_fingerprint():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in ("retrieval", "starter", "scripts", "evaluator"):
        for path in sorted((root / directory).glob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="ParticipationKit/catalog.jsonl.gz")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--contexts", help="JSONL frozen cases with context and target_asin")
    parser.add_argument("--output", default="results/retrieval.json")
    parser.add_argument("--official-output", help="Also run the unmodified official evaluate()")
    parser.add_argument("--cache-dir", default=".cache/retrieval")
    parser.add_argument("--ks", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--no-bm25", action="store_true")
    parser.add_argument("--no-structured", action="store_true")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--hard-filter", action="store_true")
    parser.add_argument("--model-dir")
    parser.add_argument("--semantic-index")
    parser.add_argument("--semantic-weight", type=float, default=0.3)
    parser.add_argument("--semantic-candidate-limit", type=int, default=40)
    parser.add_argument("--dynamic-semantic-gate", action="store_true")
    parser.add_argument("--semantic-min-lexical-fill", type=float, default=0.75)
    parser.add_argument("--semantic-shadow-min-overlap", type=int, default=2)
    parser.add_argument("--semantic-shadow-lexical-window", type=int, default=160)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")
    args = parser.parse_args()
    if any(k < 1 or k > 200 for k in args.ks):
        parser.error("--ks must be in [1, 200]")
    if bool(args.model_dir) != bool(args.semantic_index):
        parser.error("--model-dir and --semantic-index must be supplied together")
    if args.semantic_weight < 0:
        parser.error("--semantic-weight must be non-negative")
    if not 0 <= args.semantic_candidate_limit <= 200:
        parser.error("--semantic-candidate-limit must be in [0, 200]")
    if not 0 < args.semantic_min_lexical_fill <= 1:
        parser.error("--semantic-min-lexical-fill must be in (0, 1]")
    if args.semantic_shadow_min_overlap < 0:
        parser.error("--semantic-shadow-min-overlap must be non-negative")
    if args.semantic_shadow_lexical_window < 1:
        parser.error("--semantic-shadow-lexical-window must be positive")
    started = time.perf_counter()
    records = read_rows(args.catalog)
    products = {p["parent_asin"]: p for p in records}
    store = ProductStore.from_records(records)
    samples = read_rows(args.dataset) if not args.contexts or args.official_output else []
    cases = read_rows(args.contexts) if args.contexts else public_cases(samples, products)
    config = RetrievalConfig(enable_bm25=not args.no_bm25, enable_structured=not args.no_structured,
                             enable_semantic=bool(args.model_dir), catalog_fallback=not args.no_fallback,
                             filter_known_hard_failures=args.hard_filter,
                             semantic_candidate_limit=args.semantic_candidate_limit,
                             dynamic_semantic_gate=args.dynamic_semantic_gate,
                             semantic_min_lexical_fill=args.semantic_min_lexical_fill,
                             semantic_shadow_min_lexical_overlap=
                                 args.semantic_shadow_min_overlap,
                             semantic_shadow_lexical_window=args.semantic_shadow_lexical_window)
    semantic, semantic_error = None, None
    if args.model_dir:
        try:
            from retrieval.semantic_retriever import LocalSentenceEncoder, SemanticRetriever
            encoder = LocalSentenceEncoder(args.model_dir, args.query_prefix, args.document_prefix)
            semantic = SemanticRetriever.load(store, encoder, args.semantic_index)
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
    retriever = HybridRetriever(store, config, args.cache_dir, semantic, semantic_error)
    initialization_seconds = time.perf_counter() - started
    try:
        metrics = run_benchmark(retriever, cases, args.ks)
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
        result = {
            "benchmark": "frozen_contexts" if args.contexts else "public_first_turn_only",
            "note": "Candidate recall, not official session HitRate@10. No parser/Memory/reranker is evaluated.",
            "sample_count": len(cases), "catalog_size": len(store),
            "catalog_fingerprint": store.fingerprint,
            "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
            "git_revision": revision, "working_tree_dirty": dirty,
            "source_sha256": source_fingerprint(),
            "python": sys.version, "platform": platform.platform(),
            "configuration": asdict(config), "semantic_startup_error": semantic_error,
            "fts5_cache_hit": retriever.bm25.cache_hit if retriever.bm25 else None,
            "initialization_seconds": initialization_seconds, "metrics": metrics,
            "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                (1024 * 1024 if sys.platform == "darwin" else 1024),
            "memory_note": "Whole benchmark process peak, including raw catalog, simulator and store; not isolated retriever memory.",
        }
        write_result(args.output, result)
        print(json.dumps(result, indent=2))
        if args.official_output:
            categories = {asin: p.get("categories") or [] for asin, p in products.items()}
            official = evaluate(
                Agent(retriever=retriever, semantic_weight=args.semantic_weight),
                samples,
                set(products),
                categories,
                products,
            )
            write_result(args.official_output, official)
            print(json.dumps({k: v for k, v in official.items() if k != "sessions"}, indent=2))
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
