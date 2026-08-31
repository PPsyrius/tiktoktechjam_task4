"""Replay labeled public cases and explain recall/ranking failures turn by turn.

This script is development-only. Ground-truth IDs are supplied to an external
diagnostic hook and are never passed into Agent retrieval or ranking logic.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from evaluator.local_evaluator import evaluate, materialize_hidden_fields
from retrieval import HybridRetriever, ProductStore, RetrievalConfig
from retrieval.semantic_retriever import LocalSentenceEncoder, SemanticRetriever
from starter.agent import Agent
from starter.reranker import rank_candidates


def read_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def classify_case(turns, eligible_from):
    eligible = [turn for turn in turns if turn["turn"] >= eligible_from]
    before = [turn for turn in turns if turn["turn"] < eligible_from]
    if any(turn["full_rank"] is not None and turn["full_rank"] <= 10 for turn in eligible):
        return "eligible_top10_seen"
    if any(turn["target_in_candidates"] for turn in before) and not any(
        turn["target_in_candidates"] for turn in eligible
    ):
        return "candidate_continuity"
    # A very low-ranked snippet-only payload is not meaningful retrieval
    # evidence. Treat it as recall/query selection rather than reranking.
    if (
        not any(turn["target_in_pool"] or turn["target_in_shadow"] for turn in turns)
        and min(
            (turn["full_rank"] for turn in eligible if turn["full_rank"] is not None),
            default=10**9,
        ) > 50
    ):
        return "recall_or_query_selection"
    if not any(turn["target_in_candidates"] for turn in eligible):
        return "recall_or_query_selection"
    return "reranking"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="ParticipationKit/catalog.jsonl.gz")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--semantic-index", required=True)
    parser.add_argument("--output", default="results/public-failure-diagnostics.json")
    parser.add_argument("--low-rank-threshold", type=int, default=5)
    args = parser.parse_args()

    records = read_rows(args.catalog)
    products = {str(row["parent_asin"]): row for row in records}
    categories = {
        asin: product.get("categories") or [] for asin, product in products.items()
    }
    samples = read_rows(args.dataset)
    samples_by_id = {str(sample["sample_id"]): sample for sample in samples}
    evaluation = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    misses = {
        str(session["sample_id"])
        for session in evaluation["sessions"]
        if not session["hit"]
    }
    low_rank = {
        str(session["sample_id"])
        for session in evaluation["sessions"]
        if session["hit"] and int(session["best_rank"]) >= args.low_rank_threshold
    }
    selected_ids = sorted(misses | low_rank)

    store = ProductStore.from_records(records)
    encoder = LocalSentenceEncoder(args.model_dir)
    semantic = SemanticRetriever.load(store, encoder, args.semantic_index)
    config = RetrievalConfig(
        enable_semantic=True,
        semantic_candidate_limit=40,
        dynamic_semantic_gate=True,
        semantic_min_lexical_fill=0.75,
    )
    retriever = HybridRetriever(store, config, ".cache/retrieval", semantic)
    active = {"sample_id": None, "target": None}
    traces = {sample_id: [] for sample_id in selected_ids}

    def diagnostic_hook(event):
        sample_id = active["sample_id"]
        target = active["target"]
        payloads = list(event["candidate_payloads"])
        full_ranking = rank_candidates(
            event["ranking_context"],
            payloads,
            top_k=len(payloads),
            semantic_weight=event["semantic_weight"],
        )
        full_rank = next(
            (index for index, item in enumerate(full_ranking.items, 1)
             if item.parent_asin == target),
            None,
        )
        target_item = next(
            (item for item in full_ranking.items if item.parent_asin == target),
            None,
        )
        target_payload = next(
            (payload for payload in payloads if payload["parent_asin"] == target),
            None,
        )
        pool = event["candidate_pool"]
        pool_rank = next(
            (index for index, candidate in enumerate(pool, 1)
             if candidate.parent_asin == target),
            None,
        )
        target_sources = []
        if pool_rank is not None:
            candidate = pool.candidates[pool_rank - 1]
            target_sources = [
                {"source": hit.source, "rank": hit.rank, "score": hit.score}
                for hit in candidate.hits
            ]
        traces[sample_id].append({
            "turn": event["turn"],
            "message": event["user_message"],
            "mode": event["retrieval_context"].mode,
            "queries": list(event["retrieval_context"].queries),
            "preference_override": event["retrieval_context"].preference_override,
            "semantic_gate": pool.diagnostics.semantic_gate,
            "route_counts": dict(pool.diagnostics.route_counts),
            "target_in_pool": pool_rank is not None,
            "target_pool_rank": pool_rank,
            "target_pool_sources": target_sources,
            "target_in_shadow": target in pool.deferred_candidates,
            "target_in_candidates": target_payload is not None,
            "target_source_ranks": (
                dict(target_payload.get("source_ranks") or {})
                if target_payload else {}
            ),
            "target_source_scores": (
                dict(target_payload.get("source_scores") or {})
                if target_payload else {}
            ),
            "full_rank": full_rank,
            "target_final_score": target_item.final_score if target_item else None,
            "top10_cutoff_score": (
                full_ranking.items[9].final_score
                if len(full_ranking.items) >= 10 else None
            ),
            "top10": [
                {"parent_asin": item.parent_asin, "score": item.final_score}
                for item in full_ranking.items[:10]
            ],
        })

    agent = Agent(
        retriever=retriever,
        semantic_weight=0.4,
        diagnostic_hook=diagnostic_hook,
    )
    case_results = []
    try:
        for sample_id in selected_ids:
            sample = samples_by_id[sample_id]
            target = str(sample["ground_truth"]["parent_asin"])
            active.update(sample_id=sample_id, target=target)
            card, behavior = materialize_hidden_fields(sample, products)
            eligible_from = int((behavior.get("override") or {}).get("turn", 1))
            result = evaluate(
                agent,
                [sample],
                set(products),
                categories,
                products,
            )["sessions"][0]
            classification = (
                classify_case(traces[sample_id], eligible_from)
                if sample_id in misses else "low_rank_hit"
            )
            case_results.append({
                "sample_id": sample_id,
                "scenario_type": sample["scenario_type"],
                "target_asin": target,
                "selected_because": "miss" if sample_id in misses else "low_rank_hit",
                "eligible_from_turn": eligible_from,
                "classification": classification,
                "evaluation": result,
                "turns": traces[sample_id],
            })
    finally:
        retriever.close()

    counts = Counter(
        case["classification"] for case in case_results
        if case["selected_because"] == "miss"
    )
    output = {
        "evaluation_source": args.evaluation,
        "miss_count": len(misses),
        "low_rank_count": len(low_rank),
        "failure_classification_counts": dict(sorted(counts.items())),
        "cases": case_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in output.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
