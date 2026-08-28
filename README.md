# TechJam Shopping Copilot

Section 5 branch for `Candidate Fusion, Constraint Scoring, and Final Reranking`.

Current branch: `feature5`

## Goal

This branch focuses on the last ranking stage of the shopping agent pipeline.
Given a `SearchContext` from the memory/state module and a `Candidate Pool`
from the retrieval module, Section 5 is responsible for producing the final
Top 10 recommendations in a stable, constraint-aware order.

High-level pipeline:

```text
Catalog -> ParsedUpdate -> SearchContext -> Candidate Pool -> Ranked Top 10
```

Section 5 owns the final step:

```text
SearchContext + Candidate Pool -> Candidate Fusion -> Constraint Scoring -> Final Reranking -> Top 10
```

## Repository Layout

Core competition files:

- `starter/agent.py`: official editable agent entrypoint used by the evaluator
- `evaluator/local_evaluator.py`: local public-set evaluator
- `docs/agent_api_contract.json`: required request/response contract
- `data/public_set.jsonl`: 200 public development sessions
- `data/catalog.jsonl`: local catalog file, required for evaluation and ignored by Git

Section 5 files added on this branch:

- `starter/score_normalizer.py`: normalizes source scores into comparable floats
- `starter/candidate_fusion.py`: deduplicates candidates and merges source scores/ranks
- `starter/constraint_scorer.py`: scores hard constraints against candidate product text
- `starter/reranker.py`: produces the final ranked output structure

## Current Implementation Status

Implemented today:

- Section 5 file structure is created
- package-relative imports are in place
- candidate deduplication and source-score merging are working
- hard-constraint scoring is implemented in a first-pass rule-based form
- final ranking returns deterministic order by `final_score` then `parent_asin`
- local smoke test passes for a simple hard-constraint example

Not integrated yet:

- `starter/agent.py` still uses the original BM25 baseline
- there is no shared `SearchContext` or `Candidate Pool` contract file yet
- soft-preference scoring is not implemented yet
- Buying vs Browsing weighting is not implemented yet
- `clarification_slot` / `ask_attribute` selection is not implemented yet
- rating / review-count features are not implemented yet
- Section 5 is not yet wired into end-to-end evaluation

## Local Setup

From the repository root:

```bash
cd /Users/macbook/projects/tiktok/tiktoktechjam_task4
```

Prepare the catalog once:

```bash
shasum -a 256 ParticipationKit/catalog.jsonl.gz
cat ParticipationKit/SHA256SUMS
gzip -dk ParticipationKit/catalog.jsonl.gz
mv ParticipationKit/catalog.jsonl data/catalog.jsonl
```

Run the official baseline evaluator:

```bash
python3 -m evaluator.local_evaluator
```

Expected baseline metrics before end-to-end integration:

- `HitRate@10 = 0.125`
- `MRR = 0.068034`
- `MTTC = 9.81`

## Section 5 API Shape

Current internal reranker entrypoint:

```python
from starter.reranker import rank_candidates

search_context = {
    "hard_constraints": ["cotton", "black"],
}

candidate_pool = [
    {
        "parent_asin": "A1",
        "title": "Black cotton running shirt",
        "source_scores": {"bm25": 3.2, "semantic": 0.8},
    }
]

result = rank_candidates(search_context, candidate_pool, top_k=10)
```

Current expected inputs:

- `search_context`
  - `hard_constraints`: list of strings, or `constraints["hard"]`
  - `intent`: optional, reserved for future Buying/Browsing weighting
- `candidate_pool`
  - `parent_asin`
  - product text fields such as `title`, `categories`, `features`, `details`, `description`, `store`
  - optional `source_scores`
  - optional `source_ranks`

Current output:

- `RankingResult.items`: list of `RankedItem`
- `RankedItem.parent_asin`
- `RankedItem.final_score`
- `RankedItem.reason_codes`
- `RankedItem.matched_preferences`
- `RankedItem.hard_failures`
- `RankingResult.clarification_slot`: currently `None`

## Quick Validation

Syntax check:

```bash
python3 -m py_compile \
  starter/score_normalizer.py \
  starter/candidate_fusion.py \
  starter/constraint_scorer.py \
  starter/reranker.py
```

Smoke test:

```bash
python3 - <<'PY'
from starter.reranker import rank_candidates

search_context = {
    "hard_constraints": ["cotton", "black"],
}

candidate_pool = [
    {
        "parent_asin": "A1",
        "title": "Black cotton running shirt",
        "source_scores": {"bm25": 3.2, "semantic": 0.8},
    },
    {
        "parent_asin": "A2",
        "title": "Red polyester jacket",
        "source_scores": {"bm25": 4.0},
    },
]

print(rank_candidates(search_context, candidate_pool, top_k=10))
PY
```

The expected behavior is that `A1` ranks above `A2` because it matches both
hard constraints.

## Recommended Next Steps

1. Agree with Sections 3 and 4 on a stable `SearchContext` and `Candidate Pool` contract.
2. Extend `constraint_scorer.py` with soft preferences and intent-aware weights.
3. Add product quality signals such as rating and review count.
4. Add `clarification_slot` selection when top candidates are difficult to distinguish.
5. Wire `rank_candidates(...)` into `starter/agent.py` after upstream modules are ready.
6. Add unit tests for hard fail, soft match, and deterministic ranking behavior.

## Notes

- Do not modify `evaluator/local_evaluator.py` for competition scoring.
- Do not rely on `ground_truth` or `scenario_type` inside runtime agent logic.
- `data/catalog.jsonl` and `results.json` are ignored by Git and are local-only files.
