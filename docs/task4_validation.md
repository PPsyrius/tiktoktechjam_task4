# Task 4 validation snapshot — 2026-08-28

Branch: feature4. Local changes are not committed or pushed.

## Automated checks

- Python 3.9.6, standard-library-only environment: 30 tests passed; 4 optional
  NumPy tests skipped.
- Python 3.12.13 with NumPy: all 34 tests passed.
- Compilation and Git whitespace checks passed.
- Original catalog, public dataset, evaluator and evaluator tests are unchanged.
- FTS5 index built for 50,000 unique products; content-addressed cache reload verified.
- Dense tests use deterministic fixture embeddings. No real SentenceTransformer
  model has been installed, downloaded or evaluated.

## Public first-turn candidate recall

200 cases, original simulator first messages, no parsed constraints, semantic off.
Only BM25 contributes candidates in this benchmark. These are **not** final
recommendation metrics and do not demonstrate hybrid improvement.

| Pool size | Target found | Recall | P50 latency | P95 latency |
|---|---|---|---|---|
| 50 | 76 / 200 | 38.0% | 12.51 ms | 36.22 ms |
| 100 | 105 / 200 | 52.5% | 11.27 ms | 27.90 ms |
| 200 | 137 / 200 | 68.5% | 17.91 ms | 48.90 ms |

No route errors or catalog fallbacks. Warm-cache initialization: 1.305 seconds.
Benchmark peak process RSS: 366.67 MiB, including raw catalog and simulator state.
Numbers are a local macOS run, not deployment guarantees.

Machine-readable result: results/task4_recall.json (Git-ignored).
It includes source, catalog and frozen-case fingerprints.

## Official full-session regression

Result: results/task4_official_final.json (Git-ignored).

| Metric | Result |
|---|---|
| HitRate@10 | 0.125 |
| MRR | 0.068034 |
| MTTC | 9.81 |
| Efficiency | 0.119 |
| Technical score | 0.10671 |

These match docs/baseline_results.json exactly. The adapter intentionally still
lacks task 2 parsing, task 3 Memory and task 5 reranking/clarification.

## Mixed-route smoke check

Using the full catalog with two footwear queries and explicit category/price
constraints: both BM25 and structured routes returned 200 pre-admission candidates;
the admitted pool contained 100 unique valid IDs and evidence from both routes,
with no errors or fallback. This verifies integration, not relevance improvement.

## Remaining empirical work

1. Agree on canonical attributes and adapt task 1/3 objects to the documented API.
2. Freeze real upstream contexts; compare identical inputs with route ablations.
3. Select/package a local semantic model; build the real asset and measure recall,
   runtime, memory and model licensing suitability.
4. Integrate task 5 and rerun the official multi-turn evaluation.

See task4_retrieval.md for commands and the full interface.
