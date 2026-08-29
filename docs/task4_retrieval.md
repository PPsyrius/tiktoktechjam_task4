# Task 4: Hybrid Candidate Retrieval

## Scope and status

This module returns a candidate pool, **not the final recommendation ranking**.
It implements lexical multi-query retrieval, explicit-attribute retrieval, an
optional offline dense route, fair admission/deduplication, diagnostics and fallback.
It never reads evaluation labels, maintains conversation history, or chooses questions.

Default: Python 3.9+ with SQLite FTS5, **no pip dependencies**.
Optional dense route: a Python 3.10+ environment, NumPy, Sentence Transformers,
an existing local model, and a prebuilt vector asset.
No model is bundled or automatically downloaded.

`starter/agent.py` is a runnable adapter. Its current-message-to-context
conversion and first-10 slice are temporary, **not** task 2/3/5 implementations.
Structured retrieval needs upstream constraints; running the adapter alone does
not activate those constraints.

## Quick start (repository root)

```bash
python3 -m unittest discover -v
python3 -m retrieval.build_index --catalog ParticipationKit/catalog.jsonl.gz
python3 -m scripts.evaluate_retrieval \
  --output results/task4_recall.json \
  --official-output results/task4_official.json
```

The benchmark accepts the gzip catalog directly and calls the unchanged official
`evaluate()` for the optional full-session run. If `data/catalog.jsonl` is already
prepared, the original command also works:

```bash
python3 -m evaluator.local_evaluator --output results/task4_official.json
```

Derived SQLite indices live under `.cache/retrieval/`; results under `results/`.
Both are ignored by Git. Catalog/evaluator files are never modified.
Use new result filenames to preserve experiments; existing output files are overwritten.

## Public contract

```python
from retrieval import Constraint, HybridRetriever, ProductStore, SearchContext

# Task 1 may instead provide ProductStore.from_records(normalized_records).
store = ProductStore.from_jsonl("ParticipationKit/catalog.jsonl.gz")
retriever = HybridRetriever(store, cache_dir=".cache/retrieval")
context = SearchContext(
    queries=("black running shoes", "breathable running footwear"),
    constraints=(
        Constraint("category", ("Shoes",), hard=True),
        Constraint("color", ("black",)),
        Constraint("price", maximum=80, hard=True),
    ),
    mode="buying",
    semantic_query="comfortable black shoes for running",
)
pool = retriever.retrieve(context, limit=100)
for candidate in pool:
    product = store[candidate.parent_asin]
    for hit in candidate.hits:
        print(hit.source, hit.rank, hit.score, hit.higher_is_better, hit.query)
print(pool.diagnostics)
retriever.close()
```

- `SearchContext` is immutable; sequences are copied into tuples.
- `queries`: current active strings; up to eight distinct queries, each with at most
  40 distinct tokens. Rewriting belongs to task 2. Drop obsolete queries after override.
- `constraints`: immutable Constraint objects. Supported fields: category, brand,
  material, color, size, style, use_case, feature, price. Attribute values are exact,
  case/whitespace-normalized alternatives; numeric bounds are inclusive.
  Adapt budget to `field="price"`. No-preference means no constraint.
- `mode`: buying/browsing/unknown, inferred upstream, never a supplied evaluation label.
- `semantic_query`: optional positive semantic description; otherwise lexical
  queries are combined. Do not put negated preferences in positive queries.
- `limit`: 0–200 by default; configurable via `max_candidates`.
- `CandidatePool.candidates`: unique, catalog-valid candidates, each containing
  `parent_asin` and a tuple of `SourceHit` records.
- `SourceHit`: source, original 1-based rank within source/query, raw score, query,
  and score direction. FTS5 BM25 is **lower-is-better**; cosine and structured
  known-match counts are higher-is-better. Raw scores are not comparable across routes.
- `diagnostics`: route counts/timing/errors, filtered occurrences, fallback flag,
  total time. Route counts are before admission, not final unique counts.

Use `dataclasses.asdict(pool)` for JSON transport. Task 5 may consume the Python
objects directly. ProductStore is read-only after construction, preserving index identity.

## Team boundaries

| Owner | Provides / consumes |
|---|---|
| Task 1 | Canonical facts and optional normalized attributes; preserve official IDs. Agree on vocabulary with task 2. |
| Task 2 | Query rewrites and current-message changes. |
| Task 3 | Complete active SearchContext; overrides, sessions and no-preference. |
| Task 4 | Index construction/loading, candidate retrieval, deduplication and admission. |
| Task 5 | Cross-source normalization/fusion, detailed constraint scoring, final Top 10. |

The ProductStore here is a compatibility adapter, not a second cleaning project.
Input retains official text fields. Optional `attributes` overrides explicit detail
fields. Category paths, exact detail keys (Color/Material/etc.) and scalar prices
are supported. Store names are not assumed to be brands. Free text is not treated
as a reliable extracted attribute.

Task 1 may generate embeddings if agreed, but model/text versions, ID mapping and
asset formats must be coordinated with task 4. Built-in assets fingerprint both.

## Retrieval behavior

1. BM25 uses the original FTS5 field weights and safe parameterized OR queries.
   Multiple query results are interleaved. Weights are configurable.
2. Structured retrieval uses exact-attribute postings and sorted price lookup.
   At least one known match is needed; local known hard conflicts are excluded.
   Unknown fields are not false. Soft criteria contribute route match counts.
3. Optional dense retrieval normalizes vectors and uses exact cosine search with
   stable ID tie-breaks. The full catalog is embedded offline, never in respond().
4. Admission round-robins across routes, deduplicates ASINs and preserves all
   retrieved source evidence. This is a budget rule, **not score fusion**.
5. Buying requests up to 2× pool size per route; browsing/unknown requests 3×.
   These are engineering defaults, not proven optimal settings.
6. Optional global known-hard-conflict filtering applies after retrieval and to
   fallback. It is off by default to protect recall.
7. If all routes fail or are empty, optional catalog-ID fallback returns valid IDs
   with `source="fallback"`. This is not a relevance claim. Partial relevant pools
   are not padded with unrelated products.

Fallback never silently relaxes active filters. Pools may contain fewer than K.
Opt-in global filtering is bounded by route depth, not exhaustive filtered search.

Every route can be disabled via RetrievalConfig. Semantic startup failures and
runtime route exceptions are visible in diagnostics; other routes continue.
Core catalog/index construction errors fail loudly. FTS5 caches are content-
fingerprinted, atomically created and reopened read-only. A corrupt derived cache
must be explicitly rebuilt; it is not silently trusted.

One instance uses a SQLite connection tied to its creating thread. Reuse across
sequential sessions, not concurrent threads; create one instance per worker.
Call close() after use. There is no query/session cache.

## Optional offline semantic route

In your chosen Python 3.10+ environment:

```bash
python -m pip install -r requirements-semantic.txt
python -m retrieval.build_index \
  --model-dir /path/to/local/sentence-transformer \
  --semantic-output artifacts/retrieval/dense.npz
python -m scripts.evaluate_retrieval \
  --model-dir /path/to/local/sentence-transformer \
  --semantic-index artifacts/retrieval/dense.npz \
  --output results/task4_dense_recall.json
```

For prefix-dependent models, pass identical `--query-prefix` and
`--document-prefix` settings to both commands. These are fingerprinted with model
files. No particular model's quality, license or final runtime budget is assumed.

For integration: create LocalSentenceEncoder, call SemanticRetriever.load, and
pass it as `semantic=` to HybridRetriever with
`RetrievalConfig(enable_semantic=True)`. Inference never auto-builds missing assets.
The benchmark reports missing/mismatched assets and degrades; the explicit builder
fails loudly. Custom encoders need `key`, `encode_documents(texts)` and
`encode_queries(texts)`.

Optional requirements specify compatibility ranges, not a locked environment.
Record exact installed versions and model files before final submission.
Dense unit tests use deterministic fixture embeddings: they verify mechanics,
not real-model quality.

API references: [SQLite FTS5](https://www.sqlite.org/fts5.html),
[SentenceTransformer](https://sbert.net/docs/package_reference/sentence_transformer/model.html).

## Evaluation and ablations

Default benchmark: **only the first message** produced by the public simulator.
Target metadata/labels never enter SearchContext. With one target per case,
Recall@K is the fraction of targets entering the K-sized admitted pool.
Each K is tested separately. Intent-override first-turn recall is diagnostic,
not an official pre-override conversion.

Reports include Recall@50/100/200, per-scenario recall, P50/P95 latency, route errors,
fallbacks, initialization time, peak process RSS, configuration and fingerprints.
RSS includes the benchmark's raw catalog/simulator, not only retrieval. Measure
timings in separate runs and distinguish cold vs warm index load.

The starter provides no parsed constraints, so default first-turn tests mainly
measure BM25. To measure mixed-route improvements independently of tasks 2/3,
freeze actual upstream contexts in JSONL:

```json
{"case_id":"example","target_asin":"VALID_CATALOG_ID","context":{"queries":["running shoes"],"constraints":[{"field":"price","maximum":80,"hard":true}],"mode":"buying"}}
```

Labels stay outside context. Use actual available user information, not attributes
copied from the hidden target.

```bash
python3 -m scripts.evaluate_retrieval --contexts /path/to/cases.jsonl \
  --no-structured --output results/bm25_only.json
python3 -m scripts.evaluate_retrieval --contexts /path/to/cases.jsonl \
  --output results/lexical_structured.json
```

Available ablations: `--no-bm25`, `--no-structured`, `--no-fallback`, `--hard-filter`.
Keep the same case fingerprint, K and filtering settings for fair comparisons.
Labels are development-only: never hardcode answers or pass scenario labels at runtime.
Final HitRate@10/MRR/MTTC also depend on parsing, Memory, clarification and reranking.
This module extraction alone does not promise end-to-end improvement.
