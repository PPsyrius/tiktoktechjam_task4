# TechJam Shopping Copilot

An offline conversational product-search agent for the TechJam Shopping Copilot challenge. The system turns multi-turn customer language into structured state, routes each request through lexical, structured, and selectively activated semantic retrieval, then returns a constraint-aware Top 10.

## Why this project

Keyword-only shopping search breaks when a customer starts vaguely, adds constraints over several turns, or changes direction. This agent is designed around those transitions:

- **Buying:** prioritize explicit categories and hard constraints.
- **Browsing:** preserve diversity and defer semantic candidates for later turns.
- **Intent override:** replace obsolete preferences without carrying stale ranking scores.
- **Boundary/no preference:** keep unknown catalog facts eligible and avoid inventing constraints.

## Architecture

```text
customer message
      |
      v
intent + requirement parser ---> session Memory / SearchContext
                                      |
                   +------------------+------------------+
                   |                  |                  |
                   v                  v                  v
               BM25/FTS5       structured postings   MiniLM dense
                   +------------------+------------------+
                                      |
                         gated candidate admission
                                      |
                  constraint scoring + source-score fusion
                                      |
                         ordered Top 10 + next question
```

The semantic route uses only observable runtime signals. It activates on preference overrides, insufficient lexical fill, or first-turn lexical/semantic agreement; semantic candidates are capped at 40 and use ranking weight `0.3`.

## Verified result

Public set: 200 sessions and the frozen 50,000-product catalog.

| Metric | Result |
|---|---:|
| HitRate@10 | **0.955** |
| MRR | **0.638062** |
| MTTC | **2.74** |
| Efficiency | **0.826** |
| TechnicalScore | **0.834119** |

| Scenario | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|
| Boundary | 1.0000 | 0.756667 | 3.80 |
| Browsing | 0.9625 | 0.599588 | 2.675 |
| Buying | 0.9375 | 0.595774 | 2.25 |
| Intent Override | 0.966667 | 0.813889 | 3.866667 |

These are public-development results, not a claim about the organizer's disjoint 800-session private set.

## Setup

Tested with Python `3.12.11`. Python 3.10+ is supported by the source and CI.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
gzip -dc ParticipationKit/catalog.jsonl.gz > data/catalog.jsonl
```

For the exact tested dependency set, use `requirements-lock.txt` instead of `requirements.txt`.

### Prepare semantic assets in a development clone

The public source repository does not require large binary files in Git history. This command downloads the pinned model revision, rebuilds the catalog-bound index, and runs the public evaluation:

```bash
python -m scripts.run_best_semantic --download-model
```

The final offline submission archive contains the already prepared model and index, so official inference does not require network access.

## Official evaluation command

Once `artifacts/models/all-MiniLM-L6-v2` and `artifacts/retrieval/all-MiniLM-L6-v2.npz` are present, the unmodified official harness automatically loads the verified semantic configuration:

```bash
python -m evaluator.local_evaluator
```

Expected summary:

```text
HitRate@10=0.955  MRR=0.638062  MTTC=2.74  TechnicalScore=0.834119
```

Set `TECHJAM_DISABLE_SEMANTIC=1` to exercise the deterministic lexical/structured fallback. Optional path overrides are `TECHJAM_SEMANTIC_MODEL_DIR` and `TECHJAM_SEMANTIC_INDEX`.

## Build the offline submission

```bash
python -m scripts.build_submission
```

This produces `dist/techjam-shopping-copilot-offline.zip` containing:

- root `agent.py` exporting `Agent`;
- all required local helper modules;
- the frozen public catalog and evaluator for reproduction;
- the pinned Apache-2.0 MiniLM model and catalog-matched dense index;
- a machine-readable manifest with checksums and expected metrics.

The builder refuses to package a model or index whose fingerprint does not match the catalog.

## Demo

Run a human-readable multi-turn example:

```bash
python -m scripts.demo_session
```

The recording plan and shot list are in [`docs/DEMO_VIDEO_SCRIPT.md`](docs/DEMO_VIDEO_SCRIPT.md). Devpost-ready copy is in [`docs/DEVPOST_SUBMISSION.md`](docs/DEVPOST_SUBMISSION.md).

## Required interface

The submission exports `agent.Agent` and implements:

```python
agent.reset(session_id, user_profile)
response = agent.respond(session_id, user_message, turn, top_k=10)
```

Each response contains a message, an allowed `ask_attribute`, ordered unique `parent_asin` recommendations, and non-negative token usage. The agent never reads ground truth or private scenario labels and never mutates the catalog.

## Tools, APIs, libraries, and assets

- **Development tools:** Python CLI, PyCharm, Git/GitHub, GitHub Actions.
- **Runtime APIs:** no external API is required by the verified submission. An optional DeepSeek parser exists for experiments but is disabled in the verified offline configuration.
- **Libraries:** Sentence Transformers, Hugging Face Transformers, PyTorch, NumPy, scikit-learn, and SQLite FTS5.
- **Dataset:** organizer-provided frozen Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry` catalog and 200 public sessions.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, Apache-2.0.
- **Generated asset:** 50,000-product normalized dense index, fingerprinted against both catalog and model.

See [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) for dataset attribution.

## Cost and feasibility disclosure

- External API calls in the verified configuration: **0**.
- Prompt/completion tokens: **0 / 0**.
- Estimated model/API cost: **$0**.
- Network required during official inference: **No**.
- Local model size: approximately **87 MB**.
- Dense index size: approximately **71.6 MB**.
- Warm initialization observed locally: approximately **13 seconds**.
- Full index rebuild observed locally: approximately **415 seconds**; this is performed before submission, not during inference.
- Benchmark-process peak RSS observed locally: approximately **1.1 GB**, including catalog, simulator, model, and index.
- First-turn retrieval P95 observed locally: approximately **39 ms** after initialization.

Times are measurements from a macOS ARM64 development machine, not deployment guarantees.

## Tests

```bash
python -m unittest discover -s tests
```

The suite covers catalog normalization, intent parsing, memory transitions, overrides, route fusion, semantic asset validation, ranking, API behavior, and end-to-end session flow.

## Limitations and next improvements

- The aggregate `user_profile` is accepted but not yet used for ranking; safe low-weight personalization is future work.
- Clarification follows state-aware attribute ordering rather than explicit information-gain estimation over the candidate pool.
- Exact dense scoring over 50,000 vectors is intentionally simple and reproducible but uses more memory than an approximate index.
- Rating and review-count quality signals are not yet included in final ranking.
- The semantic gate and weights were selected on the 200 public sessions and may not transfer perfectly to the disjoint private set.
- The current English parser assumes the challenge's pre-cleaned text and does not address spelling or ASR errors, as allowed by the task scope.

## Team contributions

- **Panpakorn Siripanich / PPsyrius:** repository integration, participant-kit setup, CI, linting, compatibility fixes, and end-to-end test infrastructure.
- **Jia Huang / jiahuang-ui:** session memory/state contracts, local hybrid-parser work, canonical catalog data layer, normalization, and catalog caching.
- **Mrigakshi Roy Choudhury:** user-requirement understanding, constraint parsing, query rewriting, follow-up/override interpretation, and parsing/ranking integration.
- **LiiiKiii:** candidate fusion, constraint scoring, deterministic reranking, snippet evidence integration, and ranking documentation.
- **Nico:** multi-route retrieval, BM25/structured/semantic integration, selective semantic gating, override/candidate-admission stabilization, evaluation diagnostics, and reproducible offline packaging.

## Repository layout

```text
agent.py                  official submission export
starter/                  agent, parser, memory, catalog, ranking
retrieval/                BM25, structured, semantic, fusion
evaluator/                organizer-compatible local evaluator
scripts/                  profiling, evaluation, demo, packaging
tests/                    unit and integration regression suite
docs/                     contracts, design notes, submission material
artifacts/                local model/index (ignored in source Git)
```

Detailed retrieval contracts are documented in [`docs/task4_retrieval.md`](docs/task4_retrieval.md).
