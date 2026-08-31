# TechJam Shopping Copilot — Devpost Copy

## Elevator pitch

TechJam Shopping Copilot is an offline conversational retrieval agent that follows a shopper as their intent evolves. It combines stateful requirement parsing, BM25, structured filtering, selectively gated MiniLM semantic retrieval, and constraint-aware reranking to return the right product sooner without paid APIs.

## Inspiration

Shopping conversations rarely arrive as one clean query. A customer may begin with “I’m still exploring,” add a price or material constraint later, then replace the entire preference. Static keyword search either over-filters too early or carries stale terms into later results. We wanted an agent whose retrieval strategy changes with the conversation rather than treating every message as an independent search box.

## What it does

The agent classifies Buying, Browsing, and Unknown intent; converts each turn into structured state updates; and safely handles additions, exclusions, no-preference responses, and intent overrides. It retrieves candidates from three complementary routes:

- weighted BM25 over catalog fields;
- exact structured postings for category, attributes, and price;
- local MiniLM vector similarity when runtime evidence shows semantic retrieval is useful.

Candidates are deduplicated, admitted under route-specific caps, scored against hard constraints and preferences, and returned as an ordered Top 10. The system also asks a valid next clarification question and maintains candidate continuity without retaining obsolete scores after an override.

## How it addresses the challenge

- **Intent routing:** Buying uses deeper constraint precision; Browsing preserves a wider and more diverse candidate path.
- **Hybrid pipeline:** keyword, structured, snippet, and dense evidence are fused in memory.
- **Multi-turn evolution:** session state accumulates constraints and explicitly erases or replaces stale slots.
- **Runtime adaptation:** semantic retrieval is gated by lexical fill, override signals, and lexical/semantic agreement rather than enabled blindly for every query.
- **Efficiency:** the verified system uses no paid API, no inference network access, and reports zero prompt/completion tokens.

## Results

On the organizer's 200 public development sessions:

- HitRate@10: **0.955**
- MRR: **0.638062**
- MTTC: **2.74**
- Efficiency: **0.826**
- TechnicalScore: **0.834119**

The public and private users and targets are disjoint, so these values are development evidence rather than a private-set guarantee.

## Built with

- Python 3.12
- SQLite FTS5
- NumPy and scikit-learn
- PyTorch and Hugging Face Transformers
- Sentence Transformers `all-MiniLM-L6-v2`
- Git, GitHub, GitHub Actions, and PyCharm

No external API is required by the submitted configuration. An optional DeepSeek parser was explored, but the verified final path is fully offline.

## Data and assets

We use the organizer-provided frozen 50,000-product catalog derived from Amazon Reviews 2023 `Clothing_Shoes_and_Jewelry`, plus the 200 labeled public development sessions. The dense model is the Apache-2.0 `sentence-transformers/all-MiniLM-L6-v2` model at pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. We do not use private labels, mutate the catalog, or reconstruct the upstream Amazon dataset.

## Challenges

The hardest problem was not adding more retrieval routes; it was deciding when a route should be trusted. Uniform semantic fusion introduced new candidates but could displace strong lexical matches. We therefore measured failure cases by turn and scenario, added route-specific admission limits, and used observable confidence signals to activate semantics selectively. Intent override required a second invariant: old candidates could remain available for later reranking, but old scores could never survive a changed task.

## Accomplishments

- Reproduced the organizer baseline and raised HitRate@10 from 0.125 to 0.955.
- Built a catalog-fingerprinted offline vector asset that rejects stale model/catalog combinations.
- Made the original official evaluator command load the verified semantic configuration automatically.
- Added 137 regression tests across data, parsing, memory, retrieval, ranking, and full-session behavior.
- Produced a self-contained offline submission with no API credentials or runtime network dependency.

## Limitations

The aggregate user profile is accepted but not yet used in ranking. Clarification is state-aware but does not yet estimate information gain from the current candidate distribution. Exact dense scoring favors simplicity and reproducibility over the memory efficiency of approximate search. Rating and review-count quality signals are not yet part of final ranking, and public-set parameter selection may not transfer perfectly to the private set.

## What we would improve next

We would add low-weight, auditable profile personalization; choose questions using candidate entropy; incorporate product-quality priors; and validate route thresholds on a larger held-out set. For a production catalog, we would replace exact dense scoring with a compact approximate index while preserving the same catalog/model fingerprint contract.

## Team contributions

- **Panpakorn Siripanich / PPsyrius:** repository integration, participant kit, CI, linting, compatibility, and end-to-end tests.
- **Jia Huang / jiahuang-ui:** memory/state contracts, hybrid parser work, catalog normalization, and caching.
- **Mrigakshi Roy Choudhury:** requirement understanding, constraints, query rewriting, and follow-up/override parsing.
- **LiiiKiii:** candidate fusion, constraint scoring, reranking, and snippet evidence.
- **Nico:** multi-route retrieval, semantic gating, override/candidate stabilization, diagnostics, and offline packaging.

## Links to finalize

- Public repository: https://github.com/PPsyrius/tiktoktechjam_task4
- Public YouTube demo: **ADD FINAL YOUTUBE URL BEFORE SUBMISSION**
