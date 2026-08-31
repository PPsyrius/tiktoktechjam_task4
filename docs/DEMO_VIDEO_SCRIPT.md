# Demo Video Script

Target length: 2–3 minutes. A backend walkthrough is explicitly accepted by the challenge; no UI is required.

## 0:00–0:20 — Problem and architecture

Show the README architecture diagram.

Narration: “Shopping intent changes across turns. Our agent converts each message into state, retrieves through BM25, structured fields, snippets, and selectively gated MiniLM semantics, then reranks a valid Top 10 entirely offline.”

## 0:20–0:40 — Reproducibility

Show:

```bash
python -m unittest discover -s tests
python -m evaluator.local_evaluator
```

Show the final metric summary: HitRate@10 `0.955`, MRR `0.638062`, MTTC `2.74`, TechnicalScore `0.834119`.

Narration: “The unmodified official evaluator loads our local model and catalog-matched index. Inference uses no external API, network, tokens, or paid credits.”

## 0:40–1:50 — Multi-turn behavior

Run:

```bash
python -m scripts.demo_session
```

Point out:

1. `semantic_enabled=True` confirms the submitted offline route is active.
2. Turn 1 begins in Browsing mode without over-committing.
3. Turn 2 accumulates black, lightweight, and budget constraints.
4. Turn 3 replaces the shoe task with polyester jackets while stale scores are discarded.
5. Every turn returns an allowed clarification field and ordered unique product IDs.

## 1:50–2:20 — Engineering decisions

Show `retrieval/hybrid_retriever.py` and `starter/agent.py` briefly.

Narration: “Semantic search is not averaged blindly. We cap semantic admission at 40 and activate it using lexical fill, override signals, and first-turn agreement. Every dense index is fingerprinted against both the frozen catalog and pinned model.”

## 2:20–2:40 — Limitations and impact

Narration: “The current system still uses fixed state-aware clarification rather than information gain, and it does not yet use the aggregate profile for personalization. The architecture is nevertheless practical: it is deterministic, testable, offline, and can be extended to larger retail catalogs.”

## Upload checklist

- Record terminal and source only; do not show API keys, personal notifications, or private messages.
- Do not use third-party music, logos, product images, or copyrighted clips.
- Upload to YouTube with visibility set to **Public**.
- Add the URL to `docs/DEVPOST_SUBMISSION.md` and the final Devpost entry.
- Verify the YouTube link in a signed-out/private browser window.
