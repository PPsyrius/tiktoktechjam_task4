# Assignment 2: User Requirement Understanding

Convert the current customer message into a structured `ParseUpdate` for Assignment 3 (Memory). This package does not search the catalog or produce the Top 10 ranking.

```text
user message + SearchContext summary → parse_requirement() → ParseUpdate → Memory.apply_update() → SearchContext
```

`starter/agent.py` calls this module, then Memory. Understanding never writes session state. Memory never parses natural language.

---

## Layout

```text
starter/understanding/
  __init__.py
  catalog_vocab.py   catalog-aligned tokens for field mapping
  query_parser.py    rule-based parser (offline fallback)
  query_rewriter.py  retrieval query construction
  llm_parser.py      optional LLM parser
  README.md
```

Tests: `tests/test_understanding.py`.

| File | Responsibility |
| --- | --- |
| `query_parser.py` | `parse_user_message(...)`, `classify_intent(...)`, `constraint_kind(...)` — intent, hard/soft/negative slots, override vs same-task replace, decline, exclude; uses SearchContext when provided |
| `catalog_vocab.py` | Shared color/material/use-case lists plus parser-only brands, styles, and feature tokens |
| `llm_parser.py` | `parse_requirement(...)` — rules always run (with SearchContext); optional DeepSeek may add slots; rule `feature` snippets are never dropped |
| `query_rewriter.py` | `rewrite_queries(...)` — retrieval-oriented strings from this turn and SearchContext |
| `__init__.py` | Public API: `parse_requirement`, `parse_user_message`, `classify_intent`, `constraint_kind`, `rewrite_queries`, `ParseResult` |

Agent entry point:

```python
from starter.understanding import parse_requirement

result = parse_requirement(session_id, user_message, turn, search_context=retrieval_state)
```

`ParseResult` fields: `parsed` (`ParseUpdate` or `None`), `source` (`rules` or `deepseek`), `prompt_tokens`, `completion_tokens`, optional `error`, `intent_confidence` (0–1), `constraint_kinds` (`hard` / `soft` / `negative` / `decline` / `clear` / `task` per update), and `query_rewrites` (retrieval strings, max 8).

Rules always run first. A model is optional and may only add understanding. Identifying catalog phrases from the rule matcher are merged back in, so a model cannot replace `100% Cotton Lightweight` with `cotton`.

---

## Integration with Assignment 3

Memory defines the `ParseUpdate` contract (`starter/memory/README.md`). This module only emits that shape.

Required / supported fields:

- `session_id`
- `intent`: `buying`, `browsing`, `unknown`, or omitted
- `source_turn`: dialogue turn index
- `reset_task`: `true` only when the user replaces the active task (intent override)
- `updates`: `{slot, op, value}` operations Memory already implements

Slots include `category`, `product_type`, `brand`, `color`, `size`, `price_min`, `price_max`, `rating_min`, `material`, `style`, `feature`, `use_case`.  
Ops: `set`, `add`, `remove`, `clear`, `decline`, `exclude`.

Do not include `ground_truth`, `scenario_type`, or other evaluation labels.

Assignment 2 describes **this turn**. The rules parser also reads a summary of the current SearchContext so follow-ups such as "not that color" or same-task replacements can target an existing slot. Assignment 3 merges turns into the session `SearchContext` (`get_retrieval_state`). Later turns send only new operations; Memory keeps prior constraints unless `reset_task` is set.

When `reset_task` is true, Memory clears the shopping task. With an active SearchContext, the parser sets `reset_task` only when the user names a **new category**. Same-task phrases such as "ignore my earlier preference" plus a new color or material emit a replacement, not a full reset. Without SearchContext, override phrases still default to `reset_task` so standalone tests stay conservative. The Agent may also re-scope a reset before `apply_update`.

Constraint strings are cleaned to **180 characters** (the same cap the local evaluator uses for intent-card snippets). Values that classify as a short token (`cotton`, `black`) still emit that slot. If the raw phrase is longer, the parser also `add`s the full snippet on `feature` so later retrieval can exact-match catalog text.

Example when the requirement is only the token `black`:

```json
{
  "session_id": "s1",
  "intent": "buying",
  "source_turn": 1,
  "updates": [
    {"slot": "category", "op": "set", "value": "running shoes"},
    {"slot": "color", "op": "set", "value": "black"}
  ]
}
```

Example when the requirement is a longer catalog line:

```json
{
  "session_id": "s1",
  "intent": "buying",
  "source_turn": 1,
  "updates": [
    {"slot": "category", "op": "set", "value": "running shoes"},
    {"slot": "material", "op": "set", "value": "cotton"},
    {"slot": "feature", "op": "add", "value": "100% Cotton Lightweight Breathable"}
  ]
}
```

Corresponding Memory retrieval snapshot for the first example:

```json
{
  "intent": "buying",
  "category": "running shoes",
  "hard_constraints": {"color": ["black"]},
  "soft_preferences": {},
  "excluded": {}
}
```

---

## Optional model path

Default is **rules only** (same path as the public-set 0.8 run). Official scoring may disable the network.

| Variable | Role |
| --- | --- |
| `LLM_PARSER=0` | Force rules only; disables DeepSeek |
| `TECHJAM_PARSER_MODE=hybrid` | Allow DeepSeek when `DEEPSEEK_API_KEY` is set |
| `DEEPSEEK_API_KEY` | Optional DeepSeek key (not read unless mode is `hybrid` or `deepseek`) |
| `DEEPSEEK_PARSER_MODEL` | Default `deepseek-chat` |
| `DEEPSEEK_BASE_URL` | Default `https://api.deepseek.com` |
| `DEEPSEEK_PARSER_TIMEOUT_SECONDS` | Default 2.5s, capped at 8s |

API keys must not be committed. Token counts are reported on `ParseResult` and forwarded in the Agent `usage` field. Missing key, timeout, or invalid JSON falls back to rules.

The model extracts slots from the user message only. It is not used to generate product IDs. Short color/material tokens from the model are kept; identifying catalog phrases still come from the rule merge.

---

## Tests

From the repository root:

```bash
python -m unittest tests.test_memory tests.test_understanding
```

Use Python 3.10+. `tests.test_memory` covers Assignment 3. `tests.test_understanding` covers this module (rules plus mocked LLM). Agent integration tests disable the LLM via `LLM_PARSER=0`.

---

## Constraint taxonomy

`constraint_kind(slot, op)` labels each Memory update without changing the Memory contract:

- `hard`: brand, color, size, price, rating (`set` / `add`)
- `soft`: material, style, feature, use_case (`set` / `add`)
- `negative`: `exclude` (not black, don't want leather, without polyester)
- `decline`: explicit no-preference
- `clear`: remove / clear of a previous slot
- `task`: category / product type

`classify_intent(message, parsed)` returns `(Intent, confidence)`. Confidence is on `ParseResult`. Memory still receives `intent` only when this turn is confident enough to write it, so an unclear turn does not overwrite the session with `unknown`. Looking-without-a-hard-requirement writes `browsing`. A high confidence score on an override+decline turn is reported but not written into `ParseUpdate`.

`rewrite_queries(message, parsed, search_context)` builds at most eight retrieval strings (category, category+tokens, catalog feature lines, cleaned message). Tokens from slots this turn replaced are not reused from SearchContext. The Agent prepends these to `SearchContext.queries`.

Live DeepSeek calls are not part of CI (mocks only).

Catalog ProductStore loading (Assignment 1), hybrid/semantic retrieval (Assignment 4), and fusion/reranking (Assignment 5) are out of scope for this package. Field mapping uses `catalog_vocab.py`, aligned with Assignment 1 token lists.
