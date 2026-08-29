# Assignment 2: User Requirement Understanding

Convert the current customer message into a structured `ParseUpdate` for Assignment 3 (Memory). This package does not search the catalog or produce the Top 10 ranking.

```text
user message → parse_requirement() → ParseUpdate → Memory.apply_update() → SearchContext
```

`starter/agent.py` calls this module, then Memory. Understanding never writes session state. Memory never parses natural language.

---

## Layout

```text
starter/understanding/
  __init__.py
  query_parser.py    rule-based parser (offline fallback)
  llm_parser.py      optional LLM parser
  README.md
```

Tests: `tests/test_understanding.py`.

| File | Responsibility |
| --- | --- |
| `query_parser.py` | `parse_user_message(session_id, user_message, turn)` using patterns for Buying, Browsing, decline, and intent override |
| `llm_parser.py` | `parse_requirement(...)` — LLM when configured, otherwise the rule-based parser |
| `__init__.py` | Public API: `parse_requirement`, `parse_user_message`, `ParseResult` |

Agent entry point:

```python
from starter.understanding import parse_requirement

result = parse_requirement(session_id, user_message, turn, search_context=retrieval_state)
```

`ParseResult` fields: `parsed` (`ParseUpdate` or `None`), `source` (`llm` or `rules`), `prompt_tokens`, `completion_tokens`, optional `error`.

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

Assignment 2 describes **this turn**. Assignment 3 merges turns into the session `SearchContext` (`get_retrieval_state`). Later turns send only new operations; Memory keeps prior constraints unless `reset_task` is set.

When `reset_task` is true, Memory clears the shopping task. The Agent may re-attach `category` before `apply_update` so the product type is not dropped on override.

Example `ParseUpdate`:

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

Corresponding Memory retrieval snapshot:

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

## LLM path

Used only when `OPENAI_API_KEY` or `LLM_API_KEY` is present in the process environment. Missing key, timeout, HTTP failure, or invalid JSON falls back to `query_parser.py`. Official scoring may disable network; the rule-based path must remain sufficient.

| Variable | Role |
| --- | --- |
| `OPENAI_API_KEY` / `LLM_API_KEY` | Enable LLM parsing |
| `LLM_PARSER=0` | Force rule-based parser |
| `LLM_MODEL` / `OPENAI_MODEL` | Default `gpt-4o-mini` |
| `OPENAI_BASE_URL` | Default `https://api.openai.com/v1` |
| `LLM_TIMEOUT` | Default 1.5s |

API keys must not be committed. Token counts are reported on `ParseResult` and forwarded in the Agent `usage` field.

The LLM extracts slots from the user message only. It is not used to generate product IDs from retrieved catalog text.

---

## Tests

From the repository root:

```bash
python -m unittest tests.test_memory tests.test_understanding
```

Use Python 3.10+. `tests.test_memory` covers Assignment 3. `tests.test_understanding` covers this module (rules plus mocked LLM). Agent integration tests disable the LLM via `LLM_PARSER=0`.

---

## Pending

Current implementation is a first-pass Assignment 2 that matches the Memory contract.

- Team spec files (`intent_classifier.py`, `constraint_parser.py`, `query_rewriter.py`) are not split out; logic lives in the two parsers above.
- Intent has no confidence score.
- No separate retrieval query-rewrite string; BM25 still uses the user message plus Memory `retrieval_text()`.
- Rule-based extraction is limited on long or messy constraint phrases; the LLM path is untested against a live API in CI (mocks only).
- Hard / soft / negative constraints use Memory ops in a basic way, not a full constraint taxonomy.

Catalog ProductStore (Assignment 1), hybrid/semantic retrieval (Assignment 4), and fusion/reranking (Assignment 5) are out of scope for this package.
