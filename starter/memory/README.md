# Memory State V4

This module stores the active shopping task for each session. Catalog records and raw
conversation history remain outside Memory.

## Input format

`MemoryService.apply_update()` and `MemoryService.apply_update_with_delta()` accept a
`ParseUpdate` object or an equivalent JSON-compatible dictionary. The session must be
initialized with `reset_state(session_id)` before an update is applied.

### `ParseUpdate`

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `session_id` | string | yes | Non-empty session identifier. |
| `intent` | `buying`, `browsing`, `unknown`, or `null` | no | Replaces the current shopping intent when supplied. |
| `updates` | array of `StateUpdate` | no | Deterministic slot operations; defaults to an empty array. |
| `reset_task` | boolean | no | Clears the active shopping task before applying this batch; defaults to `false`. |
| `source_turn` | positive integer or `null` | no | Dialogue turn used for idempotency and out-of-order protection. |

Each item in `updates` has this shape:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `slot` | string | yes | One of the supported direct, constraint, preference, or exclusion slots. |
| `op` | string | yes | `set`, `add`, `remove`, `clear`, `decline`, or `exclude`. |
| `value` | string or number | depends on `op` | Required for `set`, `add`, `remove`, and `exclude`; omitted for `clear` and `decline`. |

Supported slots are grouped as follows:

| Group | Slots |
| --- | --- |
| Direct state | `category`, `product_type`, `current_product` |
| Hard constraints | `price_min`, `price_max`, `brand`, `color`, `size`, `rating_min` |
| Soft preferences | `material`, `style`, `feature`, `use_case` |
| Legacy untyped exclusion | `excluded` |

Example input:

```json
{
  "session_id": "s001",
  "intent": "buying",
  "source_turn": 1,
  "updates": [
    {"slot": "category", "op": "set", "value": "running shoes"},
    {"slot": "price_max", "op": "set", "value": 120},
    {"slot": "color", "op": "set", "value": "black"}
  ]
}
```

## Output formats

### Full state output

`apply_update(parse_update)` returns the complete `CurrentState`. Internal fields are
included because they are needed for retry safety, clarification tracking, debugging,
and persistence.

| Field | Description |
| --- | --- |
| `session_id`, `schema_version` | State identity and persisted schema version. |
| `intent`, `category`, `product_type`, `current_product` | Active shopping task. |
| `constraints`, `preferences` | Positive hard constraints and soft preferences. |
| `excluded`, `excluded_by_slot` | Untyped legacy exclusions and typed exclusions. |
| `attribute_status` | `specified` or `no_preference`; a missing key means `unknown`. |
| `asked_attributes`, `asked_attribute_by_turn` | Clarification-question state. |
| `applied_turn_signatures` | Idempotency records for accepted source turns. |
| `change_history` | At most 20 recent state changes; raw messages are not stored. |
| `task_version`, `updated_at` | Shopping-task version and logical state version. |

Serialized output after applying the example input above:

```json
{
  "session_id": "s001",
  "schema_version": "4.0",
  "intent": "buying",
  "category": "running shoes",
  "product_type": null,
  "constraints": {
    "color": "black",
    "price_max": 120
  },
  "preferences": {},
  "excluded": [],
  "excluded_by_slot": {},
  "current_product": null,
  "attribute_status": {
    "category": "specified",
    "color": "specified",
    "price_max": "specified"
  },
  "asked_attributes": [],
  "asked_attribute_by_turn": {},
  "applied_turn_signatures": {
    "1": "{\"intent\":\"buying\",\"session_id\":\"s001\",\"source_turn\":1,\"updates\":[{\"op\":\"set\",\"slot\":\"category\",\"value\":\"running shoes\"},{\"op\":\"set\",\"slot\":\"price_max\",\"value\":120},{\"op\":\"set\",\"slot\":\"color\",\"value\":\"black\"}]}"
  },
  "change_history": [
    {
      "turn": 1,
      "slot": "category",
      "old": null,
      "new": "running shoes",
      "op": "set"
    },
    {
      "turn": 1,
      "slot": "color",
      "old": null,
      "new": {
        "value": "black",
        "status": "specified",
        "excluded": []
      },
      "op": "set"
    },
    {
      "turn": 1,
      "slot": "intent",
      "old": "unknown",
      "new": "buying",
      "op": "set"
    },
    {
      "turn": 1,
      "slot": "price_max",
      "old": null,
      "new": {
        "value": 120,
        "status": "specified",
        "excluded": []
      },
      "op": "set"
    }
  ],
  "task_version": 0,
  "updated_at": 1
}
```

### State and delta output

`apply_update_with_delta(parse_update)` returns a Python tuple of
`(CurrentState, StateDelta)`. A transport layer can serialize it as this envelope:

```json
{
  "state": {
    "session_id": "s001",
    "schema_version": "4.0",
    "intent": "buying",
    "category": "running shoes",
    "constraints": {
      "color": "black",
      "price_max": 120
    }
  },
  "delta": {
    "changed_slots": ["category", "color", "intent", "price_max"],
    "removed_slots": []
  }
}
```

The `state` object in this envelope is abbreviated for readability; the actual
`CurrentState` has the complete shape shown in the previous example.

## State semantics

`CurrentState.attribute_status` distinguishes three states for an attribute:

- missing key: `unknown` — the user has not expressed a decision;
- `specified` — the state contains an explicit value;
- `no_preference` — the user explicitly declined to constrain that attribute.

`asked_attributes` records which clarification attributes have already been requested.
`task_version` identifies shopping-task boundaries inside the same session.

`source_turn` makes updates idempotent. Retrying the same turn with identical content is
a no-op; reusing the turn number with different content raises a conflict. Question
selection is stored in `asked_attribute_by_turn`, so a retried request returns the same
`ask_attribute` instead of advancing the dialogue.

`schema_version` identifies the persisted state shape and is independent from
`task_version`. V4 keeps at most 20 structured state changes in `change_history` for
debugging; raw conversation messages are not stored.

## Update protocol

The original `set`, `add`, `remove`, and `clear` operations remain compatible. V2 adds
`decline`, and V4 adds typed `exclude` updates:

```json
{
  "session_id": "s001",
  "source_turn": 2,
  "updates": [
    {"slot": "brand", "op": "decline"}
  ]
}
```

`decline` removes an existing value and records `no_preference`. `clear` removes the
value and returns the attribute to `unknown`.

```json
{
  "session_id": "s001",
  "source_turn": 3,
  "updates": [
    {"slot": "brand", "op": "exclude", "value": "Nike"}
  ]
}
```

Setting or adding the same value later removes it from the structured exclusion. A
single attribute cannot contain the same positive and excluded value.

A new task can replace the old task atomically while preserving the session:

```json
{
  "session_id": "s001",
  "intent": "browsing",
  "reset_task": true,
  "updates": [
    {"slot": "category", "op": "set", "value": "laptop"}
  ]
}
```

If any update in the batch is invalid, neither the reset nor any partial update is
saved.

## Retrieval output

`get_retrieval_state(session_id)` returns only fields consumed by retrieval and ranking:

```json
{
  "schema_version": "4.0",
  "intent": "buying",
  "category": "running shoes",
  "product_type": null,
  "hard_constraints": {
    "price_max": 120,
    "color": ["black"]
  },
  "soft_preferences": {
    "material": ["cotton"]
  },
  "excluded": {
    "brand": ["Nike"]
  }
}
```

Dialogue metadata, applied-turn signatures, task versions, history, and timestamps are
not exposed to Retrieval.

`apply_update_with_delta(parse_update)` additionally returns a `StateDelta` containing
`changed_slots` and `removed_slots`. This lets downstream components invalidate only
the filters or ranking stages affected by an update.

## Public service operations

- `get_state(session_id)`
- `apply_update(parse_update)`
- `apply_update_with_delta(parse_update)`
- `get_retrieval_state(session_id)`
- `get_metrics(session_id=None)`
- `mark_attribute_asked(session_id, attribute)`
- `next_unasked_attribute(session_id, candidates)`
- `get_or_record_asked_attribute(session_id, source_turn, candidates)`
- `start_new_task(session_id, intent)`
- `reset_state(session_id, intent)`
- `delete_state(session_id)`

The in-memory store returns detached snapshots, so callers cannot mutate saved state
without going through the service.
