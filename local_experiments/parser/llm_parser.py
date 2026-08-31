from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol

from local_experiments.parser.models import (
    Ambiguity,
    CandidateSource,
    FieldCandidate,
    NormalizedMessage,
    SemanticParse,
    TokenUsage,
)
from starter.memory import Intent
from starter.memory.models import CurrentState

SYSTEM_PROMPT = """You are the semantic layer of a shopping dialogue parser.
Return one JSON object only. Do not write prose.

The deterministic layer separately extracts literal prices, ratings, sizes, colors,
brands, materials, and explicit negations. Your job is semantic interpretation:
- extract what the user explicitly said; do not complete what they might want;
- infer intent only when supported: buying, browsing, or null;
- resolve category/product type, contextual preferences, references, refinements,
  removals, exclusions, and task overrides using current_state;
- use set/add/remove/clear/decline/exclude with the exact slot names below;
- use remove when a prior value is merely withdrawn, exclude when it remains rejected,
  decline when the user says they have no preference;
- never invent a numeric value. It must occur literally in the message or current_state;
- for relative language without an exact value, emit an ambiguity and leave the current
  field unchanged: do not clear, remove, or replace it;
- an override should clear/remove only the superseded fields, not every field;
- evidence must be a short exact span from the user message.
- never infer category, brand, material, color, price, style, or feature from lifestyle,
  occasion, season, or use-case language;
- for browsing, preserve explicit open descriptions as use_case/feature/style values
  using the user's own words instead of converting them into narrower constraints.

Operation limits:
- price_min, price_max, rating_min: set, clear, or decline only;
- category, product_type, current_product: set or clear only;
- excluded: set, add, remove, or clear only;
- all other slots: set, add, remove, clear, decline, or exclude.

Allowed slots:
category, product_type, current_product, price_min, price_max, brand, color, size,
rating_min, material, style, feature, use_case, excluded.

JSON shape:
{
  "intent": "buying" | "browsing" | null,
  "reset_task": false,
  "updates": [
    {
      "slot": "color",
      "op": "set",
      "value": "black",
      "confidence": 0.93,
      "evidence": "black"
    },
    {
      "slot": "color",
      "op": "remove",
      "value": "white",
      "confidence": 0.96,
      "evidence": "forget about white"
    }
  ],
  "ambiguities": [
    {"field": "price_max", "reason": "No exact new maximum was provided."}
  ]
}

Every update must contain value. Use null only for clear and decline. For
set/add/remove/exclude, value must be one string or number. Represent multiple
acceptable values as multiple add updates. Use empty arrays when there are none.
"""

def _update_schema(slots: list[str], operations: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["slot", "op", "value", "confidence", "evidence"],
        "properties": {
            "slot": {"enum": slots},
            "op": {"enum": operations},
            "value": {"type": ["string", "number", "null"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string", "minLength": 1},
        },
    }


PARSER_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "reset_task", "updates", "ambiguities"],
    "properties": {
        "intent": {"enum": ["buying", "browsing", None]},
        "reset_task": {"type": "boolean"},
        "updates": {
            "type": "array",
            "items": {
                "oneOf": [
                    _update_schema(
                        ["price_min", "price_max", "rating_min"],
                        ["set", "clear", "decline"],
                    ),
                    _update_schema(
                        ["category", "product_type", "current_product"],
                        ["set", "clear"],
                    ),
                    _update_schema(
                        ["brand", "color", "size", "material", "style", "feature", "use_case"],
                        ["set", "add", "remove", "clear", "decline", "exclude"],
                    ),
                    _update_schema(
                        ["excluded"],
                        ["set", "add", "remove", "clear"],
                    ),
                ]
            },
        },
        "ambiguities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "reason"],
                "properties": {
                    "field": {
                        "enum": [
                            "category",
                            "product_type",
                            "current_product",
                            "price_min",
                            "price_max",
                            "brand",
                            "color",
                            "size",
                            "rating_min",
                            "material",
                            "style",
                            "feature",
                            "use_case",
                            "excluded",
                            "budget",
                            "other",
                        ]
                    },
                    "reason": {"type": "string", "minLength": 1},
                },
            },
        },
    },
}


class SemanticParser(Protocol):
    def parse(
        self,
        message: NormalizedMessage,
        current_state: CurrentState | None,
    ) -> SemanticParse: ...


class DeepSeekParserError(RuntimeError):
    pass


class DeepSeekSemanticParser:
    """Strict DeepSeek V4 Flash adapter; failures are surfaced to the caller."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 15.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("DeepSeek api_key must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    @classmethod
    def from_environment(cls) -> DeepSeekSemanticParser:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "TECHJAM_PARSER_MODE=hybrid requires DEEPSEEK_API_KEY; "
                "the parser will not silently fall back to rules"
            )
        timeout_text = os.environ.get("DEEPSEEK_PARSER_TIMEOUT_SECONDS", "15")
        try:
            timeout = float(timeout_text)
        except ValueError as exc:
            raise ValueError("DEEPSEEK_PARSER_TIMEOUT_SECONDS must be numeric") from exc
        return cls(
            api_key,
            model=os.environ.get("DEEPSEEK_PARSER_MODEL", "deepseek-v4-flash"),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout_seconds=timeout,
        )

    def parse(
        self,
        message: NormalizedMessage,
        current_state: CurrentState | None,
    ) -> SemanticParse:
        payload = self.post_response(self.request_body(message, current_state))
        return self.parse_response(payload)

    def post_response(self, body: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise DeepSeekParserError(f"DeepSeek HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise DeepSeekParserError(f"DeepSeek request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise DeepSeekParserError("DeepSeek request timed out") from exc
        except json.JSONDecodeError as exc:
            raise DeepSeekParserError("DeepSeek returned a non-JSON HTTP response") from exc
        if not isinstance(payload, dict):
            raise DeepSeekParserError("DeepSeek returned a non-object HTTP response")
        return payload

    def request_body(
        self,
        message: NormalizedMessage,
        current_state: CurrentState | None,
    ) -> dict:
        user_payload = json.dumps(
            {"message": message.text, "current_state": self._state_context(current_state)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return {
            "model": self.model,
            "instructions": SYSTEM_PROMPT,
            "input": user_payload,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "shopping_dialogue_parse",
                    "schema": PARSER_OUTPUT_SCHEMA,
                }
            },
            "reasoning": {"effort": "none"},
            "temperature": 0,
            "max_output_tokens": 1200,
        }

    @staticmethod
    def parse_response(payload: object) -> SemanticParse:
        try:
            content = DeepSeekSemanticParser.response_text(payload)
            parsed = json.loads(content)
            return DeepSeekSemanticParser._semantic_parse(parsed, payload.get("usage"))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DeepSeekParserError(f"invalid DeepSeek parser response: {exc}") from exc

    @staticmethod
    def response_text(payload: object) -> str:
        if not isinstance(payload, dict):
            raise TypeError("response must be an object")
        if payload.get("status") != "completed":
            raise ValueError(f"unexpected response status: {payload.get('status')!r}")
        output = payload["output"]
        if not isinstance(output, list):
            raise TypeError("response output must be an array")
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                raise TypeError("message content must be an array")
            texts.extend(
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            )
        result = "".join(texts)
        if not result.strip():
            raise ValueError("response content is empty")
        return result

    @staticmethod
    def _semantic_parse(parsed: object, usage_payload: object) -> SemanticParse:
        if not isinstance(parsed, dict):
            raise TypeError("model JSON must be an object")
        unknown = set(parsed) - {"intent", "reset_task", "updates", "ambiguities"}
        if unknown:
            raise ValueError(f"unknown model fields: {sorted(unknown)}")

        intent_value = parsed.get("intent")
        intent = None if intent_value is None else Intent(intent_value)
        reset_task = parsed.get("reset_task", False)
        if not isinstance(reset_task, bool):
            raise TypeError("reset_task must be boolean")

        updates_payload = parsed.get("updates", [])
        if not isinstance(updates_payload, list):
            raise TypeError("updates must be an array")
        candidates: list[FieldCandidate] = []
        for item in updates_payload:
            if not isinstance(item, dict):
                raise TypeError("each model update must be an object")
            unknown_update = set(item) - {"slot", "op", "value", "confidence", "evidence"}
            if unknown_update:
                raise ValueError(f"unknown model update fields: {sorted(unknown_update)}")
            required = {"slot", "op", "value", "confidence", "evidence"}
            missing = required - set(item)
            if missing:
                raise ValueError(f"missing model update fields: {sorted(missing)}")
            candidates.append(FieldCandidate(
                slot=item["slot"],
                op=item["op"],
                value=item.get("value"),
                source=CandidateSource.LLM,
                confidence=item["confidence"],
                evidence=item["evidence"],
            ))

        ambiguities_payload = parsed.get("ambiguities", [])
        if not isinstance(ambiguities_payload, list):
            raise TypeError("ambiguities must be an array")
        ambiguities: list[Ambiguity] = []
        for item in ambiguities_payload:
            if not isinstance(item, dict) or set(item) != {"field", "reason"}:
                raise ValueError("each ambiguity requires exactly field and reason")
            ambiguities.append(Ambiguity(item["field"], item["reason"]))

        usage = TokenUsage()
        if usage_payload is not None:
            if not isinstance(usage_payload, dict):
                raise TypeError("usage must be an object")
            usage = TokenUsage(
                prompt_tokens=usage_payload.get("input_tokens", 0),
                completion_tokens=usage_payload.get("output_tokens", 0),
            )
        return SemanticParse(
            intent=intent,
            candidates=tuple(candidates),
            reset_task=reset_task,
            ambiguities=tuple(ambiguities),
            usage=usage,
        )

    @staticmethod
    def _state_context(current_state: CurrentState | None) -> dict | None:
        if current_state is None:
            return None
        serialized = current_state.to_dict()
        return {
            "intent": current_state.intent.value,
            "category": current_state.category,
            "product_type": current_state.product_type,
            "current_product": current_state.current_product,
            "constraints": serialized["constraints"],
            "preferences": serialized["preferences"],
            "excluded": list(current_state.excluded),
            "excluded_by_slot": serialized["excluded_by_slot"],
            "attribute_status": serialized["attribute_status"],
        }
