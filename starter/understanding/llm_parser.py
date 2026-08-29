from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from starter.memory.models import (
    ALLOWED_SLOTS,
    NUMERIC_SLOTS,
    ParseUpdate,
    RetrievalState,
)
from starter.understanding.query_parser import parse_user_message


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT_SECONDS = 1.5
DEFAULT_BASE_URL = "https://api.openai.com/v1"

SYSTEM_PROMPT = """\
You extract shopping requirements from one customer message for a clothing catalog.
Return JSON only. Do not use ground_truth, scenario_type, or product IDs.

Schema:
{
  "intent": "buying" | "browsing" | "unknown" | null,
  "reset_task": boolean,
  "updates": [{"slot": string, "op": string, "value": string|number}]
}

Rules:
- intent: buying if a hard requirement is stated; browsing if still exploring; unknown if unclear.
- reset_task: true only if the user replaces an earlier preference (intent override).
- slots: category, product_type, brand, color, size, price_min, price_max, rating_min, material, style, feature, use_case
- ops: set, add, remove, clear, decline, exclude
- decline has no value. Use it for "no preference" / "use your judgment".
- price_min / price_max / rating_min must be numbers. Map budget-under-$X to price_max.
- color/material should be short normalized tokens (black, cotton), not full sentences.
- Only include updates supported by this turn. Empty updates are allowed.
- Do not invent constraints the user did not state.

Examples:
User: I'm looking for running shoes. A key requirement is: cotton.
{"intent":"buying","reset_task":false,"updates":[{"slot":"category","op":"set","value":"running shoes"},{"slot":"material","op":"set","value":"cotton"}]}

User: I'm looking for winter boots, but I'm still exploring.
{"intent":"browsing","reset_task":false,"updates":[{"slot":"category","op":"set","value":"winter boots"}]}

User: I don't have a preference for color; please use your judgment.
{"intent":null,"reset_task":false,"updates":[{"slot":"color","op":"decline"}]}

User: Actually, ignore my earlier preference. What I need is: black.
{"intent":"buying","reset_task":true,"updates":[{"slot":"color","op":"set","value":"black"}]}
"""


@dataclass
class ParseResult:
    parsed: ParseUpdate | None
    source: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def llm_enabled() -> bool:
    flag = os.environ.get("LLM_PARSER", "1").strip().lower()
    if flag in {"0", "false", "off", "no"}:
        return False
    return bool(api_key())


def api_key() -> str:
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    ).strip()


def _timeout_seconds() -> float:
    raw = os.environ.get("LLM_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        return max(0.2, min(float(raw), 8.0))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _model_name() -> str:
    return (
        os.environ.get("LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_MODEL
    ).strip()


def _base_url() -> str:
    return (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or DEFAULT_BASE_URL
    ).rstrip("/")


def _context_summary(search_context: RetrievalState | dict[str, Any] | None) -> str:
    if search_context is None:
        return "{}"
    payload = search_context.to_dict() if isinstance(search_context, RetrievalState) else dict(search_context)
    compact = {
        "intent": payload.get("intent"),
        "category": payload.get("category"),
        "product_type": payload.get("product_type"),
        "hard_constraints": payload.get("hard_constraints") or {},
        "soft_preferences": payload.get("soft_preferences") or {},
        "excluded": payload.get("excluded") or {},
    }
    return json.dumps(compact, ensure_ascii=True)


def extract_json_object(text: str) -> dict[str, Any]:
    content = text.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM response did not contain a JSON object")
    parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON must be an object")
    return parsed


def _sanitize_updates(raw_updates: object) -> list[dict[str, Any]]:
    if not isinstance(raw_updates, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in raw_updates:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("slot") or "").strip()
        if slot == "budget":
            slot = "price_max"
        op = str(item.get("op") or "").strip().lower()
        if slot not in ALLOWED_SLOTS or op not in {"set", "add", "remove", "clear", "decline", "exclude"}:
            continue
        update: dict[str, Any] = {"slot": slot, "op": op}
        if op in {"clear", "decline"}:
            cleaned.append(update)
            continue
        value = item.get("value")
        if value is None or value == "":
            continue
        if slot in NUMERIC_SLOTS:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            update["value"] = int(number) if number.is_integer() else number
        else:
            text = str(value).strip()
            if not text:
                continue
            update["value"] = text[:120]
        cleaned.append(update)
    return cleaned


def parsed_update_from_llm(
    session_id: str,
    turn: int,
    payload: dict[str, Any],
) -> ParseUpdate:
    intent = payload.get("intent")
    if intent not in {"buying", "browsing", "unknown", None}:
        intent = None
    return ParseUpdate.from_dict({
        "session_id": session_id,
        "intent": intent,
        "reset_task": bool(payload.get("reset_task", False)),
        "source_turn": turn,
        "updates": _sanitize_updates(payload.get("updates")),
    })


def complete_chat(user_prompt: str) -> dict[str, Any] | None:
    key = api_key()
    if not key:
        return None
    body = json.dumps({
        "model": _model_name(),
        "temperature": 0,
        "max_tokens": 250,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{_base_url()}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_timeout_seconds()) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return None
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else 0
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else 0
    return {
        "content": content,
        "prompt_tokens": int(prompt_tokens) if isinstance(prompt_tokens, int) and prompt_tokens >= 0 else 0,
        "completion_tokens": int(completion_tokens) if isinstance(completion_tokens, int) and completion_tokens >= 0 else 0,
    }


def parse_requirement(
    session_id: str,
    user_message: str,
    turn: int,
    search_context: RetrievalState | dict[str, Any] | None = None,
) -> ParseResult:
    """Parse a turn with the LLM when configured; otherwise use the regex parser."""
    if llm_enabled():
        user_prompt = (
            f"session_id: {session_id}\n"
            f"turn: {turn}\n"
            f"current_search_context: {_context_summary(search_context)}\n"
            f"user_message: {user_message}"
        )
        response = complete_chat(user_prompt)
        if response is not None:
            try:
                payload = extract_json_object(response["content"])
                parsed = parsed_update_from_llm(session_id, turn, payload)
                if parsed.intent is not None or parsed.updates or parsed.reset_task:
                    return ParseResult(
                        parsed=parsed,
                        source="llm",
                        prompt_tokens=response["prompt_tokens"],
                        completion_tokens=response["completion_tokens"],
                    )
                fallback = parse_user_message(session_id, user_message, turn)
                return ParseResult(
                    parsed=fallback,
                    source="rules",
                    prompt_tokens=response["prompt_tokens"],
                    completion_tokens=response["completion_tokens"],
                    error="empty_llm_update",
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                fallback = parse_user_message(session_id, user_message, turn)
                return ParseResult(
                    parsed=fallback,
                    source="rules",
                    prompt_tokens=response["prompt_tokens"],
                    completion_tokens=response["completion_tokens"],
                    error="invalid_llm_json",
                )
    parsed = parse_user_message(session_id, user_message, turn)
    return ParseResult(parsed=parsed, source="rules")
