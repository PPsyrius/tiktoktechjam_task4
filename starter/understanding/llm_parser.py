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
    UpdateOperation,
)
from starter.understanding.query_parser import classify_intent, constraint_kind, parse_user_message
from starter.understanding.query_rewriter import rewrite_queries


DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_DEEPSEEK_TIMEOUT_SECONDS = 2.5
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

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
- reset_task: true only if the user starts a new shopping task (new category). Same-task replacements are false.
- slots: category, product_type, brand, color, size, price_min, price_max, rating_min, material, style, feature, use_case
- ops: set, add, remove, clear, decline, exclude
- decline has no value. Use it for "no preference" / "use your judgment".
- exclude is for negation ("not black", "don't want leather", "without polyester").
- remove/clear is for dropping an earlier slot ("forget about color").
- price_min / price_max / rating_min must be numbers. Map budget-under-$X to price_max.
- color/material should be short normalized tokens (black, cotton), not full sentences.
- Also add the user's exact requirement phrase as feature when it is longer than one token.
- Only include updates supported by this turn. Empty updates are allowed.
- Do not invent constraints the user did not state. Do not replace a long catalog line with a short token.

Examples:
User: I'm looking for running shoes. A key requirement is: cotton.
{"intent":"buying","reset_task":false,"updates":[{"slot":"category","op":"set","value":"running shoes"},{"slot":"material","op":"set","value":"cotton"}]}

User: I'm looking for winter boots, but I'm still exploring.
{"intent":"browsing","reset_task":false,"updates":[{"slot":"category","op":"set","value":"winter boots"}]}

User: I don't have a preference for color; please use your judgment.
{"intent":null,"reset_task":false,"updates":[{"slot":"color","op":"decline"}]}

User: Actually, ignore my earlier preference. What I need is: 100% Cotton Lightweight.
{"intent":"buying","reset_task":true,"updates":[{"slot":"material","op":"set","value":"cotton"},{"slot":"feature","op":"add","value":"100% Cotton Lightweight"}]}

User: I need Nike shoes, not black, under $80.
{"intent":"buying","reset_task":false,"updates":[{"slot":"category","op":"set","value":"Nike shoes"},{"slot":"brand","op":"set","value":"Nike"},{"slot":"color","op":"exclude","value":"black"},{"slot":"price_max","op":"set","value":80}]}
"""


@dataclass
class ParseResult:
    parsed: ParseUpdate | None
    source: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None
    intent_confidence: float = 0.0
    constraint_kinds: tuple[str, ...] = ()
    query_rewrites: tuple[str, ...] = ()


def _models_disabled() -> bool:
    flag = os.environ.get("LLM_PARSER", "1").strip().lower()
    return flag in {"0", "false", "off", "no"}


def deepseek_enabled() -> bool:
    if _models_disabled():
        return False
    mode = os.environ.get("TECHJAM_PARSER_MODE", "rules").strip().lower()
    if mode not in {"hybrid", "deepseek"}:
        return False
    return bool(deepseek_api_key())


def deepseek_api_key() -> str:
    return os.environ.get("DEEPSEEK_API_KEY", "").strip()


def _timeout_seconds() -> float:
    raw = os.environ.get(
        "DEEPSEEK_PARSER_TIMEOUT_SECONDS",
        str(DEFAULT_DEEPSEEK_TIMEOUT_SECONDS),
    )
    try:
        return max(0.2, min(float(raw), 8.0))
    except ValueError:
        return DEFAULT_DEEPSEEK_TIMEOUT_SECONDS


def _model_name() -> str:
    return (
        os.environ.get("DEEPSEEK_PARSER_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or DEFAULT_DEEPSEEK_MODEL
    ).strip()


def _base_url() -> str:
    return (
        os.environ.get("DEEPSEEK_BASE_URL")
        or DEFAULT_DEEPSEEK_BASE_URL
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
            update["value"] = text[:180]
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


def _merge_rule_snippets(parsed: ParseUpdate, fallback: ParseUpdate | None) -> ParseUpdate:
    """Keep rule catalog snippets, override, category, and declines the model dropped."""
    if fallback is None:
        return parsed
    existing = {(update.slot, update.op, update.value) for update in parsed.updates}
    existing_slot_ops = {(update.slot, update.op) for update in parsed.updates}
    extra: list = []
    for update in fallback.updates:
        key = (update.slot, update.op, update.value)
        if key in existing:
            continue
        if update.slot == "feature":
            extra.append(update)
        elif update.op in {
            UpdateOperation.DECLINE,
            UpdateOperation.EXCLUDE,
            UpdateOperation.REMOVE,
            UpdateOperation.CLEAR,
        } and (update.slot, update.op, update.value) not in existing:
            extra.append(update)
        elif update.slot == "category" and not any(item.slot == "category" for item in parsed.updates):
            extra.append(update)
        elif update.slot in {"brand", "size", "price_min", "price_max"} and (
            update.slot, update.op
        ) not in existing_slot_ops:
            extra.append(update)
    reset_task = fallback.reset_task
    intent = parsed.intent or fallback.intent
    if not extra and reset_task == parsed.reset_task and intent == parsed.intent:
        return parsed
    return ParseUpdate(
        session_id=parsed.session_id,
        intent=intent,
        updates=(*parsed.updates, *extra),
        reset_task=reset_task,
        source_turn=parsed.source_turn,
    )


def complete_chat(user_prompt: str) -> dict[str, Any] | None:
    key = deepseek_api_key()
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


def _apply_model_payload(
    session_id: str,
    user_message: str,
    turn: int,
    response: dict[str, Any],
    source: str,
    fallback: ParseUpdate | None,
) -> ParseResult:
    try:
        payload = extract_json_object(response["content"])
        parsed = parsed_update_from_llm(session_id, turn, payload)
        if parsed.intent is not None or parsed.updates or parsed.reset_task:
            parsed = _merge_rule_snippets(parsed, fallback)
            return ParseResult(
                parsed=parsed,
                source=source,
                prompt_tokens=response["prompt_tokens"],
                completion_tokens=response["completion_tokens"],
            )
        return ParseResult(
            parsed=fallback,
            source="rules",
            prompt_tokens=response["prompt_tokens"],
            completion_tokens=response["completion_tokens"],
            error="empty_llm_update",
        )
    except (ValueError, TypeError, json.JSONDecodeError):
        return ParseResult(
            parsed=fallback,
            source="rules",
            prompt_tokens=response["prompt_tokens"],
            completion_tokens=response["completion_tokens"],
            error="invalid_llm_json",
        )


def _enrich_parse_result(
    result: ParseResult,
    user_message: str,
    search_context: RetrievalState | dict[str, Any] | None,
) -> ParseResult:
    parsed = result.parsed
    intent, confidence = classify_intent(user_message, parsed)
    if parsed is None:
        result.intent_confidence = confidence
        result.constraint_kinds = ()
        result.query_rewrites = rewrite_queries(user_message, None, search_context)
        return result
    result.intent_confidence = confidence
    result.constraint_kinds = tuple(
        constraint_kind(update.slot, update.op) for update in parsed.updates
    )
    result.query_rewrites = rewrite_queries(user_message, parsed, search_context)
    return result


def parse_requirement(
    session_id: str,
    user_message: str,
    turn: int,
    search_context: RetrievalState | dict[str, Any] | None = None,
) -> ParseResult:
    """Rules always run. Optional DeepSeek may add slots; catalog snippets are kept."""
    fallback = parse_user_message(
        session_id,
        user_message,
        turn,
        search_context=search_context,
    )
    if not deepseek_enabled():
        return _enrich_parse_result(
            ParseResult(parsed=fallback, source="rules"),
            user_message,
            search_context,
        )
    user_prompt = (
        f"session_id: {session_id}\n"
        f"turn: {turn}\n"
        f"current_search_context: {_context_summary(search_context)}\n"
        f"user_message: {user_message}"
    )
    response = complete_chat(user_prompt)
    if response is not None:
        return _enrich_parse_result(
            _apply_model_payload(
                session_id, user_message, turn, response, "deepseek", fallback,
            ),
            user_message,
            search_context,
        )
    return _enrich_parse_result(
        ParseResult(parsed=fallback, source="rules"),
        user_message,
        search_context,
    )
