from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from local_experiments.parser.fusion_resolver import FusionResolver
from local_experiments.parser.llm_parser import DeepSeekSemanticParser, SemanticParser
from local_experiments.parser.models import ParserTrace, ResolvedParse
from local_experiments.parser.rule_extractor import CatalogLexicon, MessageNormalizer, RuleExtractor
from starter.memory import ParseUpdate
from starter.memory.models import CurrentState
from starter.memory.state_manager import StateManager

PARSER_ENV_KEYS = frozenset({
    "TECHJAM_PARSER_MODE",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_PARSER_MODEL",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_PARSER_TIMEOUT_SECONDS",
    "TECHJAM_LLM_QUESTIONS",
    "TECHJAM_QUESTION_POLICY",
})


def _load_project_parser_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in PARSER_ENV_KEYS:
            os.environ.setdefault(key, value.strip().strip("'\""))


@dataclass(frozen=True, slots=True)
class ParseResult:
    update: ParseUpdate
    trace: ParserTrace


class ParserValidator:
    """Builds the Memory contract and reuses StateManager for invariant checks."""

    def __init__(self, state_manager: StateManager | None = None) -> None:
        self.state_manager = state_manager or StateManager()

    def validate(
        self,
        session_id: str,
        turn: int | None,
        resolved: ResolvedParse,
        current_state: CurrentState | None,
    ) -> ParseUpdate:
        update = ParseUpdate(
            session_id=session_id,
            intent=resolved.intent,
            updates=tuple(candidate.to_state_update() for candidate in resolved.candidates),
            reset_task=resolved.reset_task,
            source_turn=turn,
        )
        if current_state is not None:
            self.state_manager.apply_update(current_state, update)
        return update


class DialogueParser:
    """Normalizer -> Rules + LLM -> field fusion -> validator -> ParseUpdate."""

    def __init__(
        self,
        *,
        semantic_parser: SemanticParser | None = None,
        normalizer: MessageNormalizer | None = None,
        rule_extractor: RuleExtractor | None = None,
        fusion_resolver: FusionResolver | None = None,
        validator: ParserValidator | None = None,
        selective_llm: bool = False,
    ) -> None:
        self.semantic_parser = semantic_parser
        self.normalizer = normalizer or MessageNormalizer()
        self.rule_extractor = rule_extractor or RuleExtractor()
        self.fusion_resolver = fusion_resolver or FusionResolver()
        self.validator = validator or ParserValidator()
        self.selective_llm = selective_llm
        self._cache: dict[tuple[str, int, str], ParseResult] = {}
        self.semantic_calls = 0

    @classmethod
    def from_environment(
        cls,
        catalog_path: str | Path = "data/catalog.jsonl",
    ) -> DialogueParser:
        _load_project_parser_env()
        mode = os.environ.get("TECHJAM_PARSER_MODE", "rules").strip().lower()
        if mode not in {"rules", "hybrid"}:
            raise ValueError("TECHJAM_PARSER_MODE must be 'rules' or 'hybrid'")
        lexicon = CatalogLexicon.from_catalog(catalog_path)
        semantic_parser = DeepSeekSemanticParser.from_environment() if mode == "hybrid" else None
        return cls(
            semantic_parser=semantic_parser,
            rule_extractor=RuleExtractor(lexicon),
            selective_llm=True,
        )

    def parse(
        self,
        session_id: str,
        message: str,
        turn: int | None = None,
        current_state: CurrentState | None = None,
    ) -> ParseUpdate:
        return self.parse_with_trace(session_id, message, turn, current_state).update

    def parse_with_trace(
        self,
        session_id: str,
        message: str,
        turn: int | None = None,
        current_state: CurrentState | None = None,
    ) -> ParseResult:
        normalized = self.normalizer.normalize(message)
        cache_key = (session_id, turn, normalized.text) if turn is not None else None
        if cache_key is not None and cache_key in self._cache:
            cached = self._cache[cache_key]
            self.validator.validate(session_id, turn, cached.trace.resolved, current_state)
            return cached

        rules = self.rule_extractor.extract(normalized)
        semantic = None
        if (
            normalized.text
            and self.semantic_parser is not None
            and (not self.selective_llm or self._needs_semantic_parser(normalized.text, rules))
        ):
            self.semantic_calls += 1
            semantic = self.semantic_parser.parse(normalized, current_state)
        resolved = self.fusion_resolver.resolve(rules, semantic, current_state)
        update = self.validator.validate(session_id, turn, resolved, current_state)
        result = ParseResult(update, ParserTrace(normalized, rules, semantic, resolved))
        if cache_key is not None:
            self._cache[cache_key] = result
        return result

    def reset_session(self, session_id: str) -> None:
        self._cache = {
            key: value for key, value in self._cache.items() if key[0] != session_id
        }

    @staticmethod
    def _needs_semantic_parser(message: str, rules) -> bool:
        if rules.override_signal or "exploration_language" in rules.intent_signals:
            return True
        if any(fact.kind == "unclassified_requirement" for fact in rules.facts):
            return True
        contextual = re.search(
            r"\b(?:actually|instead|earlier|anymore|those|them|it|"
            r"something|suitable|comfortable|casual|nicer|cheaper|"
            r"a little|a bit|stretch|travell?ing|vacation|occasion)\b",
            message,
            re.I,
        )
        if contextual:
            return True
        return not rules.candidates
