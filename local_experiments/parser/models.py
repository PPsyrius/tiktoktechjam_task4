from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from starter.memory import Intent, StateUpdate, UpdateOperation
from starter.memory.models import ALLOWED_SLOTS, DIALOGUE_ATTRIBUTES, JsonScalar


class CandidateSource(str, Enum):
    RULE = "rule"
    LLM = "llm"
    FUSED = "fused"


@dataclass(frozen=True, slots=True)
class NormalizedMessage:
    raw: str
    text: str
    lowered: str


@dataclass(frozen=True, slots=True)
class FieldCandidate:
    slot: str
    op: UpdateOperation
    value: JsonScalar | None
    source: CandidateSource
    confidence: float
    evidence: str

    def __post_init__(self) -> None:
        if self.slot not in ALLOWED_SLOTS:
            raise ValueError(f"unsupported candidate slot: {self.slot!r}")
        object.__setattr__(self, "op", UpdateOperation(self.op))
        object.__setattr__(self, "source", CandidateSource(self.source))
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("candidate confidence must be numeric")
        confidence = float(self.confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("candidate confidence must be between 0 and 1")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence", str(self.evidence).strip())

        validated = StateUpdate(self.slot, self.op, self.value)
        object.__setattr__(self, "value", validated.value)

    def to_state_update(self) -> StateUpdate:
        return StateUpdate(self.slot, self.op, self.value)

    def value_key(self) -> object:
        if isinstance(self.value, str):
            return self.value.casefold()
        return self.value

    def identity(self) -> tuple[str, UpdateOperation, object]:
        return self.slot, self.op, self.value_key()


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    kind: str
    value: JsonScalar
    confidence: float
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("fact kind must be a non-empty string")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("fact confidence must be numeric")
        confidence = float(self.confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("fact confidence must be between 0 and 1")
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "evidence", str(self.evidence).strip())


@dataclass(frozen=True, slots=True)
class Ambiguity:
    field: str
    reason: str

    def __post_init__(self) -> None:
        allowed = set(ALLOWED_SLOTS) | set(DIALOGUE_ATTRIBUTES)
        if self.field not in allowed:
            raise ValueError(f"unsupported ambiguity field: {self.field!r}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("ambiguity reason must be a non-empty string")
        object.__setattr__(self, "reason", self.reason.strip())


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass(frozen=True, slots=True)
class RuleExtraction:
    candidates: tuple[FieldCandidate, ...] = ()
    facts: tuple[ExtractedFact, ...] = ()
    intent_signals: tuple[str, ...] = ()
    reset_task: bool = False
    override_signal: bool = False


@dataclass(frozen=True, slots=True)
class SemanticParse:
    intent: Intent | None = None
    candidates: tuple[FieldCandidate, ...] = ()
    reset_task: bool = False
    ambiguities: tuple[Ambiguity, ...] = ()
    usage: TokenUsage = field(default_factory=TokenUsage)

    def __post_init__(self) -> None:
        if self.intent is not None:
            object.__setattr__(self, "intent", Intent(self.intent))
        if not isinstance(self.reset_task, bool):
            raise TypeError("reset_task must be a boolean")


@dataclass(frozen=True, slots=True)
class ResolvedParse:
    intent: Intent | None
    candidates: tuple[FieldCandidate, ...]
    reset_task: bool
    ambiguities: tuple[Ambiguity, ...] = ()


@dataclass(frozen=True, slots=True)
class ParserTrace:
    normalized: NormalizedMessage
    rules: RuleExtraction
    semantic: SemanticParse | None
    resolved: ResolvedParse

    @property
    def usage(self) -> TokenUsage:
        return self.semantic.usage if self.semantic is not None else TokenUsage()
