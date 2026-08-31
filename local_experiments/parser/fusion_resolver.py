from __future__ import annotations

from collections import defaultdict

from local_experiments.parser.models import (
    CandidateSource,
    FieldCandidate,
    ResolvedParse,
    RuleExtraction,
    SemanticParse,
)
from starter.memory import Intent, UpdateOperation
from starter.memory.models import CONSTRAINT_SLOTS, PREFERENCE_SLOTS, CurrentState

RULE_PRIMARY_SLOTS = frozenset({
    "price_min",
    "price_max",
    "rating_min",
    "size",
    "brand",
    "color",
})
LLM_PRIMARY_SLOTS = frozenset({
    "category",
    "product_type",
    "current_product",
    "style",
    "feature",
    "use_case",
})
NUMERIC_SLOTS = frozenset({"price_min", "price_max", "rating_min"})
NEGATIVE_OPS = frozenset({UpdateOperation.REMOVE, UpdateOperation.EXCLUDE})
POSITIVE_OPS = frozenset({UpdateOperation.SET, UpdateOperation.ADD})
COLOR_CANONICAL = {
    "grey": "gray",
    "multicolour": "multicolor",
    "multicolored": "multicolor",
}


class FusionConflictError(ValueError):
    pass


class FusionResolver:
    """Resolves candidates per slot instead of choosing one parser globally."""

    def resolve(
        self,
        rules: RuleExtraction,
        semantic: SemanticParse | None,
        current_state: CurrentState | None = None,
    ) -> ResolvedParse:
        rule_candidates = [self._canonical_candidate(candidate) for candidate in rules.candidates]
        candidates = list(rule_candidates)
        if rules.override_signal:
            candidates = [*self._targeted_override_clears(current_state), *candidates]
        candidates.extend(
            FieldCandidate(
                "feature",
                UpdateOperation.ADD,
                fact.value,
                CandidateSource.RULE,
                0.72,
                fact.evidence,
            )
            for fact in rules.facts
            if fact.kind == "unclassified_requirement"
        )
        if semantic is None:
            intent = self._rule_only_intent(rules)
            ambiguities = ()
            reset_task = rules.reset_task
        else:
            signals = set(rules.intent_signals)
            browsing_mode = (
                "exploration_language" in signals
                or (
                    current_state is not None
                    and current_state.intent is Intent.BROWSING
                    and "hard_constraint_language" not in signals
                )
            )
            intent = Intent.BROWSING if browsing_mode else semantic.intent
            ambiguities = semantic.ambiguities
            ambiguous_slots = {item.field for item in ambiguities}
            if "budget" in ambiguous_slots:
                ambiguous_slots.update({"price_min", "price_max"})
            semantic_candidates = tuple(
                self._canonical_candidate(candidate)
                for candidate in semantic.candidates
                if candidate.slot not in ambiguous_slots
            )
            semantic_candidates = self._protect_explicit_rule_values(
                rule_candidates,
                semantic_candidates,
            )
            if browsing_mode:
                semantic_candidates = self._conservative_browsing_candidates(
                    rule_candidates,
                    semantic_candidates,
                )
            candidates.extend(self._ground_semantic_numerics(rules, semantic_candidates))
            reset_task = rules.reset_task or semantic.reset_task

        merged = self._merge_identical(candidates)
        resolved: list[FieldCandidate] = []
        by_slot: dict[str, list[FieldCandidate]] = defaultdict(list)
        for candidate in merged:
            by_slot[candidate.slot].append(candidate)
        for slot, slot_candidates in by_slot.items():
            resolved.extend(self._resolve_slot(slot, slot_candidates, rules.override_signal))
        return ResolvedParse(intent, tuple(resolved), reset_task, ambiguities)

    @staticmethod
    def _conservative_browsing_candidates(
        rule_candidates: list[FieldCandidate],
        semantic_candidates: tuple[FieldCandidate, ...],
    ) -> tuple[FieldCandidate, ...]:
        rule_support = {
            (candidate.slot, candidate.value_key())
            for candidate in rule_candidates
            if candidate.value is not None
        }
        accepted: list[FieldCandidate] = []
        for candidate in semantic_candidates:
            if candidate.slot in CONSTRAINT_SLOTS or candidate.slot in {
                "category",
                "product_type",
                "current_product",
            }:
                if candidate.value is not None and (candidate.slot, candidate.value_key()) in rule_support:
                    accepted.append(candidate)
                continue
            if candidate.op in {
                UpdateOperation.CLEAR,
                UpdateOperation.DECLINE,
                UpdateOperation.REMOVE,
            }:
                accepted.append(candidate)
                continue
            if isinstance(candidate.value, str) and candidate.value.casefold() in candidate.evidence.casefold():
                accepted.append(candidate)
        return tuple(accepted)

    @staticmethod
    def _protect_explicit_rule_values(
        rule_candidates: list[FieldCandidate],
        semantic_candidates: tuple[FieldCandidate, ...],
    ) -> tuple[FieldCandidate, ...]:
        explicit_values: dict[str, set[object]] = defaultdict(set)
        for candidate in rule_candidates:
            if candidate.op in POSITIVE_OPS and candidate.value is not None:
                explicit_values[candidate.slot].add(candidate.value_key())
        return tuple(
            candidate
            for candidate in semantic_candidates
            if not (
                candidate.op in POSITIVE_OPS
                and candidate.slot in explicit_values
                and candidate.value_key() not in explicit_values[candidate.slot]
            )
        )

    @staticmethod
    def _rule_only_intent(rules: RuleExtraction) -> Intent | None:
        signals = set(rules.intent_signals)
        if "hard_constraint_language" in signals:
            return Intent.BUYING
        if "exploration_language" in signals:
            return Intent.BROWSING
        return None

    @staticmethod
    def _targeted_override_clears(
        current_state: CurrentState | None,
    ) -> tuple[FieldCandidate, ...]:
        if current_state is None:
            return ()
        mutable_slots = CONSTRAINT_SLOTS | PREFERENCE_SLOTS
        relevant = [
            change
            for change in current_state.change_history
            if change.slot in mutable_slots and change.new is not None
        ]
        if not relevant:
            return ()
        latest_turn = max(change.turn or 0 for change in relevant)
        recent = [change for change in relevant if (change.turn or 0) == latest_turn]
        for slot_group in (CONSTRAINT_SLOTS, PREFERENCE_SLOTS):
            change = next(
                (item for item in reversed(recent) if item.slot in slot_group),
                None,
            )
            if change is not None:
                return (FieldCandidate(
                    change.slot,
                    UpdateOperation.CLEAR,
                    None,
                    CandidateSource.RULE,
                    0.80,
                    "explicit override of the most recent preference",
                ),)
        return ()

    @staticmethod
    def _ground_semantic_numerics(
        rules: RuleExtraction,
        candidates: tuple[FieldCandidate, ...],
    ) -> tuple[FieldCandidate, ...]:
        grounded_values = {
            float(fact.value)
            for fact in rules.facts
            if fact.kind in {"money", "rating"} and isinstance(fact.value, (int, float))
        }
        result: list[FieldCandidate] = []
        for candidate in candidates:
            if (
                candidate.slot in NUMERIC_SLOTS
                and candidate.op is UpdateOperation.SET
                and float(candidate.value) not in grounded_values
            ):
                raise FusionConflictError(
                    f"LLM numeric candidate {candidate.slot}={candidate.value!r} "
                    "is not grounded in a rule-extracted fact"
                )
            result.append(candidate)
        return tuple(result)

    @staticmethod
    def _merge_identical(candidates: list[FieldCandidate]) -> tuple[FieldCandidate, ...]:
        merged: dict[tuple[str, UpdateOperation, object], FieldCandidate] = {}
        for candidate in candidates:
            key = candidate.identity()
            existing = merged.get(key)
            if existing is None:
                merged[key] = candidate
                continue
            if existing.source is candidate.source:
                if candidate.confidence > existing.confidence:
                    merged[key] = candidate
                continue
            evidence = " | ".join(dict.fromkeys(filter(None, (existing.evidence, candidate.evidence))))
            value = (
                existing.value
                if existing.source in {CandidateSource.RULE, CandidateSource.FUSED}
                else candidate.value
            )
            merged[key] = FieldCandidate(
                candidate.slot,
                candidate.op,
                value,
                CandidateSource.FUSED,
                1 - (1 - existing.confidence) * (1 - candidate.confidence),
                evidence,
            )
        return tuple(merged.values())

    @staticmethod
    def _canonical_candidate(candidate: FieldCandidate) -> FieldCandidate:
        if candidate.value is None or not isinstance(candidate.value, str):
            return candidate
        value = candidate.value
        if candidate.slot == "color":
            lowered = value.casefold()
            value = COLOR_CANONICAL.get(lowered, lowered)
        elif candidate.slot in {"material", "use_case"}:
            value = value.casefold()
        if value == candidate.value:
            return candidate
        return FieldCandidate(
            candidate.slot,
            candidate.op,
            value,
            candidate.source,
            candidate.confidence,
            candidate.evidence,
        )

    def _resolve_slot(
        self,
        slot: str,
        candidates: list[FieldCandidate],
        is_override: bool,
    ) -> tuple[FieldCandidate, ...]:
        declines = [candidate for candidate in candidates if candidate.op is UpdateOperation.DECLINE]
        clear = [candidate for candidate in candidates if candidate.op is UpdateOperation.CLEAR]
        remaining = [
            candidate
            for candidate in candidates
            if candidate.op not in {UpdateOperation.DECLINE, UpdateOperation.CLEAR}
        ]
        if declines:
            if remaining:
                raise FusionConflictError(f"{slot} cannot be declined and updated in the same parse")
            return (max(declines, key=self._priority),)
        if clear and remaining and not is_override:
            raise FusionConflictError(f"{slot} cannot be cleared and updated without an override")

        negative = [candidate for candidate in remaining if candidate.op in NEGATIVE_OPS]
        positive = [candidate for candidate in remaining if candidate.op in POSITIVE_OPS]
        negative_values = {candidate.value_key() for candidate in negative}
        for candidate in list(positive):
            if candidate.value_key() not in negative_values:
                continue
            matching_negative = [item for item in negative if item.value_key() == candidate.value_key()]
            if any(item.source in {CandidateSource.RULE, CandidateSource.FUSED} for item in matching_negative):
                positive.remove(candidate)
            else:
                raise FusionConflictError(
                    f"semantic parse both accepts and rejects {slot}={candidate.value!r}"
                )

        sets = [candidate for candidate in positive if candidate.op is UpdateOperation.SET]
        adds = [candidate for candidate in positive if candidate.op is UpdateOperation.ADD]
        selected_set = self._select_set(slot, sets)
        if selected_set is not None:
            adds = [candidate for candidate in adds if candidate.value_key() != selected_set.value_key()]

        ordered: list[FieldCandidate] = []
        if clear:
            ordered.append(max(clear, key=self._priority))
        ordered.extend(sorted(negative, key=self._operation_order))
        if selected_set is not None:
            ordered.append(selected_set)
        ordered.extend(adds)
        return tuple(ordered)

    @staticmethod
    def _select_set(slot: str, candidates: list[FieldCandidate]) -> FieldCandidate | None:
        if not candidates:
            return None
        if slot in LLM_PRIMARY_SLOTS:
            llm = [item for item in candidates if item.source in {CandidateSource.LLM, CandidateSource.FUSED}]
            if llm:
                return max(llm, key=FusionResolver._priority)
        if slot in RULE_PRIMARY_SLOTS:
            rules = [item for item in candidates if item.source in {CandidateSource.RULE, CandidateSource.FUSED}]
            if rules:
                return max(rules, key=FusionResolver._priority)
        return max(candidates, key=FusionResolver._priority)

    @staticmethod
    def _priority(candidate: FieldCandidate) -> tuple[float, int]:
        source_rank = {
            CandidateSource.RULE: 1,
            CandidateSource.LLM: 2,
            CandidateSource.FUSED: 3,
        }[candidate.source]
        return candidate.confidence, source_rank

    @staticmethod
    def _operation_order(candidate: FieldCandidate) -> int:
        return {
            UpdateOperation.REMOVE: 0,
            UpdateOperation.EXCLUDE: 1,
        }[candidate.op]
