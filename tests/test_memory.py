from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from starter.agent import Agent
from starter.memory import (
    AttributeStatus,
    CurrentState,
    Intent,
    MemoryService,
    OutOfOrderTurnError,
    ParseUpdate,
    StateUpdate,
    TurnConflictError,
    UpdateOperation,
)


def parsed(
    session_id: str,
    *updates: StateUpdate,
    intent: Intent | None = None,
    reset_task: bool = False,
    source_turn: int | None = None,
) -> ParseUpdate:
    return ParseUpdate(
        session_id=session_id,
        intent=intent,
        updates=updates,
        reset_task=reset_task,
        source_turn=source_turn,
    )


class MemoryServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = MemoryService()
        self.memory.reset_state("s001")

    def test_protocol_round_trip_matches_parse_update_shape(self) -> None:
        payload = {
            "session_id": "s001",
            "intent": "buying",
            "updates": [
                {"slot": "price_max", "op": "set", "value": 120},
                {"slot": "color", "op": "set", "value": "black"},
            ],
        }

        update = ParseUpdate.from_dict(payload)
        state = self.memory.apply_update(update)

        self.assertEqual(update.to_dict(), payload)
        self.assertEqual(state.intent, Intent.BUYING)
        self.assertEqual(state.constraints, {"price_max": 120, "color": "black"})
        self.assertEqual(state.updated_at, 1)

    def test_accumulates_distinct_attributes_across_turns(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("category", "set", "running shoes"),
        ))
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("price_max", "set", 100),
        ))
        state = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("color", "set", "black"),
        ))

        self.assertEqual(state.category, "running shoes")
        self.assertEqual(state.constraints, {"price_max": 100, "color": "black"})
        self.assertEqual(state.updated_at, 3)

    def test_set_overwrites_same_slot(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("price_max", "set", 100),
        ))
        state = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("price_max", "set", 150),
        ))

        self.assertEqual(state.constraints["price_max"], 150)
        self.assertEqual(state.updated_at, 2)

    def test_clear_removes_one_slot_and_preserves_others(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("brand", "set", "Nike"),
            StateUpdate("price_max", "set", 100),
        ))
        state = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("brand", "clear"),
        ))

        self.assertNotIn("brand", state.constraints)
        self.assertEqual(state.constraints["price_max"], 100)
        self.assertEqual(state.status_for("brand"), AttributeStatus.UNKNOWN)

    def test_decline_distinguishes_no_preference_from_unknown(self) -> None:
        initial = self.memory.get_state("s001")
        self.assertEqual(initial.status_for("material"), AttributeStatus.UNKNOWN)

        asked = self.memory.mark_attribute_asked("s001", "material")
        declined = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("material", "decline"),
        ))

        self.assertTrue(asked.was_asked("material"))
        self.assertNotIn("material", declined.preferences)
        self.assertEqual(declined.status_for("material"), AttributeStatus.NO_PREFERENCE)

        specified = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("material", "add", "cotton"),
        ))
        self.assertEqual(specified.status_for("material"), AttributeStatus.SPECIFIED)

        cleared = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("material", "clear"),
        ))
        self.assertEqual(cleared.status_for("material"), AttributeStatus.UNKNOWN)

    def test_budget_status_aggregates_both_price_slots(self) -> None:
        declined = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("price_min", "decline"),
            StateUpdate("price_max", "decline"),
        ))
        self.assertEqual(declined.status_for("budget"), AttributeStatus.NO_PREFERENCE)

        specified = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("price_max", "set", 100),
        ))
        self.assertEqual(specified.status_for("budget"), AttributeStatus.SPECIFIED)

    def test_asked_attributes_are_ordered_and_idempotent(self) -> None:
        first = self.memory.mark_attribute_asked("s001", "material")
        repeated = self.memory.mark_attribute_asked("s001", "material")

        self.assertEqual(first.asked_attributes, ["material"])
        self.assertEqual(repeated.asked_attributes, ["material"])
        self.assertEqual(first.updated_at, 1)
        self.assertEqual(repeated.updated_at, 1)
        self.assertEqual(
            self.memory.next_unasked_attribute("s001", ["material", "color"]),
            "color",
        )

    def test_question_selection_is_stable_for_turn_retries(self) -> None:
        first = self.memory.get_or_record_asked_attribute(
            "s001",
            1,
            ["material", "feature"],
        )
        retry = self.memory.get_or_record_asked_attribute(
            "s001",
            1,
            ["material", "feature"],
        )
        second_turn = self.memory.get_or_record_asked_attribute(
            "s001",
            2,
            ["material", "feature"],
        )
        state = self.memory.get_state("s001")

        self.assertEqual(first, "material")
        self.assertEqual(retry, "material")
        self.assertEqual(second_turn, "feature")
        self.assertEqual(state.asked_attributes, ["material", "feature"])
        self.assertEqual(
            state.asked_attribute_by_turn,
            {1: "material", 2: "feature"},
        )
        self.assertEqual(state.updated_at, 2)

    def test_duplicate_source_turn_is_idempotent(self) -> None:
        update = parsed(
            "s001",
            StateUpdate("color", "set", "black"),
            source_turn=1,
        )

        first = self.memory.apply_update(update)
        retry = self.memory.apply_update(update)

        self.assertEqual(retry.to_dict(), first.to_dict())
        self.assertEqual(retry.updated_at, 1)
        self.assertEqual(list(retry.applied_turn_signatures), [1])

    def test_same_source_turn_with_different_content_is_rejected(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("color", "set", "black"),
            source_turn=1,
        ))

        with self.assertRaisesRegex(TurnConflictError, "different content"):
            self.memory.apply_update(parsed(
                "s001",
                StateUpdate("color", "set", "white"),
                source_turn=1,
            ))

        self.assertEqual(self.memory.get_state("s001").constraints["color"], "black")

    def test_unseen_older_turn_is_rejected(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("color", "set", "black"),
            source_turn=2,
        ))

        with self.assertRaisesRegex(OutOfOrderTurnError, "older"):
            self.memory.apply_update(parsed(
                "s001",
                StateUpdate("brand", "set", "Nike"),
                source_turn=1,
            ))

    def test_add_remove_and_duplicate_are_deterministic(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("brand", "set", "Nike"),
        ))
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("brand", "add", "Adidas"),
            StateUpdate("brand", "add", "Adidas"),
        ))
        state = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("brand", "remove", "Nike"),
        ))

        self.assertEqual(state.constraints["brand"], "Adidas")
        self.assertEqual(state.updated_at, 3)

    def test_excluded_has_set_add_remove_and_clear_semantics(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("excluded", "set", "leather"),
            StateUpdate("excluded", "add", "red"),
        ))
        state = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("excluded", "remove", "leather"),
        ))
        self.assertEqual(state.excluded, ["red"])

        cleared = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("excluded", "clear"),
        ))
        self.assertEqual(cleared.excluded, [])

    def test_specified_value_can_become_a_structured_exclusion(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("color", "set", "black"),
        ))
        state, delta = self.memory.apply_update_with_delta(parsed(
            "s001",
            StateUpdate("color", "exclude", "black"),
        ))

        self.assertNotIn("color", state.constraints)
        self.assertEqual(state.excluded_by_slot, {"color": "black"})
        self.assertEqual(state.status_for("color"), AttributeStatus.UNKNOWN)
        self.assertEqual(delta.to_dict(), {
            "changed_slots": ["color"],
            "removed_slots": [],
        })

    def test_positive_value_removes_matching_structured_exclusion(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("brand", "exclude", "Nike"),
        ))
        state = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("brand", "set", "Nike"),
        ))

        self.assertEqual(state.constraints["brand"], "Nike")
        self.assertNotIn("brand", state.excluded_by_slot)
        self.assertEqual(state.status_for("brand"), AttributeStatus.SPECIFIED)

    def test_decline_clear_and_set_have_explicit_status_transitions(self) -> None:
        declined = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("material", "decline"),
        ))
        self.assertEqual(declined.status_for("material"), AttributeStatus.NO_PREFERENCE)

        specified = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("material", "add", "cotton"),
        ))
        self.assertEqual(specified.preferences["material"], ("cotton",))
        self.assertEqual(specified.status_for("material"), AttributeStatus.SPECIFIED)

        cleared = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("material", "clear"),
        ))
        self.assertEqual(cleared.status_for("material"), AttributeStatus.UNKNOWN)

        reset_value = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("material", "set", "wool"),
        ))
        self.assertEqual(reset_value.preferences["material"], "wool")
        self.assertEqual(reset_value.status_for("material"), AttributeStatus.SPECIFIED)

    def test_retrieval_state_projects_only_downstream_fields(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("category", "set", "running shoes"),
            StateUpdate("price_max", "set", 120),
            StateUpdate("color", "set", "black"),
            StateUpdate("material", "add", "cotton"),
            StateUpdate("brand", "exclude", "Nike"),
            StateUpdate("excluded", "add", "leather"),
            intent=Intent.BUYING,
            source_turn=1,
        ))
        self.memory.get_or_record_asked_attribute("s001", 1, ["material"])

        retrieval = self.memory.get_retrieval_state("s001")
        payload = retrieval.to_dict()

        self.assertEqual(payload, {
            "schema_version": "4.0",
            "intent": "buying",
            "category": "running shoes",
            "product_type": None,
            "hard_constraints": {
                "color": ["black"],
                "price_max": 120,
            },
            "soft_preferences": {
                "material": ["cotton"],
            },
            "excluded": {
                "brand": ["Nike"],
                "other": ["leather"],
            },
        })
        self.assertNotIn("asked_attributes", payload)
        self.assertNotIn("applied_turn_signatures", payload)

    def test_state_delta_and_history_capture_changes_and_removals(self) -> None:
        first, first_delta = self.memory.apply_update_with_delta(parsed(
            "s001",
            StateUpdate("brand", "set", "Nike"),
            StateUpdate("price_max", "set", 100),
            source_turn=1,
        ))
        second, second_delta = self.memory.apply_update_with_delta(parsed(
            "s001",
            StateUpdate("brand", "clear"),
            StateUpdate("price_max", "set", 150),
            source_turn=2,
        ))

        self.assertEqual(first_delta.to_dict(), {
            "changed_slots": ["brand", "price_max"],
            "removed_slots": [],
        })
        self.assertEqual(second_delta.to_dict(), {
            "changed_slots": ["price_max"],
            "removed_slots": ["brand"],
        })
        self.assertEqual(first.change_history[-1].turn, 1)
        self.assertEqual(second.change_history[-2].op, "set")
        self.assertEqual(second.change_history[-1].op, "clear")

    def test_history_is_limited_to_twenty_changes(self) -> None:
        for turn in range(1, 26):
            self.memory.apply_update(parsed(
                "s001",
                StateUpdate("color", "set", "black" if turn % 2 else "white"),
                source_turn=turn,
            ))

        history = self.memory.get_state("s001").change_history
        self.assertEqual(len(history), 20)
        self.assertEqual(history[0].turn, 6)
        self.assertEqual(history[-1].turn, 25)

    def test_metrics_record_success_noop_and_rollbacks(self) -> None:
        first = parsed(
            "s001",
            StateUpdate("color", "set", "black"),
            source_turn=1,
        )
        self.memory.apply_update(first)
        self.memory.apply_update(first)
        with self.assertRaises(TurnConflictError):
            self.memory.apply_update(parsed(
                "s001",
                StateUpdate("color", "set", "white"),
                source_turn=1,
            ))
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("brand", "set", "Nike"),
            source_turn=3,
        ))
        with self.assertRaises(OutOfOrderTurnError):
            self.memory.apply_update(parsed(
                "s001",
                StateUpdate("size", "set", "9"),
                source_turn=2,
            ))
        with self.assertRaises(ValueError):
            self.memory.apply_update(parsed(
                "s001",
                StateUpdate("price_min", "set", 100),
                StateUpdate("price_max", "set", 50),
                source_turn=4,
            ))

        metrics = self.memory.get_metrics("s001")
        self.assertEqual(metrics["update_count"], 6)
        self.assertEqual(metrics["applied_update_count"], 2)
        self.assertEqual(metrics["noop_count"], 1)
        self.assertEqual(metrics["rollback_count"], 3)
        self.assertEqual(metrics["turn_conflict_count"], 1)
        self.assertEqual(metrics["out_of_order_count"], 1)
        self.assertEqual(metrics["reset_count"], 1)
        self.assertGreater(metrics["memory_update_latency_ms"], 0)
        self.assertGreater(metrics["state_size_bytes"], 0)
        self.assertEqual(metrics["active_constraint_count"], 2)

    def test_reset_replaces_old_task(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("category", "set", "running shoes"),
            StateUpdate("color", "set", "black"),
            intent=Intent.BUYING,
        ))

        state = self.memory.reset_state("s001", Intent.BROWSING)

        self.assertEqual(state.intent, Intent.BROWSING)
        self.assertIsNone(state.category)
        self.assertEqual(state.constraints, {})
        self.assertEqual(state.updated_at, 0)

    def test_in_session_task_boundary_clears_old_task_atomically(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("category", "set", "running shoes"),
            StateUpdate("color", "set", "black"),
            StateUpdate("excluded", "add", "leather"),
            intent=Intent.BUYING,
        ))
        self.memory.mark_attribute_asked("s001", "material")
        before = self.memory.get_state("s001")

        state = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("category", "set", "laptop"),
            intent=Intent.BROWSING,
            reset_task=True,
        ))

        self.assertEqual(state.session_id, "s001")
        self.assertEqual(state.task_version, 1)
        self.assertEqual(state.updated_at, before.updated_at + 1)
        self.assertEqual(state.category, "laptop")
        self.assertEqual(state.intent, Intent.BROWSING)
        self.assertEqual(state.constraints, {})
        self.assertEqual(state.preferences, {})
        self.assertEqual(state.excluded, [])
        self.assertEqual(state.asked_attributes, [])
        self.assertEqual(state.attribute_status, {"category": AttributeStatus.SPECIFIED})

    def test_invalid_new_task_batch_preserves_previous_task(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("category", "set", "running shoes"),
        ))

        with self.assertRaisesRegex(ValueError, "price_min cannot exceed price_max"):
            self.memory.apply_update(parsed(
                "s001",
                StateUpdate("price_min", "set", 100),
                StateUpdate("price_max", "set", 50),
                reset_task=True,
            ))

        state = self.memory.get_state("s001")
        self.assertEqual(state.category, "running shoes")
        self.assertEqual(state.task_version, 0)

    def test_retried_task_reset_does_not_advance_task_version_twice(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("category", "set", "running shoes"),
            source_turn=1,
        ))
        new_task = parsed(
            "s001",
            StateUpdate("category", "set", "laptop"),
            reset_task=True,
            source_turn=2,
        )

        first = self.memory.apply_update(new_task)
        retry = self.memory.apply_update(new_task)

        self.assertEqual(first.task_version, 1)
        self.assertEqual(retry.task_version, 1)
        self.assertEqual(retry.category, "laptop")
        self.assertEqual(retry.updated_at, first.updated_at)

    def test_sessions_are_isolated(self) -> None:
        self.memory.reset_state("s002")
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("color", "set", "black"),
        ))
        self.memory.apply_update(parsed(
            "s002",
            StateUpdate("color", "set", "white"),
        ))

        self.assertEqual(self.memory.get_state("s001").constraints["color"], "black")
        self.assertEqual(self.memory.get_state("s002").constraints["color"], "white")

    def test_empty_or_idempotent_update_does_not_change_version(self) -> None:
        empty = self.memory.apply_update(parsed("s001"))
        self.assertEqual(empty.updated_at, 0)

        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("color", "set", "black"),
        ))
        repeated = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("color", "set", "black"),
        ))
        self.assertEqual(repeated.updated_at, 1)

    def test_invalid_batch_is_atomic(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("price_min", "set", 100),
        ))

        with self.assertRaisesRegex(ValueError, "price_min cannot exceed price_max"):
            self.memory.apply_update(parsed(
                "s001",
                StateUpdate("color", "set", "black"),
                StateUpdate("price_max", "set", 50),
            ))

        state = self.memory.get_state("s001")
        self.assertEqual(state.constraints, {"price_min": 100})
        self.assertEqual(state.updated_at, 1)

    def test_store_returns_detached_snapshots(self) -> None:
        state = self.memory.apply_update(parsed(
            "s001",
            StateUpdate("color", "set", "black"),
        ))
        state.constraints["color"] = "white"

        stored = self.memory.get_state("s001")
        self.assertEqual(stored.constraints["color"], "black")

    def test_invalid_protocol_values_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported slot"):
            StateUpdate("bogus", "set", "value")
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            StateUpdate("color", "set", " ")
        with self.assertRaisesRegex(ValueError, "does not accept"):
            StateUpdate("brand", "clear", "Nike")
        with self.assertRaisesRegex(ValueError, "use set or clear"):
            StateUpdate("price_max", "add", 50)
        with self.assertRaisesRegex(ValueError, "decline is not supported"):
            StateUpdate("category", "decline")
        with self.assertRaisesRegex(ValueError, "exclude is not supported"):
            StateUpdate("price_max", "exclude", 100)
        with self.assertRaises(ValueError):
            StateUpdate("color", "unknown-op", "black")
        with self.assertRaisesRegex(TypeError, "reset_task"):
            ParseUpdate("s001", reset_task="yes")
        with self.assertRaisesRegex(ValueError, "source_turn"):
            ParseUpdate("s001", source_turn=0)
        with self.assertRaisesRegex(ValueError, "session_id"):
            ParseUpdate("", updates=())
        with self.assertRaisesRegex(ValueError, "schema_version"):
            CurrentState("s002", schema_version="3.0")

    def test_save_state_revalidates_mutated_snapshots(self) -> None:
        state = self.memory.get_state("s001")
        state.constraints["bogus"] = "value"

        with self.assertRaisesRegex(ValueError, "unsupported constraint"):
            self.memory.save_state(state)

    def test_save_state_rejects_version_and_turn_metadata_regression(self) -> None:
        self.memory.apply_update(parsed(
            "s001",
            StateUpdate("color", "set", "black"),
            source_turn=2,
        ))
        current = self.memory.get_state("s001")

        stale = current.clone()
        stale.updated_at = 0
        with self.assertRaisesRegex(ValueError, "updated_at"):
            self.memory.save_state(stale)

        missing_signature = current.clone()
        missing_signature.updated_at += 1
        missing_signature.applied_turn_signatures.clear()
        with self.assertRaisesRegex(ValueError, "signatures"):
            self.memory.save_state(missing_signature)

    def test_direct_state_construction_normalizes_and_validates_strings(self) -> None:
        state = CurrentState("s002", category="  running shoes  ", excluded=[" leather "])
        self.assertEqual(state.category, "running shoes")
        self.assertEqual(state.excluded, ["leather"])

        with self.assertRaises((TypeError, ValueError)):
            CurrentState("s003", excluded=[float("nan")])


class AgentMemoryIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["LLM_PARSER"] = "0"
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.catalog_path = Path(self.temporary_directory.name) / "catalog.jsonl"
        products = [
            {
                "parent_asin": "A_BASIC",
                "title": "Basic shoe",
                "features": [],
                "description": [],
                "categories": ["Shoes"],
                "details": {},
                "store": "Example",
            },
            {
                "parent_asin": "B_BLACK",
                "title": "Black running shoe",
                "features": ["black"],
                "description": [],
                "categories": ["Shoes"],
                "details": {},
                "store": "Example",
            },
        ]
        self.catalog_path.write_text(
            "".join(json.dumps(product) + "\n" for product in products),
            encoding="utf-8",
        )
        self.agent = Agent(self.catalog_path)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_agent_reset_update_state_and_retrieval_interface(self) -> None:
        self.agent.reset("agent-session", {})
        self.agent.apply_update({
            "session_id": "agent-session",
            "intent": "buying",
            "updates": [
                {"slot": "category", "op": "set", "value": "running shoe"},
                {"slot": "color", "op": "set", "value": "black"},
            ],
        })

        response = self.agent.respond("agent-session", "shoe", 1, 10)
        state = self.agent.get_state("agent-session")

        self.assertEqual(state.to_retrieval_payload(), {
            "schema_version": "4.0",
            "intent": "buying",
            "category": "running shoe",
            "product_type": None,
            "hard_constraints": {"color": ["black"]},
            "soft_preferences": {},
            "excluded": {},
        })
        self.assertEqual(response["recommendations"][0]["parent_asin"], "B_BLACK")
        self.assertEqual(
            set(response),
            {"message", "ask_attribute", "recommendations", "usage"},
        )

    def test_agent_reset_clears_existing_state(self) -> None:
        self.agent.reset("reset-session", {})
        self.agent.apply_update(parsed(
            "reset-session",
            StateUpdate("color", UpdateOperation.SET, "black"),
        ))
        self.agent.reset("reset-session", {})

        self.assertEqual(self.agent.get_state("reset-session").constraints, {})

    def test_agent_respond_requires_reset(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reset must be called"):
            self.agent.respond("missing", "shoe", 1, 10)


if __name__ == "__main__":
    unittest.main()
