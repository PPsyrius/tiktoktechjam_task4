from __future__ import annotations

import unittest
from unittest.mock import patch

from starter.understanding import (
    classify_intent,
    constraint_kind,
    parse_requirement,
    parse_user_message,
    rewrite_queries,
)
from starter.understanding.llm_parser import extract_json_object, parsed_update_from_llm
from starter.memory.models import Intent, UpdateOperation


class QueryParserTest(unittest.TestCase):
    def test_buying_message_updates_category_and_hard_constraint(self) -> None:
        parsed = parse_user_message(
            "s001",
            "I'm looking for running shoes. A key requirement is: cotton.",
            1,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.intent.value, "buying")
        slots = {update.slot: update.value for update in parsed.updates}
        self.assertEqual(slots["category"], "running shoes")
        self.assertEqual(slots["material"], "cotton")

    def test_browsing_message_sets_browsing_intent(self) -> None:
        parsed = parse_user_message(
            "s001",
            "I'm looking for winter boots, but I'm still exploring.",
            1,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.intent.value, "browsing")
        self.assertEqual(parsed.updates[0].value, "winter boots")

    def test_decline_and_override_messages(self) -> None:
        declined = parse_user_message(
            "s001",
            "I don't have a preference for color; please use your judgment.",
            2,
        )
        self.assertIsNotNone(declined)
        assert declined is not None
        self.assertEqual(declined.updates[0].slot, "color")
        self.assertEqual(declined.updates[0].op.value, "decline")

        override = parse_user_message(
            "s001",
            "Actually, ignore my earlier preference. What I need is: black.",
            3,
        )
        self.assertIsNotNone(override)
        assert override is not None
        self.assertTrue(override.reset_task)
        self.assertEqual(override.updates[0].slot, "color")
        self.assertEqual(override.updates[0].value, "black")

    def test_paraphrased_wrappers_keep_catalog_line(self) -> None:
        parsed = parse_user_message(
            "s001",
            "I'm looking for running shoes. They must be: 100% Cotton Lightweight Breathable.",
            1,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.intent.value, "buying")
        values = [(update.slot, update.value) for update in parsed.updates]
        self.assertIn(("category", "running shoes"), values)
        self.assertIn(("material", "cotton"), values)
        self.assertIn(("feature", "100% Cotton Lightweight Breathable"), values)

    def test_reply_keeps_joined_line_and_each_part(self) -> None:
        parsed = parse_user_message(
            "s001",
            "For that, what matters is: Machine Wash; Tumble Dry.",
            2,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        features = [update.value for update in parsed.updates if update.slot == "feature"]
        self.assertIn("Machine Wash; Tumble Dry", features)
        self.assertIn("Machine Wash", features)
        self.assertIn("Tumble Dry", features)

    def test_catalog_no_label_is_a_positive_requirement(self) -> None:
        parsed = parse_user_message(
            "s001",
            "For that, what matters is: Imported; No Closure closure.",
            2,
        )
        assert parsed is not None
        operations = {
            (update.slot, update.op, update.value) for update in parsed.updates
        }
        self.assertIn(
            ("feature", UpdateOperation.ADD, "No Closure closure"),
            operations,
        )
        self.assertFalse(any(update.op is UpdateOperation.EXCLUDE for update in parsed.updates))

    def test_explicit_negative_inside_requirement_stays_negative(self) -> None:
        parsed = parse_user_message(
            "s001",
            "For that, what matters is: not black; without polyester.",
            2,
        )
        assert parsed is not None
        operations = {
            (update.slot, update.op, update.value) for update in parsed.updates
        }
        self.assertIn(("color", UpdateOperation.EXCLUDE, "black"), operations)
        self.assertIn(("material", UpdateOperation.EXCLUDE, "polyester"), operations)

    def test_override_keeps_full_new_requirement(self) -> None:
        parsed = parse_user_message(
            "s001",
            "Forget what I said. What I need is: Color: Black Waterproof Shell.",
            3,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertTrue(parsed.reset_task)
        values = [(update.slot, update.value) for update in parsed.updates]
        self.assertIn(("color", "black"), values)
        self.assertIn(("feature", "Color: Black Waterproof Shell"), values)

    def test_intent_confidence_for_buying_and_browsing(self) -> None:
        buying = parse_user_message(
            "s001",
            "I'm looking for running shoes. A key requirement is: cotton.",
            1,
        )
        browsing = parse_user_message(
            "s001",
            "I'm looking for winter boots, but I'm still exploring.",
            1,
        )
        self.assertEqual(classify_intent("", buying)[0], Intent.BUYING)
        self.assertGreaterEqual(classify_intent("", buying)[1], 0.9)
        self.assertEqual(classify_intent("", browsing)[0], Intent.BROWSING)
        self.assertGreaterEqual(classify_intent("", browsing)[1], 0.9)

    def test_extracts_brand_size_budget_and_negation(self) -> None:
        parsed = parse_user_message(
            "s001",
            "I need running shoes, brand: Nike, size 10, not black, under $80.",
            1,
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.intent, Intent.BUYING)
        ops = {(update.slot, update.op.value, update.value) for update in parsed.updates}
        self.assertIn(("brand", "set", "Nike"), ops)
        self.assertIn(("size", "set", "10"), ops)
        self.assertIn(("color", "exclude", "black"), ops)
        self.assertIn(("price_max", "set", 80), ops)
        kinds = {constraint_kind(update.slot, update.op) for update in parsed.updates}
        self.assertIn("hard", kinds)
        self.assertIn("negative", kinds)

    def test_decline_variants_and_remove_slot(self) -> None:
        flexible = parse_user_message("s001", "I am flexible about brand.", 2)
        self.assertIsNotNone(flexible)
        assert flexible is not None
        self.assertEqual(flexible.updates[0].slot, "brand")
        self.assertEqual(flexible.updates[0].op, UpdateOperation.DECLINE)

        removed = parse_user_message("s001", "Please remove the color constraint.", 3)
        self.assertIsNotNone(removed)
        assert removed is not None
        self.assertEqual(removed.updates[0].slot, "color")
        self.assertEqual(removed.updates[0].op, UpdateOperation.CLEAR)

    def test_does_not_treat_simulator_miss_as_exclusion(self) -> None:
        parsed = parse_user_message(
            "s001",
            "Those options are not quite right yet. Ask me about one specific attribute.",
            2,
        )
        self.assertIsNone(parsed)

    def test_query_rewrites_prefer_catalog_phrases(self) -> None:
        parsed = parse_user_message(
            "s001",
            "I'm looking for running shoes. A key requirement is: 100% Cotton Lightweight.",
            1,
        )
        rewrites = rewrite_queries(
            "I'm looking for running shoes. A key requirement is: 100% Cotton Lightweight.",
            parsed,
        )
        self.assertIn("running shoes", rewrites)
        self.assertTrue(any("Cotton" in item or "cotton" in item.lower() for item in rewrites))
        self.assertLessEqual(len(rewrites), 8)
        self.assertFalse(any("ground_truth" in item for item in rewrites))


class LlmParserHelpersTest(unittest.TestCase):
    def test_extract_json_object_from_fenced_text(self) -> None:
        payload = extract_json_object(
            "```json\n{\"intent\":\"buying\",\"reset_task\":false,\"updates\":[]}\n```"
        )
        self.assertEqual(payload["intent"], "buying")

    def test_sanitizes_budget_and_drops_unknown_slots(self) -> None:
        parsed = parsed_update_from_llm(
            "s001",
            1,
            {
                "intent": "buying",
                "reset_task": False,
                "updates": [
                    {"slot": "budget", "op": "set", "value": "80"},
                    {"slot": "not_a_slot", "op": "set", "value": "x"},
                    {"slot": "color", "op": "decline"},
                ],
            },
        )
        slots = [(update.slot, update.op.value, update.value) for update in parsed.updates]
        self.assertEqual(parsed.intent.value, "buying")
        self.assertIn(("price_max", "set", 80), slots)
        self.assertIn(("color", "decline", None), slots)
        self.assertFalse(any(slot == "not_a_slot" for slot, _, _ in slots))


class LlmParserFallbackTest(unittest.TestCase):
    def test_uses_regex_when_deepseek_disabled(self) -> None:
        with patch("starter.understanding.llm_parser.deepseek_enabled", return_value=False):
            result = parse_requirement(
                "s001",
                "I'm looking for running shoes. A key requirement is: cotton.",
                1,
            )
        self.assertEqual(result.source, "rules")
        assert result.parsed is not None
        self.assertEqual(result.parsed.intent.value, "buying")
        self.assertEqual(result.prompt_tokens, 0)
        self.assertGreaterEqual(result.intent_confidence, 0.9)
        self.assertTrue(result.query_rewrites)
        self.assertIn("task", result.constraint_kinds)
        self.assertIn("soft", result.constraint_kinds)

    def test_inferred_intent_is_reported_but_not_written_into_update(self) -> None:
        with patch("starter.understanding.llm_parser.deepseek_enabled", return_value=False):
            result = parse_requirement(
                "s001",
                "Ignore my earlier preference. I am flexible about color.",
                2,
            )
        assert result.parsed is not None
        self.assertIsNone(result.parsed.intent)
        self.assertGreaterEqual(result.intent_confidence, 0.9)
        self.assertIn("decline", result.constraint_kinds)

    def test_uses_deepseek_json_when_chat_succeeds(self) -> None:
        fake_response = {
            "content": (
                '{"intent":"buying","reset_task":false,'
                '"updates":[{"slot":"category","op":"set","value":"running shoes"},'
                '{"slot":"color","op":"set","value":"black"}]}'
            ),
            "prompt_tokens": 120,
            "completion_tokens": 30,
        }
        with patch("starter.understanding.llm_parser.deepseek_enabled", return_value=True), patch(
            "starter.understanding.llm_parser.complete_chat",
            return_value=fake_response,
        ):
            result = parse_requirement("s001", "I need black running shoes", 1)
        self.assertEqual(result.source, "deepseek")
        assert result.parsed is not None
        slots = {update.slot: update.value for update in result.parsed.updates}
        self.assertEqual(slots["category"], "running shoes")
        self.assertEqual(slots["color"], "black")
        self.assertEqual(result.prompt_tokens, 120)
        self.assertEqual(result.completion_tokens, 30)

    def test_merges_rule_snippet_when_model_returns_short_token(self) -> None:
        fake_response = {
            "content": (
                '{"intent":"buying","reset_task":false,'
                '"updates":[{"slot":"material","op":"set","value":"cotton"}]}'
            ),
            "prompt_tokens": 40,
            "completion_tokens": 12,
        }
        with patch("starter.understanding.llm_parser.deepseek_enabled", return_value=True), patch(
            "starter.understanding.llm_parser.complete_chat",
            return_value=fake_response,
        ):
            result = parse_requirement(
                "s001",
                "I'm looking for shirts. A key requirement is: 100% Cotton Lightweight.",
                1,
            )
        self.assertEqual(result.source, "deepseek")
        assert result.parsed is not None
        values = [(update.slot, update.value) for update in result.parsed.updates]
        self.assertIn(("material", "cotton"), values)
        self.assertIn(("feature", "100% Cotton Lightweight"), values)
        self.assertIn(("category", "shirts"), values)

    def test_deepseek_hybrid_merges_snippets_and_can_be_forced_off(self) -> None:
        fake_response = {
            "content": (
                '{"intent":"buying","reset_task":true,'
                '"updates":[{"slot":"color","op":"set","value":"black"}]}'
            ),
            "prompt_tokens": 80,
            "completion_tokens": 20,
        }
        with patch("starter.understanding.llm_parser.deepseek_enabled", return_value=True), patch(
            "starter.understanding.llm_parser.complete_chat",
            return_value=fake_response,
        ):
            result = parse_requirement(
                "s001",
                "Forget what I said. What I need is: Color: Black Waterproof Shell.",
                3,
            )
        self.assertEqual(result.source, "deepseek")
        assert result.parsed is not None
        self.assertTrue(result.parsed.reset_task)
        values = [(update.slot, update.value) for update in result.parsed.updates]
        self.assertIn(("color", "black"), values)
        self.assertIn(("feature", "Color: Black Waterproof Shell"), values)

        with patch.dict("os.environ", {"LLM_PARSER": "0", "DEEPSEEK_API_KEY": "sk-test", "TECHJAM_PARSER_MODE": "hybrid"}):
            from starter.understanding.llm_parser import deepseek_enabled
            self.assertFalse(deepseek_enabled())

    def test_falls_back_to_regex_on_invalid_deepseek_json(self) -> None:
        fake_response = {
            "content": "not json",
            "prompt_tokens": 10,
            "completion_tokens": 2,
        }
        with patch("starter.understanding.llm_parser.deepseek_enabled", return_value=True), patch(
            "starter.understanding.llm_parser.complete_chat",
            return_value=fake_response,
        ):
            result = parse_requirement(
                "s001",
                "I'm looking for running shoes. A key requirement is: cotton.",
                1,
            )
        self.assertEqual(result.source, "rules")
        self.assertEqual(result.error, "invalid_llm_json")
        assert result.parsed is not None
        self.assertEqual(result.parsed.intent.value, "buying")
        self.assertEqual(result.prompt_tokens, 10)


if __name__ == "__main__":
    unittest.main()
