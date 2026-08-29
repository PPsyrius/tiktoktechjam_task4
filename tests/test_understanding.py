from __future__ import annotations

import unittest
from unittest.mock import patch

from starter.understanding import parse_requirement, parse_user_message
from starter.understanding.llm_parser import extract_json_object, parsed_update_from_llm


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
    def test_uses_regex_when_llm_disabled(self) -> None:
        with patch("starter.understanding.llm_parser.llm_enabled", return_value=False):
            result = parse_requirement(
                "s001",
                "I'm looking for running shoes. A key requirement is: cotton.",
                1,
            )
        self.assertEqual(result.source, "rules")
        assert result.parsed is not None
        self.assertEqual(result.parsed.intent.value, "buying")
        self.assertEqual(result.prompt_tokens, 0)

    def test_uses_llm_json_when_chat_succeeds(self) -> None:
        fake_response = {
            "content": (
                '{"intent":"buying","reset_task":false,'
                '"updates":[{"slot":"category","op":"set","value":"running shoes"},'
                '{"slot":"color","op":"set","value":"black"}]}'
            ),
            "prompt_tokens": 120,
            "completion_tokens": 30,
        }
        with patch("starter.understanding.llm_parser.llm_enabled", return_value=True), patch(
            "starter.understanding.llm_parser.complete_chat",
            return_value=fake_response,
        ):
            result = parse_requirement("s001", "I need black running shoes", 1)
        self.assertEqual(result.source, "llm")
        assert result.parsed is not None
        slots = {update.slot: update.value for update in result.parsed.updates}
        self.assertEqual(slots["category"], "running shoes")
        self.assertEqual(slots["color"], "black")
        self.assertEqual(result.prompt_tokens, 120)
        self.assertEqual(result.completion_tokens, 30)

    def test_falls_back_to_regex_on_invalid_llm_json(self) -> None:
        fake_response = {
            "content": "not json",
            "prompt_tokens": 10,
            "completion_tokens": 2,
        }
        with patch("starter.understanding.llm_parser.llm_enabled", return_value=True), patch(
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
