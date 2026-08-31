"""Print a short multi-turn walkthrough for the demo video."""
from __future__ import annotations

import argparse
import json

from agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an offline Shopping Copilot demo session.")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    agent = Agent(args.catalog)
    session_id = "demo-session"
    profile = {
        "purchase_frequency": "occasional",
        "average_prior_rating": 4.2,
        "rating_style": "selective",
        "preference_tags": ["comfortable", "casual"],
        "summary": "Usually prefers practical casual clothing.",
    }
    messages = (
        "I'm still exploring running shoes and keeping my options open.",
        "A key requirement is: black, lightweight, and under $90.",
        "Actually, ignore my earlier preference. I'm looking for jackets. A key requirement is: polyester.",
    )

    try:
        agent.reset(session_id, profile)
        print(f"semantic_enabled={agent.semantic_enabled}")
        for turn, user_message in enumerate(messages, 1):
            response = agent.respond(session_id, user_message, turn, args.top_k)
            concise = {
                "message": response["message"],
                "ask_attribute": response["ask_attribute"],
                "recommendations": response["recommendations"][:args.top_k],
                "usage": response["usage"],
            }
            print(f"\nTURN {turn}\nUSER: {user_message}\nAGENT:")
            print(json.dumps(concise, indent=2))
    finally:
        agent.close()


if __name__ == "__main__":
    main()
