"""Build a one-off 100-session holdout from catalog ASINs unused by the public set."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


SCENARIOS = (
    ["buying"] * 40
    + ["browsing"] * 40
    + ["intent_override"] * 15
    + ["boundary"] * 5
)
TAGS = (
    "fit", "comfort", "style", "durability", "material",
    "weather", "warmth", "color",
)
FREQUENCIES = (
    "1-2 prior purchases",
    "3-4 prior purchases",
    "5+ prior purchases",
)
RATING_STYLES = ("usually positive", "critical", "mixed")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_intent_signal(product: dict) -> bool:
    if not str(product.get("title") or "").strip():
        return False
    features = product.get("features") or []
    details = product.get("details") or {}
    return bool(features or details)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 100-session local holdout set")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--public-set", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results/holdout_100.jsonl")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    used = {
        str(sample["ground_truth"]["parent_asin"])
        for sample in load_jsonl(Path(args.public_set))
    }
    eligible: list[dict] = []
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            product = json.loads(line)
            asin = str(product.get("parent_asin") or "").strip()
            if not asin or asin in used or not has_intent_signal(product):
                continue
            eligible.append(product)

    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    if len(eligible) < args.count:
        raise SystemExit(f"only {len(eligible)} unused catalog products available")

    scenarios = list(SCENARIOS)
    if args.count != len(scenarios):
        raise SystemExit("this script builds the fixed 100-session mix")
    rng.shuffle(scenarios)

    samples: list[dict] = []
    for index, (product, scenario) in enumerate(zip(eligible, scenarios), start=1):
        tags = rng.sample(TAGS, k=rng.randint(2, 4))
        rating_style = rng.choice(RATING_STYLES)
        samples.append({
            "sample_id": f"holdout_{index:04d}",
            "scenario_type": scenario,
            "difficulty_bucket": "holdout",
            "category_bucket": "clothing",
            "ground_truth": {"parent_asin": str(product["parent_asin"])},
            "user_profile": {
                "average_prior_rating": rng.choice([1.0, 3.0, 4.0, 5.0, None]),
                "preference_tags": tags,
                "purchase_frequency": rng.choice(FREQUENCIES),
                "rating_style": rating_style,
                "summary": (
                    f"Prior purchases emphasize {', '.join(tags)}; "
                    f"ratings are {rating_style}."
                ),
            },
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(sample, ensure_ascii=True) + "\n" for sample in samples),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output),
        "sample_count": len(samples),
        "excluded_public_asins": len(used),
        "scenario_counts": {
            name: sum(1 for sample in samples if sample["scenario_type"] == name)
            for name in ("buying", "browsing", "intent_override", "boundary")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
