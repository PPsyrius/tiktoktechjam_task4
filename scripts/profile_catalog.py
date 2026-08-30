"""Profile the participant-visible catalog without modifying it."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from starter.catalog import CatalogLoader
from starter.catalog.feature_extractor import DETAIL_ATTRIBUTE_KEYS, parse_price


PROFILE_FIELDS = (
    "parent_asin",
    "title",
    "categories",
    "features",
    "description",
    "details",
    "store",
    "price",
    "average_rating",
    "rating_number",
)


def _is_empty(value: object) -> bool:
    return (
        value is None
        or isinstance(value, str) and not value.strip()
        or isinstance(value, (list, tuple, dict)) and not value
    )


def profile(path: str | Path) -> dict:
    field_stats = {
        field: {
            "present": 0,
            "empty": 0,
            "types": collections.Counter(),
        }
        for field in PROFILE_FIELDS
    }
    asin_counts: collections.Counter[str] = collections.Counter()
    category_lengths: collections.Counter[int] = collections.Counter()
    category_paths: collections.Counter[str] = collections.Counter()
    detail_keys: collections.Counter[str] = collections.Counter()
    mapped_detail_keys: collections.Counter[str] = collections.Counter()
    price_parseable = 0
    price_nonempty = 0
    price_unparseable_examples: list[object] = []
    total = 0

    for record in CatalogLoader(path).records():
        total += 1
        for field in PROFILE_FIELDS:
            if field in record:
                field_stats[field]["present"] += 1
            value = record.get(field)
            field_stats[field]["types"][type(value).__name__] += 1
            field_stats[field]["empty"] += int(_is_empty(value))

        asin = record.get("parent_asin")
        if isinstance(asin, str):
            asin_counts[asin] += 1
        categories = record.get("categories")
        if isinstance(categories, list):
            category_lengths[len(categories)] += 1
            category_paths[" > ".join(str(value) for value in categories)] += 1
        details = record.get("details")
        if isinstance(details, dict):
            for key in details:
                detail_keys[str(key)] += 1
                mapped_field = DETAIL_ATTRIBUTE_KEYS.get(" ".join(str(key).casefold().split()))
                if mapped_field:
                    mapped_detail_keys[mapped_field] += 1
        price = record.get("price")
        if not _is_empty(price):
            price_nonempty += 1
            if parse_price(price) is not None:
                price_parseable += 1
            elif price not in price_unparseable_examples and len(price_unparseable_examples) < 20:
                price_unparseable_examples.append(price)

    fields = {
        field: {
            "present": stats["present"],
            "missing": total - stats["present"],
            "empty": stats["empty"],
            "empty_rate": round(stats["empty"] / total if total else 0.0, 6),
            "types": dict(sorted(stats["types"].items())),
        }
        for field, stats in field_stats.items()
    }
    duplicate_ids = sum(count - 1 for count in asin_counts.values() if count > 1)
    return {
        "catalog": str(path),
        "source_fingerprint": CatalogLoader(path).source_fingerprint(),
        "row_count": total,
        "unique_parent_asin": len(asin_counts),
        "duplicate_parent_asin_rows": duplicate_ids,
        "fields": fields,
        "price": {
            "nonempty": price_nonempty,
            "parseable": price_parseable,
            "unparseable": price_nonempty - price_parseable,
            "unparseable_examples": price_unparseable_examples,
        },
        "category_path_lengths": {
            str(length): count for length, count in sorted(category_lengths.items())
        },
        "top_category_paths": category_paths.most_common(20),
        "top_details_keys": detail_keys.most_common(30),
        "mapped_details_attribute_counts": dict(sorted(mapped_detail_keys.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile catalog.jsonl fields")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = profile(args.catalog)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
