"""Minimal catalog adapter; task 1 can supply normalized rows via from_records."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Tuple
from types import MappingProxyType

from .types import ATTRIBUTE_FIELDS, Constraint

TEXT_FIELDS = ("title", "categories", "features", "details", "store", "description")


def text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _price(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip().removeprefix("$").replace(",", "")
        if not re.fullmatch(r"\d+(?:\.\d+)?", value):
            return None
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class Product:
    parent_asin: str
    fields: Tuple[str, ...]
    attributes: Mapping[str, Tuple[str, ...]]
    price: Optional[float]

    def __post_init__(self):
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "attributes", MappingProxyType(
            {key: tuple(values) for key, values in self.attributes.items()}))

    @property
    def search_text(self) -> str:
        return "\n".join(self.fields)


class ProductStore:
    def __init__(self, products) -> None:
        self.products = {}
        digest = hashlib.sha256(b"task4-product-store-v1\n")
        for product in products:
            if not product.parent_asin or product.parent_asin in self.products:
                raise ValueError("Missing or duplicate parent_asin: " + product.parent_asin)
            self.products[product.parent_asin] = product
            payload = [product.parent_asin, product.fields, dict(product.attributes), product.price]
            digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))
            digest.update(b"\n")
        self.fingerprint = digest.hexdigest()
        self.products = MappingProxyType(self.products)

    @classmethod
    def from_records(cls, records):
        def convert():
            for row in records:
                asin = row.get("parent_asin")
                if not isinstance(asin, str) or not asin.strip():
                    raise ValueError("parent_asin must be a nonempty string")
                attributes = {}
                # Only explicit facts: never infer brand from store or material from arbitrary prose.
                categories = row.get("categories") or []
                attributes["category"] = [categories] if isinstance(categories, str) else categories
                for key, value in (row.get("details") or {}).items():
                    name = normalized(key)
                    if name in ATTRIBUTE_FIELDS and name != "category":
                        attributes[name] = value
                attributes.update(row.get("attributes") or {})
                cleaned = {}
                for name, values in attributes.items():
                    if name not in ATTRIBUTE_FIELDS:
                        continue
                    if not isinstance(values, (list, tuple, set)):
                        values = [values]
                    cleaned[name] = tuple(sorted({normalized(str(v)) for v in values if v is not None and str(v).strip()}))
                yield Product(asin.strip(), tuple(text(row.get(f)) for f in TEXT_FIELDS), cleaned, _price(row.get("price")))
        return cls(convert())

    @classmethod
    def from_jsonl(cls, path):
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            return cls.from_records(json.loads(line) for line in handle if line.strip())

    def __len__(self):
        return len(self.products)

    def __contains__(self, asin):
        return asin in self.products

    def __getitem__(self, asin):
        return self.products[asin]


def constraint_status(product: Product, constraint: Constraint) -> str:
    """Conservative fact comparison for opt-in retrieval filtering, not final scoring."""
    if constraint.field == "price":
        if product.price is None:
            return "unknown"
        matched = ((constraint.minimum is None or product.price >= constraint.minimum)
                   and (constraint.maximum is None or product.price <= constraint.maximum))
    else:
        observed = product.attributes.get(constraint.field, ())
        if not observed:
            return "unknown"
        matched = bool(set(observed) & {normalized(v) for v in constraint.values})
        if constraint.negative:
            matched = not matched
    return "pass" if matched else "fail"
