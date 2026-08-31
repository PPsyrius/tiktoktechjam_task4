"""Task 4 compatibility surface over the canonical Part 1 catalog layer."""
from __future__ import annotations

from starter.catalog.feature_extractor import (
    normalize_text as normalized,
)
from starter.catalog.feature_extractor import (
    value_to_text as text,
)
from starter.catalog.product import TEXT_FIELDS, Product
from starter.catalog.product_store import ProductStore

from .types import Constraint


def constraint_status(product: Product, constraint: Constraint) -> str:
    """Conservative fact comparison for opt-in retrieval filtering, not final scoring."""
    if constraint.field == "price":
        if product.price is None:
            return "unknown"
        matched = (
            (constraint.minimum is None or product.price >= constraint.minimum)
            and (constraint.maximum is None or product.price <= constraint.maximum)
        )
    else:
        observed = product.attributes.get(constraint.field, ())
        if not observed:
            return "unknown"
        matched = bool(set(observed) & {normalized(value) for value in constraint.values})
        if constraint.negative:
            matched = not matched
    return "pass" if matched else "fail"


__all__ = [
    "Product",
    "ProductStore",
    "TEXT_FIELDS",
    "constraint_status",
    "normalized",
    "text",
]
