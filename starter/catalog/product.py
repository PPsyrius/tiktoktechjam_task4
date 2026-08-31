"""Immutable canonical product object for the frozen competition catalog."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Optional
from collections.abc import Mapping


TEXT_FIELDS = (
    "title",
    "categories",
    "features",
    "details",
    "store",
    "description",
)
ATTRIBUTE_FIELDS = frozenset({
    "category",
    "brand",
    "material",
    "color",
    "size",
    "style",
    "use_case",
    "feature",
})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Product:
    parent_asin: str
    title: str
    category: str | None
    category_path: tuple[str, ...]
    brand: str | None
    price: float | None
    color: tuple[str, ...]
    size: tuple[str, ...]
    material: tuple[str, ...]
    style: tuple[str, ...]
    use_case: tuple[str, ...]
    feature: tuple[str, ...]
    features: tuple[str, ...]
    description: str
    details: Mapping[str, Any]
    store: str | None
    average_rating: float | None
    rating_number: int
    search_text: str
    fields: tuple[str, ...]
    attributes: Mapping[str, tuple[str, ...]]
    raw: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.parent_asin, str) or not self.parent_asin.strip():
            raise ValueError("parent_asin must be a nonempty string")
        if len(self.fields) != len(TEXT_FIELDS):
            raise ValueError("fields must follow the six-field retrieval contract")
        object.__setattr__(self, "parent_asin", self.parent_asin.strip())
        for name in (
            "category_path", "color", "size", "material", "style", "use_case",
            "feature", "features", "fields",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "details", _freeze(self.details))
        object.__setattr__(self, "attributes", MappingProxyType({
            key: tuple(values) for key, values in self.attributes.items()
        }))
        object.__setattr__(self, "raw", _freeze(self.raw))

    def to_dict(self, *, include_raw: bool = True) -> dict[str, Any]:
        payload = {
            "parent_asin": self.parent_asin,
            "title": self.title,
            "category": self.category,
            "category_path": list(self.category_path),
            "brand": self.brand,
            "price": self.price,
            "color": list(self.color),
            "size": list(self.size),
            "material": list(self.material),
            "style": list(self.style),
            "use_case": list(self.use_case),
            "feature": list(self.feature),
            "features": list(self.features),
            "description": self.description,
            "details": _thaw(self.details),
            "store": self.store,
            "average_rating": self.average_rating,
            "rating_number": self.rating_number,
            "search_text": self.search_text,
            "fields": list(self.fields),
            "attributes": {key: list(values) for key, values in self.attributes.items()},
        }
        if include_raw:
            payload["raw"] = _thaw(self.raw)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Product:
        return cls(
            parent_asin=str(payload["parent_asin"]),
            title=str(payload.get("title") or ""),
            category=payload.get("category"),
            category_path=tuple(payload.get("category_path") or ()),
            brand=payload.get("brand"),
            price=payload.get("price"),
            color=tuple(payload.get("color") or ()),
            size=tuple(payload.get("size") or ()),
            material=tuple(payload.get("material") or ()),
            style=tuple(payload.get("style") or ()),
            use_case=tuple(payload.get("use_case") or ()),
            feature=tuple(payload.get("feature") or ()),
            features=tuple(payload.get("features") or ()),
            description=str(payload.get("description") or ""),
            details=payload.get("details") or {},
            store=payload.get("store"),
            average_rating=payload.get("average_rating"),
            rating_number=int(payload.get("rating_number") or 0),
            search_text=str(payload.get("search_text") or ""),
            fields=tuple(payload.get("fields") or ()),
            attributes={
                str(key): tuple(values)
                for key, values in (payload.get("attributes") or {}).items()
            },
            raw=payload.get("raw") or {},
        )
