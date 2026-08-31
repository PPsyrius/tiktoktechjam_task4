"""Read-only product collection and deterministic catalog indexes."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from types import MappingProxyType
from typing import Optional
from collections.abc import Iterable, Mapping

from .catalog_loader import CatalogLoader
from .feature_extractor import FeatureExtractor, normalize_text
from .product import Product


STORE_SCHEMA_VERSION = "task1-product-store-v3"


class ProductStore:
    def __init__(
        self,
        products: Iterable[Product],
        *,
        source_fingerprint: str | None = None,
        cache_path: Path | None = None,
        cache_hit: bool = False,
    ) -> None:
        product_map: dict[str, Product] = {}
        category_index: dict[str, list[str]] = defaultdict(list)
        brand_index: dict[str, list[str]] = defaultdict(list)
        digest = hashlib.sha256(f"{STORE_SCHEMA_VERSION}\n".encode())
        for product in products:
            if product.parent_asin in product_map:
                raise ValueError("Missing or duplicate parent_asin: " + product.parent_asin)
            product_map[product.parent_asin] = product
            digest.update(json.dumps(
                product.to_dict(include_raw=False),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8"))
            digest.update(b"\n")
            category_keys = {
                normalize_text(value) for value in product.category_path if value
            }
            if product.category_path:
                category_keys.add(normalize_text(" > ".join(product.category_path)))
            for key in sorted(category_keys):
                category_index[key].append(product.parent_asin)
            if product.brand:
                brand_index[normalize_text(product.brand)].append(product.parent_asin)

        self.fingerprint = digest.hexdigest()
        self.source_fingerprint = source_fingerprint
        self.cache_path = cache_path
        self.cache_hit = cache_hit
        self.products = MappingProxyType(product_map)
        self.category_index = MappingProxyType({
            key: tuple(asins) for key, asins in category_index.items()
        })
        self.brand_index = MappingProxyType({
            key: tuple(asins) for key, asins in brand_index.items()
        })

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, object] | Product],
        *,
        extractor: FeatureExtractor | None = None,
        source_fingerprint: str | None = None,
    ) -> ProductStore:
        active_extractor = extractor or FeatureExtractor()

        def convert() -> Iterable[Product]:
            for record in records:
                yield record if isinstance(record, Product) else active_extractor.extract(record)

        return cls(convert(), source_fingerprint=source_fingerprint)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        *,
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
    ) -> ProductStore:
        if cache_dir is not None:
            from .catalog_cache import CatalogCache
            return CatalogCache(cache_dir).load_or_build(
                path,
                rebuild=rebuild_cache,
            )
        loader = CatalogLoader(path)
        return cls.from_records(
            loader.records(),
            source_fingerprint=loader.source_fingerprint(),
        )

    def get(self, parent_asin: str) -> Product | None:
        return self.products.get(parent_asin)

    def get_all(self) -> tuple[Product, ...]:
        return tuple(self.products.values())

    def get_by_category(self, category: str) -> tuple[Product, ...]:
        return tuple(
            self.products[asin]
            for asin in self.category_index.get(normalize_text(category), ())
        )

    def get_by_brand(self, brand: str) -> tuple[Product, ...]:
        return tuple(
            self.products[asin]
            for asin in self.brand_index.get(normalize_text(brand), ())
        )

    def __len__(self) -> int:
        return len(self.products)

    def __contains__(self, parent_asin: object) -> bool:
        return parent_asin in self.products

    def __getitem__(self, parent_asin: str) -> Product:
        return self.products[parent_asin]

    def __iter__(self):
        return iter(self.products.values())
