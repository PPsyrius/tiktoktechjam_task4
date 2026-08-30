from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from starter.catalog import (
    CatalogCacheError,
    CatalogLoadError,
    CatalogLoader,
    FeatureExtractor,
    ProductStore,
)
from starter.catalog.feature_extractor import parse_price
from scripts.profile_catalog import profile


def sample_record(parent_asin: str = "A") -> dict:
    return {
        "parent_asin": parent_asin,
        "title": "BLACK Cotton Running Shirt",
        "categories": [
            "Clothing, Shoes & Jewelry",
            "Men",
            "Clothing",
            "Shirts",
            "T-Shirts",
        ],
        "features": ["Lightweight", "Machine Wash", "Breathable mesh"],
        "description": ["Comfortable for running and gym workouts."],
        "details": {
            "Brand": "NIKE",
            "Color": "Black / White",
            "Material": "Cotton Blend",
            "Size": "Large",
            "Style": "Athletic",
        },
        "store": "Nike Outlet",
        "price": "$1,234.50",
        "average_rating": 4.7,
        "rating_number": 8000,
    }


class FeatureExtractorTest(unittest.TestCase):
    def test_normalizes_schema_and_preserves_raw_record(self) -> None:
        product = FeatureExtractor().extract(sample_record())

        self.assertEqual(product.parent_asin, "A")
        self.assertEqual(product.category, "T-Shirts")
        self.assertEqual(product.category_path[-2:], ("Shirts", "T-Shirts"))
        self.assertEqual(product.brand, "Nike")
        self.assertEqual(product.price, 1234.5)
        self.assertIn("black", product.color)
        self.assertIn("cotton", product.material)
        self.assertEqual(product.size, ("large",))
        self.assertEqual(product.style, ("athletic",))
        self.assertIn("machine wash", product.feature)
        self.assertIn("running", product.use_case)
        self.assertEqual(product.average_rating, 4.7)
        self.assertEqual(product.rating_number, 8000)
        self.assertIn("T-Shirts", product.search_text)
        self.assertIn("Nike", product.search_text)
        self.assertIn("Color Black / White", product.search_text)
        self.assertEqual(product.raw["title"], "BLACK Cotton Running Shirt")
        with self.assertRaises(TypeError):
            product.raw["title"] = "changed"

    def test_store_name_is_not_inferred_as_brand(self) -> None:
        product = FeatureExtractor().extract({
            "parent_asin": "A",
            "title": "Shoe",
            "store": "Nike",
        })
        self.assertIsNone(product.brand)
        self.assertNotIn("brand", product.attributes)

    def test_price_formats_are_deterministic(self) -> None:
        self.assertEqual(parse_price("$29.99"), 29.99)
        self.assertEqual(parse_price("from 12.99"), 12.99)
        self.assertEqual(parse_price("1,234.50"), 1234.5)
        for value in ("—", "10-20", True, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIsNone(parse_price(value))

    def test_extraction_is_repeatable(self) -> None:
        extractor = FeatureExtractor()
        first = extractor.extract(sample_record()).to_dict()
        second = extractor.extract(sample_record()).to_dict()
        self.assertEqual(first, second)

    def test_invalid_mapping_fields_and_nonfinite_rating_count_fail_safely(self) -> None:
        for field, value in (("details", 0), ("attributes", False)):
            with self.subTest(field=field):
                record = sample_record()
                record[field] = value
                with self.assertRaises(TypeError):
                    FeatureExtractor().extract(record)

        record = sample_record()
        record["rating_number"] = float("inf")
        self.assertEqual(FeatureExtractor().extract(record).rating_number, 0)


class CatalogLoaderTest(unittest.TestCase):
    def test_plain_and_gzip_content_have_the_same_source_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plain = Path(directory) / "catalog.jsonl"
            zipped = Path(directory) / "catalog.jsonl.gz"
            content = json.dumps(sample_record()) + "\n"
            plain.write_text(content, encoding="utf-8")
            with gzip.open(zipped, "wt", encoding="utf-8") as handle:
                handle.write(content)

            self.assertEqual(
                CatalogLoader(plain).source_fingerprint(),
                CatalogLoader(zipped).source_fingerprint(),
            )
            self.assertEqual(
                ProductStore.from_jsonl(plain).fingerprint,
                ProductStore.from_jsonl(zipped).fingerprint,
            )

    def test_invalid_json_and_non_object_rows_fail_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text("not-json\n", encoding="utf-8")
            with self.assertRaises(CatalogLoadError):
                list(CatalogLoader(path).records())
            path.write_text("[]\n", encoding="utf-8")
            with self.assertRaises(CatalogLoadError):
                list(CatalogLoader(path).records())

    def test_profiler_reports_duplicates_and_field_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.jsonl"
            path.write_text(
                json.dumps(sample_record("A")) + "\n"
                + json.dumps(sample_record("A")) + "\n",
                encoding="utf-8",
            )
            result = profile(path)

        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["unique_parent_asin"], 1)
        self.assertEqual(result["duplicate_parent_asin_rows"], 1)
        self.assertEqual(result["fields"]["details"]["types"], {"dict": 2})
        self.assertEqual(result["price"]["parseable"], 2)


class ProductStoreTest(unittest.TestCase):
    def test_indexes_and_mapping_are_read_only(self) -> None:
        first = sample_record("A")
        second = sample_record("B")
        second["details"] = {**second["details"], "Brand": "ADIDAS"}
        store = ProductStore.from_records((first, second))

        self.assertIs(store.get("A"), store["A"])
        self.assertEqual(len(store.get_all()), 2)
        self.assertEqual(
            {product.parent_asin for product in store.get_by_category("T-Shirts")},
            {"A", "B"},
        )
        self.assertEqual(
            [product.parent_asin for product in store.get_by_brand("nike")],
            ["A"],
        )
        with self.assertRaises(TypeError):
            store.products["C"] = store["A"]
        with self.assertRaises(TypeError):
            store.category_index["new"] = ("A",)

    def test_duplicate_parent_asin_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate parent_asin"):
            ProductStore.from_records((sample_record("A"), sample_record("A")))


class CatalogCacheTest(unittest.TestCase):
    def test_cache_round_trip_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            cache = root / "cache"
            catalog.write_text(json.dumps(sample_record()) + "\n", encoding="utf-8")

            first = ProductStore.from_jsonl(catalog, cache_dir=cache)
            second = ProductStore.from_jsonl(catalog, cache_dir=cache)

            self.assertFalse(first.cache_hit)
            self.assertTrue(second.cache_hit)
            self.assertEqual(
                first.source_fingerprint,
                CatalogLoader(catalog).source_fingerprint(),
            )
            self.assertEqual(first.cache_path, second.cache_path)
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.assertEqual(second["A"].raw["details"]["Brand"], "NIKE")
            self.assertEqual(list(cache.glob("catalog-build-*")), [])

            changed = sample_record()
            changed["title"] = "Changed title"
            catalog.write_text(json.dumps(changed) + "\n", encoding="utf-8")
            third = ProductStore.from_jsonl(catalog, cache_dir=cache)
            self.assertNotEqual(third.cache_path, first.cache_path)

    def test_corrupt_cache_is_not_a_silent_miss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            cache = root / "cache"
            catalog.write_text(json.dumps(sample_record()) + "\n", encoding="utf-8")
            first = ProductStore.from_jsonl(catalog, cache_dir=cache)
            first.cache_path.write_bytes(b"broken")

            with self.assertRaises(CatalogCacheError):
                ProductStore.from_jsonl(catalog, cache_dir=cache)

            rebuilt = ProductStore.from_jsonl(catalog, cache_dir=cache, rebuild_cache=True)
            self.assertFalse(rebuilt.cache_hit)
            self.assertEqual(rebuilt["A"].parent_asin, "A")


if __name__ == "__main__":
    unittest.main()
