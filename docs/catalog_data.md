# Part 1: Catalog data layer

Part 1 turns the participant-visible catalog into one deterministic, read-only
product model shared by retrieval and reranking:

```text
catalog.jsonl[.gz]
        |
   CatalogLoader
        |
 FeatureExtractor
        |
      Product
        |
 CatalogCache -> ProductStore -> Retrieval / Rerank
```

`starter/catalog/` is the canonical implementation. `retrieval.ProductStore`
is only a compatibility re-export; retrievers do not parse JSONL independently.

## Public catalog profile

The following measurements come from `data/catalog.jsonl` and can be reproduced
with:

```bash
python3 -m scripts.profile_catalog \
  --catalog data/catalog.jsonl \
  --output /tmp/catalog-profile.json
```

| Field | Empty rows | Empty rate | Observed type |
|---|---:|---:|---|
| `parent_asin` | 0 | 0% | string |
| `title` | 2 | 0.004% | string |
| `categories` | 0 | 0% | list |
| `features` | 5,219 | 10.438% | list |
| `description` | 23,887 | 47.774% | list |
| `details` | 1,670 | 3.340% | object |
| `store` | 314 | 0.628% | string/null |
| `price` | 39,473 | 78.946% | float/string/null |
| `average_rating` | 0 | 0% | float |
| `rating_number` | 0 | 0% | integer |

- 50,000 rows, 50,000 unique `parent_asin` values, no duplicate rows by ID.
- Category paths contain 2 to 8 levels; 5 levels is most common (23,544 rows).
- 10,527 prices are nonempty. 10,415 normalize to non-negative floats; the
  112 rejected values are the em dash placeholder (`—`).
- Frequent useful `details` keys include `Color` (2,439), `Brand` (2,328),
  `Material` (2,069), `Style` (1,752), and `Size` (925).

The profiler only reads participant-visible fields. It does not use evaluator
labels or ground truth.

## Canonical Product contract

```python
from starter.catalog import ProductStore

store = ProductStore.from_jsonl(
    "data/catalog.jsonl",
    cache_dir=".cache/retrieval/catalog",
)
product = store["B001..."]

print(product.category_path)
print(product.price)
print(product.attributes)
print(product.search_text)
print(product.raw)
```

Each immutable `Product` contains the official ID and raw record, normalized
category path, explicit and conservatively extracted attributes, rating data,
the six stable BM25 fields, and deterministic `search_text`. `ProductStore`
provides ID, category, and brand lookups. Its product map and indexes are
read-only after construction.

## Normalization rules

- IDs are stripped and must be nonempty; duplicates fail loudly.
- Price accepts non-negative numbers, `$1,234.50`, and `from 12.99`. Ranges,
  placeholders, booleans, negative numbers, NaN, and infinity become unknown.
- Category order is preserved as the official path; the final node is the
  canonical leaf category. A string path separated by `>` is also supported.
- Brand comes only from an explicit `brand`, `details.Brand`, or
  `details.Brand Name`; `store` is never guessed to be the brand.
- `details` keys are mapped to brand, color, material, size, style, feature,
  and use case. A small versioned lexicon also enriches the corresponding
  `Product` fields and `search_text` from visible text. Only explicit catalog
  facts enter `product.attributes`, so structured filters do not treat a
  text-derived guess as a verified fact.
- Unknown values stay unknown. Catalog errors and corrupt caches are not hidden
  by a silent fallback.
- `search_text` is built in a fixed order from title, category path, features,
  details, store, description, and canonical structured terms. Identical input
  produces an identical store fingerprint.

## Cache and index build

```bash
python3 -m retrieval.build_index \
  --catalog data/catalog.jsonl \
  --cache-dir .cache/retrieval
```

The catalog cache key contains the pipeline schema version and the SHA-256 of
the decompressed JSONL, so equivalent `.jsonl` and `.jsonl.gz` sources share an
identity while an extractor/schema change cannot reuse stale products. It is
written atomically as SQLite and reopened read-only. The builder reports
catalog-cache and FTS5-cache hits separately. If a cache is corrupt, delete that
exact derived file or call `ProductStore.from_jsonl(..., rebuild_cache=True)`
explicitly. The command-line equivalent for both derived caches is
`python3 -m retrieval.build_index ... --rebuild-cache`.
