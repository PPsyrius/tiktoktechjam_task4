"""Content-addressed, atomic SQLite cache for canonical Product objects."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import zlib
from pathlib import Path
from .catalog_loader import CatalogLoader
from .product import Product
from .product_store import ProductStore, STORE_SCHEMA_VERSION


CACHE_SCHEMA_VERSION = "catalog-cache-v1"


class CatalogCacheError(RuntimeError):
    pass


class CatalogCache:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, source_fingerprint: str) -> Path:
        return self.directory / (
            f"{CACHE_SCHEMA_VERSION}-{STORE_SCHEMA_VERSION}-{source_fingerprint}.sqlite3"
        )

    def load_or_build(
        self,
        catalog_path: str | Path,
        *,
        rebuild: bool = False,
    ) -> ProductStore:
        loader = CatalogLoader(catalog_path)
        source_fingerprint = loader.source_fingerprint()
        cache_path = self.path_for(source_fingerprint)
        if cache_path.exists() and not rebuild:
            return self._load(cache_path, source_fingerprint)

        store = ProductStore.from_records(
            loader.records(),
            source_fingerprint=source_fingerprint,
        )
        self._write(cache_path, store)
        return ProductStore(
            store.products.values(),
            source_fingerprint=source_fingerprint,
            cache_path=cache_path,
            cache_hit=False,
        )

    def _write(self, cache_path: Path, store: ProductStore) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="catalog-build-",
            suffix=".sqlite3",
            dir=self.directory,
        )
        os.close(descriptor)
        connection = None
        try:
            connection = sqlite3.connect(temporary_name)
            connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE products(position INTEGER PRIMARY KEY, payload BLOB NOT NULL)")
            connection.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                (
                    ("cache_schema", CACHE_SCHEMA_VERSION),
                    ("store_schema", STORE_SCHEMA_VERSION),
                    ("source_fingerprint", store.source_fingerprint or ""),
                    ("store_fingerprint", store.fingerprint),
                    ("product_count", str(len(store))),
                ),
            )
            connection.executemany(
                "INSERT INTO products VALUES (?, ?)",
                (
                    (
                        position,
                        sqlite3.Binary(zlib.compress(json.dumps(
                            product.to_dict(include_raw=True),
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8"), level=1)),
                    )
                    for position, product in enumerate(store.products.values())
                ),
            )
            connection.commit()
            connection.close()
            connection = None
            os.replace(temporary_name, cache_path)
        finally:
            if connection is not None:
                connection.close()
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()

    def _load(self, cache_path: Path, source_fingerprint: str) -> ProductStore:
        try:
            connection = sqlite3.connect(cache_path.as_uri() + "?mode=ro", uri=True)
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata.get("cache_schema") != CACHE_SCHEMA_VERSION:
                raise CatalogCacheError("catalog cache schema mismatch; rebuild explicitly")
            if metadata.get("store_schema") != STORE_SCHEMA_VERSION:
                raise CatalogCacheError("product store schema mismatch; rebuild explicitly")
            if metadata.get("source_fingerprint") != source_fingerprint:
                raise CatalogCacheError("catalog cache source fingerprint mismatch")
            products = [
                Product.from_dict(json.loads(zlib.decompress(payload).decode("utf-8")))
                for (payload,) in connection.execute(
                    "SELECT payload FROM products ORDER BY position"
                )
            ]
            connection.close()
        except CatalogCacheError:
            raise
        except (OSError, sqlite3.DatabaseError, ValueError, zlib.error, json.JSONDecodeError) as error:
            raise CatalogCacheError(
                f"invalid catalog cache {cache_path}; rebuild it explicitly"
            ) from error
        finally:
            if "connection" in locals():
                connection.close()

        store = ProductStore(
            products,
            source_fingerprint=source_fingerprint,
            cache_path=cache_path,
            cache_hit=True,
        )
        if metadata.get("product_count") != str(len(store)):
            raise CatalogCacheError("catalog cache product count mismatch")
        if metadata.get("store_fingerprint") != store.fingerprint:
            raise CatalogCacheError("catalog cache product fingerprint mismatch")
        return store
