"""FTS5 retrieval, preserving baseline tokenization and default column weights."""
from __future__ import annotations

import math
import os
import re
import sqlite3
import tempfile
from pathlib import Path

from .merge import interleave
from .types import Candidate, SourceHit

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}
DEFAULT_WEIGHTS = (0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)
CACHE_SCHEMA_VERSION = "fts5-v2"


class BM25CacheError(RuntimeError):
    pass


def terms(query):
    return list(dict.fromkeys(t.lower() for t in TOKEN_RE.findall(query)
                              if len(t) > 1 and t.lower() not in STOPWORDS))[:40]


def lexical_queries(context):
    queries = [q.strip() for q in context.queries if q.strip()]
    if not queries:
        query = " ".join(v for c in context.constraints if not c.negative for v in c.values)
        if query:
            queries.append(query)
    return list(dict.fromkeys(queries))[:8]


class BM25Retriever:
    def __init__(
        self,
        store,
        cache_dir=None,
        weights=DEFAULT_WEIGHTS,
        rebuild_cache=False,
    ):
        if len(weights) != 7 or any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("FTS5 requires seven finite non-negative column weights")
        self.weights = tuple(weights)
        self.cache_hit = False
        self.cache_path = None
        if cache_dir is None:
            self.connection = sqlite3.connect(":memory:")
            self._build(self.connection, store)
        else:
            directory = Path(cache_dir).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self.cache_path = directory / (
                CACHE_SCHEMA_VERSION + "-" + store.fingerprint + ".sqlite3"
            )
            self.cache_hit = self.cache_path.exists() and not rebuild_cache
            if not self.cache_hit:
                fd, name = tempfile.mkstemp(prefix="fts5-build-", suffix=".sqlite3", dir=directory)
                os.close(fd)
                connection = None
                try:
                    connection = sqlite3.connect(name)
                    self._build(connection, store)
                    connection.close()
                    connection = None
                    os.replace(name, self.cache_path)
                finally:
                    if connection is not None:
                        connection.close()
                    if Path(name).exists():
                        Path(name).unlink()
            self.connection = self._open_cache(self.cache_path, store)

    @staticmethod
    def _build(connection, store):
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            (
                ("cache_schema", CACHE_SCHEMA_VERSION),
                ("catalog_fingerprint", store.fingerprint),
                ("product_count", str(len(store))),
            ),
        )
        connection.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
                               ((p.parent_asin, *p.fields) for p in store.products.values()))
        connection.commit()

    @staticmethod
    def _open_cache(path, store):
        connection = None
        try:
            connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if metadata != {
                "cache_schema": CACHE_SCHEMA_VERSION,
                "catalog_fingerprint": store.fingerprint,
                "product_count": str(len(store)),
            }:
                raise BM25CacheError("FTS5 cache metadata mismatch; rebuild explicitly")
            if connection.execute("PRAGMA quick_check(1)").fetchone() != ("ok",):
                raise BM25CacheError("FTS5 cache integrity check failed; rebuild explicitly")
            count = connection.execute("SELECT count(*) FROM products").fetchone()[0]
            if count != len(store):
                raise BM25CacheError("FTS5 cache product count mismatch; rebuild explicitly")
            return connection
        except BM25CacheError:
            if connection is not None:
                connection.close()
            raise
        except (OSError, sqlite3.DatabaseError) as error:
            if connection is not None:
                connection.close()
            raise BM25CacheError(
                f"invalid FTS5 cache {path}; rebuild it explicitly"
            ) from error

    def search(self, context, limit):
        if limit <= 0:
            return []
        routes = []
        placeholders = ", ".join("?" for _ in self.weights)
        for query in lexical_queries(context):
            expression = " OR ".join('"' + term + '"' for term in terms(query))
            if not expression:
                continue
            rows = self.connection.execute(
                f"SELECT parent_asin, bm25(products, {placeholders}) AS score "
                "FROM products WHERE products MATCH ? ORDER BY score, rowid LIMIT ?",
                (*self.weights, expression, limit),
            ).fetchall()
            routes.append([Candidate(str(asin), (SourceHit("bm25", rank, score, query, False),))
                           for rank, (asin, score) in enumerate(rows, 1)])
        return interleave(routes, limit)

    def close(self):
        self.connection.close()
