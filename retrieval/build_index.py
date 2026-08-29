"""python -m retrieval.build_index --catalog ... [--model-dir ... --semantic-output ...]"""
from __future__ import annotations

import argparse
import json
import time
from .bm25_retriever import BM25Retriever
from .product_store import ProductStore


def main():
    parser = argparse.ArgumentParser(description="Build local task 4 retrieval assets")
    parser.add_argument("--catalog", default="ParticipationKit/catalog.jsonl.gz")
    parser.add_argument("--cache-dir", default=".cache/retrieval")
    parser.add_argument("--model-dir")
    parser.add_argument("--semantic-output")
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--document-prefix", default="")
    args = parser.parse_args()
    if bool(args.model_dir) != bool(args.semantic_output):
        parser.error("--model-dir and --semantic-output must be supplied together")
    started = time.perf_counter()
    store = ProductStore.from_jsonl(args.catalog)
    bm25 = BM25Retriever(store, cache_dir=args.cache_dir)
    result = {"catalog_size": len(store), "catalog_fingerprint": store.fingerprint,
              "fts5_cache": str(bm25.cache_path), "cache_hit": bm25.cache_hit}
    bm25.close()
    if args.model_dir:
        from .semantic_retriever import LocalSentenceEncoder, SemanticRetriever
        encoder = LocalSentenceEncoder(args.model_dir, args.query_prefix, args.document_prefix)
        SemanticRetriever.build(store, encoder, args.semantic_output)
        result["semantic_asset"] = args.semantic_output
        result["encoder_key"] = encoder.key
    result["initialization_seconds"] = time.perf_counter() - started
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
