"""Optional offline dense search. No downloads or model imports on the default path."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

from .bm25_retriever import lexical_queries
from .types import Candidate, SourceHit


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Dense retrieval requires the optional numpy dependency") from exc
    return np


class LocalSentenceEncoder:
    """Use only an existing local SentenceTransformer directory, never a Hub ID."""
    def __init__(self, model_path, query_prefix="", document_prefix=""):
        path = Path(model_path)
        if not path.is_dir():
            raise ValueError("model_path must be an existing local model directory")
        from sentence_transformers import SentenceTransformer
        digest = hashlib.sha256()
        for file in sorted(p for p in path.rglob("*") if p.is_file() and not any(
                part.startswith(".") for part in p.relative_to(path).parts)):
            digest.update(str(file.relative_to(path)).encode())
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        digest.update(json.dumps([query_prefix, document_prefix]).encode())
        self.key = "sentence-transformer:" + digest.hexdigest()
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.model = SentenceTransformer(str(path.resolve()), device="cpu",
                                         local_files_only=True, trust_remote_code=False)

    def encode_documents(self, texts):
        return self.model.encode([self.document_prefix + t for t in texts],
                                 convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False, batch_size=32)

    def encode_queries(self, texts):
        return self.model.encode([self.query_prefix + t for t in texts],
                                 convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)


class SemanticRetriever:
    def __init__(self, store, encoder, ids, vectors):
        np = _numpy()
        self.encoder = encoder
        self.ids = tuple(str(asin) for asin in ids)
        if len(set(self.ids)) != len(self.ids) or set(self.ids) != set(store.products):
            raise ValueError("Dense index IDs must cover the catalog exactly, without duplicates")
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.ids) or matrix.shape[1] == 0:
            raise ValueError("Dense matrix dimensions do not match catalog IDs")
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite dense vectors")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms == 0).any() or not np.isfinite(norms).all():
            raise ValueError("Invalid document vector norm")
        self.vectors = matrix / norms
        self._id_array = np.asarray(self.ids)

    @classmethod
    def build(cls, store, encoder, output, batch_size=128):
        """Explicit offline asset creation; inference never embeds the whole catalog."""
        np = _numpy()
        if batch_size < 1 or not len(store):
            raise ValueError("Building embeddings requires a nonempty store and positive batch_size")
        ids = tuple(store.products)
        blocks = []
        for start in range(0, len(ids), batch_size):
            batch = ids[start:start + batch_size]
            vectors = np.asarray(encoder.encode_documents([store[asin].search_text for asin in batch]))
            if vectors.ndim != 2 or vectors.shape[0] != len(batch):
                raise ValueError("Encoder returned the wrong document batch shape")
            blocks.append(vectors)
        retriever = cls(store, encoder, ids, np.concatenate(blocks, axis=0))
        metadata = json.dumps({"version": 1, "catalog": store.fingerprint, "encoder": encoder.key})
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="dense-build-", suffix=".npz", dir=output.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, ids=np.asarray(ids), vectors=retriever.vectors,
                                    metadata=np.asarray(metadata))
            os.replace(name, output)
        finally:
            if Path(name).exists():
                Path(name).unlink()
        return retriever

    @classmethod
    def load(cls, store, encoder, path):
        np = _numpy()
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            expected = {"version": 1, "catalog": store.fingerprint, "encoder": encoder.key}
            if metadata != expected:
                raise ValueError("Dense index catalog/model fingerprint mismatch; rebuild the asset")
            return cls(store, encoder, data["ids"], data["vectors"])

    def search(self, context, limit):
        np = _numpy()
        query = context.semantic_query.strip() or " ".join(lexical_queries(context))
        if not query or limit <= 0:
            return []
        vectors = np.asarray(self.encoder.encode_queries([query]), dtype=np.float32)
        if vectors.shape != (1, self.vectors.shape[1]) or not np.isfinite(vectors).all():
            raise ValueError("Invalid query embedding")
        norm = np.linalg.norm(vectors[0])
        if not np.isfinite(norm):
            raise ValueError("Invalid query vector norm")
        if norm == 0:
            return []
        scores = self.vectors @ (vectors[0] / norm)
        # Stable ID tie-break; no approximate-search service is required for 50k items.
        indices = np.lexsort((self._id_array, -scores))[:limit]
        return [Candidate(self.ids[i], (SourceHit("semantic", rank, float(scores[i]), query),))
                for rank, i in enumerate(indices, 1)]
