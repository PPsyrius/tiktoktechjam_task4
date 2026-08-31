"""Strict JSONL/gzip loader for the frozen catalog."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path


class CatalogLoadError(ValueError):
    pass


class CatalogLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _open_text(self):
        if self.path.suffix == ".gz":
            return gzip.open(self.path, "rt", encoding="utf-8")
        return self.path.open("rt", encoding="utf-8")

    def _open_binary(self):
        if self.path.suffix == ".gz":
            return gzip.open(self.path, "rb")
        return self.path.open("rb")

    def records(self) -> Iterator[Mapping[str, object]]:
        with self._open_text() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CatalogLoadError(
                        f"invalid JSON at {self.path}:{line_number}: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise CatalogLoadError(
                        f"catalog row must be an object at {self.path}:{line_number}"
                    )
                yield record

    def source_fingerprint(self) -> str:
        digest = hashlib.sha256()
        with self._open_binary() as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
