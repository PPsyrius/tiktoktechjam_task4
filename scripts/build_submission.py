"""Build a self-contained offline submission archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EXPECTED_METRICS = {
    "hit_rate_at_10": 0.955,
    "mrr": 0.638062,
    "mttc": 2.74,
    "recommended_technical_score": 0.834119,
}
SOURCE_DIRECTORIES = (
    "docs",
    "evaluator",
    "local_experiments",
    "retrieval",
    "scripts",
    "starter",
    "tests",
)
ROOT_FILES = (
    "agent.py",
    "DATA_ATTRIBUTION.md",
    "README.md",
    "requirements.txt",
    "requirements-lock.txt",
    "ruff.toml",
    "data/catalog.jsonl",
    "data/public_set.jsonl",
    "ParticipationKit/SHA256SUMS",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(root: Path) -> list[Path]:
    files = [root / name for name in ROOT_FILES]
    for directory in SOURCE_DIRECTORIES:
        files.extend(
            path
            for path in (root / directory).rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".log"}
        )
    return sorted(set(files))


def _asset_files(root: Path) -> list[Path]:
    model_dir = root / "artifacts/models/all-MiniLM-L6-v2"
    semantic_index = root / "artifacts/retrieval/all-MiniLM-L6-v2.npz"
    if not model_dir.is_dir():
        raise SystemExit(f"Missing local semantic model: {model_dir}")
    if not semantic_index.is_file():
        raise SystemExit(f"Missing semantic index: {semantic_index}")
    return sorted(path for path in model_dir.rglob("*") if path.is_file()) + [semantic_index]


def _validate_assets(root: Path) -> dict:
    from retrieval.semantic_retriever import LocalSentenceEncoder, read_semantic_metadata
    from starter.catalog import ProductStore

    catalog = root / "data/catalog.jsonl"
    model_dir = root / "artifacts/models/all-MiniLM-L6-v2"
    semantic_index = root / "artifacts/retrieval/all-MiniLM-L6-v2.npz"
    store = ProductStore.from_jsonl(catalog, cache_dir=root / ".cache/retrieval/catalog")
    metadata = read_semantic_metadata(semantic_index)
    if metadata.get("catalog") != store.fingerprint:
        raise SystemExit("Semantic index does not match the packaged catalog; rebuild it first")
    encoder = LocalSentenceEncoder(model_dir)
    if metadata.get("encoder") != encoder.key:
        raise SystemExit("Semantic index does not match the packaged model; rebuild it first")
    return {
        "catalog_fingerprint": store.fingerprint,
        "encoder_key": encoder.key,
        "semantic_index_sha256": _sha256(semantic_index),
    }


def _write_archive(root: Path, output: Path, files: list[Path], manifest: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="techjam-submission-",
        suffix=".zip",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary_path, "w", allowZip64=True) as archive:
            for path in files:
                if not path.is_file():
                    raise SystemExit(f"Required submission file is missing: {path}")
                relative = path.relative_to(root)
                compression = (
                    zipfile.ZIP_STORED
                    if path.suffix in {".gz", ".npz", ".safetensors", ".zip"}
                    else zipfile.ZIP_DEFLATED
                )
                archive.write(path, Path("submission") / relative, compress_type=compression)
            archive.writestr(
                "submission/submission_manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                compress_type=zipfile.ZIP_DEFLATED,
            )
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the offline TechJam submission ZIP.")
    parser.add_argument("--output", default="dist/techjam-shopping-copilot-offline.zip")
    args = parser.parse_args()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = root / output

    validation = _validate_assets(root)
    files = _source_files(root) + _asset_files(root)
    manifest = {
        "agent_entrypoint": "agent.Agent",
        "evaluation_command": "python -m evaluator.local_evaluator",
        "network_required_for_inference": False,
        "python_tested": "3.12.11",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": MODEL_REVISION,
        "model_license": "Apache-2.0",
        "expected_public_metrics": EXPECTED_METRICS,
        "file_count": len(files),
        **validation,
    }
    _write_archive(root, output, files, manifest)
    print(json.dumps({
        "output": str(output),
        "size_bytes": output.stat().st_size,
        "sha256": _sha256(output),
        **manifest,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
