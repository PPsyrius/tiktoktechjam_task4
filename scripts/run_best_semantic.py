"""Reproduce the verified MiniLM semantic-retrieval evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MODEL_REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILES = (
    "1_Pooling/config.json",
    "README.md",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
MODEL_REQUIRED_FILES = ("model.safetensors", "modules.json", "tokenizer.json")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _download_model(model_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install dependencies first: python -m pip install -r requirements.txt") from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPOSITORY,
        revision=MODEL_REVISION,
        local_dir=model_dir,
        allow_patterns=list(MODEL_FILES),
    )


def _model_is_ready(model_dir: Path) -> bool:
    return model_dir.is_dir() and all((model_dir / name).is_file() for name in MODEL_REQUIRED_FILES)


def _run(root: Path, *arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=root, check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build the pinned MiniLM index and run the verified 200-session evaluation."
    )
    parser.add_argument("--catalog", default="ParticipationKit/catalog.jsonl.gz")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--model-dir", default="artifacts/models/all-MiniLM-L6-v2")
    parser.add_argument("--semantic-index", default="artifacts/retrieval/all-MiniLM-L6-v2.npz")
    parser.add_argument("--retrieval-output", default="results/semantic-best-retrieval.json")
    parser.add_argument("--official-output", default="results/semantic-best-official.json")
    parser.add_argument(
        "--download-model",
        action="store_true",
        help=f"Download pinned {MODEL_REPOSITORY} files when the local model is absent.",
    )
    parser.add_argument(
        "--reuse-index",
        action="store_true",
        help="Reuse an existing compatible index instead of rebuilding it.",
    )
    args = parser.parse_args()

    catalog = _resolve(root, args.catalog)
    dataset = _resolve(root, args.dataset)
    model_dir = _resolve(root, args.model_dir)
    semantic_index = _resolve(root, args.semantic_index)
    retrieval_output = _resolve(root, args.retrieval_output)
    official_output = _resolve(root, args.official_output)

    if not catalog.is_file():
        parser.error(f"catalog not found: {catalog}")
    if not dataset.is_file():
        parser.error(f"public dataset not found: {dataset}")
    if not _model_is_ready(model_dir):
        if args.download_model:
            _download_model(model_dir)
        else:
            parser.error(
                f"model not found: {model_dir}\n"
                "Run this command again with --download-model, or place the model at that path."
            )
    if not _model_is_ready(model_dir):
        parser.error(f"model download is incomplete: {model_dir}")

    semantic_index.parent.mkdir(parents=True, exist_ok=True)
    retrieval_output.parent.mkdir(parents=True, exist_ok=True)
    official_output.parent.mkdir(parents=True, exist_ok=True)

    if not args.reuse_index:
        _run(
            root,
            "-m",
            "retrieval.build_index",
            "--catalog",
            str(catalog),
            "--model-dir",
            str(model_dir),
            "--semantic-output",
            str(semantic_index),
            "--rebuild-cache",
        )
    elif not semantic_index.is_file():
        parser.error(f"semantic index not found: {semantic_index}; remove --reuse-index to build it")

    _run(
        root,
        "-m",
        "scripts.evaluate_retrieval",
        "--catalog",
        str(catalog),
        "--dataset",
        str(dataset),
        "--model-dir",
        str(model_dir),
        "--semantic-index",
        str(semantic_index),
        "--semantic-candidate-limit",
        "40",
        "--semantic-weight",
        "0.3",
        "--dynamic-semantic-gate",
        "--semantic-min-lexical-fill",
        "0.75",
        "--semantic-shadow-min-overlap",
        "2",
        "--semantic-shadow-lexical-window",
        "160",
        "--output",
        str(retrieval_output),
        "--official-output",
        str(official_output),
    )

    retrieval_result = json.loads(retrieval_output.read_text(encoding="utf-8"))
    startup_error = retrieval_result.get("semantic_startup_error")
    if startup_error:
        raise SystemExit(f"Semantic evaluation degraded and is not reproducible: {startup_error}")

    official_result = json.loads(official_output.read_text(encoding="utf-8"))
    print(
        "Verified semantic result: "
        f"Hit@10={official_result['hit_rate_at_10']}, "
        f"MRR={official_result['mrr']}, "
        f"MTTC={official_result['mttc']}, "
        f"TechnicalScore={official_result['recommended_technical_score']}"
    )


if __name__ == "__main__":
    main()
