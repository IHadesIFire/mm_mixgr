"""Cache helpers for query and corpus embeddings."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def slugify(text: str) -> str:
    text = text.lower().replace("/", "-")
    text = re.sub(r"[^a-z0-9._-]+", "-", text)
    return text.strip("-")


def make_run_id(
    *,
    task_name: str,
    model_name: str,
    split: str,
    domains: list[str] | None,
    max_corpus: int | None,
    max_queries: int | None,
    max_length: int,
    max_image_tokens: int,
) -> str:
    payload = {
        "task": task_name,
        "model": model_name,
        "split": split,
        "domains": sorted(domains or []),
        "max_corpus": max_corpus,
        "max_queries": max_queries,
        "max_length": max_length,
        "max_image_tokens": max_image_tokens,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    model_slug = slugify(model_name.split("/")[-1])
    domain_slug = "all" if not domains else "-".join(sorted(d.lower() for d in domains))
    corpus_slug = f"c{max_corpus}" if max_corpus is not None else "call"
    query_slug = f"q{max_queries}" if max_queries is not None else "qall"
    return f"{task_name}_{model_slug}_{domain_slug}_{corpus_slug}_{query_slug}_{digest}"


def get_run_dirs(cache_root: str | Path, results_root: str | Path, run_id: str) -> dict[str, Path]:
    cache_dir = Path(cache_root) / run_id
    results_dir = Path(results_root) / run_id
    checkpoint_dir = cache_dir / "checkpoints"
    for path in [cache_dir, results_dir, checkpoint_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return {
        "cache_dir": cache_dir,
        "results_dir": results_dir,
        "query_cache": cache_dir / "queries.npz",
        "corpus_cache": cache_dir / "corpus.npz",
        "checkpoint_dir": checkpoint_dir,
        "predictions": results_dir / "predictions.json",
        "metrics": results_dir / "metrics.json",
        "run_config": results_dir / "run_config.json",
    }


def load_embedding_cache(path: str | Path):
    path = Path(path)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return data["ids"].tolist(), data["embeddings"]


def save_embedding_cache(path: str | Path, ids: list[str], embeddings) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray(ids), embeddings=np.asarray(embeddings))


def save_checkpoint(path: str | Path, ids: list[str], embeddings) -> None:
    save_embedding_cache(path, ids, embeddings)


def find_latest_checkpoint(checkpoint_dir: str | Path):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("corpus_checkpoint_*.npz"))
    if not checkpoints:
        return None

    def _step(path: Path) -> int:
        match = re.search(r"corpus_checkpoint_(\d+)\.npz$", path.name)
        return int(match.group(1)) if match else -1

    latest = max(checkpoints, key=_step)
    cached = load_embedding_cache(latest)
    if cached is None:
        return None
    return latest, cached[0], cached[1]


def clear_run_cache(run_cache_dir: str | Path, run_results_dir: str | Path) -> None:
    for path in [Path(run_cache_dir), Path(run_results_dir)]:
        if path.exists():
            shutil.rmtree(path)


def dump_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
