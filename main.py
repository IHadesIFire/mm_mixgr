"""GME-only MRMR retrieval pipeline.

Examples
--------
python main.py baseline
python main.py baseline --domains Science Medicine
python main.py baseline --max_corpus 500 --batch_size 2
python main.py baseline --model Alibaba-NLP/gme-Qwen2-VL-2B-Instruct --max_corpus 200
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from config import cfg
from data.loader import available_domains, iter_corpus, load_qrels, load_queries
from embeddings.cache import (
    clear_run_cache,
    dump_json,
    find_latest_checkpoint,
    get_run_dirs,
    load_embedding_cache,
    make_run_id,
    save_checkpoint,
    save_embedding_cache,
)
from embeddings.visual_encoder import create_encoder
from evaluation.metrics import evaluate_mrmr
from retrieval.granular import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GME-only MRMR baseline.")
    parser.add_argument("command", nargs="?", default="baseline", choices=["baseline"], help="Only the baseline command is kept in this refactor.")
    parser.add_argument("--model", default=cfg.model.model_name)
    parser.add_argument("--device", default=cfg.model.device)
    parser.add_argument("--batch_size", type=int, default=cfg.model.batch_size)
    parser.add_argument("--max_length", type=int, default=cfg.model.max_length)
    parser.add_argument("--max_image_tokens", type=int, default=cfg.model.max_image_tokens)
    parser.add_argument("--domains", nargs="*", choices=available_domains(), default=None)
    parser.add_argument("--split", default=cfg.data.split)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--max_corpus", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=cfg.retrieval.top_k)
    parser.add_argument("--corpus_chunk_size", type=int, default=cfg.retrieval.corpus_chunk_size)
    parser.add_argument("--checkpoint_every", type=int, default=cfg.retrieval.checkpoint_every)
    parser.add_argument("--clear_cache", action="store_true")
    return parser.parse_args()


def encode_queries(encoder, queries: list[dict], query_cache: Path, batch_size: int):
    cached = load_embedding_cache(query_cache)
    if cached is not None:
        logger.info("Loaded cached query embeddings from %s", query_cache)
        return cached

    query_ids, query_embeddings = encoder.encode_items(
        queries,
        instruction=cfg.data.query_instruction,
        batch_size=batch_size,
        desc="Encoding queries",
    )
    save_embedding_cache(query_cache, query_ids, query_embeddings)
    return query_ids, query_embeddings


def encode_corpus_streaming(
    encoder,
    *,
    split: str,
    max_corpus: int | None,
    corpus_cache: Path,
    checkpoint_dir: Path,
    batch_size: int,
    checkpoint_every: int,
):
    cached = load_embedding_cache(corpus_cache)
    if cached is not None:
        logger.info("Loaded cached corpus embeddings from %s", corpus_cache)
        return cached

    resume = find_latest_checkpoint(checkpoint_dir)
    corpus_ids: list[str] = []
    corpus_embeddings: list[np.ndarray] = []
    resume_count = 0

    if resume is not None:
        checkpoint_path, saved_ids, saved_embeddings = resume
        corpus_ids = list(saved_ids)
        corpus_embeddings = [np.asarray(row) for row in saved_embeddings]
        resume_count = len(corpus_ids)
        logger.info("Resuming corpus encoding from %s (%d docs)", checkpoint_path.name, resume_count)

    batch_items: list[dict] = []
    batch_has_image: bool | None = None
    t0 = time.time()

    def flush_batch() -> None:
        nonlocal batch_items, batch_has_image, corpus_ids, corpus_embeddings
        if not batch_items:
            return
        batch_ids, batch_embs = encoder.encode_batch_items(batch_items, instruction=None)
        corpus_ids.extend(batch_ids)
        corpus_embeddings.extend(batch_embs)
        batch_items = []
        batch_has_image = None

    for idx, item in enumerate(iter_corpus(split=split, max_docs=max_corpus)):
        if idx < resume_count:
            continue

        has_image = item.get("image") is not None
        if batch_has_image is None:
            batch_has_image = has_image
        if batch_items and (has_image != batch_has_image or len(batch_items) >= batch_size):
            flush_batch()
        if batch_has_image is None:
            batch_has_image = has_image
        batch_items.append(item)
        batch_has_image = has_image

        if checkpoint_every > 0 and (idx + 1) % checkpoint_every == 0:
            flush_batch()
            checkpoint_path = checkpoint_dir / f"corpus_checkpoint_{len(corpus_ids):06d}.npz"
            save_checkpoint(checkpoint_path, corpus_ids, np.asarray(corpus_embeddings))
            elapsed = max(time.time() - t0, 1e-6)
            speed = (len(corpus_ids) - resume_count) / elapsed if len(corpus_ids) > resume_count else 0.0
            logger.info("Checkpoint saved to %s | encoded=%d | speed=%.2f docs/s", checkpoint_path.name, len(corpus_ids), speed)

    flush_batch()
    corpus_embeddings_np = np.asarray(corpus_embeddings)
    save_embedding_cache(corpus_cache, corpus_ids, corpus_embeddings_np)
    return corpus_ids, corpus_embeddings_np


def print_summary(metrics: dict) -> None:
    primary = cfg.eval.primary_metric
    print("\n" + "=" * 72)
    print("MRMR / GME baseline results")
    print("=" * 72)
    overall = metrics["overall"]
    print(f"All        | {primary}: {overall.get(primary, 0.0):.5f} | queries: {overall.get('num_queries', 0)}")
    for group_name, group_metrics in metrics["coarse"].items():
        print(f"{group_name:<10} | {primary}: {group_metrics.get(primary, 0.0):.5f} | queries: {group_metrics.get('num_queries', 0)}")
    print("=" * 72)


def run_baseline(args: argparse.Namespace) -> None:
    domains = args.domains or cfg.data.domains
    run_id = make_run_id(
        task_name="knowledge",
        model_name=args.model,
        split=args.split,
        domains=domains,
        max_corpus=args.max_corpus,
        max_queries=args.max_queries,
        max_length=args.max_length,
        max_image_tokens=args.max_image_tokens,
    )
    run_dirs = get_run_dirs(cfg.paths.embedding_cache_dir, cfg.paths.results_dir, run_id)

    if args.clear_cache:
        clear_run_cache(run_dirs["cache_dir"], run_dirs["results_dir"])
        run_dirs = get_run_dirs(cfg.paths.embedding_cache_dir, cfg.paths.results_dir, run_id)

    run_config = {
        "run_id": run_id,
        "model": args.model,
        "device": args.device,
        "domains": domains,
        "split": args.split,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_image_tokens": args.max_image_tokens,
        "max_queries": args.max_queries,
        "max_corpus": args.max_corpus,
        "top_k": args.top_k,
        "corpus_chunk_size": args.corpus_chunk_size,
        "checkpoint_every": args.checkpoint_every,
        "query_instruction": cfg.data.query_instruction,
    }
    dump_json(run_dirs["run_config"], run_config)

    print("=" * 72)
    print("GME-only MRMR baseline")
    print(f"run_id      : {run_id}")
    print(f"model       : {args.model}")
    print(f"domains     : {', '.join(domains)}")
    print(f"split       : {args.split}")
    print(f"batch_size  : {args.batch_size}")
    print(f"max_queries : {args.max_queries or 'all'}")
    print(f"max_corpus  : {args.max_corpus or 'all'}")
    print(f"cache_dir   : {run_dirs['cache_dir']}")
    print(f"results_dir : {run_dirs['results_dir']}")
    print("=" * 72)

    queries = load_queries(domains=domains, split=args.split, max_queries=args.max_queries)
    qrels = load_qrels({item["id"] for item in queries}, split=args.split)
    queries = [item for item in queries if item["id"] in qrels]
    logger.info("Using %d queries after qrels filtering", len(queries))

    encoder = create_encoder(
        model_name=args.model,
        device=args.device,
        max_image_tokens=args.max_image_tokens,
        max_length=args.max_length,
    )

    try:
        query_ids, query_embeddings = encode_queries(
            encoder,
            queries,
            run_dirs["query_cache"],
            batch_size=args.batch_size,
        )
        corpus_ids, corpus_embeddings = encode_corpus_streaming(
            encoder,
            split=args.split,
            max_corpus=args.max_corpus,
            corpus_cache=run_dirs["corpus_cache"],
            checkpoint_dir=run_dirs["checkpoint_dir"],
            batch_size=args.batch_size,
            checkpoint_every=args.checkpoint_every,
        )
    finally:
        encoder.unload()

    predictions = retrieve(
        query_ids,
        query_embeddings,
        corpus_ids,
        corpus_embeddings,
        top_k=args.top_k,
        corpus_chunk_size=args.corpus_chunk_size,
    )
    metrics = evaluate_mrmr(qrels, predictions, queries, cfg.eval.k_values)

    dump_json(run_dirs["predictions"], predictions)
    dump_json(run_dirs["metrics"], metrics)
    print_summary(metrics)
    print(f"Predictions saved to: {run_dirs['predictions']}")
    print(f"Metrics saved to:     {run_dirs['metrics']}")


if __name__ == "__main__":
    run_baseline(parse_args())
