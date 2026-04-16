"""Rebuild predictions.json + metrics.json from existing cache — no GPU.

Also deletes the stale old-format cache files after confirmation.

Usage:
  python -m analysis.rebuild_predictions \
      --run-dir /content/drive/MyDrive/mm_mixgr_cache/results/knowledge_gme-qwen2-vl-7b-instruct_art-humanities-medicine-science_call_qall_af651daa09 \
      --cache-dir /content/drive/MyDrive/mm_mixgr_cache/results/embeddings/knowledge_gme-qwen2-vl-7b-instruct_art-humanities-medicine-science_call_qall_af651daa09 \
      --delete-old-cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

from config import cfg
from data.loader import load_qrels, load_queries
from embeddings.cache import dump_json, load_embedding_cache
from evaluation.metrics import evaluate_mrmr
from retrieval.granular import retrieve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--top-k", type=int, default=cfg.retrieval.top_k)
    ap.add_argument("--delete-old-cache", action="store_true",
                    help="delete old-format cache files under results/embeddings/")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    q_cache = load_embedding_cache(cache_dir / "queries.npz")
    c_cache = load_embedding_cache(cache_dir / "corpus.npz")
    if q_cache is None or c_cache is None:
        print(f"[ERROR] cache not found under {cache_dir}")
        return
    q_ids, q_embs = q_cache
    c_ids, c_embs = c_cache
    print(f"[cache] queries={len(q_ids)} corpus={len(c_ids)}")

    predictions = retrieve(
        q_ids, q_embs, c_ids, c_embs,
        top_k=args.top_k,
        corpus_chunk_size=cfg.retrieval.corpus_chunk_size,
    )

    # evaluate
    queries = load_queries(domains=cfg.data.domains, split=cfg.data.split, max_queries=None)
    qrels = load_qrels({item["id"] for item in queries}, split=cfg.data.split)
    queries = [item for item in queries if item["id"] in qrels]
    metrics = evaluate_mrmr(qrels, predictions, queries, cfg.eval.k_values)

    pred_path = run_dir / "predictions.json"
    metrics_path = run_dir / "metrics.json"
    dump_json(pred_path, predictions)
    dump_json(metrics_path, metrics)
    print(f"[out] {pred_path}")
    print(f"[out] {metrics_path}")
    print(f"\n[nDCG@10 overall] {metrics['overall'].get('nDCG@10', 0.0):.5f}")
    for name, m in metrics["coarse"].items():
        print(f"  {name:<12} nDCG@10={m.get('nDCG@10', 0.0):.5f}  n={m.get('num_queries', 0)}")

    if args.delete_old_cache:
        parent = cache_dir.parent  # .../results/embeddings/
        targets = [
            parent / "gme-qwen2-vl-7b-instruct_query_all.npz",
            parent / "gme-qwen2-vl-7b-instruct_corpus.npz",
        ]
        # also stale corpus checkpoints directly under parent
        for p in parent.glob("corpus_checkpoint_*.npz"):
            targets.append(p)
        print("\n[cleanup] deleting stale old-format cache files:")
        for p in targets:
            if p.exists():
                p.unlink()
                print(f"  rm {p}")


if __name__ == "__main__":
    main()
