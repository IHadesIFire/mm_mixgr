"""Check if queries.npz / corpus.npz match predictions.json scores.

No GPU needed. Just loads cache + predictions and compares.

Usage:
  python -m analysis.check_cache_match --run-dir <results-run-dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import cfg
from embeddings.cache import load_embedding_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-queries", type=int, default=20)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir \
        else (Path(cfg.paths.embedding_cache_dir) / run_dir.name).resolve()

    predictions = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    q_cache = load_embedding_cache(cache_dir / "queries.npz")
    c_cache = load_embedding_cache(cache_dir / "corpus.npz")
    if q_cache is None or c_cache is None:
        print(f"[ERROR] cache not found under {cache_dir}")
        return
    q_ids, q_embs = q_cache
    c_ids, c_embs = c_cache
    q_pos = {qid: i for i, qid in enumerate(q_ids)}
    c_pos = {did: i for i, did in enumerate(c_ids)}
    print(f"[cache] queries={len(q_ids)} corpus={len(c_ids)}")
    print(f"[predictions] queries={len(predictions)}")

    n_checked = 0
    n_mismatch = 0
    max_diff = 0.0

    for qid in list(predictions.keys())[:args.max_queries]:
        if qid not in q_pos:
            print(f"[WARN] {qid} not in query cache")
            continue
        q_vec = q_embs[q_pos[qid]]
        ranked = sorted(predictions[qid].items(), key=lambda kv: -kv[1])[:args.top_k]
        for did, saved in ranked:
            if did not in c_pos:
                print(f"[WARN] {did} not in corpus cache")
                continue
            recomputed = float(c_embs[c_pos[did]] @ q_vec)
            diff = abs(recomputed - saved)
            max_diff = max(max_diff, diff)
            n_checked += 1
            if diff > 1e-3:
                n_mismatch += 1
                print(f"[MISMATCH] {qid} | {did} | saved={saved:.6f} recomputed={recomputed:.6f} diff={diff:.6f}")

    print(f"\n[result] checked={n_checked} mismatch={n_mismatch} max_diff={max_diff:.6f}")
    if n_mismatch == 0:
        print("[OK] cache and predictions.json are consistent")
    else:
        print("[BAD] cache and predictions.json are NOT from the same encoding")


if __name__ == "__main__":
    main()
