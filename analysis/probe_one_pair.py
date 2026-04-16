"""Encode one (query, gold_doc) pair via BASELINE path and V2 path, compare.

Baseline path: batched encoding like encode_corpus_streaming in main.py.
V2 path:       single-item encoding like dilution_v2.py.

Usage:
  python -m analysis.probe_one_pair --qid test_Agriculture_145
"""
from __future__ import annotations

import argparse

import numpy as np
from datasets import load_dataset

from config import cfg
from data.loader import _ensure_pil_image, resize_image
from embeddings.visual_encoder import create_encoder

SPLIT = "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", required=True, help="raw qid, e.g. test_Agriculture_145")
    args = ap.parse_args()

    qds = load_dataset(cfg.data.hf_dataset_id, "query",  split=SPLIT)
    cds = load_dataset(cfg.data.hf_dataset_id, "corpus", split=SPLIT)
    rds = load_dataset(cfg.data.hf_dataset_id, "qrels",  split=SPLIT)
    q_idx = {str(v): i for i, v in enumerate(qds["id"])}
    c_idx = {str(v): i for i, v in enumerate(cds["id"])}

    gold_src = [str(r["corpus_id"]) for r in rds if str(r["query_id"]) == args.qid]
    print(f"[qrels] gold for {args.qid}: {gold_src}")
    if not gold_src:
        return
    gold = gold_src[0]

    q_row = qds[q_idx[args.qid]]
    d_row = cds[c_idx[gold]]
    print(f"[query] has_image={q_row.get('image') is not None} modality={q_row.get('modality')}")
    print(f"[gold ] has_image={d_row.get('image') is not None} modality={d_row.get('modality')}")

    q_item = {
        "id": args.qid, "text": q_row.get("text"),
        "image": resize_image(_ensure_pil_image(q_row.get("image"))),
        "modality": q_row.get("modality", "image,text"),
    }
    d_item = {
        "id": gold, "text": d_row.get("text"),
        "image": resize_image(_ensure_pil_image(d_row.get("image"))),
        "modality": d_row.get("modality", "image,text"),
    }

    print("[setup] loading GME ...")
    enc = create_encoder(
        model_name=cfg.model.model_name, device=cfg.model.device,
        max_image_tokens=cfg.model.max_image_tokens, max_length=cfg.model.max_length,
    )
    enc.load()

    # ---- BASELINE path: encode_items with batch_size (like main.py) ----
    _, q_base = enc.encode_items([q_item], instruction=cfg.data.query_instruction,
                                  batch_size=cfg.model.batch_size, show_progress=False)
    _, d_base = enc.encode_items([d_item], instruction=None,
                                  batch_size=cfg.model.batch_size, show_progress=False)
    sim_base = float(q_base[0] @ d_base[0])

    # ---- V2 path: encode_batch_items, single item ----
    _, q_v2 = enc.encode_batch_items([q_item], instruction=cfg.data.query_instruction)
    _, d_v2 = enc.encode_batch_items([d_item], instruction=None)
    sim_v2 = float(q_v2[0] @ d_v2[0])

    print()
    print(f"[baseline path] sim(q, gold) = {sim_base:.6f}")
    print(f"[v2 path     ]  sim(q, gold) = {sim_v2:.6f}")
    print(f"[diff] {abs(sim_base - sim_v2):.6f}")
    print(f"[query vec cos] baseline vs v2 = {float(q_base[0] @ q_v2[0]):.6f}")
    print(f"[gold  vec cos] baseline vs v2 = {float(d_base[0] @ d_v2[0]):.6f}")


if __name__ == "__main__":
    main()
