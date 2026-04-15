"""Probe one query's dilution: full gold doc vs truncated prefix.

Hard-coded for: query-test-validation_Agriculture_1
Truncates the gold doc text at the first occurrence of the user-specified marker.
Default encoding: image+text for query, text-only for gold doc (gold has no image).
"""
from __future__ import annotations

import numpy as np
from datasets import load_dataset

from config import cfg
from data.loader import _ensure_pil_image, resize_image
from embeddings.visual_encoder import create_encoder

SPLIT = "test"
QID_SOURCE = "validation_Agriculture_1"          # → query-test-validation_Agriculture_1
TRUNC_MARKER = "# Leaf- Blight (Corynespora cassiicola)"


def main():
    print("[setup] loading datasets ...")
    qds = load_dataset(cfg.data.hf_dataset_id, "query",  split=SPLIT)
    cds = load_dataset(cfg.data.hf_dataset_id, "corpus", split=SPLIT)
    rds = load_dataset(cfg.data.hf_dataset_id, "qrels",  split=SPLIT)

    q_idx = {str(v): i for i, v in enumerate(qds["id"])}
    c_idx = {str(v): i for i, v in enumerate(cds["id"])}

    q_row = qds[q_idx[QID_SOURCE]]
    print(f"[query] id={QID_SOURCE} | category={q_row.get('category')} | "
          f"has_image={q_row.get('image') is not None}")
    print(f"[query] text: {(q_row.get('text') or '')[:300]!r}")

    gold_src = [str(r["corpus_id"]) for r in rds if str(r["query_id"]) == QID_SOURCE]
    print(f"[gold] {len(gold_src)} gold doc(s): {gold_src}")

    print("[setup] loading GME ...")
    enc = create_encoder(
        model_name=cfg.model.model_name,
        device=cfg.model.device,
        max_image_tokens=cfg.model.max_image_tokens,
        max_length=cfg.model.max_length,
    )
    enc.load()

    q_item = {
        "id": f"query-{SPLIT}-{QID_SOURCE}",
        "text": q_row.get("text"),
        "image": resize_image(_ensure_pil_image(q_row.get("image"))),
        "modality": q_row.get("modality", "image,text"),
    }
    _, q_emb = enc.encode_batch_items([q_item], instruction=cfg.data.query_instruction)
    q_vec = q_emb[0]
    print(f"[encode] query vec shape={q_vec.shape}, |v|={np.linalg.norm(q_vec):.4f}")

    for src in gold_src:
        if src not in c_idx:
            print(f"[gold] {src} not found in corpus, skip")
            continue
        d_row = cds[c_idx[src]]
        full_text = d_row.get("text") or ""
        has_img = d_row.get("image") is not None
        print(f"\n[doc] id={src} | has_image={has_img} | full_len={len(full_text)} chars")

        if TRUNC_MARKER in full_text:
            cut = full_text.index(TRUNC_MARKER)
            prefix = full_text[:cut].rstrip()
            print(f"[trunc] marker found at char {cut} | prefix_len={len(prefix)}")
        else:
            print(f"[trunc] marker {TRUNC_MARKER!r} NOT FOUND — using full text as prefix (sanity check)")
            prefix = full_text

        full_item = {"id": f"{src}__full",  "text": full_text, "image": None, "modality": "text"}
        pref_item = {"id": f"{src}__prefix","text": prefix,    "image": None, "modality": "text"}
        _, e = enc.encode_batch_items([full_item, pref_item])
        sim_full   = float(e[0] @ q_vec)
        sim_prefix = float(e[1] @ q_vec)
        print(f"  sim(query, FULL  doc) = {sim_full:.4f}")
        print(f"  sim(query, PREFIX  )  = {sim_prefix:.4f}")
        print(f"  delta (prefix - full) = {sim_prefix - sim_full:+.4f}")


if __name__ == "__main__":
    main()
