"""Sentence-level dilution diagnosis for zero-nDCG bad cases.

For each zero-nDCG query, checks whether the gold document contains a
sentence that individually matches the query better than any sentence in
the top-K retrieved docs — i.e., whether the right information exists but
got diluted in the whole-doc embedding.

Classes:
  A_dilution : sim(gold_best_sent) > sim(top_best_sent) AND sim(gold_doc) < sim(top_doc_max)
               → key info exists, whole-doc pooling drowns it (proposition-level retrieval should help)
  B_misalign : sim(gold_best_sent) < sim(top_best_sent)
               → key info not lexically/semantically aligned (query rewrite / HyDE territory)
  C_noise    : sim(gold_best_sent) < LOW_SIM_THRESHOLD (absolute)
               → qrel likely noisy
  other      : rare remainder

Outputs under <run-dir>/dilution/:
  dilution_per_query.csv
  dilution_summary.json
  dilution_cases.html
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm

from config import cfg
from data.loader import _ensure_pil_image, resize_image
from embeddings.visual_encoder import create_encoder


SPLIT = "test"
LOW_SIM_THRESHOLD = 0.35  # empirical; absolute cosine below this means the gold sent is barely related
SENT_ENCODE_BATCH = 16


def docid_to_source(did: str) -> str:
    return did.split("-", 2)[2]


def qid_to_source(qid: str) -> str:
    return qid.split("-", 2)[2]


def split_sentences(text: str | None) -> List[str]:
    if not text:
        return []
    t = text.replace("\r", "")
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    parts = [p.strip(" \t-#*>") for p in parts]
    return [p for p in parts if len(p) >= 15]


def classify(sim_gold_doc, sim_top_doc_max, sim_gold_best, sim_top_best) -> str:
    if sim_gold_best < LOW_SIM_THRESHOLD:
        return "C_noise"
    if sim_gold_best > sim_top_best and sim_gold_doc < sim_top_doc_max:
        return "A_dilution"
    if sim_gold_best < sim_top_best:
        return "B_misalign"
    return "other"


def encode_text_batch(encoder, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 3584), dtype=np.float32)
    embeddings = []
    for i in range(0, len(texts), SENT_ENCODE_BATCH):
        batch = texts[i : i + SENT_ENCODE_BATCH]
        items = [{"id": f"s{i + k}", "text": t, "image": None, "modality": "text"} for k, t in enumerate(batch)]
        _, emb = encoder.encode_batch_items(items)
        embeddings.append(emb)
    return np.concatenate(embeddings, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to results/<run_id>/ containing predictions.json")
    ap.add_argument("--top-k", type=int, default=5, help="Compare against top-K retrieved docs' sentences")
    ap.add_argument("--max-queries", type=int, default=None, help="Limit queries for dev")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    predictions = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    per_query_csv = run_dir / "badcase" / "per_query.csv"
    if not per_query_csv.exists():
        raise FileNotFoundError(
            f"{per_query_csv} not found — run `python -m analysis.badcase --run-dir ... --stage 1` first"
        )

    out_dir = run_dir / "dilution"
    out_dir.mkdir(exist_ok=True)

    zero_rows = []
    with per_query_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["ndcg@10"]) == 0.0:
                zero_rows.append(row)
    if args.max_queries:
        zero_rows = zero_rows[: args.max_queries]
    print(f"[setup] {len(zero_rows)} zero-nDCG queries to diagnose")

    zqid_set = {r["qid"] for r in zero_rows}
    print("[setup] loading qrels ...")
    ds_qrels = load_dataset(cfg.data.hf_dataset_id, "qrels", split=SPLIT)
    qrels: Dict[str, Dict[str, int]] = {}
    for row in ds_qrels:
        qid = f"query-{SPLIT}-{row['query_id']}"
        if qid not in zqid_set:
            continue
        did = f"corpus-{SPLIT}-{row['corpus_id']}"
        qrels.setdefault(qid, {})[did] = int(row.get("score", 1))

    print("[setup] loading query + corpus datasets ...")
    query_ds = load_dataset(cfg.data.hf_dataset_id, "query", split=SPLIT)
    q_src_to_idx = {str(v): i for i, v in enumerate(query_ds["id"])}
    corpus_ds = load_dataset(cfg.data.hf_dataset_id, "corpus", split=SPLIT)
    c_src_to_idx = {str(v): i for i, v in enumerate(corpus_ds["id"])}

    print("[setup] loading GME encoder ...")
    encoder = create_encoder(
        model_name=cfg.model.model_name,
        device=cfg.model.device,
        max_image_tokens=cfg.model.max_image_tokens,
        max_length=cfg.model.max_length,
    )
    encoder.load()

    rows: List[dict] = []
    samples_by_class: Dict[str, List[dict]] = defaultdict(list)

    for z in tqdm(zero_rows, desc="dilution"):
        qid = z["qid"]
        q_src = qid_to_source(qid)
        if q_src not in q_src_to_idx:
            continue
        q_row = query_ds[q_src_to_idx[q_src]]
        query_item = {
            "id": qid,
            "text": q_row.get("text"),
            "image": resize_image(_ensure_pil_image(q_row.get("image"))),
            "modality": q_row.get("modality", "image,text"),
        }
        _, q_emb = encoder.encode_batch_items([query_item], instruction=cfg.data.query_instruction)
        q_vec = q_emb[0]

        gold_dids = list(qrels.get(qid, {}).keys())
        gold_sent_pairs: List[tuple] = []
        gold_doc_items: List[dict] = []
        for did in gold_dids:
            src = docid_to_source(did)
            if src not in c_src_to_idx:
                continue
            d_row = corpus_ds[c_src_to_idx[src]]
            for s in split_sentences(d_row.get("text")):
                gold_sent_pairs.append((did, s))
            gold_doc_items.append({
                "id": did,
                "text": d_row.get("text"),
                "image": resize_image(_ensure_pil_image(d_row.get("image"))),
                "modality": d_row.get("modality", "image,text"),
            })

        ranked = sorted(predictions.get(qid, {}).items(), key=lambda kv: -kv[1])[: args.top_k]
        top_dids = [d for d, _ in ranked]
        top_doc_scores = {d: float(s) for d, s in ranked}
        top_sent_pairs: List[tuple] = []
        for did in top_dids:
            src = docid_to_source(did)
            if src not in c_src_to_idx:
                continue
            d_row = corpus_ds[c_src_to_idx[src]]
            for s in split_sentences(d_row.get("text")):
                top_sent_pairs.append((did, s))

        gold_emb = encode_text_batch(encoder, [s for _, s in gold_sent_pairs])
        top_emb = encode_text_batch(encoder, [s for _, s in top_sent_pairs])

        def max_sim(mat):
            if len(mat) == 0:
                return -1.0, -1
            sims = mat @ q_vec
            idx = int(np.argmax(sims))
            return float(sims[idx]), idx

        gold_best_sim, gold_best_idx = max_sim(gold_emb)
        top_best_sim, top_best_idx = max_sim(top_emb)

        if gold_doc_items:
            _, gd_emb = encoder.encode_batch_items(gold_doc_items)
            sim_gold_doc = float(np.max(gd_emb @ q_vec))
        else:
            sim_gold_doc = -1.0

        sim_top_doc_max = max(top_doc_scores.values()) if top_doc_scores else -1.0

        cls = classify(sim_gold_doc, sim_top_doc_max, gold_best_sim, top_best_sim)
        row = {
            "qid": qid,
            "coarse_domain": z["coarse_domain"],
            "category": z["category"],
            "n_gold_sents": len(gold_sent_pairs),
            "n_top_sents": len(top_sent_pairs),
            "sim_gold_doc": round(sim_gold_doc, 4),
            "sim_top_doc_max": round(sim_top_doc_max, 4),
            "sim_gold_best_sent": round(gold_best_sim, 4),
            "sim_top_best_sent": round(top_best_sim, 4),
            "class": cls,
        }
        if gold_best_idx >= 0:
            row["gold_best_sent_did"] = gold_sent_pairs[gold_best_idx][0]
            row["gold_best_sent_text"] = gold_sent_pairs[gold_best_idx][1][:300]
        if top_best_idx >= 0:
            row["top_best_sent_did"] = top_sent_pairs[top_best_idx][0]
            row["top_best_sent_text"] = top_sent_pairs[top_best_idx][1][:300]
        rows.append(row)
        samples_by_class[cls].append(row)

    csv_path = out_dir / "dilution_per_query.csv"
    fieldnames = [
        "qid", "coarse_domain", "category", "class",
        "sim_gold_doc", "sim_top_doc_max", "sim_gold_best_sent", "sim_top_best_sent",
        "n_gold_sents", "n_top_sents",
        "gold_best_sent_did", "gold_best_sent_text",
        "top_best_sent_did", "top_best_sent_text",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[out] {csv_path}")

    summary = {
        "n_queries": len(rows),
        "top_k_for_comparison": args.top_k,
        "low_sim_threshold": LOW_SIM_THRESHOLD,
        "by_class": dict(Counter(r["class"] for r in rows)),
        "by_domain_class": {},
    }
    by_dc: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_dc[r["coarse_domain"]][r["class"]] += 1
    summary["by_domain_class"] = {d: dict(c) for d, c in sorted(by_dc.items())}
    (out_dir / "dilution_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[out] dilution_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    css = (
        "body{font-family:system-ui,sans-serif;max-width:1200px;margin:20px auto;padding:0 20px;}"
        "h1,h2{border-bottom:2px solid #333;padding-bottom:4px;}"
        ".case{border:1px solid #ddd;padding:12px;margin:14px 0;background:#fafafa;border-radius:6px;}"
        ".meta{color:#555;font-size:13px;margin-bottom:8px;}"
        ".sent{padding:8px;margin:6px 0;border-left:4px solid #888;background:#fff;font-size:14px;line-height:1.5;}"
        ".sent.gold{border-left-color:#2f855a;}"
        ".sent.top{border-left-color:#c05621;}"
        "pre{background:#f5f5f5;padding:10px;border-radius:4px;overflow:auto;}"
    )
    parts = [
        f'<!doctype html><html><head><meta charset="utf-8"><title>Dilution cases</title><style>{css}</style></head><body>',
        "<h1>Dilution diagnosis</h1>",
        f"<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>",
    ]
    for cls in ["A_dilution", "B_misalign", "C_noise", "other"]:
        examples = samples_by_class[cls][:10]
        if not examples:
            continue
        parts.append(f"<h2>{cls} — {len(samples_by_class[cls])} total, showing {len(examples)}</h2>")
        for r in examples:
            parts.append('<div class="case">')
            parts.append(
                f'<div class="meta">qid={html.escape(r["qid"])} | {r["coarse_domain"]}/{r["category"]} | '
                f"sim_gold_doc={r['sim_gold_doc']} sim_top_doc={r['sim_top_doc_max']} | "
                f"sim_gold_best_sent={r['sim_gold_best_sent']} sim_top_best_sent={r['sim_top_best_sent']}</div>"
            )
            if "gold_best_sent_text" in r:
                parts.append(
                    f'<div class="sent gold"><b>Gold best sentence</b> '
                    f'(<code>{html.escape(r.get("gold_best_sent_did", ""))}</code>): '
                    f'{html.escape(r["gold_best_sent_text"])}</div>'
                )
            if "top_best_sent_text" in r:
                parts.append(
                    f'<div class="sent top"><b>Top best sentence</b> '
                    f'(<code>{html.escape(r.get("top_best_sent_did", ""))}</code>): '
                    f'{html.escape(r["top_best_sent_text"])}</div>'
                )
            parts.append("</div>")
    parts.append("</body></html>")
    (out_dir / "dilution_cases.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"[out] dilution_cases.html")


if __name__ == "__main__":
    main()
