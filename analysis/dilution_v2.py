"""Dilution analysis v2 — same coordinate system as baseline.

q_vec, sim_gold_full and sim_top_full_max are READ from the baseline's cached
embeddings (queries.npz / corpus.npz) — exactly the vectors that produced the
nDCG@10 numbers. Only "sentence + gold_image" must be re-encoded since baseline
never computed it; we use single-item batches to minimise drift.

  sim_gold_full     = q_vec @ corpus_cache[gold_did]                  (cached)
  sim_top_full_max  = max over top-K of  q_vec @ corpus_cache[did]    (cached)
  sim_gold_sent_max = max over sentences s of  sim(query, s + gold_image)  (re-encoded)

Classes:
  dilution_winning : gold_sent_max > gold_full AND gold_sent_max > top_full_max
                     → granular retrieval would have won this query
  dilution_partial : gold_sent_max > gold_full BUT gold_sent_max <= top_full_max
                     → there IS dilution, but even the best sentence can't beat top
  no_dilution      : gold_sent_max <= gold_full
                     → whole doc is at least as good as any single sentence

Outputs under <run-dir>/dilution_v2/:
  per_query.csv       — one row per query with all sims + chosen class
  summary.json        — counts by class, by domain
  cases.html          — top examples per class for eyeballing

Run:
  python -m analysis.dilution_v2 --run-dir <results-run-dir>
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
from embeddings.cache import load_embedding_cache
from embeddings.visual_encoder import create_encoder

SPLIT = "test"
SENT_BATCH = 8           # sentences-with-image are heavy; keep small


def docid_to_source(d):  return d.split("-", 2)[2]
def qid_to_source(d):    return d.split("-", 2)[2]


def split_sentences(text):
    if not text:
        return []
    t = text.replace("\r", "")
    parts = re.split(r"(?<=[.!?])\s+|\n+", t)
    parts = [p.strip(" \t-#*>") for p in parts]
    return [p for p in parts if len(p) >= 15]


def classify(sim_full, sim_sent_max, sim_top_max):
    if sim_sent_max > sim_full:
        if sim_sent_max > sim_top_max:
            return "dilution_winning"
        return "dilution_partial"
    return "no_dilution"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cache-dir", default=None,
                    help="path containing queries.npz/corpus.npz; auto-resolved if omitted")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-queries", type=int, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    predictions = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    per_query_csv = run_dir / "badcase" / "per_query.csv"
    if not per_query_csv.exists():
        raise FileNotFoundError(f"{per_query_csv} not found")

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir \
        else (Path(cfg.paths.embedding_cache_dir) / run_dir.name).resolve()
    q_cache = load_embedding_cache(cache_dir / "queries.npz")
    c_cache = load_embedding_cache(cache_dir / "corpus.npz")
    if q_cache is None or c_cache is None:
        raise FileNotFoundError(f"queries.npz / corpus.npz not found under {cache_dir}")
    q_ids, q_embs = q_cache
    c_ids, c_embs = c_cache
    q_pos = {qid: i for i, qid in enumerate(q_ids)}
    c_pos = {did: i for i, did in enumerate(c_ids)}
    print(f"[cache] queries={len(q_ids)} corpus={len(c_ids)} from {cache_dir}")

    out_dir = run_dir / "dilution_v2"
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
    qrels: Dict[str, Dict[str, int]] = {}
    for row in load_dataset(cfg.data.hf_dataset_id, "qrels", split=SPLIT):
        qid = f"query-{SPLIT}-{row['query_id']}"
        if qid not in zqid_set:
            continue
        did = f"corpus-{SPLIT}-{row['corpus_id']}"
        qrels.setdefault(qid, {})[did] = int(row.get("score", 1))

    qds = load_dataset(cfg.data.hf_dataset_id, "query",  split=SPLIT)
    cds = load_dataset(cfg.data.hf_dataset_id, "corpus", split=SPLIT)
    q_idx = {str(v): i for i, v in enumerate(qds["id"])}
    c_idx = {str(v): i for i, v in enumerate(cds["id"])}

    print("[setup] loading GME ...")
    enc = create_encoder(
        model_name=cfg.model.model_name, device=cfg.model.device,
        max_image_tokens=cfg.model.max_image_tokens, max_length=cfg.model.max_length,
    )
    enc.load()

    # checkpoint / resume
    ckpt = out_dir / "per_query.csv"
    rows: List[dict] = []
    done: set = set()
    FIELDS = [
        "qid", "coarse_domain", "category", "class",
        "sim_gold_full", "sim_gold_sent_max", "sim_top_full_max",
        "n_sents", "best_sent_did", "best_sent_text",
    ]
    if ckpt.exists():
        with ckpt.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append(r)
                done.add(r["qid"])
        print(f"[resume] {len(done)} queries already done")

    def flush():
        with ckpt.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    def encode_with_image(items):
        if not items:
            return np.zeros((0, 3584), dtype=np.float32)
        out = []
        for i in range(0, len(items), SENT_BATCH):
            _, e = enc.encode_batch_items(items[i:i + SENT_BATCH])
            out.append(e)
        return np.concatenate(out, axis=0)

    new_since = 0
    for z in tqdm(zero_rows, desc="dilution_v2"):
        qid = z["qid"]
        if qid in done:
            continue
        if qid not in q_pos:
            continue
        q_vec = q_embs[q_pos[qid]]
        q_src = qid_to_source(qid)
        if q_src not in q_idx:
            continue

        # ---- gold side ----
        # sim_gold_full from baseline cache (same coordinate system as nDCG).
        # sentences+image are re-encoded (baseline never computed them).
        sim_gold_full = -1.0
        sim_sent_max = -1.0
        best_sent_did = ""
        best_sent_text = ""
        n_sents_total = 0
        for did in qrels.get(qid, {}):
            if did in c_pos:
                s_full = float(c_embs[c_pos[did]] @ q_vec)
                if s_full > sim_gold_full:
                    sim_gold_full = s_full

            src = docid_to_source(did)
            if src not in c_idx:
                continue
            d_row = cds[c_idx[src]]
            d_text = d_row.get("text") or ""
            d_img = resize_image(_ensure_pil_image(d_row.get("image")))

            sents = split_sentences(d_text)
            n_sents_total += len(sents)
            if not sents:
                continue
            modality = "image,text" if d_img is not None else "text"
            sent_items = [{"id": f"{did}__s{i}", "text": s, "image": d_img, "modality": modality}
                          for i, s in enumerate(sents)]
            sent_emb = encode_with_image(sent_items)
            sims = sent_emb @ q_vec
            j = int(np.argmax(sims))
            if float(sims[j]) > sim_sent_max:
                sim_sent_max = float(sims[j])
                best_sent_did = did
                best_sent_text = sents[j][:300]

        # ---- top side: read from baseline cache ----
        ranked = sorted(predictions.get(qid, {}).items(), key=lambda kv: -kv[1])[: args.top_k]
        sim_top_full_max = -1.0
        for did, saved in ranked:
            if did in c_pos:
                s = float(c_embs[c_pos[did]] @ q_vec)
                if abs(s - saved) > 1e-4:
                    print(f"[cache-mismatch] {qid} {did} saved={saved:.4f} recomputed={s:.4f}")
                if s > sim_top_full_max:
                    sim_top_full_max = s

        cls = classify(sim_gold_full, sim_sent_max, sim_top_full_max)
        rows.append({
            "qid": qid,
            "coarse_domain": z["coarse_domain"],
            "category": z["category"],
            "class": cls,
            "sim_gold_full": round(sim_gold_full, 4),
            "sim_gold_sent_max": round(sim_sent_max, 4),
            "sim_top_full_max": round(sim_top_full_max, 4),
            "n_sents": n_sents_total,
            "best_sent_did": best_sent_did,
            "best_sent_text": best_sent_text,
        })
        new_since += 1
        if new_since >= 5:
            flush()
            new_since = 0

    flush()
    print(f"[out] {ckpt}")

    # summary
    summary = {
        "n_queries": len(rows),
        "top_k": args.top_k,
        "by_class": dict(Counter(r["class"] for r in rows)),
        "by_domain_class": {},
    }
    by_dc: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_dc[r["coarse_domain"]][r["class"]] += 1
    summary["by_domain_class"] = {d: dict(c) for d, c in sorted(by_dc.items())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # html
    css = ("body{font-family:system-ui;max-width:1200px;margin:20px auto;padding:0 20px;}"
           "h1,h2{border-bottom:2px solid #333;padding-bottom:4px;}"
           ".case{border:1px solid #ddd;padding:12px;margin:14px 0;background:#fafafa;border-radius:6px;}"
           ".meta{color:#555;font-size:13px;margin-bottom:8px;}"
           ".sent{padding:8px;margin:6px 0;border-left:4px solid #2f855a;background:#fff;font-size:14px;}"
           "pre{background:#f5f5f5;padding:10px;border-radius:4px;}")
    parts = [f'<!doctype html><html><head><meta charset="utf-8"><style>{css}</style></head><body>',
             "<h1>Dilution v2</h1>",
             f"<pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>"]
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)
    for cls in ["dilution_winning", "dilution_partial", "no_dilution"]:
        examples = sorted(by_cls[cls], key=lambda r: -float(r["sim_gold_sent_max"]))[:10]
        if not examples: continue
        parts.append(f"<h2>{cls} — {len(by_cls[cls])} total, showing {len(examples)}</h2>")
        for r in examples:
            parts.append('<div class="case">')
            parts.append(
                f'<div class="meta">qid={html.escape(r["qid"])} | {r["coarse_domain"]}/{r["category"]} | '
                f"sim_gold_full={r['sim_gold_full']} sim_gold_sent_max={r['sim_gold_sent_max']} "
                f"sim_top_full_max={r['sim_top_full_max']}</div>"
            )
            if r.get("best_sent_text"):
                parts.append(
                    f'<div class="sent"><b>Gold best sent</b> '
                    f'(<code>{html.escape(r.get("best_sent_did",""))}</code>): '
                    f'{html.escape(r["best_sent_text"])}</div>'
                )
            parts.append("</div>")
    parts.append("</body></html>")
    (out_dir / "cases.html").write_text("\n".join(parts), encoding="utf-8")
    print(f"[out] {out_dir / 'cases.html'}")


if __name__ == "__main__":
    main()
