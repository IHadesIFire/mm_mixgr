"""Bad-case analysis for MRMR Knowledge retrieval runs.

Two-stage output under <run_dir>/badcase/:
  stage 1 (stats, no images) -> per_query.csv, stats.json
  stage 2 (samples per coarse domain, with images) -> <domain>.html, index.html
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import pytrec_eval
from datasets import load_dataset
from PIL import Image

from config import cfg
from data.loader import CATEGORY_MAP, coarse_domain_from_category, resize_image, _ensure_pil_image


SPLIT = "test"
THUMB_MAX = 400


def per_query_ndcg10(qrels: Dict[str, Dict[str, int]], predictions: Dict[str, Dict[str, float]]):
    merged = {qid: dict(predictions.get(qid, {})) for qid in qrels}
    ev = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    scored = ev.evaluate(merged)
    return {qid: s.get("ndcg_cut_10", 0.0) for qid, s in scored.items()}


def best_gold_rank(gold_ids: set, ranked_docs: List[str]) -> int | None:
    for i, d in enumerate(ranked_docs, 1):
        if d in gold_ids:
            return i
    return None


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "miss"
    if rank == 1:
        return "1"
    if rank <= 5:
        return "2-5"
    if rank <= 10:
        return "6-10"
    if rank <= 50:
        return "11-50"
    return "51+"


def load_query_metadata() -> Dict[str, dict]:
    ds = load_dataset(cfg.data.hf_dataset_id, "query", split=SPLIT)
    out: Dict[str, dict] = {}
    for row in ds:
        qid = f"query-{SPLIT}-{row['id']}"
        out[qid] = {
            "id": qid,
            "source_id": str(row["id"]),
            "category": row.get("category"),
            "coarse_domain": coarse_domain_from_category(row.get("category")),
            "modality": row.get("modality", "image,text"),
            "has_image": row.get("image") is not None,
        }
    return out


def build_corpus_index():
    ds = load_dataset(cfg.data.hf_dataset_id, "corpus", split=SPLIT)
    ids = ds["id"]
    src_to_idx = {str(v): i for i, v in enumerate(ids)}
    return ds, src_to_idx


def docid_to_source(docid: str) -> str:
    # format: corpus-<split>-<source_id>
    return docid.split("-", 2)[2]


def qid_to_source(qid: str) -> str:
    return qid.split("-", 2)[2]


def pil_to_data_uri(img: Image.Image | None, max_size: int = THUMB_MAX) -> str | None:
    if img is None:
        return None
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_block(title: str, text: str | None, img_uri: str | None, tag: str = "") -> str:
    tag_html = f'<span class="tag">{html.escape(tag)}</span>' if tag else ""
    text_html = html.escape(text or "(no text)")
    img_html = f'<img src="{img_uri}" />' if img_uri else '<div class="noimg">(no image)</div>'
    return f"""
    <div class="block">
      <div class="title">{html.escape(title)} {tag_html}</div>
      <div class="row">
        <div class="img">{img_html}</div>
        <div class="text">{text_html}</div>
      </div>
    </div>"""


_CSS = """
body{font-family:system-ui,sans-serif;max-width:1200px;margin:20px auto;padding:0 20px;}
h1{border-bottom:2px solid #333;}
h2{margin-top:40px;border-bottom:1px solid #ccc;padding-bottom:4px;}
.case{border:1px solid #ddd;padding:16px;margin:20px 0;background:#fafafa;border-radius:6px;}
.meta{color:#666;font-size:13px;margin-bottom:10px;}
.block{margin:10px 0;padding:8px;background:#fff;border-left:3px solid #888;}
.block.query{border-left-color:#2b6cb0;}
.block.gold{border-left-color:#2f855a;}
.block.top{border-left-color:#c05621;}
.block.top.hit{border-left-color:#2f855a;background:#f0fff4;}
.title{font-weight:bold;margin-bottom:6px;}
.tag{display:inline-block;padding:1px 6px;margin-left:6px;background:#eee;border-radius:3px;font-size:11px;color:#555;}
.row{display:flex;gap:12px;}
.img{flex:0 0 420px;}
.img img{max-width:400px;max-height:400px;border:1px solid #ddd;}
.noimg{color:#999;font-style:italic;padding:20px;text-align:center;border:1px dashed #ccc;}
.text{flex:1;white-space:pre-wrap;font-size:14px;line-height:1.5;}
"""


def html_head(title: str) -> str:
    return f'<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{_CSS}</style></head><body>'


def stage1_stats(
    predictions: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    query_meta: Dict[str, dict],
    out_dir: Path,
):
    ndcg = per_query_ndcg10(qrels, predictions)

    rows = []
    for qid, gold in qrels.items():
        ranked = sorted(predictions.get(qid, {}).items(), key=lambda kv: -kv[1])
        ranked_ids = [d for d, _ in ranked]
        r = best_gold_rank(set(gold.keys()), ranked_ids)
        meta = query_meta.get(qid, {})
        rows.append({
            "qid": qid,
            "coarse_domain": meta.get("coarse_domain", "Unknown"),
            "category": meta.get("category", "Unknown"),
            "has_image": meta.get("has_image", False),
            "n_gold": len(gold),
            "ndcg@10": round(ndcg.get(qid, 0.0), 5),
            "best_gold_rank": r if r is not None else -1,
            "rank_bucket": rank_bucket(r),
        })

    csv_path = out_dir / "per_query.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    stats = {
        "num_queries": len(rows),
        "gold_count_distribution": dict(Counter(r["n_gold"] for r in rows)),
        "overall": {
            "mean_ndcg@10": round(sum(r["ndcg@10"] for r in rows) / len(rows), 5),
            "zero_ndcg": sum(1 for r in rows if r["ndcg@10"] == 0),
            "rank_buckets": dict(Counter(r["rank_bucket"] for r in rows)),
        },
        "by_coarse_domain": {},
        "by_has_image": {},
    }
    for key_name, key_fn in [("by_coarse_domain", lambda r: r["coarse_domain"]),
                              ("by_has_image", lambda r: "image+text" if r["has_image"] else "text-only")]:
        groups: Dict[str, list] = defaultdict(list)
        for r in rows:
            groups[key_fn(r)].append(r)
        for k, items in sorted(groups.items()):
            stats[key_name][k] = {
                "n": len(items),
                "mean_ndcg@10": round(sum(i["ndcg@10"] for i in items) / len(items), 5),
                "zero_ndcg": sum(1 for i in items if i["ndcg@10"] == 0),
                "rank_buckets": dict(Counter(i["rank_bucket"] for i in items)),
            }

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stage1] per_query.csv + stats.json -> {out_dir}")
    return rows, stats


def stage2_samples(
    predictions: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    query_meta: Dict[str, dict],
    per_query_rows: list,
    out_dir: Path,
    n_per_domain: int,
    seed: int,
):
    print(f"[stage2] loading HF datasets (query + corpus)...")
    query_ds = load_dataset(cfg.data.hf_dataset_id, "query", split=SPLIT)
    q_src_to_idx = {str(v): i for i, v in enumerate(query_ds["id"])}
    corpus_ds, c_src_to_idx = build_corpus_index()

    by_domain: Dict[str, list] = defaultdict(list)
    for r in per_query_rows:
        by_domain[r["coarse_domain"]].append(r)

    rng = random.Random(seed)
    samples_dir = out_dir
    index_lines = ['<h1>Bad case samples</h1><ul>']

    for domain in sorted(by_domain.keys()):
        rows = by_domain[domain]
        zero_rows = [r for r in rows if r["ndcg@10"] == 0]
        low_rows = sorted([r for r in rows if r["ndcg@10"] > 0], key=lambda r: r["ndcg@10"])

        pool = list(zero_rows)
        rng.shuffle(pool)
        picked = pool[:n_per_domain]
        if len(picked) < n_per_domain:
            picked += low_rows[: n_per_domain - len(picked)]

        html_parts = [html_head(f"Bad cases: {domain}"), f"<h1>Bad cases — {domain}</h1>",
                      f"<p>{len(rows)} queries; {len(zero_rows)} with nDCG@10=0. Showing {len(picked)}.</p>"]

        for r in picked:
            qid = r["qid"]
            q_src = qid_to_source(qid)
            q_row = query_ds[q_src_to_idx[q_src]]
            q_img = resize_image(_ensure_pil_image(q_row.get("image")), max_size=1000)
            q_text = q_row.get("text")

            gold_ids = set(qrels[qid].keys())
            ranked = sorted(predictions.get(qid, {}).items(), key=lambda kv: -kv[1])[:5]

            html_parts.append(f'<div class="case">')
            html_parts.append(
                f'<div class="meta">qid={html.escape(qid)} | category={html.escape(str(r["category"]))} '
                f'| n_gold={r["n_gold"]} | nDCG@10={r["ndcg@10"]} | best_gold_rank={r["best_gold_rank"]}</div>'
            )
            html_parts.append(render_block("Query", q_text, pil_to_data_uri(q_img), tag=r.get("category", "")).replace(
                'class="block"', 'class="block query"'))

            for gid in list(gold_ids)[:3]:
                g_src = docid_to_source(gid)
                if g_src not in c_src_to_idx:
                    html_parts.append(f'<div class="block gold"><div class="title">Gold (missing from corpus): {html.escape(gid)}</div></div>')
                    continue
                g_row = corpus_ds[c_src_to_idx[g_src]]
                g_img = resize_image(_ensure_pil_image(g_row.get("image")), max_size=1000)
                html_parts.append(render_block(f"Gold {gid}", g_row.get("text"), pil_to_data_uri(g_img)).replace(
                    'class="block"', 'class="block gold"'))

            for i, (did, score) in enumerate(ranked, 1):
                d_src = docid_to_source(did)
                if d_src not in c_src_to_idx:
                    continue
                d_row = corpus_ds[c_src_to_idx[d_src]]
                d_img = resize_image(_ensure_pil_image(d_row.get("image")), max_size=1000)
                is_hit = did in gold_ids
                tag = f"rank {i} | score={score:.4f}" + (" | GOLD" if is_hit else "")
                block = render_block(f"Retrieved #{i}", d_row.get("text"), pil_to_data_uri(d_img), tag=tag)
                cls = "block top hit" if is_hit else "block top"
                html_parts.append(block.replace('class="block"', f'class="{cls}"'))

            html_parts.append('</div>')

        html_parts.append("</body></html>")
        out_path = samples_dir / f"{domain}.html"
        out_path.write_text("\n".join(html_parts), encoding="utf-8")
        print(f"[stage2] {domain}: {len(picked)} cases -> {out_path}")
        index_lines.append(f'<li><a href="{domain}.html">{domain}</a> ({len(rows)} queries, {len(zero_rows)} zero-nDCG)</li>')

    index_lines.append("</ul>")
    (samples_dir / "index.html").write_text(
        html_head("Badcase index") + "\n".join(index_lines) + "</body></html>",
        encoding="utf-8",
    )
    print(f"[stage2] index -> {samples_dir / 'index.html'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to results/<run_id>/ containing predictions.json")
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both")
    ap.add_argument("--n-per-domain", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    predictions = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))

    out_dir = run_dir / "badcase"
    out_dir.mkdir(exist_ok=True)

    print(f"[setup] loading query metadata + qrels from HF...")
    query_meta = load_query_metadata()
    ds_qrels = load_dataset(cfg.data.hf_dataset_id, "qrels", split=SPLIT)
    qrels: Dict[str, Dict[str, int]] = {}
    for row in ds_qrels:
        qid = f"query-{SPLIT}-{row['query_id']}"
        if qid not in query_meta:
            continue
        did = f"corpus-{SPLIT}-{row['corpus_id']}"
        qrels.setdefault(qid, {})[did] = int(row.get("score", 1))

    # restrict to qids actually in predictions (in case run was partial)
    qrels = {q: g for q, g in qrels.items() if q in predictions}
    print(f"[setup] {len(qrels)} queries with gold + predictions")

    rows, _ = stage1_stats(predictions, qrels, query_meta, out_dir)

    if args.stage in ("2", "both"):
        stage2_samples(predictions, qrels, query_meta, rows, out_dir, args.n_per_domain, args.seed)


if __name__ == "__main__":
    main()
