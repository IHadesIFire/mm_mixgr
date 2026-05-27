"""对 benchmark corpus 的带图 doc 做 proposition grounding, 产出多模态子 doc.

对每个带图 corpus doc:
  - 查它的 proposition(从 cache/decompositions, md5("proposition:"+text))
  - 每条 prop 调 qwen3.6-plus 做 grounding, 输出 regions(可多框)
  - 结果增量写 doc_propositions.jsonl(可断点续跑)

并发: 按 doc 分线程(每 doc 图只编码一次, 其 prop 串行); 默认 10 路.
续跑: 启动时读已有 jsonl, 跳过已完成的 (corpus_id, prop_idx).

跑(pilot 5 doc): python -u -m scripts.run_grounding --limit 5 --workers 5
跑(全量):       python -u -m scripts.run_grounding --workers 12
"""
import argparse
import base64
import hashlib
import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("HF_DATASETS_CACHE", r"D:\hf_cache\datasets")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import load_dataset  # noqa: E402
from openai import OpenAI  # noqa: E402
from PIL import Image  # noqa: E402

CACHE_DIR = Path("cache/decompositions")
CORPUS_PARQUET = "data/benchmark_v1/corpus.parquet"
OUT_PATH = Path("data/benchmark_v1/doc_propositions.jsonl")
MODEL = "qwen3.6-plus"
MAX_SIDE = 1280   # 发图前长边上限

# grounding prompt(与 tests/test_qwen_grounding_multi.py 最终版一致)
PROMPT_TEMPLATE = """Look at the image and localize the key visual entity mentioned in the proposition below.

Proposition: "{prop}"

Grounding rules (important, follow strictly):
- Set grounded=true as long as the KEY NOUN (key visual entity) of the proposition
  has a corresponding visual element in the image.
- Do NOT require the proposition's predicate, function, attribute, or external
  knowledge (e.g. specific example names) to also be visible.
- Set grounded=false ONLY when the key noun itself is an abstract concept with no
  corresponding visual entity in the image (e.g. "cell communication",
  "signal transduction" --- concepts with no concrete visual form).
- A SINGLE proposition may correspond to MULTIPLE regions. Output one entry in
  "regions" for EACH distinct region the proposition refers to.

Output JSON strictly in this format:
{{
  "grounded": true/false,
  "regions": [
    {{"key_visual_entity": "the visual entity for this region",
     "bbox_norm": [x1, y1, x2, y2]}}
  ],
  "confidence": 0.0-1.0,
  "reason": "1-2 sentences on whether the key entity is visible and how regions were chosen"
}}

Requirements:
- IGNORE all printed text/words/labels/captions in the image. Ground ONLY to actual
  visual/pictorial content (anatomical structures, cells, diagrams, photos), never
  to any text. If the image is essentially a text document with no real pictorial
  content, set grounded=false and regions=[].
- Localize as PRECISELY as possible to the exact region(s) the proposition refers to.
  If the proposition does not refer to any specific region, set grounded=false and regions=[].
- Each bbox_norm is normalized to 0-1000 (top-left [0,0], bottom-right [1000,1000]).
- grounded depends ONLY on whether the key visual entity is visible.
- Output ONLY the JSON, no text outside it."""

_write_lock = threading.Lock()


def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def props_of(text: str):
    """查 doc 文本对应的 proposition 列表, 没缓存返回 None."""
    f = CACHE_DIR / f"{md5('proposition:' + text)}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def downscale(img: Image.Image, max_side: int = MAX_SIDE) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)


def img_to_b64(img: Image.Image) -> str:
    img = downscale(img.convert("RGB"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ground_one(client: OpenAI, img_b64: str, prop: str) -> dict:
    """调 Qwen grounding, 返回解析后的 dict(失败抛异常给上层捕获)."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT_TEMPLATE.format(prop=prop)},
            ],
        }],
        temperature=0,
        response_format={"type": "json_object"},
        extra_body={"enable_thinking": False},
    )
    return json.loads(resp.choices[0].message.content)


def load_done(path: Path) -> set:
    """读已有 jsonl, 返回已完成的 (corpus_id, prop_idx) 集合, 支持断点续跑."""
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["corpus_id"], r["prop_idx"]))
            except Exception:
                continue
    return done


def process_doc(client, doc_id, img, props, done, stats):
    """处理一个 doc 的所有 prop, 增量写 jsonl. 返回本 doc 新完成的条数."""
    img_b64 = img_to_b64(img)   # 每个 doc 图只编码一次
    n_new = 0
    for prop_idx, prop in enumerate(props):
        if (doc_id, prop_idx) in done:
            continue
        rec = {"corpus_id": doc_id, "prop_id": f"{doc_id}_p{prop_idx}",
               "prop_idx": prop_idx, "text": prop}
        try:
            r = ground_one(client, img_b64, prop)
            rec.update({
                "grounded": r.get("grounded"),
                "regions": r.get("regions", []),
                "confidence": r.get("confidence"),
                "reason": r.get("reason"),
            })
        except Exception as e:
            rec.update({"grounded": None, "regions": [], "error": str(e)})
            with stats["lock"]:
                stats["errors"] += 1
        with _write_lock:
            with open(OUT_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_new += 1
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个带图 doc(pilot 用)")
    ap.add_argument("--workers", type=int, default=10, help="并发 doc 数")
    args = ap.parse_args()

    assert os.environ.get("DASHSCOPE_API_KEY"), "没设 DASHSCOPE_API_KEY"
    client = OpenAI(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=180.0, max_retries=2,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] 加载 corpus + 收集带图 doc ...")
    corpus = load_dataset("parquet", data_files=CORPUS_PARQUET, split="train")
    done = load_done(OUT_PATH)
    print(f"      已完成 {len(done)} 条 prop(断点续跑会跳过)")

    # 收集带图 doc 的 (id, image, props)
    tasks = []
    n_img, n_noprop = 0, 0
    for i in range(len(corpus)):
        if corpus[i]["image"] is None:
            continue
        n_img += 1
        doc_id = corpus[i]["id"]
        props = props_of(corpus[i]["text"])
        if not props:
            n_noprop += 1
            continue
        tasks.append((doc_id, corpus[i]["image"], props))
        if args.limit and len(tasks) >= args.limit:
            break
    print(f"      带图 doc={n_img}, 无 prop 缓存={n_noprop}, 本次处理 doc={len(tasks)}")
    total_props = sum(len(p) for _, _, p in tasks)
    print(f"      本次涉及 prop 总数={total_props}")

    print(f"[2/3] 并发 grounding(workers={args.workers}) ...")
    stats = {"errors": 0, "lock": threading.Lock()}
    t0 = time.time()
    n_done_docs = 0
    n_new_props = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_doc, client, did, img, props, done, stats): did
                for did, img, props in tasks}
        for fut in as_completed(futs):
            n_done_docs += 1
            n_new_props += fut.result()
            if n_done_docs % 5 == 0 or n_done_docs == len(tasks):
                el = time.time() - t0
                rate = n_new_props / el if el > 0 else 0
                print(f"      doc {n_done_docs}/{len(tasks)}, 新增 prop {n_new_props}, "
                      f"错误 {stats['errors']}, {rate:.2f} prop/s, 已用 {el:.0f}s")

    el = time.time() - t0
    print(f"[3/3] 完成. 新增 {n_new_props} 条, 错误 {stats['errors']}, 耗时 {el:.0f}s")
    if n_new_props > 0:
        print(f"      平均 {el/n_new_props:.1f}s/prop")
        # 用本次速率估全量(假设全量 ~32000 prop)
        full_est = 32000 / (n_new_props / el) / 3600 if el > 0 else 0
        print(f"      按此速率, 全量 ~32000 prop 约需 {full_est:.1f} 小时")
    print(f"      输出: {OUT_PATH}")


if __name__ == "__main__":
    main()
