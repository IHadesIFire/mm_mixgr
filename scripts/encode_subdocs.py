"""把每条 proposition 编码成"多模态子文档(sub-doc)"向量, 用于后续细粒度检索.

方法(对应你的方法主线 + FGVP, NeurIPS 2023):
  - grounded=True 的 prop: 用 **Blur Reverse Box (BRB)** 把 grounding 出来的 region
      之外的像素做高斯模糊、region 内保留清晰原像素, 得到"焦点图",
      再和 prop 文本一起喂 GME 做 fused 编码 (modality="image,text").
      依据: FGVP (Yang et al., NeurIPS 2023) 证明 Blur Reverse Mask/Box 在 VLM 上
      做区域聚焦优于 crop / 灰底 mask / 红框, 因为 VLM 训练数据自带浅景深(bokeh)先验.
      我们只有 bbox 没有像素 mask, 所以用其 Box 变体(BRB), 论文 Table 2 验证 BRB 仍强于其它。
  - grounded=False 的 prop(抽象概念, 无视觉锚点): 退化成纯文本编码 (modality="text").
  - 纯文本 corpus doc(没有 image 的)的 prop: 从命题缓存现读, 纯文本编码.
  - 完全没有命题缓存的 doc: 整 doc 兜底编码一条(prop_idx=-1), 保证每个 corpus doc 至少一个向量、可被检索到.

焦点图(grounded prop)有三种做法, 由 --focus_mode 选:
  - context_crop (默认): 裁 grounding 框 + 四周留 --context_margin 边距(目标高分辨率 + 局部上下文)
  - crop                : 裁紧 grounding 框(目标吃满 token, 无背景)
  - brb                 : Blur Reverse Box, 框内清晰、框外高斯模糊 σ=--blur_sigma(默认 100, 对标 FGVP)
  依据: FGVP(NeurIPS 2023)用固定 σ=100, 且其 Box 变体在 RefCOCO 上还弱于直接 crop;
  故默认改用 context_crop。各方案请用 nDCG 在子集上 ablate 后再定全量用哪个。

输出: data/benchmark_v1/subdoc_embeddings_<tag>/  (<tag> 默认=focus_mode, brb 带 σ; --out_tag 可覆盖)
  - embeddings.npy : (N, D) float32, 所有子文档向量
  - meta.jsonl     : 每行一条, 与 embeddings 行一一对应:
                     {row, prop_id, corpus_id, prop_idx, modality, grounded, has_image, source}
  - encode_stats.json : 统计
  - shards/ + manifest.jsonl : 中间产物, 支持断点续跑(按 chunk 粒度), finalize 时重建上面两个最终文件
  注: 不同 focus_mode 写到不同目录, 互不串断点 → 可各自全量编、出多版做对比。

依赖: 需要 GPU(GME-7B ~15GB). 在 Colab 跑(先把 data/benchmark_v1 + cache/decompositions 传到位).

跑(pilot, 前 20 个 doc, 顺便存焦点图抽查):
    python -u -m scripts.encode_subdocs --focus_mode context_crop --limit 20 --batch_size 8 --save_focus_samples 12
跑(全量):
    python -u -m scripts.encode_subdocs --focus_mode context_crop --batch_size 16
对比其它方案: --focus_mode 换成 crop / brb (brb 可再调 --blur_sigma) 各跑一版。
断点续跑: 重跑同一条命令即可(已完成的 chunk 会跳过).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os

# 关掉 tokenizers 多进程 fork 警告(纯噪声)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFilter

from config import cfg
from data.loader import _ensure_pil_image, resize_image
from embeddings.visual_encoder import create_encoder

# ---------------- 路径 / 常量 ----------------
BENCH = Path("data/benchmark_v1")
CORPUS_PARQUET = BENCH / "corpus.parquet"
GROUND_JSONL = BENCH / "doc_propositions.jsonl"
CACHE_DIR = Path("cache/decompositions")

OUT_DIR = BENCH / "subdoc_embeddings"
SHARD_DIR = OUT_DIR / "shards"
MANIFEST = OUT_DIR / "manifest.jsonl"
EMB_OUT = OUT_DIR / "embeddings.npy"
META_OUT = OUT_DIR / "meta.jsonl"
STATS_OUT = OUT_DIR / "encode_stats.json"
FOCUS_SAMPLE_DIR = OUT_DIR / "focus_samples"

MIN_BLUR_RADIUS = 6.0   # 高斯模糊半径下限(像素), 防止小图模糊几乎无效


# ---------------- 命题缓存读取 ----------------
def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def props_from_cache(text: str):
    """按 md5('proposition:'+text) 读命题缓存; 没有返回 None."""
    if not text:
        return None
    f = CACHE_DIR / f"{md5('proposition:' + text)}.json"
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, list) and data else None
    except Exception:
        return None


# ---------------- 焦点图: 把 grounding 区域做成 sub-doc 视图 ----------------
def _regions_to_px_boxes(regions, W, H):
    """bbox_norm(0-1000) -> 像素框列表. 纠正 min/max 顺序、裁到画面内、取整、小框 clamp 到 >=1px;
    画不出的丢弃。返回 [(x1,y1,x2,y2), ...] (空 = 无有效框)。"""
    boxes = []
    for reg in regions or []:
        bb = reg.get("bbox_norm")
        if not bb or len(bb) != 4:
            continue
        x1, y1, x2, y2 = bb
        px1 = max(0.0, min(float(W), min(x1, x2) / 1000.0 * W))
        px2 = max(0.0, min(float(W), max(x1, x2) / 1000.0 * W))
        py1 = max(0.0, min(float(H), min(y1, y2) / 1000.0 * H))
        py2 = max(0.0, min(float(H), max(y1, y2) / 1000.0 * H))
        ix1, iy1, ix2, iy2 = round(px1), round(py1), round(px2), round(py2)
        if ix2 <= ix1:
            ix2 = min(W, ix1 + 1)
        if iy2 <= iy1:
            iy2 = min(H, iy1 + 1)
        if ix2 <= ix1 or iy2 <= iy1:   # 图本身 <1px, 实在画不出框
            continue
        boxes.append((ix1, iy1, ix2, iy2))
    return boxes


def blur_reverse_box(img: Image.Image, boxes: list, blur_sigma: float, blur_ratio: float = 0.0):
    """Blur Reverse Box(FGVP 的 Box 变体): box 内清晰、box 外高斯模糊.

    FGVP 原文用固定 σ=100(executor.py blur_std_dev=100, 同一个 PIL GaussianBlur),
    所以默认走固定 blur_sigma; 仅当 blur_ratio>0 时退回"按长边比例"的老行为。
    """
    img = img.convert("RGB")
    W, H = img.size
    radius = blur_ratio * max(W, H) if blur_ratio and blur_ratio > 0 else blur_sigma
    radius = max(MIN_BLUR_RADIUS, radius)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for (ix1, iy1, ix2, iy2) in boxes:
        draw.rectangle([ix1, iy1, ix2, iy2], fill=255)
    return Image.composite(img, blurred, mask)   # 255 处清晰, 0 处模糊


def crop_union_box(img: Image.Image, boxes: list, margin: float = 0.0):
    """裁剪所有 box 的并集外接框; margin>0 时四周按框尺寸比例外扩(context crop)。"""
    img = img.convert("RGB")
    W, H = img.size
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    if margin and margin > 0:
        mw = (x2 - x1) * margin
        mh = (y2 - y1) * margin
        x1 = max(0, int(round(x1 - mw)))
        y1 = max(0, int(round(y1 - mh)))
        x2 = min(W, int(round(x2 + mw)))
        y2 = min(H, int(round(y2 + mh)))
    return img.crop((x1, y1, x2, y2))


def build_focus(pil_image, regions, focus_mode, blur_sigma, blur_ratio, context_margin):
    """按 focus_mode 把 grounding 区域做成焦点图. 无有效 box -> None(调用方退化纯文本)。"""
    if pil_image is None:
        return None
    W, H = pil_image.size
    boxes = _regions_to_px_boxes(regions, W, H)
    if not boxes:
        return None
    if focus_mode == "crop":
        return crop_union_box(pil_image, boxes, margin=0.0)
    if focus_mode == "context_crop":
        return crop_union_box(pil_image, boxes, margin=context_margin)
    return blur_reverse_box(pil_image, boxes, blur_sigma, blur_ratio)   # brb


# ---------------- 构造一个 doc 的所有子文档 item ----------------
def build_items_for_doc(cid, text, modality, pil_image, ground_recs,
                        focus_mode, blur_sigma, blur_ratio, context_margin, ungrounded_mode):
    """返回 (items, metas), 两者一一对应. item 给 encoder(含 id/modality/text/image),
    meta 给落盘(记录来源/是否 grounded 等). 同时返回少量焦点图供抽查 (focus_samples)。
    """
    items, metas, focus_samples = [], [], []

    def add(prop_id, prop_idx, item_modality, item_text, item_image, grounded, source):
        # 文本清洗 + 模态以实际内容为准:
        #   空白文本且无图 -> 跳过(不产生无意义向量)
        #   空白文本但有图 -> 纯图像模态
        #   有文本无图     -> 纯文本模态
        #   有文本有图     -> image,text
        t = (item_text or "").strip()
        if not t and item_image is None:
            return
        if not t:
            item_modality, item_text = "image", None
        elif item_image is None:
            item_modality, item_text = "text", t
        else:
            item_modality, item_text = "image,text", t
        items.append({"id": prop_id, "modality": item_modality,
                      "text": item_text, "image": item_image})
        metas.append({"prop_id": prop_id, "corpus_id": cid, "prop_idx": prop_idx,
                      "modality": item_modality, "grounded": grounded,
                      "has_image": item_image is not None, "source": source})

    if ground_recs:
        # 该 doc 在 grounding 阶段处理过(带图 doc)
        for rec in ground_recs:
            pidx = rec.get("prop_idx")
            pid = rec.get("prop_id") or f"{cid}_p{pidx}"
            ptext = rec.get("text") or ""
            grounded = rec.get("grounded") is True
            regions = rec.get("regions") or []
            if grounded and regions and pil_image is not None:
                focus = build_focus(pil_image, regions, focus_mode, blur_sigma, blur_ratio, context_margin)
                if focus is not None:
                    add(pid, pidx, "image,text", ptext, focus, True, f"grounded_{focus_mode}")
                    focus_samples.append((pid, focus))
                    continue
                # grounded=True 但 box 全部无效 -> 退化纯文本(grounded 标记如实保留)
                add(pid, pidx, "text", ptext, None, True, "grounded_box_invalid")
            elif grounded:
                # grounded=True 但缺 region 或缺图 -> 退化(保留 grounded=True, 标签如实)
                if ungrounded_mode == "wholeimage" and pil_image is not None:
                    add(pid, pidx, "image,text", ptext, pil_image, True, "grounded_no_region_wholeimg")
                else:
                    add(pid, pidx, "text", ptext, None, True, "grounded_degraded_text")
            else:
                # grounded=False(抽象概念, 无视觉锚点)
                if ungrounded_mode == "wholeimage" and pil_image is not None:
                    add(pid, pidx, "image,text", ptext, pil_image, False, "ungrounded_wholeimg")
                else:
                    add(pid, pidx, "text", ptext, None, False, "ungrounded_text")
    else:
        # 没有 grounding 记录: 纯文本 doc(1195 个), 或个别带图但当初无命题缓存的 doc
        props = props_from_cache(text)
        if props:
            for i, p in enumerate(props):
                add(f"{cid}_p{i}", i, "text", p, None, None, "textdoc_prop")
        else:
            # 命题缓存也没有 -> 整 doc 兜底编码一条, 保证可检索
            if pil_image is not None:
                add(f"{cid}_pwhole", -1, "image,text", text, pil_image, None, "wholedoc_fallback")
            else:
                add(f"{cid}_pwhole", -1, "text", text, None, None, "wholedoc_fallback")

    return items, metas, focus_samples


# ---------------- 断点续跑: 读 manifest 已完成的 doc ----------------
def load_done_docs():
    """返回 (已完成 doc 集合, 最大 chunk_idx).

    只有 shard 文件还在的 chunk 才把它的 doc 计入"已完成"; manifest 在但 shard 丢了的,
    那批 doc 会被重新处理(避免静默丢数据)。max_chunk 仍按所有可解析行取最大,
    防止 chunk_idx 复用。半截/损坏的行(断线写一半)直接跳过。"""
    done = set()
    max_chunk = -1
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                max_chunk = max(max_chunk, rec.get("chunk_idx", -1))
                if (SHARD_DIR / rec.get("shard", "")).exists():
                    done.update(rec.get("doc_ids", []))
    return done, max_chunk


# ---------------- finalize: 用 manifest + shards 重建 embeddings.npy + meta.jsonl ----------------
def finalize():
    if not MANIFEST.exists():
        print("      [finalize] 没有 manifest, 跳过")
        return 0
    chunks = []
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except Exception:
                continue
    # 同一 chunk_idx 可能因断点续跑出现多条(旧的 + 重写的); 只保留最后一条,
    # shard 同名文件已被新一次 np.save 覆盖为最新, 故"保留最后"与 shard 一致, 不会重复计入
    dedup = {}
    for c in chunks:
        dedup[c["chunk_idx"]] = c
    chunks = sorted(dedup.values(), key=lambda c: c["chunk_idx"])

    embs, metas = [], []
    row = 0
    for c in chunks:
        shard = SHARD_DIR / c["shard"]
        if not shard.exists():
            print(f"      [finalize][警告] 缺 shard {shard}, 跳过该 chunk")
            continue
        arr = np.load(shard)
        cmetas = c["metas"]
        if len(cmetas) != arr.shape[0]:
            print(f"      [finalize][警告] chunk {c['chunk_idx']} meta({len(cmetas)})"
                  f" 与 shard 行数({arr.shape[0]}) 不一致, 跳过")
            continue
        embs.append(arr)
        for m in cmetas:
            m2 = dict(m)
            m2["row"] = row
            metas.append(m2)
            row += 1

    if not embs:
        print("      [finalize] 没有可用 shard")
        return 0
    full = np.concatenate(embs, axis=0).astype(np.float32)
    np.save(EMB_OUT, full)
    with open(META_OUT, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"      [finalize] 写出 embeddings.npy {full.shape} + meta.jsonl {len(metas)} 行")
    return len(metas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default=cfg.model.device)
    ap.add_argument("--focus_mode", choices=["context_crop", "crop", "brb"], default="context_crop",
                    help="grounded prop 的焦点图: context_crop=裁框+留边距(默认) / crop=裁紧框 / brb=模糊背景(FGVP Box变体)")
    ap.add_argument("--context_margin", type=float, default=0.25,
                    help="context_crop 四周外扩比例(占框尺寸), 默认 0.25")
    ap.add_argument("--blur_sigma", type=float, default=100.0,
                    help="brb 模式的高斯模糊 σ(固定像素, 对标 FGVP=100)")
    ap.add_argument("--blur_ratio", type=float, default=0.0,
                    help="brb 模式: >0 则模糊半径=blur_ratio*长边(老行为); =0(默认)用固定 --blur_sigma")
    ap.add_argument("--out_tag", default=None,
                    help="输出子目录后缀, 默认按 focus_mode 自动命名; 不同 mode 写不同目录, 互不串断点")
    ap.add_argument("--chunk_docs", type=int, default=50,
                    help="每多少个 doc 落一次盘(断点粒度; 太大会在内存里堆太多 PIL 焦点图, Colab 易 OOM)")
    ap.add_argument("--ungrounded_mode", choices=["text", "wholeimage"], default="text",
                    help="grounded=False 的 prop 怎么编码: text=纯文本(默认) / wholeimage=prop+整图")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 个 doc(pilot)")
    ap.add_argument("--save_focus_samples", type=int, default=0,
                    help="额外存几张焦点图供肉眼抽查")
    ap.add_argument("--finalize_only", action="store_true",
                    help="跳过编码, 只用已有 shards 重建 embeddings.npy + meta.jsonl")
    args = ap.parse_args()

    # 按 focus 配置选输出目录: 不同方案互不覆盖、互不串断点, 方便做 ablation
    global OUT_DIR, SHARD_DIR, MANIFEST, EMB_OUT, META_OUT, STATS_OUT, FOCUS_SAMPLE_DIR
    tag = args.out_tag or (f"brb_s{int(args.blur_sigma)}" if args.focus_mode == "brb" else args.focus_mode)
    OUT_DIR = BENCH / f"subdoc_embeddings_{tag}"
    SHARD_DIR = OUT_DIR / "shards"
    MANIFEST = OUT_DIR / "manifest.jsonl"
    EMB_OUT = OUT_DIR / "embeddings.npy"
    META_OUT = OUT_DIR / "meta.jsonl"
    STATS_OUT = OUT_DIR / "encode_stats.json"
    FOCUS_SAMPLE_DIR = OUT_DIR / "focus_samples"
    print(f"      输出目录 = {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)

    if args.finalize_only:
        finalize()
        return

    # 1. grounding 结果按 doc 聚合
    print("[1/4] 读 grounding 结果 ...")
    ground_by_doc: dict[str, list] = {}
    if GROUND_JSONL.exists():
        with open(GROUND_JSONL, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ground_by_doc.setdefault(rec["corpus_id"], []).append(rec)
        for cid in ground_by_doc:   # 每个 doc 内按 prop_idx 排序, 顺序稳定
            ground_by_doc[cid].sort(key=lambda r: r.get("prop_idx", 0))
    print(f"      grounding 覆盖 doc 数 = {len(ground_by_doc)}")

    # 2. 加载 corpus(只建索引, 图像按需解码)
    print("[2/4] 加载 corpus.parquet ...")
    corpus = load_dataset("parquet", data_files=str(CORPUS_PARQUET), split="train")
    n_docs = len(corpus) if args.limit is None else min(len(corpus), args.limit)
    print(f"      corpus 行数 = {len(corpus)}, 本次处理 = {n_docs}")

    done_docs, max_chunk = load_done_docs()
    print(f"      已完成 doc = {len(done_docs)}(断点续跑会跳过), 下一个 chunk_idx 从 {max_chunk + 1} 开始")

    # 3. 建 encoder
    print("[3/4] 加载 GME ...")
    encoder = create_encoder(
        model_name=cfg.model.model_name,
        device=args.device,
        max_image_tokens=cfg.model.max_image_tokens,
        max_length=cfg.model.max_length,
    )

    # 4. 按 chunk_docs 分块处理
    print(f"[4/4] 分块编码(chunk_docs={args.chunk_docs}, batch={args.batch_size}, "
          f"focus_mode={args.focus_mode}, blur_sigma={args.blur_sigma}, blur_ratio={args.blur_ratio}, "
          f"context_margin={args.context_margin}, ungrounded={args.ungrounded_mode}) ...")
    chunk_idx = max_chunk + 1
    n_focus_saved = 0
    src_counter: dict[str, int] = {}
    total_new = 0
    try:
        buf_items, buf_metas, buf_docids = [], [], []

        def flush(cidx):
            """把缓冲的一组 doc 的 items 编码 + 落 shard + 写 manifest."""
            nonlocal total_new
            if not buf_items:
                return
            # 按 has_image 排序以利 encoder 批处理(返回顺序=输入顺序, 之后按 id 对齐 meta)
            order = sorted(range(len(buf_items)),
                           key=lambda i: buf_items[i]["image"] is not None)
            items_sorted = [buf_items[i] for i in order]
            metas_sorted = [buf_metas[i] for i in order]
            ids, emb = encoder.encode_items(
                items_sorted, instruction=None,
                batch_size=args.batch_size, show_progress=False,
            )
            emb = np.asarray(emb, dtype=np.float32)
            # encode_items 保序: ids[i] 对应 emb[i] 对应 metas_sorted[i]
            assert len(ids) == len(metas_sorted) == emb.shape[0], "编码返回数量不匹配"
            for mi, idd in zip(metas_sorted, ids):
                assert mi["prop_id"] == idd, f"id 错位: {mi['prop_id']} != {idd}"
            shard_name = f"shard_{cidx:05d}.npy"
            np.save(SHARD_DIR / shard_name, emb)
            with open(MANIFEST, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({
                    "chunk_idx": cidx, "shard": shard_name,
                    "doc_ids": list(dict.fromkeys(buf_docids)),
                    "n": emb.shape[0], "metas": metas_sorted,
                }, ensure_ascii=False) + "\n")
            total_new += emb.shape[0]
            for m in metas_sorted:
                src_counter[m["source"]] = src_counter.get(m["source"], 0) + 1
            print(f"      chunk {cidx}: {len(buf_docids)} docs -> {emb.shape[0]} subdocs "
                  f"(累计新增 {total_new})")

        docs_in_buf = 0
        for i in range(n_docs):
            row = corpus[i]
            cid = row["id"]
            if cid in done_docs:
                continue
            text = row.get("text") or ""
            modality = row.get("modality", "image,text")
            pil = resize_image(_ensure_pil_image(row.get("image")))
            items, metas, focus_samples = build_items_for_doc(
                cid, text, modality, pil, ground_by_doc.get(cid),
                args.focus_mode, args.blur_sigma, args.blur_ratio, args.context_margin,
                args.ungrounded_mode)

            # 抽查焦点图
            while args.save_focus_samples and n_focus_saved < args.save_focus_samples and focus_samples:
                pid, fimg = focus_samples.pop(0)
                FOCUS_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
                fimg.save(FOCUS_SAMPLE_DIR / f"{pid}.png")
                n_focus_saved += 1

            buf_items.extend(items)
            buf_metas.extend(metas)
            buf_docids.extend([cid] * len(items))
            docs_in_buf += 1

            if docs_in_buf >= args.chunk_docs:
                flush(chunk_idx)
                chunk_idx += 1
                buf_items, buf_metas, buf_docids = [], [], []
                docs_in_buf = 0
                gc.collect()   # 及时回收上一批的 PIL 焦点图, 控制宿主内存

        flush(chunk_idx)  # 收尾
    finally:
        encoder.unload()

    # 5. 重建最终文件 + 统计
    n_total = finalize()
    stats = {
        "n_subdoc_total": n_total,
        "n_new_this_run": total_new,
        "source_breakdown_this_run": src_counter,
        "focus_mode": args.focus_mode,
        "blur_sigma": args.blur_sigma,
        "blur_ratio": args.blur_ratio,
        "context_margin": args.context_margin,
        "ungrounded_mode": args.ungrounded_mode,
        "focus_samples_saved": n_focus_saved,
    }
    json.dump(stats, open(STATS_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[done]", json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
