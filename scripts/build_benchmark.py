"""构建多模态检索 benchmark 子集(基于 MRMR Knowledge + GME 全量检索结果).

规则:
  - query: MRMR query test split 全部(555 条), 只保留单张 image 列
  - qrels: 全部(query_id, corpus_id, score)
  - corpus: gold(qrels 命中) + 难负例(predictions 高分非 gold) 凑到 TARGET_SIZE;
           难负例不足则随机同领域填充。只保留单张 image 列。

输出: data/benchmark_v1/{corpus,queries,qrels}.parquet + build_stats.json
跑:  python -u -m scripts.build_benchmark
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

if os.name == "nt":
    os.environ.setdefault("HF_DATASETS_CACHE", r"D:\hf_cache\datasets")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from datasets import load_dataset  # noqa: E402

# ============ 配置 ============
TARGET_SIZE = 3000
SEED = 42
PRED_PATH = "predictions.json"           # GME 全量检索结果
OUT_DIR = Path("data/benchmark_v1")
# corpus 只保留这些列(去掉 image 1-4 / vision, 只留单张 image)
CORPUS_KEEP_COLS = ["id", "modality", "text", "image"]
# query 保留这些列(同样只留单张 image, 去掉 image 1-7)
QUERY_KEEP_COLS = ["id", "modality", "text", "image", "category", "question", "options", "answer"]


def strip_prefix(s: str) -> str:
    """predictions 里 id 带 query-test-/corpus-test- 前缀, 去掉以对齐 corpus/qrels."""
    for p in ("query-test-", "corpus-test-"):
        if s.startswith(p):
            return s[len(p):]
    return s


def main():
    random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/7] 加载 MRMR query / qrels / corpus ...")
    query_ds = load_dataset("MRMRbenchmark/knowledge", "query", split="test")
    qrels_ds = load_dataset("MRMRbenchmark/knowledge", "qrels", split="test")
    corpus_ds = load_dataset("MRMRbenchmark/knowledge", "corpus", split="test")
    print(f"      query={len(query_ds)}, qrels={len(qrels_ds)}, corpus={len(corpus_ds)}")

    # ---- gold: qrels 里 score>=1 的 corpus_id ----
    print("[2/7] 收集 gold doc ...")
    gold = set()
    for r in qrels_ds:
        if int(r["score"]) >= 1:
            gold.add(r["corpus_id"])
    print(f"      gold doc 数 = {len(gold)}")

    # ---- corpus id 集合(裸 id), 用于校验 predictions 命中 ----
    print("[3/7] 读 corpus id / modality(不解码图, 快) ...")
    corpus_ids_all = corpus_ds["id"]                  # list[str], 不触发图解码
    corpus_modality = corpus_ds["modality"]
    id_to_idx = {cid: i for i, cid in enumerate(corpus_ids_all)}
    has_image = {cid: ("image" in (m or "")) for cid, m in zip(corpus_ids_all, corpus_modality)}
    corpus_id_set = set(corpus_ids_all)

    # 校验 gold 都在 corpus 里
    gold_missing = [g for g in gold if g not in corpus_id_set]
    if gold_missing:
        print(f"      [警告] {len(gold_missing)} 个 gold 不在 corpus, 例: {gold_missing[:3]}")
    gold = {g for g in gold if g in corpus_id_set}

    # ---- 难负例: predictions 里高分非 gold doc, 按最高分聚合排序 ----
    print(f"[4/7] 从 {PRED_PATH} 挖难负例 ...")
    preds = json.load(open(PRED_PATH, encoding="utf-8"))
    print(f"      predictions query 数 = {len(preds)}")
    hardneg_best_score = defaultdict(float)   # corpus_id -> 跨 query 的最高检索分
    miss_pred = 0
    for qid, ranking in preds.items():
        for cid_raw, score in ranking.items():
            cid = strip_prefix(cid_raw)
            if cid not in corpus_id_set:
                miss_pred += 1
                continue
            if cid in gold:
                continue
            if score > hardneg_best_score[cid]:
                hardneg_best_score[cid] = score
    print(f"      难负例候选池 = {len(hardneg_best_score)} (predictions 里未命中 corpus 的条目 {miss_pred})")

    # 按分数降序, 取够填满 TARGET_SIZE 的数量
    need = TARGET_SIZE - len(gold)
    hardneg_sorted = sorted(hardneg_best_score, key=lambda c: hardneg_best_score[c], reverse=True)
    hardneg_pick = hardneg_sorted[:need]
    print(f"      需要 {need} 个负例, 难负例池给了 {min(need, len(hardneg_sorted))} 个")

    # ---- 不足则随机同领域填充(优先带图) ----
    corpus_pick = set(gold) | set(hardneg_pick)
    if len(corpus_pick) < TARGET_SIZE:
        gap = TARGET_SIZE - len(corpus_pick)
        print(f"[5/7] 难负例不足, 随机填充 {gap} 个 ...")
        # 候选: 不在已选集合里的; 优先带图
        remaining = [c for c in corpus_ids_all if c not in corpus_pick]
        remaining.sort(key=lambda c: (not has_image[c], random.random()))  # 带图优先, 其余随机
        for c in remaining[:gap]:
            corpus_pick.add(c)
    else:
        print("[5/7] 难负例充足, 无需随机填充")
    print(f"      最终 corpus 选中 = {len(corpus_pick)}")

    # ---- 物化 3 个子集并写出 ----
    print("[6/7] 物化子集并写 parquet ...")
    # corpus: select 选中行 + 只留单图列
    keep_idx = [id_to_idx[c] for c in corpus_pick]
    corpus_sub = corpus_ds.select(keep_idx)
    drop_cols = [c for c in corpus_sub.column_names if c not in CORPUS_KEEP_COLS]
    corpus_sub = corpus_sub.remove_columns(drop_cols)

    # query: 全部 + 只留单图列
    q_drop = [c for c in query_ds.column_names if c not in QUERY_KEEP_COLS]
    query_sub = query_ds.remove_columns(q_drop)

    # qrels: 过滤掉引用不在最终 corpus 的条目
    # (MRMR 本身有个别 gold 不在 corpus, 要去掉保证 benchmark 自洽)
    n_qrels_before = len(qrels_ds)
    qrels_sub = qrels_ds.filter(lambda r: r["corpus_id"] in corpus_pick)
    n_qrels_dropped = n_qrels_before - len(qrels_sub)
    # 检查有没有 query 丢光所有 gold
    q_gold_count = defaultdict(int)
    for r in qrels_sub:
        q_gold_count[r["query_id"]] += 1
    all_qids = set(query_ds["id"])
    q_lost_all = [q for q in all_qids if q_gold_count[q] == 0]
    print(f"      qrels 过滤: 丢 {n_qrels_dropped} 条孤儿; "
          f"丢光所有 gold 的 query: {len(q_lost_all)} {q_lost_all[:3]}")

    corpus_sub.to_parquet(str(OUT_DIR / "corpus.parquet"))
    query_sub.to_parquet(str(OUT_DIR / "queries.parquet"))
    qrels_sub.to_parquet(str(OUT_DIR / "qrels.parquet"))
    print(f"      写出到 {OUT_DIR}")

    # ---- 统计 + 自查 ----
    print("[7/7] 统计 + 自查 ...")
    # 领域分布(从 corpus id 解析: test_/validation_ 后第一段大写词)
    def domain_of(cid: str) -> str:
        body = cid.replace("test_", "").replace("validation_", "")
        # 取领域关键词(子领域名), 简单按已知大领域子串归类
        for dom, keys in {
            "Medicine": ["Clinical_Medicine", "Basic_Medical_Science",
                         "Diagnostics_and_Laboratory_Medicine", "Pharmacy"],
            "Science": ["Biology", "Chemistry", "Agriculture", "Geography",
                        "Physics", "Math", "Materials"],
            "Art": ["Art_Theory", "Design", "Music", "Art"],
            "Humanities": ["History", "Psychology", "Sociology", "Literature"],
        }.items():
            if any(k in body for k in keys):
                return dom
        return "Other"

    dom_count = defaultdict(int)
    for c in corpus_pick:
        dom_count[domain_of(c)] += 1

    n_with_img = sum(1 for c in corpus_pick if has_image[c])
    gold_in = sum(1 for g in gold if g in corpus_pick)

    stats = {
        "target_size": TARGET_SIZE,
        "final_corpus_size": len(corpus_pick),
        "n_gold": len(gold),
        "gold_all_in_corpus": gold_in == len(gold),
        "n_hardneg_picked": len(set(hardneg_pick) & corpus_pick),
        "n_random_fill": len(corpus_pick) - len(gold) - len(set(hardneg_pick) & corpus_pick),
        "n_corpus_with_image": n_with_img,
        "pct_corpus_with_image": round(n_with_img / len(corpus_pick), 3),
        "domain_distribution": dict(dom_count),
        "n_queries": len(query_sub),
        "n_qrels": len(qrels_sub),
        "n_qrels_dropped_orphan": n_qrels_dropped,
        "n_queries_lost_all_gold": len(q_lost_all),
    }
    json.dump(stats, open(OUT_DIR / "build_stats.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    # 自查断言
    assert stats["gold_all_in_corpus"], "有 gold 没进 corpus!"
    assert stats["final_corpus_size"] == TARGET_SIZE, "corpus 数量不等于目标!"
    assert stats["n_queries"] == 555, "query 数异常!"
    print("\n[自查通过] gold 全覆盖 / corpus=3000 / query=555")


if __name__ == "__main__":
    main()
