"""在 3000 子集上跑 GME 整-doc baseline(不分解 doc).

复用现有 pipeline 组件(data.loader 的 item 构造 + visual_encoder + retrieve + metrics),
只把数据源从全量 MRMR 换成 data/benchmark_v1 的三个 parquet。

需要 GPU(GME-7B ~15GB). 在 Colab 跑(先把 data/benchmark_v1 传到 Drive),或本地有 CUDA 跑。

跑: python -u -m scripts.baseline_subset
"""
import argparse
import logging
from pathlib import Path

from datasets import load_dataset

from config import cfg
from data.loader import _corpus_to_item, _query_to_item
from embeddings.cache import dump_json
from embeddings.visual_encoder import create_encoder
from evaluation.metrics import evaluate_mrmr
from retrieval.granular import retrieve

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SPLIT = "test"
BENCH = Path("data/benchmark_v1")
OUT_DIR = Path("results/baseline_subset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_size", type=int, default=cfg.model.batch_size,
                    help="GME 编码 batch(显存大就调大, 如 16/32)")
    ap.add_argument("--device", default=cfg.model.device)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 从 parquet 读三件套, 用 loader 的 item 构造器转成 pipeline 要的格式
    #    (item id 会加 query-test-/corpus-test- 前缀, 跟 qrels 对齐)
    print("[1/4] 加载 benchmark_v1 三个 parquet ...")
    q_rows = load_dataset("parquet", data_files=str(BENCH / "queries.parquet"), split="train")
    c_rows = load_dataset("parquet", data_files=str(BENCH / "corpus.parquet"), split="train")
    qrels_rows = load_dataset("parquet", data_files=str(BENCH / "qrels.parquet"), split="train")

    queries = [_query_to_item(r, SPLIT) for r in q_rows]
    corpus_items = [_corpus_to_item(r, SPLIT) for r in c_rows]
    # qrels: {query-test-<qid>: {corpus-test-<cid>: score}}
    qrels = {}
    for r in qrels_rows:
        qid = f"query-{SPLIT}-{r['query_id']}"
        did = f"corpus-{SPLIT}-{r['corpus_id']}"
        qrels.setdefault(qid, {})[did] = int(r["score"])
    # 只保留有 qrels 的 query(子集里应该全有)
    queries = [q for q in queries if q["id"] in qrels]
    print(f"      query={len(queries)}, corpus={len(corpus_items)}, qrels query={len(qrels)}")

    # 2. GME 编码(整 query / 整 doc, 不分解)
    print("[2/4] GME 编码 query + corpus ...")
    encoder = create_encoder(
        model_name=cfg.model.model_name,
        device=args.device,
        max_image_tokens=cfg.model.max_image_tokens,
        max_length=cfg.model.max_length,
    )
    try:
        query_ids, query_emb = encoder.encode_items(
            queries, instruction=cfg.data.query_instruction,
            batch_size=args.batch_size, desc="Encoding queries",
        )
        corpus_ids, corpus_emb = encoder.encode_items(
            corpus_items, instruction=None,
            batch_size=args.batch_size, desc="Encoding corpus",
        )
    finally:
        encoder.unload()

    # 3. 检索(在 3000 子集内排名)+ 评测
    print("[3/4] 检索 + 评测 ...")
    predictions = retrieve(
        query_ids, query_emb, corpus_ids, corpus_emb,
        top_k=cfg.retrieval.top_k, corpus_chunk_size=cfg.retrieval.corpus_chunk_size,
    )
    metrics = evaluate_mrmr(qrels, predictions, queries, cfg.eval.k_values)

    # 4. 保存 + 打印
    dump_json(OUT_DIR / "predictions.json", predictions)
    dump_json(OUT_DIR / "metrics.json", metrics)
    print("[4/4] 完成. baseline(3000 子集, 不分解 doc):")
    overall = metrics["overall"]
    print(f"  nDCG@10 = {overall.get('nDCG@10'):.4f}  "
          f"Recall@10 = {overall.get('Recall@10'):.4f}  "
          f"queries = {overall.get('num_queries')}")
    for dom, m in metrics["coarse"].items():
        print(f"  {dom:<10} nDCG@10={m.get('nDCG@10'):.4f}  queries={m.get('num_queries')}")
    print(f"  结果存于 {OUT_DIR}")


if __name__ == "__main__":
    main()
