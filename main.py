"""
MM-MixGR: Multi-Modal Mixed-Granularity Retrieval

Entry point for all experiments.

Usage:
  python main.py baseline                         # full baseline
  python main.py baseline --max_corpus 500         # quick test
  python main.py baseline --domains Science        # single domain
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Set

import numpy as np
from PIL import Image

from config import cfg

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# Data loading
# ============================================================

DOMAIN_PREFIXES = {
    "Science": ["agriculture", "geography", "chemistry", "biology"],
    "Medicine": ["diagnostics", "clinical_medicine", "basic_medical", "pharmacy"],
    "Art": ["music", "design", "art_theory", "art"],
    "Humanities": ["history", "sociology", "psychology", "literature"],
}


def infer_domain(qid: str) -> str:
    for domain, prefixes in DOMAIN_PREFIXES.items():
        for p in prefixes:
            if p in qid.lower():
                return domain
    return "Unknown"


def load_queries(domains):
    from datasets import load_dataset

    target_prefixes = set()
    for d in domains:
        if d in DOMAIN_PREFIXES:
            target_prefixes.update(DOMAIN_PREFIXES[d])

    logger.info(f"Loading queries for {domains}...")
    ds = load_dataset("MRMRbenchmark/knowledge", "query", split="test")

    queries = {}
    for row in ds:
        qid = row["id"]
        if target_prefixes and not any(p in qid.lower() for p in target_prefixes):
            continue

        image = None
        for key in ["image", "image 1", "image 2"]:
            if row.get(key) is not None and isinstance(row[key], Image.Image):
                image = row[key].convert("RGB")
                break

        queries[qid] = {
            "text": row.get("text", ""),
            "image": image,
            "instruction": row.get("instruction",
                "Retrieve relevant documents that help answer the question."),
        }

    logger.info(f"  {len(queries)} queries loaded")
    return queries


def load_qrels(query_ids: Set[str]):
    from datasets import load_dataset

    logger.info("Loading qrels...")
    ds = load_dataset("MRMRbenchmark/knowledge", "qrels", split="test")

    qrels = {}
    for row in ds:
        qid = str(row["query_id"])
        if qid not in query_ids:
            continue
        did = str(row["corpus_id"])
        score = int(row.get("score", 1))
        qrels.setdefault(qid, {})[did] = score

    logger.info(f"  {len(qrels)} queries with relevance judgments")
    return qrels


def load_corpus(max_docs=None, load_images=True):
    """加载 corpus 文本元数据，图片在编码时逐条加载。"""
    from datasets import load_dataset

    logger.info(f"Loading corpus metadata...")
    ds = load_dataset("MRMRbenchmark/knowledge", "corpus", split="test")

    corpus = {}
    total = min(len(ds), max_docs) if max_docs else len(ds)
    for i in range(total):
        row = ds[i]
        corpus[row["id"]] = {
            "text": row.get("text", ""),
            "image": None,
        }

        if (i + 1) % 5000 == 0:
            logger.info(f"  {i+1} docs loaded...")

    logger.info(f"  {len(corpus)} corpus docs loaded (text only, images loaded during encoding)")
    return corpus


def encode_corpus_streaming(encoder, cache_path, max_docs=None, _interim_eval=None):
    """
    边加载 corpus 图片边编码边释放，一次只有一张图在内存里。
    _interim_eval: (query_ids, query_embs, qrels) 用于每 2000 条打印中间 nDCG@10。
    """
    import time
    from datasets import load_dataset

    if cache_path and cache_path.exists():
        logger.info(f"Loading cached corpus embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["ids"].tolist(), data["embeddings"]

    encoder.load()

    logger.info("Encoding corpus (from local cache, with images)...")
    from datasets import load_dataset
    ds = load_dataset("MRMRbenchmark/knowledge", "corpus", split="test")

    ids = []
    embeddings = []
    t0 = time.time()

    # 检查有没有中间 checkpoint 可以恢复
    skip_count = 0
    if cache_path:
        checkpoints = sorted(cache_path.parent.glob("corpus_checkpoint_*.npz"))
        if checkpoints:
            latest = checkpoints[-1]
            data = np.load(latest, allow_pickle=True)
            ids = data["ids"].tolist()
            embeddings = list(data["embeddings"])
            skip_count = len(ids)
            logger.info(f"  Resuming from checkpoint: {latest.name} ({skip_count} docs done)")

    total = min(len(ds), max_docs) if max_docs else len(ds)

    for i in range(total):
        if i < skip_count:
            if (i + 1) % 5000 == 0:
                logger.info(f"  Skipping {i+1}/{skip_count}...")
            continue

        row = ds[i]

        # 逐条加载图片
        image = None
        try:
            for key in ["image", "image 1", "image 2", "vision"]:
                if row.get(key) is not None and isinstance(row[key], Image.Image):
                    image = row[key].convert("RGB")
                    break
        except Exception as e:
            logger.warning(f"  Failed to load image for doc {row.get('id', i)}: {e}")
            image = None

        text = row.get("text", "")
        emb = encoder.encode(text=text, image=image)

        ids.append(row["id"])
        embeddings.append(emb)

        # 立即释放图片
        del image

        if len(ids) % 50 == 0:
            elapsed = time.time() - t0
            speed = (len(ids) - skip_count) / max(elapsed, 1)
            remaining = (total - len(ids)) / max(speed, 0.1)
            logger.info(f"  [corpus] {len(ids)}/{total} "
                        f"({speed:.1f}/s, ~{remaining/60:.1f}min left)")

        # 每 2000 条保存一次中间结果，并跑一次评估
        if len(ids) % 2000 == 0 and cache_path:
            tmp_embs = np.stack(embeddings)
            tmp_path = cache_path.parent / f"corpus_checkpoint_{len(ids)}.npz"
            np.savez(tmp_path, ids=np.array(ids), embeddings=tmp_embs)
            logger.info(f"  Checkpoint saved: {tmp_path} ({len(ids)} docs)")

            if _interim_eval:
                qids, qembs, qrels_data = _interim_eval
                from retrieval.granular import retrieve as _retrieve
                from evaluation.metrics import compute_ndcg
                tmp_results = _retrieve(qids, qembs, ids, tmp_embs, top_k=10)
                ndcg10 = compute_ndcg(qrels_data, tmp_results, k=10)
                all_rel = set()
                for rels in qrels_data.values():
                    all_rel.update(rels.keys())
                covered = len(all_rel & set(ids))
                logger.info(f"  >> Interim nDCG@10 = {ndcg10:.4f} "
                            f"(corpus: {len(ids)}, relevant docs covered: {covered}/{len(all_rel)})")

    embeddings = np.stack(embeddings)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, ids=np.array(ids), embeddings=embeddings)
        logger.info(f"  Cached to {cache_path}")

    elapsed = time.time() - t0
    logger.info(f"  Encoded {len(ids)} corpus docs in {elapsed/60:.1f}min")

    return ids, embeddings


# ============================================================
# Baseline experiment
# ============================================================

def run_baseline(args):
    """
    Baseline: single granularity (query+image → doc).
    """
    from embeddings.visual_encoder import create_encoder
    from retrieval.granular import retrieve
    from evaluation.metrics import evaluate_all, evaluate_by_domain

    cache_dir = cfg.paths.embedding_cache_dir
    suffix = f"_top{args.max_corpus}" if args.max_corpus else ""

    # 从模型名生成短标签用于文件名
    # "Qwen/Qwen3-VL-Embedding-2B" → "qwen3-vl-embedding-2b"
    model_tag = args.model.split("/")[-1].lower()

    print("=" * 60)
    print("  MM-MixGR Baseline")
    print(f"  Model: {args.model}")
    print(f"  Tag: {model_tag}")
    print(f"  Domains: {args.domains}")
    if args.max_corpus:
        print(f"  Corpus limit: {args.max_corpus}")
    print("=" * 60)

    # Load data
    queries = load_queries(args.domains)
    qrels = load_qrels(set(queries.keys()))

    # Load corpus text metadata (no images in memory)
    corpus = load_corpus(max_docs=args.max_corpus)

    # Check coverage
    all_relevant = set()
    for rels in qrels.values():
        all_relevant.update(rels.keys())
    covered = all_relevant & set(corpus.keys())
    print(f"\n  Relevant docs in corpus: {len(covered)}/{len(all_relevant)}")

    # Encode
    encoder = create_encoder(
        model_name=args.model,
        device=cfg.model.device,
    )

    query_ids, query_embs = encoder.encode_batch(
        queries,
        cache_path=cache_dir / f"{model_tag}_query_{'_'.join(args.domains)}{suffix}.npz",
        item_type="query",
        use_instruction=True,
    )

    # Corpus: 边加载图片边编码边释放，不占内存
    corpus_ids, corpus_embs = encode_corpus_streaming(
        encoder,
        cache_path=cache_dir / f"{model_tag}_corpus{suffix}.npz",
        max_docs=args.max_corpus,
        _interim_eval=(query_ids, query_embs, qrels),
    )

    # Free GPU memory
    encoder.unload()

    # Retrieve
    results = retrieve(query_ids, query_embs, corpus_ids, corpus_embs, top_k=100)

    # Evaluate
    query_domains = {qid: infer_domain(qid) for qid in queries}
    all_metrics = evaluate_by_domain(qrels, results, query_domains)

    # Print
    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")

    for domain, metrics in all_metrics.items():
        n = sum(1 for qid, d in query_domains.items() if d == domain and qid in qrels) if domain != "All" else len(qrels)
        print(f"\n  {domain} ({n} queries):")
        for name, val in metrics.items():
            print(f"    {name:15s}: {val:.4f}")

    print(f"\n  --- MRMR paper reference (nDCG@10) ---")
    print(f"  Qwen3-Embedding-8B (T2T):      Sci 72.5, Med 53.2")
    print(f"  Ops-MM-Embedding-7B (IT2IT):    Sci 70.0, Med 52.5")
    print(f"  GME-Qwen2-VL-7B (IT2IT):        Sci 46.8, Med 40.1")
    print(f"{'='*60}")

    # Save
    results_path = cfg.paths.results_dir / f"baseline_{model_tag}{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n  Saved to {results_path}")


# ============================================================
# MixGR multi-granularity experiment
# ============================================================

def encode_subqueries(encoder, queries, decomposer, cache_path, model_tag):
    """
    Decompose queries into subqueries and encode each (subquery + query_image).

    Returns:
        sq_data: {qid: [(subquery_text, embedding), ...]}
    """
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached subquery embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return dict(data["sq_data"].item())

    encoder.load()

    sq_data = {}
    total_sqs = 0

    items = list(queries.items())
    for i, (qid, query) in enumerate(items):
        subqueries = decomposer.decompose_query(query["text"])
        image = query.get("image")
        instruction = query.get("instruction", "")

        sq_embeddings = []
        for sq in subqueries:
            emb = encoder.encode(text=sq, image=image, instruction=instruction)
            sq_embeddings.append((sq, emb))

        sq_data[qid] = sq_embeddings
        total_sqs += len(subqueries)

        if (i + 1) % 50 == 0:
            logger.info(f"  [subquery] {i+1}/{len(items)} queries, {total_sqs} subqueries total")

    logger.info(f"  Encoded {total_sqs} subqueries for {len(sq_data)} queries "
                f"(avg {total_sqs/max(len(sq_data),1):.1f})")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, sq_data=sq_data)
        logger.info(f"  Cached to {cache_path}")

    return sq_data


def encode_propositions_streaming(encoder, decomposer, cache_path, max_docs=None):
    """
    Decompose corpus docs into propositions and encode each (proposition + doc_image).
    Streaming: one image in memory at a time.

    Returns:
        prop_data: {did: [(proposition_text, embedding), ...]}
        doc_ids: list of all doc IDs in order
    """
    import time
    from datasets import load_dataset

    if cache_path and cache_path.exists():
        logger.info(f"Loading cached proposition embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return dict(data["prop_data"].item()), data["doc_ids"].tolist()

    encoder.load()

    logger.info("Encoding propositions (streaming, with images)...")
    ds = load_dataset("MRMRbenchmark/knowledge", "corpus", split="test")

    prop_data = {}
    doc_ids = []
    total_props = 0
    t0 = time.time()

    total = min(len(ds), max_docs) if max_docs else len(ds)

    for i in range(total):
        row = ds[i]
        did = row["id"]
        text = row.get("text", "")

        # Load image
        image = None
        try:
            for key in ["image", "image 1", "image 2", "vision"]:
                if row.get(key) is not None and isinstance(row[key], Image.Image):
                    image = row[key].convert("RGB")
                    break
        except Exception:
            image = None

        # Decompose into propositions
        propositions = decomposer.decompose_document(text)

        # Encode each (proposition + image)
        prop_embeddings = []
        for prop in propositions:
            emb = encoder.encode(text=prop, image=image)
            prop_embeddings.append((prop, emb))

        prop_data[did] = prop_embeddings
        doc_ids.append(did)
        total_props += len(propositions)

        del image

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            remaining = (total - i - 1) / max(speed, 0.1)
            logger.info(f"  [proposition] {i+1}/{total} docs, {total_props} props "
                        f"({speed:.1f} docs/s, ~{remaining/60:.1f}min left)")

        if (i + 1) % 2000 == 0 and cache_path:
            tmp_path = cache_path.parent / f"prop_checkpoint_{i+1}.npz"
            np.savez(tmp_path, prop_data=prop_data, doc_ids=np.array(doc_ids))
            logger.info(f"  Checkpoint saved: {tmp_path}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, prop_data=prop_data, doc_ids=np.array(doc_ids))
        logger.info(f"  Cached to {cache_path}")

    elapsed = time.time() - t0
    logger.info(f"  Encoded {total_props} propositions for {len(doc_ids)} docs "
                f"(avg {total_props/max(len(doc_ids),1):.1f}) in {elapsed/60:.1f}min")

    return prop_data, doc_ids


def score_mm_d(sq_data, corpus_ids, corpus_embeddings):
    """
    mm-d granularity: score(q, d) = max_i sim(sq_i, doc_d)
    subquery embeddings vs full document embeddings.
    """
    import torch

    c_tensor = torch.tensor(corpus_embeddings, dtype=torch.float32)
    results = {}

    for qid, sq_list in sq_data.items():
        # Stack all subquery embeddings: (M, dim)
        sq_embs = torch.tensor(np.stack([emb for _, emb in sq_list]), dtype=torch.float32)

        # sim: (M, N_corpus)
        sims = torch.mm(sq_embs, c_tensor.T)

        # max over subqueries: (N_corpus,)
        max_sims = sims.max(dim=0).values

        top_k = min(100, len(corpus_ids))
        top_scores, top_indices = torch.topk(max_sims, top_k)
        results[qid] = [(corpus_ids[idx.item()], score.item())
                        for score, idx in zip(top_scores, top_indices)]

    return results


def score_mm_p(sq_data, prop_data, doc_ids, mode="max"):
    """
    mm-p / ms-p granularity.
    mode="max": score(q,d) = max_i max_j sim(sq_i, prop_j)
    mode="mean": score(q,d) = (1/M) Σ_i max_j sim(sq_i, prop_j)
    """
    import torch

    results = {}

    for qid, sq_list in sq_data.items():
        sq_embs = torch.tensor(np.stack([emb for _, emb in sq_list]), dtype=torch.float32)
        M = sq_embs.shape[0]

        doc_scores = {}

        for did in doc_ids:
            if did not in prop_data or not prop_data[did]:
                continue

            prop_embs = torch.tensor(
                np.stack([emb for _, emb in prop_data[did]]), dtype=torch.float32
            )

            # sim: (M, N_props)
            sims = torch.mm(sq_embs, prop_embs.T)

            # max over propositions for each subquery: (M,)
            max_per_sq = sims.max(dim=1).values

            if mode == "max":
                doc_scores[did] = max_per_sq.max().item()
            else:  # mean
                doc_scores[did] = max_per_sq.mean().item()

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results[qid] = sorted_docs[:100]

    return results


def run_mixgr(args):
    """
    MixGR multi-granularity retrieval.
    Three granularities + RRF fusion.
    """
    from embeddings.visual_encoder import create_encoder
    from retrieval.granular import retrieve
    from retrieval.fusion import reciprocal_rank_fusion
    from evaluation.metrics import evaluate_all, evaluate_by_domain
    from decomposition.text_decompose import TextDecomposer

    cache_dir = cfg.paths.embedding_cache_dir
    model_tag = args.model.split("/")[-1].lower()
    suffix = f"_top{args.max_corpus}" if args.max_corpus else ""

    print("=" * 60)
    print("  MM-MixGR Multi-Granularity Retrieval")
    print(f"  Model: {args.model}")
    print(f"  Domains: {args.domains}")
    if args.max_corpus:
        print(f"  Corpus limit: {args.max_corpus}")
    print("=" * 60)

    # Step 1: Load data
    print("\n[Step 1] Loading data...")
    queries = load_queries(args.domains)
    qrels = load_qrels(set(queries.keys()))
    corpus = load_corpus(max_docs=args.max_corpus)

    all_relevant = set()
    for rels in qrels.values():
        all_relevant.update(rels.keys())
    covered = all_relevant & set(corpus.keys())
    print(f"  Relevant docs in corpus: {len(covered)}/{len(all_relevant)}")

    # Step 2: Decompose
    print("\n[Step 2] Decomposing queries and corpus...")
    decomposer = TextDecomposer(device="cuda")
    decomposer.load_model()

    # Step 3: Load encoder
    print("\n[Step 3] Loading encoder...")
    encoder = create_encoder(model_name=args.model, device=cfg.model.device)

    # Free decomposer GPU memory if needed
    decomposer._model = decomposer._model.cpu()
    import torch
    torch.cuda.empty_cache()

    # Step 4: Encode subqueries
    print("\n[Step 4] Encoding subqueries...")
    sq_data = encode_subqueries(
        encoder, queries, decomposer,
        cache_path=cache_dir / f"{model_tag}_subqueries{suffix}.npz",
        model_tag=model_tag,
    )

    # Step 5: Encode full corpus (reuse baseline cache if exists)
    print("\n[Step 5] Encoding full corpus (doc-level)...")
    corpus_ids, corpus_embs = encode_corpus_streaming(
        encoder,
        cache_path=cache_dir / f"{model_tag}_corpus{suffix}.npz",
        max_docs=args.max_corpus,
    )

    # Step 6: Encode propositions
    print("\n[Step 6] Encoding propositions...")
    prop_data, prop_doc_ids = encode_propositions_streaming(
        encoder, decomposer,
        cache_path=cache_dir / f"{model_tag}_propositions{suffix}.npz",
        max_docs=args.max_corpus,
    )

    encoder.unload()

    # Step 7: Score at each granularity
    print("\n[Step 7] Scoring at each granularity...")

    print("  mm-d: subquery ↔ full doc (max)...")
    results_mm_d = score_mm_d(sq_data, corpus_ids, corpus_embs)

    print("  mm-p: subquery ↔ proposition (max-max)...")
    results_mm_p = score_mm_p(sq_data, prop_data, prop_doc_ids, mode="max")

    print("  ms-p: subquery ↔ proposition (mean-max)...")
    results_ms_p = score_mm_p(sq_data, prop_data, prop_doc_ids, mode="mean")

    # Step 8: RRF fusion
    print("\n[Step 8] RRF fusion...")
    rankings = {
        "mm-d": results_mm_d,
        "mm-p": results_mm_p,
        "ms-p": results_ms_p,
    }
    fused_results = reciprocal_rank_fusion(rankings, k=60)

    # Step 9: Evaluate
    print("\n[Step 9] Evaluating...")
    query_domains = {qid: infer_domain(qid) for qid in queries}

    # Evaluate each granularity individually
    individual_metrics = {}
    for name, results in rankings.items():
        individual_metrics[name] = evaluate_by_domain(qrels, results, query_domains)

    # Evaluate fused
    fused_metrics = evaluate_by_domain(qrels, fused_results, query_domains)

    # Print results
    print(f"\n{'='*60}")
    print(f"  Results: {model_tag}")
    print(f"{'='*60}")

    for gran_name, metrics_by_domain in individual_metrics.items():
        print(f"\n  --- {gran_name} ---")
        for domain, metrics in metrics_by_domain.items():
            print(f"  {domain}: nDCG@10={metrics.get('nDCG@10', 0):.4f}")

    print(f"\n  --- RRF Fused ---")
    for domain, metrics in fused_metrics.items():
        n = sum(1 for qid, d in query_domains.items() if d == domain and qid in qrels) if domain != "All" else len(qrels)
        print(f"\n  {domain} ({n} queries):")
        for name, val in metrics.items():
            print(f"    {name:15s}: {val:.4f}")

    print(f"\n  --- Baseline reference ---")
    baseline_path = cfg.paths.results_dir / f"baseline_{model_tag}{suffix}.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        for domain, metrics in baseline.items():
            ndcg = metrics.get("nDCG@10", 0)
            print(f"  {domain}: nDCG@10={ndcg:.4f}")
    else:
        print(f"  (no baseline found at {baseline_path})")

    print(f"{'='*60}")

    # Save
    all_results = {
        "individual": {k: v for k, v in individual_metrics.items()},
        "fused": fused_metrics,
    }
    results_path = cfg.paths.results_dir / f"mixgr_{model_tag}{suffix}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {results_path}")


# ============================================================
# Decompose only (propositionizer)
# ============================================================

def run_decompose(args):
    """
    Run propositionizer on queries and corpus. Cache results for later use.
    No encoding needed — just text decomposition.
    """
    from decomposition.text_decompose import TextDecomposer
    import time

    print("=" * 60)
    print("  Propositionizer Decomposition")
    print(f"  Domains: {args.domains}")
    if args.max_corpus:
        print(f"  Corpus limit: {args.max_corpus}")
    print("=" * 60)

    # Load data
    queries = load_queries(args.domains)
    corpus = load_corpus(max_docs=args.max_corpus)

    # Init decomposer
    decomposer = TextDecomposer(device="cuda")
    decomposer.load_model()

    # Decompose queries
    print(f"\n[Step 1] Decomposing {len(queries)} queries...")
    t0 = time.time()
    query_results = {}
    for i, (qid, query) in enumerate(queries.items()):
        subqueries = decomposer.decompose_query(query["text"])
        query_results[qid] = subqueries
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(queries)} queries done")

    n_sq = sum(len(v) for v in query_results.values())
    elapsed = time.time() - t0
    print(f"  {len(queries)} queries → {n_sq} subqueries "
          f"(avg {n_sq/max(len(queries),1):.1f}) in {elapsed:.1f}s")

    # Decompose corpus
    print(f"\n[Step 2] Decomposing {len(corpus)} corpus docs...")
    t0 = time.time()
    corpus_results = {}
    for i, (did, doc) in enumerate(corpus.items()):
        propositions = decomposer.decompose_document(doc["text"])
        corpus_results[did] = propositions
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            remaining = (len(corpus) - i - 1) / max(speed, 0.1)
            print(f"  {i+1}/{len(corpus)} docs "
                  f"({speed:.1f}/s, ~{remaining/60:.1f}min left)")

    n_props = sum(len(v) for v in corpus_results.values())
    elapsed = time.time() - t0
    print(f"  {len(corpus)} docs → {n_props} propositions "
          f"(avg {n_props/max(len(corpus),1):.1f}) in {elapsed/60:.1f}min")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Done! All results cached in {decomposer.cache_dir}")
    print(f"  {len(list(decomposer.cache_dir.glob('*.json')))} cache files total")
    print(f"{'='*60}")


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MM-MixGR experiments")
    subparsers = parser.add_subparsers(dest="command")

    # baseline
    p_baseline = subparsers.add_parser("baseline", help="Run baseline retrieval")
    p_baseline.add_argument("--model", type=str,
                            default="Qwen/Qwen3-VL-Embedding-2B",
                            help="Model name or path")
    p_baseline.add_argument("--domains", nargs="+",
                            default=["Science", "Medicine"],
                            help="Domains to evaluate")
    p_baseline.add_argument("--max_corpus", type=int, default=None,
                            help="Limit corpus size for quick testing")

    # mixgr
    p_mixgr = subparsers.add_parser("mixgr", help="Run MixGR multi-granularity retrieval")
    p_mixgr.add_argument("--model", type=str,
                         default="Qwen/Qwen3-VL-Embedding-2B",
                         help="Model name or path")
    p_mixgr.add_argument("--domains", nargs="+",
                         default=["Science", "Medicine"],
                         help="Domains to evaluate")
    p_mixgr.add_argument("--max_corpus", type=int, default=None,
                         help="Limit corpus size for quick testing")

    # decompose only (no encoding)
    p_decompose = subparsers.add_parser("decompose", help="Run propositionizer only, cache results")
    p_decompose.add_argument("--domains", nargs="+",
                             default=["Science", "Medicine"],
                             help="Domains to evaluate")
    p_decompose.add_argument("--max_corpus", type=int, default=None,
                             help="Limit corpus size")

    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline(args)
    elif args.command == "mixgr":
        run_mixgr(args)
    elif args.command == "decompose":
        run_decompose(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()