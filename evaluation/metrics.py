"""
Evaluation metrics for retrieval.

Supports: nDCG@k, Recall@k
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_ndcg(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, List[Tuple[str, float]]],
    k: int = 10,
) -> float:
    """
    Compute nDCG@k.

    Args:
        qrels: {query_id: {doc_id: relevance}}
        results: {query_id: [(doc_id, score), ...]} sorted by score desc
        k: cutoff
    """
    scores = []

    for qid in qrels:
        if qid not in results:
            scores.append(0.0)
            continue

        dcg = 0.0
        for i, (did, _) in enumerate(results[qid][:k]):
            rel = qrels[qid].get(did, 0)
            dcg += rel / np.log2(i + 2)

        ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
        idcg = sum(r / np.log2(i + 2) for i, r in enumerate(ideal_rels))

        scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(scores))


def compute_recall(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, List[Tuple[str, float]]],
    k: int = 10,
) -> float:
    """
    Compute Recall@k.

    Args:
        qrels: {query_id: {doc_id: relevance}}
        results: {query_id: [(doc_id, score), ...]} sorted by score desc
        k: cutoff
    """
    scores = []

    for qid in qrels:
        if qid not in results:
            scores.append(0.0)
            continue

        retrieved = set(did for did, _ in results[qid][:k])
        relevant = set(did for did, rel in qrels[qid].items() if rel > 0)

        if len(relevant) > 0:
            scores.append(len(retrieved & relevant) / len(relevant))
        else:
            scores.append(0.0)

    return float(np.mean(scores))


def evaluate_all(qrels, results, ks=None):
    """
    Compute all metrics.

    Returns:
        dict: {"nDCG@5": ..., "nDCG@10": ..., "Recall@10": ..., etc.}
    """
    if ks is None:
        ks = {"ndcg": [5, 10, 20], "recall": [10, 100]}

    metrics = {}
    for k in ks.get("ndcg", [5, 10, 20]):
        metrics[f"nDCG@{k}"] = compute_ndcg(qrels, results, k=k)
    for k in ks.get("recall", [10, 100]):
        metrics[f"Recall@{k}"] = compute_recall(qrels, results, k=k)

    return metrics


def evaluate_by_domain(qrels, results, query_domains, ks=None):
    """
    Compute metrics broken down by domain.

    Args:
        query_domains: {query_id: domain_name}

    Returns:
        dict: {"Science": {"nDCG@10": ...}, "Medicine": {...}, "All": {...}}
    """
    domains = {}
    for qid, domain in query_domains.items():
        domains.setdefault(domain, set()).add(qid)

    all_metrics = {"All": evaluate_all(qrels, results, ks)}

    for domain, domain_qids in sorted(domains.items()):
        domain_qrels = {qid: qrels[qid] for qid in domain_qids if qid in qrels}
        if domain_qrels:
            all_metrics[domain] = evaluate_all(domain_qrels, results, ks)

    return all_metrics