"""MRMR-compatible retrieval evaluation based on pytrec_eval."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable

import pytrec_eval


def _ensure_all_queries_present(qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]]):
    merged = {qid: dict(results.get(qid, {})) for qid in qrels}
    return merged


def _per_query_scores(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    k_values: Iterable[int],
):
    safe_results = _ensure_all_queries_present(qrels, results)
    k_values = list(k_values)
    map_string = "map_cut." + ",".join(str(k) for k in k_values)
    ndcg_string = "ndcg_cut." + ",".join(str(k) for k in k_values)
    recall_string = "recall." + ",".join(str(k) for k in k_values)
    precision_string = "P." + ",".join(str(k) for k in k_values)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {map_string, ndcg_string, recall_string, precision_string},
    )
    return evaluator.evaluate(safe_results)


def _aggregate(scores: dict[str, dict[str, float]], qids: list[str], k_values: Iterable[int]):
    metrics: dict[str, float] = {}
    if not qids:
        return metrics
    for k in k_values:
        metrics[f"nDCG@{k}"] = round(sum(scores[qid].get(f"ndcg_cut_{k}", 0.0) for qid in qids) / len(qids), 5)
        metrics[f"MAP@{k}"] = round(sum(scores[qid].get(f"map_cut_{k}", 0.0) for qid in qids) / len(qids), 5)
        metrics[f"Recall@{k}"] = round(sum(scores[qid].get(f"recall_{k}", 0.0) for qid in qids) / len(qids), 5)
        metrics[f"P@{k}"] = round(sum(scores[qid].get(f"P_{k}", 0.0) for qid in qids) / len(qids), 5)
    metrics["num_queries"] = len(qids)
    return metrics


def evaluate_mrmr(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    query_items: list[dict],
    k_values: Iterable[int],
):
    scores = _per_query_scores(qrels, results, k_values)
    qids = [item["id"] for item in query_items if item["id"] in qrels]

    fine_groups: dict[str, list[str]] = defaultdict(list)
    coarse_groups: dict[str, list[str]] = defaultdict(list)
    for item in query_items:
        qid = item["id"]
        if qid not in qrels:
            continue
        fine_groups[item.get("category", "Unknown")].append(qid)
        coarse_groups[item.get("coarse_domain", "Unknown")].append(qid)

    return {
        "overall": _aggregate(scores, qids, k_values),
        "coarse": {
            group: _aggregate(scores, group_qids, k_values)
            for group, group_qids in sorted(coarse_groups.items())
        },
        "fine": {
            group: _aggregate(scores, group_qids, k_values)
            for group, group_qids in sorted(fine_groups.items())
        },
    }
