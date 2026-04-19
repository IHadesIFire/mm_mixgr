"""
Reciprocal Rank Fusion (RRF) for combining multiple ranking lists.

Following MixGR (Cai et al., EMNLP 2024), Section 3.3.
RRF(d) = Σ_g 1 / (k + rank_g(d))
where g iterates over granularities and k is a constant (default 60).
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: Dict[str, Dict[str, List[Tuple[str, float]]]],
    k: int = 60,
    top_k: int = 100,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Fuse multiple ranking lists using RRF.

    Args:
        rankings: {granularity_name: {query_id: [(doc_id, score), ...]}}
            Each inner list should be sorted by score descending.
        k: RRF constant (default 60, following standard practice)
        top_k: number of results to return per query

    Returns:
        {query_id: [(doc_id, rrf_score), ...]} sorted by rrf_score descending
    """
    # Collect all query IDs
    all_qids = set()
    for granularity_results in rankings.values():
        all_qids.update(granularity_results.keys())

    fused = {}

    for qid in all_qids:
        doc_scores = {}

        for granularity_name, granularity_results in rankings.items():
            if qid not in granularity_results:
                continue

            ranked_list = granularity_results[qid]
            for rank, (did, _) in enumerate(ranked_list):
                rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed, +1 for 1-indexed
                doc_scores[did] = doc_scores.get(did, 0.0) + rrf_score

        # Sort by RRF score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        fused[qid] = sorted_docs[:top_k]

    logger.info(f"RRF fused {len(rankings)} granularities for {len(fused)} queries")
    return fused