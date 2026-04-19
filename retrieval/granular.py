"""
Retrieval: compute similarity and rank documents.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def retrieve(
    query_ids: List[str],
    query_embeddings: np.ndarray,
    corpus_ids: List[str],
    corpus_embeddings: np.ndarray,
    top_k: int = 100,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Retrieve top-k corpus docs for each query by cosine similarity.
    Assumes embeddings are already L2 normalized.

    Args:
        query_ids: list of query IDs
        query_embeddings: (N_q, dim)
        corpus_ids: list of corpus doc IDs
        corpus_embeddings: (N_c, dim)
        top_k: number of results per query

    Returns:
        {query_id: [(doc_id, score), ...]} sorted by score desc
    """
    logger.info(f"Retrieving top-{top_k} for {len(query_ids)} queries "
                f"from {len(corpus_ids)} corpus docs...")

    q_tensor = torch.tensor(query_embeddings, dtype=torch.float32)
    c_tensor = torch.tensor(corpus_embeddings, dtype=torch.float32)

    results = {}
    chunk_size = 2000

    for i, qid in enumerate(query_ids):
        q_vec = q_tensor[i:i+1]  # (1, dim)

        all_scores = []
        for c_start in range(0, len(corpus_ids), chunk_size):
            c_chunk = c_tensor[c_start:c_start+chunk_size]
            scores = torch.mm(q_vec, c_chunk.T).squeeze(0)
            all_scores.append(scores)

        all_scores = torch.cat(all_scores)

        k = min(top_k, len(corpus_ids))
        top_scores, top_indices = torch.topk(all_scores, k)

        results[qid] = [
            (corpus_ids[idx.item()], score.item())
            for score, idx in zip(top_scores, top_indices)
        ]

    logger.info(f"  Retrieval done.")
    return results