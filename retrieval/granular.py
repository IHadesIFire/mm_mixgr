"""Dense retrieval over query and corpus embeddings."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import torch
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def retrieve(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    corpus_ids: list[str],
    corpus_embeddings: np.ndarray,
    *,
    top_k: int = 100,
    query_batch_size: int = 32,
    corpus_chunk_size: int = 4096,
) -> Dict[str, Dict[str, float]]:
    """Return {qid: {doc_id: score}} using cosine similarity on normalized embeddings."""
    logger.info(
        "Retrieving top-%d from %d corpus docs for %d queries...",
        top_k,
        len(corpus_ids),
        len(query_ids),
    )

    q_tensor = torch.as_tensor(query_embeddings, dtype=torch.float32)
    c_tensor = torch.as_tensor(corpus_embeddings, dtype=torch.float32)
    top_k = min(top_k, len(corpus_ids))

    results: Dict[str, Dict[str, float]] = {}

    for q_start in tqdm(range(0, len(query_ids), query_batch_size), desc="Scoring"):
        q_end = min(q_start + query_batch_size, len(query_ids))
        q_batch = q_tensor[q_start:q_end]

        score_parts = []
        for c_start in range(0, len(corpus_ids), corpus_chunk_size):
            c_end = min(c_start + corpus_chunk_size, len(corpus_ids))
            c_chunk = c_tensor[c_start:c_end]
            score_parts.append(torch.matmul(q_batch, c_chunk.T))

        scores = torch.cat(score_parts, dim=1)
        values, indices = torch.topk(scores, k=top_k, dim=1)

        for row_idx, qid in enumerate(query_ids[q_start:q_end]):
            results[qid] = {
                corpus_ids[col_idx]: float(score)
                for score, col_idx in zip(values[row_idx].tolist(), indices[row_idx].tolist())
            }

    return results
