"""Exact embedding retrieval для одного vacancy-запроса"""

from __future__ import annotations

import numpy as np
import pandas as pd


def retrieve_top_k(
    vacancy_id_hash: str,
    vacancy_embedding: np.ndarray,
    cv_embeddings: np.ndarray,
    cv_ids: np.ndarray,
    top_k: int,
) -> pd.DataFrame:
    """Вернуть exact top-K CV-кандидатов по dot product"""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if cv_embeddings.ndim != 2:
        raise ValueError("cv_embeddings must be a 2D matrix")

    query = np.asarray(vacancy_embedding, dtype=np.float32).reshape(-1)
    if query.shape[0] != cv_embeddings.shape[1]:
        raise ValueError("vacancy and CV embeddings must have the same dimension")
    if cv_embeddings.shape[0] != len(cv_ids):
        raise ValueError("cv_embeddings row count must match cv_ids")
    if cv_embeddings.shape[0] == 0:
        raise ValueError("cv_embeddings is empty")

    k_eff = min(int(top_k), cv_embeddings.shape[0])
    scores = cv_embeddings @ query

    if k_eff == len(scores):
        candidate_idx = np.arange(len(scores))
    else:
        candidate_idx = np.argpartition(-scores, kth=k_eff - 1)[:k_eff]

    candidate_scores = scores[candidate_idx]
    order = np.lexsort((candidate_idx, -candidate_scores))
    top_idx = candidate_idx[order]
    top_scores = candidate_scores[order]

    return pd.DataFrame(
        {
            "cv_idx": top_idx.astype(np.int32),
            "vacancy_id_hash": vacancy_id_hash,
            "cv_id_hash": cv_ids[top_idx].astype(str),
            "embedding_score": top_scores.astype(np.float32),
            "embedding_rank": np.arange(1, k_eff + 1, dtype=np.int32),
        }
    )


__all__ = ["retrieve_top_k"]
