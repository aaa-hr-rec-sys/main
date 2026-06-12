"""Shared retrieval helpers for vacancy-to-CV experiments."""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd


def build_embedding_candidates_for_vacancies(
    vacancy_matrix: np.ndarray,
    cv_matrix: np.ndarray,
    vacancies_embeddings: pd.DataFrame,
    cv_embeddings: pd.DataFrame,
    vacancy_indices: np.ndarray | None = None,
    top_k: int = 500,
    chunk_size: int = 256,
) -> pd.DataFrame:
    """Build exact embedding top-K candidates for selected vacancies.
    Chunking is only over vacancies. For every vacancy in a chunk, the function compares this vacancy with all CVs.

    Returns
    -------
    pd.DataFrame
        Columns:
        - vacancy_idx
        - cv_idx
        - vacancy_id_hash
        - cv_id_hash
        - embedding_score
        - embedding_rank
    """
    if vacancy_matrix.ndim != 2 or cv_matrix.ndim != 2:
        raise ValueError("vacancy_matrix and cv_matrix must be 2D arrays")
    if vacancy_matrix.shape[1] != cv_matrix.shape[1]:
        raise ValueError("vacancy_matrix and cv_matrix must have the same embedding dimension")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    n_vacancies = vacancy_matrix.shape[0]
    n_cv = cv_matrix.shape[0]
    if n_cv == 0:
        raise ValueError("cv_matrix is empty")

    top_k_eff = min(top_k, n_cv)

    if vacancy_indices is None:
        vacancy_indices = np.arange(n_vacancies, dtype=np.int32)
    else:
        vacancy_indices = np.asarray(vacancy_indices, dtype=np.int32)

    cv_matrix_t = np.ascontiguousarray(cv_matrix.T)

    vacancy_ids = vacancies_embeddings["vacancy_id_hash"].to_numpy()
    cv_ids = cv_embeddings["cv_id_hash"].to_numpy()

    parts = []

    for start in range(0, len(vacancy_indices), chunk_size):
        chunk_vacancy_idx = vacancy_indices[start:start + chunk_size]

        # Shape: n_vacancies_in_chunk x n_cv.
        # Every vacancy in this chunk is compared with every CV.
        sims = vacancy_matrix[chunk_vacancy_idx] @ cv_matrix_t

        unsorted_top_idx = np.argpartition(
            -sims,
            kth=top_k_eff - 1,
            axis=1,
        )[:, :top_k_eff]
        unsorted_top_scores = np.take_along_axis(sims, unsorted_top_idx, axis=1)
        order = np.argsort(-unsorted_top_scores, axis=1)

        top_idx = np.take_along_axis(unsorted_top_idx, order, axis=1)
        top_scores = np.take_along_axis(unsorted_top_scores, order, axis=1)

        vacancy_pos_repeated = np.repeat(chunk_vacancy_idx, top_k_eff)
        cv_pos_flat = top_idx.reshape(-1)
        score_flat = top_scores.reshape(-1)
        rank_flat = np.tile(
            np.arange(1, top_k_eff + 1, dtype=np.int32),
            len(chunk_vacancy_idx),
        )

        parts.append(
            pd.DataFrame(
                {
                    "vacancy_idx": vacancy_pos_repeated.astype(np.int32),
                    "cv_idx": cv_pos_flat.astype(np.int32),
                    "vacancy_id_hash": vacancy_ids[vacancy_pos_repeated],
                    "cv_id_hash": cv_ids[cv_pos_flat],
                    "embedding_score": score_flat.astype(np.float32),
                    "embedding_rank": rank_flat,
                }
            )
        )

        del sims, unsorted_top_idx, unsorted_top_scores, order, top_idx, top_scores
        gc.collect()

    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


__all__ = ["build_embedding_candidates_for_vacancies"]
