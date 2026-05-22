"""Утилиты для embeddings, candidate generation и baseline-ноутбуков"""

from __future__ import annotations

import numpy as np
import pandas as pd


def embedding_to_matrix(
    df: pd.DataFrame, column: str = "embedding", dtype: str | np.dtype = "float32"
) -> np.ndarray:
    """Преобразует колонку DataFrame из list-like embeddings в 2D-матрицу

    Параметры
    ----------
    df:
        DataFrame с колонкой embeddings
    column:
        Название колонки с list-like векторами
    dtype:
        NumPy dtype итоговой матрицы
        ``float32`` подходит для dot product и избегает потерь точности
        ``float16``

    Возвращает
    -------
    np.ndarray
        Матрица с формой ``(len(df), embedding_dim)``

    Исключения
    ------
    ValueError
        Если embeddings имеют разную длину или входной DataFrame пустой
    """
    if df.empty:
        raise ValueError("Пустой DataFrame")

    lengths = df[column].map(len)
    if lengths.nunique() != 1:
        raise ValueError(f"Найдены embeddings разной размерности: {lengths.value_counts().head()}")

    return np.vstack(df[column].to_numpy()).astype(dtype, copy=False)


def embedding_norm_report(matrix: np.ndarray, name: str) -> pd.DataFrame:
    """Собирает отчет о форме embedding-матрицы и L2-нормах

    Параметры
    ----------
    matrix:
        2D embedding-матрица
    name:
        Читаемое название матрицы для отчета

    Возвращает
    -------
    pd.DataFrame
        Однострочный отчет с числом строк, размерностью и перцентилями нормы
    """
    if matrix.ndim != 2:
        raise ValueError("Ожидалась 2D embedding-матрица")

    if matrix.shape[0] == 0:
        return pd.DataFrame(
            [
                {
                    "name": name,
                    "rows": 0,
                    "dim": matrix.shape[1] if matrix.ndim == 2 else None,
                    "norm_min": np.nan,
                    "norm_p01": np.nan,
                    "norm_p50": np.nan,
                    "norm_p99": np.nan,
                    "norm_max": np.nan,
                }
            ]
        )

    norms = np.linalg.norm(matrix, axis=1)
    return pd.DataFrame(
        [
            {
                "name": name,
                "rows": matrix.shape[0],
                "dim": matrix.shape[1],
                "norm_min": norms.min(),
                "norm_p01": np.percentile(norms, 1),
                "norm_p50": np.percentile(norms, 50),
                "norm_p99": np.percentile(norms, 99),
                "norm_max": norms.max(),
            }
        ]
    )


def chunked_topk(
    query_matrix: np.ndarray,
    doc_matrix: np.ndarray,
    k: int,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Находит top-K документов (cv) для каждого query (вакансий) через dot product чанками

    Параметры
    ----------
    query_matrix:
        Матрица query embeddings, например embeddings вакансий
    doc_matrix:
        Матрица document embeddings, например embeddings CV
    k:
        Число ближайших документов для каждого query
    chunk_size:
        Число query, обрабатываемых за один шаг
        Это держит временную similarity-матрицу маленькой и не хранит полную
        матрицу ``query x doc``

    Возвращает
    -------
    tuple[np.ndarray, np.ndarray]
        ``(top_indices, top_scores)``
        Оба массива имеют форму ``(n_queries, min(k, n_docs))``
        Индексы указывают на строки ``doc_matrix``, scores отсортированы по
        убыванию внутри каждой строки query

    Пример
    -------
    >>> query_matrix = np.array([
    ...     [1.0, 0.0],
    ...     [0.0, 1.0],
    ...     [0.7, 0.7],
    ... ], dtype=np.float32)
    >>> doc_matrix = np.array([
    ...     [1.0, 0.0],
    ...     [0.0, 1.0],
    ...     [0.6, 0.8],
    ...     [-1.0, 0.0],
    ... ], dtype=np.float32)
    >>> top_indices, top_scores = chunked_topk(query_matrix, doc_matrix, k=2, chunk_size=2)
    >>> top_indices
    array([[0, 2],
           [1, 2],
           [2, 0]])
    >>> np.round(top_scores, 2)
    array([[1.  , 0.6 ],
           [1.  , 0.8 ],
           [0.98, 0.7 ]], dtype=float32)
    """
    if query_matrix.ndim != 2 or doc_matrix.ndim != 2:
        raise ValueError("query_matrix и doc_matrix должны быть 2D-массивами")
    if query_matrix.shape[1] != doc_matrix.shape[1]:
        raise ValueError("query_matrix и doc_matrix должны иметь одинаковую embedding-размерность")
    if k <= 0:
        raise ValueError("k должен быть положительным")
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть положительным")

    n_queries = query_matrix.shape[0]
    n_docs = doc_matrix.shape[0]
    k_eff = min(k, n_docs)
    if n_docs == 0:
        return np.empty((n_queries, 0), dtype=np.int64), np.empty((n_queries, 0), dtype=query_matrix.dtype)

    top_indices = np.empty((n_queries, k_eff), dtype=np.int64)
    top_scores = np.empty((n_queries, k_eff), dtype=np.float32)
    doc_matrix_t = np.ascontiguousarray(doc_matrix.T)

    for start in range(0, n_queries, chunk_size):
        stop = min(start + chunk_size, n_queries)
        scores = query_matrix[start:stop] @ doc_matrix_t
        unsorted_idx = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
        unsorted_scores = np.take_along_axis(scores, unsorted_idx, axis=1)
        order = np.argsort(-unsorted_scores, axis=1)
        top_indices[start:stop] = np.take_along_axis(unsorted_idx, order, axis=1)
        top_scores[start:stop] = np.take_along_axis(unsorted_scores, order, axis=1)

    return top_indices, top_scores


def build_embedding_candidates(
    vacancies_embeddings: pd.DataFrame,
    cv_embeddings: pd.DataFrame,
    top_k: int = 100,
    chunk_size: int = 256,
) -> pd.DataFrame:
    """Строит кандидатов vacancy-to-CV по embedding dot product

    Параметры
    ----------
    vacancies_embeddings:
        DataFrame с колонками ``vacancy_id_hash`` и ``embedding``
    cv_embeddings:
        DataFrame с колонками ``cv_id_hash`` и ``embedding``
    top_k:
        Число CV-кандидатов для каждой вакансии
    chunk_size:
        Число embeddings вакансий, обрабатываемых за один шаг

    Возвращает
    -------
    pd.DataFrame
        Таблица кандидатов с колонками ``vacancy_id_hash``, ``cv_id_hash``,
        ``score`` и 1-based ``rank``
        Scores являются dot product, что эквивалентно cosine similarity для
        L2-нормированных embeddings

    Пример
    -------
    >>> vacancies_embeddings = pd.DataFrame({
    ...     "vacancy_id_hash": ["v1", "v2"],
    ...     "embedding": [
    ...         np.array([1.0, 0.0], dtype=np.float32),
    ...         np.array([0.0, 1.0], dtype=np.float32),
    ...     ],
    ... })
    >>> cv_embeddings = pd.DataFrame({
    ...     "cv_id_hash": ["cv1", "cv2", "cv3"],
    ...     "embedding": [
    ...         np.array([1.0, 0.0], dtype=np.float32),
    ...         np.array([0.0, 1.0], dtype=np.float32),
    ...         np.array([0.6, 0.8], dtype=np.float32),
    ...     ],
    ... })
    >>> build_embedding_candidates(
    ...     vacancies_embeddings,
    ...     cv_embeddings,
    ...     top_k=2,
    ...     chunk_size=1,
    ... )
      vacancy_id_hash cv_id_hash  score  rank
    0              v1        cv1    1.0     1
    1              v1        cv3    0.6     2
    2              v2        cv2    1.0     1
    3              v2        cv3    0.8     2
    """
    vacancy_matrix = embedding_to_matrix(vacancies_embeddings)
    cv_matrix = embedding_to_matrix(cv_embeddings)
    top_indices, top_scores = chunked_topk(vacancy_matrix, cv_matrix, top_k, chunk_size)

    vacancy_ids = vacancies_embeddings["vacancy_id_hash"].to_numpy()
    cv_ids = cv_embeddings["cv_id_hash"].to_numpy()
    rows = []
    for vacancy_pos, vacancy_id in enumerate(vacancy_ids):
        for rank_pos, cv_pos in enumerate(top_indices[vacancy_pos], start=1):
            rows.append(
                {
                    "vacancy_id_hash": vacancy_id,
                    "cv_id_hash": cv_ids[cv_pos],
                    "score": float(top_scores[vacancy_pos, rank_pos - 1]),
                    "rank": rank_pos,
                }
            )

    return pd.DataFrame(rows, columns=["vacancy_id_hash", "cv_id_hash", "score", "rank"])


__all__ = [
    "build_embedding_candidates",
    "chunked_topk",
    "embedding_norm_report",
    "embedding_to_matrix",
]
