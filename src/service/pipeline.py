"""Синхронный recommendation pipeline для существующего vacancy id"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from recommender.features import add_pair_features

from service.artifacts import InferenceArtifacts
from service.ranking import add_model_ranking, score_candidates
from service.retrieval import retrieve_top_k
from service.schemas import RecommendationItem, RecommendationResponse


class UnknownVacancyError(KeyError):
    """Ошибка, когда запрошенного vacancy id нет в bundle"""


CV_DISPLAY_COLUMNS = [
    "profession",
    "group_profession",
    "business_category",
    "sfera",
    "experience_bucket",
    "education",
    "federal_district",
    "salary_bucketed",
    "employment_type",
    "schedule",
]


def _jsonable(value: Any) -> Any:
    """Преобразовать pandas/numpy scalar values в JSON-friendly объекты"""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


class RecommendationPipeline:
    """Синхронно выполнить retrieval, feature building и CatBoost reranking"""

    def __init__(self, artifacts: InferenceArtifacts):
        self.artifacts = artifacts

    def recommend(self, vacancy_id_hash: str, top_k: int = 10) -> RecommendationResponse:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_k > self.artifacts.max_response_top_k:
            raise ValueError(f"top_k must be <= {self.artifacts.max_response_top_k}")

        vacancy_id_hash = vacancy_id_hash.strip()
        if vacancy_id_hash not in self.artifacts.vacancy_id_to_pos:
            raise UnknownVacancyError(vacancy_id_hash)

        vacancy_pos = self.artifacts.vacancy_id_to_pos[vacancy_id_hash]
        vacancy_embedding = self.artifacts.vacancy_embeddings[vacancy_pos]
        retrieval_k = max(top_k, self.artifacts.retrieval_top_k)

        candidates = retrieve_top_k(
            vacancy_id_hash=vacancy_id_hash,
            vacancy_embedding=vacancy_embedding,
            cv_embeddings=self.artifacts.cv_embeddings,
            cv_ids=self.artifacts.cv_ids,
            top_k=retrieval_k,
        )

        vacancy_row = self.artifacts.vacancy_store[
            self.artifacts.vacancy_store["vacancy_id_hash"].astype(str) == vacancy_id_hash
        ]
        if vacancy_row.empty:
            raise UnknownVacancyError(vacancy_id_hash)

        features = add_pair_features(
            candidates=candidates,
            cv_norm=self.artifacts.cv_store,
            vacancies_norm=vacancy_row,
            keep_debug_columns=True,
        )
        scores = score_candidates(
            model=self.artifacts.model,
            features=features,
            numeric_columns=self.artifacts.numeric_feature_columns,
            categorical_columns=self.artifacts.categorical_feature_columns,
        )
        ranked = add_model_ranking(features, scores=scores).head(top_k)

        display_by_cv = (
            self.artifacts.cv_store.set_index("cv_id_hash", drop=False)
            if "cv_id_hash" in self.artifacts.cv_store.columns
            else pd.DataFrame()
        )

        items: list[RecommendationItem] = []
        for _, row in ranked.iterrows():
            cv_id = str(row["cv_id_hash"])
            display: dict[str, Any] = {}
            if not display_by_cv.empty and cv_id in display_by_cv.index:
                display_row = display_by_cv.loc[cv_id]
                for col in CV_DISPLAY_COLUMNS:
                    if col in display_row.index:
                        display[col] = _jsonable(display_row[col])

            items.append(
                RecommendationItem(
                    cv_id_hash=cv_id,
                    rank=int(row["rank"]),
                    model_score=float(row["model_score"]),
                    embedding_score=float(row["embedding_score"]),
                    embedding_rank=int(row["embedding_rank"]),
                    display=display,
                )
            )

        return RecommendationResponse(
            vacancy_id_hash=vacancy_id_hash,
            top_k=top_k,
            recommendations=items,
        )


__all__ = ["CV_DISPLAY_COLUMNS", "RecommendationPipeline", "UnknownVacancyError"]
