"""Вспомогательные функции CatBoost scoring для inference-кандидатов"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def prepare_model_frame(
    features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> pd.DataFrame:
    """Подготовить вход модели с той же dtype policy, что в training"""
    required = set(numeric_columns + categorical_columns)
    missing = required - set(features.columns)
    if missing:
        raise KeyError(f"Missing feature columns: {sorted(missing)}")

    result = features[numeric_columns + categorical_columns].copy()
    for col in numeric_columns:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype("float32")
    for col in categorical_columns:
        result[col] = result[col].astype("string").fillna("unknown").astype(str)
    return result


def score_candidates(
    model: Any,
    features: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> np.ndarray:
    """Посчитать score кандидатов загруженной CatBoost-моделью"""
    try:
        from catboost import Pool
    except ModuleNotFoundError as exc:
        raise RuntimeError("catboost is required for ranking") from exc

    model_frame = prepare_model_frame(
        features=features,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )
    pool = Pool(data=model_frame, cat_features=categorical_columns)
    return np.asarray(model.predict(pool), dtype=np.float32)


def add_model_ranking(
    candidates: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Добавить CatBoost scores и финальные 1-based ranks"""
    if len(scores) != len(candidates):
        raise ValueError("Score count must match candidate count")

    result = candidates.copy()
    result["model_score"] = scores.astype(np.float32)
    result = result.sort_values(
        ["model_score", "embedding_score", "embedding_rank"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1, dtype=np.int32)
    return result


__all__ = ["add_model_ranking", "prepare_model_frame", "score_candidates"]
