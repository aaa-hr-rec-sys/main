"""Загрузка и валидация самодостаточного inference bundle"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ArtifactLoadError(RuntimeError):
    """Ошибка безопасной загрузки inference bundle"""


@dataclass(frozen=True)
class InferenceArtifacts:
    """Все артефакты для синхронной выдачи рекомендаций"""

    bundle_dir: Path
    manifest: dict[str, Any]
    feature_schema: dict[str, Any]
    model: Any
    cv_store: pd.DataFrame
    vacancy_store: pd.DataFrame
    cv_ids: np.ndarray
    vacancy_ids: np.ndarray
    cv_embeddings: np.ndarray
    vacancy_embeddings: np.ndarray
    numeric_feature_columns: list[str]
    categorical_feature_columns: list[str]
    feature_columns: list[str]
    retrieval_top_k: int
    max_response_top_k: int
    cv_id_to_pos: dict[str, int]
    vacancy_id_to_pos: dict[str, int]

    @property
    def embedding_dim(self) -> int:
        return int(self.cv_embeddings.shape[1])


DEFAULT_FILES = {
    "manifest": "manifest.json",
    "feature_schema": "feature_schema.json",
    "model": "model.cbm",
    "cv_store": "cv_store.parquet",
    "vacancy_store": "vacancy_store.parquet",
    "cv_ids": "cv_ids.parquet",
    "vacancy_ids": "vacancy_ids.parquet",
    "cv_embeddings": "cv_embeddings.npy",
    "vacancy_embeddings": "vacancy_embeddings.npy",
}


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactLoadError(f"Required artifact file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ArtifactLoadError(f"Invalid JSON artifact: {path}") from exc


def _file_path(bundle_dir: Path, manifest: dict[str, Any], logical_name: str) -> Path:
    files = manifest.get("files", {})
    return bundle_dir / files.get(logical_name, DEFAULT_FILES[logical_name])


def _load_model(path: Path) -> Any:
    try:
        from catboost import CatBoostRanker
    except ModuleNotFoundError as exc:
        raise ArtifactLoadError("catboost is required to load model.cbm") from exc

    if not path.exists():
        raise ArtifactLoadError(f"Required artifact file is missing: {path}")

    model = CatBoostRanker()
    try:
        model.load_model(str(path))
    except Exception as exc:  # pragma: no cover - типы исключений CatBoost различаются
        raise ArtifactLoadError(f"Could not load CatBoost model from {path}: {exc}") from exc
    return model


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except FileNotFoundError as exc:
        raise ArtifactLoadError(f"Required artifact file is missing: {path}") from exc
    except Exception as exc:
        raise ArtifactLoadError(f"Could not read parquet artifact {path}: {exc}") from exc


def _load_matrix(path: Path) -> np.ndarray:
    if not path.exists():
        raise ArtifactLoadError(f"Required artifact file is missing: {path}")
    try:
        matrix = np.load(path)
    except Exception as exc:
        raise ArtifactLoadError(f"Could not read NumPy artifact {path}: {exc}") from exc
    if matrix.ndim != 2:
        raise ArtifactLoadError(f"{path.name} must be a 2D matrix")
    return np.asarray(matrix, dtype=np.float32)


def _load_ids(path: Path, column: str) -> np.ndarray:
    ids_df = _read_parquet(path)
    if column not in ids_df.columns:
        raise ArtifactLoadError(f"{path.name} must contain column {column}")
    if ids_df[column].isna().any():
        raise ArtifactLoadError(f"{path.name} contains empty ids")
    if ids_df[column].duplicated().any():
        raise ArtifactLoadError(f"{path.name} contains duplicate ids")
    return ids_df[column].astype(str).to_numpy()


def _require_columns(df: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ArtifactLoadError(f"{name} is missing columns: {sorted(missing)}")


def _build_pos_map(ids: np.ndarray, name: str) -> dict[str, int]:
    result = {str(item_id): pos for pos, item_id in enumerate(ids)}
    if len(result) != len(ids):
        raise ArtifactLoadError(f"{name} contains duplicate ids")
    return result


def validate_artifacts(artifacts: InferenceArtifacts) -> None:
    """Проверить shape, id и совместимость model/feature-schema"""
    if artifacts.cv_embeddings.shape[0] != len(artifacts.cv_ids):
        raise ArtifactLoadError("cv_embeddings row count must match cv_ids")
    if artifacts.vacancy_embeddings.shape[0] != len(artifacts.vacancy_ids):
        raise ArtifactLoadError("vacancy_embeddings row count must match vacancy_ids")
    if artifacts.cv_embeddings.shape[1] != artifacts.vacancy_embeddings.shape[1]:
        raise ArtifactLoadError("CV and vacancy embedding dimensions must match")

    _require_columns(artifacts.cv_store, {"cv_id_hash"}, "cv_store")
    _require_columns(artifacts.vacancy_store, {"vacancy_id_hash"}, "vacancy_store")
    if artifacts.cv_store["cv_id_hash"].astype(str).duplicated().any():
        raise ArtifactLoadError("cv_store contains duplicate cv_id_hash values")
    if artifacts.vacancy_store["vacancy_id_hash"].astype(str).duplicated().any():
        raise ArtifactLoadError("vacancy_store contains duplicate vacancy_id_hash values")

    cv_store_ids = set(artifacts.cv_store["cv_id_hash"].astype(str))
    vacancy_store_ids = set(artifacts.vacancy_store["vacancy_id_hash"].astype(str))
    missing_cv = set(map(str, artifacts.cv_ids)) - cv_store_ids
    missing_vacancies = set(map(str, artifacts.vacancy_ids)) - vacancy_store_ids
    if missing_cv:
        raise ArtifactLoadError(f"cv_store is missing {len(missing_cv)} ids from cv_ids")
    if missing_vacancies:
        raise ArtifactLoadError(
            f"vacancy_store is missing {len(missing_vacancies)} ids from vacancy_ids"
        )

    model_feature_names = list(getattr(artifacts.model, "feature_names_", []) or [])
    if model_feature_names and model_feature_names != artifacts.feature_columns:
        raise ArtifactLoadError(
            "Feature schema does not match CatBoost model feature names: "
            f"schema={artifacts.feature_columns}, model={model_feature_names}"
        )


def load_inference_artifacts(bundle_dir: str | Path) -> InferenceArtifacts:
    """Загрузить и проверить директорию inference bundle"""
    bundle_dir = Path(bundle_dir)
    manifest = load_json(bundle_dir / DEFAULT_FILES["manifest"])

    feature_schema = load_json(_file_path(bundle_dir, manifest, "feature_schema"))
    numeric_cols = list(feature_schema.get("numeric_feature_columns", []))
    categorical_cols = list(feature_schema.get("categorical_feature_columns", []))
    feature_cols = list(feature_schema.get("feature_columns") or numeric_cols + categorical_cols)
    if not numeric_cols or not feature_cols:
        raise ArtifactLoadError("feature_schema.json must define model feature columns")

    model = _load_model(_file_path(bundle_dir, manifest, "model"))
    cv_store = _read_parquet(_file_path(bundle_dir, manifest, "cv_store"))
    vacancy_store = _read_parquet(_file_path(bundle_dir, manifest, "vacancy_store"))
    cv_ids = _load_ids(_file_path(bundle_dir, manifest, "cv_ids"), "cv_id_hash")
    vacancy_ids = _load_ids(
        _file_path(bundle_dir, manifest, "vacancy_ids"),
        "vacancy_id_hash",
    )
    cv_embeddings = _load_matrix(_file_path(bundle_dir, manifest, "cv_embeddings"))
    vacancy_embeddings = _load_matrix(_file_path(bundle_dir, manifest, "vacancy_embeddings"))

    config = manifest.get("config", {})
    artifacts = InferenceArtifacts(
        bundle_dir=bundle_dir,
        manifest=manifest,
        feature_schema=feature_schema,
        model=model,
        cv_store=cv_store,
        vacancy_store=vacancy_store,
        cv_ids=cv_ids,
        vacancy_ids=vacancy_ids,
        cv_embeddings=cv_embeddings,
        vacancy_embeddings=vacancy_embeddings,
        numeric_feature_columns=numeric_cols,
        categorical_feature_columns=categorical_cols,
        feature_columns=feature_cols,
        retrieval_top_k=int(config.get("retrieval_top_k", 500)),
        max_response_top_k=int(config.get("max_response_top_k", 50)),
        cv_id_to_pos=_build_pos_map(cv_ids, "cv_ids"),
        vacancy_id_to_pos=_build_pos_map(vacancy_ids, "vacancy_ids"),
    )
    validate_artifacts(artifacts)
    return artifacts


__all__ = [
    "ArtifactLoadError",
    "DEFAULT_FILES",
    "InferenceArtifacts",
    "load_inference_artifacts",
    "load_json",
    "validate_artifacts",
]
