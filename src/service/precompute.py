"""Сборка самодостаточного inference bundle из существующих offline artifacts"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommender.features import get_categorical_feature_columns
from utils.embeddings import embedding_to_matrix


FINAL_MODEL_NAME = "catboost_ranker_lossYetiRank_it500_depth6_lr0p05_l23p0_onehot10_wnone.cbm"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_unique_ids(df: pd.DataFrame, column: str, table_name: str) -> None:
    if column not in df.columns:
        raise KeyError(f"{table_name} must contain {column}")
    if df[column].isna().any():
        raise ValueError(f"{table_name} contains empty {column} values")
    if df[column].duplicated().any():
        raise ValueError(f"{table_name} contains duplicate {column} values")


def _load_model_feature_names(model_path: Path) -> list[str]:
    try:
        from catboost import CatBoostRanker
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("catboost is required to inspect the model") from exc

    model = CatBoostRanker()
    model.load_model(str(model_path))
    return list(model.feature_names_)


def build_inference_bundle(
    output_dir: Path,
    data_dir: Path = Path("data/aaa-out"),
    processed_dir: Path = Path("data/processed/v1"),
    dataset_dir: Path = Path("data/modeling/ltr_v1_top500_neg5_ohe"),
    model_path: Path = Path("models") / FINAL_MODEL_NAME,
    retrieval_top_k: int = 500,
    max_response_top_k: int = 500,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Собрать и проверить директорию inference bundle"""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Bundle directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_store_path = processed_dir / "cv_normalized.parquet"
    vacancy_store_path = processed_dir / "vacancies_normalized.parquet"
    cv_embeddings_path = data_dir / "cv_embeddings.parquet"
    vacancy_embeddings_path = data_dir / "vacancies_embeddings.parquet"
    feature_columns_path = dataset_dir / "feature_columns.json"
    normalization_manifest_path = processed_dir / "manifest.json"

    cv_store = pd.read_parquet(cv_store_path)
    vacancy_store = pd.read_parquet(vacancy_store_path)
    cv_embeddings = pd.read_parquet(cv_embeddings_path)
    vacancy_embeddings = pd.read_parquet(vacancy_embeddings_path)
    feature_config = load_json(feature_columns_path)

    _validate_unique_ids(cv_store, "cv_id_hash", "cv_store")
    _validate_unique_ids(vacancy_store, "vacancy_id_hash", "vacancy_store")
    _validate_unique_ids(cv_embeddings, "cv_id_hash", "cv_embeddings")
    _validate_unique_ids(vacancy_embeddings, "vacancy_id_hash", "vacancy_embeddings")

    cv_matrix = embedding_to_matrix(cv_embeddings, dtype=np.float32)
    vacancy_matrix = embedding_to_matrix(vacancy_embeddings, dtype=np.float32)
    if cv_matrix.shape[1] != vacancy_matrix.shape[1]:
        raise ValueError("CV and vacancy embedding dimensions must match")

    numeric_columns = list(feature_config["numeric_feature_columns"])
    categorical_columns = get_categorical_feature_columns()
    feature_columns = numeric_columns + categorical_columns
    model_feature_names = _load_model_feature_names(model_path)
    if model_feature_names != feature_columns:
        raise ValueError(
            "Model feature names differ from generated inference schema: "
            f"model={model_feature_names}, schema={feature_columns}"
        )

    cv_store.to_parquet(output_dir / "cv_store.parquet", index=False)
    vacancy_store.to_parquet(output_dir / "vacancy_store.parquet", index=False)
    cv_embeddings[["cv_id_hash"]].to_parquet(output_dir / "cv_ids.parquet", index=False)
    vacancy_embeddings[["vacancy_id_hash"]].to_parquet(
        output_dir / "vacancy_ids.parquet",
        index=False,
    )
    np.save(output_dir / "cv_embeddings.npy", cv_matrix)
    np.save(output_dir / "vacancy_embeddings.npy", vacancy_matrix)
    shutil.copy2(model_path, output_dir / "model.cbm")
    if normalization_manifest_path.exists():
        shutil.copy2(normalization_manifest_path, output_dir / "normalization_manifest.json")

    feature_schema = {
        "numeric_feature_columns": numeric_columns,
        "categorical_feature_columns": categorical_columns,
        "feature_columns": feature_columns,
        "label_column": feature_config.get("label_column", "label"),
        "query_column": feature_config.get("query_column", "vacancy_id_hash"),
        "item_column": feature_config.get("item_column", "cv_id_hash"),
    }
    write_json(output_dir / "feature_schema.json", feature_schema)

    files = {
        "feature_schema": "feature_schema.json",
        "model": "model.cbm",
        "cv_store": "cv_store.parquet",
        "vacancy_store": "vacancy_store.parquet",
        "cv_ids": "cv_ids.parquet",
        "vacancy_ids": "vacancy_ids.parquet",
        "cv_embeddings": "cv_embeddings.npy",
        "vacancy_embeddings": "vacancy_embeddings.npy",
    }
    if (output_dir / "normalization_manifest.json").exists():
        files["normalization_manifest"] = "normalization_manifest.json"

    source_files = {
        "cv_store": cv_store_path,
        "vacancy_store": vacancy_store_path,
        "cv_embeddings": cv_embeddings_path,
        "vacancy_embeddings": vacancy_embeddings_path,
        "feature_columns": feature_columns_path,
        "model": model_path,
    }
    manifest = {
        "bundle_version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "config": {
            "retrieval_top_k": int(retrieval_top_k),
            "max_response_top_k": int(max_response_top_k),
        },
        "counts": {
            "cv_rows": int(len(cv_store)),
            "vacancy_rows": int(len(vacancy_store)),
            "cv_embeddings_rows": int(cv_matrix.shape[0]),
            "vacancy_embeddings_rows": int(vacancy_matrix.shape[0]),
        },
        "embedding": {
            "dim": int(cv_matrix.shape[1]),
            "dtype": str(cv_matrix.dtype),
            "retrieval": "exact_dot_product",
        },
        "model": {
            "type": "CatBoostRanker",
            "feature_count": len(feature_columns),
            "source_path": str(model_path),
        },
        "source_files": {name: str(path) for name, path in source_files.items()},
        "source_checksums": {
            name: file_sha256(path) for name, path in source_files.items() if path.exists()
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


__all__ = ["FINAL_MODEL_NAME", "build_inference_bundle"]
