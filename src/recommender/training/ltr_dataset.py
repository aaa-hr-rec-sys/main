from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from recommender.training.common import load_json
from utils.splits import check_split_leakage, temporal_split


def load_ltr_dataset(
    dataset_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load fixed LTR train/valid tables and metadata."""
    required_files = [
        "train_features.parquet",
        "valid_features.parquet",
        "valid_positive_pairs.parquet",
        "feature_columns.json",
        "dataset_summary.json",
    ]
    missing = [name for name in required_files if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "В dataset-dir не хватает файлов: "
            f"{missing}. Перезапустите scripts/build_ltr_dataset.py."
        )

    train = pd.read_parquet(dataset_dir / "train_features.parquet")
    valid = pd.read_parquet(dataset_dir / "valid_features.parquet")
    valid_positive_pairs = pd.read_parquet(dataset_dir / "valid_positive_pairs.parquet")
    feature_config = load_json(dataset_dir / "feature_columns.json")
    dataset_summary = load_json(dataset_dir / "dataset_summary.json")

    return train, valid, valid_positive_pairs, feature_config, dataset_summary


def get_n_items_total(
    dataset_summary: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> int:
    """Infer total CV item count used by ranking metrics."""
    counts = dataset_summary.get("counts", {})
    if "cv_embeddings_rows" in counts:
        return int(counts["cv_embeddings_rows"])

    max_idx = max(
        int(train["cv_idx"].max()) if "cv_idx" in train.columns else 0,
        int(valid["cv_idx"].max()) if "cv_idx" in valid.columns else 0,
    )
    return max_idx + 1


def load_processed_tables_for_recommendations(
    dataset_summary: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normalized CV/vacancy tables referenced by an LTR dataset."""
    processed_dir = Path(dataset_summary["processed_dir"])
    cv_path = processed_dir / "cv_normalized.parquet"
    vacancies_path = processed_dir / "vacancies_normalized.parquet"

    if not cv_path.exists() or not vacancies_path.exists():
        raise FileNotFoundError(
            f"Для --save-recommendations нужны {cv_path} и {vacancies_path}"
        )

    return pd.read_parquet(cv_path), pd.read_parquet(vacancies_path)


def prepare_validation(
    applies: pd.DataFrame,
    cv_embeddings: pd.DataFrame,
    vacancies_embeddings: pd.DataFrame,
    valid_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Build temporal validation pairs covered by embedding tables."""
    train_applies, valid_applies = temporal_split(
        applies,
        date_col="applied_at_jittered",
        valid_frac=valid_frac,
    )
    split_report = check_split_leakage(train_applies, valid_applies)

    cv_id_to_idx = pd.Series(
        np.arange(len(cv_embeddings), dtype=np.int32),
        index=cv_embeddings["cv_id_hash"],
    )
    vacancy_id_to_idx = pd.Series(
        np.arange(len(vacancies_embeddings), dtype=np.int32),
        index=vacancies_embeddings["vacancy_id_hash"],
    )

    valid_pairs = valid_applies[["cv_id_hash", "vacancy_id_hash"]].drop_duplicates().copy()
    valid_pairs["cv_idx"] = valid_pairs["cv_id_hash"].map(cv_id_to_idx)
    valid_pairs["vacancy_idx"] = valid_pairs["vacancy_id_hash"].map(vacancy_id_to_idx)

    missing_mask = valid_pairs["cv_idx"].isna() | valid_pairs["vacancy_idx"].isna()
    if missing_mask.any():
        print(f"Warning: dropping {int(missing_mask.sum())} validation pairs without embeddings")

    valid_pairs = valid_pairs.loc[~missing_mask].copy()
    valid_pairs["cv_idx"] = valid_pairs["cv_idx"].astype(np.int32)
    valid_pairs["vacancy_idx"] = valid_pairs["vacancy_idx"].astype(np.int32)

    validation_vacancy_indices = np.array(
        sorted(valid_pairs["vacancy_idx"].unique()),
        dtype=np.int32,
    )

    return train_applies, valid_pairs, split_report, validation_vacancy_indices
