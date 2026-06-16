from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ExperimentInputs:
    """Common data for validation experiments."""

    cv: pd.DataFrame
    vacancies: pd.DataFrame
    train_positive_pairs: pd.DataFrame
    valid_positive_pairs: pd.DataFrame
    valid_vacancies: pd.DataFrame


def load_processed_tables(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load normalized CV and vacancy tables."""
    cv_path = processed_dir / "cv_normalized.parquet"
    vacancies_path = processed_dir / "vacancies_normalized.parquet"

    if not cv_path.exists():
        raise FileNotFoundError(f"Required file not found: {cv_path}")
    if not vacancies_path.exists():
        raise FileNotFoundError(f"Required file not found: {vacancies_path}")

    cv = pd.read_parquet(cv_path)
    vacancies = pd.read_parquet(vacancies_path)

    cv["cv_id_hash"] = cv["cv_id_hash"].astype(str)
    vacancies["vacancy_id_hash"] = vacancies["vacancy_id_hash"].astype(str)

    return cv, vacancies


def load_train_valid_positive_pairs(
    dataset_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/valid positive pairs from an existing modeling dataset."""
    train_pos_path = dataset_dir / "train_positive_pairs.parquet"
    valid_pos_path = dataset_dir / "valid_positive_pairs.parquet"

    if not train_pos_path.exists():
        raise FileNotFoundError(f"Required file not found: {train_pos_path}")
    if not valid_pos_path.exists():
        raise FileNotFoundError(f"Required file not found: {valid_pos_path}")

    train_positive_pairs = pd.read_parquet(train_pos_path)
    valid_positive_pairs = pd.read_parquet(valid_pos_path)

    for pairs in [train_positive_pairs, valid_positive_pairs]:
        pairs["vacancy_id_hash"] = pairs["vacancy_id_hash"].astype(str)
        pairs["cv_id_hash"] = pairs["cv_id_hash"].astype(str)

    return train_positive_pairs, valid_positive_pairs


def get_validation_vacancies(
    vacancies: pd.DataFrame,
    valid_positive_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Use the same validation vacancy set as existing experiments."""
    valid_vacancy_ids = valid_positive_pairs["vacancy_id_hash"].drop_duplicates().tolist()
    order = {vacancy_id: i for i, vacancy_id in enumerate(valid_vacancy_ids)}

    result = vacancies[vacancies["vacancy_id_hash"].isin(order)].copy()
    result["_order"] = result["vacancy_id_hash"].map(order)
    result = result.sort_values("_order", kind="mergesort").drop(columns="_order")

    return result.reset_index(drop=True)


def load_experiment_data(
    processed_dir: Path,
    dataset_dir: Path,
) -> ExperimentInputs:
    """Load normalized tables and the same train/validation split."""
    cv, vacancies = load_processed_tables(processed_dir)
    train_positive_pairs, valid_positive_pairs = load_train_valid_positive_pairs(dataset_dir)
    valid_vacancies = get_validation_vacancies(vacancies, valid_positive_pairs)

    return ExperimentInputs(
        cv=cv,
        vacancies=vacancies,
        train_positive_pairs=train_positive_pairs,
        valid_positive_pairs=valid_positive_pairs,
        valid_vacancies=valid_vacancies,
    )