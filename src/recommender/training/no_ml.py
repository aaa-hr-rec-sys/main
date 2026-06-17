from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

from recommender.metrics import evaluate_ranked_candidates


#normalized fields
CV_TEXT_COLUMNS = [
    "profession_norm",
    "group_profession_norm",
    "business_category_norm",
    "sfera_norm",
    "experience_common",
    "schedule_norm",
    "employment_type_norm",
    "education_norm",
]

VACANCY_TEXT_COLUMNS = [
    "profession_norm",
    "group_profession_norm",
    "business_category_norm",
    "sfera_norm",
    "experience_common",
    "schedule_norm",
    "employment_type_norm",
    "education_level_norm",
]

BAD_VALUES = {"", "nan", "none", "null", "unknown", "other", "<na>"}
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", flags=re.IGNORECASE)


def norm_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_good_value(value: object) -> bool:
    return norm_value(value) not in BAD_VALUES


def tokens_from_fields(row: pd.Series, columns: list[str]) -> set[str]:
    """Simple token set from normalized fields.

    This is intentionally simple:
    - uses only normalized fields;
    - lowercases;
    - splits by words;
    - removes empty/unknown/other-like tokens.
    """
    tokens: set[str] = set()

    for col in columns:
        if col not in row.index:
            continue

        value = norm_value(row[col])
        if value in BAD_VALUES:
            continue

        for token in TOKEN_RE.findall(value):
            token = token.lower().replace("ё", "е")
            if len(token) < 2:
                continue
            if token in BAD_VALUES:
                continue
            tokens.add(token)

    return tokens


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

def build_exact_title_candidates(
    cv: pd.DataFrame,
    valid_vacancies: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Build candidates by exact normalized profession match."""
    required_cv = {"cv_id_hash", "profession_norm"}
    required_vac = {"vacancy_id_hash", "profession_norm"}
    if missing := required_cv - set(cv.columns):
        raise KeyError(f"CV table misses columns: {sorted(missing)}")
    if missing := required_vac - set(valid_vacancies.columns):
        raise KeyError(f"Vacancy table misses columns: {sorted(missing)}")

    cv_tmp = cv[["cv_id_hash", "profession_norm"]].copy()
    cv_tmp["profession_key"] = cv_tmp["profession_norm"].map(norm_value)
    cv_tmp = cv_tmp[cv_tmp["profession_key"].map(lambda x: x not in BAD_VALUES)]

    cv_by_title = {
        title: group["cv_id_hash"].sort_values().tolist()
        for title, group in cv_tmp.groupby("profession_key", sort=False)
    }

    chunks: list[pd.DataFrame] = []

    for _, row in valid_vacancies[["vacancy_id_hash", "profession_norm"]].iterrows():
        vacancy_id = str(row["vacancy_id_hash"])
        title = norm_value(row["profession_norm"])

        if title in BAD_VALUES:
            continue

        cv_ids = cv_by_title.get(title, [])
        if not cv_ids:
            continue

        cv_ids = cv_ids[:top_k]
        chunks.append(
            pd.DataFrame(
                {
                    "vacancy_id_hash": vacancy_id,
                    "cv_id_hash": cv_ids,
                    "exact_title_score": 1.0,
                    "exact_title_rank": np.arange(1, len(cv_ids) + 1, dtype="int32"),
                }
            )
        )

    if not chunks:
        return pd.DataFrame(
            columns=["vacancy_id_hash", "cv_id_hash", "exact_title_score", "exact_title_rank"]
        )

    return pd.concat(chunks, ignore_index=True)


def build_word_overlap_candidates(
    cv: pd.DataFrame,
    valid_vacancies: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Build candidates by normalized word overlap."""
    cv_ids = cv["cv_id_hash"].astype(str).to_numpy()

    # CV index: token -> CV row indexes with this token.
    token_to_cv_indexes: dict[str, list[int]] = defaultdict(list)

    for cv_idx, (_, row) in enumerate(cv.iterrows()):
        cv_tokens = tokens_from_fields(row, CV_TEXT_COLUMNS)
        for token in cv_tokens:
            token_to_cv_indexes[token].append(cv_idx)

    chunks: list[pd.DataFrame] = []

    for _, vacancy_row in valid_vacancies.iterrows():
        vacancy_id = str(vacancy_row["vacancy_id_hash"])
        vacancy_tokens = tokens_from_fields(vacancy_row, VACANCY_TEXT_COLUMNS)

        if not vacancy_tokens:
            continue
        
        # Score = number of shared unique tokens.
        scores: dict[int, int] = defaultdict(int)
        for token in vacancy_tokens:
            for cv_idx in token_to_cv_indexes.get(token, []):
                scores[cv_idx] += 1

        if not scores:
            continue

        ranked = sorted(
            scores.items(),
            key=lambda x: (-x[1], str(cv_ids[x[0]])),
        )[:top_k]

        chunks.append(
            pd.DataFrame(
                {
                    "vacancy_id_hash": vacancy_id,
                    "cv_id_hash": [str(cv_ids[cv_idx]) for cv_idx, _ in ranked],
                    "word_overlap_score": [int(score) for _, score in ranked],
                    "word_overlap_rank": np.arange(1, len(ranked) + 1, dtype="int32"),
                }
            )
        )

    if not chunks:
        return pd.DataFrame(
            columns=["vacancy_id_hash", "cv_id_hash", "word_overlap_score", "word_overlap_rank"]
        )

    return pd.concat(chunks, ignore_index=True)


@dataclass(frozen=True)
class RankingEvaluationInput:
    """One ranked candidate table to evaluate."""

    candidates: pd.DataFrame
    rank_col: str
    ranking_name: str


def evaluate_ranking(
    item: RankingEvaluationInput,
    positive_pairs: pd.DataFrame,
    ks: list[int],
    n_items_total: int,
) -> pd.DataFrame:
    """Evaluate one ranking with the shared project metrics."""
    return evaluate_ranked_candidates(
        candidates=item.candidates,
        positive_pairs=positive_pairs,
        rank_col=item.rank_col,
        ks=ks,
        n_items_total=n_items_total,
        ranking_name=item.ranking_name,
    )


def evaluate_rankings(
    rankings: list[RankingEvaluationInput],
    positive_pairs: pd.DataFrame,
    ks: list[int],
    n_items_total: int,
) -> pd.DataFrame:
    """Evaluate several rankings and concatenate metrics in the same order."""
    frames = [
        evaluate_ranking(
            item=item,
            positive_pairs=positive_pairs,
            ks=ks,
            n_items_total=n_items_total,
        )
        for item in rankings
    ]

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def evaluate_rankings_and_save(
    rankings: list[RankingEvaluationInput],
    positive_pairs: pd.DataFrame,
    ks: list[int],
    n_items_total: int,
    output_dir: Path,
    filename: str = "metrics.csv",
) -> tuple[pd.DataFrame, Path]:
    """Evaluate rankings and save metrics.csv."""
    metrics = evaluate_rankings(
        rankings=rankings,
        positive_pairs=positive_pairs,
        ks=ks,
        n_items_total=n_items_total,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / filename
    metrics.to_csv(metrics_path, index=False)

    return metrics, metrics_path