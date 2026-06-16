from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
import pandas as pd

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