"""Build shared vacancy-CV pair features for ranking models.

Features include embedding scores, normalized field matches, compatibility
signals, and optional processed categorical columns.
"""

from __future__ import annotations

import pandas as pd

from utils.normalization import (
    education_compatible_series,
    employment_type_compatible_series,
    experience_compatible_series,
    schedule_compatible_series,
)


NUMERIC_FEATURE_COLUMNS = [
    "embedding_score",
    "embedding_rank",
    "same_profession_norm",
    "same_group_profession_norm",
    "same_business_category_norm",
    "same_sfera_norm",
    "experience_compatible_feature",
    "schedule_compatible_feature",
    "employment_type_compatible_feature",
    "education_compatible_feature",
    "salary_missing",
]


CATEGORICAL_FEATURE_COLUMNS = [
    "cv_profession_norm",
    "vac_profession_norm",
    "cv_group_profession_norm",
    "vac_group_profession_norm",
    "cv_business_category_norm",
    "vac_business_category_norm",
    "cv_sfera_norm",
    "vac_sfera_norm",
    "cv_schedule_norm",
    "vac_schedule_norm",
    "cv_employment_type_norm",
    "vac_employment_type_norm",
    "cv_education_norm",
    "vac_education_level_norm",
    "cv_experience_common",
    "vac_experience_common",
]


DEBUG_TEXT_COLUMNS = list(CATEGORICAL_FEATURE_COLUMNS)


def bool_feature(left: pd.Series, right: pd.Series) -> pd.Series:
    """Return 1.0 if both values are known and equal, else 0.0."""
    known = left.notna() & right.notna()
    return (known & (left.astype("string") == right.astype("string"))).astype("float32")


def compatibility_to_float(
    series: pd.Series,
    incompatible_value: float = -1.0,
    unknown_value: float = 0.0,
) -> pd.Series:
    """Convert nullable boolean compatibility result to numeric feature.

    True  -> 1.0
    False -> incompatible_value
    NA    -> unknown_value
    """
    return (
        series
        .map({True: 1.0, False: incompatible_value})
        .fillna(unknown_value)
        .astype("float32")
    )


def add_pair_features(
    candidates: pd.DataFrame,
    cv_norm: pd.DataFrame,
    vacancies_norm: pd.DataFrame,
    incompatible_value: float = -1.0,
    unknown_value: float = 0.0,
    keep_debug_columns: bool = True,
) -> pd.DataFrame:
    """Add pair-level features to candidate pairs.

    Parameters
    ----------
    candidates:
        Candidate table with vacancy_id_hash and cv_id_hash.
    cv_norm:
        Normalized CV table from cv_normalized.parquet.
    vacancies_norm:
        Normalized vacancy table from vacancies_normalized.parquet.
    incompatible_value:
        Numeric value for explicit incompatibility.
    unknown_value:
        Numeric value for unknown compatibility.
    keep_debug_columns:
        If True, keep joined normalized categorical columns.
        Required for LogReg with OneHotEncoder.
        If False, keep only numeric/model-friendly columns.

    Returns
    -------
    pd.DataFrame
        Candidate table with additional feature columns.
    """
    required_candidates = {"vacancy_id_hash", "cv_id_hash"}
    missing = required_candidates - set(candidates.columns)
    if missing:
        raise KeyError(f"Missing columns in candidates: {sorted(missing)}")

    cv_cols = [
        "cv_id_hash",
        "profession_norm",
        "group_profession_norm",
        "business_category_norm",
        "sfera_norm",
        "schedule_norm",
        "employment_type_norm",
        "education_norm",
        "experience_common",
        "salary_bucketed",
    ]
    vacancy_cols = [
        "vacancy_id_hash",
        "profession_norm",
        "group_profession_norm",
        "business_category_norm",
        "sfera_norm",
        "schedule_norm",
        "employment_type_norm",
        "education_level_norm",
        "experience_common",
    ]

    missing_cv = set(cv_cols) - set(cv_norm.columns)
    missing_vac = set(vacancy_cols) - set(vacancies_norm.columns)
    if missing_cv:
        raise KeyError(f"Missing columns in cv_norm: {sorted(missing_cv)}")
    if missing_vac:
        raise KeyError(f"Missing columns in vacancies_norm: {sorted(missing_vac)}")

    result = (
        candidates
        .merge(
            cv_norm[cv_cols].add_prefix("cv_"),
            left_on="cv_id_hash",
            right_on="cv_cv_id_hash",
            how="left",
        )
        .merge(
            vacancies_norm[vacancy_cols].add_prefix("vac_"),
            left_on="vacancy_id_hash",
            right_on="vac_vacancy_id_hash",
            how="left",
        )
    )

    result["same_profession_norm"] = bool_feature(
        result["cv_profession_norm"],
        result["vac_profession_norm"],
    )
    result["same_group_profession_norm"] = bool_feature(
        result["cv_group_profession_norm"],
        result["vac_group_profession_norm"],
    )
    result["same_business_category_norm"] = bool_feature(
        result["cv_business_category_norm"],
        result["vac_business_category_norm"],
    )
    result["same_sfera_norm"] = bool_feature(
        result["cv_sfera_norm"],
        result["vac_sfera_norm"],
    )

    result["schedule_compatible_feature"] = compatibility_to_float(
        schedule_compatible_series(result["cv_schedule_norm"], result["vac_schedule_norm"]),
        incompatible_value=incompatible_value,
        unknown_value=unknown_value,
    )
    result["employment_type_compatible_feature"] = compatibility_to_float(
        employment_type_compatible_series(
            result["cv_employment_type_norm"],
            result["vac_employment_type_norm"],
        ),
        incompatible_value=incompatible_value,
        unknown_value=unknown_value,
    )
    result["education_compatible_feature"] = compatibility_to_float(
        education_compatible_series(
            result["cv_education_norm"],
            result["vac_education_level_norm"],
        ),
        incompatible_value=incompatible_value,
        unknown_value=unknown_value,
    )
    result["experience_compatible_feature"] = compatibility_to_float(
        experience_compatible_series(
            result["cv_experience_common"],
            result["vac_experience_common"],
        ),
        incompatible_value=incompatible_value,
        unknown_value=unknown_value,
    )

    result["salary_missing"] = result["cv_salary_bucketed"].isna().astype("float32")

    # Normalize numeric feature dtypes and missing values.
    for col in NUMERIC_FEATURE_COLUMNS:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0).astype("float32")

    # Normalize categorical feature dtypes and missing values.
    for col in CATEGORICAL_FEATURE_COLUMNS:
        if col in result.columns:
            result[col] = result[col].astype("string").fillna("unknown")

    if not keep_debug_columns:
        drop_cols = [
            col for col in DEBUG_TEXT_COLUMNS
            if col in result.columns
        ]
        drop_cols += [
            col for col in ["cv_cv_id_hash", "vac_vacancy_id_hash"]
            if col in result.columns
        ]
        result = result.drop(columns=drop_cols)

    return result


def get_numeric_feature_columns() -> list[str]:
    """Return default numeric features for simple ML baselines."""
    return list(NUMERIC_FEATURE_COLUMNS)


def get_categorical_feature_columns() -> list[str]:
    """Return processed categorical features for OHE-based models."""
    return list(CATEGORICAL_FEATURE_COLUMNS)


__all__ = [
    "CATEGORICAL_FEATURE_COLUMNS",
    "DEBUG_TEXT_COLUMNS",
    "NUMERIC_FEATURE_COLUMNS",
    "add_pair_features",
    "bool_feature",
    "compatibility_to_float",
    "get_categorical_feature_columns",
    "get_numeric_feature_columns",
]
