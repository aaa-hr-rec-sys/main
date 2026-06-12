"""Build top-K recommendation outputs.

Works with embedding, rule-based, and ML rankers by accepting rank and score
column names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_CV_DETAIL_COLUMNS = [
    "cv_id_hash",
    "created_at_jittered",
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
    "profession_norm",
    "group_profession_norm",
    "business_category_norm",
    "sfera_norm",
    "schedule_norm",
    "employment_type_norm",
    "education_norm",
    "experience_common",
]

DEFAULT_VACANCY_DETAIL_COLUMNS = [
    "vacancy_id_hash",
    "published_at_jittered",
    "profession",
    "group_profession",
    "business_category",
    "sfera",
    "experience",
    "schedule",
    "employment_type",
    "education_level",
    "profession_norm",
    "group_profession_norm",
    "business_category_norm",
    "sfera_norm",
    "schedule_norm",
    "employment_type_norm",
    "education_level_norm",
    "experience_common",
]

DEFAULT_FEATURE_COLUMNS = [
    "same_profession_norm",
    "same_group_profession_norm",
    "same_business_category_norm",
    "same_sfera_norm",
    "experience_compatible_feature",
    "schedule_compatible_feature",
    "employment_type_compatible_feature",
    "education_compatible_feature",
    "salary_missing",
    "rule_bonus",
]


def _existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return columns that exist in df, preserving the requested order."""
    return [col for col in columns if col in df.columns]


def select_top_k_recommendations(
    candidates: pd.DataFrame,
    rank_col: str,
    score_col: str,
    k_inf: int = 20,
) -> pd.DataFrame:
    """Select top-k recommendations per vacancy and add rank/score columns."""
    if k_inf <= 0:
        raise ValueError("k_inf must be positive")

    required = {"vacancy_id_hash", "cv_id_hash", rank_col, score_col}
    missing = required - set(candidates.columns)
    if missing:
        raise KeyError(f"Missing columns in candidates: {sorted(missing)}")

    recs = candidates.loc[candidates[rank_col] <= k_inf].copy()

    recs["rank"] = recs[rank_col].astype("int32")
    recs["score"] = recs[score_col]

    recs = recs.sort_values(
        ["vacancy_id_hash", "rank", "score"],
        ascending=[True, True, False],
        kind="mergesort",
    ).copy()

    return recs


def attach_cv_details(
    recommendations: pd.DataFrame,
    cv_table: pd.DataFrame,
    detail_columns: list[str] | None = None,
    prefix: str = "cv_",
) -> pd.DataFrame:
    """Attach human-readable CV fields to recommendation rows."""
    detail_columns = detail_columns or DEFAULT_CV_DETAIL_COLUMNS
    cols = _existing_columns(cv_table, detail_columns)

    if "cv_id_hash" not in cols:
        raise KeyError("cv_table must contain cv_id_hash")

    details = cv_table[cols].copy()
    details = details.add_prefix(prefix)

    result = recommendations.copy()
    columns_to_replace = [
        col for col in details.columns
        if col in result.columns and col != f"{prefix}cv_id_hash"
    ]
    if columns_to_replace:
        result = result.drop(columns=columns_to_replace)

    result = result.merge(
        details,
        left_on="cv_id_hash",
        right_on=f"{prefix}cv_id_hash",
        how="left",
    )

    return result


def attach_vacancy_details(
    recommendations: pd.DataFrame,
    vacancies_table: pd.DataFrame,
    detail_columns: list[str] | None = None,
    prefix: str = "vac_",
) -> pd.DataFrame:
    """Attach human-readable vacancy fields to recommendation rows."""
    detail_columns = detail_columns or DEFAULT_VACANCY_DETAIL_COLUMNS
    cols = _existing_columns(vacancies_table, detail_columns)

    if "vacancy_id_hash" not in cols:
        raise KeyError("vacancies_table must contain vacancy_id_hash")

    details = vacancies_table[cols].copy()
    details = details.add_prefix(prefix)

    result = recommendations.copy()
    columns_to_replace = [
        col for col in details.columns
        if col in result.columns and col != f"{prefix}vacancy_id_hash"
    ]
    if columns_to_replace:
        result = result.drop(columns=columns_to_replace)

    result = result.merge(
        details,
        left_on="vacancy_id_hash",
        right_on=f"{prefix}vacancy_id_hash",
        how="left",
    )

    return result


def make_recommendation_output(
    candidates: pd.DataFrame,
    cv_table: pd.DataFrame,
    vacancies_table: pd.DataFrame,
    rank_col: str,
    score_col: str,
    k_inf: int = 20,
    keep_feature_columns: bool = True,
) -> pd.DataFrame:
    """Create top-k recommendation table."""

    recs = select_top_k_recommendations(
        candidates=candidates,
        rank_col=rank_col,
        score_col=score_col,
        k_inf=k_inf,
    )

    base_cols = [
        "vacancy_id_hash",
        "cv_id_hash",
        "rank",
        "score",
        "embedding_score",
        "embedding_rank",
        "final_score",
        "final_rank",
        "model_score",
        "model_rank",
    ]
    keep_cols = _existing_columns(recs, base_cols)

    if keep_feature_columns:
        keep_cols += [
            col for col in DEFAULT_FEATURE_COLUMNS
            if col in recs.columns and col not in keep_cols
        ]

    recs = recs[keep_cols].copy()

    recs = attach_cv_details(recs, cv_table=cv_table)
    recs = attach_vacancy_details(recs, vacancies_table=vacancies_table)

    # Put key ranking and detail columns first.
    preferred_order = [
        "vacancy_id_hash",
        "cv_id_hash",
        "rank",
        "score",
        "embedding_score",
        "embedding_rank",
        "final_score",
        "final_rank",
        "model_score",
        "model_rank",
        "rule_bonus",
        "same_profession_norm",
        "same_group_profession_norm",
        "same_business_category_norm",
        "experience_compatible_feature",
        "schedule_compatible_feature",
        "employment_type_compatible_feature",
        "education_compatible_feature",
        "cv_profession",
        "vac_profession",
        "cv_group_profession",
        "vac_group_profession",
        "cv_business_category",
        "vac_business_category",
        "cv_experience_bucket",
        "vac_experience",
        "cv_education",
        "vac_education_level",
        "cv_schedule",
        "vac_schedule",
        "cv_employment_type",
        "vac_employment_type",
        "cv_salary_bucketed",
    ]
    ordered = [col for col in preferred_order if col in recs.columns]
    remaining = [col for col in recs.columns if col not in ordered]
    recs = recs[ordered + remaining]

    recs = recs.sort_values(
        ["vacancy_id_hash", "rank"],
        ascending=[True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return recs


def save_recommendations(
    recommendations: pd.DataFrame,
    output_dir: Path,
    filename_stem: str,
    csv_limit: int = 20_000,
) -> dict[str, str | int | None]:
    """Save recommendations to parquet and optionally CSV.

    csv_limit:
    - csv_limit > 0: save first N rows to CSV
    - csv_limit == -1: save all rows to CSV
    - csv_limit == 0: do not save CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / f"{filename_stem}.parquet"
    recommendations.to_parquet(parquet_path, index=False)

    csv_path = None
    csv_rows = 0

    if csv_limit != 0:
        if csv_limit < 0:
            csv_df = recommendations
        else:
            csv_df = recommendations.head(csv_limit)

        csv_path = output_dir / f"{filename_stem}.csv"
        csv_df.to_csv(csv_path, index=False)
        csv_rows = len(csv_df)

    return {
        "parquet_path": str(parquet_path),
        "csv_path": str(csv_path) if csv_path is not None else None,
        "rows_total": int(len(recommendations)),
        "csv_rows": int(csv_rows),
    }


__all__ = [
    "DEFAULT_CV_DETAIL_COLUMNS",
    "DEFAULT_FEATURE_COLUMNS",
    "DEFAULT_VACANCY_DETAIL_COLUMNS",
    "attach_cv_details",
    "attach_vacancy_details",
    "make_recommendation_output",
    "save_recommendations",
    "select_top_k_recommendations",
]
