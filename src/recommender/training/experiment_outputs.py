from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommender.inference import make_recommendation_output, save_recommendations
from recommender.training.common import json_default, save_json


def get_selection_value(
    metrics: pd.DataFrame,
    select_metric: str,
    select_k: int,
) -> float:
    """Return the model-selection metric value for the requested K."""
    selection_rows = metrics[metrics["K"] == select_k]
    if selection_rows.empty or select_metric not in selection_rows.columns:
        selection_value = (
            float(metrics[select_metric].mean())
            if select_metric in metrics.columns
            else -np.inf
        )
    else:
        selection_value = float(selection_rows.iloc[0][select_metric])

    return selection_value


def save_recommendation_outputs(
    candidates: pd.DataFrame,
    cv_table: pd.DataFrame,
    vacancies_table: pd.DataFrame,
    output_dir: Path,
    filename_stem: str,
    rank_col: str,
    score_col: str,
    k_inf: int,
    csv_limit: int,
    keep_feature_columns: bool,
) -> dict[str, Any]:
    """Build and save recommendation files for experiment."""
    recommendations = make_recommendation_output(
        candidates=candidates,
        cv_table=cv_table,
        vacancies_table=vacancies_table,
        rank_col=rank_col,
        score_col=score_col,
        k_inf=k_inf,
        keep_feature_columns=keep_feature_columns,
    )
    return save_recommendations(
        recommendations=recommendations,
        output_dir=output_dir,
        filename_stem=filename_stem,
        csv_limit=csv_limit,
    )


def save_metrics_outputs(
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save metrics as CSV and JSON experiment outputs."""
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    save_json(
        {"metrics": metrics.to_dict(orient="records")},
        output_dir / "metrics.json",
        default=json_default,
    )
