from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from recommender.metrics import evaluate_ranked_candidates


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