"""Shared ranking metrics for vacancy-to-CV recommendations.

Evaluates ranked CV candidates against historical apply pairs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


def build_positives_by_query(
    positive_pairs: pd.DataFrame,
    query_col: str = "vacancy_id_hash",
    item_col: str = "cv_id_hash",
) -> dict[object, set[object]]:
    """Group real apply pairs by vacancy."""

    required = {query_col, item_col}
    missing = required - set(positive_pairs.columns)
    if missing:
        raise KeyError(f"Missing columns in positive_pairs: {sorted(missing)}")

    positives: dict[object, set[object]] = defaultdict(set)
    deduped = positive_pairs[[query_col, item_col]].drop_duplicates()

    for query_id, item_id in deduped.itertuples(index=False):
        positives[query_id].add(item_id)

    return dict(positives)


def _ranking_metrics_for_one_query(
    ranked_items: list[object],
    positives: set[object],
    k: int,
) -> dict[str, float]:
    """Compute ranking metrics for one query with binary relevance."""
    if k <= 0:
        raise ValueError("k must be positive")

    top_items = ranked_items[:k]
    if not positives:
        return {
            "recall": 0.0,
            "hit": 0.0,
            "precision": 0.0,
            "mrr": 0.0,
            "ap": 0.0,
            "ndcg": 0.0,
            "recovered": 0.0,
        }

    recovered = 0
    dcg = 0.0
    precision_sum_at_hits = 0.0
    reciprocal_rank = 0.0

    for rank, item_id in enumerate(top_items, start=1):
        if item_id in positives:
            recovered += 1
            dcg += 1.0 / np.log2(rank + 1)

            precision_at_rank = recovered / rank
            precision_sum_at_hits += precision_at_rank

            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / rank

    max_hits_possible = min(len(positives), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, max_hits_possible + 1))

    recall = recovered / len(positives)
    hit = 1.0 if recovered > 0 else 0.0
    precision = recovered / k
    ap = precision_sum_at_hits / max_hits_possible if max_hits_possible > 0 else 0.0
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return {
        "recall": recall,
        "hit": hit,
        "precision": precision,
        "mrr": reciprocal_rank,
        "ap": ap,
        "ndcg": ndcg,
        "recovered": float(recovered),
    }


def evaluate_ranked_candidates(
    candidates: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    rank_col: str,
    ks: Iterable[int],
    n_items_total: int,
    query_col: str = "vacancy_id_hash",
    item_col: str = "cv_id_hash",
    ranking_name: str = "ranking",
) -> pd.DataFrame:
    """Evaluate ranked candidates against real apply pairs.

    Returns one metrics row per K.
    """
    if n_items_total <= 0:
        raise ValueError("n_items_total must be positive")

    required_candidates = {query_col, item_col, rank_col}
    missing_candidates = required_candidates - set(candidates.columns)
    if missing_candidates:
        raise KeyError(f"Missing columns in candidates: {sorted(missing_candidates)}")

    positives_by_query = build_positives_by_query(
        positive_pairs=positive_pairs,
        query_col=query_col,
        item_col=item_col,
    )

    ks = sorted(set(int(k) for k in ks if int(k) > 0))
    if not ks:
        raise ValueError("ks must contain at least one positive integer")

    positive_pair_total = sum(len(items) for items in positives_by_query.values())
    query_with_positive_total = len(positives_by_query)

    if positive_pair_total == 0 or query_with_positive_total == 0:
        raise ValueError("positive_pairs has no positive pairs to evaluate")

    # Sort once before evaluating all K values.
    sorted_candidates = candidates[[query_col, item_col, rank_col]].sort_values(
        [query_col, rank_col],
        ascending=[True, True],
        kind="mergesort",
    )

    ranked_items_by_query: dict[object, list[object]] = {
        query_id: part[item_col].tolist()
        for query_id, part in sorted_candidates.groupby(query_col, sort=False)
    }

    candidate_counts = sorted_candidates.groupby(query_col).size()
    queries_without_candidates = sum(
        1 for query_id in positives_by_query if query_id not in ranked_items_by_query
    )

    rows = []
    for k in ks:
        recovered_pairs = 0
        recall_sum = 0.0
        hit_sum = 0.0
        precision_sum = 0.0
        mrr_sum = 0.0
        map_sum = 0.0
        ndcg_sum = 0.0

        for query_id, positives in positives_by_query.items():
            ranked_items = ranked_items_by_query.get(query_id, [])
            query_metrics = _ranking_metrics_for_one_query(
                ranked_items=ranked_items,
                positives=positives,
                k=k,
            )

            recovered_pairs += int(query_metrics["recovered"])
            recall_sum += query_metrics["recall"]
            hit_sum += query_metrics["hit"]
            precision_sum += query_metrics["precision"]
            mrr_sum += query_metrics["mrr"]
            map_sum += query_metrics["ap"]
            ndcg_sum += query_metrics["ndcg"]

        micro_recall = recovered_pairs / positive_pair_total * 100
        macro_recall = recall_sum / query_with_positive_total * 100
        hit_rate = hit_sum / query_with_positive_total * 100
        precision_at_k = precision_sum / query_with_positive_total * 100
        mrr_at_k = mrr_sum / query_with_positive_total
        map_at_k = map_sum / query_with_positive_total
        ndcg_at_k = ndcg_sum / query_with_positive_total

        effective_random_k = min(k, n_items_total)
        expected_random_recall = effective_random_k / n_items_total * 100

        rows.append(
            {
                "ranking": ranking_name,
                "K": k,
                "validation_vacancies_with_positive": query_with_positive_total,
                "validation_positive_pairs": positive_pair_total,
                "recovered_positive_pairs": recovered_pairs,
                "micro_recall_positive_pairs_pct": micro_recall,
                "macro_recall_per_vacancy_pct": macro_recall,
                "hit_rate_vacancies_with_positive_pct": hit_rate,
                "ndcg_at_k": ndcg_at_k,
                "mrr_at_k": mrr_at_k,
                "map_at_k": map_at_k,
                "precision_at_k_pct": precision_at_k,
                "vacancies_without_candidates_pct": (
                    queries_without_candidates / query_with_positive_total * 100
                ),
                "mean_candidates_per_vacancy": float(candidate_counts.mean())
                if len(candidate_counts)
                else 0.0,
                "median_candidates_per_vacancy": float(candidate_counts.median())
                if len(candidate_counts)
                else 0.0,
                "expected_random_recall_pct": expected_random_recall,
                "lift_over_random_micro_recall": (
                    micro_recall / expected_random_recall
                    if expected_random_recall > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


__all__ = [
    "build_positives_by_query",
    "evaluate_ranked_candidates",
]
