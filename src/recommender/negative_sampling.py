"""Helpers for labeling and sampling ranking training pairs.

Positive pairs are real applies. Negative pairs are sampled unobserved
vacancy-CV pairs, not guaranteed ground-truth negatives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_binary_label(
    candidates: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    query_col: str = "vacancy_id_hash",
    item_col: str = "cv_id_hash",
    label_col: str = "label",
) -> pd.DataFrame:
    """Add binary label to candidate pairs.

    label = 1 if pair is present in positive_pairs, else 0.
    """
    required_candidates = {query_col, item_col}
    required_positives = {query_col, item_col}

    missing_candidates = required_candidates - set(candidates.columns)
    missing_positives = required_positives - set(positive_pairs.columns)

    if missing_candidates:
        raise KeyError(f"Missing columns in candidates: {sorted(missing_candidates)}")
    if missing_positives:
        raise KeyError(f"Missing columns in positive_pairs: {sorted(missing_positives)}")

    positives = positive_pairs[[query_col, item_col]].drop_duplicates().copy()
    positives[label_col] = 1

    result = candidates.merge(
        positives,
        on=[query_col, item_col],
        how="left",
    )
    result[label_col] = result[label_col].fillna(0).astype("int8")

    return result


def sample_hard_negatives_per_query(
    labeled_candidates: pd.DataFrame,
    negative_ratio: int = 5,
    query_col: str = "vacancy_id_hash",
    label_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """Keep positives and sample negatives from the candidate pool."""

    if negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive")

    required = {query_col, label_col}
    missing = required - set(labeled_candidates.columns)
    if missing:
        raise KeyError(f"Missing columns in labeled_candidates: {sorted(missing)}")

    rng = np.random.default_rng(random_state)
    parts = []

    for _, group in labeled_candidates.groupby(query_col, sort=False):
        positives = group[group[label_col] == 1]
        if positives.empty:
            continue

        negatives = group[group[label_col] == 0]
        n_negatives = min(len(negatives), len(positives) * negative_ratio)

        if n_negatives > 0:
            sampled_idx = rng.choice(
                negatives.index.to_numpy(),
                size=n_negatives,
                replace=False,
            )
            sampled_negatives = negatives.loc[sampled_idx]
            parts.append(pd.concat([positives, sampled_negatives], ignore_index=False))
        else:
            parts.append(positives)

    if not parts:
        return labeled_candidates.iloc[0:0].copy()

    sampled = pd.concat(parts, ignore_index=True)

    order = rng.permutation(len(sampled))
    sampled = sampled.iloc[order].reset_index(drop=True)

    return sampled


def _sample_easy_negative_cv_indices(
    rng: np.random.Generator,
    n_cv_total: int,
    forbidden_cv_idx: set[int],
    n_needed: int,
    max_attempt_multiplier: int = 20,
) -> list[int]:
    """Sample CV indices outside the forbidden set."""

    if n_needed <= 0:
        return []

    if len(forbidden_cv_idx) >= n_cv_total:
        return []

    result: list[int] = []
    result_set: set[int] = set()

    max_attempts = max(n_needed * max_attempt_multiplier, 100)
    attempts = 0

    while len(result) < n_needed and attempts < max_attempts:
        attempts += 1
        candidate = int(rng.integers(0, n_cv_total))
        if candidate in forbidden_cv_idx or candidate in result_set:
            continue
        result.append(candidate)
        result_set.add(candidate)

    # Fill the remaining quota from all available CVs.
    if len(result) < n_needed:
        available = np.array(
            list(set(range(n_cv_total)) - forbidden_cv_idx - result_set),
            dtype=np.int32,
        )
        if len(available) > 0:
            take = min(n_needed - len(result), len(available))
            extra = rng.choice(available, size=take, replace=False)
            result.extend([int(x) for x in extra])

    return result


def sample_mixed_negatives_per_query(
    labeled_topk_candidates: pd.DataFrame,
    positive_pairs: pd.DataFrame,
    vacancy_matrix: np.ndarray,
    cv_matrix: np.ndarray,
    vacancies_embeddings: pd.DataFrame,
    cv_embeddings: pd.DataFrame,
    hard_negative_ratio: int = 5,
    easy_negative_ratio: int = 5,
    hard_min_rank: int = 50,
    easy_embedding_rank_value: int | None = None,
    query_col: str = "vacancy_id_hash",
    item_col: str = "cv_id_hash",
    label_col: str = "label",
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample positives with hard and easy negatives.

    Hard negatives come from the candidate pool. Easy negatives are random CVs
    outside the pool and outside known positives.
    """
    if hard_negative_ratio < 0:
        raise ValueError("hard_negative_ratio must be non-negative")
    if easy_negative_ratio < 0:
        raise ValueError("easy_negative_ratio must be non-negative")
    if hard_negative_ratio == 0 and easy_negative_ratio == 0:
        raise ValueError("At least one of hard/easy negative ratios must be positive")
    if hard_min_rank <= 0:
        raise ValueError("hard_min_rank must be positive")

    required = {
        query_col,
        item_col,
        "vacancy_idx",
        "cv_idx",
        "embedding_score",
        "embedding_rank",
        label_col,
    }
    missing = required - set(labeled_topk_candidates.columns)
    if missing:
        raise KeyError(f"Missing columns in labeled_topk_candidates: {sorted(missing)}")

    n_cv_total = int(cv_matrix.shape[0])
    rng = np.random.default_rng(random_state)

    if easy_embedding_rank_value is None:
        easy_embedding_rank_value = int(labeled_topk_candidates["embedding_rank"].max()) + 1

    cv_ids = cv_embeddings["cv_id_hash"].to_numpy()

    # Known positives include pairs outside the candidate pool.
    pos_idx = positive_pairs[["vacancy_idx", "cv_idx"]].drop_duplicates()
    positives_by_vacancy_idx: dict[int, set[int]] = (
        pos_idx.groupby("vacancy_idx")["cv_idx"]
        .apply(lambda s: set(map(int, s.to_numpy())))
        .to_dict()
    )

    parts = []
    easy_parts = []

    grouped = labeled_topk_candidates.groupby("vacancy_idx", sort=False)

    for vacancy_idx, group in grouped:
        vacancy_idx_int = int(vacancy_idx)

        positives_in_topk = group[group[label_col] == 1]
        if positives_in_topk.empty:
            continue

        n_pos = len(positives_in_topk)

        # Always keep observed positives in the top-K candidate pool.
        positives_to_keep = positives_in_topk.copy()
        positives_to_keep["sample_source"] = "positive"
        parts.append(positives_to_keep)

        # Hard negatives: not from the very top ranks.
        if hard_negative_ratio > 0:
            hard_pool = group[
                (group[label_col] == 0)
                & (group["embedding_rank"] >= hard_min_rank)
            ]
            n_hard = min(len(hard_pool), n_pos * hard_negative_ratio)

            if n_hard > 0:
                sampled_idx = rng.choice(
                    hard_pool.index.to_numpy(),
                    size=n_hard,
                    replace=False,
                )
                hard_sample = hard_pool.loc[sampled_idx].copy()
                hard_sample["sample_source"] = "hard_topk_filtered"
                parts.append(hard_sample)

        # Easy negatives: random CV outside top-K and outside positives.
        if easy_negative_ratio > 0:
            n_easy_target = n_pos * easy_negative_ratio

            topk_cv_idx = set(map(int, group["cv_idx"].to_numpy()))
            positive_cv_idx = positives_by_vacancy_idx.get(vacancy_idx_int, set())
            forbidden_cv_idx = topk_cv_idx | positive_cv_idx

            sampled_easy_cv_idx = _sample_easy_negative_cv_indices(
                rng=rng,
                n_cv_total=n_cv_total,
                forbidden_cv_idx=forbidden_cv_idx,
                n_needed=n_easy_target,
            )

            if sampled_easy_cv_idx:
                vacancy_vector = vacancy_matrix[vacancy_idx_int].astype("float32")
                easy_cv_matrix = cv_matrix[np.array(sampled_easy_cv_idx, dtype=np.int32)].astype("float32")
                easy_scores = easy_cv_matrix @ vacancy_vector

                vacancy_id = group[query_col].iloc[0]

                easy_part = pd.DataFrame(
                    {
                        "vacancy_idx": np.full(len(sampled_easy_cv_idx), vacancy_idx_int, dtype=np.int32),
                        "cv_idx": np.array(sampled_easy_cv_idx, dtype=np.int32),
                        query_col: vacancy_id,
                        item_col: cv_ids[np.array(sampled_easy_cv_idx, dtype=np.int32)],
                        "embedding_score": easy_scores.astype("float32"),
                        "embedding_rank": np.full(
                            len(sampled_easy_cv_idx),
                            easy_embedding_rank_value,
                            dtype=np.int32,
                        ),
                        label_col: np.zeros(len(sampled_easy_cv_idx), dtype=np.int8),
                        "sample_source": "easy_random_outside_topk",
                    }
                )
                easy_parts.append(easy_part)

    all_parts = parts + easy_parts
    if not all_parts:
        return labeled_topk_candidates.iloc[0:0].copy()

    sampled = pd.concat(all_parts, ignore_index=True)

    order = rng.permutation(len(sampled))
    sampled = sampled.iloc[order].reset_index(drop=True)

    return sampled


def label_summary(
    labeled_data: pd.DataFrame,
    label_col: str = "label",
) -> dict[str, int | float]:
    """Summarize label counts."""
    rows = len(labeled_data)
    positives = int((labeled_data[label_col] == 1).sum())
    negatives = int((labeled_data[label_col] == 0).sum())

    return {
        "rows": int(rows),
        "positives": positives,
        "negatives": negatives,
        "positive_pct": positives / rows * 100 if rows else 0.0,
        "negative_pct": negatives / rows * 100 if rows else 0.0,
    }


def sample_source_summary(
    data: pd.DataFrame,
    source_col: str = "sample_source",
) -> dict[str, int]:
    """Return row counts by sample source."""
    if source_col not in data.columns:
        return {}
    return {
        str(k): int(v)
        for k, v in data[source_col].value_counts(dropna=False).to_dict().items()
    }


__all__ = [
    "add_binary_label",
    "label_summary",
    "sample_hard_negatives_per_query",
    "sample_mixed_negatives_per_query",
    "sample_source_summary",
]
