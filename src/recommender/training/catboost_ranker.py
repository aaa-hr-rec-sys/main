"""CatBoost ranker helpers for fixed LTR datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from recommender.features import get_categorical_feature_columns

def sanitize_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def make_model_name(
    loss_function: str,
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    one_hot_max_size: int,
    label_weighting: str,
) -> str:
    return (
        f"catboost_ranker_loss{loss_function}_"
        f"it{iterations}_depth{depth}_"
        f"lr{sanitize_float(learning_rate)}_"
        f"l2{sanitize_float(l2_leaf_reg)}_"
        f"onehot{one_hot_max_size}_"
        f"w{label_weighting}"
    )


def get_feature_columns(
    feature_config: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    """Return numeric, categorical and combined feature columns."""
    numeric_cols = feature_config["numeric_feature_columns"]
    categorical_cols = get_categorical_feature_columns()
    feature_cols = numeric_cols + categorical_cols

    required_train = set(feature_cols + ["label", "vacancy_id_hash"])
    required_valid = set(feature_cols + ["label", "vacancy_id_hash"])

    missing_train = required_train - set(train.columns)
    missing_valid = required_valid - set(valid.columns)

    if missing_train or missing_valid:
        message = []
        if missing_train:
            message.append(f"missing in train: {sorted(missing_train)}")
        if missing_valid:
            message.append(f"missing in valid: {sorted(missing_valid)}")
        message.append(
            "For CatBoost native categorical features, rebuild dataset with --keep-debug-columns."
        )
        raise KeyError("; ".join(message))

    return numeric_cols, categorical_cols, feature_cols


def prepare_pool_dataframe(
    df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Prepare feature DataFrame for CatBoost."""
    X = df[feature_cols].copy()

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0).astype("float32")

    for col in categorical_cols:
        X[col] = X[col].astype("string").fillna("unknown").astype(str)

    return X


def sort_by_group(df: pd.DataFrame, group_col: str = "vacancy_id_hash") -> pd.DataFrame:
    """CatBoost Pool with group_id requires objects of the same group to be contiguous."""
    return df.sort_values(
        [group_col, "label", "embedding_rank"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def make_label_weights(
    labels: pd.Series | np.ndarray,
    label_weighting: str,
) -> np.ndarray | None:
    """Create optional positive-label weights for ranking data:
    - none: no weights
    - balanced: positive weight = n_negative / n_positive
    - sqrt_balanced: positive weight = sqrt(n_negative / n_positive)

    Negative weight is always 1.0.
    """
    if label_weighting == "none":
        return None

    y = np.asarray(labels).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    if n_pos == 0 or n_neg == 0:
        return np.ones(len(y), dtype="float32")

    ratio = n_neg / n_pos
    if label_weighting == "balanced":
        pos_weight = ratio
    elif label_weighting == "sqrt_balanced":
        pos_weight = float(np.sqrt(ratio))
    else:
        raise ValueError(f"Unknown label_weighting: {label_weighting}")

    weights = np.ones(len(y), dtype="float32")
    weights[y == 1] = np.float32(pos_weight)
    return weights


def make_pool(
    df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    label_weighting: str = "none",
) -> Pool:
    """Create CatBoost Pool with group_id and optional object weights."""
    sorted_df = sort_by_group(df)

    X = prepare_pool_dataframe(
        sorted_df,
        feature_cols=feature_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
    y = sorted_df["label"].astype(float).to_numpy()
    group_id = sorted_df["vacancy_id_hash"].astype(str).to_numpy()
    weights = make_label_weights(sorted_df["label"], label_weighting=label_weighting)

    return Pool(
        data=X,
        label=y,
        group_id=group_id,
        cat_features=categorical_cols,
        weight=weights,
    )


def make_prediction_frame(
    valid: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Add model scores/ranks without sorting the full validation table.

    The full table may contain large Arrow-backed categorical columns, so metrics
    and recommendations use only ids, ranks, scores, and compact feature columns.
    """
    PREDICTION_KEEP_COLUMNS = [
        "vacancy_idx",
        "cv_idx",
        "vacancy_id_hash",
        "cv_id_hash",
        "embedding_score",
        "embedding_rank",
        "label",
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
    keep_cols = [col for col in PREDICTION_KEEP_COLUMNS if col in valid.columns]
    result = valid[keep_cols].copy()

    scores = np.asarray(scores)
    if len(scores) != len(result):
        raise ValueError(f"Score length mismatch: scores={len(scores)}, rows={len(result)}")

    result["model_score"] = scores.astype("float32")

    group_col = "vacancy_idx" if "vacancy_idx" in result.columns else "vacancy_id_hash"

    result = result.sort_values(
        [group_col, "model_score", "embedding_score", "embedding_rank"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).copy()

    result["model_rank"] = result.groupby(group_col, sort=False).cumcount() + 1
    result["model_rank"] = result["model_rank"].astype(np.int32)

    return result


def iter_model_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = []
    for loss_function in args.loss_function:
        for iterations in args.iterations:
            for depth in args.depth:
                for learning_rate in args.learning_rate:
                    for l2_leaf_reg in args.l2_leaf_reg:
                        for one_hot_max_size in args.one_hot_max_size:
                            for label_weighting in args.label_weighting:
                                configs.append(
                                    {
                                        "loss_function": loss_function,
                                        "iterations": int(iterations),
                                        "depth": int(depth),
                                        "learning_rate": float(learning_rate),
                                        "l2_leaf_reg": float(l2_leaf_reg),
                                        "one_hot_max_size": int(one_hot_max_size),
                                        "label_weighting": label_weighting,
                                    }
                                )
    return configs


def build_ranker(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> CatBoostRanker:
    params = {
        "loss_function": config["loss_function"],
        "iterations": config["iterations"],
        "depth": config["depth"],
        "learning_rate": config["learning_rate"],
        "l2_leaf_reg": config["l2_leaf_reg"],
        "one_hot_max_size": config["one_hot_max_size"],
        "random_seed": args.random_seed,
        "thread_count": args.thread_count,
        "task_type": args.task_type,
        "verbose": args.catboost_verbose,
        "allow_writing_files": False,
    }

    if args.early_stopping_rounds and args.early_stopping_rounds > 0:
        params["early_stopping_rounds"] = args.early_stopping_rounds
        params["use_best_model"] = True

    return CatBoostRanker(**params)


def save_feature_importance(
    model: CatBoostRanker,
    pool: Pool,
    output_path: Path,
) -> None:
    """Best-effort feature importance export."""
    try:
        importance = model.get_feature_importance(pool, prettified=True)
        if isinstance(importance, pd.DataFrame):
            importance.to_csv(output_path, index=False)
        else:
            pd.DataFrame({"importance": importance}).to_csv(output_path, index=False)
    except Exception as exc:
        print(f"Warning: could not save feature importance: {exc}", flush=True)
