"""Logistic Regression ranker helpers for fixed LTR datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recommender.features import get_categorical_feature_columns


def sanitize_float(value: float) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def make_model_name(
    C: float,
    class_weight: str,
    solver: str,
    feature_set: str,
    penalty: str,
    l1_ratio: float | None,
) -> str:
    base = (
        f"logreg_{feature_set}_penalty{penalty}_"
        f"C{sanitize_float(C)}_cw{class_weight}_solver{solver}"
    )
    if penalty == "elasticnet":
        base += f"_l1r{sanitize_float(float(l1_ratio))}"
    return base


def get_feature_columns(
    feature_set: str,
    feature_config: dict[str, Any],
    train: pd.DataFrame,
    valid: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    numeric_cols = feature_config["numeric_feature_columns"]

    if feature_set == "numeric":
        categorical_cols: list[str] = []
    elif feature_set == "numeric_ohe":
        categorical_cols = get_categorical_feature_columns()
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    required_train = set(numeric_cols + categorical_cols + ["label"])
    required_valid = set(numeric_cols + categorical_cols)

    missing_train = required_train - set(train.columns)
    missing_valid = required_valid - set(valid.columns)

    if missing_train or missing_valid:
        message = []
        if missing_train:
            message.append(f"missing in train: {sorted(missing_train)}")
        if missing_valid:
            message.append(f"missing in valid: {sorted(missing_valid)}")
        if feature_set == "numeric_ohe":
            message.append(
                "For --feature-set numeric_ohe rebuild dataset with --keep-debug-columns."
            )
        raise KeyError("; ".join(message))

    return numeric_cols, categorical_cols


def prepare_xy(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    label_col: str = "label",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_cols = numeric_cols + categorical_cols

    X_train = train[feature_cols].copy()
    X_valid = valid[feature_cols].copy()

    for col in numeric_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(0)
        X_valid[col] = pd.to_numeric(X_valid[col], errors="coerce").fillna(0)

    for col in categorical_cols:
        X_train[col] = X_train[col].astype("string").fillna("unknown")
        X_valid[col] = X_valid[col].astype("string").fillna("unknown")

    y_train = train[label_col].astype(int)

    return X_train, y_train, X_valid


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def validate_solver_penalty(solver: str, penalty: str) -> None:
    """Raise clear errors for unsupported sklearn solver/penalty combinations."""
    if penalty in {"l1", "elasticnet"} and solver != "saga":
        raise ValueError(
            f"penalty={penalty} requires --solver saga in this script. "
            "Use --solver saga for L1/ElasticNet experiments."
        )
    if penalty == "elasticnet" and solver != "saga":
        raise ValueError("ElasticNet is supported only with solver=saga.")
    if penalty == "l2" and solver not in {"lbfgs", "liblinear", "saga", "sag", "newton-cg", "newton-cholesky"}:
        raise ValueError(f"Unsupported solver for l2: {solver}")


def build_logreg_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    C: float,
    class_weight_name: str,
    solver: str,
    penalty: str,
    l1_ratio: float | None,
    max_iter: int,
    verbose: int,
) -> Pipeline:
    validate_solver_penalty(solver=solver, penalty=penalty)

    class_weight = None if class_weight_name == "none" else "balanced"

    transformers = []
    if numeric_cols:
        # with_mean=False keeps the output sparse-compatible when combined with OHE.
        transformers.append(("num", StandardScaler(with_mean=False), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", make_one_hot_encoder(), categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )

    model_kwargs = {
        "C": C,
        "penalty": penalty,
        "class_weight": class_weight,
        "solver": solver,
        "max_iter": max_iter,
        "verbose": verbose,
    }
    if penalty == "elasticnet":
        model_kwargs["l1_ratio"] = l1_ratio

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(**model_kwargs)),
        ]
    )


def iter_model_configs(
    Cs: list[float],
    penalties: list[str],
    class_weights: list[str],
    l1_ratios: list[float],
) -> list[dict[str, Any]]:
    """Build model configs, expanding l1_ratio only for ElasticNet."""

    configs = []

    for penalty in penalties:
        ratios: list[float | None]
        if penalty == "elasticnet":
            ratios = [float(x) for x in l1_ratios]
        else:
            ratios = [None]

        for C in Cs:
            for class_weight in class_weights:
                for l1_ratio in ratios:
                    configs.append(
                        {
                            "C": float(C),
                            "penalty": penalty,
                            "class_weight": class_weight,
                            "l1_ratio": l1_ratio,
                        }
                    )

    return configs


def add_model_scores_and_ranks(valid: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    result = valid.copy(deep=False)
    result["model_score"] = scores.astype("float32")

    result = result.sort_values(
        ["vacancy_id_hash", "model_score", "embedding_score", "embedding_rank"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )

    result["model_rank"] = result.groupby("vacancy_id_hash").cumcount() + 1
    result["model_rank"] = result["model_rank"].astype(np.int32)

    return result


def predict_positive_proba_in_chunks(
    model: Pipeline,
    X: pd.DataFrame,
    chunk_size: int = 100_000,
) -> np.ndarray:
    """Predict positive-class probabilities in row chunks."""
    scores = []

    for start in range(0, len(X), chunk_size):
        stop = min(start + chunk_size, len(X))
        chunk_scores = model.predict_proba(X.iloc[start:stop])[:, 1]
        scores.append(chunk_scores)

    return np.concatenate(scores)