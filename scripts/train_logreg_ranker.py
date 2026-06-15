"""Train Logistic Regression rankers on a fixed LTR dataset.

Supports numeric features and optional one-hot encoded categorical features.
Runs one or more regularization configs and saves metrics, models, and summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MAIN_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MAIN_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import joblib
    import sklearn
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Для train_logreg_ranker.py нужны scikit-learn и joblib. "
        "Установите: python -m pip install scikit-learn joblib"
    ) from exc

from recommender.features import get_categorical_feature_columns
from recommender.inference import make_recommendation_output, save_recommendations
from recommender.metrics import evaluate_ranked_candidates


class StepTimer:
    """Print start/end logs and elapsed time for a pipeline step."""

    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        print(f"\n[START] {self.name}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        print(f"[DONE]  {self.name}: {elapsed:.1f}s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Logistic Regression ranker")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/modeling/ltr_v1_top500_neg5"),
        help="Directory created by scripts/build_ltr_dataset.py",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments"))
    parser.add_argument("--model-root", type=Path, default=Path("data/models"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument(
        "--feature-set",
        type=str,
        default="numeric",
        choices=["numeric", "numeric_ohe"],
        help="numeric = numeric features only; numeric_ohe = numeric + processed categorical OneHot",
    )
    parser.add_argument(
        "--C",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0],
        help="Inverse regularization strength. Smaller C means stronger regularization.",
    )
    parser.add_argument(
        "--penalty",
        type=str,
        nargs="+",
        default=["l2"],
        choices=["l2", "l1", "elasticnet"],
        help="Regularization type. For l1/elasticnet use solver=saga.",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        nargs="+",
        default=[0.5],
        help="Only used for penalty=elasticnet. Ignored for l1/l2.",
    )
    parser.add_argument(
        "--class-weight",
        type=str,
        nargs="+",
        default=["none", "balanced"],
        choices=["none", "balanced"],
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        help="Recommended: lbfgs for l2, saga for l1/elasticnet/OHE experiments.",
    )
    parser.add_argument(
        "--sklearn-verbose",
        type=int,
        default=0,
        help="sklearn internal verbosity. For saga, 1 can show convergence progress.",
    )
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100, 500])
    parser.add_argument("--select-metric", type=str, default="ndcg_at_k")
    parser.add_argument("--select-k", type=int, default=100)

    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--save-valid-predictions", action="store_true")
    parser.add_argument("--save-recommendations", action="store_true")
    parser.add_argument("--k-inf", type=int, default=20)
    parser.add_argument("--recommendations-csv-limit", type=int, default=20_000)
    return parser.parse_args()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    required_files = [
        "train_features.parquet",
        "valid_features.parquet",
        "valid_positive_pairs.parquet",
        "feature_columns.json",
        "dataset_summary.json",
    ]
    missing = [name for name in required_files if not (dataset_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "В dataset-dir не хватает файлов: "
            f"{missing}. Перезапустите scripts/build_ltr_dataset.py."
        )

    train = pd.read_parquet(dataset_dir / "train_features.parquet")
    valid = pd.read_parquet(dataset_dir / "valid_features.parquet")
    valid_positive_pairs = pd.read_parquet(dataset_dir / "valid_positive_pairs.parquet")
    feature_config = load_json(dataset_dir / "feature_columns.json")
    dataset_summary = load_json(dataset_dir / "dataset_summary.json")

    return train, valid, valid_positive_pairs, feature_config, dataset_summary


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
    """Create OneHotEncoder compatible with newer/older sklearn versions."""
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
    result = valid.copy()
    result["model_score"] = scores.astype("float32")

    result = result.sort_values(
        ["vacancy_id_hash", "model_score", "embedding_score", "embedding_rank"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).copy()

    result["model_rank"] = result.groupby("vacancy_id_hash").cumcount() + 1
    result["model_rank"] = result["model_rank"].astype(np.int32)

    return result


def get_n_items_total(dataset_summary: dict[str, Any], train: pd.DataFrame, valid: pd.DataFrame) -> int:
    counts = dataset_summary.get("counts", {})
    if "cv_embeddings_rows" in counts:
        return int(counts["cv_embeddings_rows"])

    max_idx = max(
        int(train["cv_idx"].max()) if "cv_idx" in train.columns else 0,
        int(valid["cv_idx"].max()) if "cv_idx" in valid.columns else 0,
    )
    return max_idx + 1


def load_processed_tables_for_recommendations(dataset_summary: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(dataset_summary["processed_dir"])
    cv_path = processed_dir / "cv_normalized.parquet"
    vacancies_path = processed_dir / "vacancies_normalized.parquet"

    if not cv_path.exists() or not vacancies_path.exists():
        raise FileNotFoundError(
            f"Для --save-recommendations нужны {cv_path} и {vacancies_path}"
        )

    return pd.read_parquet(cv_path), pd.read_parquet(vacancies_path)


def main() -> None:
    args = parse_args()

    if args.k_inf <= 0:
        raise ValueError("--k-inf must be positive")

    ks = sorted(set(int(k) for k in args.ks if int(k) > 0))
    if not ks:
        raise ValueError("--ks must contain positive integers")

    for penalty in args.penalty:
        validate_solver_penalty(solver=args.solver, penalty=penalty)

    dataset_dir = args.dataset_dir
    dataset_name = dataset_dir.name
    penalty_part = "_".join(args.penalty)
    run_name = args.run_name or f"logreg_{args.feature_set}_{penalty_part}_{dataset_name}"

    output_dir = args.output_root / run_name
    model_dir = args.model_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"Feature set: {args.feature_set}", flush=True)
    print(f"Penalties: {args.penalty}", flush=True)
    print(f"Output:  {output_dir}", flush=True)
    print(f"Models:  {model_dir}", flush=True)
    print(f"sklearn version: {sklearn.__version__}", flush=True)

    with StepTimer("Loading fixed LTR dataset"):
        train, valid, valid_positive_pairs, feature_config, dataset_summary = load_dataset(dataset_dir)

    numeric_cols, categorical_cols = get_feature_columns(
        feature_set=args.feature_set,
        feature_config=feature_config,
        train=train,
        valid=valid,
    )

    print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}", flush=True)
    print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}", flush=True)
    print(f"Train rows: {len(train):,}", flush=True)
    print(f"Valid rows: {len(valid):,}", flush=True)
    print("Train label distribution:", train["label"].value_counts(dropna=False).to_dict(), flush=True)
    print("Valid label distribution in candidate pool:", valid["label"].value_counts(dropna=False).to_dict(), flush=True)

    with StepTimer("Preparing X/y"):
        X_train, y_train, X_valid = prepare_xy(
            train=train,
            valid=valid,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            label_col="label",
        )

    n_items_total = get_n_items_total(dataset_summary, train=train, valid=valid)

    with StepTimer("Evaluating embedding baseline on fixed valid candidate pool"):
        embedding_metrics = evaluate_ranked_candidates(
            candidates=valid,
            positive_pairs=valid_positive_pairs,
            rank_col="embedding_rank",
            ks=ks,
            n_items_total=n_items_total,
            query_col="vacancy_id_hash",
            item_col="cv_id_hash",
            ranking_name="embedding_only_fixed_dataset",
        )

    configs = iter_model_configs(
        Cs=args.C,
        penalties=args.penalty,
        class_weights=args.class_weight,
        l1_ratios=args.l1_ratio,
    )
    print(f"Total model configs: {len(configs)}", flush=True)

    all_metrics = [embedding_metrics]
    model_summaries = []
    best = {
        "metric_value": -np.inf,
        "model_name": None,
        "model_path": None,
        "metrics": None,
        "predictions": None,
    }

    for i, cfg in enumerate(configs, start=1):
        C = cfg["C"]
        penalty = cfg["penalty"]
        class_weight = cfg["class_weight"]
        l1_ratio = cfg["l1_ratio"]

        model_name = make_model_name(
            C=C,
            class_weight=class_weight,
            solver=args.solver,
            feature_set=args.feature_set,
            penalty=penalty,
            l1_ratio=l1_ratio,
        )
        model_path = model_dir / f"{model_name}.joblib"

        print(f"\n[MODEL {i}/{len(configs)}] {model_name}", flush=True)

        if model_path.exists() and not args.force_train:
            with StepTimer(f"Loading existing model {model_path.name}"):
                model = joblib.load(model_path)
            trained_now = False
        else:
            with StepTimer(f"Training {model_name}"):
                model = build_logreg_pipeline(
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    C=C,
                    class_weight_name=class_weight,
                    solver=args.solver,
                    penalty=penalty,
                    l1_ratio=l1_ratio,
                    max_iter=args.max_iter,
                    verbose=args.sklearn_verbose,
                )
                model.fit(X_train, y_train)
                joblib.dump(model, model_path)
            trained_now = True

        with StepTimer(f"Scoring validation for {model_name}"):
            scores = model.predict_proba(X_valid)[:, 1]
            valid_scored = add_model_scores_and_ranks(valid, scores=scores)

        with StepTimer(f"Evaluating {model_name}"):
            metrics = evaluate_ranked_candidates(
                candidates=valid_scored,
                positive_pairs=valid_positive_pairs,
                rank_col="model_rank",
                ks=ks,
                n_items_total=n_items_total,
                query_col="vacancy_id_hash",
                item_col="cv_id_hash",
                ranking_name=model_name,
            )
            all_metrics.append(metrics)

        selection_rows = metrics[metrics["K"] == args.select_k]
        if selection_rows.empty or args.select_metric not in selection_rows.columns:
            selection_value = (
                float(metrics[args.select_metric].mean())
                if args.select_metric in metrics.columns
                else -np.inf
            )
        else:
            selection_value = float(selection_rows.iloc[0][args.select_metric])

        model_summary = {
            "model_name": model_name,
            "model_path": str(model_path),
            "feature_set": args.feature_set,
            "C": C,
            "penalty": penalty,
            "l1_ratio": l1_ratio,
            "class_weight": class_weight,
            "solver": args.solver,
            "max_iter": args.max_iter,
            "trained_now": trained_now,
            "select_metric": args.select_metric,
            "select_k": args.select_k,
            "select_value": selection_value,
        }
        model_summaries.append(model_summary)

        if selection_value > best["metric_value"]:
            best = {
                "metric_value": selection_value,
                "model_name": model_name,
                "model_path": str(model_path),
                "metrics": metrics,
                "predictions": valid_scored,
            }

        print(
            f"[MODEL RESULT] {model_name}: {args.select_metric}@{args.select_k} = {selection_value:.6f}",
            flush=True,
        )

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    metrics_all.to_csv(output_dir / "metrics.csv", index=False)
    write_json(output_dir / "metrics.json", {"metrics": metrics_all.to_dict(orient="records")})

    model_results = pd.DataFrame(model_summaries).sort_values("select_value", ascending=False)
    model_results.to_csv(output_dir / "model_results.csv", index=False)

    best_predictions_path = None
    if args.save_valid_predictions and best["predictions"] is not None:
        best_predictions_path = output_dir / f"valid_predictions_{best['model_name']}.parquet"
        best["predictions"].to_parquet(best_predictions_path, index=False)

    recommendation_info = None
    if args.save_recommendations and best["predictions"] is not None:
        with StepTimer("Saving recommendations for best model"):
            cv_norm, vacancies_norm = load_processed_tables_for_recommendations(dataset_summary)
            recommendations = make_recommendation_output(
                candidates=best["predictions"],
                cv_table=cv_norm,
                vacancies_table=vacancies_norm,
                rank_col="model_rank",
                score_col="model_score",
                k_inf=args.k_inf,
                keep_feature_columns=True,
            )
            recommendation_info = save_recommendations(
                recommendations=recommendations,
                output_dir=output_dir,
                filename_stem=f"recommendations_top{args.k_inf}_{best['model_name']}",
                csv_limit=args.recommendations_csv_limit,
            )

    best_summary = {
        "best_model_name": best["model_name"],
        "best_model_path": best["model_path"],
        "selection_metric": args.select_metric,
        "selection_k": args.select_k,
        "selection_value": best["metric_value"],
        "valid_predictions_path": str(best_predictions_path) if best_predictions_path is not None else None,
        "recommendations": recommendation_info,
    }
    write_json(output_dir / "best_model.json", best_summary)

    run_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
        "config": {
            "feature_set": args.feature_set,
            "C": args.C,
            "penalty": args.penalty,
            "l1_ratio": args.l1_ratio,
            "class_weight": args.class_weight,
            "solver": args.solver,
            "max_iter": args.max_iter,
            "ks": ks,
            "select_metric": args.select_metric,
            "select_k": args.select_k,
            "force_train": args.force_train,
            "save_valid_predictions": args.save_valid_predictions,
            "save_recommendations": args.save_recommendations,
            "k_inf": args.k_inf,
        },
        "numeric_feature_columns": numeric_cols,
        "categorical_feature_columns": categorical_cols,
        "counts": {
            "train_rows": int(len(train)),
            "valid_rows": int(len(valid)),
            "valid_positive_pairs": int(len(valid_positive_pairs)),
            "n_items_total": int(n_items_total),
        },
        "model_results": model_summaries,
        "best_model": best_summary,
    }
    write_json(output_dir / "run_summary.json", run_summary)

    print("\n=== Model results ===")
    print(model_results.to_string(index=False))

    print("\n=== Best model ===")
    print(json.dumps(best_summary, ensure_ascii=False, indent=2, default=json_default))

    print(f"\nSaved metrics to: {output_dir / 'metrics.csv'}")
    print(f"Saved models to:  {model_dir}")


if __name__ == "__main__":
    main()
