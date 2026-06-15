"""Train CatBoost rankers on a fixed LTR dataset.

Uses CatBoost's native ranking mode with vacancy groups and CV candidates.
Saves metrics, models, summaries, and optional recommendation outputs.
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
    import catboost
    from catboost import CatBoostRanker, Pool
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Для train_catboost_ranker.py нужен catboost. "
        "Установите: python -m pip install catboost"
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
        if exc_type is None:
            print(f"[DONE]  {self.name}: {elapsed:.1f}s", flush=True)
        else:
            print(f"[FAILED] {self.name}: {elapsed:.1f}s ({exc_type.__name__}: {exc})", flush=True)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoostRanker on fixed LTR dataset")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/modeling/ltr_v1_top500_neg5_ohe"),
        help="Directory created by scripts/build_ltr_dataset.py",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments"))
    parser.add_argument("--model-root", type=Path, default=Path("data/models"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument(
        "--loss-function",
        type=str,
        nargs="+",
        default=["YetiRank"],
        help="Ranking loss functions to try. Start with YetiRank; later try PairLogit.",
    )
    parser.add_argument("--iterations", type=int, nargs="+", default=[300])
    parser.add_argument("--depth", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--learning-rate", type=float, nargs="+", default=[0.05, 0.1])
    parser.add_argument("--l2-leaf-reg", type=float, nargs="+", default=[3.0])
    parser.add_argument(
        "--one-hot-max-size",
        type=int,
        nargs="+",
        default=[10],
        help=(
            "CatBoost one_hot_max_size values to try. "
            "This is a cardinality threshold for one-hot encoding, not a limit on the number of categorical features."
        ),
    )
    parser.add_argument(
        "--label-weighting",
        type=str,
        nargs="+",
        default=["none"],
        choices=["none", "balanced", "sqrt_balanced"],
        help=(
            "Manual object weights for labels in ranking Pool. "
            "none: all weights=1; balanced: positive weight ~= n_neg/n_pos; "
            "sqrt_balanced: positive weight ~= sqrt(n_neg/n_pos)."
        ),
    )
    parser.add_argument(
        "--weight-valid-pool",
        action="store_true",
        help="Also apply label weighting to validation Pool used for CatBoost early stopping. Metrics remain unweighted.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument("--task-type", type=str, default="CPU", choices=["CPU", "GPU"])
    parser.add_argument("--catboost-verbose", type=int, default=100)
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="0 disables early stopping",
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def make_prediction_frame(
    valid: pd.DataFrame,
    scores: np.ndarray,
) -> pd.DataFrame:
    """Add model scores/ranks without sorting the full validation table.

    The full table may contain large Arrow-backed categorical columns, so metrics
    and recommendations use only ids, ranks, scores, and compact feature columns.
    """
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


def main() -> None:
    args = parse_args()

    if args.k_inf <= 0:
        raise ValueError("--k-inf must be positive")

    ks = sorted(set(int(k) for k in args.ks if int(k) > 0))
    if not ks:
        raise ValueError("--ks must contain positive integers")

    dataset_dir = args.dataset_dir
    dataset_name = dataset_dir.name
    losses_part = "_".join(args.loss_function)
    run_name = args.run_name or f"catboost_ranker_{losses_part}_{dataset_name}"

    output_dir = args.output_root / run_name
    model_dir = args.model_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_dir}", flush=True)
    print(f"Output:  {output_dir}", flush=True)
    print(f"Models:  {model_dir}", flush=True)
    print(f"CatBoost version: {catboost.__version__}", flush=True)

    with StepTimer("Loading fixed LTR dataset"):
        train, valid, valid_positive_pairs, feature_config, dataset_summary = load_dataset(dataset_dir)

    numeric_cols, categorical_cols, feature_cols = get_feature_columns(
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

    with StepTimer("Sorting data by group"):
        train_sorted = sort_by_group(train)
        valid_sorted = sort_by_group(valid)

    with StepTimer("Building validation CatBoost Pool"):
        # Keep validation unweighted by default; metrics are computed separately.
        valid_pool = make_pool(
            valid_sorted,
            feature_cols=feature_cols,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            label_weighting=("none" if not args.weight_valid_pool else args.label_weighting[0]),
        )

    train_pool_cache: dict[str, Pool] = {}

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

    configs = iter_model_configs(args)
    print(f"Total CatBoost configs: {len(configs)}", flush=True)

    all_metrics = [embedding_metrics]
    model_summaries = []
    best = {
        "metric_value": -np.inf,
        "model_name": None,
        "model_path": None,
        "metrics": None,
        "predictions": None,
        "model": None,
    }

    for i, config in enumerate(configs, start=1):
        model_name = make_model_name(**config)
        model_path = model_dir / f"{model_name}.cbm"

        print(f"\n[MODEL {i}/{len(configs)}] {model_name}", flush=True)
        print(f"Config: {config}", flush=True)

        label_weighting = config["label_weighting"]
        if label_weighting not in train_pool_cache:
            with StepTimer(f"Building train CatBoost Pool with weights={label_weighting}"):
                train_pool_cache[label_weighting] = make_pool(
                    train_sorted,
                    feature_cols=feature_cols,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    label_weighting=label_weighting,
                )
        train_pool = train_pool_cache[label_weighting]

        model = CatBoostRanker()
        if model_path.exists() and not args.force_train:
            with StepTimer(f"Loading existing model {model_path.name}"):
                model.load_model(str(model_path))
            trained_now = False
        else:
            with StepTimer(f"Training {model_name}"):
                model = build_ranker(config=config, args=args)
                model.fit(
                    train_pool,
                    eval_set=valid_pool,
                )
                model.save_model(str(model_path))
            trained_now = True

        with StepTimer(f"Scoring validation for {model_name}"):
            scores = model.predict(valid_pool)
            valid_scored = make_prediction_frame(valid_sorted, scores=scores)

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
            "trained_now": trained_now,
            "select_metric": args.select_metric,
            "select_k": args.select_k,
            "select_value": selection_value,
            **config,
        }
        model_summaries.append(model_summary)

        if selection_value > best["metric_value"]:
            best = {
                "metric_value": selection_value,
                "model_name": model_name,
                "model_path": str(model_path),
                "metrics": metrics,
                "predictions": valid_scored,
                "model": model,
            }

        # Save progress after each model so completed results are not lost.
        metrics_partial = pd.concat(all_metrics, ignore_index=True)
        metrics_partial.to_csv(output_dir / "metrics.csv", index=False)
        pd.DataFrame(model_summaries).sort_values(
            "select_value",
            ascending=False,
        ).to_csv(output_dir / "model_results.csv", index=False)

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

    if best["model"] is not None:
        save_feature_importance(
            model=best["model"],
            pool=valid_pool,
            output_path=output_dir / f"feature_importance_{best['model_name']}.csv",
        )

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
            "loss_function": args.loss_function,
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
            "l2_leaf_reg": args.l2_leaf_reg,
            "one_hot_max_size": args.one_hot_max_size,
            "label_weighting": args.label_weighting,
            "weight_valid_pool": args.weight_valid_pool,
            "random_seed": args.random_seed,
            "thread_count": args.thread_count,
            "task_type": args.task_type,
            "early_stopping_rounds": args.early_stopping_rounds,
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
