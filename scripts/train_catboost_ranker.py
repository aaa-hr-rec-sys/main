"""Train CatBoost rankers on a fixed LTR dataset.

Uses CatBoost's native ranking mode with vacancy groups and CV candidates.
Saves metrics, models, summaries, and optional recommendation outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MAIN_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import catboost
from catboost import CatBoostRanker, Pool

from recommender.training.common import StepTimer, json_default, save_json, normalize_ks, validate_k_inf
from recommender.training.ltr_dataset import load_ltr_dataset, get_n_items_total, load_processed_tables_for_recommendations
from recommender.training.experiment_outputs import save_recommendation_outputs, save_metrics_outputs
from recommender.metrics import evaluate_ranked_candidates
from recommender.training.experiment_outputs import get_selection_value
from recommender.training.catboost_ranker import (
    build_ranker,
    get_feature_columns,
    iter_model_configs,
    make_model_name,
    make_pool,
    make_prediction_frame,
    save_feature_importance,
    sort_by_group,
)

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


def main() -> None:
    args = parse_args()

    validate_k_inf(args.k_inf)
    ks = normalize_ks(args.ks)
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
        train, valid, valid_positive_pairs, feature_config, dataset_summary = load_ltr_dataset(dataset_dir)

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

        selection_value = get_selection_value(
            metrics=metrics, select_metric=args.select_metric, select_k=args.select_k,
        )

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

        # Save progress after each model
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
    save_metrics_outputs(metrics_all, output_dir)
        
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
            recommendation_info = save_recommendation_outputs(
                candidates=best["predictions"],
                cv_table=cv_norm,
                vacancies_table=vacancies_norm,
                output_dir=output_dir,
                filename_stem=f"recommendations_top{args.k_inf}_{best['model_name']}",
                rank_col="model_rank",
                score_col="model_score",
                k_inf=args.k_inf,
                csv_limit=args.recommendations_csv_limit,
                keep_feature_columns=True,
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
    save_json(best_summary, output_dir / "best_model.json", default=json_default)

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
    save_json(run_summary, output_dir / "run_summary.json", default=json_default)

    print("\n=== Model results ===")
    print(model_results.to_string(index=False))

    print("\n=== Best model ===")
    print(json.dumps(best_summary, ensure_ascii=False, indent=2, default=json_default))

    print(f"\nSaved metrics to: {output_dir / 'metrics.csv'}")
    print(f"Saved models to:  {model_dir}")


if __name__ == "__main__":
    main()
