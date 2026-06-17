"""Train Logistic Regression rankers on a fixed LTR dataset.

Supports numeric features and optional one-hot encoded categorical features.
Runs one or more regularization configs and saves metrics, models, and summaries.
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

import joblib
import sklearn

from recommender.training.common import StepTimer, json_default, save_json, normalize_ks, validate_k_inf
from recommender.training.ltr_dataset import load_ltr_dataset, get_n_items_total, load_processed_tables_for_recommendations
from recommender.training.experiment_outputs import save_recommendation_outputs, save_metrics_outputs
from recommender.metrics import evaluate_ranked_candidates
from recommender.training.experiment_outputs import get_selection_value
from recommender.training.logreg_ranker import (
    add_model_scores_and_ranks,
    build_logreg_pipeline,
    get_feature_columns,
    iter_model_configs,
    make_model_name,
    prepare_xy,
    validate_solver_penalty,
    predict_positive_proba_in_chunks
)


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


def main() -> None:
    args = parse_args()

    validate_k_inf(args.k_inf)
    ks = normalize_ks(args.ks)
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
        train, valid, valid_positive_pairs, feature_config, dataset_summary = load_ltr_dataset(dataset_dir)

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
            scores = predict_positive_proba_in_chunks(model, X_valid)
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

        selection_value = get_selection_value(
            metrics=metrics, select_metric=args.select_metric, select_k=args.select_k,
        )

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
    save_metrics_outputs(metrics_all, output_dir)

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
    save_json(run_summary, output_dir / "run_summary.json", default=json_default)
    
    print("\n=== Model results ===")
    print(model_results.to_string(index=False))

    print("\n=== Best model ===")
    print(json.dumps(best_summary, ensure_ascii=False, indent=2, default=json_default))

    print(f"\nSaved metrics to: {output_dir / 'metrics.csv'}")
    print(f"Saved models to:  {model_dir}")


if __name__ == "__main__":
    main()
