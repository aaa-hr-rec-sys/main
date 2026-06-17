"""Evaluate a rule-based reranking baseline on embedding candidates."""


from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd


MAIN_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MAIN_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.training.common import (
    json_default,
    normalize_ks,
    resolve_candidate_top_k,
    save_json,
    validate_k_inf,
)
from recommender.features import add_pair_features
from recommender.training.experiment_outputs import save_recommendation_outputs
from recommender.metrics import evaluate_ranked_candidates
from recommender.retrieval import build_embedding_candidates_for_vacancies
from recommender.training.ltr_dataset import prepare_validation
from utils.data import load_tables
from utils.embeddings import embedding_norm_report, embedding_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate no-ML rule-based reranking baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/aaa-out"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--processed-version", type=str, default="latest")
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100, 500])
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--chunk-size", type=int, default=256)

    parser.add_argument("--w-profession", type=float, default=0.03)
    parser.add_argument("--w-group-profession", type=float, default=0.05)
    parser.add_argument("--w-business-category", type=float, default=0.03)
    parser.add_argument("--w-experience", type=float, default=0.03)
    parser.add_argument("--w-schedule", type=float, default=0.02)
    parser.add_argument("--w-employment-type", type=float, default=0.02)
    parser.add_argument("--w-education", type=float, default=0.01)
    parser.add_argument("--compat-incompatible-value", type=float, default=-1.0)
    parser.add_argument("--compat-unknown-value", type=float, default=0.0)

    parser.add_argument("--save-candidates", action="store_true")
    parser.add_argument(
        "--save-recommendations",
        action="store_true",
        help="Save top-k-inf recommendations with CV/vacancy details",
    )
    parser.add_argument(
        "--k-inf",
        type=int,
        default=20,
        help="How many recommendations per vacancy to save for inspection/inference",
    )
    parser.add_argument(
        "--recommendations-csv-limit",
        type=int,
        default=20_000,
        help="Rows to save to recommendations CSV. Use -1 for all, 0 for no CSV.",
    )
    return parser.parse_args()


def resolve_processed_dir(processed_root: Path, requested_version: str) -> tuple[str, Path]:
    if requested_version == "latest":
        latest_path = processed_root / "latest.txt"
        if not latest_path.exists():
            raise FileNotFoundError(
                f"Не найден {latest_path}. Сначала запустите scripts/normalize_data.py"
            )
        version = latest_path.read_text(encoding="utf-8").strip()
    else:
        version = requested_version

    processed_dir = processed_root / version
    if not processed_dir.exists():
        raise FileNotFoundError(f"Не найдена processed-версия: {processed_dir}")

    if not (processed_dir / "cv_normalized.parquet").exists():
        raise FileNotFoundError(f"Не найден {processed_dir / 'cv_normalized.parquet'}")
    if not (processed_dir / "vacancies_normalized.parquet").exists():
        raise FileNotFoundError(f"Не найден {processed_dir / 'vacancies_normalized.parquet'}")

    return version, processed_dir


def apply_rule_based_score(candidates: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    result = candidates.copy(deep=False)

    result["rule_bonus"] = (
        weights["profession"] * result["same_profession_norm"]
        + weights["group_profession"] * result["same_group_profession_norm"]
        + weights["business_category"] * result["same_business_category_norm"]
        + weights["experience"] * result["experience_compatible_feature"]
        + weights["schedule"] * result["schedule_compatible_feature"]
        + weights["employment_type"] * result["employment_type_compatible_feature"]
        + weights["education"] * result["education_compatible_feature"]
    ).astype("float32")

    result["final_score"] = result["embedding_score"] + result["rule_bonus"]

    result = result.sort_values(
        ["vacancy_id_hash", "final_score", "embedding_score", "embedding_rank"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )

    result["final_rank"] = result.groupby("vacancy_id_hash").cumcount() + 1
    result["final_rank"] = result["final_rank"].astype(np.int32)
    return result


def main() -> None:
    args = parse_args()

    validate_k_inf(args.k_inf)
    ks = normalize_ks(args.ks)
    top_k = resolve_candidate_top_k(args.top_k, ks, args.k_inf)

    processed_version, processed_dir = resolve_processed_dir(
        args.processed_root,
        args.processed_version,
    )
    run_name = args.run_name or f"rule_based_baseline_{processed_version}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = {
        "profession": args.w_profession,
        "group_profession": args.w_group_profession,
        "business_category": args.w_business_category,
        "experience": args.w_experience,
        "schedule": args.w_schedule,
        "employment_type": args.w_employment_type,
        "education": args.w_education,
    }

    print(f"Using processed data: {processed_dir}")
    print("Loading raw tables...")
    tables = load_tables(
        args.data_dir,
        names=["applies", "cv_embeddings", "vacancies_embeddings"],
    )
    applies = tables["applies"]
    cv_embeddings = tables["cv_embeddings"]
    vacancies_embeddings = tables["vacancies_embeddings"]

    print("Loading normalized CV/vacancies...")
    cv_norm = pd.read_parquet(processed_dir / "cv_normalized.parquet")
    vacancies_norm = pd.read_parquet(processed_dir / "vacancies_normalized.parquet")

    print("Converting embeddings to matrices...")
    cv_matrix = embedding_to_matrix(cv_embeddings)
    vacancy_matrix = embedding_to_matrix(vacancies_embeddings)

    embedding_report = pd.concat(
        [
            embedding_norm_report(cv_matrix, "cv_embeddings"),
            embedding_norm_report(vacancy_matrix, "vacancies_embeddings"),
        ],
        ignore_index=True,
    )
    embedding_report.to_csv(output_dir / "embedding_norm_report.csv", index=False)

    print("Preparing temporal validation split...")
    train_applies, valid_pairs, split_report, validation_vacancy_indices = prepare_validation(
        applies=applies,
        cv_embeddings=cv_embeddings,
        vacancies_embeddings=vacancies_embeddings,
        valid_frac=args.valid_frac,
    )
    split_report.to_csv(output_dir / "split_report.csv", index=False)

    print(
        f"Building embedding candidate pool: top_k={top_k}, "
        f"validation_vacancies={len(validation_vacancy_indices)}"
    )
    candidates = build_embedding_candidates_for_vacancies(
        vacancy_matrix=vacancy_matrix,
        cv_matrix=cv_matrix,
        vacancies_embeddings=vacancies_embeddings,
        cv_embeddings=cv_embeddings,
        vacancy_indices=validation_vacancy_indices,
        top_k=top_k,
        chunk_size=args.chunk_size,
    )

    print("Adding pair features...")
    candidates = add_pair_features(
        candidates=candidates,
        cv_norm=cv_norm,
        vacancies_norm=vacancies_norm,
        incompatible_value=args.compat_incompatible_value,
        unknown_value=args.compat_unknown_value,
        keep_debug_columns=True,
    )

    print("Applying rule-based reranking...")
    candidates = apply_rule_based_score(candidates, weights=weights)

    candidate_path = None
    if args.save_candidates:
        candidate_path = output_dir / "candidates.parquet"
        candidates.to_parquet(candidate_path, index=False)

    print("Evaluating embedding order and rule-based order...")
    metrics_embedding = evaluate_ranked_candidates(
        candidates=candidates,
        positive_pairs=valid_pairs,
        rank_col="embedding_rank",
        ks=ks,
        n_items_total=len(cv_embeddings),
        query_col="vacancy_id_hash",
        item_col="cv_id_hash",
        ranking_name="embedding_only",
    )
    metrics_rule = evaluate_ranked_candidates(
        candidates=candidates,
        positive_pairs=valid_pairs,
        rank_col="final_rank",
        ks=ks,
        n_items_total=len(cv_embeddings),
        query_col="vacancy_id_hash",
        item_col="cv_id_hash",
        ranking_name="rule_based_rerank",
    )
    metrics = pd.concat([metrics_embedding, metrics_rule], ignore_index=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    save_json({"metrics": metrics.to_dict(orient="records")}, output_dir / "metrics.json", default=json_default)

    feature_cols = [
        "same_profession_norm",
        "same_group_profession_norm",
        "same_business_category_norm",
        "same_sfera_norm",
        "experience_compatible_feature",
        "schedule_compatible_feature",
        "employment_type_compatible_feature",
        "education_compatible_feature",
        "salary_missing",
        "rule_bonus",
        "embedding_score",
        "final_score",
    ]
    feature_summary = candidates[feature_cols].describe().T
    feature_summary.to_csv(output_dir / "feature_summary.csv")

    recommendation_info = None
    if args.save_recommendations:
        print(f"Saving top-{args.k_inf} recommendations with CV/vacancy details...")
        recommendation_info = save_recommendation_outputs(
            candidates=candidates,
            cv_table=cv_norm,
            vacancies_table=vacancies_norm,
            output_dir=output_dir,
            filename_stem=f"recommendations_top{args.k_inf}_rule_based",
            rank_col="final_rank",
            score_col="final_score",
            k_inf=args.k_inf,
            csv_limit=args.recommendations_csv_limit,
            keep_feature_columns=True,
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": "rule_based_rerank_no_ml",
        "data_dir": str(args.data_dir),
        "processed_root": str(args.processed_root),
        "processed_version": processed_version,
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "config": {
            "valid_frac": args.valid_frac,
            "top_k": top_k,
            "ks": ks,
            "chunk_size": args.chunk_size,
            "weights": weights,
            "compat_incompatible_value": args.compat_incompatible_value,
            "compat_unknown_value": args.compat_unknown_value,
            "save_candidates": args.save_candidates,
            "save_recommendations": args.save_recommendations,
            "k_inf": args.k_inf,
            "recommendations_csv_limit": args.recommendations_csv_limit,
        },
        "counts": {
            "applies_rows": int(len(applies)),
            "train_applies_rows": int(len(train_applies)),
            "valid_positive_pairs_evaluated": int(len(valid_pairs)),
            "validation_vacancies_evaluated": int(len(validation_vacancy_indices)),
            "candidate_rows": int(len(candidates)),
            "cv_embeddings_rows": int(len(cv_embeddings)),
            "vacancies_embeddings_rows": int(len(vacancies_embeddings)),
        },
        "candidate_path": str(candidate_path) if candidate_path is not None else None,
        "recommendations": recommendation_info,
        "split_report": split_report.to_dict(orient="records"),
        "embedding_norm_report": embedding_report.to_dict(orient="records"),
        "metrics": metrics.to_dict(orient="records"),
    }
    save_json(summary, output_dir / "run_summary.json", default=json_default)

    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")
    if recommendation_info:
        print(f"Recommendations parquet: {recommendation_info['parquet_path']}")
        if recommendation_info["csv_path"]:
            print(f"Recommendations csv: {recommendation_info['csv_path']}")


if __name__ == "__main__":
    main()
