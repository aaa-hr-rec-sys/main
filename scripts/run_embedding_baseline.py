"""Evaluate an embedding-only retrieval baseline and save metrics."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MAIN_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MAIN_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.inference import make_recommendation_output, save_recommendations
from recommender.metrics import evaluate_ranked_candidates
from recommender.retrieval import build_embedding_candidates_for_vacancies
from utils.data import load_tables
from utils.embeddings import embedding_norm_report, embedding_to_matrix
from utils.splits import check_split_leakage, temporal_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate no-ML embedding retrieval baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data/aaa-out"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--processed-version", type=str, default="latest")
    parser.add_argument("--output-root", type=Path, default=Path("data/experiments"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100, 500])
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--chunk-size", type=int, default=256)
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


def resolve_processed_dir(processed_root: Path, requested: str) -> tuple[str | None, Path | None]:
    if requested == "latest":
        latest_path = processed_root / "latest.txt"
        if not latest_path.exists():
            return None, None
        version = latest_path.read_text(encoding="utf-8").strip()
    else:
        version = requested

    processed_dir = processed_root / version
    return version, processed_dir


def load_processed_tables(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if processed_dir is None or not processed_dir.exists():
        raise FileNotFoundError(
            "Processed data is required for --save-recommendations. "
            "Run scripts/normalize_data.py first or pass --processed-version v1."
        )

    cv_path = processed_dir / "cv_normalized.parquet"
    vacancies_path = processed_dir / "vacancies_normalized.parquet"

    if not cv_path.exists() or not vacancies_path.exists():
        raise FileNotFoundError(
            f"Expected {cv_path} and {vacancies_path}"
        )

    return pd.read_parquet(cv_path), pd.read_parquet(vacancies_path)


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


def prepare_validation(
    applies: pd.DataFrame,
    cv_embeddings: pd.DataFrame,
    vacancies_embeddings: pd.DataFrame,
    valid_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    train_applies, valid_applies = temporal_split(
        applies,
        date_col="applied_at_jittered",
        valid_frac=valid_frac,
    )
    split_report = check_split_leakage(train_applies, valid_applies)

    cv_id_to_idx = pd.Series(
        np.arange(len(cv_embeddings), dtype=np.int32),
        index=cv_embeddings["cv_id_hash"],
    )
    vacancy_id_to_idx = pd.Series(
        np.arange(len(vacancies_embeddings), dtype=np.int32),
        index=vacancies_embeddings["vacancy_id_hash"],
    )

    valid_pairs = valid_applies[["cv_id_hash", "vacancy_id_hash"]].drop_duplicates().copy()
    valid_pairs["cv_idx"] = valid_pairs["cv_id_hash"].map(cv_id_to_idx)
    valid_pairs["vacancy_idx"] = valid_pairs["vacancy_id_hash"].map(vacancy_id_to_idx)

    missing_mask = valid_pairs["cv_idx"].isna() | valid_pairs["vacancy_idx"].isna()
    if missing_mask.any():
        print(f"Warning: dropping {int(missing_mask.sum())} validation pairs without embeddings")

    valid_pairs = valid_pairs.loc[~missing_mask].copy()
    valid_pairs["cv_idx"] = valid_pairs["cv_idx"].astype(np.int32)
    valid_pairs["vacancy_idx"] = valid_pairs["vacancy_idx"].astype(np.int32)

    validation_vacancy_indices = np.array(
        sorted(valid_pairs["vacancy_idx"].unique()),
        dtype=np.int32,
    )

    return train_applies, valid_pairs, split_report, validation_vacancy_indices


def main() -> None:
    args = parse_args()

    if args.k_inf <= 0:
        raise ValueError("--k-inf must be positive")

    ks = sorted(set(int(k) for k in args.ks if int(k) > 0))
    if not ks:
        raise ValueError("--ks must contain positive integers")
    top_k = max(args.top_k, max(ks), args.k_inf)

    processed_version, processed_dir = resolve_processed_dir(
        args.processed_root,
        args.processed_version,
    )

    run_name = args.run_name or f"embedding_baseline_{processed_version or 'raw'}"
    output_dir = args.output_root / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw tables...")
    tables = load_tables(
        args.data_dir,
        names=["applies", "cv_embeddings", "vacancies_embeddings"],
    )
    applies = tables["applies"]
    cv_embeddings = tables["cv_embeddings"]
    vacancies_embeddings = tables["vacancies_embeddings"]

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
        f"Building exact embedding candidates: top_k={top_k}, "
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

    print("Evaluating...")
    metrics = evaluate_ranked_candidates(
        candidates=candidates,
        positive_pairs=valid_pairs,
        rank_col="embedding_rank",
        ks=ks,
        n_items_total=len(cv_embeddings),
        query_col="vacancy_id_hash",
        item_col="cv_id_hash",
        ranking_name="embedding_only",
    )
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    write_json(output_dir / "metrics.json", {"metrics": metrics.to_dict(orient="records")})

    candidate_path = None
    if args.save_candidates:
        candidate_path = output_dir / "candidates.parquet"
        candidates.to_parquet(candidate_path, index=False)

    recommendation_info = None
    if args.save_recommendations:
        print(f"Saving top-{args.k_inf} recommendations with CV/vacancy details...")
        cv_norm, vacancies_norm = load_processed_tables(processed_dir)

        recommendations = make_recommendation_output(
            candidates=candidates,
            cv_table=cv_norm,
            vacancies_table=vacancies_norm,
            rank_col="embedding_rank",
            score_col="embedding_score",
            k_inf=args.k_inf,
            keep_feature_columns=False,
        )
        recommendation_info = save_recommendations(
            recommendations=recommendations,
            output_dir=output_dir,
            filename_stem=f"recommendations_top{args.k_inf}_embedding",
            csv_limit=args.recommendations_csv_limit,
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline": "embedding_only_exact_topk",
        "data_dir": str(args.data_dir),
        "processed_root": str(args.processed_root),
        "processed_version": processed_version,
        "processed_dir": str(processed_dir) if processed_dir is not None else None,
        "output_dir": str(output_dir),
        "config": {
            "valid_frac": args.valid_frac,
            "top_k": top_k,
            "ks": ks,
            "chunk_size": args.chunk_size,
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
    write_json(output_dir / "run_summary.json", summary)

    print("\nMetrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved results to: {output_dir}")
    if recommendation_info:
        print(f"Recommendations parquet: {recommendation_info['parquet_path']}")
        if recommendation_info["csv_path"]:
            print(f"Recommendations csv: {recommendation_info['csv_path']}")


if __name__ == "__main__":
    main()
