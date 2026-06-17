"""Build train and validation feature tables for LTR models.

The script creates a fixed modeling dataset from raw applies, embeddings,
and normalized CV/vacancy tables, then saves features and evaluation metadata.
"""

from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MAIN_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MAIN_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.training.common import StepTimer, json_default, save_json, normalize_ks, resolve_candidate_top_k
from recommender.features import add_pair_features, get_numeric_feature_columns
from recommender.metrics import evaluate_ranked_candidates
from recommender.negative_sampling import (
    add_binary_label,
    label_summary,
    sample_hard_negatives_per_query,
    sample_mixed_negatives_per_query,
    sample_source_summary,
)
from recommender.retrieval import build_embedding_candidates_for_vacancies
from utils.data import load_tables
from utils.embeddings import embedding_norm_report, embedding_to_matrix
from utils.splits import check_split_leakage, temporal_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed LTR dataset for ML rankers")
    parser.add_argument("--data-dir", type=Path, default=Path("data/aaa-out"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--processed-version", type=str, default="latest")
    parser.add_argument("--output-root", type=Path, default=Path("data/modeling"))
    parser.add_argument("--dataset-name", type=str, default=None)

    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100, 500])
    parser.add_argument("--valid-frac", type=float, default=0.2)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--negative-sampling-strategy",
        type=str,
        default="hard",
        choices=["hard", "mixed"],
        help="hard = old behavior; mixed = filtered hard negatives + easy random negatives",
    )
    parser.add_argument(
        "--negative-ratio",
        type=int,
        default=5,
        help="Old hard-only ratio. Used when --negative-sampling-strategy hard.",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=int,
        default=5,
        help="For mixed strategy: hard negatives per positive from top-K candidate pool.",
    )
    parser.add_argument(
        "--easy-negative-ratio",
        type=int,
        default=5,
        help="For mixed strategy: easy random negatives per positive outside top-K.",
    )
    parser.add_argument(
        "--hard-min-rank",
        type=int,
        default=50,
        help="For mixed strategy: sample hard negatives only with embedding_rank >= this value.",
    )
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--compat-incompatible-value", type=float, default=-1.0)
    parser.add_argument("--compat-unknown-value", type=float, default=0.0)
    parser.add_argument(
        "--keep-debug-columns",
        action="store_true",
        help="Keep joined normalized text columns in feature parquet files. Larger but easier to debug.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing dataset directory",
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

    cv_path = processed_dir / "cv_normalized.parquet"
    vacancies_path = processed_dir / "vacancies_normalized.parquet"

    if not cv_path.exists() or not vacancies_path.exists():
        raise FileNotFoundError(
            f"В {processed_dir} должны быть cv_normalized.parquet и vacancies_normalized.parquet"
        )

    return version, processed_dir


def prepare_split_and_indices(
    applies: pd.DataFrame,
    cv_embeddings: pd.DataFrame,
    vacancies_embeddings: pd.DataFrame,
    valid_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
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

    return train_applies, valid_applies, split_report, cv_id_to_idx, vacancy_id_to_idx


def apply_indices(
    pairs: pd.DataFrame,
    cv_id_to_idx: pd.Series,
    vacancy_id_to_idx: pd.Series,
) -> pd.DataFrame:
    result = pairs[["cv_id_hash", "vacancy_id_hash"]].drop_duplicates().copy()
    result["cv_idx"] = result["cv_id_hash"].map(cv_id_to_idx)
    result["vacancy_idx"] = result["vacancy_id_hash"].map(vacancy_id_to_idx)

    missing_mask = result["cv_idx"].isna() | result["vacancy_idx"].isna()
    if missing_mask.any():
        print(f"Warning: dropping {int(missing_mask.sum())} pairs without embeddings", flush=True)

    result = result.loc[~missing_mask].copy()
    result["cv_idx"] = result["cv_idx"].astype(np.int32)
    result["vacancy_idx"] = result["vacancy_idx"].astype(np.int32)

    return result


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.negative_sampling_strategy == "hard" and args.negative_ratio <= 0:
        raise ValueError("--negative-ratio must be positive")
    if args.negative_sampling_strategy == "mixed":
        if args.hard_negative_ratio < 0:
            raise ValueError("--hard-negative-ratio must be non-negative")
        if args.easy_negative_ratio < 0:
            raise ValueError("--easy-negative-ratio must be non-negative")
        if args.hard_negative_ratio == 0 and args.easy_negative_ratio == 0:
            raise ValueError("At least one of hard/easy negative ratios must be positive")
        if args.hard_min_rank <= 0:
            raise ValueError("--hard-min-rank must be positive")

    ks = normalize_ks(args.ks)
    top_k = resolve_candidate_top_k(args.top_k, ks)

    processed_version, processed_dir = resolve_processed_dir(
        args.processed_root,
        args.processed_version,
    )

    if args.dataset_name:
        dataset_name = args.dataset_name
    elif args.negative_sampling_strategy == "hard":
        dataset_name = f"ltr_{processed_version}_top{top_k}_neg{args.negative_ratio}"
    else:
        dataset_name = (
            f"ltr_{processed_version}_top{top_k}_mixed_"
            f"hard{args.hard_negative_ratio}_easy{args.easy_negative_ratio}_"
            f"minrank{args.hard_min_rank}"
        )
    output_dir = args.output_root / dataset_name

    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Dataset directory already exists: {output_dir}. "
            f"Use --overwrite or choose --dataset-name."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dataset directory: {output_dir}", flush=True)
    print(f"Processed data: {processed_dir}", flush=True)

    with StepTimer("Loading raw tables"):
        tables = load_tables(
            args.data_dir,
            names=["applies", "cv_embeddings", "vacancies_embeddings"],
        )
        applies = tables["applies"]
        cv_embeddings = tables["cv_embeddings"]
        vacancies_embeddings = tables["vacancies_embeddings"]

    with StepTimer("Loading normalized CV/vacancies"):
        cv_norm = pd.read_parquet(processed_dir / "cv_normalized.parquet")
        vacancies_norm = pd.read_parquet(processed_dir / "vacancies_normalized.parquet")

    with StepTimer("Converting embeddings to matrices"):
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

    with StepTimer("Temporal split"):
        train_applies, valid_applies, split_report, cv_id_to_idx, vacancy_id_to_idx = (
            prepare_split_and_indices(
                applies=applies,
                cv_embeddings=cv_embeddings,
                vacancies_embeddings=vacancies_embeddings,
                valid_frac=args.valid_frac,
            )
        )
        split_report.to_csv(output_dir / "split_report.csv", index=False)

        train_positive_pairs = apply_indices(train_applies, cv_id_to_idx, vacancy_id_to_idx)
        valid_positive_pairs = apply_indices(valid_applies, cv_id_to_idx, vacancy_id_to_idx)

        # Save real vacancy-CV matches for later evaluation
        train_positive_pairs.to_parquet(output_dir / "train_positive_pairs.parquet", index=False)
        valid_positive_pairs.to_parquet(output_dir / "valid_positive_pairs.parquet", index=False)

        train_vacancy_indices = np.array(
            sorted(train_positive_pairs["vacancy_idx"].unique()),
            dtype=np.int32,
        )
        valid_vacancy_indices = np.array(
            sorted(valid_positive_pairs["vacancy_idx"].unique()),
            dtype=np.int32,
        )

        print(
            f"Train positives: {len(train_positive_pairs)}, "
            f"train vacancies: {len(train_vacancy_indices)}",
            flush=True,
        )
        print(
            f"Valid positives: {len(valid_positive_pairs)}, "
            f"valid vacancies: {len(valid_vacancy_indices)}",
            flush=True,
        )

    with StepTimer(f"Building train embedding candidates top-{top_k}"):
        train_candidates = build_embedding_candidates_for_vacancies(
            vacancy_matrix=vacancy_matrix,
            cv_matrix=cv_matrix,
            vacancies_embeddings=vacancies_embeddings,
            cv_embeddings=cv_embeddings,
            vacancy_indices=train_vacancy_indices,
            top_k=top_k,
            chunk_size=args.chunk_size,
        )

    with StepTimer(f"Building validation embedding candidates top-{top_k}"):
        valid_candidates = build_embedding_candidates_for_vacancies(
            vacancy_matrix=vacancy_matrix,
            cv_matrix=cv_matrix,
            vacancies_embeddings=vacancies_embeddings,
            cv_embeddings=cv_embeddings,
            vacancy_indices=valid_vacancy_indices,
            top_k=top_k,
            chunk_size=args.chunk_size,
        )

    with StepTimer("Adding labels"):
        train_labeled_pool = add_binary_label(
            candidates=train_candidates,
            positive_pairs=train_positive_pairs,
            label_col="label",
        )
        valid_labeled = add_binary_label(
            candidates=valid_candidates,
            positive_pairs=valid_positive_pairs,
            label_col="label",
        )

        train_pool_label_summary = label_summary(train_labeled_pool)
        valid_label_summary = label_summary(valid_labeled)

        print(f"Train candidate pool labels: {train_pool_label_summary}", flush=True)
        print(f"Valid candidate labels: {valid_label_summary}", flush=True)

    with StepTimer("Evaluating candidate generation"):
        train_candidate_metrics = evaluate_ranked_candidates(
            candidates=train_labeled_pool,
            positive_pairs=train_positive_pairs,
            rank_col="embedding_rank",
            ks=ks,
            n_items_total=len(cv_embeddings),
            ranking_name="train_embedding_candidate_pool",
        )
        valid_candidate_metrics = evaluate_ranked_candidates(
            candidates=valid_labeled,
            positive_pairs=valid_positive_pairs,
            rank_col="embedding_rank",
            ks=ks,
            n_items_total=len(cv_embeddings),
            ranking_name="valid_embedding_candidate_pool",
        )
        candidate_generation_metrics = pd.concat(
            [train_candidate_metrics, valid_candidate_metrics],
            ignore_index=True,
        )
        candidate_generation_metrics.to_csv(
            output_dir / "candidate_generation_metrics.csv",
            index=False,
        )

    with StepTimer("Sampling negatives for train"):
        if args.negative_sampling_strategy == "hard":
            train_sampled = sample_hard_negatives_per_query(
                labeled_candidates=train_labeled_pool,
                negative_ratio=args.negative_ratio,
                random_state=args.random_state,
            )
        else:
            train_sampled = sample_mixed_negatives_per_query(
                labeled_topk_candidates=train_labeled_pool,
                positive_pairs=train_positive_pairs,
                vacancy_matrix=vacancy_matrix,
                cv_matrix=cv_matrix,
                vacancies_embeddings=vacancies_embeddings,
                cv_embeddings=cv_embeddings,
                hard_negative_ratio=args.hard_negative_ratio,
                easy_negative_ratio=args.easy_negative_ratio,
                hard_min_rank=args.hard_min_rank,
                easy_embedding_rank_value=top_k + 1,
                random_state=args.random_state,
            )

        train_sampled_label_summary = label_summary(train_sampled)
        train_sample_source_summary = sample_source_summary(train_sampled)
        print(f"Train sampled labels: {train_sampled_label_summary}", flush=True)
        print(f"Train sampled sources: {train_sample_source_summary}", flush=True)

        # Free memory before feature join.
        del train_candidates, train_labeled_pool
        gc.collect()

    with StepTimer("Adding pair features to train"):
        train_features = add_pair_features(
            candidates=train_sampled,
            cv_norm=cv_norm,
            vacancies_norm=vacancies_norm,
            incompatible_value=args.compat_incompatible_value,
            unknown_value=args.compat_unknown_value,
            keep_debug_columns=args.keep_debug_columns,
        )
        train_features.to_parquet(output_dir / "train_features.parquet", index=False)

        del train_sampled, train_features
        gc.collect()

    with StepTimer("Adding pair features to validation"):
        valid_features = add_pair_features(
            candidates=valid_labeled,
            cv_norm=cv_norm,
            vacancies_norm=vacancies_norm,
            incompatible_value=args.compat_incompatible_value,
            unknown_value=args.compat_unknown_value,
            keep_debug_columns=args.keep_debug_columns,
        )
        valid_features.to_parquet(output_dir / "valid_features.parquet", index=False)

        del valid_labeled, valid_features
        gc.collect()

    feature_columns = get_numeric_feature_columns()
    save_json(
        {
            "numeric_feature_columns": feature_columns,
            "label_column": "label",
            "query_column": "vacancy_id_hash",
            "item_column": "cv_id_hash",
        },
        output_dir / "feature_columns.json",
        default=json_default,
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "data_dir": str(args.data_dir),
        "processed_root": str(args.processed_root),
        "processed_version": processed_version,
        "processed_dir": str(processed_dir),
        "config": {
            "top_k": top_k,
            "ks": ks,
            "valid_frac": args.valid_frac,
            "chunk_size": args.chunk_size,
            "negative_sampling_strategy": args.negative_sampling_strategy,
            "negative_ratio": args.negative_ratio,
            "hard_negative_ratio": args.hard_negative_ratio,
            "easy_negative_ratio": args.easy_negative_ratio,
            "hard_min_rank": args.hard_min_rank,
            "random_state": args.random_state,
            "compat_incompatible_value": args.compat_incompatible_value,
            "compat_unknown_value": args.compat_unknown_value,
            "keep_debug_columns": args.keep_debug_columns,
        },
        "counts": {
            "applies_rows": int(len(applies)),
            "train_applies_rows": int(len(train_applies)),
            "valid_applies_rows": int(len(valid_applies)),
            "train_positive_pairs_with_embeddings": int(len(train_positive_pairs)),
            "valid_positive_pairs_with_embeddings": int(len(valid_positive_pairs)),
            "train_vacancies": int(len(train_vacancy_indices)),
            "valid_vacancies": int(len(valid_vacancy_indices)),
            "train_candidate_pool_rows_before_sampling": int(train_pool_label_summary["rows"]),
            "valid_candidate_rows": int(valid_label_summary["rows"]),
            "cv_embeddings_rows": int(len(cv_embeddings)),
            "vacancies_embeddings_rows": int(len(vacancies_embeddings)),
        },
        "files": {
            "train_features": str(output_dir / "train_features.parquet"),
            "valid_features": str(output_dir / "valid_features.parquet"),
            "train_positive_pairs": str(output_dir / "train_positive_pairs.parquet"),
            "valid_positive_pairs": str(output_dir / "valid_positive_pairs.parquet"),
            "feature_columns": str(output_dir / "feature_columns.json"),
        },
        "label_summaries": {
            "train_candidate_pool": train_pool_label_summary,
            "train_sampled": train_sampled_label_summary,
            "valid_candidates": valid_label_summary,
        },
        "sample_source_summaries": {
            "train_sampled": train_sample_source_summary,
        },
        "split_report": split_report.to_dict(orient="records"),
        "candidate_generation_metrics": candidate_generation_metrics.to_dict(orient="records"),
    }
    save_json(summary, output_dir / "dataset_summary.json", default=json_default)

    print("\nCandidate generation metrics:")
    print(candidate_generation_metrics.to_string(index=False))
    print(f"\nSaved dataset to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
