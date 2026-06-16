from __future__ import annotations

from pathlib import Path

from recommender.training.no_ml import (
    CV_TEXT_COLUMNS,
    VACANCY_TEXT_COLUMNS,
    build_exact_title_candidates,
    build_word_overlap_candidates,
)
from recommender.training.common import ensure_dir, save_json
from recommender.training.experiment_inputs import load_experiment_data
from recommender.training.evaluation import (
    RankingEvaluationInput,
    evaluate_rankings_and_save,
)


def run_no_ml_feature_baselines(
    processed_dir: Path,
    dataset_dir: Path,
    output_dir: Path,
    top_k: int,
    ks: list[int],
    save_candidates: bool = False,
) -> None:
    """Run simple no-ML baselines."""
    ensure_dir(output_dir)

    print("[1/4] Loading data and SAME train/validation split", flush=True)
    data = load_experiment_data(
        processed_dir=processed_dir,
        dataset_dir=dataset_dir,
    )

    print(f"CV rows: {len(data.cv):,}", flush=True)
    print(f"Train positives from same split: {len(data.train_positive_pairs):,}", flush=True)
    print(f"Valid positives from same split: {len(data.valid_positive_pairs):,}", flush=True)
    print(f"Valid vacancies from same split: {len(data.valid_vacancies):,}", flush=True)

    print("[2/4] Baseline 1: exact normalized title match", flush=True)
    exact_candidates = build_exact_title_candidates(
        cv=data.cv,
        valid_vacancies=data.valid_vacancies,
        top_k=top_k,
    )
    print(f"Exact title candidate rows: {len(exact_candidates):,}", flush=True)

    print("[3/4] Baseline 2: simple normalized word overlap", flush=True)
    word_candidates = build_word_overlap_candidates(
        cv=data.cv,
        valid_vacancies=data.valid_vacancies,
        top_k=top_k,
    )
    print(f"Word overlap candidate rows: {len(word_candidates):,}", flush=True)

    print("[4/4] Evaluating with the same metrics code", flush=True)
    n_items_total = data.cv["cv_id_hash"].nunique()

    metrics, metrics_path = evaluate_rankings_and_save(
        rankings=[
            RankingEvaluationInput(
                candidates=exact_candidates,
                rank_col="exact_title_rank",
                ranking_name="no_ml_exact_title_match_norm",
            ),
            RankingEvaluationInput(
                candidates=word_candidates,
                rank_col="word_overlap_rank",
                ranking_name="no_ml_word_overlap_norm",
            ),
        ],
        positive_pairs=data.valid_positive_pairs,
        ks=ks,
        n_items_total=n_items_total,
        output_dir=output_dir,
    )

    print(f"[SAVE] {metrics_path}", flush=True)

    summary = {
        "important": "No ML, one-stage baselines. Same train/validation split as dataset_dir.",
        "processed_dir": str(processed_dir),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "top_k": top_k,
        "ks": ks,
        "cv_rows": int(len(data.cv)),
        "train_positive_pairs_from_same_split": int(len(data.train_positive_pairs)),
        "valid_positive_pairs_from_same_split": int(len(data.valid_positive_pairs)),
        "valid_vacancies_from_same_split": int(len(data.valid_vacancies)),
        "baselines": {
            "no_ml_exact_title_match_norm": {
                "description": "Exact vacancy.profession_norm == cv.profession_norm.",
                "uses": [
                    "profession_norm"
                ],
                "candidate_rows": int(len(exact_candidates)),
            },
            "no_ml_word_overlap_norm": {
                "description": "Simple count of common unique words over normalized fields.",
                "cv_text_columns": CV_TEXT_COLUMNS,
                "vacancy_text_columns": VACANCY_TEXT_COLUMNS,
                "candidate_rows": int(len(word_candidates)),
            },
        },
    }

    summary_path = output_dir / "run_summary.json"
    save_json(summary, summary_path)
    print(f"[SAVE] {summary_path}", flush=True)

    if save_candidates:
        exact_path = output_dir / "no_ml_exact_title_match_norm_candidates.parquet"
        word_path = output_dir / "no_ml_word_overlap_norm_candidates.parquet"

        exact_candidates.to_parquet(exact_path, index=False)
        word_candidates.to_parquet(word_path, index=False)

        print(f"[SAVE] {exact_path}", flush=True)
        print(f"[SAVE] {word_path}", flush=True)

    print("[DONE]", flush=True)