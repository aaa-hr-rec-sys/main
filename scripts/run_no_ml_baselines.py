"""Run simple no-ML baselines on the validation split.

Evaluates exact normalized title matching and normalized word-overlap retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MAIN_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MAIN_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.metrics import evaluate_ranked_candidates


#normalized fields
CV_TEXT_COLUMNS = [
    "profession_norm",
    "group_profession_norm",
    "business_category_norm",
    "sfera_norm",
    "experience_common",
    "schedule_norm",
    "employment_type_norm",
    "education_norm",
]

VACANCY_TEXT_COLUMNS = [
    "profession_norm",
    "group_profession_norm",
    "business_category_norm",
    "sfera_norm",
    "experience_common",
    "schedule_norm",
    "employment_type_norm",
    "education_level_norm",
]

BAD_VALUES = {"", "nan", "none", "null", "unknown", "other", "<na>"}
TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/v1"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/modeling/ltr_v1_top500_neg5_ohe"),
        help=(
            "Existing modeling dataset directory. "
            "Used only to take the SAME train/validation split as other experiments."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/experiments/no_ml_feature_baselines"),
    )
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50, 100, 500])
    parser.add_argument("--save-candidates", action="store_true")
    return parser.parse_args()


def norm_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def is_good_value(value: object) -> bool:
    return norm_value(value) not in BAD_VALUES


def tokens_from_fields(row: pd.Series, columns: list[str]) -> set[str]:
    """Simple token set from normalized fields.

    This is intentionally simple:
    - uses only normalized fields;
    - lowercases;
    - splits by words;
    - removes empty/unknown/other-like tokens.
    """
    tokens: set[str] = set()

    for col in columns:
        if col not in row.index:
            continue

        value = norm_value(row[col])
        if value in BAD_VALUES:
            continue

        for token in TOKEN_RE.findall(value):
            token = token.lower().replace("ё", "е")
            if len(token) < 2:
                continue
            if token in BAD_VALUES:
                continue
            tokens.add(token)

    return tokens


def load_inputs(processed_dir: Path, dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cv_path = processed_dir / "cv_normalized.parquet"
    vacancies_path = processed_dir / "vacancies_normalized.parquet"
    train_pos_path = dataset_dir / "train_positive_pairs.parquet"
    valid_pos_path = dataset_dir / "valid_positive_pairs.parquet"

    for path in [cv_path, vacancies_path, train_pos_path, valid_pos_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    cv = pd.read_parquet(cv_path)
    vacancies = pd.read_parquet(vacancies_path)
    train_positive_pairs = pd.read_parquet(train_pos_path)
    valid_positive_pairs = pd.read_parquet(valid_pos_path)

    cv["cv_id_hash"] = cv["cv_id_hash"].astype(str)
    vacancies["vacancy_id_hash"] = vacancies["vacancy_id_hash"].astype(str)

    for pairs in [train_positive_pairs, valid_positive_pairs]:
        pairs["vacancy_id_hash"] = pairs["vacancy_id_hash"].astype(str)
        pairs["cv_id_hash"] = pairs["cv_id_hash"].astype(str)

    return cv, vacancies, train_positive_pairs, valid_positive_pairs


def get_validation_vacancies(
    vacancies: pd.DataFrame,
    valid_positive_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Use the same validation vacancy set as existing experiments."""
    valid_vacancy_ids = valid_positive_pairs["vacancy_id_hash"].drop_duplicates().tolist()
    order = {vacancy_id: i for i, vacancy_id in enumerate(valid_vacancy_ids)}

    result = vacancies[vacancies["vacancy_id_hash"].isin(order)].copy()
    result["_order"] = result["vacancy_id_hash"].map(order)
    result = result.sort_values("_order", kind="mergesort").drop(columns="_order")
    return result.reset_index(drop=True)


def build_exact_title_candidates(
    cv: pd.DataFrame,
    valid_vacancies: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Build candidates by exact normalized profession match."""
    required_cv = {"cv_id_hash", "profession_norm"}
    required_vac = {"vacancy_id_hash", "profession_norm"}
    if missing := required_cv - set(cv.columns):
        raise KeyError(f"CV table misses columns: {sorted(missing)}")
    if missing := required_vac - set(valid_vacancies.columns):
        raise KeyError(f"Vacancy table misses columns: {sorted(missing)}")

    cv_tmp = cv[["cv_id_hash", "profession_norm"]].copy()
    cv_tmp["profession_key"] = cv_tmp["profession_norm"].map(norm_value)
    cv_tmp = cv_tmp[cv_tmp["profession_key"].map(lambda x: x not in BAD_VALUES)]

    cv_by_title = {
        title: group["cv_id_hash"].sort_values().tolist()
        for title, group in cv_tmp.groupby("profession_key", sort=False)
    }

    chunks: list[pd.DataFrame] = []

    for _, row in valid_vacancies[["vacancy_id_hash", "profession_norm"]].iterrows():
        vacancy_id = str(row["vacancy_id_hash"])
        title = norm_value(row["profession_norm"])

        if title in BAD_VALUES:
            continue

        cv_ids = cv_by_title.get(title, [])
        if not cv_ids:
            continue

        cv_ids = cv_ids[:top_k]
        chunks.append(
            pd.DataFrame(
                {
                    "vacancy_id_hash": vacancy_id,
                    "cv_id_hash": cv_ids,
                    "exact_title_score": 1.0,
                    "exact_title_rank": np.arange(1, len(cv_ids) + 1, dtype="int32"),
                }
            )
        )

    if not chunks:
        return pd.DataFrame(
            columns=["vacancy_id_hash", "cv_id_hash", "exact_title_score", "exact_title_rank"]
        )

    return pd.concat(chunks, ignore_index=True)


def build_word_overlap_candidates(
    cv: pd.DataFrame,
    valid_vacancies: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    """Build candidates by normalized word overlap."""
    cv_ids = cv["cv_id_hash"].astype(str).to_numpy()

    # CV index: token -> CV row indexes with this token.
    token_to_cv_indexes: dict[str, list[int]] = defaultdict(list)

    for cv_idx, (_, row) in enumerate(cv.iterrows()):
        cv_tokens = tokens_from_fields(row, CV_TEXT_COLUMNS)
        for token in cv_tokens:
            token_to_cv_indexes[token].append(cv_idx)

    chunks: list[pd.DataFrame] = []

    for _, vacancy_row in valid_vacancies.iterrows():
        vacancy_id = str(vacancy_row["vacancy_id_hash"])
        vacancy_tokens = tokens_from_fields(vacancy_row, VACANCY_TEXT_COLUMNS)

        if not vacancy_tokens:
            continue

        # Score = number of shared unique tokens.
        scores: dict[int, int] = defaultdict(int)
        for token in vacancy_tokens:
            for cv_idx in token_to_cv_indexes.get(token, []):
                scores[cv_idx] += 1

        if not scores:
            continue

        ranked = sorted(
            scores.items(),
            key=lambda x: (-x[1], str(cv_ids[x[0]])),
        )[:top_k]

        chunks.append(
            pd.DataFrame(
                {
                    "vacancy_id_hash": vacancy_id,
                    "cv_id_hash": [str(cv_ids[cv_idx]) for cv_idx, _ in ranked],
                    "word_overlap_score": [int(score) for _, score in ranked],
                    "word_overlap_rank": np.arange(1, len(ranked) + 1, dtype="int32"),
                }
            )
        )

    if not chunks:
        return pd.DataFrame(
            columns=["vacancy_id_hash", "cv_id_hash", "word_overlap_score", "word_overlap_rank"]
        )

    return pd.concat(chunks, ignore_index=True)


def evaluate_and_save(
    candidates: pd.DataFrame,
    valid_positive_pairs: pd.DataFrame,
    rank_col: str,
    ranking_name: str,
    ks: list[int],
    n_items_total: int,
) -> pd.DataFrame:
    return evaluate_ranked_candidates(
        candidates=candidates,
        positive_pairs=valid_positive_pairs,
        rank_col=rank_col,
        ks=ks,
        n_items_total=n_items_total,
        ranking_name=ranking_name,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading data and SAME train/validation split", flush=True)
    cv, vacancies, train_positive_pairs, valid_positive_pairs = load_inputs(
        processed_dir=args.processed_dir,
        dataset_dir=args.dataset_dir,
    )
    valid_vacancies = get_validation_vacancies(vacancies, valid_positive_pairs)

    print(f"CV rows: {len(cv):,}", flush=True)
    print(f"Train positives from same split: {len(train_positive_pairs):,}", flush=True)
    print(f"Valid positives from same split: {len(valid_positive_pairs):,}", flush=True)
    print(f"Valid vacancies from same split: {len(valid_vacancies):,}", flush=True)

    print("[2/4] Baseline 1: exact normalized title match", flush=True)
    exact_candidates = build_exact_title_candidates(
        cv=cv,
        valid_vacancies=valid_vacancies,
        top_k=args.top_k,
    )
    print(f"Exact title candidate rows: {len(exact_candidates):,}", flush=True)

    print("[3/4] Baseline 2: simple normalized word overlap", flush=True)
    word_candidates = build_word_overlap_candidates(
        cv=cv,
        valid_vacancies=valid_vacancies,
        top_k=args.top_k,
    )
    print(f"Word overlap candidate rows: {len(word_candidates):,}", flush=True)

    print("[4/4] Evaluating with the same metrics code", flush=True)
    n_items_total = cv["cv_id_hash"].nunique()

    metrics = pd.concat(
        [
            evaluate_and_save(
                candidates=exact_candidates,
                valid_positive_pairs=valid_positive_pairs,
                rank_col="exact_title_rank",
                ranking_name="no_ml_exact_title_match_norm",
                ks=args.ks,
                n_items_total=n_items_total,
            ),
            evaluate_and_save(
                candidates=word_candidates,
                valid_positive_pairs=valid_positive_pairs,
                rank_col="word_overlap_rank",
                ranking_name="no_ml_word_overlap_norm",
                ks=args.ks,
                n_items_total=n_items_total,
            ),
        ],
        ignore_index=True,
    )

    metrics_path = args.output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"[SAVE] {metrics_path}", flush=True)

    summary = {
        "important": "No ML, one-stage baselines. Same train/validation split as dataset_dir.",
        "processed_dir": str(args.processed_dir),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(args.output_dir),
        "top_k": args.top_k,
        "ks": args.ks,
        "cv_rows": int(len(cv)),
        "train_positive_pairs_from_same_split": int(len(train_positive_pairs)),
        "valid_positive_pairs_from_same_split": int(len(valid_positive_pairs)),
        "valid_vacancies_from_same_split": int(len(valid_vacancies)),
        "baselines": {
            "no_ml_exact_title_match_norm": {
                "description": "Exact vacancy.profession_norm == cv.profession_norm.",
                "uses": ["profession_norm"],
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
    summary_path = args.output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVE] {summary_path}", flush=True)

    if args.save_candidates:
        exact_path = args.output_dir / "no_ml_exact_title_match_norm_candidates.parquet"
        word_path = args.output_dir / "no_ml_word_overlap_norm_candidates.parquet"
        exact_candidates.to_parquet(exact_path, index=False)
        word_candidates.to_parquet(word_path, index=False)
        print(f"[SAVE] {exact_path}", flush=True)
        print(f"[SAVE] {word_path}", flush=True)

    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
