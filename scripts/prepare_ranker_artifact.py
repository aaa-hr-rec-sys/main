from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


RANKER_FEATURE_COLUMNS_FILE = "ranker_feature_columns.json"
RANKER_MANIFEST_FILE = "ranker_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare ranker inference artifacts for ranking-service."
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the final trained CatBoost .cbm model.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to the modeling dataset directory containing feature_columns.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory where ranker inference artifacts will be written.",
    )
    parser.add_argument(
        "--artifact-version",
        default="v1",
        help="Version label for the ranker artifact.",
    )
    parser.add_argument(
        "--processed-data-version",
        default="v1",
        help="Processed data version expected at runtime.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=500,
        help="Stage 1 retrieval candidate count used for training.",
    )
    parser.add_argument(
        "--model-type",
        default="CatBoostRanker",
        help="Model type stored in the manifest.",
    )
    parser.add_argument(
        "--loss-function",
        default="YetiRank",
        help="Training loss function stored in the manifest.",
    )
    parser.add_argument(
        "--selection-metric",
        default="NDCG@10",
        help="Selection metric stored in the manifest.",
    )

    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{description} is not a directory: {path}")


def copy_model_if_needed(model_path: Path, output_dir: Path) -> Path:
    target_model_path = output_dir / model_path.name

    if model_path.resolve() == target_model_path.resolve():
        return target_model_path

    shutil.copy2(model_path, target_model_path)
    return target_model_path


def write_manifest(
    *,
    manifest_path: Path,
    artifact_version: str,
    model_path: Path,
    feature_columns_path: Path,
    dataset_dir: Path,
    processed_data_version: str,
    retrieval_top_k: int,
    model_type: str,
    loss_function: str,
    selection_metric: str,
) -> None:
    manifest = {
        "artifact_name": "ranker",
        "artifact_version": artifact_version,
        "model_file": model_path.name,
        "feature_columns_file": feature_columns_path.name,
        "training_dataset": dataset_dir.as_posix(),
        "processed_data_version": processed_data_version,
        "runtime_cv_store": f"data/processed/{processed_data_version}/cv_normalized.parquet",
        "model_type": model_type,
        "loss_function": loss_function,
        "retrieval_top_k": retrieval_top_k,
        "selection_metric": selection_metric,
        "git_policy": {
            "commit": [
                f"models/{model_path.name}",
                f"models/{feature_columns_path.name}",
                f"models/{manifest_path.name}",
            ],
            "do_not_commit": [
                "data/",
                f"data/processed/{processed_data_version}/cv_normalized.parquet",
            ],
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    model_path = args.model_path
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    feature_columns_source = dataset_dir / "feature_columns.json"

    require_file(model_path, "Model file")
    require_dir(dataset_dir, "Dataset directory")
    require_file(feature_columns_source, "feature_columns.json")

    output_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = copy_model_if_needed(model_path, output_dir)

    feature_columns_target = output_dir / RANKER_FEATURE_COLUMNS_FILE
    shutil.copy2(feature_columns_source, feature_columns_target)

    manifest_path = output_dir / RANKER_MANIFEST_FILE
    write_manifest(
        manifest_path=manifest_path,
        artifact_version=args.artifact_version,
        model_path=final_model_path,
        feature_columns_path=feature_columns_target,
        dataset_dir=dataset_dir,
        processed_data_version=args.processed_data_version,
        retrieval_top_k=args.retrieval_top_k,
        model_type=args.model_type,
        loss_function=args.loss_function,
        selection_metric=args.selection_metric,
    )

    print("Prepared ranker artifact:")
    print(f"  model: {final_model_path}")
    print(f"  feature columns: {feature_columns_target}")
    print(f"  manifest: {manifest_path}")
    print()
    print("Runtime CV store is required but must not be committed:")
    print(f"  data/processed/{args.processed_data_version}/cv_normalized.parquet")


if __name__ == "__main__":
    main()
