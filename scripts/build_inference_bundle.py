"""CLI-обертка для сборки синхронного inference bundle"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MAIN_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MAIN_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from service.precompute import FINAL_MODEL_NAME, build_inference_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build inference bundle for FastAPI service")
    parser.add_argument("--data-dir", type=Path, default=Path("data/aaa-out"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/v1"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/modeling/ltr_v1_top500_neg5_ohe"),
    )
    parser.add_argument("--model-path", type=Path, default=Path("models") / FINAL_MODEL_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/inference_bundle_v1"),
    )
    parser.add_argument("--retrieval-top-k", type=int, default=500)
    parser.add_argument("--max-response-top-k", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_inference_bundle(
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        retrieval_top_k=args.retrieval_top_k,
        max_response_top_k=args.max_response_top_k,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"\nSaved inference bundle to: {args.output_dir}")


if __name__ == "__main__":
    main()
