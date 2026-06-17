"""Run simple no-ML baselines on the validation split.

Evaluates exact normalized title matching and normalized word-overlap retrieval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MAIN_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.training.no_ml_runner import run_no_ml_feature_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/v1"))
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help=(
            "Existing modeling dataset directory. "
            "Used only to take the SAME train/validation split as other experiments."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where metrics, summary and optional candidates will be saved.",
    )

    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 50, 100, 500])
    parser.add_argument("--save-candidates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_no_ml_feature_baselines(
        processed_dir=args.processed_dir,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        ks=args.ks,
        save_candidates=args.save_candidates,
    )


if __name__ == "__main__":
    main()
    