from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


class StepTimer:
    """Small context manager for consistent step logging."""

    def __init__(self, name: str):
        self.name = name
        self.start: float | None = None

    def __enter__(self):
        self.start = time.perf_counter()
        print(f"[START] {self.name}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        if exc_type is None:
            print(f"[DONE]  {self.name}: {elapsed:.1f}s", flush=True)
        else:
            print(f"[FAILED] {self.name}: {elapsed:.1f}s ({exc_type.__name__}: {exc})", flush=True)
        return False


def ensure_dir(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def require_files(paths: list[Path]) -> None:
    """Check that all required files exist."""
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")


def cast_columns_to_str(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Cast selected columns to string in-place and return the same DataFrame."""
    for column in columns:
        df[column] = df[column].astype(str)
    return df


def concat_or_empty(chunks: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    """Concatenate chunks or return an empty DataFrame with the requested columns."""
    if not chunks:
        return pd.DataFrame(columns=columns)

    return pd.concat(chunks, ignore_index=True)


def select_ordered_subset_by_ids(
    df: pd.DataFrame,
    id_col: str,
    ordered_ids: list[str],
) -> pd.DataFrame:
    """Filter rows by ids and preserve the order of ordered_ids.
    """
    order = {item_id: i for i, item_id in enumerate(ordered_ids)}

    result = df[df[id_col].isin(order)].copy()
    result["_order"] = result[id_col].map(order)
    result = result.sort_values("_order", kind="mergesort").drop(columns="_order")

    return result.reset_index(drop=True)