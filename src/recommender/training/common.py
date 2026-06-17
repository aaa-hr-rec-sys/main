from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
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


def save_json(
    data: dict[str, Any],
    path: Path,
    default: Callable[[Any], Any] | None = None,
) -> None:
    """Save JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)

    dumps_kwargs: dict[str, Any] = {
        "ensure_ascii": False,
        "indent": 2,
    }
    if default is not None:
        dumps_kwargs["default"] = default

    path.write_text(
        json.dumps(data, **dumps_kwargs),
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


def validate_k_inf(k_inf: int) -> None:
    """Validate the number of recommendations."""
    if k_inf <= 0:
        raise ValueError("--k-inf must be positive")


def normalize_ks(ks: list[int]) -> list[int]:
    """Return sorted unique positive K values for ranking metrics."""
    result = sorted(set(int(k) for k in ks if int(k) > 0))
    if not result:
        raise ValueError("--ks must contain positive integers")
    return result


def resolve_candidate_top_k(
    top_k: int,
    ks: list[int],
    k_inf: int | None = None,
) -> int:
    """Choose candidate pool size large enough for metrics and saved recommendations."""
    values = [top_k, max(ks)]
    if k_inf is not None:
        values.append(k_inf)
    return max(values)