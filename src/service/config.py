"""Вспомогательные настройки HTTP inference-сервиса"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceSettings:
    """Настройки runtime из переменных окружения"""

    artifact_dir: Path = Path("artifacts/inference_bundle_v1")
    max_result_limit: int = 500
    worker_count: int = 1
    max_jobs: int = 1_000
    recommend_timeout_seconds: float = 0.2


def load_settings() -> ServiceSettings:
    """Загрузить настройки сервиса из переменных окружения"""
    return ServiceSettings(
        artifact_dir=Path(os.getenv("ARTIFACT_DIR", "artifacts/inference_bundle_v1")),
        max_result_limit=int(os.getenv("ORCH_MAX_RESULT_LIMIT", "500")),
        worker_count=int(os.getenv("ORCH_WORKER_COUNT", "1")),
        max_jobs=int(os.getenv("ORCH_MAX_JOBS", "1000")),
        recommend_timeout_seconds=float(os.getenv("RECOMMEND_TIMEOUT_SECONDS", "0.2")),
    )


__all__ = ["ServiceSettings", "load_settings"]
