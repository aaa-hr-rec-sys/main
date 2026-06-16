"""Вспомогательные настройки HTTP inference-сервиса"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServiceSettings:
    """Настройки runtime из переменных окружения"""

    artifact_dir: Path = Path("artifacts/inference_bundle_v1")


def load_settings() -> ServiceSettings:
    """Загрузить настройки сервиса из переменных окружения"""
    return ServiceSettings(
        artifact_dir=Path(os.getenv("ARTIFACT_DIR", "artifacts/inference_bundle_v1")),
    )


__all__ = ["ServiceSettings", "load_settings"]
