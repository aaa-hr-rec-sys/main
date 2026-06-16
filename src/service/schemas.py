"""Pydantic-схемы синхронного recommendation API"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class RecommendationRequest(BaseModel):
    """Запрос рекомендаций по существующему vacancy id"""

    vacancy_id_hash: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)

    @field_validator("vacancy_id_hash", mode="before")
    @classmethod
    def strip_vacancy_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class RecommendationItem(BaseModel):
    """Одна CV-рекомендация с полями retrieval и model ranking"""

    cv_id_hash: str
    rank: int
    model_score: float
    embedding_score: float
    embedding_rank: int
    display: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    """Синхронный ответ с top-K рекомендациями"""

    vacancy_id_hash: str
    top_k: int
    recommendations: list[RecommendationItem]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    bundle_dir: str
    cv_count: int
    vacancy_count: int
    embedding_dim: int
    model_feature_count: int


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


__all__ = [
    "ErrorBody",
    "ErrorResponse",
    "HealthResponse",
    "ReadyResponse",
    "RecommendationItem",
    "RecommendationRequest",
    "RecommendationResponse",
]
