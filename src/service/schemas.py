"""Pydantic-схемы HTTP inference API"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class RecommendationRequest(BaseModel):
    """Запрос рекомендаций по существующему vacancy id"""

    vacancy_id_hash: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=500)

    @field_validator("vacancy_id_hash", mode="before")
    @classmethod
    def strip_vacancy_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class VacancyFields(BaseModel):
    """Поля вакансии, приходящие от frontend или stage 1 клиента"""

    vacancy_text: str = Field(default="", min_length=0)
    profession: str = Field(..., min_length=1)
    group_profession: str = Field(..., min_length=1)
    business_category: str = Field(..., min_length=1)
    sfera: str = Field(..., min_length=1)
    experience: str = Field(..., min_length=1)
    schedule: str = Field(..., min_length=1)
    employment_type: str = Field(..., min_length=1)
    education_level: str = Field(..., min_length=1)

    @field_validator("*", mode="before")
    @classmethod
    def strip_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class FrontendRecommendRequest(BaseModel):
    """Текущий JSON-контракт формы из frontend submodule"""

    vacancy_text: str = Field(default="", min_length=0)
    vac_profession: str = Field(..., min_length=1)
    vac_group_profession: str = Field(..., min_length=1)
    vac_business_category: str = Field(..., min_length=1)
    vac_sfera: str = Field(..., min_length=1)
    vac_experience: str = Field(..., min_length=1)
    vac_schedule: str = Field(..., min_length=1)
    vac_employment_type: str = Field(..., min_length=1)
    vac_education_level: str = Field(..., min_length=1)
    candidate_limit: int = Field(default=500, ge=1, le=500)
    result_limit: int = Field(default=500, ge=1, le=500)

    @field_validator("*", mode="before")
    @classmethod
    def strip_frontend_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    def to_vacancy_fields(self) -> VacancyFields:
        return VacancyFields(
            vacancy_text=self.vacancy_text,
            profession=self.vac_profession,
            group_profession=self.vac_group_profession,
            business_category=self.vac_business_category,
            sfera=self.vac_sfera,
            experience=self.vac_experience,
            schedule=self.vac_schedule,
            employment_type=self.vac_employment_type,
            education_level=self.vac_education_level,
        )


class InferenceJobCreateRequest(BaseModel):
    """Запрос на асинхронный inference job"""

    vacancy_id_hash: str | None = Field(default=None, min_length=1)
    vacancy: VacancyFields | None = None
    candidate_limit: int = Field(default=500, ge=1, le=500)
    result_limit: int = Field(default=500, ge=1, le=500)

    @field_validator("vacancy_id_hash", mode="before")
    @classmethod
    def strip_optional_vacancy_id(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @model_validator(mode="after")
    def require_vacancy_input(self) -> "InferenceJobCreateRequest":
        if self.vacancy_id_hash is None and self.vacancy is None:
            raise ValueError("Either vacancy_id_hash or vacancy must be provided")
        return self


JobStatus = Literal[
    "queued",
    "running",
    "stage1_running",
    "stage2_running",
    "postprocessing_running",
    "succeeded",
    "failed",
]


class InferenceJobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus


class InferenceJobError(BaseModel):
    code: str
    message: str
    details: Any | None = None


class InferenceJobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    candidate_count: int | None = None
    result_count: int | None = None
    error: InferenceJobError | None = None


class FrontendRecommendationItem(BaseModel):
    """Формат одной рекомендации, удобный текущей странице frontend/recommendations"""

    id: str
    cv_id_hash: str
    rank: int
    score: float
    model_score: float
    embedding_score: float
    embedding_rank: int
    profession: str | None = None
    group_profession: str | None = None
    business_category: str | None = None
    sfera: str | None = None
    experience_bucket: str | None = None
    education: str | None = None
    federal_district: str | None = None
    salary_bucketed: Any | None = None
    employment_type: str | None = None
    schedule: str | None = None


class InferenceJobResultResponse(BaseModel):
    job_id: str
    status: Literal["succeeded"]
    result_limit: int
    recommendations: list[FrontendRecommendationItem]


class FrontendRecommendResponse(BaseModel):
    recommendations: list[FrontendRecommendationItem]


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
    "FrontendRecommendRequest",
    "FrontendRecommendResponse",
    "FrontendRecommendationItem",
    "HealthResponse",
    "InferenceJobCreateRequest",
    "InferenceJobCreateResponse",
    "InferenceJobError",
    "InferenceJobResultResponse",
    "InferenceJobStatusResponse",
    "JobStatus",
    "ReadyResponse",
    "RecommendationItem",
    "RecommendationRequest",
    "RecommendationResponse",
    "VacancyFields",
]
