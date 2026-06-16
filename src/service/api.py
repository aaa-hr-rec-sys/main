"""FastAPI-приложение синхронного inference-сервиса"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from service.artifacts import ArtifactLoadError, InferenceArtifacts, load_inference_artifacts
from service.config import ServiceSettings, load_settings
from service.pipeline import RecommendationPipeline, UnknownVacancyError
from service.schemas import HealthResponse, ReadyResponse, RecommendationRequest


def _error_response(
    http_status: int,
    code: str,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content=jsonable_encoder(
            {
                "error": {
                    "code": code,
                    "message": message,
                    "details": details,
                }
            }
        ),
    )


def _build_pipeline(artifacts: InferenceArtifacts | None) -> RecommendationPipeline | None:
    return RecommendationPipeline(artifacts) if artifacts is not None else None


def create_app(
    settings: ServiceSettings | None = None,
    artifacts: InferenceArtifacts | None = None,
) -> FastAPI:
    """Создать FastAPI app и загрузить артефакты при старте"""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.artifacts is None:
            try:
                app.state.artifacts = load_inference_artifacts(app.state.settings.artifact_dir)
                app.state.pipeline = _build_pipeline(app.state.artifacts)
                app.state.load_error = None
            except ArtifactLoadError as exc:
                app.state.load_error = str(exc)
        yield

    app = FastAPI(
        title="Resume Recommendation Inference Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.artifacts = artifacts
    app.state.pipeline = _build_pipeline(artifacts)
    app.state.load_error = None

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return _error_response(
            422,
            "validation_error",
            "Invalid request body",
            exc.errors(),
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> ReadyResponse:
        bundle = app.state.artifacts
        if bundle is None:
            return _error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "service_not_ready",
                "Inference artifacts are not loaded",
                app.state.load_error,
            )
        return ReadyResponse(
            status="ready",
            bundle_dir=str(bundle.bundle_dir),
            cv_count=int(len(bundle.cv_ids)),
            vacancy_count=int(len(bundle.vacancy_ids)),
            embedding_dim=bundle.embedding_dim,
            model_feature_count=len(bundle.feature_columns),
        )

    @app.post("/recommendations")
    def recommendations(request: RecommendationRequest):
        pipeline = app.state.pipeline
        if pipeline is None:
            return _error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "service_not_ready",
                "Inference artifacts are not loaded",
                app.state.load_error,
            )
        try:
            return pipeline.recommend(
                vacancy_id_hash=request.vacancy_id_hash,
                top_k=request.top_k,
            )
        except UnknownVacancyError:
            return _error_response(
                status.HTTP_404_NOT_FOUND,
                "unknown_vacancy_id",
                f"Unknown vacancy_id_hash: {request.vacancy_id_hash}",
            )
        except ValueError as exc:
            return _error_response(
                status.HTTP_400_BAD_REQUEST,
                "bad_request",
                str(exc),
            )
        except Exception as exc:  # pragma: no cover - защитная граница API
            return _error_response(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "inference_error",
                "Recommendation pipeline failed",
                str(exc),
            )

    return app


app = create_app()


__all__ = ["app", "create_app"]
