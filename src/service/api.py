"""FastAPI-приложение inference-сервиса и MVP orchestrator"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from service.artifacts import ArtifactLoadError, InferenceArtifacts, load_inference_artifacts
from service.config import ServiceSettings, load_settings
from service.jobs import JobNotFoundError
from service.orchestrator import InferenceOrchestrator
from service.pipeline import RecommendationPipeline, UnknownVacancyError
from service.schemas import (
    FrontendRecommendRequest,
    FrontendRecommendResponse,
    HealthResponse,
    InferenceJobCreateRequest,
    InferenceJobCreateResponse,
    InferenceJobResultResponse,
    InferenceJobStatusResponse,
    ReadyResponse,
    RecommendationRequest,
)


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


def _job_error_status(code: str) -> int:
    if code in {"runtime_embedder_not_configured", "service_not_ready"}:
        return status.HTTP_503_SERVICE_UNAVAILABLE
    if code == "unknown_vacancy_id":
        return status.HTTP_404_NOT_FOUND
    if code == "bad_request":
        return status.HTTP_400_BAD_REQUEST
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def _get_orchestrator(app: FastAPI) -> InferenceOrchestrator:
    orchestrator = getattr(app.state, "orchestrator", None)
    if orchestrator is None:
        raise RuntimeError("Orchestrator is not started")
    return orchestrator


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
        app.state.orchestrator = InferenceOrchestrator(
            pipeline=app.state.pipeline,
            settings=app.state.settings,
        )
        await app.state.orchestrator.start()
        yield
        await app.state.orchestrator.stop()

    app = FastAPI(
        title="Resume Recommendation Inference Service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.artifacts = artifacts
    app.state.pipeline = _build_pipeline(artifacts)
    app.state.load_error = None
    app.state.orchestrator = None

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

    @app.post(
        "/inference/jobs",
        response_model=InferenceJobCreateResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_inference_job(request: InferenceJobCreateRequest):
        try:
            return await _get_orchestrator(app).submit(request)
        except RuntimeError as exc:
            return _error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "orchestrator_not_available",
                str(exc),
            )

    @app.get("/inference/jobs/{job_id}", response_model=InferenceJobStatusResponse)
    def get_inference_job(job_id: str):
        try:
            return _get_orchestrator(app).status(job_id)
        except JobNotFoundError:
            return _error_response(
                status.HTTP_404_NOT_FOUND,
                "unknown_job_id",
                f"Unknown job_id: {job_id}",
            )
        except RuntimeError as exc:
            return _error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "orchestrator_not_available",
                str(exc),
            )

    @app.get("/inference/jobs/{job_id}/result", response_model=InferenceJobResultResponse)
    def get_inference_job_result(job_id: str):
        try:
            orchestrator = _get_orchestrator(app)
            job_status = orchestrator.status(job_id)
            if job_status.status == "failed" and job_status.error is not None:
                return _error_response(
                    _job_error_status(job_status.error.code),
                    job_status.error.code,
                    job_status.error.message,
                    job_status.error.details,
                )
            result = orchestrator.result(job_id)
            if result is None:
                return JSONResponse(
                    status_code=status.HTTP_202_ACCEPTED,
                    content=jsonable_encoder(job_status),
                )
            return result
        except JobNotFoundError:
            return _error_response(
                status.HTTP_404_NOT_FOUND,
                "unknown_job_id",
                f"Unknown job_id: {job_id}",
            )
        except RuntimeError as exc:
            return _error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "orchestrator_not_available",
                str(exc),
            )

    @app.post("/recommend", response_model=FrontendRecommendResponse)
    async def frontend_recommend(request: FrontendRecommendRequest):
        job_request = InferenceJobCreateRequest(
            vacancy=request.to_vacancy_fields(),
            candidate_limit=request.candidate_limit,
            result_limit=request.result_limit,
        )
        try:
            orchestrator = _get_orchestrator(app)
            created = await orchestrator.submit(job_request)
            deadline = asyncio.get_running_loop().time() + app.state.settings.recommend_timeout_seconds

            while asyncio.get_running_loop().time() < deadline:
                job_status = orchestrator.status(created.job_id)
                if job_status.status == "succeeded":
                    result = orchestrator.result(created.job_id)
                    return FrontendRecommendResponse(
                        recommendations=result.recommendations if result is not None else []
                    )
                if job_status.status == "failed" and job_status.error is not None:
                    return _error_response(
                        _job_error_status(job_status.error.code),
                        job_status.error.code,
                        job_status.error.message,
                        job_status.error.details,
                    )
                await asyncio.sleep(0.02)

            return _error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "inference_job_not_ready",
                "Inference job did not finish within frontend compatibility timeout",
                {"job_id": created.job_id},
            )
        except RuntimeError as exc:
            return _error_response(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "orchestrator_not_available",
                str(exc),
            )

    return app


app = create_app()


__all__ = ["app", "create_app"]
