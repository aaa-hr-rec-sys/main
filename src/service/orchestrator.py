"""Легкий in-memory orchestrator для async inference jobs"""

from __future__ import annotations

import asyncio
from typing import Any

from service.config import ServiceSettings
from service.jobs import InMemoryJobStore, InferenceJob, job_status_response
from service.pipeline import RecommendationPipeline, UnknownVacancyError
from service.schemas import (
    FrontendRecommendationItem,
    InferenceJobCreateRequest,
    InferenceJobCreateResponse,
    InferenceJobResultResponse,
    InferenceJobStatusResponse,
    RecommendationItem,
    RecommendationResponse,
)


class RuntimeEmbedderNotConfigured(RuntimeError):
    """Новые vacancy fields нельзя обработать без stage 1/embedder service"""


class ServiceNotReadyError(RuntimeError):
    """Inference artifacts или pipeline ещё не загружены"""


def _display_value(display: dict[str, Any], key: str) -> Any:
    return display.get(key)


def recommendation_to_frontend(item: RecommendationItem) -> FrontendRecommendationItem:
    display = item.display
    return FrontendRecommendationItem(
        id=item.cv_id_hash,
        cv_id_hash=item.cv_id_hash,
        rank=item.rank,
        score=item.model_score,
        model_score=item.model_score,
        embedding_score=item.embedding_score,
        embedding_rank=item.embedding_rank,
        profession=_display_value(display, "profession"),
        group_profession=_display_value(display, "group_profession"),
        business_category=_display_value(display, "business_category"),
        sfera=_display_value(display, "sfera"),
        experience_bucket=_display_value(display, "experience_bucket"),
        education=_display_value(display, "education"),
        federal_district=_display_value(display, "federal_district"),
        salary_bucketed=_display_value(display, "salary_bucketed"),
        employment_type=_display_value(display, "employment_type"),
        schedule=_display_value(display, "schedule"),
    )


def response_to_frontend_result(
    job_id: str,
    response: RecommendationResponse,
    result_limit: int,
) -> InferenceJobResultResponse:
    return InferenceJobResultResponse(
        job_id=job_id,
        status="succeeded",
        result_limit=result_limit,
        recommendations=[
            recommendation_to_frontend(item)
            for item in response.recommendations[:result_limit]
        ],
    )


class InferenceOrchestrator:
    """In-process queue + workers для MVP async inference"""

    def __init__(
        self,
        pipeline: RecommendationPipeline | None,
        settings: ServiceSettings,
        store: InMemoryJobStore | None = None,
    ):
        self.pipeline = pipeline
        self.settings = settings
        self.store = store or InMemoryJobStore(max_jobs=settings.max_jobs)
        self._queue: asyncio.Queue[str] | None = None
        self._workers: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        if self._queue is not None:
            return
        self._queue = asyncio.Queue()
        worker_count = max(1, self.settings.worker_count)
        self._workers = [
            asyncio.create_task(self._worker(), name=f"inference-worker-{idx}")
            for idx in range(worker_count)
        ]

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        self._queue = None

    async def submit(self, request: InferenceJobCreateRequest) -> InferenceJobCreateResponse:
        if self._queue is None:
            raise RuntimeError("Orchestrator is not started")
        job = self.store.create(request)
        await self._queue.put(job.job_id)
        return InferenceJobCreateResponse(job_id=job.job_id, status=job.status)

    def status(self, job_id: str) -> InferenceJobStatusResponse:
        return job_status_response(self.store.get(job_id))

    def result(self, job_id: str) -> InferenceJobResultResponse | None:
        return self.store.get(job_id).result

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            job_id = await self._queue.get()
            try:
                await self._run_job(self.store.get(job_id))
            finally:
                self._queue.task_done()

    async def _run_job(self, job: InferenceJob) -> None:
        self.store.set_status(job.job_id, "running", stage="start")
        try:
            response = await self._run_pipeline(job)
            result_limit = min(job.request.result_limit, self.settings.max_result_limit)
            result = response_to_frontend_result(
                job_id=job.job_id,
                response=response,
                result_limit=result_limit,
            )
            self.store.set_status(job.job_id, "postprocessing_running", stage="postprocessing")
            self.store.set_result(
                job.job_id,
                result=result,
                candidate_count=len(response.recommendations),
            )
        except RuntimeEmbedderNotConfigured as exc:
            self.store.set_error(job.job_id, "runtime_embedder_not_configured", str(exc))
        except ServiceNotReadyError as exc:
            self.store.set_error(job.job_id, "service_not_ready", str(exc))
        except UnknownVacancyError:
            self.store.set_error(
                job.job_id,
                "unknown_vacancy_id",
                f"Unknown vacancy_id_hash: {job.request.vacancy_id_hash}",
            )
        except ValueError as exc:
            self.store.set_error(job.job_id, "bad_request", str(exc))
        except Exception as exc:  # pragma: no cover - защитная граница worker
            self.store.set_error(job.job_id, "inference_error", "Inference job failed", str(exc))

    async def _run_pipeline(self, job: InferenceJob) -> RecommendationResponse:
        if self.pipeline is None:
            raise ServiceNotReadyError("Inference artifacts are not loaded")

        if job.request.vacancy_id_hash is None:
            self.store.set_status(job.job_id, "stage1_running", stage="embedder")
            raise RuntimeEmbedderNotConfigured(
                "Runtime embedder/stage 1 service is not configured for new vacancy fields"
            )

        result_limit = min(job.request.result_limit, self.settings.max_result_limit)
        self.store.set_status(job.job_id, "stage1_running", stage="local_retrieval")
        self.store.set_status(job.job_id, "stage2_running", stage="catboost_rerank")
        return await asyncio.to_thread(
            self.pipeline.recommend,
            vacancy_id_hash=job.request.vacancy_id_hash,
            top_k=result_limit,
        )


__all__ = [
    "InferenceOrchestrator",
    "RuntimeEmbedderNotConfigured",
    "ServiceNotReadyError",
    "recommendation_to_frontend",
    "response_to_frontend_result",
]
