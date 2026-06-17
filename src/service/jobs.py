"""In-memory хранилище async inference jobs"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from service.schemas import (
    InferenceJobCreateRequest,
    InferenceJobError,
    InferenceJobResultResponse,
    InferenceJobStatusResponse,
    JobStatus,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class InferenceJob:
    job_id: str
    request: InferenceJobCreateRequest
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    stage: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    candidate_count: int | None = None
    result_count: int | None = None
    error: InferenceJobError | None = None
    result: InferenceJobResultResponse | None = None


class JobNotFoundError(KeyError):
    """Запрошенный job id не найден"""


class InMemoryJobStore:
    """Минимальное volatile-хранилище job state для MVP"""

    def __init__(self, max_jobs: int = 1_000):
        self.max_jobs = max_jobs
        self._jobs: dict[str, InferenceJob] = {}

    def create(self, request: InferenceJobCreateRequest) -> InferenceJob:
        if len(self._jobs) >= self.max_jobs:
            raise RuntimeError("Job store capacity exceeded")
        now = utc_now()
        job = InferenceJob(
            job_id=str(uuid4()),
            request=request,
            status="queued",
            created_at=now,
            updated_at=now,
        )
        self._jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> InferenceJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFoundError(job_id) from exc

    def set_status(self, job_id: str, status: JobStatus, stage: str | None = None) -> InferenceJob:
        job = self.get(job_id)
        now = utc_now()
        job.status = status
        job.stage = stage
        job.updated_at = now
        if status == "running" and job.started_at is None:
            job.started_at = now
        return job

    def set_result(
        self,
        job_id: str,
        result: InferenceJobResultResponse,
        candidate_count: int | None = None,
    ) -> InferenceJob:
        job = self.get(job_id)
        now = utc_now()
        job.status = "succeeded"
        job.stage = None
        job.updated_at = now
        job.completed_at = now
        job.result = result
        job.candidate_count = candidate_count
        job.result_count = len(result.recommendations)
        return job

    def set_error(
        self,
        job_id: str,
        code: str,
        message: str,
        details: object | None = None,
    ) -> InferenceJob:
        job = self.get(job_id)
        now = utc_now()
        job.status = "failed"
        job.updated_at = now
        job.completed_at = now
        job.error = InferenceJobError(code=code, message=message, details=details)
        return job


def job_status_response(job: InferenceJob) -> InferenceJobStatusResponse:
    return InferenceJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        stage=job.stage,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        candidate_count=job.candidate_count,
        result_count=job.result_count,
        error=job.error,
    )


__all__ = [
    "InferenceJob",
    "InMemoryJobStore",
    "JobNotFoundError",
    "job_status_response",
]
