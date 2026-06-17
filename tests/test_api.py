from __future__ import annotations

import time

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from service.api import create_app
from service.schemas import FrontendRecommendRequest, InferenceJobCreateRequest, RecommendationRequest


def test_recommendation_request_validation():
    request = RecommendationRequest(vacancy_id_hash=" vac_1 ")
    assert request.vacancy_id_hash == "vac_1"
    assert request.top_k == 10

    with pytest.raises(ValidationError):
        RecommendationRequest(vacancy_id_hash="vac_1", top_k=0)

    with pytest.raises(ValidationError):
        RecommendationRequest(vacancy_id_hash="vac_1", top_k=501)


def test_frontend_recommend_request_maps_to_vacancy_fields():
    request = FrontendRecommendRequest(
        vacancy_text=" text ",
        vac_profession=" администратор ",
        vac_group_profession="административный персонал",
        vac_business_category="Офисные",
        vac_sfera="бытовые и персональные услуги",
        vac_experience="более 1 года",
        vac_schedule="фиксированный",
        vac_employment_type="полная занятость",
        vac_education_level="не имеет значения",
    )

    vacancy = request.to_vacancy_fields()

    assert vacancy.vacancy_text == "text"
    assert vacancy.profession == "администратор"
    assert vacancy.education_level == "не имеет значения"


def test_inference_job_request_requires_input():
    with pytest.raises(ValidationError):
        InferenceJobCreateRequest()


def test_api_endpoints_smoke(tiny_artifacts):
    app = create_app(artifacts=tiny_artifacts)

    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        recommendations = client.post(
            "/recommendations",
            json={"vacancy_id_hash": "vac_1", "top_k": 2},
        )
        unknown = client.post(
            "/recommendations",
            json={"vacancy_id_hash": "missing", "top_k": 2},
        )
        invalid = client.post(
            "/recommendations",
            json={"vacancy_id_hash": "vac_1", "top_k": 0},
        )
        job_created = client.post(
            "/inference/jobs",
            json={"vacancy_id_hash": "vac_1", "result_limit": 2},
        )

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert recommendations.status_code == 200
    payload = recommendations.json()
    assert payload["vacancy_id_hash"] == "vac_1"
    assert len(payload["recommendations"]) == 2
    assert {"cv_id_hash", "rank", "model_score", "embedding_score", "embedding_rank"} <= set(
        payload["recommendations"][0]
    )
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "unknown_vacancy_id"
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "validation_error"
    assert job_created.status_code == 202
    assert job_created.json()["status"] == "queued"


def test_inference_job_lifecycle_for_existing_vacancy(tiny_artifacts):
    app = create_app(artifacts=tiny_artifacts)

    with TestClient(app) as client:
        created = client.post(
            "/inference/jobs",
            json={"vacancy_id_hash": "vac_1", "result_limit": 2},
        )
        job_id = created.json()["job_id"]

        result = None
        for _ in range(50):
            response = client.get(f"/inference/jobs/{job_id}/result")
            if response.status_code == 200:
                result = response.json()
                break
            time.sleep(0.02)

    assert created.status_code == 202
    assert result is not None
    assert result["status"] == "succeeded"
    assert len(result["recommendations"]) == 2
    assert {"id", "cv_id_hash", "score", "profession"} <= set(result["recommendations"][0])


def test_frontend_recommend_reports_missing_runtime_embedder(tiny_artifacts):
    app = create_app(artifacts=tiny_artifacts)
    payload = {
        "vacancy_text": "Нужен администратор",
        "vac_profession": "администратор",
        "vac_group_profession": "административный персонал",
        "vac_business_category": "Офисные",
        "vac_sfera": "бытовые и персональные услуги",
        "vac_experience": "более 1 года",
        "vac_schedule": "фиксированный",
        "vac_employment_type": "полная занятость",
        "vac_education_level": "не имеет значения",
    }

    with TestClient(app) as client:
        response = client.post("/recommend", json=payload)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "runtime_embedder_not_configured"
