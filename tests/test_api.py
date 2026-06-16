from __future__ import annotations

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from service.api import create_app
from service.schemas import RecommendationRequest


def test_recommendation_request_validation():
    request = RecommendationRequest(vacancy_id_hash=" vac_1 ")
    assert request.vacancy_id_hash == "vac_1"
    assert request.top_k == 10

    with pytest.raises(ValidationError):
        RecommendationRequest(vacancy_id_hash="vac_1", top_k=0)

    with pytest.raises(ValidationError):
        RecommendationRequest(vacancy_id_hash="vac_1", top_k=51)


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

