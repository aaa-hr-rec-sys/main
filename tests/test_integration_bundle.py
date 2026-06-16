from __future__ import annotations

import os

import pytest

from service.artifacts import load_inference_artifacts
from service.pipeline import RecommendationPipeline


@pytest.mark.skipif(
    not os.getenv("INFERENCE_BUNDLE_DIR"),
    reason="Set INFERENCE_BUNDLE_DIR to run real bundle smoke test",
)
def test_real_bundle_smoke_from_env():
    artifacts = load_inference_artifacts(os.environ["INFERENCE_BUNDLE_DIR"])
    pipeline = RecommendationPipeline(artifacts)
    vacancy_id = str(artifacts.vacancy_ids[0])

    response = pipeline.recommend(vacancy_id, top_k=1)

    assert response.vacancy_id_hash == vacancy_id
    assert len(response.recommendations) == 1

