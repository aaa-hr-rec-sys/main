from __future__ import annotations

from service.pipeline import RecommendationPipeline, UnknownVacancyError


def test_recommendation_pipeline_smoke(tiny_artifacts):
    pipeline = RecommendationPipeline(tiny_artifacts)

    response = pipeline.recommend("vac_1", top_k=2)

    assert response.vacancy_id_hash == "vac_1"
    assert response.top_k == 2
    assert len(response.recommendations) == 2
    assert response.recommendations[0].rank == 1
    assert response.recommendations[0].cv_id_hash.startswith("cv_")
    assert isinstance(response.recommendations[0].model_score, float)
    assert isinstance(response.recommendations[0].embedding_score, float)
    assert response.recommendations[0].embedding_rank >= 1
    assert "profession" in response.recommendations[0].display


def test_recommendation_pipeline_unknown_vacancy(tiny_artifacts):
    pipeline = RecommendationPipeline(tiny_artifacts)

    try:
        pipeline.recommend("missing", top_k=1)
    except UnknownVacancyError:
        return

    raise AssertionError("UnknownVacancyError was not raised")

