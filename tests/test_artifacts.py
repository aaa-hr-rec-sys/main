from __future__ import annotations

import json

import pytest

from service.artifacts import ArtifactLoadError, load_inference_artifacts


def test_load_inference_artifacts_success(tiny_bundle_dir):
    artifacts = load_inference_artifacts(tiny_bundle_dir)

    assert artifacts.cv_embeddings.shape == (3, 3)
    assert artifacts.vacancy_embeddings.shape == (2, 3)
    assert artifacts.retrieval_top_k == 3
    assert len(artifacts.feature_columns) == len(artifacts.model.feature_names_)


def test_load_inference_artifacts_rejects_shape_mismatch(tiny_bundle_dir):
    manifest_path = tiny_bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["cv_embeddings"] = "vacancy_embeddings.npy"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ArtifactLoadError, match="cv_embeddings row count"):
        load_inference_artifacts(tiny_bundle_dir)

