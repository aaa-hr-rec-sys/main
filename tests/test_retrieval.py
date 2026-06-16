from __future__ import annotations

import numpy as np
import pytest

from service.retrieval import retrieve_top_k


def test_retrieve_top_k_uses_dot_product_and_rank_order():
    cv_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )
    cv_ids = np.array(["cv_1", "cv_2", "cv_3"])

    result = retrieve_top_k(
        vacancy_id_hash="vac_1",
        vacancy_embedding=np.array([1.0, 0.0], dtype=np.float32),
        cv_embeddings=cv_embeddings,
        cv_ids=cv_ids,
        top_k=2,
    )

    assert result["cv_id_hash"].tolist() == ["cv_1", "cv_3"]
    assert result["embedding_rank"].tolist() == [1, 2]
    assert result["embedding_score"].tolist() == pytest.approx([1.0, 0.6])
