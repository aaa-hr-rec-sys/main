from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.features import get_categorical_feature_columns, get_numeric_feature_columns
from service.artifacts import load_inference_artifacts
from service.precompute import FINAL_MODEL_NAME


@pytest.fixture()
def tiny_bundle_dir(tmp_path: Path) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    cv_store = pd.DataFrame(
        [
            {
                "cv_id_hash": "cv_1",
                "profession": "python developer",
                "group_profession": "it",
                "business_category": "office",
                "sfera": "software",
                "experience_bucket": "has experience",
                "education": "higher",
                "federal_district": "central",
                "salary_bucketed": 120000.0,
                "employment_type": "full",
                "schedule": "full day",
                "profession_norm": "python developer",
                "group_profession_norm": "it",
                "business_category_norm": "office",
                "sfera_norm": "software",
                "schedule_norm": "full day",
                "employment_type_norm": "full",
                "education_norm": "higher",
                "experience_common": "has_experience",
            },
            {
                "cv_id_hash": "cv_2",
                "profession": "sales manager",
                "group_profession": "sales",
                "business_category": "office",
                "sfera": "retail",
                "experience_bucket": "no experience",
                "education": "middle",
                "federal_district": "siberian",
                "salary_bucketed": None,
                "employment_type": "part",
                "schedule": "shift",
                "profession_norm": "sales manager",
                "group_profession_norm": "sales",
                "business_category_norm": "office",
                "sfera_norm": "retail",
                "schedule_norm": "shift",
                "employment_type_norm": "part",
                "education_norm": "middle",
                "experience_common": "no_experience",
            },
            {
                "cv_id_hash": "cv_3",
                "profession": "data analyst",
                "group_profession": "it",
                "business_category": "office",
                "sfera": "analytics",
                "experience_bucket": "has experience",
                "education": "higher",
                "federal_district": "ural",
                "salary_bucketed": 90000.0,
                "employment_type": "full",
                "schedule": "remote",
                "profession_norm": "data analyst",
                "group_profession_norm": "it",
                "business_category_norm": "office",
                "sfera_norm": "analytics",
                "schedule_norm": "remote",
                "employment_type_norm": "full",
                "education_norm": "higher",
                "experience_common": "has_experience",
            },
        ]
    )
    vacancy_store = pd.DataFrame(
        [
            {
                "vacancy_id_hash": "vac_1",
                "profession": "python developer",
                "group_profession": "it",
                "business_category": "office",
                "sfera": "software",
                "experience": "has experience",
                "schedule": "full day",
                "employment_type": "full",
                "education_level": "higher",
                "profession_norm": "python developer",
                "group_profession_norm": "it",
                "business_category_norm": "office",
                "sfera_norm": "software",
                "schedule_norm": "full day",
                "employment_type_norm": "full",
                "education_level_norm": "higher",
                "experience_common": "has_experience",
            },
            {
                "vacancy_id_hash": "vac_2",
                "profession": "sales manager",
                "group_profession": "sales",
                "business_category": "office",
                "sfera": "retail",
                "experience": "no experience",
                "schedule": "shift",
                "employment_type": "part",
                "education_level": "middle",
                "profession_norm": "sales manager",
                "group_profession_norm": "sales",
                "business_category_norm": "office",
                "sfera_norm": "retail",
                "schedule_norm": "shift",
                "employment_type_norm": "part",
                "education_level_norm": "middle",
                "experience_common": "no_experience",
            },
        ]
    )

    cv_ids = pd.DataFrame({"cv_id_hash": ["cv_1", "cv_2", "cv_3"]})
    vacancy_ids = pd.DataFrame({"vacancy_id_hash": ["vac_1", "vac_2"]})
    cv_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    vacancy_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    cv_store.to_parquet(bundle_dir / "cv_store.parquet", index=False)
    vacancy_store.to_parquet(bundle_dir / "vacancy_store.parquet", index=False)
    cv_ids.to_parquet(bundle_dir / "cv_ids.parquet", index=False)
    vacancy_ids.to_parquet(bundle_dir / "vacancy_ids.parquet", index=False)
    np.save(bundle_dir / "cv_embeddings.npy", cv_embeddings)
    np.save(bundle_dir / "vacancy_embeddings.npy", vacancy_embeddings)
    shutil.copy2(ROOT_DIR / "models" / FINAL_MODEL_NAME, bundle_dir / "model.cbm")

    numeric_columns = get_numeric_feature_columns()
    categorical_columns = get_categorical_feature_columns()
    (bundle_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "numeric_feature_columns": numeric_columns,
                "categorical_feature_columns": categorical_columns,
                "feature_columns": numeric_columns + categorical_columns,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (bundle_dir / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_version": "test",
                "files": {
                    "feature_schema": "feature_schema.json",
                    "model": "model.cbm",
                    "cv_store": "cv_store.parquet",
                    "vacancy_store": "vacancy_store.parquet",
                    "cv_ids": "cv_ids.parquet",
                    "vacancy_ids": "vacancy_ids.parquet",
                    "cv_embeddings": "cv_embeddings.npy",
                    "vacancy_embeddings": "vacancy_embeddings.npy",
                },
                "config": {
                    "retrieval_top_k": 3,
                    "max_response_top_k": 50,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return bundle_dir


@pytest.fixture()
def tiny_artifacts(tiny_bundle_dir: Path):
    return load_inference_artifacts(tiny_bundle_dir)

