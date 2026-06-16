# Каркас inference-сервиса

Сервис синхронно возвращает top-K резюме для уже существующего
`vacancy_id_hash`. На текущей версии новые эмбеддинги не генерируются.

## Что входит

- `GET /health` - процесс жив.
- `GET /ready` - bundle загружен, сервис готов к запросам.
- `POST /recommendations` - рекомендации по `vacancy_id_hash`.
- Exact retrieval по dot product поверх `cv_embeddings.npy`.
- CatBoost reranking тем же набором признаков, что и training pipeline.

## Сборка bundle

Из каталога `main/`:

```bash
python scripts/build_inference_bundle.py --overwrite
```

По умолчанию команда читает:

- `data/aaa-out/cv_embeddings.parquet`
- `data/aaa-out/vacancies_embeddings.parquet`
- `data/processed/v1/cv_normalized.parquet`
- `data/processed/v1/vacancies_normalized.parquet`
- `data/modeling/ltr_v1_top500_neg5_ohe/feature_columns.json`
- `models/catboost_ranker_lossYetiRank_it500_depth6_lr0p05_l23p0_onehot10_wnone.cbm`

И пишет bundle в `artifacts/inference_bundle_v1/`. Папка `artifacts/`
игнорируется git, потому что содержит производные данные.

## Запуск API

```bash
export PYTHONPATH=src
export ARTIFACT_DIR=artifacts/inference_bundle_v1
python -m uvicorn service.api:app --host 127.0.0.1 --port 8000
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Пример запроса:

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"vacancy_id_hash":"9acbdf658912c3ef","top_k":10}'
```

Ответ содержит `cv_id_hash`, финальный `model_score`, `embedding_score`,
`embedding_rank`, позицию `rank` после reranking и доступные display-поля резюме

## Тесты

Обычные тесты не требуют локальной папки `data/`:

```bash
python -m pytest
```

Smoke-тест на реальном bundle запускается отдельно:

```bash
INFERENCE_BUNDLE_DIR=artifacts/inference_bundle_v1 \
  python -m pytest tests/test_integration_bundle.py
```
