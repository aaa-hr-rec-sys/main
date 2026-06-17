# Каркас inference-сервиса

Сервис содержит синхронный legacy endpoint для уже существующего
`vacancy_id_hash` и MVP-каркас backend orchestrator для async inference jobs.

На текущей версии новые vacancy fields принимаются контрактом, но job завершается ошибкой
`runtime_embedder_not_configured`. Это состояние до появления
отдельного stage 1 сервиса

## Что входит

- `GET /health` - процесс жив.
- `GET /ready` - bundle загружен, сервис готов к запросам.
- `POST /recommendations` - legacy-рекомендации по `vacancy_id_hash`.
- `POST /recommend` - compatibility endpoint для текущего frontend submodule.
- `POST /inference/jobs` - создать async inference job.
- `GET /inference/jobs/{job_id}` - статус job.
- `GET /inference/jobs/{job_id}/result` - результат job или промежуточный `202`.
- Exact retrieval по dot product поверх `cv_embeddings.npy`.
- CatBoost reranking тем же набором признаков, что и training pipeline.
- In-memory queue/job store для MVP.

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

Новый default для `--max-response-top-k` - `500`. Уже собранный bundle со старым
лимитом нужно пересобрать, чтобы legacy `/recommendations` мог вернуть больше 50.

## Запуск API

```bash
export PYTHONPATH=src
export ARTIFACT_DIR=artifacts/inference_bundle_v1
export ORCH_MAX_RESULT_LIMIT=500
python -m uvicorn service.api:app --host 127.0.0.1 --port 8000
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
```

Legacy-запрос:

```bash
curl -X POST http://127.0.0.1:8000/recommendations \
  -H "Content-Type: application/json" \
  -d '{"vacancy_id_hash":"9acbdf658912c3ef","top_k":10}'
```

Ответ содержит `cv_id_hash`, финальный `model_score`, `embedding_score`,
`embedding_rank`, позицию `rank` после reranking и доступные display-поля резюме.

## Frontend Compatibility

Текущий frontend из submodule отправляет форму вакансии в `POST /recommend`:

```json
{
  "vacancy_text": "Нужен администратор для работы с клиентами",
  "vac_profession": "администратор",
  "vac_group_profession": "административный персонал",
  "vac_business_category": "Офисные",
  "vac_sfera": "бытовые и персональные услуги",
  "vac_experience": "более 1 года",
  "vac_schedule": "фиксированный",
  "vac_employment_type": "полная занятость",
  "vac_education_level": "не имеет значения"
}
```

Пока stage 1/embedder не настроен, endpoint вернёт ошибку, а frontend продолжит
работать в demo mode:

```json
{
  "error": {
    "code": "runtime_embedder_not_configured",
    "message": "Runtime embedder/stage 1 service is not configured for new vacancy fields",
    "details": null
  }
}
```

После подключения stage 1 ответ будет иметь вид:

```json
{
  "recommendations": [
    {
      "id": "cv_id_hash",
      "cv_id_hash": "cv_id_hash",
      "rank": 1,
      "score": 0.93,
      "model_score": 0.93,
      "embedding_score": 0.81,
      "embedding_rank": 4,
      "profession": "администратор",
      "group_profession": "административный персонал",
      "business_category": "Офисные",
      "sfera": "административная работа",
      "experience_bucket": "есть опыт",
      "education": "среднее специальное",
      "federal_district": "Центральный федеральный округ",
      "employment_type": "только основная работа",
      "schedule": "полный день"
    }
  ]
}
```

## Async Orchestrator API

Создание job по существующему `vacancy_id_hash`:

```bash
curl -X POST http://127.0.0.1:8000/inference/jobs \
  -H "Content-Type: application/json" \
  -d '{"vacancy_id_hash":"9acbdf658912c3ef","result_limit":10}'
```

Создание job по новым vacancy fields:

```json
{
  "vacancy": {
    "vacancy_text": "Нужен администратор для работы с клиентами",
    "profession": "администратор",
    "group_profession": "административный персонал",
    "business_category": "Офисные",
    "sfera": "бытовые и персональные услуги",
    "experience": "более 1 года",
    "schedule": "фиксированный",
    "employment_type": "полная занятость",
    "education_level": "не имеет значения"
  },
  "candidate_limit": 500,
  "result_limit": 500
}
```

Статусы job:

```text
queued
running
stage1_running
stage2_running
postprocessing_running
succeeded
failed
```

Минимальные runtime-настройки:

```text
ARTIFACT_DIR=artifacts/inference_bundle_v1
ORCH_MAX_RESULT_LIMIT=500
ORCH_WORKER_COUNT=1
ORCH_MAX_JOBS=1000
RECOMMEND_TIMEOUT_SECONDS=0.2
```

## Контракты Stage 1 / Stage 2

Orchestrator -> stage 1/embedder:

```json
{
  "job_id": "uuid",
  "vacancy": {
    "vacancy_text": "string",
    "profession": "string",
    "group_profession": "string",
    "business_category": "string",
    "sfera": "string",
    "experience": "string",
    "schedule": "string",
    "employment_type": "string",
    "education_level": "string"
  },
  "candidate_limit": 500
}
```

Stage 1/embedder -> orchestrator:

```json
{
  "job_id": "uuid",
  "embedding_dim": 1024,
  "candidates": [
    {
      "cv_id_hash": "string",
      "embedding_score": 0.0,
      "embedding_rank": 1
    }
  ]
}
```

Orchestrator -> stage 2:

```json
{
  "job_id": "uuid",
  "vacancy": "normalized vacancy fields",
  "candidates": [
    {
      "cv_id_hash": "string",
      "embedding_score": 0.0,
      "embedding_rank": 1
    }
  ],
  "result_limit": 500
}
```

Stage 2 -> orchestrator:

```json
{
  "job_id": "uuid",
  "ranked": [
    {
      "cv_id_hash": "string",
      "rank": 1,
      "model_score": 0.0,
      "embedding_score": 0.0,
      "embedding_rank": 1,
      "display": {}
    }
  ]
}
```

## Тесты

```bash
python -m pytest
```

Smoke-тест на реальном bundle запускается отдельно:

```bash
INFERENCE_BUNDLE_DIR=artifacts/inference_bundle_v1 \
  python -m pytest tests/test_integration_bundle.py
```
