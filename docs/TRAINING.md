# Training: Vacancy -> CV Two-Stage Recommendation Model

Эта инструкция описывает, как воспроизвести обучение финальной двухстадийной модели.

Система решает задачу:

```text
на вход: вакансия
на выход: top-K подходящих резюме
```

Pipeline состоит из двух стадий:

```text
1. Retrieval / candidate generation:
   vacancy embedding -> top-500 CV по cosine similarity
2. Ranking:
   CatBoostRanker переупорядочивает top-500 кандидатов
   и формирует финальный top-10
```

## 0. Подготовка данных

Сырые данные не входят в репозиторий.

Они должны лежать в:

```text
data/aaa-out/
```

Ожидаемые файлы:

```text
cv.parquet
cv_embeddings.parquet
vacancies.parquet
vacancies_embeddings.parquet
applies.parquet
```

Перед обучением нужно выполнить нормализацию:

```bash
python scripts/normalize_data.py --input-dir data/aaa-out --output-root data/processed --version v1
```

После этого должна появиться папка:

```text
data/processed/v1/
```

В ней должны быть нормализованные таблицы:

```text
cv_normalized.parquet
vacancies_normalized.parquet
manifest.json
normalization_report.json
```

Важно: исходные embeddings всё ещё должны оставаться в `data/aaa-out/`, потому что первая стадия retrieval использует `cv_embeddings.parquet` и `vacancies_embeddings.parquet`.

---

## 1. Stage 1: build retrieval candidate pool

Первая стадия строит candidate pool.

Для каждой вакансии считается similarity между embedding вакансии и embeddings всех CV.

Так как embeddings L2-нормированы, используется dot product, который эквивалентен cosine similarity.

Для каждой вакансии выбирается top-500 CV.

Команда:

```bash
python scripts/build_ltr_dataset.py \
  --processed-version v1 \
  --top-k 500 \
  --negative-sampling-strategy hard \
  --negative-ratio 5 \
  --keep-debug-columns \
  --dataset-name ltr_v1_top500_neg5_ohe
```

После выполнения появится папка:

```text
data/modeling/ltr_v1_top500_neg5_ohe/
```

Основные файлы внутри:

```text
train_features.parquet
valid_features.parquet
train_positive_pairs.parquet
valid_positive_pairs.parquet
feature_columns.json
dataset_summary.json
candidate_generation_metrics.csv
```

Что здесь происходит:

```text
positive pairs = historical applies
negative pairs = hard negatives из embedding top-500
negative ratio = 5 negatives на каждый positive
```

Эта команда одновременно делает:

```text
1. temporal train/validation split
2. embedding retrieval top-500
3. построение pairwise features
4. negative sampling для train
5. сохранение train/valid датасетов для CatBoost
```

---

## 2. Stage 2: train CatBoostRanker

Финальная модель обучается поверх candidate pool из Stage 1.

Лучшая найденная конфигурация:

```text
model = CatBoostRanker
loss_function = YetiRank
retrieval top-k = 500
negative sampling = hard-only, 5 negatives per positive
iterations = 500
depth = 6
learning_rate = 0.05
l2_leaf_reg = 3
one_hot_max_size = 10
label_weighting = none
selection metric = NDCG@10
```

Команда обучения:

```bash
python scripts/train_catboost_ranker.py \
  --dataset-dir data/modeling/ltr_v1_top500_neg5_ohe \
  --run-name catboost_final_yetirank_neg5 \
  --loss-function YetiRank \
  --iterations 500 \
  --depth 6 \
  --learning-rate 0.05 \
  --l2-leaf-reg 3 \
  --label-weighting none \
  --one-hot-max-size 10 \
  --early-stopping-rounds 80 \
  --ks 5 10 20 50 100 500 \
  --select-metric ndcg_at_k \
  --select-k 10 \
  --save-recommendations \
  --k-inf 10
```

После обучения появятся:

```text
data/models/catboost_final_yetirank_neg5/
data/experiments/catboost_final_yetirank_neg5/
```

В `data/models/...` сохраняется `.cbm` модель.

В `data/experiments/...` сохраняются:

```text
metrics.csv
model_results.csv
best_model.json
run_summary.json
recommendations_top10_*.parquet
recommendations_top10_*.csv
```

---

## 3. Сохранение финальной модели со второй стадии

Папка `models/` содержит только одну финальную модель для воспроизведения inference без переобучения.
