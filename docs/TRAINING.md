\# Training: Vacancy -> CV Two-Stage Recommendation Model



Эта инструкция описывает, как воспроизвести обучение финальной двухстадийной модели.



Система решает задачу:



```text

на вход: вакансия

на выход: top-K подходящих резюме

```



Pipeline состоит из двух стадий:



```text

1\. Retrieval / candidate generation:

&#x20;  vacancy embedding -> top-500 CV по cosine similarity



2\. Ranking:

&#x20;  CatBoostRanker переупорядочивает top-500 кандидатов

&#x20;  и формирует финальный top-10

```



\## 0. Подготовка данных



Сырые данные не входят в репозиторий.



Они должны лежать в:



```text

data/aaa-out/

```



Ожидаемые файлы:



```text

cv.parquet

cv\_embeddings.parquet

vacancies.parquet

vacancies\_embeddings.parquet

applies.parquet

```



Перед обучением нужно выполнить нормализацию:



```bash

python scripts/normalize\_data.py --input-dir data/aaa-out --output-root data/processed --version v1

```



После этого должна появиться папка:



```text

data/processed/v1/

```



В ней должны быть нормализованные таблицы:



```text

cv\_normalized.parquet

vacancies\_normalized.parquet

manifest.json

normalization\_report.json

```



Важно: исходные embeddings всё ещё должны оставаться в `data/aaa-out/`, потому что первая стадия retrieval использует `cv\_embeddings.parquet` и `vacancies\_embeddings.parquet`.



\---



\## 1. Stage 1: build retrieval candidate pool



Первая стадия строит candidate pool.



Для каждой вакансии считается similarity между embedding вакансии и embeddings всех CV.



Так как embeddings L2-нормированы, используется dot product, который эквивалентен cosine similarity.



Для каждой вакансии выбирается top-500 CV.



Команда:



```bash

python scripts/build\_ltr\_dataset.py \\

&#x20; --processed-version v1 \\

&#x20; --top-k 500 \\

&#x20; --negative-sampling-strategy hard \\

&#x20; --negative-ratio 5 \\

&#x20; --keep-debug-columns \\

&#x20; --dataset-name ltr\_v1\_top500\_neg5\_ohe

```





После выполнения появится папка:



```text

data/modeling/ltr\_v1\_top500\_neg5\_ohe/

```



Основные файлы внутри:



```text

train\_features.parquet

valid\_features.parquet

train\_positive\_pairs.parquet

valid\_positive\_pairs.parquet

feature\_columns.json

dataset\_summary.json

candidate\_generation\_metrics.csv

```



Что здесь происходит:



```text

positive pairs = historical applies

negative pairs = hard negatives из embedding top-500

negative ratio = 5 negatives на каждый positive

```



Эта команда одновременно делает:



```text

1\. temporal train/validation split

2\. embedding retrieval top-500

3\. построение pairwise features

4\. negative sampling для train

5\. сохранение train/valid датасетов для CatBoost

```



\---



\## 2. Stage 2: train CatBoostRanker



Финальная модель обучается поверх candidate pool из Stage 1.



Лучшая найденная конфигурация:



```text

model = CatBoostRanker

loss\_function = YetiRank

retrieval top-k = 500

negative sampling = hard-only, 5 negatives per positive

iterations = 500

depth = 6

learning\_rate = 0.05

l2\_leaf\_reg = 3

one\_hot\_max\_size = 10

label\_weighting = none

selection metric = NDCG@10

```



Команда обучения:



```bash

python scripts/train\_catboost\_ranker.py \\

&#x20; --dataset-dir data/modeling/ltr\_v1\_top500\_neg5\_ohe \\

&#x20; --run-name catboost\_final\_yetirank\_neg5 \\

&#x20; --loss-function YetiRank \\

&#x20; --iterations 500 \\

&#x20; --depth 6 \\

&#x20; --learning-rate 0.05 \\

&#x20; --l2-leaf-reg 3 \\

&#x20; --label-weighting none \\

&#x20; --one-hot-max-size 10 \\

&#x20; --early-stopping-rounds 80 \\

&#x20; --ks 5 10 20 50 100 500 \\

&#x20; --select-metric ndcg\_at\_k \\

&#x20; --select-k 10 \\

&#x20; --save-recommendations \\

&#x20; --k-inf 10

```



После обучения появятся:



```text

data/models/catboost\_final\_yetirank\_neg5/

data/experiments/catboost\_final\_yetirank\_neg5/

```



В `data/models/...` сохраняется `.cbm` модель.



В `data/experiments/...` сохраняются:



```text

metrics.csv

model\_results.csv

best\_model.json

run\_summary.json

recommendations\_top10\_\*.parquet

recommendations\_top10\_\*.csv

```



\---



\## 3. Сохранение финальной модели со второй стадии

Папка `models/` содержит только одну финальную модель для воспроизведения inference без переобучения.
