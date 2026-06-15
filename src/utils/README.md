# Utils для данных и baseline-подготовки

Пакет содержит минимальный переиспользуемый слой для
нормализации данных, базовой работы с embeddings, temporal split и легких
проверок таблиц

## Структура

- `normalization.py` - нормализация CV/vacancy таблиц, salary diagnostics,
  compatibility mappings
- `embeddings.py` - перевод embeddings в матрицы, отчеты по L2-нормам,
  chunked top-K retrieval
- `splits.py` - temporal split и простая проверка утечки между train/valid
- `data.py` - пути к parquet, загрузка таблиц, density и key report
- `../../scripts/normalize_data.py` - скрипт для сохранения нормализованной версии
  датасета

## Базовый порядок использования

```python
from utils.data import load_tables
from utils.normalization import normalize_cv, normalize_vacancies
from utils.embeddings import build_embedding_candidates
from utils.splits import temporal_split, check_split_leakage

tables = load_tables("data/aaa-out")
cv_norm = normalize_cv(tables["cv"])
vacancies_norm = normalize_vacancies(tables["vacancies"])

candidates = build_embedding_candidates(
    tables["vacancies_embeddings"],
    tables["cv_embeddings"],
    top_k=100,
)

train_applies, valid_applies = temporal_split(tables["applies"])
leakage_report = check_split_leakage(train_applies, valid_applies)
```

## Нормализация

`normalize_cv(cv)` и `normalize_vacancies(vacancies)` возвращают копии входных
DataFrame
Они не удаляют строки, не меняют порядок строк и не перетирают исходные колонки
Все изменения добавляются отдельными колонками

`normalize_cv` добавляет:

- `profession_norm`
- `group_profession_norm`
- `business_category_norm`
- `sfera_norm`
- `schedule_norm`
- `employment_type_norm`
- `education_norm`
- `experience_common`

`normalize_vacancies` добавляет:

- `profession_norm`
- `group_profession_norm`
- `business_category_norm`
- `sfera_norm`
- `schedule_norm`
- `employment_type_norm`
- `education_level_norm`
- `experience_common`

`business_category` не имеет отдельной функции нормализации: EDA показал, что
словарь маленький и уже хорошо совпадает между CV и vacancy
Для него используется только общая строковая нормализация

Зарплатные derived-колонки не добавляются в `normalize_cv`: исходный
`salary_bucketed` сохраняется без изменения, а диагностику можно получить
отдельной функцией `clean_salary`. Это оставляет normalized CV ближе к raw
таблице и не навязывает downstream-ноутбукам политику обработки зарплаты.

## Compatibility mappings

Compatibility-функции возвращают nullable boolean:

- `True` - пара значений совместима по текущему правилу
- `False` - пара значений явно несовместима
- `pd.NA` - нет надежного правила или значение неизвестно

### `schedule_compatible`

CV и vacancy используют разные словари, поэтому exact match бесполезен
Базовые правила:

- `полный день` -> `фиксированный`
- `свободный график` -> `гибкий`
- `сменный график` -> `сменный`
- `неполный день` -> `гибкий`
- `вахтовый метод` -> `вахта`
- `удаленная работа` -> `pd.NA`, потому что в vacancy нет надежного аналога

### `employment_type_compatible`

Базовые правила:

- `только основная работа` -> `полная занятость`
- `только подработка` -> `частичная занятость`, `временная`
- `смешанный` -> `полная занятость`, `частичная занятость`, `временная`
- vacancy `other` -> `pd.NA`

### `education_compatible`

Базовые правила:

- vacancy `NaN` или `не имеет значения` совместима с любым CV education
- vacancy `среднее профессиональное` совместима с CV `среднее специальное`
  и `высшее`
- CV `среднее` не покрывает требование `среднее профессиональное`
- CV `образование не указано`, `незаконченное высшее` и vacancy `other`
  возвращают `pd.NA`

### `experience_compatible`

Базовые правила:

- vacancy `без опыта` совместима с любым CV experience
- vacancy `более 1 года`, `более 3 лет`, `более 5 лет` совместима только
  с CV `есть опыт`
- missing и `other` возвращают `pd.NA`

Для каждой compatibility-функции есть vectorized wrapper с суффиксом `_series`

## Embedding candidates

`chunked_topk(query_matrix, doc_matrix, k, chunk_size)` считает dot product
частями
Это нужно, чтобы не держать всю матрицу `vacancies x cv` в памяти

`build_embedding_candidates(vacancies_embeddings, cv_embeddings, top_k)`
возвращает таблицу:

- `vacancy_id_hash`
- `cv_id_hash`
- `score`
- `rank`

Такой формат можно напрямую использовать в baseline-ноутбуках и позже для
hard-negative mining

## Версионирование нормализованных данных

Скрипт:

```bash
python main/scripts/normalize_data.py \
  --input-dir data/aaa-out \
  --output-root data/processed
```

Если `--version` не указан, создается версия вида:

```text
vYYYY-MM-DD_HHMMSS
```

Структура результата:

```text
data/processed/<version>/
  cv_normalized.parquet
  vacancies_normalized.parquet
  normalization_report.json
  manifest.json
data/processed/latest.txt
```

`latest.txt` содержит имя последней версии
Используется обычный текстовый файл, а не symlink, чтобы поведение было
предсказуемым на Windows

`manifest.json` хранит:

- версию и время создания
- пути к raw и processed данным
- размеры и SHA-256 checksums входных/выходных файлов
- row counts
- параметры нормализации
- версию mapping-логики
- версии Python, pandas и pyarrow

Embeddings и `applies` не копируются в processed-версию, чтобы не плодить
тяжелые дубли
Они остаются в raw `data/aaa-out`, а manifest сохраняет ссылки и checksums
