"""Утилиты работы с данными для ноутбуков и скриптов проекта"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

TABLE_FILES = {
    "cv": "cv.parquet",
    "vacancies": "vacancies.parquet",
    "applies": "applies.parquet",
    "cv_embeddings": "cv_embeddings.parquet",
    "vacancies_embeddings": "vacancies_embeddings.parquet",
}


def get_data_paths(data_dir: str | Path) -> dict[str, Path]:
    """Возвращает канонические пути к raw parquet файлам датасета

    Параметры
    ----------
    data_dir:
        Директория с ``cv.parquet``, ``vacancies.parquet``,
        ``applies.parquet`` и parquet-файлами embeddings

    Возвращает
    -------
    dict[str, Path]
        Соответствие логических имен таблиц путям файлов
        Функция не проверяет существование файлов, чтобы вызывающий код сам
        выбирал нужную строгость
    """
    data_dir = Path(data_dir)
    return {name: data_dir / file_name for name, file_name in TABLE_FILES.items()}


def load_tables(
    data_dir: str | Path, names: Iterable[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Загружает выбранные raw parquet таблицы в обычные pandas DataFrame

    Параметры
    ----------
    data_dir:
        Директория с raw parquet файлами
    names:
        Опциональный список логических имен таблиц
        Если не передан, загружаются все известные таблицы

    Возвращает
    -------
    dict[str, pd.DataFrame]
        Загруженные таблицы по логическим именам
        Функция намеренно возвращает обычный dict вместо кастомного bundle,
        чтобы в ноутбуках можно было удобно держать отдельные DataFrame
    """
    paths = get_data_paths(data_dir)
    selected_names = list(paths) if names is None else list(names)
    unknown = sorted(set(selected_names) - set(paths))
    if unknown:
        raise KeyError(f"Неизвестные имена таблиц: {unknown}")
    return {name: pd.read_parquet(paths[name]) for name in selected_names}


def positive_density(
    cv: pd.DataFrame, vacancies: pd.DataFrame, applies: pd.DataFrame
) -> float:
    """Считает плотность позитивных пар в процентах

    Параметры
    ----------
    cv:
        Таблица CV с ``cv_id_hash``
    vacancies:
        Таблица вакансий с ``vacancy_id_hash``
    applies:
        Positive-only события откликов

    Возвращает
    -------
    float
        ``unique_positive_pairs / (unique_cv * unique_vacancies) * 100``
    """
    total_pairs = cv["cv_id_hash"].nunique() * vacancies["vacancy_id_hash"].nunique()
    if total_pairs == 0:
        return 0.0
    positive_pairs = (
        applies[["cv_id_hash", "vacancy_id_hash"]].drop_duplicates().shape[0]
    )
    return positive_pairs / total_pairs * 100


def basic_key_report(
    cv: pd.DataFrame, vacancies: pd.DataFrame, applies: pd.DataFrame
) -> pd.DataFrame:
    """Строит компактный отчет уникальности ключей основных таблиц

    Параметры
    ----------
    cv:
        Таблица CV с ``cv_id_hash``
    vacancies:
        Таблица вакансий с ``vacancy_id_hash``
    applies:
        Таблица откликов с ``cv_id_hash`` и ``vacancy_id_hash``

    Возвращает
    -------
    pd.DataFrame
        Отчет с числом строк, числом уникальных ключей и числом дублей для CV,
        вакансий и наблюдаемых пар откликов
    """
    return pd.DataFrame(
        [
            {
                "table": "cv",
                "key": "cv_id_hash",
                "rows": len(cv),
                "unique_keys": cv["cv_id_hash"].nunique(),
                "duplicate_keys": int(cv["cv_id_hash"].duplicated().sum()),
            },
            {
                "table": "vacancies",
                "key": "vacancy_id_hash",
                "rows": len(vacancies),
                "unique_keys": vacancies["vacancy_id_hash"].nunique(),
                "duplicate_keys": int(vacancies["vacancy_id_hash"].duplicated().sum()),
            },
            {
                "table": "applies",
                "key": "cv_id_hash + vacancy_id_hash",
                "rows": len(applies),
                "unique_keys": applies[["cv_id_hash", "vacancy_id_hash"]]
                .drop_duplicates()
                .shape[0],
                "duplicate_keys": int(
                    applies.duplicated(["cv_id_hash", "vacancy_id_hash"]).sum()
                ),
            },
        ]
    )


__all__ = [
    "TABLE_FILES",
    "basic_key_report",
    "get_data_paths",
    "load_tables",
    "positive_density",
]
