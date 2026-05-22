"""Утилиты разбиения positive-only событий откликов"""

from __future__ import annotations

import pandas as pd


def temporal_split(
    applies: pd.DataFrame,
    date_col: str = "applied_at_jittered",
    valid_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Делит события откликов по временному порядку

    Параметры
    ----------
    applies:
        Таблица откликов с datetime-like колонкой
    date_col:
        Колонка для хронологической сортировки
    valid_frac:
        Доля последних строк, попадающих в validation

    Возвращает
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Копии ``(train_applies, valid_applies)``
        Функция сортирует по ``date_col`` стабильной сортировкой, сохраняет
        исходные колонки и не мутирует входной DataFrame
    """
    if not 0 <= valid_frac < 1:
        raise ValueError("valid_frac должен быть в интервале [0, 1)")
    if date_col not in applies:
        raise KeyError(f"Колонка {date_col!r} отсутствует в applies")

    ordered = applies.sort_values(date_col, kind="mergesort")
    split_at = int(len(ordered) * (1 - valid_frac))
    return ordered.iloc[:split_at].copy(), ordered.iloc[split_at:].copy()


def check_split_leakage(
    train_applies: pd.DataFrame,
    valid_applies: pd.DataFrame,
    date_col: str = "applied_at_jittered",
) -> pd.DataFrame:
    """Проверяет простые leakage-риски между train и validation откликами

    Параметры
    ----------
    train_applies:
        Train-события откликов
    valid_applies:
        Validation-события откликов
    date_col:
        Колонка даты для проверки временного пересечения, если она есть в
        обеих таблицах

    Возвращает
    -------
    pd.DataFrame
        Однострочный отчет с дублями позитивных пар между split, числом unseen
        CV и вакансий в validation и информацией о временном пересечении

    Примечания
    -----
    Unseen CV или вакансии в validation не всегда являются leakage
    Они выводятся, потому что заметно меняют сценарий оценки: warm-start
    validation проще, чем validation на unseen сущностях
    """
    train_pairs = set(
        map(tuple, train_applies[["cv_id_hash", "vacancy_id_hash"]].to_numpy())
    )
    valid_pairs = set(
        map(tuple, valid_applies[["cv_id_hash", "vacancy_id_hash"]].to_numpy())
    )
    pair_overlap = train_pairs & valid_pairs

    train_cv = set(train_applies["cv_id_hash"])
    valid_cv = set(valid_applies["cv_id_hash"])
    train_vacancies = set(train_applies["vacancy_id_hash"])
    valid_vacancies = set(valid_applies["vacancy_id_hash"])

    train_max_date = pd.NaT
    valid_min_date = pd.NaT
    temporal_overlap = pd.NA
    if (
        date_col in train_applies
        and date_col in valid_applies
        and len(train_applies)
        and len(valid_applies)
    ):
        train_max_date = train_applies[date_col].max()
        valid_min_date = valid_applies[date_col].min()
        temporal_overlap = bool(train_max_date > valid_min_date)

    return pd.DataFrame(
        [
            {
                "train_rows": len(train_applies),
                "valid_rows": len(valid_applies),
                "pair_overlap_count": len(pair_overlap),
                "valid_unseen_cv_count": len(valid_cv - train_cv),
                "valid_unseen_cv_pct": (
                    len(valid_cv - train_cv) / len(valid_cv) * 100 if valid_cv else 0.0
                ),
                "valid_unseen_vacancy_count": len(valid_vacancies - train_vacancies),
                "valid_unseen_vacancy_pct": (
                    len(valid_vacancies - train_vacancies) / len(valid_vacancies) * 100
                    if valid_vacancies
                    else 0.0
                ),
                "train_max_date": train_max_date,
                "valid_min_date": valid_min_date,
                "temporal_overlap": temporal_overlap,
            }
        ]
    )


__all__ = ["check_split_leakage", "temporal_split"]
