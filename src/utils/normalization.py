"""Утилиты нормализации таблиц резюме и вакансий

Функции в этом модуле сохраняют исходные колонки и добавляют нормализованные
колонки для EDA, baseline retrieval и будущего моделирования
Compatibility-функции намеренно возвращают nullable boolean: ``True`` для
известной совместимой пары, ``False`` для известной несовместимой пары и
``pd.NA`` для неизвестного или неоднозначного правила
"""

from __future__ import annotations

import re

import pandas as pd

UNKNOWN = "unknown"
OTHER_VALUES = {"other", "другая", "другое", "другой", "прочее"}

KNOWN_VACANCY_SCHEDULES = {"гибкий", "сменный", "вахта", "фиксированный"}
SCHEDULE_COMPATIBILITY = {
    "полный день": {"фиксированный"},
    "свободный график": {"гибкий"},
    "сменный график": {"сменный"},
    "неполный день": {"гибкий"},
    "вахтовый метод": {"вахта"},
    "удаленная работа": None,
}

KNOWN_VACANCY_EMPLOYMENT_TYPES = {
    "полная занятость",
    "частичная занятость",
    "временная",
}
EMPLOYMENT_TYPE_COMPATIBILITY = {
    "только основная работа": {"полная занятость"},
    "только подработка": {"частичная занятость", "временная"},
    "смешанный": {"полная занятость", "частичная занятость", "временная"},
}

NO_EXPERIENCE = "no_experience"
HAS_EXPERIENCE = "has_experience"


def _normalize_value(value: object) -> str | None:
    """Нормализует одно категориальное значение или возвращает ``None`` для пропуска"""
    if pd.isna(value):
        return None
    normalized = str(value).replace("\xa0", " ").replace("ё", "е").strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized or None


def normalize_text_category(series: pd.Series) -> pd.Series:
    """Нормализует строковые категории без маппинга под конкретное поле

    Параметры
    ----------
    series:
        Исходные категориальные значения

    Возвращает
    -------
    pd.Series
        Series с dtype ``string``, где значения приведены к нижнему регистру,
        очищены от краевых пробелов, нормализованы из ``ё`` в ``е``, а
        повторяющиеся пробелы схлопнуты
        Пустые строки и пропуски превращаются в ``pd.NA``

    Пример
    -------
    ``"  Удалённая   работа "`` превращается в ``"удаленная работа"``
    """
    normalized = (
        series.astype("string")
        .str.replace("\xa0", " ", regex=False)
        .str.replace("ё", "е", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )
    return normalized.mask(normalized.eq(""), pd.NA)


def normalize_other_values(series: pd.Series) -> pd.Series:
    """Унифицирует обобщенные категории вида ``other``

    Параметры
    ----------
    series:
        Исходные категориальные значения

    Возвращает
    -------
    pd.Series
        Нормализованные категории, где ``other``, ``другая``, ``другое``,
        ``другой`` и ``прочее`` представлены как ``other``

    Примечания
    -----
    Функция полезна для колонок вроде profession или sfera
    """
    normalized = normalize_text_category(series)
    return normalized.mask(normalized.isin(OTHER_VALUES), "other")


def normalize_sfera(series: pd.Series) -> pd.Series:
    """Нормализует ``sfera`` и известные близкие значения

    Параметры
    ----------
    series:
        Сырые значения ``sfera`` из таблиц CV или вакансий

    Возвращает
    -------
    pd.Series
        Нормализованные значения ``sfera`` с унифицированным ``other`` и
        известными близкими значениями, приведенными к общему написанию

    Правила mapping
    -------------
    - ``медицина и фармацевтика`` -> ``медицина, фармацевтика``
    - ``производство, сырье, с/х`` остается в написании через ``е`` после
      общего правила ``ё -> е``
    """
    normalized = normalize_other_values(series)
    mapping = {
        "медицина и фармацевтика": "медицина, фармацевтика",
        "медицина, фармацевтика": "медицина, фармацевтика",
        "производство, сырье, с/х": "производство, сырье, с/х",
    }
    return normalized.replace(mapping)


def clean_salary(series: pd.Series, outlier_threshold: float = 500_000) -> pd.DataFrame:
    """Очищает и флагирует зарплатные бакеты CV

    Параметры
    ----------
    series:
        Сырые значения ``salary_bucketed`` из ``cv.parquet``
    outlier_threshold:
        Значения строго выше этого порога получают флаг выброса
        По умолчанию они не клиппятся, чтобы следующие ноутбуки могли выбрать
        собственную политику обработки

    Возвращает
    -------
    pd.DataFrame
        Четыре колонки с тем же индексом, что и входная Series:
        ``salary_clean`` с отрицательными значениями, переведенными в ``NaN``
        ``salary_missing`` для пропусков и нечисловых исходных значений
        ``salary_negative`` для значений ниже нуля после salary jittering
        ``salary_outlier`` для значений выше ``outlier_threshold``

    Пример
    -------
    >>> salary = pd.Series(
    ...     [80_000, "-1000", "not specified", None, 700_000],
    ...     name="salary_bucketed",
    ... )
    >>> clean_salary(salary, outlier_threshold=500_000)
       salary_clean  salary_missing  salary_negative  salary_outlier
    0       80000.0           False            False           False
    1           NaN           False             True           False
    2           NaN            True            False           False
    3           NaN            True            False           False
    4      700000.0           False            False            True
    """
    numeric = pd.to_numeric(series, errors="coerce")
    missing = series.isna() | numeric.isna()
    negative = numeric.lt(0).fillna(False)
    clean = numeric.mask(negative)
    outlier = clean.gt(outlier_threshold).fillna(False)
    return pd.DataFrame(
        {
            "salary_clean": clean,
            "salary_missing": missing.astype("boolean"),
            "salary_negative": negative.astype("boolean"),
            "salary_outlier": outlier.astype("boolean"),
        },
        index=series.index,
    )


def map_cv_experience_to_common(series: pd.Series) -> pd.Series:
    """Приводит опыт CV к грубой общей шкале

    Параметры
    ----------
    series:
        Сырые значения ``experience_bucket`` из таблицы CV

    Возвращает
    -------
    pd.Series
        ``no_experience`` для ``без опыта``, ``has_experience`` для
        ``есть опыт`` и ``unknown`` для пропусков или неизвестных значений
    """
    normalized = normalize_other_values(series)
    mapped = normalized.map({"без опыта": NO_EXPERIENCE, "есть опыт": HAS_EXPERIENCE})
    return mapped.fillna(UNKNOWN).astype("string")


def map_vacancy_experience_to_common(series: pd.Series) -> pd.Series:
    """Приводит требования вакансии к опыту к грубой общей шкале

    Параметры
    ----------
    series:
        Сырые значения ``experience`` из таблицы вакансий

    Возвращает
    -------
    pd.Series
        ``no_experience`` для ``без опыта``; ``has_experience`` для
        ``более 1 года``, ``более 3 лет`` и ``более 5 лет``; ``unknown``
        для пропусков, ``other`` или неизвестных значений
    """
    normalized = normalize_other_values(series)
    mapped = normalized.map(
        {
            "без опыта": NO_EXPERIENCE,
            "более 1 года": HAS_EXPERIENCE,
            "более 3 лет": HAS_EXPERIENCE,
            "более 5 лет": HAS_EXPERIENCE,
        }
    )
    return mapped.fillna(UNKNOWN).astype("string")


def schedule_compatible(cv_schedule: object, vacancy_schedule: object) -> object:
    """Проверяет совместимость графика из CV с графиком вакансии

    Параметры
    ----------
    cv_schedule:
        Значение графика со стороны CV, например ``полный день``
    vacancy_schedule:
        Значение графика со стороны вакансии, например ``фиксированный``

    Возвращает
    -------
    bool | pd.NA
        ``True`` для известной совместимой пары, ``False`` для известной
        несовместимой пары и ``pd.NA`` для пропусков или неизвестного mapping

    Правила mapping
    -------------
    - ``полный день`` совместим с ``фиксированный``
    - ``свободный график`` совместим с ``гибкий``
    - ``сменный график`` совместим с ``сменный``
    - ``неполный день`` совместим с ``гибкий``
    - ``вахтовый метод`` совместим с ``вахта``
    - ``удаленная работа`` не имеет надежного аналога в текущем словаре
      вакансий и поэтому возвращает ``pd.NA``
    """
    cv_norm = _normalize_value(cv_schedule)
    vacancy_norm = _normalize_value(vacancy_schedule)
    if cv_norm is None or vacancy_norm is None:
        return pd.NA
    allowed = SCHEDULE_COMPATIBILITY.get(cv_norm)
    if allowed is None or vacancy_norm not in KNOWN_VACANCY_SCHEDULES:
        return pd.NA
    return vacancy_norm in allowed


def schedule_compatible_series(
    cv_schedule: pd.Series, vacancy_schedule: pd.Series
) -> pd.Series:
    """Векторизованная обертка над :func:`schedule_compatible`

    Параметры
    ----------
    cv_schedule:
        Series со значениями графика со стороны CV
    vacancy_schedule:
        Series со значениями графика со стороны вакансии, выровненная по позиции

    Возвращает
    -------
    pd.Series
        Nullable boolean Series со значениями совместимости и индексом CV Series

    Пример
    -------
    >>> cv_schedule = pd.Series([
    ...     "полный день",
    ...     "свободный график",
    ...     "удаленная работа",
    ...     None,
    ... ])
    >>> vacancy_schedule = pd.Series([
    ...     "фиксированный",
    ...     "сменный",
    ...     "фиксированный",
    ...     "гибкий",
    ... ])
    >>> schedule_compatible_series(cv_schedule, vacancy_schedule)
    0     True
    1    False
    2     <NA>
    3     <NA>
    dtype: boolean
    """
    return pd.Series(
        [
            schedule_compatible(cv_value, vacancy_value)
            for cv_value, vacancy_value in zip(cv_schedule, vacancy_schedule)
        ],
        index=cv_schedule.index,
        dtype="boolean",
    )


def employment_type_compatible(
    cv_employment: object, vacancy_employment: object
) -> object:
    """Проверяет совместимость типа занятости из CV с типом занятости вакансии

    Параметры
    ----------
    cv_employment:
        Значение занятости со стороны CV: ``только основная работа``,
        ``только подработка`` или ``смешанный``
    vacancy_employment:
        Значение занятости со стороны вакансии: ``полная занятость``,
        ``частичная занятость``, ``временная`` или ``other``

    Возвращает
    -------
    bool | pd.NA
        Nullable результат совместимости

    Правила mapping
    -------------
    - ``только основная работа`` совместима с ``полная занятость``
    - ``только подработка`` совместима с ``частичная занятость`` и
      ``временная``
    - ``смешанный`` совместим с ``полная занятость``,
      ``частичная занятость`` и ``временная``
    - ``other`` или пропуск со стороны вакансии возвращает ``pd.NA``
    """
    cv_norm = _normalize_value(cv_employment)
    vacancy_norm = _normalize_value(vacancy_employment)
    if cv_norm is None or vacancy_norm is None or vacancy_norm == "other":
        return pd.NA
    allowed = EMPLOYMENT_TYPE_COMPATIBILITY.get(cv_norm)
    if allowed is None or vacancy_norm not in KNOWN_VACANCY_EMPLOYMENT_TYPES:
        return pd.NA
    return vacancy_norm in allowed


def employment_type_compatible_series(
    cv_employment: pd.Series, vacancy_employment: pd.Series
) -> pd.Series:
    """Векторизованная обертка над :func:`employment_type_compatible`

    Пример
    -------
    >>> cv_employment = pd.Series([
    ...     "только основная работа",
    ...     "только подработка",
    ...     "смешанный",
    ...     "только основная работа",
    ...     None,
    ... ])
    >>> vacancy_employment = pd.Series([
    ...     "полная занятость",
    ...     "временная",
    ...     "частичная занятость",
    ...     "частичная занятость",
    ...     "полная занятость",
    ... ])
    >>> employment_type_compatible_series(cv_employment, vacancy_employment)
    0     True
    1     True
    2     True
    3    False
    4     <NA>
    dtype: boolean
    """
    return pd.Series(
        [
            employment_type_compatible(cv_value, vacancy_value)
            for cv_value, vacancy_value in zip(cv_employment, vacancy_employment)
        ],
        index=cv_employment.index,
        dtype="boolean",
    )


def education_compatible(
    cv_education: object, vacancy_education_level: object
) -> object:
    """Проверяет, покрывает ли образование из CV требование вакансии

    Параметры
    ----------
    cv_education:
        Значение образования со стороны CV, например ``высшее`` или
        ``среднее специальное``
    vacancy_education_level:
        Требование со стороны вакансии, например ``не имеет значения`` или
        ``среднее профессиональное``

    Возвращает
    -------
    bool | pd.NA
        Nullable результат совместимости

    Правила mapping
    -------------
    - Пропуск в требовании вакансии и ``не имеет значения`` совместимы с
      любым значением образования CV
    - ``среднее профессиональное`` совместимо с CV ``среднее специальное`` и
      ``высшее``
    - CV ``среднее`` не покрывает требование ``среднее профессиональное``
    - CV ``образование не указано``, CV ``незаконченное высшее`` и vacancy
      ``other`` возвращают ``pd.NA``, потому что правило неоднозначно
    """
    vacancy_norm = _normalize_value(vacancy_education_level)
    if vacancy_norm is None or vacancy_norm == "не имеет значения":
        return True
    if vacancy_norm == "other":
        return pd.NA

    cv_norm = _normalize_value(cv_education)
    if cv_norm is None or cv_norm in {"образование не указано", "other"}:
        return pd.NA

    if vacancy_norm == "среднее профессиональное":
        if cv_norm in {"среднее специальное", "высшее"}:
            return True
        if cv_norm == "среднее":
            return False
        return pd.NA

    return pd.NA


def education_compatible_series(
    cv_education: pd.Series, vacancy_education_level: pd.Series
) -> pd.Series:
    """Векторизованная обертка над :func:`education_compatible`
    
    Пример
    -------
    >>> cv_education = pd.Series([
    ...     "высшее",
    ...     "среднее специальное",
    ...     "среднее",
    ...     "образование не указано",
    ...     "незаконченное высшее",
    ... ])
    >>> vacancy_education_level = pd.Series([
    ...     "не имеет значения",
    ...     "среднее профессиональное",
    ...     "среднее профессиональное",
    ...     "среднее профессиональное",
    ...     "среднее профессиональное",
    ... ])
    >>> education_compatible_series(cv_education, vacancy_education_level)
    0     True
    1     True
    2    False
    3     <NA>
    4     <NA>
    dtype: boolean
    """
    return pd.Series(
        [
            education_compatible(cv_value, vacancy_value)
            for cv_value, vacancy_value in zip(cv_education, vacancy_education_level)
        ],
        index=cv_education.index,
        dtype="boolean",
    )


def experience_compatible(cv_experience: object, vacancy_experience: object) -> object:
    """Проверяет, покрывает ли опыт из CV требование вакансии

    Параметры
    ----------
    cv_experience:
        Сырое или общее значение опыта со стороны CV
    vacancy_experience:
        Сырое или общее требование к опыту со стороны вакансии

    Возвращает
    -------
    bool | pd.NA
        Nullable результат совместимости

    Правила mapping
    -------------
    - Vacancy ``без опыта`` или общий код ``no_experience`` совместимы с
      любым CV, потому что опыт не требуется
    - Vacancy ``более 1 года``, ``более 3 лет``, ``более 5 лет`` или общий
      код ``has_experience`` совместимы только с CV ``есть опыт`` или общим
      кодом ``has_experience``
    - Пропуски и vacancy ``other`` возвращают ``pd.NA``
    """
    vacancy_norm = _normalize_value(vacancy_experience)
    if vacancy_norm is None or vacancy_norm == "other" or vacancy_norm == UNKNOWN:
        return pd.NA
    if vacancy_norm in {"без опыта", NO_EXPERIENCE}:
        return True

    cv_norm = _normalize_value(cv_experience)
    if cv_norm is None or cv_norm == UNKNOWN:
        return pd.NA

    vacancy_requires_experience = vacancy_norm in {
        "более 1 года",
        "более 3 лет",
        "более 5 лет",
        HAS_EXPERIENCE,
    }
    if vacancy_requires_experience:
        if cv_norm in {"есть опыт", HAS_EXPERIENCE}:
            return True
        if cv_norm in {"без опыта", NO_EXPERIENCE}:
            return False
        return pd.NA

    return pd.NA


def experience_compatible_series(
    cv_experience: pd.Series, vacancy_experience: pd.Series
) -> pd.Series:
    """Векторизованная обертка над :func:`experience_compatible`
    Пример
    -------
    >>> cv_experience = pd.Series([
    ...     "есть опыт",
    ...     "без опыта",
    ...     "без опыта",
    ...     "есть опыт",
    ...     None,
    ... ])
    >>> vacancy_experience = pd.Series([
    ...     "более 1 года",
    ...     "более 3 лет",
    ...     "без опыта",
    ...     "other",
    ...     "более 1 года",
    ... ])
    >>> experience_compatible_series(cv_experience, vacancy_experience)
    0     True
    1    False
    2     True
    3     <NA>
    4     <NA>
    dtype: boolean

    """
    return pd.Series(
        [
            experience_compatible(cv_value, vacancy_value)
            for cv_value, vacancy_value in zip(cv_experience, vacancy_experience)
        ],
        index=cv_experience.index,
        dtype="boolean",
    )


def normalize_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """Возвращает нормализованную копию таблицы CV

    Параметры
    ----------
    cv:
        Сырая таблица CV с колонками из ``cv.parquet``
    Возвращает
    -------
    pd.DataFrame
        Копия ``cv`` с сохранением всех исходных колонок и добавлением колонок:
        ``profession_norm``, ``group_profession_norm``,
        ``business_category_norm``, ``sfera_norm``, ``schedule_norm``,
        ``employment_type_norm``, ``education_norm`` и ``experience_common``

    Примечания
    -----
    Функция не удаляет строки, не мутирует входной DataFrame in-place и не
    перетирает исходные колонки
    Зарплатные derived-колонки не добавляются в normalized CV по умолчанию:
    исходный ``salary_bucketed`` сохраняется, а отдельную диагностику можно
    получить через :func:`clean_salary`
    ``business_category_norm`` использует только общую строковую нормализацию,
    потому что EDA не выявил отдельной словарной проблемы для этого поля
    """
    result = cv.copy()
    result["profession_norm"] = normalize_other_values(result["profession"])
    result["group_profession_norm"] = normalize_other_values(result["group_profession"])
    result["business_category_norm"] = normalize_text_category(
        result["business_category"]
    )
    result["sfera_norm"] = normalize_sfera(result["sfera"])
    result["schedule_norm"] = normalize_text_category(result["schedule"])
    result["employment_type_norm"] = normalize_text_category(result["employment_type"])
    result["education_norm"] = normalize_other_values(result["education"])
    result["experience_common"] = map_cv_experience_to_common(
        result["experience_bucket"]
    )

    return result


def normalize_vacancies(vacancies: pd.DataFrame) -> pd.DataFrame:
    """Возвращает нормализованную копию таблицы вакансий

    Параметры
    ----------
    vacancies:
        Сырая таблица вакансий с колонками из ``vacancies.parquet``

    Возвращает
    -------
    pd.DataFrame
        Копия ``vacancies`` с сохранением всех исходных колонок и добавлением
        колонок: ``profession_norm``, ``group_profession_norm``,
        ``business_category_norm``, ``sfera_norm``, ``schedule_norm``,
        ``employment_type_norm``, ``education_level_norm`` и
        ``experience_common``

    Примечания
    -----
    Функция не удаляет строки, не мутирует входной DataFrame in-place и не
    перетирает исходные колонки
    Зарплатные колонки не создаются, потому что в текущем датасете у вакансий
    нет поля зарплаты
    """
    result = vacancies.copy()
    result["profession_norm"] = normalize_other_values(result["profession"])
    result["group_profession_norm"] = normalize_other_values(result["group_profession"])
    result["business_category_norm"] = normalize_text_category(
        result["business_category"]
    )
    result["sfera_norm"] = normalize_sfera(result["sfera"])
    result["schedule_norm"] = normalize_text_category(result["schedule"])
    result["employment_type_norm"] = normalize_other_values(result["employment_type"])
    result["education_level_norm"] = normalize_other_values(result["education_level"])
    result["experience_common"] = map_vacancy_experience_to_common(result["experience"])
    return result


__all__ = [
    "EMPLOYMENT_TYPE_COMPATIBILITY",
    "HAS_EXPERIENCE",
    "KNOWN_VACANCY_EMPLOYMENT_TYPES",
    "KNOWN_VACANCY_SCHEDULES",
    "NO_EXPERIENCE",
    "SCHEDULE_COMPATIBILITY",
    "UNKNOWN",
    "clean_salary",
    "education_compatible",
    "education_compatible_series",
    "employment_type_compatible",
    "employment_type_compatible_series",
    "experience_compatible",
    "experience_compatible_series",
    "map_cv_experience_to_common",
    "map_vacancy_experience_to_common",
    "normalize_cv",
    "normalize_other_values",
    "normalize_sfera",
    "normalize_text_category",
    "normalize_vacancies",
    "schedule_compatible",
    "schedule_compatible_series",
]
