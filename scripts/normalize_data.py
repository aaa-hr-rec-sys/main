"""Создает версионированный слой нормализованных данных

Пример
-------
python main/scripts/normalize_data.py --input-dir data/aaa-out --output-root data/processed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa

MAIN_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = MAIN_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.data import get_data_paths  # noqa: E402
from utils.normalization import (  # noqa: E402
    EMPLOYMENT_TYPE_COMPATIBILITY,
    KNOWN_VACANCY_EMPLOYMENT_TYPES,
    KNOWN_VACANCY_SCHEDULES,
    SCHEDULE_COMPATIBILITY,
    clean_salary,
    normalize_cv,
    normalize_vacancies,
)

MAPPINGS_VERSION = "v1"


def require_pyarrow(min_major: int = 24) -> None:
    """Заранее падает, если pyarrow слишком старый для текущих parquet файлов"""
    major = int(pa.__version__.split(".")[0])
    if major < min_major:
        raise RuntimeError(
            f"pyarrow=={pa.__version__} может падать на этих parquet файлах "
            f'Используйте pyarrow>={min_major}, например через uv run --isolated --with pandas --with "pyarrow>={min_major}"'
        )


def parse_args() -> argparse.Namespace:
    """Парсит CLI-аргументы для нормализации датасета"""
    parser = argparse.ArgumentParser(
        description="Нормализует CV и вакансии в версионированный датасет"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/aaa-out"),
        help="Директория raw датасета",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Корень для processed версий",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Имя папки версии, по умолчанию timestamp",
    )
    parser.add_argument(
        "--salary-outlier-threshold",
        type=float,
        default=500_000,
        help="Порог для флага salary_outlier",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписать существующую папку версии",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Считает SHA-256 checksum файла"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_remove_version_dir(version_dir: Path, output_root: Path) -> None:
    """Удаляет существующую версию после проверки, что она лежит внутри output_root"""
    resolved_version = version_dir.resolve()
    resolved_root = output_root.resolve()
    if (
        resolved_version == resolved_root
        or resolved_root not in resolved_version.parents
    ):
        raise ValueError(
            f"Отказываюсь удалять путь вне output root: {resolved_version}"
        )
    shutil.rmtree(resolved_version)


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    """Возвращает JSON-friendly value counts для Series"""
    counts = series.astype("string").fillna("<NA>").value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def unknown_cv_schedules(cv_norm: pd.DataFrame) -> list[str]:
    """Возвращает CV schedule values без надежного compatibility mapping"""
    values = set(cv_norm["schedule_norm"].dropna().astype(str))
    unknown = [
        value
        for value in values
        if value not in SCHEDULE_COMPATIBILITY
        or SCHEDULE_COMPATIBILITY.get(value) is None
    ]
    return sorted(unknown)


def unknown_vacancy_schedules(vacancies_norm: pd.DataFrame) -> list[str]:
    """Возвращает vacancy schedule values вне известного compatibility-словаря"""
    values = set(vacancies_norm["schedule_norm"].dropna().astype(str))
    return sorted(values - KNOWN_VACANCY_SCHEDULES)


def unknown_cv_employment(cv_norm: pd.DataFrame) -> list[str]:
    """Возвращает CV employment values без надежного compatibility mapping"""
    values = set(cv_norm["employment_type_norm"].dropna().astype(str))
    return sorted(values - set(EMPLOYMENT_TYPE_COMPATIBILITY))


def unknown_vacancy_employment(vacancies_norm: pd.DataFrame) -> list[str]:
    """Возвращает vacancy employment values вне известного compatibility-словаря"""
    values = set(vacancies_norm["employment_type_norm"].dropna().astype(str))
    return sorted(values - KNOWN_VACANCY_EMPLOYMENT_TYPES)


def build_normalization_report(
    cv_raw: pd.DataFrame,
    vacancies_raw: pd.DataFrame,
    cv_norm: pd.DataFrame,
    vacancies_norm: pd.DataFrame,
    salary_flags: pd.DataFrame,
    salary_outlier_threshold: float,
) -> dict[str, Any]:
    """Строит JSON-сериализуемый отчет по нормализованным таблицам"""
    return {
        "row_counts": {
            "cv_raw": int(len(cv_raw)),
            "cv_normalized": int(len(cv_norm)),
            "vacancies_raw": int(len(vacancies_raw)),
            "vacancies_normalized": int(len(vacancies_norm)),
        },
        "salary": {
            "outlier_threshold": salary_outlier_threshold,
            "missing_count": int(salary_flags["salary_missing"].sum()),
            "negative_count": int(salary_flags["salary_negative"].sum()),
            "outlier_count": int(salary_flags["salary_outlier"].sum()),
        },
        "value_counts": {
            "cv_schedule_norm": value_counts_dict(cv_norm["schedule_norm"]),
            "vacancy_schedule_norm": value_counts_dict(vacancies_norm["schedule_norm"]),
            "cv_employment_type_norm": value_counts_dict(
                cv_norm["employment_type_norm"]
            ),
            "vacancy_employment_type_norm": value_counts_dict(
                vacancies_norm["employment_type_norm"]
            ),
            "cv_experience_common": value_counts_dict(cv_norm["experience_common"]),
            "vacancy_experience_common": value_counts_dict(
                vacancies_norm["experience_common"]
            ),
            "cv_sfera_norm": value_counts_dict(cv_norm["sfera_norm"]),
            "vacancy_sfera_norm": value_counts_dict(vacancies_norm["sfera_norm"]),
        },
        "unknown_or_unmapped_values": {
            "cv_schedule": unknown_cv_schedules(cv_norm),
            "vacancy_schedule": unknown_vacancy_schedules(vacancies_norm),
            "cv_employment_type": unknown_cv_employment(cv_norm),
            "vacancy_employment_type": unknown_vacancy_employment(vacancies_norm),
        },
        "normalization_columns": {
            "cv_added": sorted(set(cv_norm.columns) - set(cv_raw.columns)),
            "vacancies_added": sorted(
                set(vacancies_norm.columns) - set(vacancies_raw.columns)
            ),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Записывает словарь как UTF-8 JSON"""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_manifest(
    version: str,
    input_dir: Path,
    output_dir: Path,
    source_paths: dict[str, Path],
    output_paths: dict[str, Path],
    normalization_params: dict[str, Any],
    row_counts: dict[str, int],
) -> dict[str, Any]:
    """Строит manifest версии с метаданными файлов и версиями окружения"""
    existing_source_paths = {
        name: path for name, path in source_paths.items() if path.exists()
    }
    existing_output_paths = {
        name: path for name, path in output_paths.items() if path.exists()
    }
    return {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_files": {
            name: str(path) for name, path in existing_source_paths.items()
        },
        "source_file_sizes": {
            name: path.stat().st_size for name, path in existing_source_paths.items()
        },
        "source_checksums": {
            name: sha256_file(path) for name, path in existing_source_paths.items()
        },
        "output_files": {
            name: str(path) for name, path in existing_output_paths.items()
        },
        "output_checksums": {
            name: sha256_file(path) for name, path in existing_output_paths.items()
        },
        "row_counts": row_counts,
        "normalization_params": normalization_params,
        "normalization_mappings_version": MAPPINGS_VERSION,
        "python_version": sys.version.split()[0],
        "pandas_version": pd.__version__,
        "pyarrow_version": pa.__version__,
    }


def main() -> None:
    """Запускает нормализацию датасета и создает версионированную output-директорию"""
    require_pyarrow()
    args = parse_args()
    version = args.version or datetime.now().strftime("v%Y-%m-%d_%H%M%S")
    input_dir = args.input_dir
    output_root = args.output_root
    version_dir = output_root / version

    if version_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Версия уже существует: {version_dir} Используйте --overwrite для замены"
            )
        safe_remove_version_dir(version_dir, output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    version_dir.mkdir(parents=True, exist_ok=False)

    paths = get_data_paths(input_dir)
    cv_raw = pd.read_parquet(paths["cv"])
    vacancies_raw = pd.read_parquet(paths["vacancies"])

    cv_norm = normalize_cv(cv_raw)
    vacancies_norm = normalize_vacancies(vacancies_raw)
    salary_flags = clean_salary(
        cv_raw["salary_bucketed"],
        outlier_threshold=args.salary_outlier_threshold,
    )

    cv_output = version_dir / "cv_normalized.parquet"
    vacancies_output = version_dir / "vacancies_normalized.parquet"
    report_output = version_dir / "normalization_report.json"
    manifest_output = version_dir / "manifest.json"

    cv_norm.to_parquet(cv_output, index=False)
    vacancies_norm.to_parquet(vacancies_output, index=False)

    report = build_normalization_report(
        cv_raw=cv_raw,
        vacancies_raw=vacancies_raw,
        cv_norm=cv_norm,
        vacancies_norm=vacancies_norm,
        salary_flags=salary_flags,
        salary_outlier_threshold=args.salary_outlier_threshold,
    )
    write_json(report_output, report)

    output_paths = {
        "cv_normalized": cv_output,
        "vacancies_normalized": vacancies_output,
        "normalization_report": report_output,
    }
    manifest = build_manifest(
        version=version,
        input_dir=input_dir,
        output_dir=version_dir,
        source_paths=paths,
        output_paths=output_paths,
        normalization_params={
            "salary_outlier_threshold": args.salary_outlier_threshold
        },
        row_counts=report["row_counts"],
    )
    write_json(manifest_output, manifest)

    latest_path = output_root / "latest.txt"
    latest_path.write_text(version + "\n", encoding="utf-8")

    print(f"Создана версия нормализованного датасета: {version_dir}")
    print(f"Обновлен указатель latest: {latest_path}")


if __name__ == "__main__":
    main()
