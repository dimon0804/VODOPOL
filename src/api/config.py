"""Настройки слоя сервиса.

Здесь только про то, где лежит комплект запуска и на каком адресе слушать.
Никакой предметной логики: все числа приходят из `src.runtime.RunContext`.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

#: Версия панели оператора (сервисная, к версии модели отношения не имеет).
API_VERSION = "1.0.0"
API_TITLE = "Водополь — панель оператора"

#: Переменная окружения с путём к готовому комплекту запуска.
ENV_RUN = "FLOODVALUE_RUN"
ENV_HOST = "FLOODVALUE_HOST"
ENV_PORT = "FLOODVALUE_PORT"

#: Корень репозитория: src/api/config.py -> src/api -> src -> корень.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
WEB_DIR = PROJECT_ROOT / "web"

#: Файл-признак готового комплекта.
RUN_MARKER = "run_metadata.json"

#: Слои карты, которые умеет отдавать RunContext.raster_png.
RASTER_KINDS = ("s1", "probability", "mask", "uncertainty")

#: Сколько отрисованных PNG держать в памяти, чтобы не рендерить на каждый скролл.
RASTER_CACHE_SIZE = 16


def env_run_dir() -> Path | None:
    """Путь из FLOODVALUE_RUN, если переменная задана и не пуста."""
    raw = os.environ.get(ENV_RUN, "").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def list_run_dirs() -> list[Path]:
    """Все готовые комплекты в outputs/, свежие первыми.

    Признак готовности — наличие run_metadata.json: каталог, в котором расчёт
    оборвался на середине, в список не попадёт и панель на нём не сломается.
    """
    if not OUTPUTS_DIR.is_dir():
        return []
    runs = [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and (d / RUN_MARKER).is_file()]
    return sorted(runs, key=lambda d: (d / RUN_MARKER).stat().st_mtime, reverse=True)


def run_dir_by_id(run_id: str) -> Path | None:
    """Каталог комплекта по его идентификатору, с проверкой, что он наш.

    Идентификатор приходит из запроса, поэтому проверяется дважды: он обязан быть
    одним именем без разделителей пути, и он обязан найтись среди каталогов,
    которые перечисляет list_run_dirs. Подняться выше outputs/ таким путём нельзя.
    """
    name = (run_id or "").strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        return None
    for path in list_run_dirs():
        if path.name == name:
            return path
    return None


def event_parts() -> dict[str, str]:
    """Событие -> часть протокола, в которой оно лежит.

    Нужно панели: оператор должен видеть, показываем мы чип, на котором модель
    училась, или событие, которого она не видела. Без этой подписи выбор комплекта
    превращается в лотерею из непрозрачных идентификаторов.
    """
    titles = {
        "train": "обучение",
        "validation": "настройка",
        "test": "независимая проверка",
        "holdout": "отложенное событие",
    }
    mapping: dict[str, str] = {}
    splits_dir = PROJECT_ROOT / "splits"
    if not splits_dir.is_dir():
        return mapping
    for part, title in titles.items():
        path = splits_dir / f"{part}.csv"
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    event = (row.get("event") or row.get("event_id") or "").strip()
                    if event:
                        mapping.setdefault(event, title)
        except OSError:
            continue
    return mapping


def discover_run_dir() -> Path | None:
    """Самый свежий комплект в outputs/ — когда переменная окружения не задана."""
    runs = list_run_dirs()
    return runs[0] if runs else None


def host_port() -> tuple[str, int]:
    host = os.environ.get(ENV_HOST, "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.environ.get(ENV_PORT, "8000"))
    except ValueError:
        port = 8000
    return host, port

#: Файлы комплекта, которые панель отдаёт по одному — для кнопок «скачать CSV» на
#: страницах. Имя → тип содержимого. Всё, чего здесь нет, ручка не отдаёт.
RUN_FILES = {
    "asset_loss.csv": "text/csv; charset=utf-8",
    "assets.csv": "text/csv; charset=utf-8",
    "procurement_plan.csv": "text/csv; charset=utf-8",
    "strategy_comparison.csv": "text/csv; charset=utf-8",
    "sensitivity.csv": "text/csv; charset=utf-8",
    "sensitivity_ranks.csv": "text/csv; charset=utf-8",
    "run_metadata.json": "application/json",
    "impact_summary.json": "application/json",
    "source_manifest.json": "application/json",
    "strategy_plans.json": "application/json",
    "assets.geojson": "application/geo+json",
    "candidate_orders.geojson": "application/geo+json",
}

#: Отчёты модели для страницы «Методы и качество». Имя в URL → путь от корня
#: репозитория. Числа там посчитаны при обучении и проверке, панель их не трогает.
REPORT_FILES = {
    "metrics_compare.json": "reports/metrics_compare.json",
    "metrics_compare_holdout.json": "reports/metrics_compare_holdout.json",
    "metrics_main_validation.json": "reports/metrics_main_validation.json",
    "experiments.json": "reports/experiments.json",
    "split_manifest.json": "splits/split_manifest.json",
}
