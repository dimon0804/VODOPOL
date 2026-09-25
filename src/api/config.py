"""Настройки слоя сервиса.

Здесь только про то, где лежит комплект запуска и на каком адресе слушать.
Никакой предметной логики: все числа приходят из `src.runtime.RunContext`.
"""

from __future__ import annotations

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


def discover_run_dir() -> Path | None:
    """Самый свежий комплект в outputs/ — когда переменная окружения не задана."""
    if not OUTPUTS_DIR.is_dir():
        return None
    runs = [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and (d / RUN_MARKER).is_file()]
    if not runs:
        return None
    return max(runs, key=lambda d: (d / RUN_MARKER).stat().st_mtime)


def host_port() -> tuple[str, int]:
    host = os.environ.get(ENV_HOST, "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.environ.get(ENV_PORT, "8000"))
    except ValueError:
        port = 8000
    return host, port
