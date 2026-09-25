"""Генератор каталога кандидатных зон съёмки (candidate_orders.geojson).

Организатор кандидатных зон не даёт — каталог целиком наш. В файл попадают ВСЕ
предложенные зоны, а не только выбранные стратегией: выбор живёт в
procurement_plan.csv и strategy_plans.json.

Как строится каталог:
  * растр приоритета (ожидаемый ущерб, неопределённость или их комбинация) режется
    регулярной сеткой ячеек в равновеликой проекции;
  * для каждой ячейки считается суммарный приоритет и число покрытых объектов;
  * ячейки без приоритета и без объектов выбрасываются;
  * оставшиеся получают стабильный candidate_id вида cand_r03_c07 и набор
    коммерческих атрибутов из конфигурации.

Размер ячейки по умолчанию 1,2 км: 1,44 км² с запасом проходят порог 1 км² до
округления, но порог всё равно проверяется явно — конфигурацию может поменять кто угодно.

Даты observation_at и available_at для новой съёмки у нас сценарные. Внутри кода они
НЕ выдумываются: берутся из конфигурации, а если неизвестны — остаются пустой строкой,
а статус раскрывается снаружи (run_metadata.json и подпись в интерфейсе).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
from shapely.geometry import Point, Polygon, shape

from src.contracts import (
    BASE_RATE_RUB_KM2_SCENARIO,
    BASE_RATE_SOURCE_SCENARIO,
    BASE_RATE_STATUS_SCENARIO,
    CRS_EQUAL_AREA,
    DATA_ROLE_CONTEXT,
    DATA_ROLE_EVENT,
    MIN_ORDER_AREA_KM2,
    PROPERTIES_CANDIDATE,
    SENSOR_OPTICAL,
    SENSOR_SAR,
    quantize_area,
)
from src.procurement.geometry import (
    GridCell,
    area_km2,
    area_km2_raw,
    covered_assets,
    grid_cells,
    to_geojson_feature,
    validate_polygon,
)

__all__ = [
    "PROPERTIES_EXTRA",
    "CatalogConfig",
    "build_catalog",
    "catalog_to_geojson",
    "selftest_catalog",
]

#: Дополнительные (разрешённые кейсом) свойства — нужны отбору стратегий B и C.
PROPERTIES_EXTRA: Final = (
    "priority_score",
    "covered_asset_count",
    "covered_asset_ids",
    "grid_row",
    "grid_col",
    "cell_km",
)

_EPS: Final = 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Конфигурация
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CatalogConfig:
    """Параметры генерации каталога. Всё, что может меняться, меняется здесь.

    Значения по умолчанию — наш базовый сценарий: радар (проходит облачность
    паводка), 3 м, новая гарантированно выкупаемая съёмка, L2, внутреннее
    использование, сценарная ставка БРЕ.
    """

    # Геометрия
    cell_km: float = 1.2
    min_area_km2: float = MIN_ORDER_AREA_KM2
    crs_equal_area: str = CRS_EQUAL_AREA
    id_prefix: str = "cand"

    # Коммерческие атрибуты основной (событийной) съёмки
    sensor_type: str = SENSOR_SAR
    resolution_m: float = 3.0
    acquisition_type: str = "new"
    data_role: str = DATA_ROLE_EVENT
    observation_at: str = ""
    available_at: str = ""
    processing_level: str = "L2"
    usage_type: str = "internal"
    guaranteed_purchase: bool = True

    # Ставка БРЕ: держим scenario, пока нет проверяемого источника действующей ставки
    base_rate_rub_km2: Decimal = BASE_RATE_RUB_KM2_SCENARIO
    base_rate_status: str = BASE_RATE_STATUS_SCENARIO
    base_rate_source: str = BASE_RATE_SOURCE_SCENARIO

    # Контекстные архивные зоны
    context_count: int = 2
    context_cell_km: float = 2.4
    context_id_prefix: str = "ctx"
    context_sensor_type: str = SENSOR_OPTICAL
    context_resolution_m: float = 10.0
    context_acquisition_type: str = "archive"
    context_processing_level: str = "L1"
    context_usage_type: str = "internal"
    context_observation_at: str = ""
    context_available_at: str = ""

    # Отбор ячеек
    min_priority: float = 0.0  # строго больше — иначе ячейка без объектов отбрасывается
    max_candidates: int | None = None  # ограничение размера каталога, None — без лимита


# ─────────────────────────────────────────────────────────────────────────────
# Служебное: растр и точки
# ─────────────────────────────────────────────────────────────────────────────


def _apply(transform: Any, col: float, row: float) -> tuple[float, float]:
    """Применение affine-трансформации к точке.

    В affine 3 оператор `*` для этого помечен как устаревший, поэтому сначала
    пробуем `@`, а `*` оставляем для старых версий пакета.
    """
    try:
        x, y = transform @ (col, row)
    except TypeError:
        x, y = transform * (col, row)
    return float(x), float(y)


def _raster_bounds(transform: Any, shape_hw: tuple[int, int]) -> tuple[float, float, float, float]:
    """Охват растра в WGS84 по affine-трансформации и размеру (rows, cols)."""
    rows, cols = int(shape_hw[0]), int(shape_hw[1])
    corners = [(0, 0), (cols, 0), (0, rows), (cols, rows)]
    xs: list[float] = []
    ys: list[float] = []
    for col, row in corners:
        x, y = _apply(transform, col, row)
        xs.append(x)
        ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys))


def _pixel_window(
    transform: Any, shape_hw: tuple[int, int], bounds: tuple[float, float, float, float]
) -> tuple[int, int, int, int] | None:
    """Окно растра (row0, row1, col0, col1) полуинтервалами по ЦЕНТРАМ пикселей.

    Пиксель относится к ячейке, если его центр попадает в её охват. Так соседние
    ячейки не делят один пиксель и суммарный приоритет не задваивается.
    """
    rows, cols = int(shape_hw[0]), int(shape_hw[1])
    inv = ~transform
    min_x, min_y, max_x, max_y = bounds
    pix = [
        _apply(inv, min_x, min_y),
        _apply(inv, min_x, max_y),
        _apply(inv, max_x, min_y),
        _apply(inv, max_x, max_y),
    ]
    col_vals = [p[0] for p in pix]
    row_vals = [p[1] for p in pix]

    col0 = max(0, int(math.ceil(min(col_vals) - 0.5 - _EPS)))
    col1 = min(cols, int(math.floor(max(col_vals) - 0.5 + _EPS)) + 1)
    row0 = max(0, int(math.ceil(min(row_vals) - 0.5 - _EPS)))
    row1 = min(rows, int(math.floor(max(row_vals) - 0.5 + _EPS)) + 1)

    if col0 >= col1 or row0 >= row1:
        return None
    return (row0, row1, col0, col1)


def _normalize_assets(assets: Any) -> dict[str, Point]:
    """Портфель объектов в вид {asset_id: Point(lon, lat)}.

    Принимаем словарь, пары (asset_id, Point/координаты) и строки assets.csv
    (словари с longitude/latitude).
    """
    if assets is None:
        return {}
    out: dict[str, Point] = {}

    if isinstance(assets, Mapping):
        items: Iterable[Any] = assets.items()
    else:
        items = assets

    for item in items:
        if isinstance(item, Mapping):
            asset_id = str(item["asset_id"])
            value: Any = item
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            asset_id, value = str(item[0]), item[1]
        else:
            raise TypeError(f"не удалось разобрать объект портфеля: {item!r}")

        if isinstance(value, Point):
            point = value
        elif isinstance(value, Mapping):
            point = Point(float(value["longitude"]), float(value["latitude"]))
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            point = Point(float(value[0]), float(value[1]))
        else:
            raise TypeError(f"{asset_id}: не удалось получить координаты из {value!r}")
        out[asset_id] = point
    return out


@dataclass(frozen=True)
class _ScoredCell:
    """Ячейка сетки со счётом: приоритет внутри и покрытые объекты."""

    cell: GridCell
    priority: float
    assets: tuple[str, ...]


def _score_cells(
    cells: Sequence[GridCell],
    priority: np.ndarray,
    transform: Any,
    points: Mapping[str, Point],
) -> list[_ScoredCell]:
    """Суммарный приоритет и покрытые объекты для каждой ячейки сетки."""
    arr = np.asarray(priority, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"растр приоритета должен быть двумерным, получено {arr.ndim}D")
    shape_hw = (arr.shape[0], arr.shape[1])

    coverage = covered_assets(
        [(f"{c.row}:{c.col}", c.polygon) for c in cells],
        points,
    )

    scored: list[_ScoredCell] = []
    for cell in cells:
        window = _pixel_window(transform, shape_hw, cell.polygon.bounds)
        if window is None:
            total = 0.0
        else:
            row0, row1, col0, col1 = window
            block = arr[row0:row1, col0:col1]
            total = float(np.nansum(np.clip(block, 0.0, None))) if block.size else 0.0
        hit = tuple(sorted(coverage.get(f"{cell.row}:{cell.col}", set())))
        scored.append(_ScoredCell(cell=cell, priority=total, assets=hit))
    return scored


def _keep(scored: _ScoredCell, config: CatalogConfig) -> bool:
    """Ячейка остаётся, если есть приоритет ИЛИ есть хотя бы один объект."""
    return scored.priority > config.min_priority or bool(scored.assets)


# ─────────────────────────────────────────────────────────────────────────────
# Сборка фич
# ─────────────────────────────────────────────────────────────────────────────


def _feature(
    candidate_id: str,
    polygon: Polygon,
    scored: _ScoredCell,
    config: CatalogConfig,
    *,
    sensor_type: str,
    resolution_m: float,
    acquisition_type: str,
    data_role: str,
    observation_at: str,
    available_at: str,
    processing_level: str,
    usage_type: str,
    guaranteed_purchase: bool,
    cell_km: float,
) -> dict[str, Any]:
    """Одна Feature каталога. Порядок свойств — как в PROPERTIES_CANDIDATE."""
    area = area_km2(polygon, config.crs_equal_area)
    properties: dict[str, Any] = {
        "candidate_id": candidate_id,
        "sensor_type": sensor_type,
        "resolution_m": float(resolution_m),
        "acquisition_type": acquisition_type,
        "data_role": data_role,
        "observation_at": observation_at,
        "available_at": available_at,
        "processing_level": processing_level,
        "usage_type": usage_type,
        "guaranteed_purchase": bool(guaranteed_purchase),
        "area_km2": float(area),
        "base_rate_rub_km2": float(config.base_rate_rub_km2),
        "base_rate_status": config.base_rate_status,
        "base_rate_source": config.base_rate_source,
        # Дальше — дополнительные поля, кейс их разрешает.
        "priority_score": float(scored.priority),
        "covered_asset_count": len(scored.assets),
        "covered_asset_ids": list(scored.assets),
        "grid_row": int(scored.cell.row),
        "grid_col": int(scored.cell.col),
        "cell_km": float(cell_km),
    }
    return to_geojson_feature(polygon, properties)


def build_catalog(
    priority: np.ndarray,
    transform: Any,
    assets: Any = None,
    config: CatalogConfig | None = None,
) -> list[dict[str, Any]]:
    """Каталог кандидатных зон по растру приоритета.

    priority  — двумерный numpy-массив (ожидаемый ущерб на пиксель, неопределённость
                или их комбинация); NaN трактуется как ноль, отрицательные значения
                обрезаются нулём.
    transform — affine-трансформация растра (rasterio.Affine), WGS84.
    assets    — портфель точечных объектов: {asset_id: Point}, пары или строки assets.csv.
    config    — CatalogConfig; None — базовый сценарий.

    Возвращает список Feature GeoJSON, отсортированный: сначала событийные зоны по
    (row, col), затем контекстные архивные. Порог 1 км² проверяется явно, ячейки
    меньше порога в каталог не попадают.
    """
    cfg = config or CatalogConfig()
    arr = np.asarray(priority, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"растр приоритета должен быть двумерным, получено {arr.ndim}D")
    if arr.size == 0:
        raise ValueError("растр приоритета пуст")

    points = _normalize_assets(assets)
    bounds = _raster_bounds(transform, (arr.shape[0], arr.shape[1]))

    # ── Основная сетка: событийные зоны ─────────────────────────────────────
    cells = grid_cells(bounds, cfg.cell_km, cfg.crs_equal_area)
    scored = [s for s in _score_cells(cells, arr, transform, points) if _keep(s, cfg)]

    if cfg.max_candidates is not None and len(scored) > cfg.max_candidates:
        # Режем по значимости, но порядок в каталоге оставляем геометрическим.
        scored.sort(key=lambda s: (-len(s.assets), -s.priority, s.cell.row, s.cell.col))
        scored = scored[: cfg.max_candidates]
    scored.sort(key=lambda s: (s.cell.row, s.cell.col))

    features: list[dict[str, Any]] = []
    for item in scored:
        polygon = item.cell.polygon
        if area_km2_raw(polygon, cfg.crs_equal_area) < cfg.min_area_km2:
            continue  # порог 1 км² проверяется ДО округления
        if validate_polygon(polygon, cfg.min_area_km2, cfg.crs_equal_area):
            continue
        candidate_id = f"{cfg.id_prefix}_r{item.cell.row:02d}_c{item.cell.col:02d}"
        features.append(
            _feature(
                candidate_id,
                polygon,
                item,
                cfg,
                sensor_type=cfg.sensor_type,
                resolution_m=cfg.resolution_m,
                acquisition_type=cfg.acquisition_type,
                data_role=cfg.data_role,
                observation_at=cfg.observation_at,
                available_at=cfg.available_at,
                processing_level=cfg.processing_level,
                usage_type=cfg.usage_type,
                guaranteed_purchase=cfg.guaranteed_purchase,
                cell_km=cfg.cell_km,
            )
        )

    features.extend(_context_features(arr, transform, points, bounds, cfg))
    return features


def _context_features(
    arr: np.ndarray,
    transform: Any,
    points: Mapping[str, Point],
    bounds: tuple[float, float, float, float],
    cfg: CatalogConfig,
) -> list[dict[str, Any]]:
    """Небольшой набор контекстных АРХИВНЫХ зон.

    Архивный снимок дешевле (коэффициент актуальности Т = 0,6) и не выкупается
    гарантированно, поэтому он нужен для честного сравнения стратегий. Но: архив
    НЕ является наблюдением текущей воды — по нему нельзя утверждать, что вода
    есть или её нет сейчас. Поэтому у таких зон data_role = context, и в охват
    события они не засчитываются: их роль — фон, границы русла, до-событийное
    состояние местности.
    """
    if cfg.context_count <= 0:
        return []

    cells = grid_cells(bounds, cfg.context_cell_km, cfg.crs_equal_area)
    scored = [s for s in _score_cells(cells, arr, transform, points) if _keep(s, cfg)]
    scored.sort(key=lambda s: (-s.priority, -len(s.assets), s.cell.row, s.cell.col))

    out: list[dict[str, Any]] = []
    for item in scored:
        if len(out) >= cfg.context_count:
            break
        polygon = item.cell.polygon
        if area_km2_raw(polygon, cfg.crs_equal_area) < cfg.min_area_km2:
            continue
        if validate_polygon(polygon, cfg.min_area_km2, cfg.crs_equal_area):
            continue
        candidate_id = f"{cfg.context_id_prefix}_r{item.cell.row:02d}_c{item.cell.col:02d}"
        out.append(
            _feature(
                candidate_id,
                polygon,
                item,
                cfg,
                sensor_type=cfg.context_sensor_type,
                resolution_m=cfg.context_resolution_m,
                acquisition_type=cfg.context_acquisition_type,
                data_role=DATA_ROLE_CONTEXT,
                observation_at=cfg.context_observation_at,
                available_at=cfg.context_available_at,
                processing_level=cfg.context_processing_level,
                usage_type=cfg.context_usage_type,
                guaranteed_purchase=False,
                cell_km=cfg.context_cell_km,
            )
        )
    return out


def catalog_to_geojson(features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """FeatureCollection WGS84, готовый к записи в candidate_orders.geojson."""
    return {
        "type": "FeatureCollection",
        # RFC 7946: CRS по умолчанию — WGS84 / CRS84 (долгота, широта). Поле crs
        # в стандарте отсутствует, поэтому фиксируем систему координат явным
        # комментарием в метаданных запуска, а не самодельным ключом.
        "features": [dict(f) for f in features],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Самопроверка каталога
# ─────────────────────────────────────────────────────────────────────────────


def selftest_catalog(
    features: Sequence[Mapping[str, Any]],
    min_area_km2: float = MIN_ORDER_AREA_KM2,
    crs_equal_area: str = CRS_EQUAL_AREA,
) -> list[str]:
    """Проверка каталога теми же правилами, что и у валидатора пакета.

    Пустой список — каталог пригоден к сдаче. Проверяется:
      * непустой каталог;
      * набор обязательных свойств PROPERTIES_CANDIDATE у каждой фичи;
      * валидность полигона, отсутствие вырезов и самопересечений;
      * площадь не менее 1 км² ДО округления;
      * записанная area_km2 совпадает с пересчитанной;
      * уникальность candidate_id;
      * отсутствие NaN/Inf в числовых свойствах.
    """
    problems: list[str] = []
    if not features:
        return ["каталог пуст: ни одной кандидатной зоны"]

    seen: dict[str, int] = {}
    for index, feature in enumerate(features):
        props = dict(feature.get("properties") or {})
        candidate_id = str(props.get("candidate_id") or f"<без id, позиция {index}>")

        missing = [name for name in PROPERTIES_CANDIDATE if name not in props]
        if missing:
            problems.append(f"{candidate_id}: нет обязательных свойств: {', '.join(missing)}")

        if candidate_id in seen:
            problems.append(
                f"{candidate_id}: candidate_id не уникален "
                f"(позиции {seen[candidate_id]} и {index})"
            )
        else:
            seen[candidate_id] = index

        for name, value in props.items():
            if isinstance(value, float) and not math.isfinite(value):
                problems.append(f"{candidate_id}: свойство {name} содержит NaN или Inf")

        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping) or geometry.get("type") != "Polygon":
            problems.append(f"{candidate_id}: геометрия не Polygon")
            continue
        try:
            polygon = shape(dict(geometry))
        except Exception as exc:  # noqa: BLE001 — сообщение важнее типа
            problems.append(f"{candidate_id}: геометрия не читается: {exc}")
            continue

        for violation in validate_polygon(polygon, min_area_km2, crs_equal_area):
            problems.append(f"{candidate_id}: {violation}")

        if not isinstance(polygon, Polygon):
            continue
        written = props.get("area_km2")
        if written is None:
            continue
        try:
            recomputed = area_km2(polygon, crs_equal_area)
            if quantize_area(written) != recomputed:
                problems.append(
                    f"{candidate_id}: записанная площадь {written} км² не совпадает "
                    f"с вычисленной {recomputed} км²"
                )
        except (ValueError, TypeError, ArithmeticError) as exc:
            problems.append(f"{candidate_id}: площадь не пересчитана: {exc}")

    return problems
