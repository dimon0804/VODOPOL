"""Геометрия зон заказа ДЗЗ: площади, валидация, покрытие объектов, сетка ячеек.

Здесь собраны все геометрические операции, от которых зависят баллы за критерий
«зоны заказа». Правила кейса, зашитые в модуль:

  * зона — ОДИН валидный Polygon, без внутренних вырезов и без самопересечений;
  * площадь не менее 1 км² ДО округления (0,9996 км² не становится допустимой
    после округления до 1,000);
  * оплачивается площадь по контуру: ни bbox, ни выпуклая оболочка, ни чип целиком,
    и вырезать воду или пустые участки из оплачиваемой геометрии нельзя;
  * площадь считается переводом вершин в равновеликую проекцию EPSG:6933,
    м² делятся на 1 000 000, запись с точностью 0,001 км², ROUND_HALF_UP;
  * GeoJSON — WGS84, порядок координат [долгота, широта] (RFC 7946 / OGC CRS84),
    поэтому все трансформеры pyproj создаются с always_xy=True;
  * покрытие точечных объектов проверяется предикатом covers — граница включительно.

Внешних зависимостей минимум: shapely 2, pyproj, numpy. Ни geopandas, ни folium.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from typing import Any, Final, Iterable, Mapping, Sequence

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.validation import explain_validity

from src.contracts import (
    CRS_EQUAL_AREA,
    CRS_GEO,
    MIN_ORDER_AREA_KM2,
    quantize_area,
)

__all__ = [
    "GridCell",
    "area_km2",
    "area_km2_raw",
    "bbox_of",
    "covered_assets",
    "grid_cells",
    "to_geojson_feature",
    "unique_covered",
    "unique_covered_by",
    "validate_polygon",
]

#: EPSG:6933 (Lambert cylindrical equal area) определён от 86° ю.ш. до 86° с.ш.
MAX_ABS_LAT_EQUAL_AREA: Final = 86.0

#: Защита от случайного запроса гигантской сетки (numpy-массив углов растёт как n²).
MAX_GRID_CELLS: Final = 250_000

_EPS: Final = 1e-9

# Формулировки нарушений. Тексты используются валидатором пакета и тестами,
# менять их без нужды не стоит.
V_NOT_POLYGON: Final = "геометрия не является Polygon"
V_EMPTY: Final = "геометрия пустая"
V_INVALID: Final = "геометрия невалидна"
V_SELF_INTERSECTION: Final = "самопересечение контура"
V_HAS_HOLES: Final = "полигон содержит внутренние вырезы (holes)"
V_TOO_SMALL: Final = "площадь меньше минимальной"


# ─────────────────────────────────────────────────────────────────────────────
# Проекции
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=8)
def _transformer(src: str, dst: str) -> Transformer:
    """Кэшированный трансформер. always_xy=True — порядок всегда (x=lon, y=lat)."""
    return Transformer.from_crs(src, dst, always_xy=True)


def _check_lat_range(lats: Sequence[float]) -> None:
    for lat in lats:
        if not math.isfinite(lat):
            raise ValueError("в геометрии есть NaN или Inf")
        if abs(lat) > MAX_ABS_LAT_EQUAL_AREA:
            raise ValueError(
                f"широта {lat} вне области применимости {CRS_EQUAL_AREA} "
                f"(±{MAX_ABS_LAT_EQUAL_AREA}°)"
            )


def _to_equal_area(polygon: Polygon, crs_equal_area: str = CRS_EQUAL_AREA) -> Polygon:
    """Полигон WGS84 → равновеликая проекция. Вырезы переносятся как есть."""
    tr = _transformer(CRS_GEO, crs_equal_area)

    def _ring(coords: Iterable[tuple[float, ...]]) -> list[tuple[float, float]]:
        lon = [float(c[0]) for c in coords]
        lat = [float(c[1]) for c in coords]
        _check_lat_range(lat)
        x, y = tr.transform(lon, lat)
        return list(zip(x, y))

    shell = _ring(polygon.exterior.coords)
    holes = [_ring(ring.coords) for ring in polygon.interiors]
    return Polygon(shell, holes)


# ─────────────────────────────────────────────────────────────────────────────
# Площадь
# ─────────────────────────────────────────────────────────────────────────────


def area_km2_raw(polygon: Polygon, crs_equal_area: str = CRS_EQUAL_AREA) -> float:
    """Неокруглённая площадь полигона в км².

    Именно это значение сравнивается с порогом 1 км²: округление до 0,001 км²
    выполняется только при записи в файл и порог «спасать» не должно.
    """
    if not isinstance(polygon, Polygon):
        raise TypeError(f"ожидался Polygon, получен {type(polygon).__name__}")
    if polygon.is_empty:
        return 0.0
    return _to_equal_area(polygon, crs_equal_area).area / 1_000_000.0


def area_km2(polygon: Polygon, crs_equal_area: str = CRS_EQUAL_AREA) -> Decimal:
    """Площадь полигона в км², округлённая до 0,001 по ROUND_HALF_UP.

    Это то число, которое попадает в поле area_km2 GeoJSON и в расчёт цены.
    """
    return quantize_area(area_km2_raw(polygon, crs_equal_area))


# ─────────────────────────────────────────────────────────────────────────────
# Валидация зоны заказа
# ─────────────────────────────────────────────────────────────────────────────


def validate_polygon(
    polygon: Any,
    min_area_km2: float = MIN_ORDER_AREA_KM2,
    crs_equal_area: str = CRS_EQUAL_AREA,
) -> list[str]:
    """Нарушения требований кейса для одной зоны заказа.

    Пустой список — зона пригодна к заказу. Проверяется:
      * тип геометрии (ровно Polygon, не Multi-, не коллекция);
      * непустота;
      * валидность и отдельно самопересечение контура;
      * отсутствие внутренних вырезов (holes);
      * площадь не менее min_area_km2 ДО округления.
    """
    violations: list[str] = []

    if not isinstance(polygon, Polygon):
        return [f"{V_NOT_POLYGON}: {type(polygon).__name__}"]
    if polygon.is_empty:
        return [V_EMPTY]

    if not polygon.is_valid:
        reason = explain_validity(polygon)
        if "self-intersection" in reason.lower() or "self intersection" in reason.lower():
            violations.append(f"{V_SELF_INTERSECTION}: {reason}")
        else:
            violations.append(f"{V_INVALID}: {reason}")

    holes = len(polygon.interiors)
    if holes:
        violations.append(f"{V_HAS_HOLES}: {holes}")

    try:
        raw = area_km2_raw(polygon, crs_equal_area)
    except ValueError as exc:
        violations.append(f"площадь не вычислена: {exc}")
        return violations

    # Сравнение ведётся с сырой площадью, а не с quantize_area(raw):
    # 0,9996 км² округляется до 1,000, но зоной заказа не становится.
    if raw < min_area_km2:
        violations.append(
            f"{V_TOO_SMALL} {min_area_km2} км²: {raw:.6f} км² (проверка до округления)"
        )

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# Покрытие точечных объектов
# ─────────────────────────────────────────────────────────────────────────────


def _as_pairs(items: Any, what: str) -> list[tuple[str, Any]]:
    """Нормализация входа: Mapping, последовательность пар или объекты с .id."""
    if isinstance(items, Mapping):
        return [(str(k), v) for k, v in items.items()]
    pairs: list[tuple[str, Any]] = []
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            pairs.append((str(item[0]), item[1]))
        else:
            raise TypeError(
                f"{what}: ожидался словарь или пары (id, геометрия), получено {item!r}"
            )
    return pairs


def _as_point(value: Any) -> Point:
    """Точка объекта: Point, пара (lon, lat) или словарь строки assets.csv."""
    if isinstance(value, Point):
        return value
    if isinstance(value, Mapping):
        if "geometry" in value and isinstance(value["geometry"], Point):
            return value["geometry"]
        return Point(float(value["longitude"]), float(value["latitude"]))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return Point(float(value[0]), float(value[1]))
    raise TypeError(f"не удалось получить Point из {value!r}")


def covered_assets(
    polygons: Mapping[str, Polygon] | Iterable[tuple[str, Polygon]],
    points: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> dict[str, set[str]]:
    """Какие asset_id покрыты какими зонами: {candidate_id: {asset_id, ...}}.

    Предикат — covers, то есть объект ровно на границе полигона СЧИТАЕТСЯ покрытым.
    Зоны без объектов остаются в результате с пустым множеством — так удобнее
    отбору стратегий.
    """
    zones = _as_pairs(polygons, "polygons")
    assets = [(aid, _as_point(value)) for aid, value in _as_pairs(points, "points")]

    result: dict[str, set[str]] = {}
    for candidate_id, poly in zones:
        if not isinstance(poly, Polygon):
            raise TypeError(f"{candidate_id}: ожидался Polygon")
        hit = {aid for aid, pt in assets if poly.covers(pt)}
        result[candidate_id] = hit
    return result


def unique_covered(
    polygons: Mapping[str, Polygon] | Iterable[tuple[str, Polygon]],
    points: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> set[str]:
    """Объединение уникальных asset_id по всем зонам.

    Повторное покрытие одного объекта несколькими зонами НЕ умножает его ущерб:
    он попадает в множество ровно один раз. На оплату это не влияет — каждая зона
    остаётся отдельной платной позицией, даже если геометрии пересекаются.
    """
    return unique_covered_by(covered_assets(polygons, points))


def unique_covered_by(coverage: Mapping[str, set[str]]) -> set[str]:
    """То же объединение, но по уже посчитанной карте покрытия."""
    out: set[str] = set()
    for hit in coverage.values():
        out |= set(hit)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────


def bbox_of(geoms: Iterable[Any]) -> tuple[float, float, float, float]:
    """Общий охват геометрий: (min_lon, min_lat, max_lon, max_lat).

    Используется только как служебная рамка (например, для сетки), оплачиваемой
    геометрией bbox быть не может.
    """
    minx = miny = math.inf
    maxx = maxy = -math.inf
    empty = True
    for geom in geoms:
        if geom is None or getattr(geom, "is_empty", False):
            continue
        x0, y0, x1, y1 = geom.bounds
        minx, miny = min(minx, x0), min(miny, y0)
        maxx, maxy = max(maxx, x1), max(maxy, y1)
        empty = False
    if empty:
        raise ValueError("bbox_of: пустой набор геометрий")
    return (minx, miny, maxx, maxy)


def _jsonable(value: Any) -> Any:
    """Приведение значения свойства к типам, допустимым в GeoJSON (без NaN/Inf)."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("в свойствах GeoJSON запрещены NaN и Inf")
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (set, frozenset)):
        # Множества сериализуем детерминированно, иначе порядок гуляет между запусками.
        return [_jsonable(v) for v in sorted(value, key=str)]
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def to_geojson_feature(polygon: Polygon, properties: Mapping[str, Any]) -> dict[str, Any]:
    """Feature GeoJSON для одной зоны: WGS84, координаты строго [долгота, широта].

    Вырезы недопустимы по условию кейса, поэтому полигон с holes не записывается,
    а вызывает ошибку: тихо «потерять» внутреннее кольцо хуже, чем упасть.
    """
    if not isinstance(polygon, Polygon):
        raise TypeError(f"ожидался Polygon, получен {type(polygon).__name__}")
    if polygon.is_empty:
        raise ValueError("пустой полигон нельзя записать в GeoJSON")
    if polygon.interiors:
        raise ValueError("полигон с вырезами не может быть зоной заказа")

    ring: list[list[float]] = []
    for lon, lat, *_ in polygon.exterior.coords:
        lon_f, lat_f = float(lon), float(lat)
        if not (math.isfinite(lon_f) and math.isfinite(lat_f)):
            raise ValueError("в координатах GeoJSON запрещены NaN и Inf")
        ring.append([lon_f, lat_f])  # RFC 7946: сначала долгота, потом широта
    if ring[0] != ring[-1]:
        ring.append(list(ring[0]))

    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {str(k): _jsonable(v) for k, v in properties.items()},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Сетка ячеек
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GridCell:
    """Ячейка регулярной сетки. row отсчитывается с севера, col — с запада."""

    row: int
    col: int
    polygon: Polygon


def grid_cells(
    bounds: tuple[float, float, float, float],
    cell_km: float,
    crs_equal_area: str = CRS_EQUAL_AREA,
) -> list[GridCell]:
    """Регулярная сетка ячеек cell_km × cell_km, покрывающая bounds.

    bounds задаются в WGS84 как (min_lon, min_lat, max_lon, max_lat); нарезка ведётся
    в равновеликой проекции, поэтому площадь каждой ячейки одинакова независимо от
    широты. Крайние ячейки НЕ обрезаются по границе: обрезка ломала бы порог 1 км² и
    порождала бы куски произвольной формы, а лишняя площадь просто оплачивается.

    Возвращает список GridCell с полигонами в WGS84 (порядок обхода — против часовой
    стрелки, как рекомендует RFC 7946). Соседние ячейки делят вершины ровно, без щелей.
    """
    if cell_km <= 0:
        raise ValueError("cell_km должен быть строго положителен")
    min_lon, min_lat, max_lon, max_lat = (float(v) for v in bounds)
    if max_lon <= min_lon or max_lat <= min_lat:
        raise ValueError(f"вырожденные границы: {bounds!r}")
    _check_lat_range([min_lat, max_lat])

    fwd = _transformer(CRS_GEO, crs_equal_area)
    inv = _transformer(crs_equal_area, CRS_GEO)

    x0, y0 = fwd.transform(min_lon, min_lat)
    x1, y1 = fwd.transform(max_lon, max_lat)
    step = float(cell_km) * 1000.0

    n_cols = max(1, int(math.ceil((x1 - x0) / step - _EPS)))
    n_rows = max(1, int(math.ceil((y1 - y0) / step - _EPS)))
    if n_cols * n_rows > MAX_GRID_CELLS:
        raise ValueError(
            f"сетка {n_rows}×{n_cols} превышает лимит {MAX_GRID_CELLS} ячеек; "
            "увеличьте cell_km или сузьте границы"
        )

    xs = x0 + step * np.arange(n_cols + 1, dtype=float)
    ys = y1 - step * np.arange(n_rows + 1, dtype=float)  # сверху вниз
    grid_x, grid_y = np.meshgrid(xs, ys)
    lon, lat = inv.transform(grid_x, grid_y)

    cells: list[GridCell] = []
    for r in range(n_rows):
        for c in range(n_cols):
            ring = [
                (float(lon[r + 1, c]), float(lat[r + 1, c])),  # юго-запад
                (float(lon[r + 1, c + 1]), float(lat[r + 1, c + 1])),  # юго-восток
                (float(lon[r, c + 1]), float(lat[r, c + 1])),  # северо-восток
                (float(lon[r, c]), float(lat[r, c])),  # северо-запад
            ]
            cells.append(GridCell(row=r, col=c, polygon=Polygon(ring)))
    return cells
