"""Тесты геометрического модуля и генератора кандидатных зон.

Реальные данные не нужны: растр приоритета и affine-трансформация синтетические.
Эталонные полигоны строятся в равновеликой проекции и переводятся обратно в WGS84 —
так площадь квадрата задаётся точно, а не подбирается в градусах.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer
from shapely.geometry import Point, Polygon

from src.contracts import (
    BASE_RATE_STATUS_SCENARIO,
    CRS_EQUAL_AREA,
    CRS_GEO,
    DATA_ROLE_CONTEXT,
    DATA_ROLE_EVENT,
    PROPERTIES_CANDIDATE,
    quantize_area,
)
from src.procurement.candidates import (
    CatalogConfig,
    build_catalog,
    catalog_to_geojson,
    selftest_catalog,
)
from src.procurement.geometry import (
    area_km2,
    area_km2_raw,
    bbox_of,
    covered_assets,
    grid_cells,
    to_geojson_feature,
    unique_covered,
    validate_polygon,
)

_FWD = Transformer.from_crs(CRS_GEO, CRS_EQUAL_AREA, always_xy=True)
_INV = Transformer.from_crs(CRS_EQUAL_AREA, CRS_GEO, always_xy=True)


def rect_wgs84(center_lon: float, center_lat: float, width_m: float, height_m: float) -> Polygon:
    """Прямоугольник заданного размера в метрах вокруг точки, возвращён в WGS84."""
    cx, cy = _FWD.transform(center_lon, center_lat)
    half_w, half_h = width_m / 2.0, height_m / 2.0
    corners_xy = [
        (cx - half_w, cy - half_h),
        (cx + half_w, cy - half_h),
        (cx + half_w, cy + half_h),
        (cx - half_w, cy + half_h),
    ]
    xs, ys = zip(*corners_xy)
    lon, lat = _INV.transform(list(xs), list(ys))
    return Polygon(zip(lon, lat))


# ─────────────────────────────────────────────────────────────────────────────
# Площадь
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lat", [0.0, 55.0, -55.0])
def test_square_1km_area_is_one_km2_at_any_latitude(lat: float) -> None:
    """Квадрат 1 км × 1 км даёт 1,000 км² и на экваторе, и на 55° — проекция равновеликая."""
    poly = rect_wgs84(37.6, lat, 1000.0, 1000.0)
    assert area_km2(poly) == Decimal("1.000")
    assert abs(area_km2_raw(poly) - 1.0) < 1e-6


def test_area_does_not_depend_on_latitude_unlike_degrees() -> None:
    """Одинаковые по площади зоны на разных широтах имеют разный охват в градусах."""
    equator = rect_wgs84(37.6, 0.0, 1200.0, 1200.0)
    north = rect_wgs84(37.6, 55.0, 1200.0, 1200.0)

    assert area_km2(equator) == area_km2(north) == Decimal("1.440")

    # Контроль: наивный расчёт «в градусах» такого равенства не даёт.
    def degree_area(p: Polygon) -> float:
        return p.area

    assert degree_area(north) > degree_area(equator) * 1.5


def test_area_km2_rejects_non_polygon() -> None:
    with pytest.raises(TypeError):
        area_km2_raw(Point(37.6, 55.0))  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# Валидация
# ─────────────────────────────────────────────────────────────────────────────


def test_valid_zone_has_no_violations() -> None:
    assert validate_polygon(rect_wgs84(37.6, 55.0, 1200.0, 1200.0)) == []


def test_polygon_with_hole_is_rejected() -> None:
    """Вырезать воду или пустые участки из оплачиваемой геометрии нельзя."""
    outer = rect_wgs84(37.6, 55.0, 2000.0, 2000.0)
    inner = rect_wgs84(37.6, 55.0, 400.0, 400.0)
    holed = Polygon(outer.exterior.coords, [inner.exterior.coords])

    violations = validate_polygon(holed)
    assert any("вырез" in v for v in violations), violations
    # Площадь при этом всё ещё достаточна — нарушение именно из-за выреза.
    assert area_km2_raw(holed) > 1.0


def test_self_intersecting_polygon_is_rejected() -> None:
    """«Бабочка» — невалидная геометрия, самопересечение контура."""
    bowtie = Polygon(
        [
            (37.600, 55.000),
            (37.620, 55.020),
            (37.600, 55.020),
            (37.620, 55.000),
        ]
    )
    violations = validate_polygon(bowtie)
    assert violations
    assert any("самопересеч" in v for v in violations), violations


def test_area_below_threshold_before_rounding() -> None:
    """0,9996 км² округляется до 1,000, но зоной заказа не становится."""
    poly = rect_wgs84(37.6, 55.0, 999.8, 999.8)  # 999,8² м² = 0,99960004 км²

    raw = area_km2_raw(poly)
    assert raw < 1.0
    assert abs(raw - 0.9996) < 1e-4

    # Округление «спасло» бы порог — поэтому сравнение идёт до округления.
    assert quantize_area(raw) == Decimal("1.000")
    assert area_km2(poly) == Decimal("1.000")

    violations = validate_polygon(poly)
    assert any("площадь меньше минимальной" in v for v in violations), violations


def test_empty_and_wrong_type_are_reported() -> None:
    assert validate_polygon(Polygon()) == ["геометрия пустая"]
    assert validate_polygon(Point(37.6, 55.0))[0].startswith("геометрия не является Polygon")


# ─────────────────────────────────────────────────────────────────────────────
# Покрытие объектов
# ─────────────────────────────────────────────────────────────────────────────


def test_covers_includes_point_exactly_on_boundary() -> None:
    """Граница включительно: covers, а не contains."""
    zone = rect_wgs84(37.6, 55.0, 1200.0, 1200.0)
    lon, lat = list(zone.exterior.coords)[0]
    on_edge = Point(lon, lat)

    assert zone.covers(on_edge)
    assert not zone.contains(on_edge)

    coverage = covered_assets({"cand_r00_c00": zone}, {"a01": on_edge})
    assert coverage == {"cand_r00_c00": {"a01"}}


def test_overlapping_zones_give_one_unique_asset() -> None:
    """Повторное покрытие объекта не умножает его: в unique_covered он один раз."""
    left = rect_wgs84(37.600, 55.0, 1500.0, 1500.0)
    right = rect_wgs84(37.605, 55.0, 1500.0, 1500.0)
    asset = Point(37.6025, 55.0)

    assert left.covers(asset) and right.covers(asset)
    assert left.intersects(right)

    zones = {"cand_r00_c00": left, "cand_r00_c01": right}
    coverage = covered_assets(zones, {"a01": asset})
    assert coverage == {"cand_r00_c00": {"a01"}, "cand_r00_c01": {"a01"}}

    # Уникальный охват — один объект, хотя оплачиваемых позиций две.
    assert unique_covered(zones, {"a01": asset}) == {"a01"}
    assert len(zones) == 2


def test_covered_assets_accepts_pairs_and_coordinates() -> None:
    zone = rect_wgs84(37.6, 55.0, 2000.0, 2000.0)
    coverage = covered_assets([("z1", zone)], [("a01", (37.6, 55.0)), ("a02", (10.0, 10.0))])
    assert coverage == {"z1": {"a01"}}


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────


def test_bbox_of_covers_all_geometries() -> None:
    a = rect_wgs84(37.60, 55.0, 1000.0, 1000.0)
    b = rect_wgs84(37.70, 55.1, 1000.0, 1000.0)
    min_lon, min_lat, max_lon, max_lat = bbox_of([a, b])
    assert min_lon < 37.60 < 37.70 < max_lon
    assert min_lat < 55.0 < 55.1 < max_lat


def test_geojson_feature_is_lon_lat_and_rejects_holes() -> None:
    zone = rect_wgs84(37.6, 55.0, 1200.0, 1200.0)
    feature = to_geojson_feature(zone, {"candidate_id": "cand_r00_c00", "area_km2": area_km2(zone)})

    assert feature["type"] == "Feature"
    assert feature["geometry"]["type"] == "Polygon"
    ring = feature["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1], "кольцо должно быть замкнуто"
    for lon, lat in ring:
        assert 37.0 < lon < 38.0, "первая координата — долгота"
        assert 54.0 < lat < 56.0, "вторая координата — широта"
    assert isinstance(feature["properties"]["area_km2"], float)

    outer = rect_wgs84(37.6, 55.0, 2000.0, 2000.0)
    inner = rect_wgs84(37.6, 55.0, 400.0, 400.0)
    with pytest.raises(ValueError):
        to_geojson_feature(Polygon(outer.exterior.coords, [inner.exterior.coords]), {})


# ─────────────────────────────────────────────────────────────────────────────
# Сетка
# ─────────────────────────────────────────────────────────────────────────────


def test_grid_cells_have_equal_area_and_cover_bounds() -> None:
    bounds = (37.60, 55.00, 37.70, 55.05)
    cells = grid_cells(bounds, 1.2)

    assert cells
    areas = {area_km2(c.polygon) for c in cells}
    assert areas == {Decimal("1.440")}, areas

    rows = {c.row for c in cells}
    cols = {c.col for c in cells}
    assert rows == set(range(max(rows) + 1))
    assert cols == set(range(max(cols) + 1))

    # Сетка накрывает границы целиком (крайние ячейки намеренно не обрезаются).
    # Допуск 1e-6° (≈ 0,1 м) — цена round-trip WGS84 → EPSG:6933 → WGS84.
    tol = 1e-6
    min_lon, min_lat, max_lon, max_lat = bbox_of([c.polygon for c in cells])
    assert min_lon <= bounds[0] + tol and min_lat <= bounds[1] + tol
    assert max_lon >= bounds[2] - tol and max_lat >= bounds[3] - tol

    # row 0 — самый северный.
    north = [c for c in cells if c.row == 0][0]
    south = [c for c in cells if c.row == max(rows)][0]
    assert north.polygon.bounds[3] > south.polygon.bounds[3]


def test_grid_cells_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        grid_cells((37.6, 55.0, 37.7, 55.05), 0.0)
    with pytest.raises(ValueError):
        grid_cells((37.7, 55.0, 37.6, 55.05), 1.2)


# ─────────────────────────────────────────────────────────────────────────────
# Каталог кандидатных зон
# ─────────────────────────────────────────────────────────────────────────────


def synthetic_chip() -> tuple[np.ndarray, Affine, dict[str, Point]]:
    """Синтетический чип 512×512 с шагом 9e-05°, пятном приоритета и тремя объектами."""
    size = 512
    step = 9e-05
    west, north = 37.60, 55.05
    transform = Affine.translation(west, north) @ Affine.scale(step, -step)

    priority = np.zeros((size, size), dtype=float)
    priority[100:260, 120:300] = 1.0  # «пятно» ожидаемого ущерба
    priority[300:320, 400:420] = 5.0  # локальный максимум
    priority[0:10, 0:10] = np.nan  # nodata не должен ломать суммы

    def lonlat(row: int, col: int) -> Point:
        lon, lat = transform @ (col + 0.5, row + 0.5)
        return Point(float(lon), float(lat))

    assets = {
        "a01": lonlat(180, 200),
        "a02": lonlat(310, 410),
        "a03": lonlat(480, 40),  # вне зоны приоритета — ячейка остаётся из-за объекта
    }
    return priority, transform, assets


def test_build_catalog_passes_selftest() -> None:
    priority, transform, assets = synthetic_chip()
    features = build_catalog(priority, transform, assets)

    assert features
    assert selftest_catalog(features) == []

    for feature in features:
        props = feature["properties"]
        for name in PROPERTIES_CANDIDATE:
            assert name in props, name
        assert props["area_km2"] >= 1.0
        assert props["base_rate_status"] == BASE_RATE_STATUS_SCENARIO


def test_build_catalog_ids_are_stable_and_unique() -> None:
    priority, transform, assets = synthetic_chip()
    first = build_catalog(priority, transform, assets)
    second = build_catalog(priority, transform, assets)

    ids = [f["properties"]["candidate_id"] for f in first]
    assert ids == [f["properties"]["candidate_id"] for f in second]
    assert len(ids) == len(set(ids))
    assert any(i.startswith("cand_r") and "_c" in i for i in ids)


def test_build_catalog_drops_empty_cells_and_keeps_assets() -> None:
    priority, transform, assets = synthetic_chip()
    cfg = CatalogConfig(context_count=0)
    features = build_catalog(priority, transform, assets, cfg)

    total_cells = len(grid_cells(
        (
            min(p.x for p in assets.values()) - 1,  # заведомо шире — просто для контраста
            min(p.y for p in assets.values()) - 1,
            max(p.x for p in assets.values()) + 1,
            max(p.y for p in assets.values()) + 1,
        ),
        1.2,
    ))
    assert len(features) < total_cells, "пустые ячейки должны отбрасываться"

    covered = set()
    for feature in features:
        covered |= set(feature["properties"]["covered_asset_ids"])
    assert covered == set(assets), "ячейка с объектом остаётся даже при нулевом приоритете"

    assert all(f["properties"]["data_role"] == DATA_ROLE_EVENT for f in features)


def test_context_zones_are_archive_and_not_guaranteed() -> None:
    priority, transform, assets = synthetic_chip()
    features = build_catalog(priority, transform, assets, CatalogConfig(context_count=2))

    context = [f for f in features if f["properties"]["data_role"] == DATA_ROLE_CONTEXT]
    assert len(context) == 2
    for feature in context:
        props = feature["properties"]
        assert props["acquisition_type"] == "archive"
        assert props["guaranteed_purchase"] is False
        assert props["candidate_id"].startswith("ctx_")


def test_catalog_to_geojson_structure() -> None:
    priority, transform, assets = synthetic_chip()
    collection = catalog_to_geojson(build_catalog(priority, transform, assets))

    assert collection["type"] == "FeatureCollection"
    assert collection["features"]
    assert all(f["type"] == "Feature" for f in collection["features"])

    import json

    dumped = json.dumps(collection, ensure_ascii=False, allow_nan=False)
    assert "NaN" not in dumped and "Infinity" not in dumped


def test_selftest_catches_broken_catalog() -> None:
    priority, transform, assets = synthetic_chip()
    features = build_catalog(priority, transform, assets)

    # Подменённая площадь должна быть поймана.
    broken = [dict(f) for f in features]
    broken[0] = {
        **broken[0],
        "properties": {**broken[0]["properties"], "area_km2": 99.999},
    }
    problems = selftest_catalog(broken)
    assert any("не совпадает" in p for p in problems), problems

    # Дубликат candidate_id тоже.
    duplicated = list(features) + [features[0]]
    assert any("не уникален" in p for p in selftest_catalog(duplicated))

    assert selftest_catalog([]) == ["каталог пуст: ни одной кандидатной зоны"]
