"""Проверки сборки пакета сдачи и его валидатора.

Логика такая: сначала во временном каталоге собирается заведомо корректный
минимальный пакет (растры 8×8, десять объектов, две зоны, планы B и C) и
проверяется, что валидатор его принимает. Потом в этот же пакет по одному
вносятся дефекты, и проверяется, что каждый из них валидатор ловит.
"""

from __future__ import annotations

import csv
import json
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import (  # noqa: E402
    ASSET_STATUS_NO_DATA,
    ASSET_STATUS_OK,
    ASSET_STATUS_PARTIAL,
    ASSET_TYPES,
    BASE_RATE_RUB_KM2_SCENARIO,
    BASE_RATE_SOURCE_SCENARIO,
    BASE_RATE_STATUS_SCENARIO,
    BUNDLE_FILES,
    CHIP_PIXEL_DEG,
    CRS_EQUAL_AREA,
    CRS_GEO,
    DATA_ROLE_EVENT,
    F_ASSET_LOSS,
    F_CANDIDATES,
    F_MASK,
    F_PROCUREMENT,
    F_STRATEGY_COMPARISON,
    F_SENSITIVITY,
    LEGAL_EDITION,
    MASK_DRY,
    MASK_NODATA,
    MASK_WATER,
    PRICE_FORMULA,
    PROB_NODATA,
    SENSOR_SAR,
    UNC_BASELINE,
    UNC_SCENARIO,
    discount_coef,
    expected_loss,
    quantize_area,
    quantize_money,
)
from src.export.bundle import BundlePaths, assemble, new_run_id  # noqa: E402
from src.export.writers import BundleWriteError, format_value, write_json  # noqa: E402
from src.validate_bundle import (  # noqa: E402
    STATUS_FAIL,
    STATUS_OK,
    STATUS_SKIP,
    format_report,
    main,
    validate,
)

# ─────────────────────────────────────────────────────────────────────────────
# Параметры учебного чипа
# ─────────────────────────────────────────────────────────────────────────────

SIZE = 8
WEST = 77.0
NORTH = 20.0
THRESHOLD = 0.5
CHIP_ID = "India_900498"
BUDGET = Decimal("5000")
OTHER_COST = Decimal("250.00")
RESOLUTION_M = 10.0

TRANSFORM = from_origin(WEST, NORTH, CHIP_PIXEL_DEG, CHIP_PIXEL_DEG)
PROFILE: dict[str, Any] = {
    "crs": CRS_GEO,
    "transform": TRANSFORM,
    "width": SIZE,
    "height": SIZE,
}


def _probability() -> np.ndarray:
    """Вероятности 8×8 плюс два пикселя nodata в углах, свободных от объектов."""
    data = np.linspace(0.0, 1.0, SIZE * SIZE, dtype=np.float32).reshape(SIZE, SIZE)
    data[0, SIZE - 1] = np.float32(PROB_NODATA)
    data[SIZE - 1, 0] = np.float32(PROB_NODATA)
    return data


def _mask(prob: np.ndarray) -> np.ndarray:
    nodata = prob == np.float32(PROB_NODATA)
    mask = np.where(prob >= THRESHOLD, MASK_WATER, MASK_DRY).astype(np.uint8)
    mask[nodata] = MASK_NODATA
    return mask


def _pixel_center(row: int, col: int) -> tuple[float, float]:
    lon = WEST + (col + 0.5) * CHIP_PIXEL_DEG
    lat = NORTH - (row + 0.5) * CHIP_PIXEL_DEG
    return lon, lat


def _polygon_area_km2(polygon: Polygon) -> Decimal:
    """Площадь контура через равновеликую проекцию, как требует протокол расчёта."""
    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    transformer = Transformer.from_crs(CRS_GEO, CRS_EQUAL_AREA, always_xy=True)
    projected = shapely_transform(transformer.transform, polygon)
    return quantize_area(projected.area / 1_000_000.0)


def _zone(index: int) -> Polygon:
    """Квадратная зона примерно 2 на 2 км рядом с чипом."""
    west = WEST + index * 0.03
    north = NORTH - index * 0.03
    return Polygon(
        [
            (west, north),
            (west + 0.02, north),
            (west + 0.02, north - 0.02),
            (west, north - 0.02),
            (west, north),
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Сборка корректного пакета
# ─────────────────────────────────────────────────────────────────────────────


def _assets() -> list[dict[str, Any]]:
    rows = []
    for index, asset_type in enumerate(ASSET_TYPES):
        row_px, col_px = index % SIZE, (index * 3 + index // SIZE) % SIZE
        lon, lat = _pixel_center(row_px, col_px)
        rows.append(
            {
                "asset_id": f"A-{index + 1:02d}",
                "chip_id": CHIP_ID,
                "asset_class": asset_type.asset_class,
                "asset_value_rub": asset_type.value_rub,
                "vulnerability_coef": asset_type.vulnerability,
                "longitude": lon,
                "latitude": lat,
                "_row": row_px,
                "_col": col_px,
            }
        )
    return rows


def _asset_loss(assets: list[dict[str, Any]], prob: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Decimal]]:
    """Строки ущерба и словарь EL по объектам.

    Последние два объекта намеренно уходят в ``partial`` и ``no_data``: у них
    поля p/EL/rank должны остаться пустыми.
    """
    statuses = {assets[-2]["asset_id"]: ASSET_STATUS_PARTIAL, assets[-1]["asset_id"]: ASSET_STATUS_NO_DATA}
    measured: list[tuple[str, float, float]] = []
    for asset in assets:
        status = statuses.get(asset["asset_id"], ASSET_STATUS_OK)
        if status != ASSET_STATUS_OK:
            continue
        p = float(prob[asset["_row"], asset["_col"]])
        el = expected_loss(p, asset["asset_value_rub"], asset["vulnerability_coef"])
        measured.append((asset["asset_id"], p, el))

    order = sorted(measured, key=lambda item: (-item[2], item[0]))
    rank_by_id = {asset_id: position for position, (asset_id, _, _) in enumerate(order, start=1)}
    losses: dict[str, Decimal] = {}
    rows: list[dict[str, Any]] = []
    for asset in assets:
        asset_id = asset["asset_id"]
        status = statuses.get(asset_id, ASSET_STATUS_OK)
        if status == ASSET_STATUS_OK:
            p, el = next((p, el) for a, p, el in measured if a == asset_id)
            losses[asset_id] = Decimal(format_value(el))
            rows.append(
                {
                    "asset_id": asset_id,
                    "chip_id": CHIP_ID,
                    "p_flood": p,
                    "expected_loss_rub": el,
                    "uncertainty": 0.1,
                    "rank": rank_by_id[asset_id],
                    "status": status,
                }
            )
        else:
            rows.append(
                {
                    "asset_id": asset_id,
                    "chip_id": CHIP_ID,
                    "p_flood": None,
                    "expected_loss_rub": None,
                    "uncertainty": None,
                    "rank": None,
                    "status": status,
                }
            )
    return rows, losses


def _candidates() -> list[dict[str, Any]]:
    features = []
    for index in range(2):
        polygon = _zone(index)
        features.append(
            {
                "geometry": mapping(polygon),
                "candidate_id": f"Z-{index + 1:02d}",
                "sensor_type": SENSOR_SAR,
                "resolution_m": RESOLUTION_M,
                "acquisition_type": "operational",
                "data_role": DATA_ROLE_EVENT,
                "observation_at": "2026-09-20",
                "available_at": "2026-09-21",
                "processing_level": "L1",
                "usage_type": "internal",
                "guaranteed_purchase": False,
                "area_km2": _polygon_area_km2(polygon),
                "base_rate_rub_km2": BASE_RATE_RUB_KM2_SCENARIO,
                "base_rate_status": BASE_RATE_STATUS_SCENARIO,
                "base_rate_source": BASE_RATE_SOURCE_SCENARIO,
            }
        )
    return features


def _positions(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Decimal]]:
    """Позиции закупки для B (обе зоны) и C (одна зона). Скидка — на корзину целиком."""
    baskets = {"B": [0, 1], "C": [0]}
    rows: list[dict[str, Any]] = []
    totals: dict[str, Decimal] = {}
    for strategy, indices in baskets.items():
        areas = [Decimal(str(candidates[i]["area_km2"])) for i in indices]
        basket_area = sum(areas, Decimal("0"))
        discount = discount_coef(float(basket_area), RESOLUTION_M, SENSOR_SAR)
        total = Decimal("0")
        for position, index in enumerate(indices):
            candidate = candidates[index]
            area = areas[position]
            processing = Decimal("1")
            usage = Decimal("1")
            freshness = Decimal("1")
            unit_price = quantize_money(
                BASE_RATE_RUB_KM2_SCENARIO * processing * usage * freshness * discount
            )
            cost = quantize_money(unit_price * area)
            total += cost
            rows.append(
                {
                    "strategy": strategy,
                    "candidate_id": candidate["candidate_id"],
                    "area_km2": area,
                    "base_rate_rub_km2": BASE_RATE_RUB_KM2_SCENARIO,
                    "base_rate_status": BASE_RATE_STATUS_SCENARIO,
                    "processing_level": candidate["processing_level"],
                    "usage_type": candidate["usage_type"],
                    "guaranteed_purchase": candidate["guaranteed_purchase"],
                    "processing_coef": processing,
                    "usage_coef": usage,
                    "freshness_coef": freshness,
                    "discount_coef": discount,
                    "discount_group_id": f"sar-{int(RESOLUTION_M)}m-2026",
                    "group_area_km2": basket_area,
                    "unit_price_rub_km2": unit_price,
                    "cost_rub": cost,
                    "formula": PRICE_FORMULA,
                    "legal_edition": LEGAL_EDITION,
                }
            )
        totals[strategy] = total
    return rows, totals


def _comparison(totals: dict[str, Decimal], losses: dict[str, Decimal]) -> list[dict[str, Any]]:
    """Сравнение стратегий: A без закупки, C в бюджете, B за бюджетом и контрфактическая."""
    total_loss = sum(losses.values(), Decimal("0"))
    ordered = sorted(losses, key=lambda a: (-losses[a], a))
    covered_by = {"A": [], "B": list(ordered), "C": ordered[:4]}
    rows: list[dict[str, Any]] = []
    for strategy in ("A", "B", "C"):
        data_cost = totals.get(strategy, Decimal("0"))
        other_cost = OTHER_COST if strategy != "A" else Decimal("0.00")
        decision_cost = quantize_money(data_cost + other_cost)
        covered = quantize_money(sum((losses[a] for a in covered_by[strategy]), Decimal("0")))
        share = (covered / total_loss).quantize(Decimal("0.000001")) if total_loss else None
        rows.append(
            {
                "strategy": strategy,
                "data_cost_rub": quantize_money(data_cost),
                "other_cost_rub": quantize_money(other_cost),
                "decision_cost_rub": decision_cost,
                "budget_rub": quantize_money(BUDGET),
                "budget_feasible": decision_cost <= BUDGET,
                "covered_expected_loss_rub": covered,
                "coverage_share": share,
                "residual_uncertainty": Decimal("0.42") if strategy == "A" else Decimal("0.30"),
                "uncertainty_status": UNC_BASELINE if strategy == "A" else UNC_SCENARIO,
                "uncertainty_basis": "учебный сценарий, формула снижения раскрыта в паспорте запуска",
                "covered_asset_ids": json.dumps(covered_by[strategy], ensure_ascii=False),
            }
        )
    return rows


def _sensitivity(totals: dict[str, Decimal], losses: dict[str, Decimal]) -> list[dict[str, Any]]:
    total_loss = quantize_money(sum(losses.values(), Decimal("0")))
    return [
        {
            "scenario_id": "S-01",
            "strategy": "C",
            "changed_inputs_json": {"budget_rub": 2500},
            "decision_cost_rub": quantize_money(totals["C"] + OTHER_COST),
            "covered_expected_loss_rub": total_loss,
            "result_status": "stable",
            "interpretation": "уполовинивание бюджета состав выбранных зон не меняет",
        },
        {
            "scenario_id": "S-02",
            "strategy": "B",
            "changed_inputs_json": {"base_rate_rub_km2": "1000"},
            "decision_cost_rub": quantize_money(totals["B"] + OTHER_COST),
            "covered_expected_loss_rub": total_loss,
            "result_status": "sensitive",
            "interpretation": "рост ставки делает широкую закупку ещё менее выполнимой",
        },
    ]


def _write_source_chip(path: Path) -> Path:
    """Исходный S1-чип той же сетки: нужен проверке 4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(PROFILE)
    profile.update(driver="GTiff", dtype="float32", count=2, nodata=None)
    data = np.zeros((SIZE, SIZE), dtype=np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.write(data, 2)
    return path


def build_bundle(root: Path) -> Path:
    """Собирает заведомо корректный минимальный пакет и возвращает каталог запуска."""
    source_chip = _write_source_chip(root / "source" / f"{CHIP_ID}_S1Hand.tif")
    prob = _probability()
    mask = _mask(prob)
    assets = _assets()
    loss_rows, losses = _asset_loss(assets, prob)
    candidates = _candidates()
    positions, totals = _positions(candidates)
    run_id = new_run_id(CHIP_ID, BUDGET)
    run_dir = root / "outputs" / run_id

    payload = {
        "profile": PROFILE,
        "probability": {"data": prob},
        "mask": {"data": mask},
        "assets": [{k: v for k, v in a.items() if not k.startswith("_")} for a in assets],
        "asset_loss": loss_rows,
        "candidates": candidates,
        "procurement": positions,
        "strategy_plans": {
            "A": [],
            "B": [c["candidate_id"] for c in candidates],
            "C": [candidates[0]["candidate_id"]],
        },
        "strategy_comparison": _comparison(totals, losses),
        "sensitivity": _sensitivity(totals, losses),
        "run_metadata": {
            "run_id": run_id,
            "chip_id": CHIP_ID,
            "event_id": "India",
            "split": "test",
            "seed": 20260926,
            "threshold": THRESHOLD,
            "source_chip_path": str(source_chip),
            "budget_rub": str(BUDGET),
            "commands": ["python -m src.cli.run_bundle --chip India_900498 --budget 5000"],
            "uncertainty": {"status": UNC_SCENARIO, "basis": "учебный сценарий"},
        },
        "source_manifest": [
            {
                "id": f"{CHIP_ID}_S1Hand",
                "url": "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/",
                "version": "v1.1",
                "observed_at": "2016-12-01",
                "published_at": "2020-06-01",
                "license": "CC BY 4.0",
                "purpose": "исходный радарный снимок события",
                "limitations": "учебный набор, к реальному имуществу не относится",
            }
        ],
    }
    assemble(run_dir, payload)
    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# Правка собранного пакета — внесение дефектов
# ─────────────────────────────────────────────────────────────────────────────


def _read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    records = list(csv.reader(lines))
    return records[0], records[1:]


def _write_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def patch_csv(path: Path, predicate, changes: dict[str, str]) -> None:
    """Меняет поля первой строки, удовлетворяющей условию."""
    header, rows = _read_table(path)
    for row in rows:
        record = dict(zip(header, row))
        if predicate(record):
            for column, value in changes.items():
                row[header.index(column)] = value
            break
    else:  # pragma: no cover - страховка от опечатки в тесте
        raise AssertionError(f"в {path.name} не нашлось строки под правку")
    _write_table(path, header, rows)


def statuses(run_dir: Path) -> dict[int, str]:
    return {result.number: result.status for result in validate(run_dir)}


def details(run_dir: Path, number: int) -> str:
    for result in validate(run_dir):
        if result.number == number:
            return " | ".join(result.details)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Корректный пакет
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    return build_bundle(tmp_path)


def test_bundle_has_all_twelve_files(bundle: Path) -> None:
    for name in BUNDLE_FILES:
        assert (bundle / name).is_file(), f"нет файла {name}"
    assert len(BundlePaths.for_run(bundle).all_files()) == 12


def test_valid_bundle_passes_validator(bundle: Path) -> None:
    results = validate(bundle)
    assert len(results) == 10
    failed = [f"{r.number}. {r.title}: {r.details}" for r in results if r.status == STATUS_FAIL]
    assert not failed, "валидатор забраковал корректный пакет: " + "; ".join(failed)
    assert main([str(bundle)]) == 0


def test_geometry_check_is_skipped_until_module_exists(bundle: Path) -> None:
    """Проверка 6 либо пройдена, либо честно помечена пропущенной — но не провалена."""
    assert statuses(bundle)[6] in (STATUS_OK, STATUS_SKIP)


def test_report_is_readable(bundle: Path) -> None:
    report = format_report(bundle, validate(bundle))
    assert "Проверка пакета" in report
    assert "Итог:" in report
    assert report.count("\n") > 10


def test_csv_is_utf8_with_commas_and_dots(bundle: Path) -> None:
    raw = (bundle / F_PROCUREMENT).read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    text = raw.decode("utf-8")
    assert text.splitlines()[0].startswith("strategy,candidate_id,area_km2")


def test_no_data_rows_are_empty_not_zero(bundle: Path) -> None:
    header, rows = _read_table(bundle / F_ASSET_LOSS)
    records = [dict(zip(header, row)) for row in rows]
    blanks = [r for r in records if r["status"] in ("partial", "no_data")]
    assert len(blanks) == 2
    for record in blanks:
        for column in ("p_flood", "expected_loss_rub", "rank"):
            assert record[column] == "", f"{record['asset_id']}: {column} должен быть пустым"


def test_new_run_id_is_readable_and_deterministic() -> None:
    from datetime import date

    run_id = new_run_id("India_900498", 250000, at=date(2026, 9, 26))
    assert run_id == "20260926-India_900498-b250000"
    assert run_id == new_run_id("India_900498", Decimal("250000"), at=date(2026, 9, 26))


def test_writers_reject_nan(tmp_path: Path) -> None:
    with pytest.raises(BundleWriteError) as info:
        write_json(tmp_path / "broken.json", {"value": float("nan")})
    assert "NaN" in str(info.value)


# ─────────────────────────────────────────────────────────────────────────────
# Дефекты: каждый должен быть пойман
# ─────────────────────────────────────────────────────────────────────────────


def test_catches_nan_in_csv(bundle: Path) -> None:
    patch_csv(
        bundle / F_SENSITIVITY,
        lambda row: row["scenario_id"] == "S-01",
        {"covered_expected_loss_rub": "NaN"},
    )
    assert statuses(bundle)[2] == STATUS_FAIL
    assert "NaN" in details(bundle, 2)


def test_catches_zero_instead_of_empty_for_no_data(bundle: Path) -> None:
    patch_csv(
        bundle / F_ASSET_LOSS,
        lambda row: row["status"] == "no_data",
        {"p_flood": "0", "expected_loss_rub": "0", "rank": "0"},
    )
    assert statuses(bundle)[2] == STATUS_FAIL
    assert "должно быть пустым" in details(bundle, 2)


def test_catches_mask_not_matching_threshold(bundle: Path) -> None:
    path = bundle / F_MASK
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile
    row, col = np.argwhere(data == MASK_DRY)[0]
    data[row, col] = MASK_WATER
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    assert statuses(bundle)[3] == STATUS_FAIL
    assert "маска не совпадает" in details(bundle, 3)


def test_catches_duplicate_candidate_id(bundle: Path) -> None:
    path = bundle / F_CANDIDATES
    document = json.loads(path.read_text(encoding="utf-8"))
    first = document["features"][0]["properties"]["candidate_id"]
    document["features"][1]["properties"]["candidate_id"] = first
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    assert statuses(bundle)[6] == STATUS_FAIL
    assert "повторяется" in details(bundle, 6)


def test_catches_zone_smaller_than_one_km2(bundle: Path) -> None:
    """Зона меньше 1 км² до округления заказом быть не может."""
    path = bundle / F_CANDIDATES
    document = json.loads(path.read_text(encoding="utf-8"))
    tiny = Polygon(
        [(WEST, NORTH), (WEST + 0.001, NORTH), (WEST + 0.001, NORTH - 0.001), (WEST, NORTH - 0.001)]
    )
    document["features"][0]["geometry"] = mapping(tiny)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    result = statuses(bundle)[6]
    if result == STATUS_SKIP:
        pytest.skip("модуль src.procurement.geometry недоступен, площади не проверяются")
    assert result == STATUS_FAIL
    assert "км²" in details(bundle, 6)


def test_catches_strategy_c_over_budget(bundle: Path) -> None:
    patch_csv(bundle / F_STRATEGY_COMPARISON, lambda row: row["strategy"] == "C", {"budget_rub": "100.00"})
    assert statuses(bundle)[8] == STATUS_FAIL
    assert "стратегия C вышла за бюджет" in details(bundle, 8)


def test_catches_wrong_position_price(bundle: Path) -> None:
    patch_csv(
        bundle / F_PROCUREMENT,
        lambda row: row["strategy"] == "B",
        {"discount_coef": "0.900000000"},
    )
    assert statuses(bundle)[10] == STATUS_FAIL
    assert "unit_price_rub_km2" in details(bundle, 10)


def test_catches_missing_file(bundle: Path) -> None:
    (bundle / F_SENSITIVITY).unlink()
    assert statuses(bundle)[1] == STATUS_FAIL
    assert main([str(bundle)]) == 1


def test_catches_plan_out_of_sync(bundle: Path) -> None:
    path = bundle / "strategy_plans.json"
    plans = json.loads(path.read_text(encoding="utf-8"))
    plans["C"] = []
    path.write_text(json.dumps(plans, ensure_ascii=False, indent=2), encoding="utf-8")
    assert statuses(bundle)[7] == STATUS_FAIL


def test_catches_double_counted_coverage(bundle: Path) -> None:
    header, rows = _read_table(bundle / F_STRATEGY_COMPARISON)
    index = header.index("covered_expected_loss_rub")
    for row in rows:
        if row[header.index("strategy")] == "B":
            row[index] = str(Decimal(row[index]) * 2)
    _write_table(bundle / F_STRATEGY_COMPARISON, header, rows)
    assert statuses(bundle)[9] == STATUS_FAIL
    assert "учтён дважды" in details(bundle, 9)
