"""Тесты портфеля объектов и ожидаемого прямого ущерба.

Это самая проверяемая часть кейса: жюри смотрит и состав портфеля, и правило
размещения, и то, что низкая вероятность не обнуляет ущерб, и то, что у
неоценённых объектов поля пустые, а не нулевые.

Часть проверок идёт на реальном чипе India_900498 (он уже выгружен в data/cache),
часть — на маленьких синтетических растрах, где условие можно задать точно:
на реальном чипе нет ни одного пикселя nodata, поэтому «объект не сидит в nodata»
там проверяется тривиально и требует отдельного искусственного примера.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
from affine import Affine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import assets as A  # noqa: E402
from src.contracts import (  # noqa: E402
    ASSET_COUNT,
    ASSET_STATUS_NO_DATA,
    ASSET_STATUS_OK,
    ASSET_STATUS_PARTIAL,
    ASSET_TYPES,
    PROB_NODATA,
    expected_loss,
)

CHIP_ID = "India_900498"
CHIP_ROOT = PROJECT_ROOT / "data" / "cache"
SEED = 20260926
THRESHOLD = 0.5

#: Таблица кейса переписана сюда вручную: тест обязан падать, если кто-то
#: «поправит» значение в contracts.py и тем самым разойдётся с постановкой.
CASE_TABLE = (
    ("warehouse", 10_000_000, 0.25),
    ("substation", 20_000_000, 0.40),
    ("road_unit", 5_000_000, 0.30),
    ("facility", 8_000_000, 0.20),
    ("pumping_station", 12_000_000, 0.35),
    ("clinic", 30_000_000, 0.15),
    ("school", 18_000_000, 0.20),
    ("workshop", 6_000_000, 0.45),
    ("telecom_node", 4_000_000, 0.50),
    ("water_intake", 16_000_000, 0.30),
)


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательное
# ─────────────────────────────────────────────────────────────────────────────


def synthetic_transform(size: int = 8) -> Affine:
    """Простая геопривязка учебного растра: шаг 9e-05°, север сверху."""
    return Affine.translation(77.0, 20.0) * Affine.scale(9e-05, -9e-05)


def synthetic_s1(valid_cells, size: int = 8):
    """Два канала S1, где валидны только перечисленные пиксели, остальное — NaN."""
    vv = np.full((size, size), np.nan, dtype=np.float32)
    vh = np.full((size, size), np.nan, dtype=np.float32)
    for row, col in valid_cells:
        vv[row, col] = -10.0
        vh[row, col] = -15.0
    return vv, vh


def place_on_synthetic(valid_cells, size: int = 8, seed: int = 1):
    vv, vh = synthetic_s1(valid_cells, size)
    return A.place_assets("SYNTH_000000", vv, vh, synthetic_transform(size), seed)


def make_asset(asset_id: str, asset_class: str, value: int, q: float, row: int, col: int) -> A.Asset:
    """Объект собирается напрямую — для проверок ранжирования нужен точный контроль."""
    return A.Asset(
        asset_id=asset_id,
        chip_id="SYNTH_000000",
        asset_class=asset_class,
        asset_value_rub=value,
        vulnerability_coef=q,
        row=row,
        col=col,
        longitude=77.0 + col * 9e-05,
        latitude=20.0 - row * 9e-05,
    )


@pytest.fixture(scope="module")
def chip():
    """Реальный чип кейса. Если выгрузки нет — тесты на нём пропускаются."""
    from src.data.chips import load_chip

    if not (CHIP_ROOT / "S1Hand" / f"{CHIP_ID}_S1Hand.tif").exists():
        pytest.skip(f"чип {CHIP_ID} не выгружен в {CHIP_ROOT}")
    return load_chip(CHIP_ID, root=CHIP_ROOT)


@pytest.fixture(scope="module")
def chip_assets(chip):
    return A.place_assets(CHIP_ID, chip.vv, chip.vh, chip.transform, SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Состав портфеля
# ─────────────────────────────────────────────────────────────────────────────


def test_case_table_values_are_frozen():
    """V и q в contracts.py обязаны совпадать с таблицей постановки построчно."""
    actual = tuple((t.asset_class, t.value_rub, t.vulnerability) for t in ASSET_TYPES)
    assert actual == CASE_TABLE


def test_portfolio_is_exactly_ten_objects_one_per_type(chip_assets):
    """Ровно десять объектов, по одному каждого типа — прямое требование кейса."""
    assert len(chip_assets) == ASSET_COUNT == 10
    classes = [a.asset_class for a in chip_assets]
    assert classes == [t.asset_class for t in ASSET_TYPES]
    assert len(set(classes)) == 10
    assert len({a.asset_id for a in chip_assets}) == 10


def test_portfolio_carries_case_table_values(chip_assets):
    """V и q объекта берутся из таблицы, а не подбираются при размещении."""
    for asset, asset_type in zip(chip_assets, ASSET_TYPES):
        assert asset.asset_value_rub == asset_type.value_rub
        assert asset.vulnerability_coef == pytest.approx(asset_type.vulnerability)


# ─────────────────────────────────────────────────────────────────────────────
# Воспроизводимость размещения
# ─────────────────────────────────────────────────────────────────────────────


def test_same_seed_gives_identical_coordinates(chip):
    """Фиксированное зерно — обязательное условие воспроизводимости запуска."""
    first = A.place_assets(CHIP_ID, chip.vv, chip.vh, chip.transform, SEED)
    second = A.place_assets(CHIP_ID, chip.vv, chip.vh, chip.transform, SEED)
    assert [(a.row, a.col) for a in first] == [(a.row, a.col) for a in second]
    assert [(a.longitude, a.latitude) for a in first] == [
        (a.longitude, a.latitude) for a in second
    ]


def test_different_seed_gives_different_coordinates(chip):
    """Разное зерно должно давать другой портфель, иначе зерно ни на что не влияет."""
    first = A.place_assets(CHIP_ID, chip.vv, chip.vh, chip.transform, SEED)
    other = A.place_assets(CHIP_ID, chip.vv, chip.vh, chip.transform, SEED + 1)
    assert [(a.row, a.col) for a in first] != [(a.row, a.col) for a in other]


# ─────────────────────────────────────────────────────────────────────────────
# Валидная область и уникальность пикселей
# ─────────────────────────────────────────────────────────────────────────────


def test_all_assets_land_in_valid_s1_area(chip, chip_assets):
    """Объект стоит там, где оба канала S1 конечны — это и есть валидная область."""
    valid = A.s1_valid_area(chip.vv, chip.vh)
    for asset in chip_assets:
        assert valid[asset.row, asset.col]
        assert np.isfinite(chip.vv[asset.row, asset.col])
        assert np.isfinite(chip.vh[asset.row, asset.col])


def test_assets_never_land_in_nodata():
    """На растре, где валидна лишь горстка пикселей, все объекты попадают только в них."""
    valid_cells = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4), (5, 2)]
    placed = place_on_synthetic(valid_cells, size=16, seed=7)
    assert len(placed) == 10
    assert {(a.row, a.col) for a in placed} == set(valid_cells)


def test_two_assets_never_share_a_pixel(chip_assets):
    """Один пиксель — один объект: иначе портфель схлопывается и ущерб дублируется."""
    cells = [(a.row, a.col) for a in chip_assets]
    assert len(set(cells)) == len(cells) == 10


def test_placement_fails_loudly_when_valid_area_too_small():
    """Девять валидных пикселей на десять объектов — это ошибка, а не тихое усечение."""
    valid_cells = [(0, c) for c in range(9)]
    with pytest.raises(RuntimeError):
        place_on_synthetic(valid_cells, size=16, seed=3)


# ─────────────────────────────────────────────────────────────────────────────
# Независимость размещения от прогноза и меток
# ─────────────────────────────────────────────────────────────────────────────


def test_place_assets_signature_takes_no_forecast_and_no_labels():
    """Подглядеть в прогноз или метку place_assets не может: их просто нет в сигнатуре."""
    params = set(inspect.signature(A.place_assets).parameters)
    assert params == {"chip_id", "vv", "vh", "transform", "seed"}
    forbidden = {"prob", "probability", "label", "labels", "truth", "mask", "loss"}
    assert not (params & forbidden)


def test_placement_identical_regardless_of_downstream_probability(chip):
    """Один и тот же seed на одном чипе даёт те же координаты при любом прогнозе.

    Прогоняем размещение дважды и скармливаем результат двум разным вероятностным
    растрам: координаты обязаны совпасть, меняется только ущерб.
    """
    first = A.place_assets(CHIP_ID, chip.vv, chip.vh, chip.transform, SEED)
    second = A.place_assets(CHIP_ID, chip.vv, chip.vh, chip.transform, SEED)

    shape = chip.vv.shape
    prob_low = np.full(shape, 0.05, dtype=np.float32)
    prob_high = np.full(shape, 0.95, dtype=np.float32)

    rows_low = A.evaluate_assets(first, prob_low)
    rows_high = A.evaluate_assets(second, prob_high)

    assert [(a.row, a.col) for a in first] == [(a.row, a.col) for a in second]
    assert [r["asset_id"] for r in rows_low] == [r["asset_id"] for r in rows_high]
    # Прогноз влияет только на ущерб, но не на то, где стоят объекты.
    assert [r["expected_loss_rub"] for r in rows_low] != [
        r["expected_loss_rub"] for r in rows_high
    ]


def test_placement_does_not_depend_on_labels(chip):
    """Метки не участвуют: искажение label не должно двигать ни один объект."""
    before = A.place_assets(CHIP_ID, chip.vv, chip.vh, chip.transform, SEED)
    chip.label = np.zeros_like(chip.label)
    after = A.place_assets(CHIP_ID, chip.vv, chip.vh, chip.transform, SEED)
    assert [(a.row, a.col) for a in before] == [(a.row, a.col) for a in after]


# ─────────────────────────────────────────────────────────────────────────────
# Ущерб EL = p x V x q
# ─────────────────────────────────────────────────────────────────────────────


def test_expected_loss_is_exactly_p_times_v_times_q():
    """Формула ровно такая: никаких весов, порогов и нормировок сверху."""
    assert expected_loss(0.4, 10_000_000, 0.25) == pytest.approx(1_000_000.0)
    assert expected_loss(0.0, 10_000_000, 0.25) == 0.0


def test_low_probability_does_not_zero_the_loss():
    """p = 0,1 при пороге 0,5: маска сухая, а ущерб обязан остаться положительным."""
    size = 8
    valid_cells = [(r, c) for r in range(2) for c in range(5)]
    placed = place_on_synthetic(valid_cells, size=size, seed=11)

    prob = np.full((size, size), 0.1, dtype=np.float32)
    mask = prob >= THRESHOLD
    assert not mask.any()  # бинарная маска целиком «сухая»

    rows = A.evaluate_assets(placed, prob)
    assert all(row["status"] == ASSET_STATUS_OK for row in rows)
    for row, asset in zip(rows, placed):
        assert row["expected_loss_rub"] > 0
        assert row["expected_loss_rub"] == pytest.approx(
            float(prob[asset.row, asset.col]) * asset.asset_value_rub * asset.vulnerability_coef
        )
    assert A.total_expected_loss(rows) > 0


def test_loss_matches_probability_pixel_on_real_chip(chip, chip_assets):
    """p берётся из пикселя объекта без сглаживания — проверяем на реальной геометрии."""
    rng = np.random.default_rng(5)
    prob = rng.random(chip.vv.shape).astype(np.float32)
    rows = A.evaluate_assets(chip_assets, prob)
    for row, asset in zip(rows, chip_assets):
        assert row["p_flood"] == pytest.approx(float(prob[asset.row, asset.col]))
        assert row["expected_loss_rub"] == pytest.approx(
            float(prob[asset.row, asset.col]) * asset.asset_value_rub * asset.vulnerability_coef
        )


# ─────────────────────────────────────────────────────────────────────────────
# Статусы: пусто, а не ноль
# ─────────────────────────────────────────────────────────────────────────────


def test_no_data_fields_are_none_not_zero():
    """У no_data p/EL/rank именно None: ноль означал бы «ущерба нет», а он неизвестен."""
    size = 8
    valid_cells = [(r, c) for r in range(2) for c in range(5)]
    placed = place_on_synthetic(valid_cells, size=size, seed=11)

    prob = np.full((size, size), 0.4, dtype=np.float32)
    target = placed[0]
    prob[target.row, target.col] = PROB_NODATA

    rows = {row["asset_id"]: row for row in A.evaluate_assets(placed, prob)}
    bad = rows[target.asset_id]
    assert bad["status"] == ASSET_STATUS_NO_DATA
    assert bad["p_flood"] is None
    assert bad["expected_loss_rub"] is None
    assert bad["rank"] is None
    assert bad["uncertainty"] is None


def test_partial_fields_are_none_not_zero():
    """У partial поля тоже пустые: оценка есть, но она ненадёжна."""
    size = 8
    valid_cells = [(r, c) for r in range(2) for c in range(5)]
    placed = place_on_synthetic(valid_cells, size=size, seed=11)

    prob = np.full((size, size), 0.4, dtype=np.float32)
    target = placed[0]
    # Больше трети окрестности 3x3 — nodata, сам пиксель остаётся валидным.
    row, col = target.row, target.col
    holes = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = row + dr, col + dc
            if 0 <= r < size and 0 <= c < size and holes < 5:
                prob[r, c] = PROB_NODATA
                holes += 1
    assert prob[row, col] != PROB_NODATA

    rows = {r["asset_id"]: r for r in A.evaluate_assets(placed, prob)}
    partial = rows[target.asset_id]
    assert partial["status"] == ASSET_STATUS_PARTIAL
    assert partial["p_flood"] is None
    assert partial["expected_loss_rub"] is None
    assert partial["rank"] is None


def test_nan_probability_is_no_data_not_zero():
    """NaN в растре — это «неизвестно», а не «сухо»."""
    size = 8
    valid_cells = [(r, c) for r in range(2) for c in range(5)]
    placed = place_on_synthetic(valid_cells, size=size, seed=11)
    prob = np.full((size, size), 0.4, dtype=np.float32)
    prob[placed[3].row, placed[3].col] = np.nan
    rows = {r["asset_id"]: r for r in A.evaluate_assets(placed, prob)}
    assert rows[placed[3].asset_id]["status"] == ASSET_STATUS_NO_DATA
    assert rows[placed[3].asset_id]["expected_loss_rub"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Ранги
# ─────────────────────────────────────────────────────────────────────────────


def test_rank_one_is_the_largest_expected_loss(chip_assets):
    """Ранг 1 — максимальный ущерб, дальше по убыванию без пропусков."""
    prob = np.full((512, 512), 0.5, dtype=np.float32)
    rows = A.evaluate_assets(chip_assets, prob)
    ranked = sorted(rows, key=lambda r: r["rank"])
    assert [r["rank"] for r in ranked] == list(range(1, 11))
    losses = [r["expected_loss_rub"] for r in ranked]
    assert losses == sorted(losses, reverse=True)
    # При одинаковой p максимум V*q у электроподстанции (20 млн x 0,40).
    assert ranked[0]["asset_id"] == "a02"


def test_rank_ties_are_broken_by_asset_id():
    """При равном ущербе порядок определяется asset_id — иначе ранг неустойчив."""
    placed = [
        make_asset("a02", "warehouse", 10_000_000, 0.25, 0, 0),
        make_asset("a01", "warehouse", 10_000_000, 0.25, 0, 1),
        make_asset("a03", "substation", 20_000_000, 0.40, 0, 2),
    ]
    prob = np.full((4, 4), 0.5, dtype=np.float32)
    rows = {r["asset_id"]: r for r in A.evaluate_assets(placed, prob)}
    assert rows["a03"]["rank"] == 1
    assert rows["a01"]["rank"] == 2  # равный ущерб с a02, но меньший asset_id
    assert rows["a02"]["rank"] == 3


def test_unassessed_objects_get_no_rank():
    """Неоценённые объекты не ранжируются, а оценённые нумеруются подряд."""
    size = 8
    valid_cells = [(r, c) for r in range(2) for c in range(5)]
    placed = place_on_synthetic(valid_cells, size=size, seed=11)
    prob = np.full((size, size), 0.4, dtype=np.float32)
    prob[placed[0].row, placed[0].col] = PROB_NODATA

    rows = A.evaluate_assets(placed, prob)
    ranks = [r["rank"] for r in rows if r["status"] == ASSET_STATUS_OK]
    assert sorted(ranks) == list(range(1, len(ranks) + 1))
    assert all(r["rank"] is None for r in rows if r["status"] != ASSET_STATUS_OK)


# ─────────────────────────────────────────────────────────────────────────────
# Неоценённая экспозиция
# ─────────────────────────────────────────────────────────────────────────────


def test_unassessed_exposure_is_not_empty_when_object_sits_in_nodata():
    """Объект без оценки не исчезает из отчёта: его стоимость показывается отдельно."""
    size = 8
    valid_cells = [(r, c) for r in range(2) for c in range(5)]
    placed = place_on_synthetic(valid_cells, size=size, seed=11)
    prob = np.full((size, size), 0.4, dtype=np.float32)
    target = placed[2]
    prob[target.row, target.col] = PROB_NODATA

    rows = A.evaluate_assets(placed, prob)
    exposure = A.unassessed_exposure(rows, placed)
    assert exposure["count"] >= 1
    assert target.asset_id in exposure["asset_ids"]
    assert exposure["value_rub"] >= target.asset_value_rub
    assert exposure["note"]


def test_unassessed_exposure_is_empty_when_everything_assessed(chip_assets):
    """Если оценены все — экспозиция пуста, но структура та же."""
    prob = np.full((512, 512), 0.3, dtype=np.float32)
    rows = A.evaluate_assets(chip_assets, prob)
    exposure = A.unassessed_exposure(rows, chip_assets)
    assert exposure["count"] == 0
    assert exposure["asset_ids"] == []
    assert exposure["value_rub"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Прокси-проверка ущерба
# ─────────────────────────────────────────────────────────────────────────────


def test_proxy_error_is_zero_when_forecast_equals_truth():
    """Прогноз совпал с истиной — ошибка перехода «вероятность → рубли» нулевая."""
    size = 8
    valid_cells = [(r, c) for r in range(2) for c in range(5)]
    placed = place_on_synthetic(valid_cells, size=size, seed=11)

    prob = np.ones((size, size), dtype=np.float32)
    truth = np.ones((size, size), dtype=np.uint8)

    rows = A.evaluate_assets(placed, prob)
    report = A.proxy_error(rows, placed, truth)
    assert report["status"] == "checked"
    assert report["n_points"] == 10
    assert report["mae_rub"] == pytest.approx(0.0)
    assert report["rmse_rub"] == pytest.approx(0.0)
    assert report["bias_rub"] == pytest.approx(0.0)


def test_proxy_error_equals_sum_of_value_times_q_when_forecast_is_opposite():
    """Прогноз строго противоположен истине — суммарная ошибка равна сумме |V x q|."""
    size = 8
    valid_cells = [(r, c) for r in range(2) for c in range(5)]
    placed = place_on_synthetic(valid_cells, size=size, seed=11)

    prob = np.ones((size, size), dtype=np.float32)  # предсказали воду везде
    truth = np.zeros((size, size), dtype=np.uint8)  # на деле воды нет нигде

    rows = A.evaluate_assets(placed, prob)
    report = A.proxy_error(rows, placed, truth)

    expected_sum = sum(t.value_rub * t.vulnerability for t in ASSET_TYPES)
    assert report["mae_rub"] * report["n_points"] == pytest.approx(expected_sum)
    assert report["sum_predicted_rub"] == pytest.approx(expected_sum)
    assert report["sum_reference_rub"] == pytest.approx(0.0)
    assert report["bias_rub"] > 0


# Исправлено: proxy_error исключает контрольные точки без ручной метки и сообщает,
# сколько их было. Значение −1 не считается ни водой, ни сушей.
def test_proxy_error_excludes_control_points_without_a_manual_label():
    """Контрольная точка без ручной метки — «неизвестно», её нельзя считать ни водой, ни сушей."""
    size = 8
    valid_cells = [(r, c) for r in range(2) for c in range(5)]
    placed = place_on_synthetic(valid_cells, size=size, seed=11)

    prob = np.zeros((size, size), dtype=np.float32)  # прогноз: воды нет нигде
    labels = np.zeros((size, size), dtype=np.int16)  # истина: воды нет нигде
    unlabeled = placed[0]
    labels[unlabeled.row, unlabeled.col] = -1  # LABEL_INVALID: метки просто нет

    rows = A.evaluate_assets(placed, prob)
    report = A.proxy_error(rows, placed, labels)

    assert report["n_points"] == 9  # точка без метки исключена из проверки
    assert report["mae_rub"] == pytest.approx(0.0)


def test_proxy_error_reports_not_checked_without_usable_points():
    """Пригодных контрольных точек нет — так и пишем, а не подставляем ноль."""
    size = 8
    valid_cells = [(r, c) for r in range(2) for c in range(5)]
    placed = place_on_synthetic(valid_cells, size=size, seed=11)
    prob = np.full((size, size), PROB_NODATA, dtype=np.float32)
    rows = A.evaluate_assets(placed, prob)
    report = A.proxy_error(rows, placed, np.zeros((size, size), dtype=np.uint8))
    assert report["status"] == "not_checked"
    assert report["n_points"] == 0
    assert "mae_rub" not in report
