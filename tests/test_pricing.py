"""Тесты расчёта цены заказа ДЗЗ по ПП РФ № 840.

Запускается и как ``python -m pytest tests/test_pricing.py -q``, и напрямую
``python tests/test_pricing.py`` — во втором случае файл сам печатает результат.

Контрольные значения взяты из постановки кейса: 1 703,46 / 946,37 / 567,82.
Сравнение всегда со строкой через Decimal, чтобы поймать не только величину,
но и разрядность округления.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.contracts import (  # noqa: E402
    BASE_RATE_RUB_KM2_SCENARIO,
    BASE_RATE_STATUS_SCENARIO,
    COLUMNS_PROCUREMENT,
    LEGAL_EDITION,
    PRICE_FORMULA,
    discount_coef,
)
from src.procurement.pricing import (  # noqa: E402
    OrderSpec,
    PricedPosition,
    basket_total,
    discount_group_key,
    explain,
    freshness_from_age,
    position_row,
    price_basket,
)

YEAR = 2026


def make_spec(
    candidate_id: str = "c1",
    *,
    sensor_type: str = "optical",
    resolution_m: str = "1",
    acquisition_type: str = "new",
    processing_level: str = "L2",
    usage_type: str = "internal",
    guaranteed_purchase: bool | None = None,
    area_km2: str = "1.000",
    base_rate_rub_km2: Decimal | str = BASE_RATE_RUB_KM2_SCENARIO,
    base_rate_status: str = BASE_RATE_STATUS_SCENARIO,
    discount_group_id: str | None = None,
) -> OrderSpec:
    """Зона контрольного примера; любой параметр переопределяется точечно."""
    if guaranteed_purchase is None:
        guaranteed_purchase = acquisition_type == "new"
    return OrderSpec(
        candidate_id=candidate_id,
        sensor_type=sensor_type,
        resolution_m=Decimal(resolution_m),
        acquisition_type=acquisition_type,
        processing_level=processing_level,
        usage_type=usage_type,
        guaranteed_purchase=guaranteed_purchase,
        area_km2=Decimal(area_km2),
        base_rate_rub_km2=Decimal(base_rate_rub_km2),
        base_rate_status=base_rate_status,
        discount_group_id=discount_group_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Контрольный пример постановки
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "acquisition_type, expected",
    [
        ("new", "1703.46"),
        ("operational", "946.37"),
        ("archive", "567.82"),
    ],
)
def test_control_example(acquisition_type: str, expected: str) -> None:
    """Б=788,64; К=1,000 км²; оптика r=1 м; S=1,000 км²; L2; internal."""
    positions = price_basket([make_spec(acquisition_type=acquisition_type)], year=YEAR)
    assert len(positions) == 1
    position = positions[0]
    assert position.unit_price_rub_km2 == Decimal(expected)
    assert position.cost_rub == Decimal(expected)
    # S = 1 км², r = 1 м → ln(1) = 0 → скидки нет.
    assert position.discount_coef == Decimal("1.000000000")
    assert position.group_area_km2 == Decimal("1.000")
    assert position.formula == PRICE_FORMULA
    assert position.legal_edition == LEGAL_EDITION
    assert position.base_rate_status == BASE_RATE_STATUS_SCENARIO


def test_control_example_multipliers_visible() -> None:
    """Все множители формулы выходят наружу — цена пересчитывается руками."""
    position = price_basket([make_spec()], year=YEAR)[0]
    assert position.processing_coef == Decimal("1.2")
    assert position.usage_coef == Decimal("1")
    assert position.freshness_coef == Decimal("1.8")
    manual = (
        position.base_rate_rub_km2
        * position.processing_coef
        * position.usage_coef
        * position.freshness_coef
        * position.discount_coef
    )
    assert manual == Decimal("1703.4624000000")
    assert position.unit_price_rub_km2 == Decimal("1703.46")


def test_position_fields_match_procurement_columns() -> None:
    """PricedPosition — ровно колонки procurement_plan.csv, имя в имя."""
    assert tuple(PricedPosition.__dataclass_fields__) == COLUMNS_PROCUREMENT
    position = price_basket([make_spec()], year=YEAR, strategy="C")[0]
    row = position_row(position)
    assert tuple(row) == COLUMNS_PROCUREMENT
    assert row["strategy"] == "C"
    assert row["candidate_id"] == "c1"


def test_strategy_none_becomes_empty_cell() -> None:
    """Пустое значение в пакете — пустая строка, не ноль и не None."""
    position = price_basket([make_spec()], year=YEAR)[0]
    assert position.strategy is None
    assert position_row(position)["strategy"] == ""


# ─────────────────────────────────────────────────────────────────────────────
# Скидка за объём
# ─────────────────────────────────────────────────────────────────────────────


def test_discount_optical_100km2_r1() -> None:
    """P_opt при S = 100 км², r = 1 м."""
    assert discount_coef(100.0, 1.0, "optical") == Decimal("0.769900129")


def test_discount_sar_100km2_r10_clipped_to_one() -> None:
    """P_sar при S = 100 км², r = 10 м упирается в 1: объёмной скидки на радаре нет."""
    assert discount_coef(100.0, 10.0, "sar") == Decimal("1.000000000")


@pytest.mark.parametrize(
    "area_km2, resolution_m, sensor_type",
    [
        (0.0, 1.0, "optical"),
        (-1.0, 1.0, "optical"),
        (100.0, 0.0, "optical"),
        (100.0, -1.0, "sar"),
    ],
)
def test_discount_rejects_nonpositive(
    area_km2: float, resolution_m: float, sensor_type: str
) -> None:
    """S и r строго положительны."""
    with pytest.raises(ValueError):
        discount_coef(area_km2, resolution_m, sensor_type)


def test_spec_rejects_nonpositive_resolution() -> None:
    with pytest.raises(ValueError, match="resolution_m"):
        make_spec(resolution_m="0")
    with pytest.raises(ValueError, match="resolution_m"):
        make_spec(resolution_m="-1")


def test_spec_rejects_nonpositive_area() -> None:
    with pytest.raises(ValueError, match="area_km2"):
        make_spec(area_km2="0")
    with pytest.raises(ValueError, match="area_km2"):
        make_spec(area_km2="-5")


def test_spec_rejects_area_below_minimum() -> None:
    """Зона заказа — не меньше 1 км² до округления."""
    with pytest.raises(ValueError, match="км²"):
        make_spec(area_km2="0.9996")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sensor_type": "lidar"},
        {"acquisition_type": "yesterday"},
        {"processing_level": "L3"},
        {"usage_type": "public"},
    ],
)
def test_spec_rejects_unknown_values(kwargs: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        make_spec(**kwargs)


def test_new_acquisition_requires_guaranteed_purchase() -> None:
    """Т = 1,8 — это про гарантированную покупку новой съёмки."""
    with pytest.raises(ValueError, match="guaranteed_purchase"):
        make_spec(acquisition_type="new", guaranteed_purchase=False)


# ─────────────────────────────────────────────────────────────────────────────
# Пересчёт группы целиком
# ─────────────────────────────────────────────────────────────────────────────


def test_basket_recomputes_group_discount_on_change() -> None:
    """Добавление зоны в группу меняет S, а значит Р и цену уже посчитанной зоны."""
    first = make_spec("z1", area_km2="10.000", acquisition_type="operational")
    second = make_spec("z2", area_km2="15.000", acquisition_type="operational")

    alone = price_basket([first], year=YEAR)[0]
    together = price_basket([first, second], year=YEAR)

    assert alone.group_area_km2 == Decimal("10.000")
    assert together[0].group_area_km2 == Decimal("25.000")
    assert together[1].group_area_km2 == Decimal("25.000")

    # Скидка изменилась — значит изменилась и цена за км² первой зоны.
    assert alone.discount_coef == Decimal("0.903450065")
    assert together[0].discount_coef == Decimal("0.850305202")
    assert together[0].discount_coef < alone.discount_coef
    assert together[0].unit_price_rub_km2 != alone.unit_price_rub_km2
    assert together[0].unit_price_rub_km2 < alone.unit_price_rub_km2

    # Обе позиции одной группы получают одинаковый коэффициент.
    assert together[0].discount_coef == together[1].discount_coef

    # Складывать заранее посчитанные цены одиночных зон нельзя: результат другой.
    naive = alone.cost_rub + price_basket([second], year=YEAR)[0].cost_rub
    assert basket_total(together) != naive


def test_prior_area_increases_group_volume() -> None:
    """Ранее учтённые заказы года копятся в ту же S."""
    spec = make_spec("z1", area_km2="10.000", acquisition_type="operational")
    key = discount_group_key(spec, YEAR)

    plain = price_basket([spec], year=YEAR)[0]
    with_prior = price_basket([spec], prior_area_by_group={key: Decimal("15.000")}, year=YEAR)[0]

    assert with_prior.group_area_km2 == Decimal("25.000")
    assert with_prior.discount_coef < plain.discount_coef
    # Площадь позиции К от скидки независима: платим за свои 10 км².
    assert with_prior.area_km2 == Decimal("10.000")


def test_groups_are_independent() -> None:
    """Разные тип данных и разрешение — разные группы, площади не складываются."""
    optical = make_spec("o1", sensor_type="optical", resolution_m="1", area_km2="10.000")
    sar = make_spec("s1", sensor_type="sar", resolution_m="10", area_km2="10.000")
    positions = price_basket([optical, sar], year=YEAR)

    assert positions[0].discount_group_id != positions[1].discount_group_id
    assert positions[0].group_area_km2 == Decimal("10.000")
    assert positions[1].group_area_km2 == Decimal("10.000")
    assert positions[1].discount_coef == Decimal("1.000000000")


def test_discount_group_key_shape() -> None:
    """Ключ группы: тип данных + разрешение + календарный год."""
    spec = make_spec(sensor_type="optical", resolution_m="1.0")
    assert discount_group_key(spec, 2026) == "optical_r1_2026"
    assert discount_group_key(spec, 2027) == "optical_r1_2027"
    assert discount_group_key(make_spec(sensor_type="sar", resolution_m="10"), 2026) == (
        "sar_r10_2026"
    )
    # 1 и 1.0 — одна и та же группа.
    assert discount_group_key(make_spec(resolution_m="1"), 2026) == discount_group_key(
        make_spec(resolution_m="1.0"), 2026
    )


def test_explicit_group_id_wins() -> None:
    """Явный discount_group_id позволяет свести позиции в одну группу вручную."""
    a = make_spec("a", area_km2="10.000", discount_group_id="ручная_группа")
    b = make_spec("b", area_km2="15.000", discount_group_id="ручная_группа")
    positions = price_basket([a, b], year=YEAR)
    assert positions[0].discount_group_id == "ручная_группа"
    assert positions[0].group_area_km2 == Decimal("25.000")


def test_duplicate_candidate_id_rejected() -> None:
    with pytest.raises(ValueError, match="дважды"):
        price_basket([make_spec("z1"), make_spec("z1")], year=YEAR)


def test_mixed_sensor_in_manual_group_rejected() -> None:
    """Ручная группа не должна смешивать сенсоры: формула скидки станет неопределённой."""
    a = make_spec("a", sensor_type="optical", resolution_m="1", discount_group_id="g")
    b = make_spec("b", sensor_type="sar", resolution_m="1", discount_group_id="g")
    with pytest.raises(ValueError, match="скидк"):
        price_basket([a, b], year=YEAR)


# ─────────────────────────────────────────────────────────────────────────────
# Граница актуальности
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "days, expected",
    [(0, "operational"), (1, "operational"), (89, "operational"), (90, "operational"), (91, "archive"), (365, "archive")],
)
def test_freshness_boundary(days: int, expected: str) -> None:
    """Ровно на 90 днях — operational (коэффициент 1), с 91 дня — archive."""
    assert freshness_from_age(days) == expected


def test_freshness_rejects_negative_age() -> None:
    with pytest.raises(ValueError, match="дав"):
        freshness_from_age(-1)


def test_freshness_boundary_affects_price() -> None:
    """90 и 91 день дают разную цену той же зоны."""
    at_90 = price_basket([make_spec(acquisition_type=freshness_from_age(90))], year=YEAR)[0]
    at_91 = price_basket([make_spec(acquisition_type=freshness_from_age(91))], year=YEAR)[0]
    assert at_90.unit_price_rub_km2 == Decimal("946.37")
    assert at_91.unit_price_rub_km2 == Decimal("567.82")


# ─────────────────────────────────────────────────────────────────────────────
# Итог корзины
# ─────────────────────────────────────────────────────────────────────────────


def test_basket_total_equals_sum_of_positions() -> None:
    """Итог — сумма цен позиций, а не пересчёт с нуля."""
    specs = [
        make_spec("z1", area_km2="10.000", acquisition_type="operational"),
        make_spec("z2", area_km2="15.500", acquisition_type="archive"),
        make_spec("z3", sensor_type="sar", resolution_m="10", area_km2="26.214"),
    ]
    positions = price_basket(specs, year=YEAR, strategy="B")
    expected = sum((p.cost_rub for p in positions), Decimal("0"))
    assert basket_total(positions) == expected
    assert basket_total([]) == Decimal("0.00")


def test_position_cost_is_unit_price_times_area() -> None:
    position = price_basket(
        [make_spec("z1", area_km2="12.345", acquisition_type="operational")], year=YEAR
    )[0]
    assert position.cost_rub == (position.unit_price_rub_km2 * position.area_km2).quantize(
        Decimal("0.01")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Расшифровка
# ─────────────────────────────────────────────────────────────────────────────


def test_explain_contains_numbers_and_result() -> None:
    position = price_basket([make_spec()], year=YEAR)[0]
    text = explain(position)
    assert "788.64" in text
    assert "1.8" in text
    assert "1703.46" in text
    assert "optical_r1_2026" in text
    assert LEGAL_EDITION in text


def _run_standalone() -> int:
    """Запуск без pytest: python tests/test_pricing.py."""
    import inspect

    module = sys.modules[__name__]
    failures = 0
    passed = 0
    for name, func in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(func):
            continue
        marks = getattr(func, "pytestmark", [])
        cases: list[tuple] = [()]
        argnames: list[str] = []
        for mark in marks:
            if mark.name == "parametrize":
                argnames = [a.strip() for a in mark.args[0].split(",")]
                raw = list(mark.args[1])
                cases = [c if isinstance(c, tuple) else (c,) for c in raw]
        for case in cases:
            kwargs = dict(zip(argnames, case)) if argnames else {}
            label = f"{name}{case if case else ''}"
            try:
                if kwargs:
                    func(**kwargs)
                else:
                    sig = inspect.signature(func)
                    if sig.parameters:
                        continue
                    func()
                passed += 1
            except Exception as error:  # noqa: BLE001
                failures += 1
                sys.stdout.write(f"FAIL {label}: {error!r}\n")
    sys.stdout.write(f"\nпройдено {passed}, провалено {failures}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
