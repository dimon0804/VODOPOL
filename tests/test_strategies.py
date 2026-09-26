"""Тесты трёх стратегий заказа A/B/C и их сравнения.

Данные синтетические и крошечные: каталог из нескольких зон и три объекта.
Так проверяются именно правила отбора, а не качество модели, и тесты остаются
быстрыми. Отдельно закрыты два места, на которых теряют баллы:

* повторное покрытие объекта не удваивает его ущерб, но каждая зона всё равно
  оплачивается как отдельная позиция;
* без реальных новых наблюдений снижение неопределённости не объявляется
  измеренным: у A статус baseline, у B и C — scenario.
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import (  # noqa: E402
    ASSET_STATUS_OK,
    BASE_RATE_RUB_KM2_SCENARIO,
    BASE_RATE_STATUS_SCENARIO,
    DATA_ROLE_CONTEXT,
    DATA_ROLE_EVENT,
    SENSOR_SAR,
    STRATEGY_BROAD,
    STRATEGY_OPEN,
    STRATEGY_SELECTIVE,
    UNC_BASELINE,
    UNC_SCENARIO,
)
from src.procurement.pricing import basket_total, price_basket  # noqa: E402
from src.procurement.strategies import (  # noqa: E402
    StrategyConfig,
    build_strategies,
    covered_loss,
    coverage_by_candidate,
    is_event_observation,
    select_broad,
    select_selective,
    spec_from_feature,
    unique_covered,
)

DEADLINE = "2016-08-20"
OBSERVED = "2016-08-13"


# ─────────────────────────────────────────────────────────────────────────────
# Конструкторы учебного каталога
# ─────────────────────────────────────────────────────────────────────────────


def zone(
    candidate_id: str,
    lon0: float,
    lat0: float,
    *,
    width: float = 0.012,
    height: float = 0.011,
    data_role: str = DATA_ROLE_EVENT,
    acquisition_type: str = "new",
    available_at: str = OBSERVED,
    area_km2: float = 1.44,
) -> dict[str, Any]:
    """Кандидатная зона: прямоугольник плюс полный набор свойств для расчёта цены."""
    ring = [
        [lon0, lat0],
        [lon0 + width, lat0],
        [lon0 + width, lat0 + height],
        [lon0, lat0 + height],
        [lon0, lat0],
    ]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "candidate_id": candidate_id,
            "sensor_type": SENSOR_SAR,
            "resolution_m": 3,
            "acquisition_type": acquisition_type,
            "data_role": data_role,
            "observation_at": OBSERVED,
            "available_at": available_at,
            "processing_level": "L2",
            "usage_type": "internal",
            "guaranteed_purchase": acquisition_type == "new",
            "area_km2": area_km2,
            "base_rate_rub_km2": float(BASE_RATE_RUB_KM2_SCENARIO),
            "base_rate_status": BASE_RATE_STATUS_SCENARIO,
        },
    }


def asset_point(asset_id: str, lon: float, lat: float) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"asset_id": asset_id},
    }


def asset_row(
    asset_id: str,
    expected_loss_rub: float | None,
    uncertainty: float | None = 0.2,
    status: str = ASSET_STATUS_OK,
) -> dict[str, Any]:
    return {
        "asset_id": asset_id,
        "expected_loss_rub": expected_loss_rub,
        "uncertainty": uncertainty,
        "status": status,
    }


def simple_case() -> tuple[list[dict], dict, list[dict]]:
    """Три непересекающиеся зоны, в каждой ровно один объект с разным ущербом."""
    catalog = [zone("z1", 93.70, 26.70), zone("z2", 93.72, 26.70), zone("z3", 93.75, 26.70)]
    assets = {
        "type": "FeatureCollection",
        "features": [
            asset_point("a01", 93.705, 26.705),
            asset_point("a02", 93.725, 26.705),
            asset_point("a03", 93.755, 26.705),
        ],
    }
    rows = [
        asset_row("a01", 1_000_000.0, 0.2),
        asset_row("a02", 500_000.0, 0.3),
        asset_row("a03", 200_000.0, 0.1),
    ]
    return catalog, assets, rows


def config(budget: str = "6000", deadline: str = DEADLINE) -> StrategyConfig:
    return StrategyConfig(budget_rub=Decimal(budget), decision_deadline=deadline)


def by_strategy(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["strategy"]: row for row in result["comparison"]}


# ─────────────────────────────────────────────────────────────────────────────
# Стратегия A
# ─────────────────────────────────────────────────────────────────────────────


def test_strategy_a_orders_nothing_and_costs_nothing():
    """A — только открытые данные: пустой план и нулевая стоимость данных."""
    catalog, assets, rows = simple_case()
    result = build_strategies(catalog, assets, rows, config())
    assert result["plans"][STRATEGY_OPEN] == []
    a = by_strategy(result)[STRATEGY_OPEN]
    assert a["data_cost_rub"] == Decimal("0.00")
    assert a["covered_expected_loss_rub"] == 0.0
    assert a["budget_feasible"] is True


def test_strategy_a_stays_empty_even_with_huge_budget():
    """Деньги на A не тратятся ни при каком бюджете — это определение стратегии."""
    catalog, assets, rows = simple_case()
    result = build_strategies(catalog, assets, rows, config(budget="10000000"))
    assert result["plans"][STRATEGY_OPEN] == []


# ─────────────────────────────────────────────────────────────────────────────
# Стратегия B
# ─────────────────────────────────────────────────────────────────────────────


def test_strategy_b_follows_declared_rule_and_ignores_budget():
    """Правило B объявлено заранее, поэтому состав не зависит от бюджета."""
    catalog, assets, rows = simple_case()
    cheap = build_strategies(catalog, assets, rows, config(budget="100"))
    rich = build_strategies(catalog, assets, rows, config(budget="10000000"))
    assert cheap["plans"][STRATEGY_BROAD] == rich["plans"][STRATEGY_BROAD]
    assert cheap["plans"][STRATEGY_BROAD] == ["z1", "z2", "z3"]


def test_strategy_b_is_counterfactual_when_over_budget():
    """B дороже бюджета — состав не урезается, помечается budget_feasible=False."""
    catalog, assets, rows = simple_case()
    result = build_strategies(catalog, assets, rows, config(budget="100"))
    b = by_strategy(result)[STRATEGY_BROAD]
    assert result["plans"][STRATEGY_BROAD] == ["z1", "z2", "z3"]
    assert b["budget_feasible"] is False
    assert b["data_cost_rub"] > Decimal("100")


def test_strategy_b_feasible_flag_flips_when_budget_is_enough():
    """Тот же состав при достаточном бюджете становится реализуемым."""
    catalog, assets, rows = simple_case()
    result = build_strategies(catalog, assets, rows, config(budget="10000000"))
    assert by_strategy(result)[STRATEGY_BROAD]["budget_feasible"] is True


def test_select_broad_takes_every_event_zone():
    """Отбор B — это буквально «все зоны наблюдения события», без фильтра по ущербу."""
    catalog, _, _ = simple_case()
    catalog.append(zone("z_ctx", 93.80, 26.70, data_role=DATA_ROLE_CONTEXT))
    assert select_broad(catalog, config()) == ["z1", "z2", "z3"]


# ─────────────────────────────────────────────────────────────────────────────
# Стратегия C
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("budget", ["0", "1000", "2500", "4000", "6000", "100000"])
def test_strategy_c_never_exceeds_budget(budget):
    """C обязана соблюдать лимит при любом объявленном бюджете."""
    catalog, assets, rows = simple_case()
    result = build_strategies(catalog, assets, rows, config(budget=budget))
    c = by_strategy(result)[STRATEGY_SELECTIVE]
    assert c["data_cost_rub"] <= Decimal(budget)
    assert c["budget_feasible"] is True


def test_strategy_c_does_not_shrink_when_budget_grows():
    """Монотонность полезности: больше денег — покрытого ущерба не меньше."""
    catalog, assets, rows = simple_case()
    covered = []
    for budget in ("1000", "2500", "5000", "8000", "100000"):
        result = build_strategies(catalog, assets, rows, config(budget=budget))
        covered.append(by_strategy(result)[STRATEGY_SELECTIVE]["covered_expected_loss_rub"])
    assert covered == sorted(covered)
    assert covered[-1] > covered[0]


def test_strategy_c_prefers_the_most_valuable_zone_first():
    """При бюджете на одну зону C берёт ту, что закрывает наибольший ущерб."""
    catalog, assets, rows = simple_case()
    result = build_strategies(catalog, assets, rows, config(budget="2500"))
    assert result["plans"][STRATEGY_SELECTIVE] == ["z1"]


def test_select_selective_ignores_zones_without_gain():
    """Зона, не покрывающая ни одного оценённого объекта, в план C не попадает."""
    catalog, assets, rows = simple_case()
    catalog.append(zone("z_empty", 93.90, 26.70))
    coverage = coverage_by_candidate(catalog, assets, DEADLINE)
    losses = {row["asset_id"]: row["expected_loss_rub"] for row in rows}
    chosen = select_selective(catalog, coverage, losses, config(budget="100000"))
    assert "z_empty" not in chosen


# ─────────────────────────────────────────────────────────────────────────────
# Уникальность покрытия и раздельная оплата
# ─────────────────────────────────────────────────────────────────────────────


def test_overlapping_zones_count_the_asset_loss_once():
    """Две пересекающиеся зоны над одним объектом дают его ущерб ровно один раз."""
    catalog = [zone("z1", 93.70, 26.70), zone("z2", 93.705, 26.702)]
    assets = {"type": "FeatureCollection", "features": [asset_point("a01", 93.708, 26.705)]}
    rows = [asset_row("a01", 1_000_000.0)]

    coverage = coverage_by_candidate(catalog, assets, DEADLINE)
    assert coverage["z1"] == {"a01"} and coverage["z2"] == {"a01"}
    assert unique_covered(["z1", "z2"], coverage) == {"a01"}

    result = build_strategies(catalog, assets, rows, config(budget="100000"))
    b = by_strategy(result)[STRATEGY_BROAD]
    assert result["plans"][STRATEGY_BROAD] == ["z1", "z2"]
    assert b["covered_expected_loss_rub"] == pytest.approx(1_000_000.0)
    assert b["covered_asset_ids"] == "a01"


def test_overlapping_zones_are_still_two_billable_positions():
    """Пересечение геометрий не делает вторую зону бесплатной."""
    catalog = [zone("z1", 93.70, 26.70), zone("z2", 93.705, 26.702)]
    assets = {"type": "FeatureCollection", "features": [asset_point("a01", 93.708, 26.705)]}
    rows = [asset_row("a01", 1_000_000.0)]

    result = build_strategies(catalog, assets, rows, config(budget="100000"))
    broad_positions = [p for p in result["positions"] if p.strategy == STRATEGY_BROAD]
    assert len(broad_positions) == 2
    assert {p.candidate_id for p in broad_positions} == {"z1", "z2"}

    one = basket_total(price_basket([spec_from_feature(catalog[0])], None, 2026, STRATEGY_BROAD))
    both = by_strategy(result)[STRATEGY_BROAD]["data_cost_rub"]
    assert both > one


def test_covered_loss_does_not_double_count():
    """Прямая проверка агрегатора: повтор идентификатора ничего не удваивает."""
    coverage = {"z1": {"a01", "a02"}, "z2": {"a02", "a03"}}
    losses = {"a01": 100.0, "a02": 50.0, "a03": 25.0}
    assert covered_loss(["z1", "z2"], coverage, losses) == pytest.approx(175.0)
    assert covered_loss(["z1", "z1"], coverage, losses) == pytest.approx(150.0)


# ─────────────────────────────────────────────────────────────────────────────
# Что не считается наблюдением события
# ─────────────────────────────────────────────────────────────────────────────


def test_context_zone_is_not_event_observation():
    """Контекстная зона описывает состояние до события и в покрытие не идёт."""
    assert is_event_observation(zone("z1", 93.70, 26.70)) is True
    assert is_event_observation(zone("zc", 93.70, 26.70, data_role=DATA_ROLE_CONTEXT)) is False


def test_archive_context_zone_is_excluded_from_event_coverage():
    """Архивный снимок в роли контекста не закрывает объект при оценке текущей воды."""
    catalog = [
        zone("z_event", 93.70, 26.70),
        zone(
            "z_archive",
            93.72,
            26.70,
            data_role=DATA_ROLE_CONTEXT,
            acquisition_type="archive",
        ),
    ]
    assets = {
        "type": "FeatureCollection",
        "features": [asset_point("a01", 93.705, 26.705), asset_point("a02", 93.725, 26.705)],
    }
    rows = [asset_row("a01", 1_000_000.0), asset_row("a02", 900_000.0)]

    coverage = coverage_by_candidate(catalog, assets, DEADLINE)
    assert "z_archive" not in coverage

    result = build_strategies(catalog, assets, rows, config(budget="100000"))
    assert result["plans"][STRATEGY_BROAD] == ["z_event"]
    b = by_strategy(result)[STRATEGY_BROAD]
    assert b["covered_asset_ids"] == "a01"
    assert b["covered_expected_loss_rub"] == pytest.approx(1_000_000.0)


def test_zone_available_after_deadline_is_excluded():
    """Данные, которые придут после момента решения, решению не помогают."""
    late = zone("z_late", 93.70, 26.70, available_at="2016-09-01")
    assert is_event_observation(late, DEADLINE) is False
    assert is_event_observation(late, "") is True  # без срока ограничение не применяется

    catalog = [zone("z_ok", 93.72, 26.70), late]
    assets = {
        "type": "FeatureCollection",
        "features": [asset_point("a01", 93.705, 26.705), asset_point("a02", 93.725, 26.705)],
    }
    rows = [asset_row("a01", 1_000_000.0), asset_row("a02", 900_000.0)]

    result = build_strategies(catalog, assets, rows, config(budget="100000"))
    assert result["plans"][STRATEGY_BROAD] == ["z_ok"]
    assert by_strategy(result)[STRATEGY_BROAD]["covered_asset_ids"] == "a02"


# ─────────────────────────────────────────────────────────────────────────────
# Доля покрытия
# ─────────────────────────────────────────────────────────────────────────────


def test_coverage_share_is_covered_over_total():
    """coverage_share — это просто отношение покрытого ущерба к суммарному."""
    catalog, assets, rows = simple_case()
    total = sum(row["expected_loss_rub"] for row in rows)
    result = build_strategies(catalog, assets, rows, config(budget="6000"))
    for row in result["comparison"]:
        assert row["coverage_share"] == pytest.approx(row["covered_expected_loss_rub"] / total)
    assert by_strategy(result)[STRATEGY_BROAD]["coverage_share"] == pytest.approx(1.0)


def test_coverage_share_is_none_when_total_loss_is_zero():
    """Нулевой знаменатель — поле пустое, а не ноль и не деление на ноль."""
    catalog = [zone("z1", 93.70, 26.70)]
    assets = {"type": "FeatureCollection", "features": [asset_point("a01", 93.705, 26.705)]}
    rows = [asset_row("a01", 0.0)]
    result = build_strategies(catalog, assets, rows, config(budget="100000"))
    for row in result["comparison"]:
        assert row["coverage_share"] is None


# ─────────────────────────────────────────────────────────────────────────────
# Статус неопределённости
# ─────────────────────────────────────────────────────────────────────────────


def test_uncertainty_status_is_baseline_for_a_and_scenario_for_b_and_c():
    """Без реальных новых наблюдений снижение не заявляется измеренным."""
    catalog, assets, rows = simple_case()
    result = build_strategies(catalog, assets, rows, config(budget="6000"))
    statuses = {row["strategy"]: row["uncertainty_status"] for row in result["comparison"]}
    assert statuses[STRATEGY_OPEN] == UNC_BASELINE
    assert statuses[STRATEGY_BROAD] == UNC_SCENARIO
    assert statuses[STRATEGY_SELECTIVE] == UNC_SCENARIO
    assert all(row["uncertainty_basis"] for row in result["comparison"])


def test_uncertainty_never_reports_measured():
    """Статус measured недопустим, пока заказанная съёмка реально не поступила."""
    catalog, assets, rows = simple_case()
    result = build_strategies(catalog, assets, rows, config(budget="100000"))
    assert all(row["uncertainty_status"] != "measured" for row in result["comparison"])


def test_empty_selective_plan_keeps_scenario_status():
    """Бюджета не хватило ни на одну зону: C ничего не заказала.

    Число остаточной неопределённости при этом равно исходному — снижать нечего, —
    но статус остаётся scenario. Стратегия закупки не превращается в стратегию
    открытых данных от того, что закупить не удалось, а кейс допускает у B и C
    только scenario, measured или not_estimated.
    """
    catalog, assets, rows = simple_case()
    result = build_strategies(catalog, assets, rows, config(budget="10"))
    c = by_strategy(result)[STRATEGY_SELECTIVE]
    a = by_strategy(result)[STRATEGY_OPEN]
    assert result["plans"][STRATEGY_SELECTIVE] == []
    assert c["uncertainty_status"] == UNC_SCENARIO
    assert a["uncertainty_status"] == UNC_BASELINE
    assert c["residual_uncertainty"] == a["residual_uncertainty"]
    assert c["covered_expected_loss_rub"] == 0.0


def test_residual_uncertainty_drops_when_coverage_grows():
    """Сценарное остаточное значение у B не больше базового у A."""
    catalog, assets, rows = simple_case()
    result = build_strategies(catalog, assets, rows, config(budget="100000"))
    comparison = by_strategy(result)
    assert comparison[STRATEGY_BROAD]["residual_uncertainty"] <= comparison[STRATEGY_OPEN][
        "residual_uncertainty"
    ]
