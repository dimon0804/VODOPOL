"""Тесты анализа чувствительности решения.

Кейс требует отдельно проверить устойчивость ожидаемого ущерба к p, V и q и
устойчивость плана закупки как минимум к бюджету и к одному экономическому
параметру. Тесты идут на том же крошечном синтетическом каталоге, что и тесты
стратегий: здесь проверяются правила пересчёта, а не модель.

Отдельно закреплено, что сценарий p не трогает V и q и не переобучает модель —
это видно и по значениям, и по содержимому changed_inputs_json.
"""

from __future__ import annotations

import json
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
    COLUMNS_SENSITIVITY,
    DATA_ROLE_EVENT,
    SENSOR_SAR,
    STRATEGIES,
    STRATEGY_BROAD,
    STRATEGY_SELECTIVE,
)
from src.procurement.strategies import StrategyConfig, build_strategies  # noqa: E402
from src.sensitivity import default_scenarios, plan_changes, run_sensitivity  # noqa: E402

DEADLINE = "2016-08-20"
OBSERVED = "2016-08-13"
BASE_BUDGET = Decimal("6000")


# ─────────────────────────────────────────────────────────────────────────────
# Учебный вход
# ─────────────────────────────────────────────────────────────────────────────


def zone(candidate_id: str, lon0: float, lat0: float) -> dict[str, Any]:
    ring = [
        [lon0, lat0],
        [lon0 + 0.012, lat0],
        [lon0 + 0.012, lat0 + 0.011],
        [lon0, lat0 + 0.011],
        [lon0, lat0],
    ]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "candidate_id": candidate_id,
            "sensor_type": SENSOR_SAR,
            "resolution_m": 3,
            "acquisition_type": "new",
            "data_role": DATA_ROLE_EVENT,
            "observation_at": OBSERVED,
            "available_at": OBSERVED,
            "processing_level": "L2",
            "usage_type": "internal",
            "guaranteed_purchase": True,
            "area_km2": 1.44,
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


def case() -> tuple[list[dict], dict, list[dict]]:
    """Три зоны, три объекта, ущерб 1 000 000 / 500 000 / 200 000 руб."""
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
        {
            "asset_id": "a01",
            "p_flood": 0.5,
            "expected_loss_rub": 1_000_000.0,
            "uncertainty": 0.2,
            "status": ASSET_STATUS_OK,
        },
        {
            "asset_id": "a02",
            "p_flood": 0.3,
            "expected_loss_rub": 500_000.0,
            "uncertainty": 0.3,
            "status": ASSET_STATUS_OK,
        },
        {
            "asset_id": "a03",
            "p_flood": 0.1,
            "expected_loss_rub": 200_000.0,
            "uncertainty": 0.1,
            "status": ASSET_STATUS_OK,
        },
    ]
    return catalog, assets, rows


def config() -> StrategyConfig:
    return StrategyConfig(budget_rub=BASE_BUDGET, decision_deadline=DEADLINE)


@pytest.fixture(scope="module")
def sensitivity_rows():
    catalog, assets, rows = case()
    return run_sensitivity(catalog, assets, rows, config())


@pytest.fixture(scope="module")
def base_comparison():
    catalog, assets, rows = case()
    result = build_strategies(catalog, assets, rows, config())
    return {row["strategy"]: row for row in result["comparison"]}


def pick(rows, scenario_id: str, strategy: str) -> dict[str, Any]:
    for row in rows:
        if row["scenario_id"] == scenario_id and row["strategy"] == strategy:
            return row
    raise AssertionError(f"нет строки {scenario_id}/{strategy}")


# ─────────────────────────────────────────────────────────────────────────────
# Состав обязательного минимума
# ─────────────────────────────────────────────────────────────────────────────


def test_scenarios_cover_p_value_vulnerability_budget_and_rate():
    """Кейс требует p, V и q по отдельности плюс бюджет и экономический параметр."""
    ids = {s.scenario_id for s in default_scenarios(BASE_BUDGET)}
    assert {"p_x0.8", "p_x1.2"} <= ids
    assert {"V_x0.8", "V_x1.2"} <= ids
    assert {"q_x0.8", "q_x1.2"} <= ids
    assert {"budget_x0.5", "budget_x2.0"} <= ids
    assert {"base_rate_x0.5", "base_rate_x2.0"} <= ids


def test_threshold_shift_is_not_a_sensitivity_scenario():
    """Сдвиг порога бинаризации ущерб не меняет и сценарием чувствительности не является."""
    ids = {s.scenario_id for s in default_scenarios(BASE_BUDGET)}
    assert not any("threshold" in scenario_id for scenario_id in ids)


def test_every_scenario_is_reported_for_every_strategy(sensitivity_rows):
    """По строке на пару «сценарий x стратегия», иначе сравнение несопоставимо."""
    ids = {row["scenario_id"] for row in sensitivity_rows}
    assert len(sensitivity_rows) == len(ids) * len(STRATEGIES)
    for scenario_id in ids:
        strategies = {r["strategy"] for r in sensitivity_rows if r["scenario_id"] == scenario_id}
        assert strategies == set(STRATEGIES)


def test_rows_have_exactly_the_contract_columns(sensitivity_rows):
    """Имена колонок sensitivity.csv заморожены контрактом."""
    for row in sensitivity_rows:
        assert set(row) == set(COLUMNS_SENSITIVITY)


# ─────────────────────────────────────────────────────────────────────────────
# Вероятность
# ─────────────────────────────────────────────────────────────────────────────


def test_probability_scenario_changes_covered_loss(sensitivity_rows, base_comparison):
    """+20 % к p меняет покрытый ущерб линейно и ничего больше не трогает."""
    base = base_comparison[STRATEGY_SELECTIVE]["covered_expected_loss_rub"]
    up = pick(sensitivity_rows, "p_x1.2", STRATEGY_SELECTIVE)
    down = pick(sensitivity_rows, "p_x0.8", STRATEGY_SELECTIVE)
    assert up["covered_expected_loss_rub"] == pytest.approx(base * 1.2)
    assert down["covered_expected_loss_rub"] == pytest.approx(base * 0.8)
    assert up["covered_expected_loss_rub"] != base


def test_probability_scenario_does_not_touch_value_or_vulnerability(sensitivity_rows):
    """В сценарии p меняется только p: V и q в описании изменений не фигурируют."""
    changed = json.loads(pick(sensitivity_rows, "p_x1.2", STRATEGY_SELECTIVE)["changed_inputs_json"])
    assert changed["p_flood_factor"] == 1.2
    assert "V_factor" not in changed
    assert "q_factor" not in changed
    assert "budget_rub" not in changed
    assert "base_rate_rub_km2" not in changed
    assert changed["rule"] == "p := clip(p * factor, 0, 1)"


def test_probability_scenario_does_not_change_the_plan(sensitivity_rows, base_comparison):
    """Сценарий p не переобучает модель и цену данных не меняет."""
    base_cost = base_comparison[STRATEGY_SELECTIVE]["decision_cost_rub"]
    assert pick(sensitivity_rows, "p_x1.2", STRATEGY_SELECTIVE)["decision_cost_rub"] == base_cost


# ─────────────────────────────────────────────────────────────────────────────
# Стоимость и уязвимость
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("scenario_id,factor", [("V_x1.2", 1.2), ("V_x0.8", 0.8)])
def test_value_scenario_scales_loss_only(sensitivity_rows, base_comparison, scenario_id, factor):
    """Ущерб линеен по V; вероятность и уязвимость при этом зафиксированы."""
    row = pick(sensitivity_rows, scenario_id, STRATEGY_BROAD)
    base = base_comparison[STRATEGY_BROAD]["covered_expected_loss_rub"]
    assert row["covered_expected_loss_rub"] == pytest.approx(base * factor)
    changed = json.loads(row["changed_inputs_json"])
    assert changed == {"V_factor": factor}


@pytest.mark.parametrize("scenario_id,factor", [("q_x1.2", 1.2), ("q_x0.8", 0.8)])
def test_vulnerability_scenario_scales_loss_only(
    sensitivity_rows, base_comparison, scenario_id, factor
):
    """Ущерб линеен по q; вероятность и стоимость объектов зафиксированы."""
    row = pick(sensitivity_rows, scenario_id, STRATEGY_BROAD)
    base = base_comparison[STRATEGY_BROAD]["covered_expected_loss_rub"]
    assert row["covered_expected_loss_rub"] == pytest.approx(base * factor)
    changed = json.loads(row["changed_inputs_json"])
    assert changed == {"q_factor": factor}


# ─────────────────────────────────────────────────────────────────────────────
# Бюджет
# ─────────────────────────────────────────────────────────────────────────────


def test_halved_budget_does_not_increase_cost_of_c(sensitivity_rows, base_comparison):
    """Урезали бюджет вдвое — стоимость решения C не выросла."""
    base_cost = base_comparison[STRATEGY_SELECTIVE]["decision_cost_rub"]
    row = pick(sensitivity_rows, "budget_x0.5", STRATEGY_SELECTIVE)
    assert row["decision_cost_rub"] <= base_cost
    assert row["decision_cost_rub"] <= BASE_BUDGET * Decimal("0.5")


def test_budget_scenario_declares_the_new_budget(sensitivity_rows):
    """В changed_inputs_json лежит именно новый бюджет, а не множитель наугад."""
    changed = json.loads(pick(sensitivity_rows, "budget_x0.5", STRATEGY_SELECTIVE)["changed_inputs_json"])
    assert changed == {"budget_rub": float(BASE_BUDGET * Decimal("0.5"))}
    changed_up = json.loads(pick(sensitivity_rows, "budget_x2.0", STRATEGY_SELECTIVE)["changed_inputs_json"])
    assert changed_up == {"budget_rub": float(BASE_BUDGET * Decimal("2"))}


def test_budget_scenario_does_not_change_broad_plan_cost(sensitivity_rows, base_comparison):
    """Бюджет не влияет на состав B, поэтому её стоимость от бюджета не зависит."""
    base_cost = base_comparison[STRATEGY_BROAD]["decision_cost_rub"]
    for scenario_id in ("budget_x0.5", "budget_x2.0"):
        assert pick(sensitivity_rows, scenario_id, STRATEGY_BROAD)["decision_cost_rub"] == base_cost


def test_budget_scenario_does_not_change_expected_loss(sensitivity_rows, base_comparison):
    """Смена бюджета не переобучает модель: ущерб портфеля тот же, меняется корзина."""
    base = base_comparison[STRATEGY_BROAD]["covered_expected_loss_rub"]
    for scenario_id in ("budget_x0.5", "budget_x2.0"):
        row = pick(sensitivity_rows, scenario_id, STRATEGY_BROAD)
        assert row["covered_expected_loss_rub"] == pytest.approx(base)


# ─────────────────────────────────────────────────────────────────────────────
# Ставка
# ─────────────────────────────────────────────────────────────────────────────


def test_base_rate_changes_cost_but_not_expected_loss(sensitivity_rows, base_comparison):
    """Ставка — экономический параметр: двигает цену, но не ожидаемый ущерб.

    Проверяем на B, состав которой от денег не зависит: цена меняется ровно в
    заданное число раз, а покрытый ущерб остаётся прежним.
    """
    base_cost = Decimal(str(base_comparison[STRATEGY_BROAD]["decision_cost_rub"]))
    base_loss = base_comparison[STRATEGY_BROAD]["covered_expected_loss_rub"]

    cheap = pick(sensitivity_rows, "base_rate_x0.5", STRATEGY_BROAD)
    dear = pick(sensitivity_rows, "base_rate_x2.0", STRATEGY_BROAD)

    assert Decimal(str(cheap["decision_cost_rub"])) < base_cost
    assert Decimal(str(dear["decision_cost_rub"])) > base_cost
    assert cheap["covered_expected_loss_rub"] == pytest.approx(base_loss)
    assert dear["covered_expected_loss_rub"] == pytest.approx(base_loss)


def test_base_rate_scenario_declares_the_new_rate(sensitivity_rows):
    """Изменение описано числом новой ставки, воспроизводимо без чтения кода."""
    changed = json.loads(pick(sensitivity_rows, "base_rate_x2.0", STRATEGY_BROAD)["changed_inputs_json"])
    assert changed == {"base_rate_rub_km2": float(BASE_RATE_RUB_KM2_SCENARIO * Decimal("2"))}


def test_expensive_rate_shrinks_selective_basket(sensitivity_rows, base_comparison):
    """При дорогой съёмке C укладывается в тот же бюджет меньшим покрытием."""
    dear = pick(sensitivity_rows, "base_rate_x2.0", STRATEGY_SELECTIVE)
    assert dear["decision_cost_rub"] <= BASE_BUDGET
    assert (
        dear["covered_expected_loss_rub"]
        <= base_comparison[STRATEGY_SELECTIVE]["covered_expected_loss_rub"]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Описание изменений
# ─────────────────────────────────────────────────────────────────────────────


def test_changed_inputs_json_is_valid_json_everywhere(sensitivity_rows):
    """changed_inputs_json обязан разбираться как JSON-объект в каждой строке."""
    for row in sensitivity_rows:
        payload = json.loads(row["changed_inputs_json"])
        assert isinstance(payload, dict) and payload


def test_changed_inputs_json_names_exactly_the_changed_parameter(sensitivity_rows):
    """Ключ описания совпадает с тем, что действительно менялось в сценарии."""
    expected_key = {
        "p": "p_flood_factor",
        "V": "V_factor",
        "q": "q_factor",
        "budget": "budget_rub",
        "base_rate": "base_rate_rub_km2",
    }
    for row in sensitivity_rows:
        prefix = row["scenario_id"].rsplit("_x", 1)[0]
        payload = json.loads(row["changed_inputs_json"])
        assert expected_key[prefix] in payload, row["scenario_id"]
        other_keys = set(payload) - {expected_key[prefix], "rule"}
        assert not other_keys, f"{row['scenario_id']}: лишние изменения {other_keys}"


def test_interpretation_is_filled_and_status_is_explicit(sensitivity_rows):
    """Каждая строка объясняет смысл словами и честно помечает контрфактические случаи."""
    for row in sensitivity_rows:
        assert row["interpretation"].strip()
        assert row["result_status"] in {"в пределах бюджета", "контрфактический"}


def test_plan_changes_reports_only_material_shifts(sensitivity_rows, base_comparison):
    """Краткие выводы для отчёта перечисляют только реально изменившуюся стоимость C."""
    notes = plan_changes(sensitivity_rows, base_comparison[STRATEGY_SELECTIVE])
    assert any("budget_x0.5" in note for note in notes)
    assert not any("p_x1.2" in note for note in notes)
