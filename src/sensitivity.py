"""Чувствительность решения к входным параметрам.

Кейс требует проверить устойчивость ожидаемого ущерба к изменению p, V и q по
отдельности при фиксированных остальных, а плана закупки — как минимум к бюджету и
к одному экономическому параметру.

Два уточнения, на которых легко ошибиться:

* Сценарное изменение p делается БЕЗ переобучения модели: вероятности множатся или
  сдвигаются по раскрытому правилу и обрезаются в [0;1]. Правило пишется в строку
  сценария, чтобы его можно было повторить.
* Простая смена порога бинаризации сама по себе ущерб не меняет: EL считается по
  вероятностному растру, а не по маске. Поэтому «сдвинули порог» сценарием
  чувствительности не является и сюда не попадает.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Callable, Mapping, Sequence

from src import contracts as C
from src.procurement.pricing import basket_total
from src.procurement.strategies import StrategyConfig, build_strategies


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    changed: dict[str, Any]
    interpretation: str
    #: Как пересчитать вход. Возвращает (asset_rows, config).
    apply: Callable[[list[dict], StrategyConfig], tuple[list[dict], StrategyConfig]]


def _scaled_rows(rows: Sequence[Mapping[str, Any]], factor_p: float) -> list[dict]:
    """Масштабирование вероятности с обрезкой в [0;1]; V и q не трогаются."""
    out = []
    for row in rows:
        item = dict(row)
        if item.get("status") == C.ASSET_STATUS_OK and item.get("p_flood") is not None:
            old_p = float(item["p_flood"])
            new_p = min(1.0, max(0.0, old_p * factor_p))
            item["p_flood"] = new_p
            if old_p > 0:
                item["expected_loss_rub"] = float(item["expected_loss_rub"]) * (new_p / old_p)
            else:
                item["expected_loss_rub"] = 0.0
        out.append(item)
    return out


def _scaled_value(rows: Sequence[Mapping[str, Any]], field: str, factor: float) -> list[dict]:
    """Масштабирование V или q: ущерб линеен по каждому, p остаётся прежней."""
    out = []
    for row in rows:
        item = dict(row)
        if item.get("status") == C.ASSET_STATUS_OK and item.get("expected_loss_rub") is not None:
            item["expected_loss_rub"] = float(item["expected_loss_rub"]) * factor
            item[f"{field}_factor"] = factor
        out.append(item)
    return out


def default_scenarios(
    base_budget: Decimal, base_rate: Decimal = C.BASE_RATE_RUB_KM2_SCENARIO
) -> list[Scenario]:
    """Обязательный минимум: p, V, q по отдельности плюс бюджет и ставка."""
    scenarios: list[Scenario] = []

    for factor in (0.8, 1.2):
        scenarios.append(
            Scenario(
                scenario_id=f"p_x{factor}",
                changed={"p_flood_factor": factor, "rule": "p := clip(p * factor, 0, 1)"},
                interpretation=(
                    f"Вероятность изменена на {round((factor - 1) * 100):+d} % без переобучения "
                    "модели; V и q зафиксированы."
                ),
                apply=lambda rows, cfg, f=factor: (_scaled_rows(rows, f), cfg),
            )
        )

    for field, factor in (("V", 0.8), ("V", 1.2), ("q", 0.8), ("q", 1.2)):
        scenarios.append(
            Scenario(
                scenario_id=f"{field}_x{factor}",
                changed={f"{field}_factor": factor},
                interpretation=(
                    f"{'Стоимость объектов' if field == 'V' else 'Уязвимость'} изменена на "
                    f"{round((factor - 1) * 100):+d} %; вероятность и остальные параметры "
                    "зафиксированы."
                ),
                apply=lambda rows, cfg, fl=field, f=factor: (_scaled_value(rows, fl, f), cfg),
            )
        )

    for factor in (0.5, 2.0):
        scenarios.append(
            Scenario(
                scenario_id=f"budget_x{factor}",
                changed={"budget_rub": float(base_budget * Decimal(str(factor)))},
                interpretation=(
                    f"Бюджет изменён в {factor} раза. Модель не переобучается, портфель и "
                    "порог прежние — меняется только корзина заказа."
                ),
                apply=lambda rows, cfg, f=factor: (
                    list(rows),
                    _with_budget(cfg, cfg.budget_rub * Decimal(str(f))),
                ),
            )
        )

    for factor in (0.5, 2.0):
        scenarios.append(
            Scenario(
                scenario_id=f"base_rate_x{factor}",
                changed={"base_rate_rub_km2": float(base_rate * Decimal(str(factor)))},
                interpretation=(
                    f"Ставка съёмки изменена в {factor} раза. Проверяет, насколько план "
                    "закупки зависит от сценарного тарифа."
                ),
                apply=lambda rows, cfg, f=factor: (list(rows), cfg),
            )
        )

    return scenarios


def _with_budget(config: StrategyConfig, budget: Decimal) -> StrategyConfig:
    return StrategyConfig(
        budget_rub=budget,
        other_cost_rub=config.other_cost_rub,
        broad_rule=config.broad_rule,
        decision_deadline=config.decision_deadline,
        year=config.year,
        prior_area_by_group=config.prior_area_by_group,
    )


def _rate_scaled_catalog(catalog: Sequence[Mapping[str, Any]], factor: float) -> list[dict]:
    out = []
    for feature in catalog:
        item = json.loads(json.dumps(feature))
        props = item["properties"]
        props["base_rate_rub_km2"] = float(props["base_rate_rub_km2"]) * factor
        out.append(item)
    return out


def run_sensitivity(
    catalog: Sequence[Mapping[str, Any]],
    assets_geojson: Mapping[str, Any],
    asset_rows: Sequence[Mapping[str, Any]],
    config: StrategyConfig,
    scenarios: Sequence[Scenario] | None = None,
) -> list[dict[str, Any]]:
    """Строки sensitivity.csv: по строке на пару «сценарий × стратегия»."""
    scenarios = scenarios or default_scenarios(config.budget_rub)
    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        catalog_now = catalog
        if scenario.scenario_id.startswith("base_rate_x"):
            catalog_now = _rate_scaled_catalog(catalog, float(scenario.changed["base_rate_rub_km2"]) / float(C.BASE_RATE_RUB_KM2_SCENARIO))
        changed_rows, changed_config = scenario.apply(list(asset_rows), config)
        result = build_strategies(catalog_now, assets_geojson, changed_rows, changed_config)

        for comparison in result["comparison"]:
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "strategy": comparison["strategy"],
                    "changed_inputs_json": json.dumps(scenario.changed, ensure_ascii=False),
                    "decision_cost_rub": comparison["decision_cost_rub"],
                    "covered_expected_loss_rub": comparison["covered_expected_loss_rub"],
                    "result_status": (
                        "в пределах бюджета" if comparison["budget_feasible"] else "контрфактический"
                    ),
                    "interpretation": scenario.interpretation,
                }
            )
    return rows


def plan_changes(rows: Sequence[Mapping[str, Any]], base: Mapping[str, Any]) -> list[str]:
    """Короткие выводы для отчёта: где план действительно поменялся."""
    notes: list[str] = []
    base_cost = float(base.get("decision_cost_rub", 0))
    for row in rows:
        if row["strategy"] != C.STRATEGY_SELECTIVE:
            continue
        cost = float(row["decision_cost_rub"])
        if base_cost and abs(cost - base_cost) / base_cost > 0.01:
            notes.append(
                f"{row['scenario_id']}: стоимость решения {cost:.2f} против базовых "
                f"{base_cost:.2f} руб."
            )
    return notes
