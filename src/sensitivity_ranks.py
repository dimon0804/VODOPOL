"""Как изменение p, V и q сдвигает приоритеты проверки объектов.

Критерий 14 требует показать влияние p, V и q именно на **приоритеты**, а не только
на стоимость плана. Обычная чувствительность этого не покажет: если умножить
вероятность у всех десяти объектов на один и тот же множитель, ожидаемый ущерб у всех
изменится пропорционально, и порядок останется прежним. Ноль информации.

Поэтому здесь возмущения неоднородные и содержательные:

* **p у ненадёжно оценённых.** Вероятность меняется только там, где модель не уверена.
  Это и есть вопрос оператора: «что будет, если модель ошиблась именно там, где сама
  в себе сомневается».
* **V по классу объектов.** Переоценка стоимости одного типа имущества — обычная
  ситуация: реестр уточнили, и склад оказался дороже, чем думали.
* **q по классу объектов.** Уязвимость пересмотрели после первого обследования.
* **Порог оценки.** Часть объектов переводится в статус no_data — проверяем, что
  неоценённая экспозиция не исчезает из приоритетов молча.

Для каждого сценария считается новый порядок рангов и число объектов, сменивших
позицию. Если приоритеты не поменялись — так и пишем: это тоже результат.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Sequence

from src import contracts as C

COLUMNS = (
    "scenario_id",
    "changed_inputs_json",
    "asset_id",
    "asset_class",
    "rank_base",
    "rank_scenario",
    "rank_shift",
    "expected_loss_base_rub",
    "expected_loss_scenario_rub",
    "status",
    "interpretation",
)


def _rank(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    """Ранги по ожидаемому ущербу: 1 — максимальный, при равенстве по asset_id."""
    scored = [
        row
        for row in rows
        if row.get("status") == C.ASSET_STATUS_OK and row.get("expected_loss_rub") is not None
    ]
    order = sorted(scored, key=lambda row: (-float(row["expected_loss_rub"]), row["asset_id"]))
    ranks: dict[str, int | None] = {row["asset_id"]: None for row in rows}
    for position, row in enumerate(order, start=1):
        ranks[row["asset_id"]] = position
    return ranks


def _apply(
    rows: Sequence[Mapping[str, Any]],
    classes: Mapping[str, str],
    change: Callable[[dict[str, Any], str], None],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if item.get("status") == C.ASSET_STATUS_OK and item.get("expected_loss_rub") is not None:
            change(item, classes.get(item["asset_id"], ""))
        out.append(item)
    return out


def _scale_where_uncertain(threshold: float, factor: float):
    """Меняет p только у объектов с неопределённостью выше порога."""

    def change(item: dict[str, Any], _asset_class: str) -> None:
        unc = item.get("uncertainty")
        if unc is None or float(unc) < threshold:
            return
        old_p = float(item["p_flood"])
        new_p = min(1.0, max(0.0, old_p * factor))
        item["p_flood"] = new_p
        item["expected_loss_rub"] = (
            float(item["expected_loss_rub"]) * (new_p / old_p) if old_p > 0 else 0.0
        )

    return change


def _scale_class(asset_class: str, factor: float):
    """Меняет V или q у одного класса объектов — ущерб линеен по обоим."""

    def change(item: dict[str, Any], item_class: str) -> None:
        if item_class == asset_class:
            item["expected_loss_rub"] = float(item["expected_loss_rub"]) * factor

    return change


def _to_no_data(asset_ids: set[str]):
    """Переводит объекты в статус no_data: оценки нет, поля пустые."""

    def change(item: dict[str, Any], _item_class: str) -> None:
        if item["asset_id"] in asset_ids:
            item["status"] = C.ASSET_STATUS_NO_DATA
            item["p_flood"] = None
            item["expected_loss_rub"] = None
            item["uncertainty"] = None

    return change


def build_scenarios(
    rows: Sequence[Mapping[str, Any]], classes: Mapping[str, str]
) -> list[tuple[str, dict[str, Any], Callable, str]]:
    """Набор неоднородных возмущений. Каждое меняет ровно одну вещь."""
    ranked = _rank(rows)
    top = [a for a, r in ranked.items() if r == 1]
    uncertainties = [
        float(row["uncertainty"])
        for row in rows
        if row.get("uncertainty") is not None
    ]
    median_unc = sorted(uncertainties)[len(uncertainties) // 2] if uncertainties else 0.0

    scenarios: list[tuple[str, dict[str, Any], Callable, str]] = []
    for factor in (0.7, 1.3):
        scenarios.append(
            (
                f"p_uncertain_x{factor}",
                {
                    "p_flood_factor": factor,
                    "applies_to": f"объекты с неопределённостью не ниже {median_unc:.3f}",
                    "rule": "p := clip(p * factor, 0, 1) только у выбранных объектов",
                },
                _scale_where_uncertain(median_unc, factor),
                "Вероятность изменена только там, где модель сама в себе не уверена. "
                "V и q зафиксированы, модель не переобучается.",
            )
        )

    for asset_class, factor in (("clinic", 0.5), ("telecom_node", 3.0), ("substation", 0.6)):
        title = C.ASSET_TYPE_BY_CLASS[asset_class].title_ru
        scenarios.append(
            (
                f"V_{asset_class}_x{factor}",
                {"V_factor": factor, "applies_to": asset_class},
                _scale_class(asset_class, factor),
                f"Стоимость класса «{title}» пересмотрена в {factor} раза: реестр уточнили. "
                "Вероятность и уязвимость прежние.",
            )
        )

    for asset_class, factor in (("workshop", 0.4), ("school", 2.0)):
        title = C.ASSET_TYPE_BY_CLASS[asset_class].title_ru
        scenarios.append(
            (
                f"q_{asset_class}_x{factor}",
                {"q_factor": factor, "applies_to": asset_class},
                _scale_class(asset_class, factor),
                f"Уязвимость класса «{title}» пересмотрена в {factor} раза после первого "
                "обследования. Вероятность и стоимость прежние.",
            )
        )

    if top:
        scenarios.append(
            (
                "no_data_top1",
                {"status": "no_data", "applies_to": top},
                _to_no_data(set(top)),
                "Объект с наибольшим ущербом потерял оценку: данные оказались ненадёжны. "
                "Его ущерб не становится нулём — он становится неизвестным, и объект "
                "уходит из ранжирования в неоценённую экспозицию.",
            )
        )
    return scenarios


def run(
    rows: Sequence[Mapping[str, Any]], classes: Mapping[str, str]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Строки таблицы сдвига приоритетов и короткие выводы для отчёта."""
    base_rank = _rank(rows)
    base_loss = {row["asset_id"]: row.get("expected_loss_rub") for row in rows}
    table: list[dict[str, Any]] = []
    notes: list[str] = []

    for scenario_id, changed, change, interpretation in build_scenarios(rows, classes):
        changed_rows = _apply(rows, classes, change)
        new_rank = _rank(changed_rows)
        moved = [
            a
            for a in base_rank
            if base_rank[a] is not None and new_rank[a] is not None and base_rank[a] != new_rank[a]
        ]
        dropped = [a for a in base_rank if base_rank[a] is not None and new_rank[a] is None]

        for row in changed_rows:
            asset_id = row["asset_id"]
            table.append(
                {
                    "scenario_id": scenario_id,
                    "changed_inputs_json": json.dumps(changed, ensure_ascii=False),
                    "asset_id": asset_id,
                    "asset_class": classes.get(asset_id, ""),
                    "rank_base": base_rank[asset_id],
                    "rank_scenario": new_rank[asset_id],
                    "rank_shift": (
                        None
                        if base_rank[asset_id] is None or new_rank[asset_id] is None
                        else base_rank[asset_id] - new_rank[asset_id]
                    ),
                    "expected_loss_base_rub": base_loss[asset_id],
                    "expected_loss_scenario_rub": row.get("expected_loss_rub"),
                    "status": row.get("status"),
                    "interpretation": interpretation,
                }
            )

        if moved or dropped:
            parts = []
            if moved:
                parts.append(f"сменили позицию: {', '.join(sorted(moved))}")
            if dropped:
                parts.append(f"ушли из ранжирования: {', '.join(sorted(dropped))}")
            notes.append(f"{scenario_id} — {'; '.join(parts)}")
        else:
            notes.append(f"{scenario_id} — приоритеты не изменились")

    return table, notes
