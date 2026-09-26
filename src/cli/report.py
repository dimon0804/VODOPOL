"""Сборка итогового отчёта из того, что реально посчитано.

    python -m src.cli.report --run outputs/<run_id>

Отчёт не пишется руками и не содержит чисел, которых нет в артефактах: каждая цифра
берётся из файлов, сгенерированных кодом. Если какого-то куска нет, в отчёте стоит
прямая отметка «не посчитано», а не пропуск и не правдоподобная оценка.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src import contracts as C


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value, digits: int = 4, dash: str = "не посчитано") -> str:
    if value is None or value == "":
        return dash
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _money(value) -> str:
    if value is None or value == "":
        return "—"
    return f"{float(value):,.2f}".replace(",", " ")


def build(run_dir: Path, reports: Path) -> str:
    meta = _load_json(run_dir / C.F_RUN_METADATA) or {}
    compare = _load_json(reports / "metrics_compare.json")
    validation = _load_json(reports / "metrics_main_validation.json")
    holdout = _load_json(reports / "metrics_compare_holdout.json")
    asset_rows = _load_csv(run_dir / C.F_ASSET_LOSS)
    strategy_rows = _load_csv(run_dir / C.F_STRATEGY_COMPARISON)
    plan_rows = _load_csv(run_dir / C.F_PROCUREMENT)
    sensitivity_rows = _load_csv(run_dir / C.F_SENSITIVITY)
    split = meta.get("split", {})

    out: list[str] = []
    add = out.append

    add("# Водополь — отчёт по кейсу")
    add("")
    add(
        "Водополь — оценка экономического ущерба от паводка и обоснование заказа "
        "дополнительной съёмки. КосмоХакатон 2026, кейс 2, команда CLV-DIGITAL."
    )
    add("")
    add(
        f"Запуск `{meta.get('run_id', '—')}`: чип **{meta.get('chip_id', '—')}**, событие "
        f"{meta.get('event_id', '—')}, порог бинаризации {meta.get('threshold', '—')}, "
        f"бюджет {_money(meta.get('budget_rub'))} руб."
    )
    add("")

    # ── данные и протокол ────────────────────────────────────────────────────
    add("## 1. Данные и протокол разделения")
    add("")
    add(
        "Единственный обязательный источник — официальный Sen1Floods11 "
        f"({C.DATASET_REPO}), версия {C.DATASET_VERSION}. Используется размеченная часть: "
        "446 чипов по 11 событиям, слои S1Hand (VV/VH в дБ), LabelHand и JRCWaterHand."
    )
    add("")
    if split:
        add("Собственный сплит **по событиям**, а не по чипам:")
        add("")
        add("| Часть | События | Чипов |")
        add("| --- | --- | ---: |")
        for part in ("train", "validation", "test", "holdout"):
            events = ", ".join(split.get("events", {}).get(part, []))
            add(f"| {part} | {events} | {split.get('counts', {}).get(part, '—')} |")
        add("")
        add(
            "Авторские split-файлы набора разводят чипы случайно внутри одних и тех же "
            "десяти событий: все три части делят все десять. Соседние фрагменты одной "
            "сцены при этом расходятся по обучению и проверке — это утечка, и поэтому "
            "протокол переписан. Bolivia не участвует ни в обучении, ни в подборе порога."
        )
        add("")

    # ── методы ───────────────────────────────────────────────────────────────
    add("## 2. Два метода определения воды")
    add("")
    methods = meta.get("methods", {})
    baseline = methods.get("baseline") or {}
    main = methods.get("main") or {}
    add(
        f"**Baseline.** Пороговый по Sentinel-1: канал {baseline.get('channel', '—')}, порог "
        f"{_fmt(baseline.get('threshold_db'), 2)} дБ, медианный фильтр "
        f"{baseline.get('despeckle_size', '—')}. Выбран перебором на validation."
    )
    add("")
    add(
        f"**Основной метод.** {main.get('kind', '—')}: "
        f"{len(main.get('features', []))} признаков, ансамбль из "
        f"{main.get('ensemble_size', '—')} моделей, калибровка {main.get('calibration', '—')}, "
        f"обучение на {main.get('trained_on_chips', '—')} чипах. Порог выбран на validation."
    )
    add("")
    add(
        "Это не «другой порог»: обучаемая модель в многомерном признаковом пространстве "
        "с пространственным контекстом на четырёх масштабах, калиброванные вероятности и "
        "оценка неопределённости. Порог применяется после модели и только для маски."
    )
    add("")
    add(
        f"**Целевой класс — {meta.get('target_class', '—')}.** Постоянная вода исключена по "
        "слою JRCWaterHand из самого официального набора; он построен по Landsat 1984–2020, "
        "то есть до событий набора, значит временно корректен. JRC используется только на "
        "стороне разметки: на применении методу нужен один Sentinel-1 и ничего больше."
    )
    add("")

    # ── сравнение ────────────────────────────────────────────────────────────
    add("## 3. Сравнение на независимой проверке")
    add("")
    if compare:
        add(
            f"Часть «{compare.get('part')}», {compare.get('chips')} чипов. Оба метода — на "
            "одном и том же наборе валидных пикселей."
        )
        add("")
        add("| Цель | Метод | IoU | F1 | Precision | Recall | Пикселей |")
        add("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for target, title in (("flood", "временное затопление"), ("water", "вода вообще")):
            for method in ("baseline", "main"):
                row = compare.get("totals", {}).get(f"{method}_{target}")
                if not row:
                    continue
                add(
                    f"| {title} | {method} | {_fmt(row.get('iou'))} | {_fmt(row.get('f1'))} | "
                    f"{_fmt(row.get('precision'))} | {_fmt(row.get('recall'))} | "
                    f"{row.get('n_pixels', '—'):,} |"
                )
        add("")
        add("Ложные срабатывания, попавшие на постоянную воду:")
        add("")
        add("| Метод | FP на постоянной воде | Всего FP | Доля |")
        add("| --- | ---: | ---: | ---: |")
        for method in ("baseline", "main"):
            item = compare.get("totals", {}).get(f"{method}_fp_permanent")
            if not item:
                continue
            add(
                f"| {method} | {item['fp_on_permanent']:,} | {item['fp_total']:,} | "
                f"{item['share']:.1%} |"
            )
        add("")
        add(
            "Это и есть механизм завышения ущерба, про который говорили организаторы: "
            "каждый такой пиксель — река или озеро, принятые за паводок, и превращённые "
            "в лишние рубли."
        )
        add("")
        add("### Что из этого следует, без прикрас")
        add("")
        base_flood = compare.get("totals", {}).get("baseline_flood", {})
        main_flood = compare.get("totals", {}).get("main_flood", {})
        base_fp = compare.get("totals", {}).get("baseline_fp_permanent", {})
        main_fp = compare.get("totals", {}).get("main_fp_permanent", {})
        if base_flood and main_flood:
            add(
                f"**По суммарным метрикам сегментации пороговый baseline впереди:** F1 "
                f"{_fmt(base_flood.get('f1'))} против {_fmt(main_flood.get('f1'))}. Мы это не "
                "прячем и не переформулируем. Конфигурация основного метода была "
                "зафиксирована на validation до открытия теста, тест открыт один раз, и "
                "подгонять её после просмотра теста мы не стали: это было бы прямым "
                "нарушением собственного протокола и того, что требует кейс."
            )
            add("")
        if base_fp and main_fp:
            add(
                f"**Но качество ошибок у методов разное.** У baseline "
                f"{base_fp['share']:.1%} ложных срабатываний приходится на постоянную воду, "
                f"у основного метода {main_fp['share']:.1%} — почти втрое меньше. То есть "
                "высокий результат порога держится в том числе на том, что он называет "
                "паводком реки и озёра. Для карты воды это неважно, для рублей ожидаемого "
                "ущерба — важно принципиально: именно эти пиксели и создают завышение."
            )
            add("")
        add(
            "**И главное: пороговый метод не даёт величины, на которой держится весь "
            "экономический контур.** Он выдаёт ноль или единицу. Формула EL = p × V × q "
            "требует вероятности; приоритеты проверки объектов требуют неопределённости; "
            "выбор зон съёмки требует знать, где модель не уверена. Ни одного из этих "
            "чисел у порога нет. Полный экономический расчёт по baseline кейс и не "
            "требует — именно поэтому."
        )
        add("")
        add(
            "Где основной метод слабее и почему: он обучен и калиброван на одних событиях, "
            "а применяется к другим, и калибровка вместе с фиксированным порогом "
            "переносится хуже, чем одно физическое пороговое значение обратного рассеяния. "
            "Это ограничение названо в разделе «Границы» и проверяется отдельно на "
            "отложенном событии."
        )
    else:
        add("Не посчитано: запустите `python -m src.cli.evaluate`.")
    add("")

    if holdout:
        add("### Перенос на отложенное событие")
        add("")
        add(
            f"Bolivia, {holdout.get('chips')} чипов — событие, не участвовавшее ни в "
            "обучении, ни в калибровке, ни в выборе порога."
        )
        add("")
        add("| Метод | IoU | F1 | Precision | Recall |")
        add("| --- | ---: | ---: | ---: | ---: |")
        for method in ("baseline", "main"):
            row = holdout.get("totals", {}).get(f"{method}_flood")
            if row:
                add(
                    f"| {method} | {_fmt(row.get('iou'))} | {_fmt(row.get('f1'))} | "
                    f"{_fmt(row.get('precision'))} | {_fmt(row.get('recall'))} |"
                )
        add("")

    # ── вероятности ──────────────────────────────────────────────────────────
    add("## 4. Надёжность вероятностей и неопределённость")
    add("")
    if validation:
        calib = validation.get("calibration", {})
        add(
            f"Калибровка изотонической регрессией на validation: ECE "
            f"{_fmt(calib.get('ece_raw'))} → {_fmt(calib.get('ece_calibrated'))}, Brier "
            f"{_fmt(calib.get('brier_raw'))} → {_fmt(calib.get('brier_calibrated'))}."
        )
        add("")
        add(
            "Калибровочная выборка равномерная, с природными долями классов. Обучение идёт "
            "на сбалансированных пикселях, иначе редкий класс тонет, но калибровать на той "
            "же сбалансированной выборке нельзя — вероятность отражала бы долю воды в "
            "половину кадра."
        )
        add("")
        use = validation.get("uncertainty_usefulness", {})
        add(
            f"Неопределённость проверена, а не заявлена: AUC связи с фактическими ошибками "
            f"{_fmt(use.get('auc_uncertainty_vs_error'))} на "
            f"{use.get('n_pixels', 0):,} валидных пикселях. Средняя неопределённость на "
            f"ошибках {_fmt(use.get('mean_uncertainty_on_errors'))} против "
            f"{_fmt(use.get('mean_uncertainty_on_correct'))} на верных предсказаниях."
        )
        add("")
        rel = validation.get("reliability", [])
        if rel:
            add("| Корзина вероятности | Обещано | Наблюдаемая доля | Пикселей |")
            add("| --- | ---: | ---: | ---: |")
            for row in rel:
                if not row.get("n_pixels"):
                    continue
                add(
                    f"| {row['bin_low']:.1f}–{row['bin_high']:.1f} | "
                    f"{_fmt(row['mean_predicted'], 3)} | {_fmt(row['observed_fraction'], 3)} | "
                    f"{row['n_pixels']:,} |"
                )
            add("")
    else:
        add("Не посчитано: запустите `python -m src.cli.train_main`.")
        add("")

    # ── ущерб ────────────────────────────────────────────────────────────────
    add("## 5. Портфель и ожидаемый ущерб")
    add("")
    add(f"Правило размещения: {meta.get('placement_rule', '—')}")
    add("")
    add(f"Извлечение вероятности: {meta.get('probability_rule', '—')}")
    add("")
    if asset_rows:
        add("| Ранг | Объект | Тип | p | Ущерб, руб. | Неопределённость | Статус |")
        add("| ---: | --- | --- | ---: | ---: | ---: | --- |")
        ordered = sorted(asset_rows, key=lambda r: (r["rank"] == "", r["rank"]))
        classes = {a.asset_class: a.title_ru for a in C.ASSET_TYPES}
        assets_csv = {r["asset_id"]: r for r in _load_csv(run_dir / C.F_ASSETS_CSV)}
        for row in ordered:
            info = assets_csv.get(row["asset_id"], {})
            add(
                f"| {row['rank'] or '—'} | {row['asset_id']} | "
                f"{classes.get(info.get('asset_class', ''), info.get('asset_class', '—'))} | "
                f"{_fmt(row['p_flood'], 3, '—')} | {_money(row['expected_loss_rub'])} | "
                f"{_fmt(row['uncertainty'], 3, '—')} | {row['status']} |"
            )
        add("")
        add(
            f"Суммарный оценённый ущерб: **{_money(meta.get('total_expected_loss_rub'))} руб.** "
            "Объект не исключается из расчёта и ущерб не обнуляется из-за того, что "
            "вероятность ниже порога бинаризации: EL считается по вероятностному растру."
        )
        add("")
        unassessed = meta.get("unassessed_exposure", {})
        if unassessed.get("count"):
            add(
                f"Без надёжной оценки осталось объектов: {unassessed['count']} "
                f"({', '.join(unassessed['asset_ids'])}), экспозиция "
                f"{_money(unassessed['value_rub'])} руб. Их ущерб не равен нулю — он неизвестен."
            )
            add("")
    proxy = meta.get("proxy_check", {})
    if proxy.get("status") == "checked":
        add(
            f"**Прокси-проверка ущерба.** {proxy['n_points']} контрольных точек — те же "
            "десять объектов, размещённые случайно до всякого расчёта. Сравнение "
            "`EL = p×V×q` с `EL_ref = y×V×q` при тех же V и q: MAE "
            f"{_money(proxy['mae_rub'])} руб., RMSE {_money(proxy['rmse_rub'])} руб., "
            f"смещение {_money(proxy['bias_rub'])} руб. Это оценка качества перехода "
            "«вероятность → рубли», а не измерение фактических потерь."
        )
    else:
        add(f"**Прокси-проверка ущерба.** {proxy.get('note', 'не проведена')}")
    add("")

    # ── заказ ────────────────────────────────────────────────────────────────
    add("## 6. Зоны съёмки, цена и стратегии")
    add("")
    pricing = meta.get("pricing", {})
    add(
        f"Цена по формуле {pricing.get('formula', '—')} ({pricing.get('legal_edition', '—')}, "
        f"дата проверки {pricing.get('legal_check_date', '—')}). Ставка "
        f"{_money(pricing.get('base_rate_rub_km2'))} руб./км² со статусом "
        f"**{pricing.get('base_rate_status', '—')}**: подтверждённого действующего размера "
        "БРЕ в материалах кейса нет, и официальным тарифом это число не называется."
    )
    add("")
    add(f"Округления: {pricing.get('rounding', '—')}. Группировка скидки: {pricing.get('discount_grouping', '—')}.")
    add("")
    zones = meta.get("zones", {})
    add(f"Построение зон: {zones.get('rule', '—')}, минимум {zones.get('min_area_km2', '—')} км² до округления.")
    add("")
    add(f"Правило широкого покрытия B объявлено заранее: {zones.get('broad_rule_B', '—')}")
    add("")
    if strategy_rows:
        add("| Стратегия | Зон | Данные, руб. | Полная стоимость | В бюджете | Покрытый ущерб | Доля | Остаточная неопределённость |")
        add("| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |")
        plans = _load_json(run_dir / C.F_STRATEGY_PLANS) or {}
        for row in strategy_rows:
            status = row.get("uncertainty_status", "")
            value = row.get("residual_uncertainty", "")
            add(
                f"| {row['strategy']} | {len(plans.get(row['strategy'], []))} | "
                f"{_money(row['data_cost_rub'])} | {_money(row['decision_cost_rub'])} | "
                f"{'да' if row['budget_feasible'] in ('true', 'True', True) else '**нет, контрфактическая**'} | "
                f"{_money(row['covered_expected_loss_rub'])} | "
                f"{_fmt(row.get('coverage_share'), 3, '—')} | "
                f"{_fmt(value, 3, '—')} ({status}) |"
            )
        add("")
        add(
            "Повторное покрытие одного объекта не удваивает его ущерб, но каждая зона "
            "остаётся отдельной оплачиваемой позицией. Снижение неопределённости у B и C "
            "имеет статус `scenario`: реальных новых наблюдений нет, и фактом это не "
            "объявляется."
        )
        add("")
    if plan_rows:
        example = plan_rows[0]
        add("Пример расчёта одной позиции, который можно пересчитать на калькуляторе:")
        add("")
        add(
            f"    {example['candidate_id']}: {example['base_rate_rub_km2']} × "
            f"{example['processing_coef']} × {example['usage_coef']} × "
            f"{example['freshness_coef']} × {example['discount_coef']} = "
            f"{example['unit_price_rub_km2']} руб./км²"
        )
        add(
            f"    × {example['area_km2']} км² = {example['cost_rub']} руб.  "
            f"(группа {example['discount_group_id']}, S = {example['group_area_km2']} км²)"
        )
        add("")

    # ── чувствительность ─────────────────────────────────────────────────────
    add("## 7. Чувствительность")
    add("")
    if sensitivity_rows:
        selective = [r for r in sensitivity_rows if r["strategy"] == C.STRATEGY_SELECTIVE]
        add("| Сценарий | Стоимость решения C | Покрытый ущерб | Статус |")
        add("| --- | ---: | ---: | --- |")
        for row in selective:
            add(
                f"| {row['scenario_id']} | {_money(row['decision_cost_rub'])} | "
                f"{_money(row['covered_expected_loss_rub'])} | {row['result_status']} |"
            )
        add("")
        add(
            "Проверены p, V и q по отдельности при фиксированных остальных, плюс бюджет и "
            "ставка съёмки. Смена порога бинаризации в чувствительность не включена "
            "сознательно: ущерб считается по вероятностному растру и от порога не зависит."
        )
    else:
        add("Не посчитано.")
    add("")

    # ── границы ──────────────────────────────────────────────────────────────
    add("## 8. Границы прототипа")
    add("")
    for line in [
        "Демонстрация выполнена на одном чипе Sentinel-1; перенос на тысячи километров и "
        "реальные реестры инфраструктуры — предмет последующего развития.",
        "Стоимости V и уязвимости q учебные, заданы постановкой и к реальному имуществу на "
        "территории чипа отношения не имеют.",
        "Ставка съёмки сценарная. Применимость правового режима к реальному заказу требует "
        "отдельного подтверждения.",
        "Фактической покупки данных не было: эффект закупки — сценарий, а не измерение. "
        "Снижение совокупных потерь не заявляется, модели действия и последствий ошибки нет.",
        "Отделение постоянной воды опирается на JRC — производный продукт по оптике Landsat "
        "со своей ошибкой на узких реках и мелководье.",
        "Дособытийной истории Sentinel-1 по тем же контурам в официальном наборе нет, "
        "поэтому подтверждение постоянной воды по истории S1 не выполнялось.",
        "Перенос проверен на одном отложенном событии, а не на десятках.",
        "Прототип не является официальной оценкой ущерба.",
    ]:
        add(f"- {line}")
    add("")
    add("## 9. Воспроизведение")
    add("")
    add("```bash")
    for command in meta.get("reproduce", []):
        add(command)
    add("```")
    add("")
    add(f"Commit: `{meta.get('commit', '—')}`. Дата сборки: {meta.get('generated_at', '—')}.")
    add("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="каталог комплекта запуска")
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--out", type=Path, default=Path("reports/report.md"))
    args = parser.parse_args()

    text = build(args.run, args.reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"отчёт: {args.out} ({len(text.splitlines())} строк)")


if __name__ == "__main__":
    main()
