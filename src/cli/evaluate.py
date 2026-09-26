"""Независимая проверка: baseline против основного метода на одних пикселях.

    python -m src.cli.evaluate                 # test (Mekong, Pakistan, Somalia)
    python -m src.cli.evaluate --part holdout  # перенос на Bolivia

Это единственное место, где открывается тест. Ни порог, ни калибровка, ни
конфигурация здесь не подбираются: всё приходит зафиксированным с validation.

Оба метода считаются на ОДНОМ И ТОМ ЖЕ наборе валидных пикселей и с одной и той же
целью — иначе сравнение недействительно. Метрики выводятся в двух разрезах:
«временное затопление» (наш целевой класс) и «вода вообще» (чтобы числа можно было
сопоставлять с литературой и чтобы видно было, какой ценой даётся отделение
постоянной воды).

Отдельно считается то, что обычно остаётся за кадром: какая доля ложных
срабатываний каждого метода приходится на постоянную воду. Именно там пороговый
метод systematically завышает ущерб.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data import chips as chips_mod
from src.eval.metrics import (
    Confusion,
    brier_score,
    confusion,
    expected_calibration_error,
    metrics_from_confusion,
    reliability,
    uncertainty_usefulness,
)
from src.eval.runner import part_chip_ids
from src.methods.baseline_threshold import BaselineThreshold
from src.methods.main_model import MainModel


def _clean(value):
    """Заменяет NaN и Inf на None по всей структуре.

    Постановка прямо запрещает NaN и Inf в JSON, и это не придирка: такой литерал
    невалиден по RFC 8259, и строгий парсер у жюри на нём упадёт. NaN здесь берётся
    из пустых корзин диаграммы надёжности — в них просто нет пикселей, и правильное
    значение там именно «нет данных», а не ноль.
    """
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fp_on_permanent(pred: np.ndarray, target: np.ndarray, permanent: np.ndarray, valid: np.ndarray) -> tuple[int, int]:
    """Сколько ложных срабатываний метода приходится на постоянную воду."""
    false_positive = valid & pred & ~target
    return int(np.count_nonzero(false_positive & permanent)), int(np.count_nonzero(false_positive))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", default="test", choices=("test", "holdout", "validation"))
    parser.add_argument("--model", type=Path, default=Path("models/main"))
    parser.add_argument("--baseline", type=Path, default=Path("models/baseline.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, default=Path("reports/metrics_compare.csv"))
    parser.add_argument("--out-md", type=Path, default=Path("reports/comparison.md"))
    parser.add_argument("--out-json", type=Path, default=Path("reports/metrics_compare.json"))
    args = parser.parse_args()

    baseline = BaselineThreshold.load(args.baseline)
    model = MainModel.load(args.model)
    if model.threshold is None:
        raise SystemExit("у основного метода не зафиксирован порог")

    chip_ids = part_chip_ids(args.part)
    if args.limit:
        chip_ids = chip_ids[: args.limit]
    print(f"часть: {args.part}, чипов: {len(chip_ids)}", flush=True)
    print(
        f"baseline: {baseline.config.channel} <= {baseline.config.threshold_db:.2f} дБ, "
        f"фильтр {baseline.config.despeckle_size}; основной метод: порог {model.threshold:.2f}",
        flush=True,
    )

    totals: dict[tuple[str, str], Confusion] = defaultdict(lambda: Confusion(0, 0, 0, 0))
    by_event: dict[tuple[str, str, str], Confusion] = defaultdict(lambda: Confusion(0, 0, 0, 0))
    fp_permanent = defaultdict(int)
    fp_total = defaultdict(int)
    per_chip_rows: list[dict] = []
    # Вероятности, неопределённость и метки копим подвыборкой: полные растры по всей
    # части не влезут в память, а для калибровочных метрик хватает равномерной выборки.
    rng = np.random.default_rng(2026)
    prob_sample: list[np.ndarray] = []
    target_sample: list[np.ndarray] = []
    unc_sample: list[np.ndarray] = []
    pred_sample: list[np.ndarray] = []

    for index, chip_id in enumerate(chip_ids, start=1):
        chip = chips_mod.load_chip(chip_id)
        valid = chips_mod.valid_mask(chip)
        if not valid.any():
            continue
        event = chips_mod.event_of(chip_id)
        permanent = chips_mod.permanent_water(chip)

        pred_baseline = baseline.predict_mask(chip.vv, chip.vh)
        prob, uncertainty = model.predict_chip(chip.vv, chip.vh)
        pred_main = prob >= model.threshold

        flood_target = chips_mod.target_flood(chip)
        flat = np.flatnonzero(valid.ravel())
        take = rng.choice(flat, size=min(20_000, flat.size), replace=False)
        prob_sample.append(prob.ravel()[take])
        target_sample.append(flood_target.ravel()[take].astype(np.float64))
        unc_sample.append(uncertainty.ravel()[take])
        pred_sample.append(pred_main.ravel()[take])

        for target_name, target in (
            ("flood", chips_mod.target_flood(chip)),
            ("water", chips_mod.target_water(chip)),
        ):
            for method_name, pred in (("baseline", pred_baseline), ("main", pred_main)):
                matrix = confusion(pred, target, valid)
                totals[(method_name, target_name)] = totals[(method_name, target_name)] + matrix
                by_event[(method_name, target_name, event)] = (
                    by_event[(method_name, target_name, event)] + matrix
                )
                if target_name == "flood":
                    row = metrics_from_confusion(matrix)
                    per_chip_rows.append(
                        {"chip_id": chip_id, "event": event, "method": method_name, **row}
                    )

        for method_name, pred in (("baseline", pred_baseline), ("main", pred_main)):
            on_permanent, total = _fp_on_permanent(
                pred, chips_mod.target_flood(chip), permanent, valid
            )
            fp_permanent[method_name] += on_permanent
            fp_total[method_name] += total

        if index % 10 == 0:
            print(f"  обработано {index}/{len(chip_ids)}", flush=True)

    # ── надёжность вероятностей на НЕЗАВИСИМОЙ части ─────────────────────────
    flat_prob = np.concatenate(prob_sample)
    flat_target = np.concatenate(target_sample)
    flat_unc = np.concatenate(unc_sample)
    flat_pred = np.concatenate(pred_sample)
    ones = np.ones_like(flat_prob, dtype=bool)
    calibration_here = {
        "note": (
            "Посчитано на части, не использованной для обучения калибратора: "
            "это и есть проверка надёжности вероятностей, а не их подгонка."
        ),
        "n_pixels": int(flat_prob.size),
        "ece": expected_calibration_error(flat_prob, flat_target),
        "brier": brier_score(flat_prob, flat_target),
        "reliability": reliability(flat_prob, flat_target, bins=10),
    }
    uncertainty_here = uncertainty_usefulness(flat_unc, flat_pred, flat_target > 0.5, ones)
    print(
        f"\n  надёжность вероятностей на этой части: ECE {calibration_here['ece']:.4f}, "
        f"Brier {calibration_here['brier']:.4f}"
    )
    print(
        f"  неопределённость против ошибок: AUC "
        f"{uncertainty_here['auc_uncertainty_vs_error']:.4f} на "
        f"{uncertainty_here['n_pixels']:,} пикселях"
    )

    # ── вывод ────────────────────────────────────────────────────────────────
    summary: dict[str, dict] = {}
    print("\nна одних и тех же валидных пикселях:")
    for target_name in ("flood", "water"):
        title = "временное затопление" if target_name == "flood" else "вода вообще"
        print(f"\n  цель — {title}:")
        for method_name in ("baseline", "main"):
            metrics = metrics_from_confusion(totals[(method_name, target_name)])
            summary[f"{method_name}_{target_name}"] = metrics
            print(
                f"    {method_name:9s} IoU {metrics['iou']:.4f}  F1 {metrics['f1']:.4f}  "
                f"P {metrics['precision']:.4f}  R {metrics['recall']:.4f}  "
                f"пикселей {metrics['n_pixels']:,}  из них цель {metrics['n_positive']:,}"
            )

    print("\n  ложные срабатывания на постоянной воде:")
    for method_name in ("baseline", "main"):
        total = fp_total[method_name]
        share = fp_permanent[method_name] / total if total else float("nan")
        summary[f"{method_name}_fp_permanent"] = {
            "fp_on_permanent": fp_permanent[method_name],
            "fp_total": total,
            "share": share,
        }
        print(
            f"    {method_name:9s} {fp_permanent[method_name]:,} из {total:,} "
            f"({share:.1%}) — это и есть завышение ущерба за счёт рек и озёр"
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write("scope,method,target,event,iou,f1,precision,recall,n_pixels,n_positive,tp,fp,fn,tn\n")
        for (method_name, target_name), matrix in sorted(totals.items()):
            m = metrics_from_confusion(matrix)
            handle.write(
                f"total,{method_name},{target_name},,{m['iou']:.6f},{m['f1']:.6f},"
                f"{m['precision']:.6f},{m['recall']:.6f},{m['n_pixels']},{m['n_positive']},"
                f"{m['tp']},{m['fp']},{m['fn']},{m['tn']}\n"
            )
        for (method_name, target_name, event), matrix in sorted(by_event.items()):
            m = metrics_from_confusion(matrix)
            handle.write(
                f"event,{method_name},{target_name},{event},{m['iou']:.6f},{m['f1']:.6f},"
                f"{m['precision']:.6f},{m['recall']:.6f},{m['n_pixels']},{m['n_positive']},"
                f"{m['tp']},{m['fp']},{m['fn']},{m['tn']}\n"
            )

    args.out_json.write_text(
        json.dumps(
            _clean(
            {
                "part": args.part,
                "chips": len(chip_ids),
                "baseline": {
                    "channel": baseline.config.channel,
                    "threshold_db": baseline.config.threshold_db,
                    "despeckle_size": baseline.config.despeckle_size,
                },
                "main_threshold": model.threshold,
                "calibration_on_this_part": calibration_here,
                "uncertainty_on_this_part": uncertainty_here,
                "totals": summary,
                "by_event": {
                    f"{m}|{t}|{e}": metrics_from_confusion(c) for (m, t, e), c in by_event.items()
                },
            },
            ),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    _write_markdown(args.out_md, args.part, chip_ids, baseline, model, totals, by_event, summary)
    print(f"\nтаблица: {args.out_csv}\nотчёт: {args.out_md}")


def _write_markdown(path, part, chip_ids, baseline, model, totals, by_event, summary) -> None:
    events = sorted({event for (_, _, event) in by_event})
    lines = [
        f"# Сравнение методов на части «{part}»",
        "",
        f"Чипов: {len(chip_ids)}. События: {', '.join(events)}.",
        "",
        f"Baseline: канал {baseline.config.channel}, порог {baseline.config.threshold_db:.2f} дБ, "
        f"медианный фильтр {baseline.config.despeckle_size}. Выбран на validation.",
        f"Основной метод: ансамбль LightGBM по признакам S1, порог {model.threshold:.2f}, "
        "выбран на validation. К этой части оба применены без изменений.",
        "",
        "Оба метода считаются на одном и том же наборе валидных пикселей: метка не равна −1 "
        "и оба канала S1 конечны.",
        "",
        "## Итог по всем пикселям части",
        "",
        "| Цель | Метод | IoU | F1 | Precision | Recall | Пикселей | Из них цель |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target_name, title in (("flood", "временное затопление"), ("water", "вода вообще")):
        for method_name in ("baseline", "main"):
            m = metrics_from_confusion(totals[(method_name, target_name)])
            lines.append(
                f"| {title} | {method_name} | {m['iou']:.4f} | {m['f1']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['n_pixels']:,} | {m['n_positive']:,} |"
            )

    lines += [
        "",
        "## Ложные срабатывания на постоянной воде",
        "",
        "| Метод | FP на постоянной воде | Всего FP | Доля |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method_name in ("baseline", "main"):
        item = summary[f"{method_name}_fp_permanent"]
        lines.append(
            f"| {method_name} | {item['fp_on_permanent']:,} | {item['fp_total']:,} | "
            f"{item['share']:.1%} |"
        )
    lines += [
        "",
        "Это прямая мера того, о чём говорили организаторы: порог принимает реки и озёра "
        "за паводок, и каждый такой пиксель превращается в лишние рубли ожидаемого ущерба.",
        "",
        "## По событиям (цель — временное затопление)",
        "",
        "| Событие | Метод | IoU | F1 | Precision | Recall | Пикселей |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for event in events:
        for method_name in ("baseline", "main"):
            key = (method_name, "flood", event)
            if key not in by_event:
                continue
            m = metrics_from_confusion(by_event[key])
            lines.append(
                f"| {event} | {method_name} | {m['iou']:.4f} | {m['f1']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['n_pixels']:,} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
