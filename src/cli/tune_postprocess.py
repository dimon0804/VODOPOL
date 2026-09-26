"""E07. Порог и постобработка маски — подбор на validation.

    python -m src.cli.tune_postprocess

Зачем. Разбивка по событиям показала, что основной метод выигрывает у baseline на
Spain и Sri-Lanka, но проигрывает на Nigeria, а микро-среднее по пикселям тянет на
себя именно Nigeria: там больше всего целевых пикселей. Это классическая ловушка
одного усреднённого числа, поэтому дальше смотрим обе величины сразу — микро по
пикселям и макро по событиям.

Что перебираем. Порог бинаризации и минимальный размер связной области: одиночные
тёмные пиксели спекла водой быть не могут, и их отсев должен поднять точность там,
где она провалена. Инференс делается один раз, дальше перебор идёт по сохранённым
вероятностям — иначе каждая комбинация стоила бы десять минут.

Выбор делается ТОЛЬКО на validation. Тест здесь не открывается.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

from src.eval.metrics import Confusion, confusion, metrics_from_confusion
from src.eval.runner import collect_samples
from src.methods.baseline_threshold import BaselineThreshold
from src.methods.main_model import MainModel


def drop_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    """Убирает связные области меньше заданного размера.

    Вода — связное тело. Отдельный тёмный пиксель посреди суши это почти всегда
    спекл или тень, а не затопление; отсев таких областей поднимает точность,
    почти не трогая полноту.
    """
    if min_pixels <= 1:
        return mask
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    small = np.flatnonzero(sizes < min_pixels)
    if small.size == 0:
        return mask
    return mask & ~np.isin(labels, small)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/main"))
    parser.add_argument("--baseline", type=Path, default=Path("models/baseline.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/postprocess_sweep.csv"))
    parser.add_argument("--json", type=Path, default=Path("reports/postprocess_best.json"))
    args = parser.parse_args()

    model = MainModel.load(args.model)
    baseline = BaselineThreshold.load(args.baseline)

    print("инференс по validation…", flush=True)
    cached: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for index, (chip_id, vv, vh, target, valid) in enumerate(
        collect_samples("validation", "flood", args.limit or None), start=1
    ):
        prob, _ = model.predict_chip(vv, vh)
        cached.append((chip_id, prob, target, valid))
        if index % 20 == 0:
            print(f"  {index} чипов", flush=True)
    print(f"  всего {len(cached)} чипов", flush=True)

    thresholds = [round(x, 2) for x in np.arange(0.15, 0.81, 0.05)]
    min_sizes = (1, 4, 9, 16, 30, 60, 120)
    rows: list[dict] = []

    for min_size in min_sizes:
        for threshold in thresholds:
            total = Confusion(0, 0, 0, 0)
            by_event: dict[str, Confusion] = defaultdict(lambda: Confusion(0, 0, 0, 0))
            for chip_id, prob, target, valid in cached:
                mask = drop_small_components(prob >= threshold, min_size)
                matrix = confusion(mask, target, valid)
                total = total + matrix
                event = chip_id.split("_")[0]
                by_event[event] = by_event[event] + matrix
            micro = metrics_from_confusion(total)
            per_event = {e: metrics_from_confusion(c) for e, c in by_event.items()}
            macro_f1 = float(np.mean([m["f1"] for m in per_event.values()]))
            macro_iou = float(np.mean([m["iou"] for m in per_event.values()]))
            rows.append(
                {
                    "min_component_px": min_size,
                    "threshold": threshold,
                    "f1_micro": micro["f1"],
                    "iou_micro": micro["iou"],
                    "precision": micro["precision"],
                    "recall": micro["recall"],
                    "f1_macro": macro_f1,
                    "iou_macro": macro_iou,
                    **{f"f1_{e}": m["f1"] for e, m in per_event.items()},
                }
            )
        best_line = max(
            (r for r in rows if r["min_component_px"] == min_size), key=lambda r: r["f1_micro"]
        )
        print(
            f"  минимальная область {min_size:>3} px: лучший микро-F1 {best_line['f1_micro']:.4f} "
            f"при пороге {best_line['threshold']:.2f}, макро-F1 {best_line['f1_macro']:.4f}",
            flush=True,
        )

    # Baseline на тех же пикселях — точка отсчёта.
    base_total = Confusion(0, 0, 0, 0)
    base_by_event: dict[str, Confusion] = defaultdict(lambda: Confusion(0, 0, 0, 0))
    for (chip_id, _, target, valid), (_, vv, vh, _, _) in zip(
        cached, collect_samples("validation", "flood", args.limit or None)
    ):
        matrix = confusion(baseline.predict_mask(vv, vh), target, valid)
        base_total = base_total + matrix
        base_by_event[chip_id.split("_")[0]] = base_by_event[chip_id.split("_")[0]] + matrix
    base_micro = metrics_from_confusion(base_total)
    base_macro = float(np.mean([metrics_from_confusion(c)["f1"] for c in base_by_event.values()]))

    best = max(rows, key=lambda r: r["f1_micro"])
    best_macro = max(rows, key=lambda r: r["f1_macro"])

    print()
    print(f"baseline:              микро-F1 {base_micro['f1']:.4f}  макро-F1 {base_macro:.4f}")
    print(
        f"лучшее по микро-F1:    {best['f1_micro']:.4f}  (порог {best['threshold']:.2f}, "
        f"область {best['min_component_px']} px, макро {best['f1_macro']:.4f})"
    )
    print(
        f"лучшее по макро-F1:    {best_macro['f1_macro']:.4f}  (порог {best_macro['threshold']:.2f}, "
        f"область {best_macro['min_component_px']} px, микро {best_macro['f1_micro']:.4f})"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in columns) + "\n")
    args.json.write_text(
        json.dumps(
            {
                "baseline": {"f1_micro": base_micro["f1"], "f1_macro": base_macro},
                "best_by_micro": best,
                "best_by_macro": best_macro,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nтаблица: {args.out}")


if __name__ == "__main__":
    main()
