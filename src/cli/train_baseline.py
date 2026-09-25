"""Подбор порога baseline на validation и его фиксация.

    python -m src.cli.train_baseline
    python -m src.cli.train_baseline --target water

Перебор идёт по гистограммам: для каждой пары «канал + фильтр» значения раскладываются
в гистограмму отдельно для воды и для суши, после чего метрики для любого порога
получаются из накопленных сумм. Это превращает перебор 300 с лишним конфигураций по
20 миллионам пикселей из получаса в несколько секунд и ничего не приближает — числа
точные с точностью до ширины корзины в 0,05 дБ.

Порог выбирается ТОЛЬКО на validation. Тест здесь не открывается.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.runner import collect_samples
from src.methods.baseline_threshold import (
    BaselineConfig,
    BaselineThreshold,
    channel_values,
    despeckle,
)

BIN_LOW, BIN_HIGH, BIN_STEP = -45.0, 5.0, 0.05


def _accumulate(samples, channel: str, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Гистограммы значений канала отдельно для целевого класса и фона."""
    edges = np.arange(BIN_LOW, BIN_HIGH + BIN_STEP, BIN_STEP)
    hist_pos = np.zeros(len(edges) - 1, dtype=np.int64)
    hist_neg = np.zeros(len(edges) - 1, dtype=np.int64)
    for _, vv, vh, target, valid in samples:
        values = despeckle(channel_values(vv, vh, channel), size)
        usable = valid & np.isfinite(values)
        positive = values[usable & target]
        negative = values[usable & ~target]
        hist_pos += np.histogram(positive, bins=edges)[0]
        hist_neg += np.histogram(negative, bins=edges)[0]
    return hist_pos, hist_neg, edges


def _sweep(hist_pos: np.ndarray, hist_neg: np.ndarray, edges: np.ndarray) -> list[dict]:
    """Метрики для каждого порога: вода — это значения не больше порога."""
    tp = np.cumsum(hist_pos)
    fp = np.cumsum(hist_neg)
    total_pos = hist_pos.sum()
    fn = total_pos - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = np.where(total_pos > 0, tp / max(total_pos, 1), np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
    return [
        {
            "threshold_db": float(edges[index + 1]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "iou": float(iou[index]),
            "tp": int(tp[index]),
            "fp": int(fp[index]),
            "fn": int(fn[index]),
        }
        for index in range(len(tp))
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("flood", "water"), default="flood")
    parser.add_argument("--limit", type=int, default=0, help="ограничить число чипов validation")
    parser.add_argument("--out", type=Path, default=Path("models/baseline.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/baseline_sweep.csv"))
    args = parser.parse_args()

    print(f"целевой класс: {args.target}", flush=True)
    samples = collect_samples("validation", args.target, args.limit or None)
    print(f"чипов validation: {len(samples)}", flush=True)

    best: dict | None = None
    rows: list[dict] = []
    for channel in ("VV", "VH", "VV+VH"):
        for size in (1, 3, 5):
            hist_pos, hist_neg, edges = _accumulate(samples, channel, size)
            for row in _sweep(hist_pos, hist_neg, edges):
                row = {"channel": channel, "despeckle_size": size, **row}
                rows.append(row)
                if not np.isnan(row["f1"]) and (best is None or row["f1"] > best["f1"]):
                    best = row
            print(
                f"  {channel:6s} фильтр {size}: лучший F1 "
                f"{max((r['f1'] for r in rows if r['channel'] == channel and r['despeckle_size'] == size and not np.isnan(r['f1'])), default=float('nan')):.4f}",
                flush=True,
            )

    if best is None:
        raise SystemExit("перебор не дал ни одной конфигурации с определённой метрикой")

    config = BaselineConfig(
        channel=best["channel"],
        threshold_db=best["threshold_db"],
        despeckle_size=best["despeckle_size"],
        selection="grid по validation, метрика F1",
        fitted_on=tuple(chip_id for chip_id, *_ in samples),
    )
    BaselineThreshold(config).save(args.out)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8", newline="") as handle:
        handle.write("channel,despeckle_size,threshold_db,precision,recall,f1,iou,tp,fp,fn\n")
        for row in rows:
            handle.write(
                f"{row['channel']},{row['despeckle_size']},{row['threshold_db']:.2f},"
                f"{row['precision']:.6f},{row['recall']:.6f},{row['f1']:.6f},{row['iou']:.6f},"
                f"{row['tp']},{row['fp']},{row['fn']}\n"
            )

    print(
        "\nвыбрано: канал {channel}, порог {threshold_db:.2f} дБ, фильтр {despeckle_size}\n"
        "на validation: F1 {f1:.4f}, IoU {iou:.4f}, precision {precision:.4f}, recall {recall:.4f}".format(
            **best
        )
    )
    print(f"конфигурация: {args.out}\nтаблица перебора: {args.report}")
    print(json.dumps({"validation_best": best}, ensure_ascii=False))


if __name__ == "__main__":
    main()
