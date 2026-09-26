"""Обучение основного метода и выбор порога — всё на train и validation.

    python -m src.cli.train_main
    python -m src.cli.train_main --limit-train 40 --limit-valid 20   # быстрый прогон

Порядок ровно такой, как требует кейс: обучение на train, калибровка вероятностей на
validation, выбор порога бинаризации на validation. Независимый тест здесь не
открывается — для него есть отдельная команда `python -m src.cli.evaluate`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.eval.metrics import (
    Confusion,
    brier_score,
    confusion,
    expected_calibration_error,
    metrics_from_confusion,
    reliability,
    uncertainty_usefulness,
)
from src.eval.runner import collect_samples, iter_samples
from src.methods.main_model import MainModel, MainModelConfig


def _threshold_sweep_over_chips(
    model: MainModel, samples: list, grid: np.ndarray
) -> list[dict[str, float]]:
    """Перебор порога по всей validation сразу, без хранения растров целиком."""
    totals = {float(t): Confusion(0, 0, 0, 0) for t in grid}
    for _, vv, vh, target, valid in samples:
        prob, _ = model.predict_chip(vv, vh)
        for threshold in grid:
            key = float(threshold)
            totals[key] = totals[key] + confusion(prob >= threshold, target, valid)
    rows = []
    for threshold, matrix in totals.items():
        row = metrics_from_confusion(matrix)
        row["threshold"] = threshold
        rows.append(row)
    return sorted(rows, key=lambda row: row["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("flood", "water"), default="flood")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-valid", type=int, default=0)
    parser.add_argument("--pixels-per-chip", type=int, default=4000)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--positive-share", type=float, default=0.5)
    parser.add_argument("--despeckle", type=int, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=Path("models/main"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics_main_validation.json"))
    parser.add_argument("--sweep", type=Path, default=Path("reports/threshold_sweep.csv"))
    args = parser.parse_args()

    from src.methods.features import FeatureConfig

    config = MainModelConfig(
        n_estimators=args.trees,
        ensemble_size=args.ensemble,
        pixels_per_chip=args.pixels_per_chip,
        positive_share=args.positive_share,
        features=FeatureConfig(despeckle_sizes=tuple(args.despeckle)),
        target_name="временное затопление" if args.target == "flood" else "вода вообще",
    )
    model = MainModel(config)

    started = time.time()
    print(f"целевой класс: {config.target_name}", flush=True)
    print("обучение на train…", flush=True)
    model.fit(iter_samples("train", args.target, args.limit_train or None))
    print(f"  обучено на {len(model.trained_on)} чипах за {time.time() - started:.0f} с", flush=True)

    print("калибровка на validation…", flush=True)
    calibration = model.calibrate(iter_samples("validation", args.target, args.limit_valid or None))
    print(
        "  ECE {ece_raw:.4f} -> {ece_calibrated:.4f}, Brier {brier_raw:.4f} -> "
        "{brier_calibrated:.4f}".format(**calibration),
        flush=True,
    )

    print("выбор порога на validation…", flush=True)
    samples = collect_samples("validation", args.target, args.limit_valid or None)
    grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    sweep = _threshold_sweep_over_chips(model, samples, grid)
    best = max((row for row in sweep if not np.isnan(row["f1"])), key=lambda row: row["f1"])
    model.threshold = float(best["threshold"])
    print(
        f"  порог {model.threshold:.2f}: F1 {best['f1']:.4f}, IoU {best['iou']:.4f}, "
        f"precision {best['precision']:.4f}, recall {best['recall']:.4f}",
        flush=True,
    )

    # Надёжность вероятностей и полезность неопределённости — обе на validation.
    probs, targets, uncert, preds = [], [], [], []
    for _, vv, vh, target, valid in samples:
        prob, unc = model.predict_chip(vv, vh)
        probs.append(prob[valid])
        uncert.append(unc[valid])
        targets.append(target[valid])
        preds.append((prob >= model.threshold)[valid])
    flat_prob = np.concatenate(probs)
    flat_target = np.concatenate(targets)
    flat_unc = np.concatenate(uncert)
    flat_pred = np.concatenate(preds)
    ones = np.ones_like(flat_prob, dtype=bool)

    usefulness = uncertainty_usefulness(flat_unc, flat_pred, flat_target, ones)
    print(
        f"  неопределённость против ошибок: AUC {usefulness['auc_uncertainty_vs_error']:.4f}",
        flush=True,
    )

    model.save(args.out)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target": config.target_name,
        "trained_on_chips": len(model.trained_on),
        "calibrated_on_chips": len(model.calibrated_on),
        "threshold": model.threshold,
        "validation_at_threshold": best,
        "calibration": calibration,
        "reliability": reliability(flat_prob, flat_target, bins=10),
        "ece_validation": expected_calibration_error(flat_prob, flat_target),
        "brier_validation": brier_score(flat_prob, flat_target),
        "uncertainty_usefulness": usefulness,
        "feature_importance_top": model.feature_importance()[:12],
        "seconds": round(time.time() - started, 1),
    }
    args.metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    args.sweep.parent.mkdir(parents=True, exist_ok=True)
    with args.sweep.open("w", encoding="utf-8", newline="") as handle:
        handle.write("threshold,iou,f1,precision,recall,n_pixels,n_positive,tp,fp,fn\n")
        for row in sweep:
            handle.write(
                f"{row['threshold']:.2f},{row['iou']:.6f},{row['f1']:.6f},"
                f"{row['precision']:.6f},{row['recall']:.6f},{row['n_pixels']},"
                f"{row['n_positive']},{row['tp']},{row['fp']},{row['fn']}\n"
            )

    print(f"\nмодель: {args.out}\nметрики: {args.metrics}\nпорог: {args.sweep}")
    print(f"всего {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
