"""Обучение разложенной модели: голова воды и голова постоянной воды.

    python -m src.cli.train_decomposed
    python -m src.cli.train_decomposed --limit-train 70 --limit-valid 30 --trees 200 --ensemble 1

Порядок тот же, что у одноголовой: обучение на train, калибровка произведения на
validation, выбор порога на validation. Тест здесь не открывается.
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
from src.eval.runner import collect_samples, iter_samples_decomposed
from src.methods.decomposed import DecomposedConfig, DecomposedModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-valid", type=int, default=0)
    parser.add_argument("--pixels-per-chip", type=int, default=4000)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("models/main_decomposed"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/metrics_decomposed.json"))
    parser.add_argument("--sweep", type=Path, default=Path("reports/threshold_sweep_decomposed.csv"))
    args = parser.parse_args()

    config = DecomposedConfig(
        n_estimators=args.trees,
        ensemble_size=args.ensemble,
        pixels_per_chip=args.pixels_per_chip,
    )
    model = DecomposedModel(config)

    started = time.time()
    print("обучение двух голов на train…", flush=True)
    model.fit(iter_samples_decomposed("train", args.limit_train or None))
    print(f"  обучено на {len(model.trained_on)} чипах за {time.time() - started:.0f} с", flush=True)

    print("калибровка произведения на validation…", flush=True)
    calibration = model.calibrate(iter_samples_decomposed("validation", args.limit_valid or None))
    print(
        "  ECE {ece_raw:.4f} -> {ece_calibrated:.4f}, Brier {brier_raw:.4f} -> "
        "{brier_calibrated:.4f}".format(**calibration),
        flush=True,
    )

    print("выбор порога на validation…", flush=True)
    samples = collect_samples("validation", "flood", args.limit_valid or None)
    grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    totals = {float(t): Confusion(0, 0, 0, 0) for t in grid}
    probs, targets, uncerts = [], [], []
    for _, vv, vh, target, valid in samples:
        prob, unc = model.predict_chip(vv, vh)
        for threshold in grid:
            key = float(threshold)
            totals[key] = totals[key] + confusion(prob >= threshold, target, valid)
        probs.append(prob[valid])
        targets.append(target[valid])
        uncerts.append(unc[valid])

    sweep = []
    for threshold, matrix in sorted(totals.items()):
        row = metrics_from_confusion(matrix)
        row["threshold"] = threshold
        sweep.append(row)
    best = max((r for r in sweep if not np.isnan(r["f1"])), key=lambda r: r["f1"])
    model.threshold = float(best["threshold"])
    print(
        f"  порог {model.threshold:.2f}: F1 {best['f1']:.4f}, IoU {best['iou']:.4f}, "
        f"precision {best['precision']:.4f}, recall {best['recall']:.4f}",
        flush=True,
    )

    flat_prob = np.concatenate(probs)
    flat_target = np.concatenate(targets)
    flat_unc = np.concatenate(uncerts)
    flat_pred = flat_prob >= model.threshold
    ones = np.ones_like(flat_prob, dtype=bool)
    usefulness = uncertainty_usefulness(flat_unc, flat_pred, flat_target, ones)
    print(
        f"  неопределённость против ошибок: AUC {usefulness['auc_uncertainty_vs_error']:.4f}",
        flush=True,
    )

    model.save(args.out)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(
        json.dumps(
            {
                "kind": "decomposed",
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
                "feature_importance": model.feature_importance(),
                "seconds": round(time.time() - started, 1),
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )
    with args.sweep.open("w", encoding="utf-8", newline="") as handle:
        handle.write("threshold,iou,f1,precision,recall,n_pixels,n_positive,tp,fp,fn\n")
        for row in sweep:
            handle.write(
                f"{row['threshold']:.2f},{row['iou']:.6f},{row['f1']:.6f},"
                f"{row['precision']:.6f},{row['recall']:.6f},{row['n_pixels']},"
                f"{row['n_positive']},{row['tp']},{row['fp']},{row['fn']}\n"
            )

    print(f"\nмодель: {args.out}\nметрики: {args.metrics}")
    print(f"всего {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
