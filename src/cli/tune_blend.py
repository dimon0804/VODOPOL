"""E11. Смесь основного метода с пороговым: помогает ли порог как второй источник.

    python -m src.cli.tune_blend

Зачем. Методы ошибаются по-разному, и это измерено: у порогового baseline 36 процентов
ложных срабатываний приходится на постоянную воду, у основного метода 15. Порог хорошо
находит воду вообще и плохо отличает реку от разлива; модель наоборот. Когда ошибки
двух источников не совпадают, их смесь часто оказывается лучше каждого по отдельности.

Смесь простая и объяснимая:

    p = a × p_модели + (1 − a) × s_порога,
    s_порога = сигмоида((порог_дБ − VV_фильтрованный) / масштаб)

Мягкая версия порога нужна потому, что складывать вероятность с нулём или единицей
бессмысленно: вся информация о том, насколько пиксель темнее порога, при бинаризации
теряется. Масштаб в децибелах задаёт, насколько резким остаётся переход.

Вес и порог бинаризации подбираются на validation. Тест здесь не открывается.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.eval.metrics import Confusion, confusion, metrics_from_confusion
from src.eval.runner import collect_samples
from src.methods.baseline_threshold import BaselineThreshold, channel_values, despeckle
from src.methods.main_model import MainModel, drop_small_components_prob


def baseline_soft(vv: np.ndarray, vh: np.ndarray, baseline: BaselineThreshold, scale: float) -> np.ndarray:
    """Мягкий выход порогового метода: насколько пиксель темнее порога, в долях."""
    values = despeckle(
        channel_values(vv, vh, baseline.config.channel), baseline.config.despeckle_size
    )
    z = (baseline.config.threshold_db - values) / max(scale, 1e-6)
    return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/main"))
    parser.add_argument("--baseline", type=Path, default=Path("models/baseline.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scale-db", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("reports/blend_sweep.csv"))
    parser.add_argument("--json", type=Path, default=Path("reports/blend_best.json"))
    args = parser.parse_args()

    model = MainModel.load(args.model)
    baseline = BaselineThreshold.load(args.baseline)
    min_component = model.config.min_component_px
    # Кэшируем сырую вероятность: постобработку применяем уже к смеси.
    model.config.min_component_px = 0

    print("инференс по validation…", flush=True)
    cached: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for index, (chip_id, vv, vh, target, valid) in enumerate(
        collect_samples("validation", "flood", args.limit or None), start=1
    ):
        prob, _ = model.predict_chip(vv, vh)
        soft = baseline_soft(vv, vh, baseline, args.scale_db)
        cached.append((chip_id, prob, soft, target, valid))
        if index % 20 == 0:
            print(f"  {index} чипов", flush=True)
    print(f"  всего {len(cached)} чипов", flush=True)

    alphas = [round(a, 2) for a in np.arange(0.0, 1.01, 0.1)]
    thresholds = [round(t, 2) for t in np.arange(0.10, 0.86, 0.05)]
    rows: list[dict] = []

    for alpha in alphas:
        for threshold in thresholds:
            total = Confusion(0, 0, 0, 0)
            by_event: dict[str, Confusion] = defaultdict(lambda: Confusion(0, 0, 0, 0))
            for chip_id, prob, soft, target, valid in cached:
                blended = alpha * prob + (1 - alpha) * soft
                if min_component > 1:
                    blended = drop_small_components_prob(blended, threshold, min_component)
                matrix = confusion(blended >= threshold, target, valid)
                total = total + matrix
                by_event[chip_id.split("_")[0]] = by_event[chip_id.split("_")[0]] + matrix
            micro = metrics_from_confusion(total)
            per_event = [metrics_from_confusion(c) for c in by_event.values()]
            rows.append(
                {
                    "alpha": alpha,
                    "threshold": threshold,
                    "f1_micro": micro["f1"],
                    "iou_micro": micro["iou"],
                    "precision": micro["precision"],
                    "recall": micro["recall"],
                    "f1_macro": float(np.mean([m["f1"] for m in per_event])),
                    "iou_macro": float(np.mean([m["iou"] for m in per_event])),
                }
            )
        best_for_alpha = max(
            (r for r in rows if r["alpha"] == alpha), key=lambda r: r["f1_macro"]
        )
        label = "только порог" if alpha == 0 else ("только модель" if alpha == 1 else f"смесь {alpha}")
        print(
            f"  {label:16s} макро-F1 {best_for_alpha['f1_macro']:.4f}  "
            f"микро-F1 {best_for_alpha['f1_micro']:.4f}  при пороге {best_for_alpha['threshold']:.2f}",
            flush=True,
        )

    best_macro = max(rows, key=lambda r: r["f1_macro"])
    best_micro = max(rows, key=lambda r: r["f1_micro"])
    only_model = max((r for r in rows if r["alpha"] == 1.0), key=lambda r: r["f1_macro"])
    only_base = max((r for r in rows if r["alpha"] == 0.0), key=lambda r: r["f1_macro"])

    print()
    print(
        f"только модель:  макро {only_model['f1_macro']:.4f}  микро {only_model['f1_micro']:.4f}"
    )
    print(
        f"только порог:   макро {only_base['f1_macro']:.4f}  микро {only_base['f1_micro']:.4f}"
    )
    print(
        f"лучшая смесь:   макро {best_macro['f1_macro']:.4f}  микро {best_macro['f1_micro']:.4f}"
        f"  (вес модели {best_macro['alpha']}, порог {best_macro['threshold']:.2f})"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(
                ",".join(f"{row[c]:.6f}" if isinstance(row[c], float) else str(row[c]) for c in columns)
                + "\n"
            )
    args.json.write_text(
        json.dumps(
            {
                "scale_db": args.scale_db,
                "min_component_px": min_component,
                "only_model": only_model,
                "only_baseline": only_base,
                "best_by_macro": best_macro,
                "best_by_micro": best_micro,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nтаблица: {args.out}")


if __name__ == "__main__":
    main()
