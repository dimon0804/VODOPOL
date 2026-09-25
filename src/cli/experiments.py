"""Контролируемые эксперименты, обосновывающие конфигурацию основного метода.

    python -m src.cli.experiments --run E02
    python -m src.cli.experiments --run all --limit-train 60 --limit-valid 30

Критерий 8 оценивает обоснованность, а не число опытов. Поэтому правила здесь жёсткие:
в каждом эксперименте меняется РОВНО ОДНА вещь, остальное фиксируется, сравнение идёт
на одной и той же валидационной выборке, и результат пишется в журнал независимо от
того, подтвердилась гипотеза или нет.

Тест в экспериментах не участвует вообще.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.eval.metrics import Confusion, confusion, metrics_from_confusion
from src.eval.runner import collect_samples, iter_samples
from src.methods.features import FeatureConfig
from src.methods.main_model import MainModel, MainModelConfig

JOURNAL = Path("reports/experiments.md")


def _evaluate(model: MainModel, samples, thresholds=np.arange(0.15, 0.81, 0.05)) -> dict:
    """Лучшая по F1 точка на валидации — и есть результат конфигурации."""
    totals = {float(t): Confusion(0, 0, 0, 0) for t in thresholds}
    for _, vv, vh, target, valid in samples:
        prob, _ = model.predict_chip(vv, vh)
        for threshold in thresholds:
            key = float(threshold)
            totals[key] = totals[key] + confusion(prob >= threshold, target, valid)
    rows = []
    for threshold, matrix in totals.items():
        row = metrics_from_confusion(matrix)
        row["threshold"] = threshold
        rows.append(row)
    best = max((r for r in rows if not np.isnan(r["f1"])), key=lambda r: r["f1"])
    return best


def _train(config: MainModelConfig, target: str, limit_train: int, limit_valid: int) -> MainModel:
    model = MainModel(config)
    model.fit(iter_samples("train", target, limit_train or None))
    model.calibrate(iter_samples("validation", target, limit_valid or None))
    return model


def experiment_e02(args, samples) -> dict:
    """E02. Дают ли контекстные признаки прирост против только VV и VH.

    Это главное обоснование метода: если контекст ничего не даёт, метод сводится к
    сложному порогу и терять на него время незачем.
    """
    base = MainModelConfig(
        n_estimators=args.trees,
        ensemble_size=1,
        pixels_per_chip=args.pixels,
        target_name="временное затопление",
    )
    variants = {
        "только VV и VH": replace(base, features=FeatureConfig(scales=(), use_chip_contrast=False)),
        "контекст 3, 9": replace(base, features=FeatureConfig(scales=(3, 9), use_chip_contrast=False)),
        "контекст 3, 9, 25, 51": replace(base, features=FeatureConfig(scales=(3, 9, 25, 51), use_chip_contrast=False)),
        "контекст + контраст чипа": replace(base, features=FeatureConfig(scales=(3, 9, 25, 51), use_chip_contrast=True)),
    }
    results = {}
    for name, config in variants.items():
        started = time.time()
        model = _train(config, args.target, args.limit_train, args.limit_valid)
        best = _evaluate(model, samples)
        best["seconds"] = round(time.time() - started, 1)
        best["n_features"] = len(model.feature_names)
        results[name] = best
        print(
            f"  {name:26s} F1 {best['f1']:.4f}  IoU {best['iou']:.4f}  "
            f"порог {best['threshold']:.2f}  признаков {best['n_features']}  "
            f"{best['seconds']:.0f} с",
            flush=True,
        )
    return results


def experiment_e02b(args, samples) -> dict:
    """E02b. Нормировка на статистику чипа и абсолютные константы сцены.

    Уровень обратного рассеяния зависит от сцены и угла съёмки, поэтому абсолютные
    децибелы плохо переносятся между событиями. Проверяем три вещи: вредят ли
    абсолютные константы чипа, помогает ли приведение каналов к собственной статистике
    чипа и даёт ли что-то медианная фильтрация, от которой заметно выигрывает baseline.
    """
    base = MainModelConfig(
        n_estimators=args.trees,
        ensemble_size=1,
        pixels_per_chip=args.pixels,
        target_name="временное затопление",
    )
    full = FeatureConfig(scales=(3, 9, 25, 51), use_chip_contrast=True)
    variants = {
        "базовая конфигурация": replace(base, features=full),
        "с абсолютными константами чипа": replace(
            base, features=replace(full, use_chip_absolute=True)
        ),
        "нормировка на статистику чипа": replace(
            base, features=replace(full, per_chip_normalize=True)
        ),
        "нормировка + медианный фильтр 5": replace(
            base, features=replace(full, per_chip_normalize=True, despeckle_sizes=(5,))
        ),
    }
    results = {}
    for name, config in variants.items():
        started = time.time()
        model = _train(config, args.target, args.limit_train, args.limit_valid)
        best = _evaluate(model, samples)
        best["seconds"] = round(time.time() - started, 1)
        best["n_features"] = len(model.feature_names)
        results[name] = best
        print(
            f"  {name:32s} F1 {best['f1']:.4f}  IoU {best['iou']:.4f}  "
            f"порог {best['threshold']:.2f}  признаков {best['n_features']}  "
            f"{best['seconds']:.0f} с",
            flush=True,
        )
    return results


def experiment_e02c(args, samples) -> dict:
    """E02c. Объём обучения и вклад отдельных событий.

    Наблюдение, ради которого поставлен опыт: модель, обученная на 70 чипах двух
    событий, на валидации оказалась заметно лучше модели, обученной на всех 257 чипах
    четырёх событий. Гипотеза — крупные события перетягивают выборку на себя: USA и
    Paraguay дают 136 чипов, больше половины обучающего материала.

    Все варианты учатся на ОДНОЙ и той же выборке чипов, меняется только правило
    отбора пикселей.
    """
    base = MainModelConfig(
        n_estimators=args.trees,
        ensemble_size=1,
        pixels_per_chip=args.pixels,
        target_name="временное затопление",
    )
    variants = {
        "выборка по чипам": replace(base, event_balanced=False),
        "выборка, выровненная по событиям": replace(base, event_balanced=True),
    }
    results = {}
    for name, config in variants.items():
        started = time.time()
        model = MainModel(config)
        model.fit(iter_samples("train", args.target, None))
        model.calibrate(iter_samples("validation", args.target, args.limit_valid or None))
        best = _evaluate(model, samples)
        best["seconds"] = round(time.time() - started, 1)
        best["n_features"] = len(model.feature_names)
        results[name] = best
        print(
            f"  {name:34s} F1 {best['f1']:.4f}  IoU {best['iou']:.4f}  "
            f"порог {best['threshold']:.2f}  чипов {len(model.trained_on)}  "
            f"{best['seconds']:.0f} с",
            flush=True,
        )
    return results


def experiment_e03(args, samples) -> dict:
    """E03. Цена отделения постоянной воды.

    Одна и та же архитектура учится на двух целях: «вода вообще» и «временное
    затопление». Сравнение показывает, насколько задача действительно труднее, и не
    позволяет выдать метрики одной цели за метрики другой.
    """
    config = MainModelConfig(
        n_estimators=args.trees, ensemble_size=1, pixels_per_chip=args.pixels
    )
    results = {}
    for target, title in (("water", "вода вообще"), ("flood", "временное затопление")):
        model = _train(config, target, args.limit_train, args.limit_valid)
        part = collect_samples("validation", target, args.limit_valid or None)
        best = _evaluate(model, part)
        results[title] = best
        print(
            f"  {title:26s} F1 {best['f1']:.4f}  IoU {best['iou']:.4f}  "
            f"доля цели {best['n_positive'] / max(best['n_pixels'], 1):.3%}",
            flush=True,
        )
    return results


def experiment_e05(args, samples) -> dict:
    """E05. Из чего полезнее собирать неопределённость.

    Сравниваются энтропия одной модели и разброс ансамбля: что из этого сильнее
    связано с реальными ошибками. Связь меряется AUC, а не утверждается.
    """
    from src.eval.metrics import roc_auc

    config = MainModelConfig(
        n_estimators=args.trees, ensemble_size=3, pixels_per_chip=args.pixels
    )
    model = _train(config, args.target, args.limit_train, args.limit_valid)
    best = _evaluate(model, samples)
    threshold = best["threshold"]

    entropy_parts, spread_parts, error_parts = [], [], []
    for _, vv, vh, target, valid in samples:
        from src.methods.features import build_features

        cube = build_features(vv, vh, model.config.features)
        flat = cube.reshape(-1, cube.shape[-1])
        stack = model._raw_stack(flat)
        raw = stack.mean(axis=0)
        spread = stack.std(axis=0)
        prob = np.clip(model.calibrator.predict(raw), 1e-6, 1 - 1e-6)
        entropy = -(prob * np.log2(prob) + (1 - prob) * np.log2(1 - prob))
        error = (prob.reshape(target.shape) >= threshold) != target
        flat_valid = valid.ravel()
        entropy_parts.append(entropy[flat_valid])
        spread_parts.append(spread[flat_valid])
        error_parts.append(error.ravel()[flat_valid])

    errors = np.concatenate(error_parts)
    results = {
        "энтропия вероятности": {"auc": roc_auc(np.concatenate(entropy_parts), errors)},
        "разброс ансамбля": {"auc": roc_auc(np.concatenate(spread_parts), errors)},
        "комбинация 0,7 / 0,3": {
            "auc": roc_auc(
                0.7 * np.concatenate(entropy_parts) + 0.3 * (np.concatenate(spread_parts) / 0.5),
                errors,
            )
        },
        "n_pixels": int(errors.size),
        "n_errors": int(errors.sum()),
    }
    for name in ("энтропия вероятности", "разброс ансамбля", "комбинация 0,7 / 0,3"):
        print(f"  {name:26s} AUC {results[name]['auc']:.4f}", flush=True)
    return results


EXPERIMENTS = {
    "E02": ("Контекстные признаки против только VV и VH", experiment_e02),
    "E02b": ("Нормировка на статистику чипа против абсолютных децибел", experiment_e02b),
    "E02c": ("Выравнивание вклада событий в обучающей выборке", experiment_e02c),
    "E03": ("Цена отделения постоянной воды", experiment_e03),
    "E05": ("Из чего собирать неопределённость", experiment_e05),
}


def append_journal(key: str, title: str, args, results: dict) -> None:
    """Дописывает запись в журнал экспериментов — включая неудачные."""
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    if not JOURNAL.exists():
        JOURNAL.write_text(
            "# Журнал экспериментов\n\nВсе конфигурации выбираются на validation. "
            "Тест в экспериментах не участвует.\n",
            encoding="utf-8",
        )
    lines = [
        "",
        f"## {key}. {title}",
        "",
        f"Выборка: train {args.limit_train or 'вся'} чипов, validation "
        f"{args.limit_valid or 'вся'} чипов; пикселей на чип {args.pixels}; "
        f"деревьев {args.trees}; цель — {args.target}.",
        "",
        "| Вариант | F1 | IoU | Precision | Recall | Порог | Прочее |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, row in results.items():
        if not isinstance(row, dict) or "f1" not in row:
            continue
        extra = []
        if "n_features" in row:
            extra.append(f"признаков {row['n_features']}")
        if "seconds" in row:
            extra.append(f"{row['seconds']:.0f} с")
        lines.append(
            f"| {name} | {row['f1']:.4f} | {row['iou']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['threshold']:.2f} | {', '.join(extra)} |"
        )
    auc_rows = {k: v for k, v in results.items() if isinstance(v, dict) and "auc" in v}
    if auc_rows:
        lines += ["", "| Показатель | AUC связи с ошибками |", "| --- | ---: |"]
        lines += [f"| {k} | {v['auc']:.4f} |" for k, v in auc_rows.items()]
    lines.append("")
    with JOURNAL.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="E02", help="E02, E03, E05 или all")
    parser.add_argument("--target", choices=("flood", "water"), default="flood")
    parser.add_argument("--limit-train", type=int, default=60)
    parser.add_argument("--limit-valid", type=int, default=30)
    parser.add_argument("--pixels", type=int, default=3000)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("reports/experiments.json"))
    args = parser.parse_args()

    keys = list(EXPERIMENTS) if args.run == "all" else [args.run]
    samples = collect_samples("validation", args.target, args.limit_valid or None)
    print(f"валидационных чипов: {len(samples)}", flush=True)

    payload = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    for key in keys:
        title, func = EXPERIMENTS[key]
        print(f"\n{key}. {title}", flush=True)
        results = func(args, samples)
        payload[key] = {"title": title, "results": results}
        append_journal(key, title, args, results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\nжурнал: {JOURNAL}\nсырые числа: {args.out}")


if __name__ == "__main__":
    main()
