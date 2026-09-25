"""Метрики сегментации, калибровки и полезности неопределённости.

Главное правило кейса: оба метода сравниваются на ОДНОМ И ТОМ ЖЕ наборе валидных
пикселей. Поэтому маска валидности приходит извне и применяется одинаково ко всем
сравниваемым предсказаниям, а число учтённых пикселей всегда попадает в отчёт —
без него сравнение недействительно.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class Confusion:
    """Матрица ошибок по валидным пикселям."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n_pixels(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def n_positive(self) -> int:
        return self.tp + self.fn

    def __add__(self, other: "Confusion") -> "Confusion":
        return Confusion(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )


def confusion(pred: np.ndarray, target: np.ndarray, valid: np.ndarray) -> Confusion:
    """Считает матрицу ошибок только по валидным пикселям."""
    if pred.shape != target.shape or pred.shape != valid.shape:
        raise ValueError("формы предсказания, цели и маски валидности должны совпадать")
    p = np.asarray(pred, dtype=bool)[valid]
    t = np.asarray(target, dtype=bool)[valid]
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(~p & ~t))
    return Confusion(tp, fp, fn, tn)


def metrics_from_confusion(matrix: Confusion) -> dict[str, float | int]:
    """IoU, F1/Dice, Precision, Recall плюс число пикселей.

    При пустом знаменателе метрика остаётся неопределённой (NaN в расчётах,
    пустое значение при записи) — ноль здесь был бы неправдой.
    """
    tp, fp, fn = matrix.tp, matrix.fp, matrix.fn
    union = tp + fp + fn
    return {
        "iou": tp / union if union else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "accuracy": (tp + matrix.tn) / matrix.n_pixels if matrix.n_pixels else float("nan"),
        "n_pixels": matrix.n_pixels,
        "n_positive": matrix.n_positive,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": matrix.tn,
    }


def segmentation_metrics(
    pred: np.ndarray, target: np.ndarray, valid: np.ndarray
) -> dict[str, float | int]:
    return metrics_from_confusion(confusion(pred, target, valid))


def aggregate(confusions: Iterable[Confusion]) -> dict[str, float | int]:
    """Микро-усреднение: складываем пиксели, потом считаем метрики.

    Это честнее макро-среднего по чипам, когда доля воды между чипами различается
    на порядки: маленький чип с тремя пикселями воды не должен весить столько же,
    сколько чип, залитый наполовину. Макро-среднее считаем отдельно, когда нужно.
    """
    total = Confusion(0, 0, 0, 0)
    for item in confusions:
        total = total + item
    return metrics_from_confusion(total)


def macro_average(per_chip: Iterable[Mapping[str, float]], keys: Iterable[str]) -> dict[str, float]:
    """Среднее метрик по чипам, NaN игнорируются."""
    rows = list(per_chip)
    out: dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        values = [v for v in values if not np.isnan(v)]
        out[f"{key}_macro"] = float(np.mean(values)) if values else float("nan")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Порог бинаризации
# ─────────────────────────────────────────────────────────────────────────────


def threshold_sweep(
    prob: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, float | int]]:
    """Влияние порога на метрики. Выбор порога делается ТОЛЬКО на validation."""
    grid = list(thresholds) if thresholds is not None else [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]
    rows = []
    for threshold in grid:
        row = segmentation_metrics(prob >= threshold, target, valid)
        row["threshold"] = float(threshold)
        rows.append(row)
    return rows


def best_threshold(sweep: list[Mapping[str, float]], key: str = "f1") -> float:
    """Порог с лучшим значением метрики. Применяется к тесту без подгонки."""
    finite = [row for row in sweep if not np.isnan(float(row[key]))]
    if not finite:
        raise ValueError("не удалось выбрать порог: метрика неопределена на всей сетке")
    return float(max(finite, key=lambda row: float(row[key]))["threshold"])


# ─────────────────────────────────────────────────────────────────────────────
# Надёжность вероятностей
# ─────────────────────────────────────────────────────────────────────────────


def reliability(
    prob: np.ndarray, target: np.ndarray, bins: int = 10
) -> list[dict[str, float | int]]:
    """Диаграмма надёжности: средняя вероятность против наблюдаемой доли воды."""
    p = np.asarray(prob, dtype=np.float64).ravel()
    y = np.asarray(target, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        selected = (p >= low) & (p < high if index < bins - 1 else p <= high)
        count = int(np.count_nonzero(selected))
        rows.append(
            {
                "bin_low": float(low),
                "bin_high": float(high),
                "n_pixels": count,
                "mean_predicted": float(p[selected].mean()) if count else float("nan"),
                "observed_fraction": float(y[selected].mean()) if count else float("nan"),
            }
        )
    return rows


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    """ECE: средневзвешенное расхождение между обещанной и наблюдаемой долей."""
    rows = reliability(prob, target, bins)
    total = sum(row["n_pixels"] for row in rows)
    if not total:
        return float("nan")
    error = 0.0
    for row in rows:
        if not row["n_pixels"]:
            continue
        error += row["n_pixels"] / total * abs(row["mean_predicted"] - row["observed_fraction"])
    return float(error)


def brier_score(prob: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(prob, dtype=np.float64).ravel()
    y = np.asarray(target, dtype=np.float64).ravel()
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Полезность неопределённости
# ─────────────────────────────────────────────────────────────────────────────


def roc_auc(score: np.ndarray, label: np.ndarray) -> float:
    """AUC через ранги. Используется для проверки «неопределённость → ошибка»."""
    s = np.asarray(score, dtype=np.float64).ravel()
    y = np.asarray(label, dtype=bool).ravel()
    positives = int(np.count_nonzero(y))
    negatives = int(y.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # Средние ранги для совпадающих значений, иначе AUC смещается на плоских участках.
    sorted_scores = s[order]
    start = 0
    for index in range(1, s.size + 1):
        if index == s.size or sorted_scores[index] != sorted_scores[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def uncertainty_usefulness(
    uncertainty: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    """Проверяет, выше ли неопределённость там, где метод действительно ошибается.

    Это и есть разница между «уверенностью модели» и проверенной неопределённостью,
    за которую спрашивает критерий 10. Если AUC близок к 0,5 — связи нет, и об этом
    надо писать прямо, а не подбирать другой показатель до красивого числа.
    """
    u = np.asarray(uncertainty)[valid]
    error = (np.asarray(pred, dtype=bool) != np.asarray(target, dtype=bool))[valid]
    auc = roc_auc(u, error)
    return {
        "auc_uncertainty_vs_error": auc,
        "mean_uncertainty_on_errors": float(u[error].mean()) if error.any() else float("nan"),
        "mean_uncertainty_on_correct": float(u[~error].mean()) if (~error).any() else float("nan"),
        "n_pixels": int(u.size),
        "n_errors": int(np.count_nonzero(error)),
    }


def as_dict(matrix: Confusion) -> dict[str, int]:
    return asdict(matrix)
