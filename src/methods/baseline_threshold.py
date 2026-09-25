"""Пороговый baseline по Sentinel-1.

Точка отсчёта, с которой сравнивается основной метод. Сознательно простой и
объяснимый: обратное рассеяние от гладкой открытой воды низкое, поэтому вода —
это пиксели темнее порога. Порог выбирается на train и validation и после этого
фиксируется; к независимой проверке он применяется без подгонки.

Что baseline НЕ делает, и это принципиально: не даёт калиброванных вероятностей,
не учитывает пространственный контекст и не отличает тень рельефа или гладкий
асфальт от воды. Ровно на этих трёх вещах его и обходит основной метод.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.ndimage import median_filter

Channel = Literal["VV", "VH", "VV+VH"]


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Порог Оцу по гистограмме значений в дБ.

    Работает по конечным значениям: NaN и невалидные пиксели должны быть отсеяны
    до вызова, иначе гистограмма поедет.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("нет конечных значений для выбора порога")
    counts, edges = np.histogram(finite, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_low = np.cumsum(counts)
    weight_high = counts.sum() - weight_low
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_low = np.cumsum(counts * centers) / weight_low
        mean_high = (np.sum(counts * centers) - np.cumsum(counts * centers)) / weight_high
        between = weight_low * weight_high * (mean_low - mean_high) ** 2
    between[~np.isfinite(between)] = -np.inf
    return float(centers[int(np.argmax(between))])


def channel_values(vv: np.ndarray, vh: np.ndarray, channel: Channel) -> np.ndarray:
    """Скалярное поле, к которому применяется порог."""
    if channel == "VV":
        return vv
    if channel == "VH":
        return vh
    if channel == "VV+VH":
        # Среднее двух поляризаций: шум частично гасится, вода остаётся тёмной.
        return (vv + vh) / 2.0
    raise ValueError(f"неизвестный канал: {channel!r}")


def despeckle(values: np.ndarray, size: int) -> np.ndarray:
    """Медианный фильтр против спекла. size=1 означает «без фильтрации»."""
    if size <= 1:
        return values
    filled = np.where(np.isfinite(values), values, np.nanmedian(values))
    return median_filter(filled, size=size, mode="nearest")


@dataclass
class BaselineConfig:
    channel: Channel = "VH"
    threshold_db: float = -18.0
    despeckle_size: int = 3
    #: Как выбран порог: 'otsu' — среднее по обучающим чипам, 'grid' — перебор по validation.
    selection: str = "grid"
    fitted_on: tuple[str, ...] = ()


class BaselineThreshold:
    """Пороговый метод с фиксированным после обучения порогом."""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()

    # ── применение ───────────────────────────────────────────────────────────

    def predict_mask(self, vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
        values = despeckle(channel_values(vv, vh, self.config.channel), self.config.despeckle_size)
        return values <= self.config.threshold_db

    def score(self, vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
        """Расстояние до порога в дБ. Не вероятность и вероятностью не называется."""
        values = despeckle(channel_values(vv, vh, self.config.channel), self.config.despeckle_size)
        return self.config.threshold_db - values

    # ── подбор порога ────────────────────────────────────────────────────────

    @staticmethod
    def otsu_over_chips(
        samples: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
        channel: Channel,
        despeckle_size: int,
    ) -> float:
        """Средний порог Оцу по обучающим чипам.

        Один общий порог на все чипы, а не свой на каждый: на защите метод должен
        применяться к новому снимку без подглядывания в его собственную гистограмму.
        """
        thresholds = []
        for vv, vh, valid in samples:
            values = despeckle(channel_values(vv, vh, channel), despeckle_size)
            selected = values[valid & np.isfinite(values)]
            if selected.size:
                thresholds.append(otsu_threshold(selected))
        if not thresholds:
            raise ValueError("не удалось посчитать порог Оцу: пустая обучающая выборка")
        return float(np.mean(thresholds))

    @staticmethod
    def grid_search(
        samples: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        channels: Sequence[Channel] = ("VV", "VH", "VV+VH"),
        despeckle_sizes: Sequence[int] = (1, 3, 5),
        grid_db: Sequence[float] | None = None,
        metric: str = "f1",
    ) -> tuple[BaselineConfig, list[dict[str, float]]]:
        """Перебор канала, фильтра и порога по валидационной выборке.

        samples: последовательность (vv, vh, target, valid) — только validation.
        Возвращает лучшую конфигурацию и полную таблицу перебора для отчёта.
        """
        from src.eval.metrics import Confusion, confusion, metrics_from_confusion

        grid = list(grid_db) if grid_db is not None else [float(x) for x in np.arange(-26.0, -8.0, 0.5)]
        rows: list[dict[str, float]] = []
        best: tuple[float, BaselineConfig] | None = None

        for channel in channels:
            for size in despeckle_sizes:
                prepared = [
                    (despeckle(channel_values(vv, vh, channel), size), target, valid)
                    for vv, vh, target, valid in samples
                ]
                for threshold in grid:
                    total = Confusion(0, 0, 0, 0)
                    for values, target, valid in prepared:
                        total = total + confusion(values <= threshold, target, valid)
                    scored = metrics_from_confusion(total)
                    row = {
                        "channel": channel,
                        "despeckle_size": size,
                        "threshold_db": threshold,
                        **{k: v for k, v in scored.items()},
                    }
                    rows.append(row)
                    value = float(scored[metric])
                    if not np.isnan(value) and (best is None or value > best[0]):
                        best = (
                            value,
                            BaselineConfig(
                                channel=channel,
                                threshold_db=threshold,
                                despeckle_size=size,
                                selection="grid",
                            ),
                        )

        if best is None:
            raise ValueError("перебор не дал ни одной конфигурации с определённой метрикой")
        return best[1], rows

    # ── сохранение ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.config)
        payload["fitted_on"] = list(self.config.fitted_on)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaselineThreshold":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["fitted_on"] = tuple(payload.get("fitted_on", ()))
        return cls(BaselineConfig(**payload))
