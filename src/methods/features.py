"""Признаки основного метода: только Sentinel-1, но с пространственным контекстом.

Идея, которой метод обходит пороговый baseline. Порог смотрит на один пиксель и
поэтому не отличает воду от тени рельефа, гладкого асфальта и отдельных тёмных
выбросов спекла. Окрестность отличает: у воды низкое среднее И низкая изменчивость
на нескольких масштабах сразу, у спекла низкое среднее при высокой изменчивости, у
тени низкое среднее в узком окне при обычном фоне в широком.

Контекст считается быстрыми box-фильтрами (`uniform_filter`), а не свёрточной сетью:
на 512×512 это доли секунды на CPU, результат воспроизводится побитово, а веса
модели весят мегабайты и спокойно лежат в репозитории.

Важно: VV и VH уже в децибелах, то есть логарифмические. Повторно логарифмировать
их нельзя, и отношение каналов в дБ — это обычная разность.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter, uniform_filter

#: Масштабы окрестности в пикселях. 10 м на пиксель, то есть 30 м … 510 м на местности.
DEFAULT_SCALES: tuple[int, ...] = (3, 9, 25, 51)


@dataclass(frozen=True)
class FeatureConfig:
    scales: tuple[int, ...] = DEFAULT_SCALES
    #: Разность с медианой чипа. Снимает разницу калибровки между сценами и остаётся
    #: относительной величиной, одинаково осмысленной на любом снимке.
    use_chip_contrast: bool = True
    #: Абсолютные константы уровня чипа (медиана и 5-й процентиль VV). По умолчанию
    #: выключены: дерево использует их как опознавательный знак сцены и запоминает
    #: событие вместо того, чтобы учиться отличать воду. На новых событиях это вредит.
    use_chip_absolute: bool = False
    #: Медианно отфильтрованные каналы. Спекл в радаре силён, и пороговому методу
    #: фильтр помогает заметно — проверяем, помогает ли он и модели.
    despeckle_sizes: tuple[int, ...] = ()
    #: Приводить ли каналы к собственной статистике чипа перед расчётом контекста.
    #: Уровень обратного рассеяния зависит от сцены, угла съёмки и типа местности,
    #: поэтому абсолютные децибелы плохо переносятся между событиями. После вычитания
    #: медианы чипа признак означает «насколько этот пиксель темнее окружающей сцены»,
    #: а это свойство воды, одинаковое в Гане и в Испании.
    per_chip_normalize: bool = False
    names: tuple[str, ...] = field(default=(), compare=False)


def feature_names(config: FeatureConfig | None = None) -> list[str]:
    config = config or FeatureConfig()
    names = ["vv", "vh", "vv_minus_vh", "vv_plus_vh"]
    for scale in config.scales:
        for channel in ("vv", "vh"):
            names += [f"{channel}_mean_{scale}", f"{channel}_std_{scale}"]
        names.append(f"vv_minus_mean_{scale}")
    if config.use_chip_contrast:
        names += ["vv_minus_chip_median", "vh_minus_chip_median"]
    if config.use_chip_absolute:
        names += ["chip_vv_median", "chip_vv_p05"]
    for size in config.despeckle_sizes:
        names += [f"vv_median_{size}", f"vh_median_{size}"]
    return names


def _fill_invalid(band: np.ndarray, fill: float) -> np.ndarray:
    """Дырки затыкаются медианой чипа: box-фильтр не умеет работать с NaN."""
    out = np.asarray(band, dtype=np.float32).copy()
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = fill
    return out


def _mean_std(band: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Среднее и стандартное отклонение в окне size×size."""
    mean = uniform_filter(band, size=size, mode="nearest")
    mean_sq = uniform_filter(band * band, size=size, mode="nearest")
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return mean, np.sqrt(variance)


def build_features(
    vv: np.ndarray, vh: np.ndarray, config: FeatureConfig | None = None
) -> np.ndarray:
    """Куб признаков (высота, ширина, число признаков), float32.

    Порядок признаков совпадает с `feature_names` и является частью контракта модели:
    менять его без переобучения нельзя.
    """
    config = config or FeatureConfig()
    vv = np.asarray(vv, dtype=np.float32)
    vh = np.asarray(vh, dtype=np.float32)
    if vv.shape != vh.shape:
        raise ValueError("каналы VV и VH должны иметь одинаковую форму")

    finite_vv = vv[np.isfinite(vv)]
    finite_vh = vh[np.isfinite(vh)]
    if finite_vv.size == 0 or finite_vh.size == 0:
        raise ValueError("в чипе нет ни одного конечного значения S1")
    median_vv = float(np.median(finite_vv))
    median_vh = float(np.median(finite_vh))
    p05_vv = float(np.percentile(finite_vv, 5))

    vv_f = _fill_invalid(vv, median_vv)
    vh_f = _fill_invalid(vh, median_vh)
    if config.per_chip_normalize:
        # Разброс берём как межквартильный размах: он устойчив к залитой половине кадра.
        scale_vv = float(np.subtract(*np.percentile(finite_vv, [75, 25]))) or 1.0
        scale_vh = float(np.subtract(*np.percentile(finite_vh, [75, 25]))) or 1.0
        vv_f = (vv_f - median_vv) / scale_vv
        vh_f = (vh_f - median_vh) / scale_vh

    layers: list[np.ndarray] = [vv_f, vh_f, vv_f - vh_f, vv_f + vh_f]
    for scale in config.scales:
        vv_mean, vv_std = _mean_std(vv_f, scale)
        vh_mean, vh_std = _mean_std(vh_f, scale)
        layers += [vv_mean, vv_std, vh_mean, vh_std, vv_f - vv_mean]

    if config.use_chip_contrast:
        layers += [vv_f - median_vv, vh_f - median_vh]
    if config.use_chip_absolute:
        layers += [np.full_like(vv_f, median_vv), np.full_like(vv_f, p05_vv)]
    for size in config.despeckle_sizes:
        layers += [
            median_filter(vv_f, size=size, mode="nearest"),
            median_filter(vh_f, size=size, mode="nearest"),
        ]

    cube = np.stack(layers, axis=-1).astype(np.float32)
    expected = len(feature_names(config))
    if cube.shape[-1] != expected:
        raise AssertionError(
            f"число признаков {cube.shape[-1]} не совпадает с именами {expected} — "
            "порядок признаков разошёлся с контрактом"
        )
    return cube


def sample_pixels(
    cube: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    per_chip: int,
    rng: np.random.Generator,
    positive_share: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Стратифицированная подвыборка пикселей одного чипа.

    Вода занимает единицы процентов площади, поэтому равномерная выборка утопила бы
    положительный класс. Берём фиксированную долю положительных, остальное — фон.
    Если положительных мало, добираем фоном, а не дублируем воду.
    """
    flat_valid = valid.ravel()
    flat_target = target.ravel()[flat_valid]
    flat_cube = cube.reshape(-1, cube.shape[-1])[flat_valid]

    positives = np.flatnonzero(flat_target)
    negatives = np.flatnonzero(~flat_target.astype(bool))
    n_pos = min(len(positives), int(per_chip * positive_share))
    n_neg = min(len(negatives), per_chip - n_pos)

    picked = np.concatenate(
        [
            rng.choice(positives, size=n_pos, replace=False) if n_pos else np.array([], dtype=int),
            rng.choice(negatives, size=n_neg, replace=False) if n_neg else np.array([], dtype=int),
        ]
    ).astype(int)
    return flat_cube[picked], flat_target[picked].astype(np.int8)


def sample_pixels_uniform(
    cube: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    per_chip: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Равномерная подвыборка валидных пикселей, БЕЗ стратификации по классу.

    Нужна для калибровки. Если калибровать на сбалансированной выборке, изотоническая
    регрессия выучит искусственную долю воды в пятьдесят процентов, и вероятность 0,7
    перестанет означать наблюдаемую долю 0,7 на реальном снимке, где воды считаные
    проценты. Поэтому здесь доли классов остаются природными.
    """
    flat_valid = valid.ravel()
    flat_target = target.ravel()[flat_valid]
    flat_cube = cube.reshape(-1, cube.shape[-1])[flat_valid]
    if flat_target.size == 0:
        empty = np.empty((0, cube.shape[-1]), dtype=np.float32)
        return empty, np.empty(0, dtype=np.int8)
    size = min(per_chip, flat_target.size)
    picked = rng.choice(flat_target.size, size=size, replace=False)
    return flat_cube[picked], flat_target[picked].astype(np.int8)
