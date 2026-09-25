"""Отделение постоянной воды от временного затопления.

Организаторы назвали это главной сложностью кейса и главным рычагом улучшения
решения. Метка `1` в Sen1Floods11 означает воду вообще: река и озеро помечены так же,
как разлив. Если постоянную воду не вычесть, ожидаемый ущерб систематически
завышается — на контрольном чипе почти половина «воды» оказалась постоянной.

Как мы это делаем и чего не делаем:

* Целевой класс обучения — `LabelHand == 1 AND JRCWaterHand == 0`. Слой JRC входит
  в сам официальный набор и построен по Landsat 1984–2020, то есть до событий
  набора (2016–2019): временная корректность соблюдена.
* JRC используется на стороне РАЗМЕТКИ. На применении модель считает вероятность
  временного затопления по одному Sentinel-1, без каких-либо дополнительных слоёв —
  требование «решение обязано работать только по S1» выполняется буквально.
* Если слой JRC для чипа доступен, его можно применить как необязательное уточнение
  поверх вероятности. Это отдельный режим, и в отчёте он показывается рядом с
  основным, а не вместо него.
* Дособытийной истории Sentinel-1 по тем же контурам в официальном наборе НЕТ.
  Папка `perm_water` содержит 814 самостоятельных чипов S1 с масками JRC по другим
  территориям, а не снимки тех же участков до паводка. Поэтому подтверждение по
  истории S1 мы не выдумываем, а честно указываем как ограничение прототипа.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WaterSplitStats:
    """Разложение воды на постоянную и временную по одному чипу."""

    chip_id: str
    n_valid: int
    n_water: int
    n_permanent: int
    n_flood: int
    n_permanent_dry: int

    @property
    def permanent_share_of_water(self) -> float:
        """Какая доля размеченной воды приходится на постоянные водоёмы."""
        return self.n_permanent_in_water / self.n_water if self.n_water else float("nan")

    @property
    def n_permanent_in_water(self) -> int:
        return self.n_water - self.n_flood

    @property
    def flood_share_of_valid(self) -> float:
        return self.n_flood / self.n_valid if self.n_valid else float("nan")

    def as_row(self) -> dict[str, float | int | str]:
        return {
            "chip_id": self.chip_id,
            "n_valid": self.n_valid,
            "n_water": self.n_water,
            "n_permanent_layer": self.n_permanent,
            "n_permanent_in_water": self.n_permanent_in_water,
            "n_flood": self.n_flood,
            "n_permanent_dry": self.n_permanent_dry,
            "permanent_share_of_water": self.permanent_share_of_water,
            "flood_share_of_valid": self.flood_share_of_valid,
        }


def permanent_mask(jrc: np.ndarray) -> np.ndarray:
    """Маска постоянной воды из слоя JRCWaterHand: значения 0/1."""
    return np.asarray(jrc) > 0


def flood_target(label: np.ndarray, jrc: np.ndarray) -> np.ndarray:
    """Временное затопление: вода по ручной метке за вычетом постоянной воды."""
    return (np.asarray(label) == 1) & ~permanent_mask(jrc)


def split_stats(chip_id: str, label: np.ndarray, jrc: np.ndarray, valid: np.ndarray) -> WaterSplitStats:
    """Разложение воды по одному чипу — материал для отчёта и для защиты.

    `n_permanent_dry` — пиксели, которые JRC считает постоянной водой, а ручная метка
    водой не считает. Это мера расхождения источников: её не прячем, а показываем.
    """
    label = np.asarray(label)
    permanent = permanent_mask(jrc)
    water = (label == 1) & valid
    flood = water & ~permanent
    return WaterSplitStats(
        chip_id=chip_id,
        n_valid=int(np.count_nonzero(valid)),
        n_water=int(np.count_nonzero(water)),
        n_permanent=int(np.count_nonzero(permanent & valid)),
        n_flood=int(np.count_nonzero(flood)),
        n_permanent_dry=int(np.count_nonzero(permanent & valid & (label == 0))),
    )


def suppress_permanent(prob: np.ndarray, jrc: np.ndarray, factor: float = 0.0) -> np.ndarray:
    """Необязательное уточнение вероятности там, где известна постоянная вода.

    factor = 0 обнуляет вероятность временного затопления на постоянных водоёмах,
    factor = 1 не меняет ничего. Промежуточные значения полезны, когда доверие к
    источнику постоянной воды неполное: JRC — производный продукт по оптике Landsat,
    и на узких реках и мелководье он ошибается.

    Режим применяется ТОЛЬКО когда слой JRC доступен. Основной путь решения — без него.
    """
    if not 0.0 <= factor <= 1.0:
        raise ValueError("factor должен лежать в диапазоне от 0 до 1")
    result = np.asarray(prob, dtype=np.float32).copy()
    permanent = permanent_mask(jrc)
    result[permanent] = result[permanent] * float(factor)
    return result


def agreement_report(stats: list[WaterSplitStats]) -> dict[str, float | int]:
    """Сводка расхождения JRC и ручной метки по набору чипов.

    Даёт три числа, которые нужны на защите: сколько воды всего, какая доля
    постоянная, и насколько часто JRC называет постоянной водой то, что размечено
    как суша. Последнее — прямая оценка надёжности источника.
    """
    water = sum(s.n_water for s in stats)
    flood = sum(s.n_flood for s in stats)
    permanent_in_water = sum(s.n_permanent_in_water for s in stats)
    permanent_dry = sum(s.n_permanent_dry for s in stats)
    permanent_total = sum(s.n_permanent for s in stats)
    return {
        "chips": len(stats),
        "n_water": water,
        "n_flood": flood,
        "n_permanent_in_water": permanent_in_water,
        "n_permanent_total": permanent_total,
        "n_permanent_dry": permanent_dry,
        "permanent_share_of_water": permanent_in_water / water if water else float("nan"),
        "permanent_dry_share_of_permanent": (
            permanent_dry / permanent_total if permanent_total else float("nan")
        ),
    }
