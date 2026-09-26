"""Общая загрузка выборок для обучения и проверки.

Один источник правды о том, что такое «валидные пиксели» и «целевой класс»:
если бы каждый скрипт решал это сам, сравнение двух методов на одних пикселях
перестало бы быть честным, а именно на нём держится критерий 4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from src.data import chips as chips_mod
from src.data import fetch, splits as splits_mod

TargetKind = Literal["flood", "water"]
Sample = tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def target_array(chip: chips_mod.Chip, kind: TargetKind) -> np.ndarray:
    """Целевой класс.

    `flood` — временное затопление, вода за вычетом постоянной по слою JRC.
    Это основной класс решения. `water` — вода вообще, нужен только для того,
    чтобы показать в отчёте обе цифры и не выдавать одну за другую.
    """
    if kind == "flood":
        return chips_mod.target_flood(chip)
    if kind == "water":
        return chips_mod.target_water(chip)
    raise ValueError(f"неизвестный целевой класс: {kind!r}")


def part_chip_ids(part: str, split_dir: Path | str = "splits") -> list[str]:
    data = splits_mod.load_splits(split_dir)
    chips = data["chips"] if "chips" in data else data
    if part not in chips:
        raise KeyError(f"в сплите нет части {part!r}: есть {sorted(chips)}")
    return list(chips[part])


def iter_samples(
    part: str,
    target: TargetKind = "flood",
    limit: int | None = None,
    split_dir: Path | str = "splits",
    root: Path = fetch.DEFAULT_ROOT,
) -> Iterator[Sample]:
    """Итератор (chip_id, vv, vh, target, valid) по части выборки."""
    ids = part_chip_ids(part, split_dir)
    if limit:
        ids = ids[:limit]
    for chip_id in ids:
        try:
            chip = chips_mod.load_chip(chip_id, root)
        except chips_mod.ChipError:
            continue
        valid = chips_mod.valid_mask(chip)
        if not valid.any():
            continue
        yield chip_id, chip.vv, chip.vh, target_array(chip, target), valid


def collect_samples(
    part: str,
    target: TargetKind = "flood",
    limit: int | None = None,
    split_dir: Path | str = "splits",
    root: Path = fetch.DEFAULT_ROOT,
) -> list[Sample]:
    return list(iter_samples(part, target, limit, split_dir, root))


def iter_samples_decomposed(
    part: str,
    limit: int | None = None,
    split_dir: Path | str = "splits",
    root: Path = fetch.DEFAULT_ROOT,
) -> Iterator[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Итератор для разложенной модели: вода и постоянная вода отдаются раздельно.

    Одноголовая модель получает сразу готовую цель «временное затопление». Здесь
    цели две, потому что и голов две: одна учится находить воду, вторая — отличать
    среди неё постоянную.
    """
    ids = part_chip_ids(part, split_dir)
    if limit:
        ids = ids[:limit]
    for chip_id in ids:
        try:
            chip = chips_mod.load_chip(chip_id, root)
        except chips_mod.ChipError:
            continue
        if chip.jrc is None or chip.label is None:
            continue
        valid = chips_mod.valid_mask(chip)
        if not valid.any():
            continue
        yield (
            chip_id,
            chip.vv,
            chip.vh,
            chips_mod.target_water(chip),
            chips_mod.permanent_water(chip),
            valid,
        )
