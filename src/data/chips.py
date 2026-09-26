"""Чтение чипов Sen1Floods11 и базовая статистика по ним.

Слой доступа к данным для всей модельной части: выше по стеку никто не открывает
GeoTIFF напрямую и не решает заново, что считать валидным пикселем и что считать
целевым классом. Эти два вопроса решаются здесь и только здесь.

Ключевые договорённости (CLAUDE.md и вольт, `00_Кейс/Данные_Sen1Floods11.md`):

* ``S1Hand``       — два канала VV/VH в дБ, float32, nodata = NaN, EPSG:4326;
* ``LabelHand``    — int16: -1 метки нет, 0 не вода, 1 вода;
* ``JRCWaterHand`` — uint8 0/1, постоянная вода (JRC Global Surface Water, Landsat
  1984-2020, то есть заведомо до событий набора 2016-2019);
* сетки слоёв одного чипа совпадают, перепроецировать не нужно — но мы это
  проверяем на каждом чтении, а не верим на слово;
* целевой класс проекта — **временное затопление**: ``LabelHand == 1 AND
  JRCWaterHand == 0``, а не вода вообще.

Пиксели без метки (-1) и пиксели с NaN в S1 исключаются и из обучения, и из метрик:
это «неизвестно», а не «сухо».
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.contracts import (
    CHIP_SIZE_PX,
    LABEL_INVALID,
    LABEL_WATER,
    LAYER_JRC,
    LAYER_LABEL,
    LAYER_S1,
)
from src.data import fetch

#: Слои, без которых чип бесполезен для обучения и оценки.
#: ``S1OtsuLabelHand`` сюда не входит: это baseline авторов, а не исходные данные.
REQUIRED_LAYERS: tuple[str, ...] = (LAYER_S1, LAYER_LABEL, LAYER_JRC)

#: Допуск при сверке геопривязки слоёв. Коэффициенты affine у слоёв одного чипа
#: различаются в последних разрядах double (порядок 1e-19 при шаге 9e-05), поэтому
#: сравнение «на равенство» давало бы ложные срабатывания. Допуск взят как одна
#: миллионная пикселя — это доли микрометра на местности.
GRID_RTOL: float = 1e-6


class ChipError(RuntimeError):
    """Базовая ошибка слоя чтения чипов."""


class MissingLayerError(ChipError):
    """Нужного слоя нет на диске (например, выгрузка ещё не дошла до этого чипа)."""


class GridMismatchError(ChipError):
    """Слои одного чипа лежат на разных сетках — совмещать их нельзя."""


@dataclass
class Chip:
    """Один чип 512x512 со всеми слоями, приведёнными к общей сетке.

    ``vv`` и ``vh`` — обратное рассеяние в дБ (float32, nodata = NaN),
    ``label`` — ручная метка (int16, -1/0/1),
    ``jrc`` — постоянная вода (uint8, 0/1) либо ``None``, если слой не скачан.
    ``transform``/``crs``/``profile`` взяты у ``S1Hand`` и служат образцом для всех
    растров, которые мы потом пишем по этому чипу.
    """

    chip_id: str
    event: str
    vv: np.ndarray
    vh: np.ndarray
    label: np.ndarray | None
    jrc: np.ndarray | None
    transform: Affine
    crs: CRS
    profile: dict[str, Any]

    @property
    def shape(self) -> tuple[int, int]:
        """Размер чипа в пикселях (строки, столбцы)."""
        return (int(self.label.shape[0]), int(self.label.shape[1]))

    @property
    def has_jrc(self) -> bool:
        """Есть ли слой постоянной воды — без него целевой класс не определён."""
        return self.jrc is not None


def event_of(chip_id: str) -> str:
    """Событие по идентификатору чипа: ``India_900498`` -> ``India``.

    Имя события может содержать дефис (``Sri-Lanka``), но не подчёркивание, поэтому
    разделитель однозначен.
    """
    event, _, rest = chip_id.partition("_")
    if not event or not rest:
        raise ValueError(f"не разобрать идентификатор чипа: {chip_id!r}")
    return event


def _check_grid(
    chip_id: str,
    layer: str,
    shape: tuple[int, int],
    transform: Affine,
    crs: CRS | None,
    ref_shape: tuple[int, int],
    ref_transform: Affine,
    ref_crs: CRS | None,
) -> None:
    """Сверяет сетку слоя с эталонной (``S1Hand``) и падает с внятным сообщением."""
    problems: list[str] = []
    if shape != ref_shape:
        problems.append(f"размер {shape} вместо {ref_shape}")
    if crs != ref_crs:
        problems.append(f"CRS {crs} вместо {ref_crs}")
    pixel = max(abs(ref_transform.a), abs(ref_transform.e))
    atol = pixel * GRID_RTOL
    got = tuple(transform)[:6]
    ref = tuple(ref_transform)[:6]
    if any(abs(g - r) > atol for g, r in zip(got, ref)):
        problems.append(f"transform {got} вместо {ref} (допуск {atol:.3e})")
    if problems:
        raise GridMismatchError(
            f"чип {chip_id}: слой {layer} не совпадает с {LAYER_S1} по сетке — "
            + "; ".join(problems)
            + ". Совмещать такие слои без перепроецирования нельзя."
        )


def _require_layer(chip_id: str, layer: str, root: Path) -> Path:
    """Путь к слою; если файла нет или он пуст — понятная ошибка."""
    path = fetch.layer_path(chip_id, layer, root)
    if not path.exists() or path.stat().st_size == 0:
        raise MissingLayerError(
            f"чип {chip_id}: нет слоя {layer} ({path}). "
            "Выгрузите его: python -m src.cli.fetch_data"
        )
    return path


def load_chip(chip_id: str, root: Path = fetch.DEFAULT_ROOT) -> Chip:
    """Читает чип целиком и проверяет, что слои лежат на одной сетке.

    Обязателен только ``S1Hand``: решение должно запускаться на голом радарном
    снимке, без каких-либо вспомогательных слоёв. ``LabelHand`` и ``JRCWaterHand``
    необязательны — без них поля ``label`` и ``jrc`` остаются ``None``, а функции,
    которым метка нужна, честно об этом сообщают. Тихой подмены одного другим не
    происходит нигде: вероятность считается по S1, метка нужна только для метрик.
    """
    root = Path(root)
    s1_path = _require_layer(chip_id, LAYER_S1, root)
    with rasterio.open(s1_path) as src:
        if src.count < 2:
            raise ChipError(
                f"чип {chip_id}: в {LAYER_S1} ожидались 2 канала (VV, VH), найдено {src.count}"
            )
        s1 = src.read(out_dtype="float32")
        ref_transform = src.transform
        ref_crs = src.crs
        profile = dict(src.profile)
    ref_shape = (int(s1.shape[1]), int(s1.shape[2]))
    if ref_shape != (CHIP_SIZE_PX, CHIP_SIZE_PX):
        raise ChipError(
            f"чип {chip_id}: ожидался размер {CHIP_SIZE_PX}x{CHIP_SIZE_PX}, получено {ref_shape}"
        )

    label: np.ndarray | None = None
    label_path = fetch.layer_path(chip_id, LAYER_LABEL, root)
    if label_path.exists() and label_path.stat().st_size > 0:
        with rasterio.open(label_path) as src:
            label = src.read(1)
            _check_grid(
                chip_id,
                LAYER_LABEL,
                (int(src.height), int(src.width)),
                src.transform,
                src.crs,
                ref_shape,
                ref_transform,
                ref_crs,
            )

    jrc: np.ndarray | None = None
    jrc_path = fetch.layer_path(chip_id, LAYER_JRC, root)
    if jrc_path.exists() and jrc_path.stat().st_size > 0:
        with rasterio.open(jrc_path) as src:
            jrc = src.read(1)
            _check_grid(
                chip_id,
                LAYER_JRC,
                (int(src.height), int(src.width)),
                src.transform,
                src.crs,
                ref_shape,
                ref_transform,
                ref_crs,
            )

    return Chip(
        chip_id=chip_id,
        event=event_of(chip_id),
        vv=np.asarray(s1[0], dtype="float32"),
        vh=np.asarray(s1[1], dtype="float32"),
        label=None if label is None else np.asarray(label, dtype="int16"),
        jrc=None if jrc is None else np.asarray(jrc, dtype="uint8"),
        transform=ref_transform,
        crs=ref_crs,
        profile=profile,
    )


def nodata_mask(chip: Chip) -> np.ndarray:
    """Пиксели без радарных данных: NaN хотя бы в одном из каналов."""
    return np.isnan(chip.vv) | np.isnan(chip.vh)


def valid_mask(chip: Chip) -> np.ndarray:
    """Пиксели, которые вообще можно использовать.

    Метка задана (не -1) И оба канала S1 не NaN. Всё остальное — «неизвестно»:
    такие пиксели не попадают ни в обучающую выборку, ни в знаменатель метрик.

    Если слоя меток нет (режим запуска по одному S1), валидными считаются пиксели
    с радарными данными: метрики в этом режиме всё равно не считаются, а маска
    нужна интерфейсу и размещению объектов.
    """
    if chip.label is None:
        return ~nodata_mask(chip)
    return (chip.label != LABEL_INVALID) & ~nodata_mask(chip)


def _require_label(chip: Chip) -> np.ndarray:
    if chip.label is None:
        raise MissingLayerError(
            f"чип {chip.chip_id}: нет слоя {LAYER_LABEL}. Целевой класс и метрики без "
            "ручной разметки не определены; запуск по одному S1 даёт вероятность и "
            "маску, но не качество"
        )
    return chip.label


def target_water(chip: Chip) -> np.ndarray:
    """Вода вообще (``LabelHand == 1``): постоянная плюс временная."""
    return chip.label == LABEL_WATER


def permanent_water(chip: Chip) -> np.ndarray:
    """Постоянная вода по официальному слою JRC (``JRCWaterHand == 1``)."""
    if chip.jrc is None:
        raise ChipError(
            f"чип {chip.chip_id}: нет слоя {LAYER_JRC}, постоянная вода неизвестна. "
            "Выгрузите слой: python -m src.cli.fetch_data"
        )
    return chip.jrc > 0


def target_flood(chip: Chip) -> np.ndarray:
    """ВРЕМЕННОЕ ЗАТОПЛЕНИЕ — основной целевой класс проекта.

    ``LabelHand == 1 AND JRCWaterHand == 0``: вода там, где её в норме нет.
    Если слоя JRC нет, функция падает: подменять целевой класс водой вообще
    нельзя, подпись результата обязана соответствовать тому, что посчитано.
    """
    if chip.jrc is None:
        raise ChipError(
            f"чип {chip.chip_id}: нет слоя {LAYER_JRC}, временное затопление посчитать нельзя. "
            "Подменять его водой вообще запрещено — это другой класс."
        )
    return target_water(chip) & (chip.jrc == 0)


def _band_stats(band: np.ndarray, prefix: str) -> dict[str, float | None]:
    """min/max/mean канала по конечным значениям (NaN отброшены)."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return {f"{prefix}_min": None, f"{prefix}_max": None, f"{prefix}_mean": None}
    return {
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_max": float(finite.max()),
        f"{prefix}_mean": float(finite.mean()),
    }


def chip_stats(chip: Chip) -> dict[str, Any]:
    """Сводка по одному чипу для EDA.

    Доли ``nodata`` и ``invalid_label`` считаются от всех пикселей чипа, а доли воды,
    постоянной воды и временного затопления — от ВАЛИДНЫХ пикселей: иначе большая
    зона NaN (а она бывает и на 60 % чипа) искажает картину.
    ``permanent_of_water`` — какая часть воды на чипе постоянная; именно она
    показывает, насколько на этом событии легко выдать реку за паводок.
    """
    total = int(chip.label.size)
    nodata = nodata_mask(chip)
    valid = valid_mask(chip)
    n_valid = int(valid.sum())
    water = target_water(chip) & valid
    n_water = int(water.sum())

    stats: dict[str, Any] = {
        "chip_id": chip.chip_id,
        "event": chip.event,
        "pixels": total,
        "valid_px": n_valid,
        "valid_share": n_valid / total,
        "nodata_share": float(nodata.sum()) / total,
        "invalid_label_share": float((chip.label == LABEL_INVALID).sum()) / total,
        "water_share": (n_water / n_valid) if n_valid else None,
    }

    if chip.jrc is None:
        stats["permanent_share"] = None
        stats["flood_share"] = None
        stats["permanent_of_water"] = None
    else:
        n_perm = int((permanent_water(chip) & valid).sum())
        n_flood = int((target_flood(chip) & valid).sum())
        stats["permanent_share"] = (n_perm / n_valid) if n_valid else None
        stats["flood_share"] = (n_flood / n_valid) if n_valid else None
        stats["permanent_of_water"] = ((n_water - n_flood) / n_water) if n_water else None

    stats.update(_band_stats(chip.vv, "vv"))
    stats.update(_band_stats(chip.vh, "vh"))
    return stats


def iter_chips(
    chip_ids: Iterable[str],
    root: Path = fetch.DEFAULT_ROOT,
    *,
    skip_missing: bool = False,
) -> Iterator[Chip]:
    """Генератор чипов: в памяти держим один чип, а не всю выборку.

    ``skip_missing`` нужен, когда выгрузка идёт фоном и часть файлов ещё не дошла.
    По умолчанию выключен: молча терять чипы обучающей выборки нельзя.
    """
    for chip_id in chip_ids:
        try:
            chip = load_chip(chip_id, root)
        except MissingLayerError:
            if skip_missing:
                continue
            raise
        yield chip


def available_chips(
    root: Path = fetch.DEFAULT_ROOT,
    layers: Sequence[str] = REQUIRED_LAYERS,
) -> list[str]:
    """Чипы, скачанные целиком: есть все обязательные слои и все файлы непустые."""
    root = Path(root)
    base = root / LAYER_S1
    if not base.is_dir():
        return []
    suffix = f"_{LAYER_S1}.tif"
    found: list[str] = []
    for path in base.glob(f"*{suffix}"):
        chip_id = path.name[: -len(suffix)]
        complete = True
        for layer in layers:
            candidate = fetch.layer_path(chip_id, layer, root)
            if not candidate.exists() or candidate.stat().st_size == 0:
                complete = False
                break
        if complete:
            found.append(chip_id)
    return sorted(found)


def normalization_stats(
    chip_ids: Iterable[str],
    root: Path = fetch.DEFAULT_ROOT,
    *,
    pixels_per_chip: int = 20_000,
    seed: int = 2026,
    skip_missing: bool = False,
) -> dict[str, Any]:
    """Среднее и СКО VV/VH ТОЛЬКО по переданным чипам.

    Считается по обучающей части и ни по какой другой: статистика нормировки —
    часть обученной модели, а не свойство набора. Взять сюда validation или test
    означает утечку.

    Устойчивость: с каждого чипа берётся случайная подвыборка валидных пикселей
    (NaN отброшены), накопление идёт в float64, дополнительно возвращаются медиана
    и процентили 1/99 — по ним видно, не утащено ли среднее выбросами.
    """
    rng = np.random.default_rng(seed)
    vv_parts: list[np.ndarray] = []
    vh_parts: list[np.ndarray] = []
    used: list[str] = []

    for chip in iter_chips(chip_ids, root, skip_missing=skip_missing):
        keep = valid_mask(chip)
        if not keep.any():
            continue
        vv = chip.vv[keep].astype("float64")
        vh = chip.vh[keep].astype("float64")
        if vv.size > pixels_per_chip:
            idx = rng.choice(vv.size, size=pixels_per_chip, replace=False)
            vv, vh = vv[idx], vh[idx]
        vv_parts.append(vv)
        vh_parts.append(vh)
        used.append(chip.chip_id)

    if not used:
        raise ChipError("нормировку не по чему считать: ни одного валидного чипа")

    vv_all = np.concatenate(vv_parts)
    vh_all = np.concatenate(vh_parts)
    out: dict[str, Any] = {
        "n_chips": len(used),
        "n_pixels": int(vv_all.size),
        "pixels_per_chip": pixels_per_chip,
        "seed": seed,
    }
    for name, arr in (("vv", vv_all), ("vh", vh_all)):
        out[f"{name}_mean"] = float(arr.mean())
        out[f"{name}_std"] = float(arr.std())
        out[f"{name}_median"] = float(np.median(arr))
        out[f"{name}_p1"] = float(np.percentile(arr, 1))
        out[f"{name}_p99"] = float(np.percentile(arr, 99))
    return out
