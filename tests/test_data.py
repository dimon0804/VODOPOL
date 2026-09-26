"""Проверки слоя чтения данных и протокола разделения выборки.

Запуск из корня проекта:

    python -m pytest tests -q

Тесты, которым нужны сами растры, работают на реально скачанных чипах и
пропускаются, пока выгрузка не дошла хотя бы до двух полных чипов. Тесты
протокола разделения от данных не зависят и выполняются всегда.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import CHIP_SIZE_PX, EVENT_CHIP_COUNTS, LABEL_INVALID, LAYER_LABEL, LAYER_S1
from src.data import fetch, splits as splits_mod
from src.data.chips import (
    ChipError,
    available_chips,
    chip_stats,
    event_of,
    iter_chips,
    load_chip,
    normalization_stats,
    target_flood,
    target_water,
    valid_mask,
)

DATA_ROOT = ROOT / fetch.DEFAULT_ROOT


@pytest.fixture(scope="module")
def ready_chips() -> list[str]:
    chips = available_chips(DATA_ROOT)
    if len(chips) < 2:
        pytest.skip(
            f"скачано полностью {len(chips)} чипов, нужно минимум 2: "
            "запустите python -m src.cli.fetch_data"
        )
    return chips[:2]


# ─────────────────────────────────────────────────────────────────────────────
# Чтение чипов
# ─────────────────────────────────────────────────────────────────────────────


def test_load_chip_shapes_and_types(ready_chips: list[str]) -> None:
    for chip_id in ready_chips:
        chip = load_chip(chip_id, DATA_ROOT)
        assert chip.chip_id == chip_id
        assert chip.event == event_of(chip_id)
        assert chip.event in EVENT_CHIP_COUNTS
        assert chip.vv.shape == (CHIP_SIZE_PX, CHIP_SIZE_PX)
        assert chip.vh.shape == (CHIP_SIZE_PX, CHIP_SIZE_PX)
        assert chip.label.shape == (CHIP_SIZE_PX, CHIP_SIZE_PX)
        assert chip.vv.dtype == np.float32 and chip.vh.dtype == np.float32
        assert chip.label.dtype == np.int16
        assert chip.has_jrc, "JRCWaterHand обязателен: available_chips его требует"
        assert chip.jrc is not None and chip.jrc.shape == (CHIP_SIZE_PX, CHIP_SIZE_PX)
        assert set(np.unique(chip.label)) <= {-1, 0, 1}
        assert set(np.unique(chip.jrc)) <= {0, 1}


def test_grids_match_between_layers(ready_chips: list[str]) -> None:
    """load_chip обязан сверять сетки: проверяем это против прямого чтения слоя."""
    for chip_id in ready_chips:
        chip = load_chip(chip_id, DATA_ROOT)
        with rasterio.open(fetch.layer_path(chip_id, LAYER_LABEL, DATA_ROOT)) as src:
            assert (src.height, src.width) == chip.shape
            assert src.crs == chip.crs
            pixel = max(abs(chip.transform.a), abs(chip.transform.e))
            got, ref = tuple(src.transform)[:6], tuple(chip.transform)[:6]
            assert all(abs(g - r) <= pixel * 1e-6 for g, r in zip(got, ref))


def test_valid_mask_not_empty(ready_chips: list[str]) -> None:
    for chip_id in ready_chips:
        chip = load_chip(chip_id, DATA_ROOT)
        mask = valid_mask(chip)
        assert mask.dtype == np.bool_
        assert mask.any(), f"чип {chip_id}: валидных пикселей нет вовсе"
        assert not (mask & (chip.label == LABEL_INVALID)).any()
        assert not np.isnan(chip.vv[mask]).any()
        assert not np.isnan(chip.vh[mask]).any()


def test_flood_is_subset_of_water(ready_chips: list[str]) -> None:
    """Временное затопление — подмножество воды вообще, и оно её не заменяет."""
    for chip_id in ready_chips:
        chip = load_chip(chip_id, DATA_ROOT)
        water = target_water(chip)
        flood = target_flood(chip)
        assert flood.dtype == np.bool_
        assert np.all(flood <= water), f"чип {chip_id}: затопление вне воды"
        permanent = water & ~flood
        assert not np.any(permanent & (chip.jrc == 0))


def test_target_flood_without_jrc_raises(ready_chips: list[str]) -> None:
    """Без слоя JRC целевой класс не подменяется водой вообще, а честно падает."""
    chip = load_chip(ready_chips[0], DATA_ROOT)
    chip.jrc = None
    with pytest.raises(ChipError):
        target_flood(chip)


def test_chip_stats_consistent(ready_chips: list[str]) -> None:
    for chip_id in ready_chips:
        stats = chip_stats(load_chip(chip_id, DATA_ROOT))
        assert stats["pixels"] == CHIP_SIZE_PX * CHIP_SIZE_PX
        for key in ("nodata_share", "invalid_label_share", "water_share", "flood_share"):
            assert 0.0 <= stats[key] <= 1.0
        assert stats["flood_share"] <= stats["water_share"] + 1e-12
        assert stats["vv_min"] <= stats["vv_mean"] <= stats["vv_max"]
        assert stats["vh_min"] <= stats["vh_mean"] <= stats["vh_max"]


def test_iter_chips_skips_missing(ready_chips: list[str]) -> None:
    ids = [ready_chips[0], "Nowhere_000000", ready_chips[1]]
    got = [chip.chip_id for chip in iter_chips(ids, DATA_ROOT, skip_missing=True)]
    assert got == [ready_chips[0], ready_chips[1]]
    with pytest.raises(ChipError):
        list(iter_chips(ids, DATA_ROOT))


def test_normalization_stats_finite(ready_chips: list[str]) -> None:
    stats = normalization_stats(ready_chips, DATA_ROOT, pixels_per_chip=2_000)
    assert stats["n_chips"] == len(ready_chips)
    for key in ("vv_mean", "vv_std", "vh_mean", "vh_std"):
        assert np.isfinite(stats[key])
    assert stats["vv_std"] > 0 and stats["vh_std"] > 0
    repeated = normalization_stats(ready_chips, DATA_ROOT, pixels_per_chip=2_000)
    assert stats == repeated, "нормировка обязана быть воспроизводимой при том же seed"


# ─────────────────────────────────────────────────────────────────────────────
# Протокол разделения
# ─────────────────────────────────────────────────────────────────────────────


def all_chip_ids() -> list[str]:
    """Синтетический полный перечень набора — от наличия данных на диске не зависит."""
    return [f"{event}_{index:06d}" for event, count in EVENT_CHIP_COUNTS.items() for index in range(count)]


@pytest.fixture(scope="module")
def full_split() -> dict:
    return splits_mod.split_by_event(all_chip_ids())


def test_split_covers_everything(full_split: dict) -> None:
    assert sum(full_split["counts"].values()) == sum(EVENT_CHIP_COUNTS.values()) == 446
    assigned = [c for part in splits_mod.PARTS for c in full_split["chips"][part]]
    assert len(assigned) == len(set(assigned)) == 446


def test_no_shared_events_between_parts(full_split: dict) -> None:
    events = {part: set(full_split["events"][part]) for part in splits_mod.PARTS}
    for left in splits_mod.PARTS:
        for right in splits_mod.PARTS:
            if left != right:
                assert not (events[left] & events[right]), f"{left} и {right} делят событие"


def test_check_no_leakage_clean(full_split: dict) -> None:
    assert splits_mod.check_no_leakage(full_split) == []


def test_bolivia_is_holdout_only(full_split: dict) -> None:
    for part in (splits_mod.PART_TRAIN, splits_mod.PART_VALID, splits_mod.PART_TEST):
        assert "Bolivia" not in full_split["events"][part]
        assert not any(c.startswith("Bolivia_") for c in full_split["chips"][part])
    assert full_split["events"][splits_mod.PART_HOLDOUT] == ["Bolivia"]
    assert full_split["counts"][splits_mod.PART_HOLDOUT] == EVENT_CHIP_COUNTS["Bolivia"]


def test_train_keeps_majority(full_split: dict) -> None:
    learnable = sum(EVENT_CHIP_COUNTS.values()) - EVENT_CHIP_COUNTS["Bolivia"]
    assert full_split["counts"][splits_mod.PART_TRAIN] > learnable / 2
    valid_n = full_split["counts"][splits_mod.PART_VALID]
    test_n = full_split["counts"][splits_mod.PART_TEST]
    assert abs(valid_n - test_n) <= 0.25 * max(valid_n, test_n), "части оценки несопоставимы"


def test_same_seed_same_result() -> None:
    ids = all_chip_ids()
    first = splits_mod.split_by_event(ids, seed=123)
    second = splits_mod.split_by_event(list(reversed(ids)), seed=123)
    assert first == second
    other = splits_mod.split_by_event(ids, seed=124)
    assert other["counts"] == first["counts"]
    assert other["events"] == first["events"]


def test_leakage_is_detected() -> None:
    """Проверка обязана ловить нарушение, а не только подтверждать хорошее."""
    broken = splits_mod.split_by_event(all_chip_ids())
    moved = broken["chips"][splits_mod.PART_TRAIN][0]
    broken["chips"][splits_mod.PART_TEST].append(moved)
    broken["events"][splits_mod.PART_TEST] = sorted(
        set(broken["events"][splits_mod.PART_TEST]) | {event_of(moved)}
    )
    problems = splits_mod.check_no_leakage(broken)
    assert problems and any("общие события" in p for p in problems)


def test_conflicting_events_rejected() -> None:
    with pytest.raises(ValueError):
        splits_mod.split_by_event(all_chip_ids(), valid_events=("Spain",), test_events=("Spain",))
    with pytest.raises(ValueError):
        splits_mod.split_by_event(all_chip_ids(), valid_events=("Atlantis",))


def test_save_and_load_roundtrip(tmp_path: Path, full_split: dict) -> None:
    out = splits_mod.save_splits(full_split, tmp_path / "splits")
    assert (out / splits_mod.MANIFEST_NAME).exists()
    restored = splits_mod.load_splits(out)
    assert restored["chips"] == full_split["chips"]
    assert restored["events"] == full_split["events"]
    assert restored["seed"] == full_split["seed"]
    assert splits_mod.check_no_leakage(restored) == []


def test_author_split_has_event_leakage() -> None:
    """Аналитика для отчёта: у авторов части действительно делят события."""
    report = splits_mod.author_split_overlap(DATA_ROOT)
    if not report.get("available"):
        pytest.skip(report.get("reason", "нет авторских split-файлов"))
    assert report["leakage"] is True
    assert report["events_shared_by_all_three_count"] >= 10
    for info in report["pairwise"].values():
        assert info["shared_chips"] == 0, "по чипам авторы не пересекаются — утечка на событиях"


# ── снимок из стороннего источника ───────────────────────────────────────────
# Эксперты на чекпоинте заметили, что источники прибиты к Sen1Floods11. Для
# обучения это так и есть по определению, а для применения — нет: на вход идут
# два канала в дБ, и происхождение файла модели безразлично. Проверяем, что
# произвольный GeoTIFF читается, а негодный отвергается внятно, а не молча.

def test_внешний_снимок_читается_как_чип(tmp_path: Path) -> None:
    import rasterio
    from src.data import external, fetch

    source = fetch.layer_path("India_900498", LAYER_S1)
    if not source.exists():
        pytest.skip("набор не выкачан: python -m src.cli.fetch_data")

    chip = external.load_external_chip(source, "Foreign_1")
    assert chip.vv.shape == chip.vh.shape == (512, 512)
    assert chip.label is None and chip.jrc is None, "разметки у чужого снимка быть не может"
    assert np.isfinite(chip.vv).any()


def test_не_децибелы_отвергаются_с_объяснением(tmp_path: Path) -> None:
    """Линейная интенсивность вместо дБ — числа другого порядка и мусор на выходе."""
    import rasterio
    from rasterio.transform import from_origin
    from src.data import external
    from src.data.chips import ChipError

    path = tmp_path / "linear.tif"
    data = np.full((2, 512, 512), 0.05, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=512, width=512, count=2,
        dtype="float32", crs="EPSG:4326", transform=from_origin(77.0, 20.0, 9e-05, 9e-05),
    ) as dst:
        dst.write(data)

    with pytest.raises(ChipError) as err:
        external.load_external_chip(path, "Linear_1")
    assert "децибел" in str(err.value)


def test_чужой_размер_кадра_отвергается(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    from src.data import external
    from src.data.chips import ChipError

    path = tmp_path / "small.tif"
    data = np.full((2, 100, 100), -15.0, dtype="float32")
    with rasterio.open(
        path, "w", driver="GTiff", height=100, width=100, count=2,
        dtype="float32", crs="EPSG:4326", transform=from_origin(77.0, 20.0, 9e-05, 9e-05),
    ) as dst:
        dst.write(data)

    with pytest.raises(ChipError) as err:
        external.load_external_chip(path, "Small_1")
    assert "512" in str(err.value)
