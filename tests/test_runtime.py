"""Тесты единственной точки входа сервиса — RunContext.

Слой API ничего не считает сам, поэтому всё, что увидит оператор, приходит
отсюда. Проверяем две ветки: демонстрационный комплект (нужен, чтобы верстать
до появления данных) и реальный комплект запуска на диске.

Тесты, которым нужен готовый комплект, пропускаются, если его каталога нет:
жюри может клонировать репозиторий без outputs/.
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.contracts import (  # noqa: E402
    ASSET_COUNT,
    ASSET_STATUSES,
    BUNDLE_FILES,
    STRATEGIES,
    STRATEGY_SELECTIVE,
    UNC_BASELINE,
    UNC_SCENARIO,
)
from src.runtime import RunContext  # noqa: E402

def _find_run_dir() -> Path | None:
    """Ищем любой собранный комплект, а не конкретное имя.

    Имя каталога содержит бюджет, а бюджет — параметр запуска: привязка к нему
    превращала тесты в пропуски при каждой пересборке пакета с другим числом.
    """
    outputs = PROJECT_ROOT / "outputs"
    if not outputs.exists():
        return None
    ready = sorted(
        (d for d in outputs.iterdir() if d.is_dir() and (d / "run_metadata.json").exists()),
        key=lambda d: d.name,
    )
    india = [d for d in ready if "India" in d.name]
    return (india or ready)[0] if ready else None


RUN_DIR = _find_run_dir() or PROJECT_ROOT / "outputs" / "_нет_комплекта"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

needs_bundle = pytest.mark.skipif(
    not RUN_DIR.exists(), reason=f"готовый комплект не найден: {RUN_DIR}"
)


@pytest.fixture(scope="module")
def demo() -> RunContext:
    return RunContext.demo()


@pytest.fixture(scope="module")
def run() -> RunContext:
    if not RUN_DIR.exists():
        pytest.skip(f"готовый комплект не найден: {RUN_DIR}")
    return RunContext.load(RUN_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Демонстрационный комплект
# ─────────────────────────────────────────────────────────────────────────────


def test_demo_summary_is_marked_as_demo(demo):
    """Синтетика обязана честно помечаться: выдавать её за расчёт нельзя."""
    summary = demo.summary()
    assert summary["demo"] is True
    assert summary["chip_id"]
    assert summary["threshold"] is not None
    assert len(summary["bounds"]) == 4


def test_demo_has_ten_assets_with_valid_statuses(demo):
    """Структура та же, что у настоящего запуска: ровно десять объектов."""
    features = demo.assets()["features"]
    assert len(features) == ASSET_COUNT == 10
    for feature in features:
        props = feature["properties"]
        assert props["status"] in ASSET_STATUSES
        assert len(feature["geometry"]["coordinates"]) == 2


def test_demo_unassessed_asset_keeps_empty_fields(demo):
    """И в демо у no_data поля пустые, а не нулевые — иначе вёрстка привыкнет к нулям."""
    for feature in demo.assets()["features"]:
        props = feature["properties"]
        if props["status"] != "ok":
            assert props["p_flood"] is None
            assert props["expected_loss_rub"] is None
            assert props["rank"] is None


def test_demo_catalog_is_not_empty(demo):
    """Каталог зон нужен интерфейсу карты, пустым он быть не должен."""
    features = demo.candidates()["features"]
    assert features
    assert all(f["properties"]["candidate_id"] for f in features)


def test_demo_comparison_has_three_strategy_rows(demo):
    """Сравнение — всегда три строки A/B/C, иначе сопоставлять нечего."""
    comparison = demo.strategies()["comparison"]
    assert len(comparison) == 3
    assert [row["strategy"] for row in comparison] == list(STRATEGIES)
    statuses = {row["strategy"]: row["uncertainty_status"] for row in comparison}
    assert statuses["A"] == UNC_BASELINE
    assert statuses["B"] == UNC_SCENARIO and statuses["C"] == UNC_SCENARIO


def test_demo_raster_is_png_with_bounds(demo):
    """Карта получает готовый PNG и четыре числа границ."""
    png, bounds = demo.raster_png("probability")
    assert png.startswith(PNG_MAGIC)
    assert len(bounds) == 4
    assert bounds[0] < bounds[2] and bounds[1] < bounds[3]


def test_demo_selective_plan_respects_budget(demo):
    """Слайдер бюджета в демо ведёт себя так же, как в настоящем расчёте."""
    small = demo.strategies(2000.0)
    large = demo.strategies(100000.0)
    assert len(small["plans"][STRATEGY_SELECTIVE]) <= len(large["plans"][STRATEGY_SELECTIVE])


# ─────────────────────────────────────────────────────────────────────────────
# Готовый комплект на диске
# ─────────────────────────────────────────────────────────────────────────────


@needs_bundle
def test_load_reads_summary_of_the_real_run(run):
    """Паспорт запуска читается из run_metadata.json и не помечается демо."""
    summary = run.summary()
    assert summary["demo"] is False
    assert summary["chip_id"] == "India_900498"
    assert summary["run_id"]
    assert summary["threshold"] is not None
    assert summary["budget_rub"] is not None
    assert len(summary["bounds"]) == 4


@needs_bundle
def test_load_returns_ten_assets_with_loss_merged(run):
    """assets.geojson склеивается с asset_loss.csv: на карте объект уже с ущербом."""
    features = run.assets()["features"]
    assert len(features) == ASSET_COUNT == 10
    assert len({f["properties"]["asset_id"] for f in features}) == 10
    for feature in features:
        props = feature["properties"]
        assert props["status"] in ASSET_STATUSES
        if props["status"] == "ok":
            assert props["p_flood"] is not None
            assert props["expected_loss_rub"] is not None
        else:
            assert props["p_flood"] is None
            assert props["expected_loss_rub"] is None
            assert props["rank"] is None


@needs_bundle
def test_probability_raster_is_png_with_four_bounds(run):
    """Слой вероятности отдаётся картинкой с сигнатурой PNG и границами наложения."""
    png, bounds = run.raster_png("probability")
    assert png.startswith(PNG_MAGIC)
    assert len(png) > 1000
    assert len(bounds) == 4
    assert all(isinstance(value, float) for value in bounds)
    assert bounds[0] < bounds[2] and bounds[1] < bounds[3]


@needs_bundle
def test_mask_raster_is_also_png(run):
    """Маска рисуется тем же способом — сервис не должен падать на втором слое."""
    png, bounds = run.raster_png("mask")
    assert png.startswith(PNG_MAGIC)
    assert len(bounds) == 4


@needs_bundle
def test_bundle_zip_contains_every_required_file(run):
    """Кнопка выгрузки отдаёт полный комплект: все файлы из BUNDLE_FILES."""
    archive = zipfile.ZipFile(io.BytesIO(run.bundle_zip()))
    names = set(archive.namelist())
    assert set(BUNDLE_FILES) <= names
    assert archive.testzip() is None
    for name in BUNDLE_FILES:
        assert archive.getinfo(name).file_size > 0


@needs_bundle
def test_two_budgets_give_different_selective_plans(run):
    """Слайдер бюджета действительно пересчитывает C, а не показывает одно и то же."""
    tight = run.strategies(2500.0)
    wide = run.strategies(20000.0)
    assert tight["plans"][STRATEGY_SELECTIVE] != wide["plans"][STRATEGY_SELECTIVE]
    assert len(tight["plans"][STRATEGY_SELECTIVE]) < len(wide["plans"][STRATEGY_SELECTIVE])
    # Стратегия B правилом объявлена заранее и от бюджета не зависит.
    assert tight["plans"]["B"] == wide["plans"]["B"]


@needs_bundle
def test_budget_change_does_not_touch_probabilities_or_portfolio(run):
    """Пересчёт под бюджет не переобучает модель: порог и ущерб объектов те же."""
    before = run.summary()["threshold"]
    losses_before = [f["properties"]["expected_loss_rub"] for f in run.assets()["features"]]
    run.strategies(3000.0)
    assert run.summary()["threshold"] == before
    assert [f["properties"]["expected_loss_rub"] for f in run.assets()["features"]] == losses_before


@needs_bundle
# Исправлено: обе ветки strategies() приводят числа к float, а budget_feasible к bool.
# Строка "false" из CSV раньше была истинной, и контрфактическая B читалась как
# уложившаяся в бюджет.
def test_strategies_returns_the_same_types_with_and_without_budget(run):
    """Одна ручка — один тип полей: иначе интерфейс читает 'false' как истину."""
    stored = {row["strategy"]: row for row in run.strategies()["comparison"]}
    recomputed = {row["strategy"]: row for row in run.strategies(20000.0)["comparison"]}
    for strategy in STRATEGIES:
        assert isinstance(stored[strategy]["budget_feasible"], bool)
        assert type(stored[strategy]["budget_feasible"]) is type(
            recomputed[strategy]["budget_feasible"]
        )


@needs_bundle
def test_stored_budget_feasible_marks_broad_as_counterfactual(run):
    """Пока типы разъезжаются, хотя бы проверяем содержимое: B не влезла в бюджет."""
    stored = {row["strategy"]: row for row in run.strategies()["comparison"]}
    assert str(stored["B"]["budget_feasible"]).lower() == "false"
    assert str(stored["C"]["budget_feasible"]).lower() == "true"


@needs_bundle
def test_sensitivity_rows_are_readable(run):
    """Таблица чувствительности из комплекта читается и не пуста."""
    rows = run.sensitivity()
    assert rows
    for row in rows:
        assert row["scenario_id"]
        assert json.loads(row["changed_inputs_json"])


@needs_bundle
def test_recomputed_plan_stays_within_budget(run):
    """C под любым бюджетом остаётся в пределах лимита и после перечёта из сервиса."""
    from decimal import Decimal

    for budget in (2500.0, 8000.0, 20000.0):
        comparison = {row["strategy"]: row for row in run.strategies(budget)["comparison"]}
        cost = Decimal(str(comparison[STRATEGY_SELECTIVE]["data_cost_rub"]))
        assert cost <= Decimal(str(budget))


# ─────────────────────────────────────────────────────────────────────────────
# Отсутствующий комплект
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_bundle_raises_readable_russian_error(tmp_path):
    """Оператор должен понять, что делать: сообщение по-русски и с командой сборки."""
    missing = tmp_path / "нет-такого-запуска"
    with pytest.raises(FileNotFoundError) as info:
        RunContext.load(missing)
    message = str(info.value)
    assert "комплект запуска не найден" in message
    assert "run_bundle" in message
