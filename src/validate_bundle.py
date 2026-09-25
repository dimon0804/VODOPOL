"""Валидатор пакета сдачи Водополь.

Самостоятельная программа::

    python -m src.validate_bundle outputs/<run_id>

Гоняется перед каждой сдачей. Десять проверок из спецификации обязательных
артефактов: состав пакета, запрет NaN/Inf, соответствие маски порогу, сетка
растров, портфель объектов, геометрия зон, согласованность планов закупки,
суммы и бюджет, уникальность охвата ущерба и пересчёт цены каждой позиции.

Отчёт печатается по-русски, по строке на проверку: ``OK`` / ``ПРОПУЩЕНО`` /
``ОШИБКА``. Код возврата 0, если ошибок нет, иначе 1. Пропуск ошибкой не
считается: он означает, что проверить было нечем, и это сказано прямо.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import rasterio
from shapely.geometry import shape as shapely_shape
from shapely.validation import explain_validity

from src.contracts import (
    ASSET_COUNT,
    ASSET_STATUS_OK,
    ASSET_STATUSES,
    ASSET_TYPE_BY_CLASS,
    BUNDLE_FILES,
    COLUMNS_ASSET_LOSS,
    COLUMNS_ASSETS,
    COLUMNS_PROCUREMENT,
    COLUMNS_SENSITIVITY,
    COLUMNS_STRATEGY_COMPARISON,
    CSV_DELIMITER,
    CSV_ENCODING,
    FRESHNESS_COEF,
    LEGAL_EDITION,
    MASK_DRY,
    MASK_NODATA,
    MASK_WATER,
    MIN_ORDER_AREA_KM2,
    PRICE_FORMULA,
    PROB_NODATA,
    PROCESSING_COEF,
    PROPERTIES_CANDIDATE,
    STRATEGIES,
    STRATEGY_BROAD,
    STRATEGY_OPEN,
    STRATEGY_SELECTIVE,
    USAGE_COEF,
    quantize_money,
)
from src.export.bundle import BundlePaths
from src.export.writers import SOURCE_CHIP_KEYS, THRESHOLD_KEYS, find_nested

__all__ = ["CheckResult", "validate", "format_report", "main", "STATUS_OK", "STATUS_SKIP", "STATUS_FAIL"]

STATUS_OK = "OK"
STATUS_SKIP = "ПРОПУЩЕНО"
STATUS_FAIL = "ОШИБКА"

#: Допуск сравнения вероятности с порогом: p хранится во Float32, порог — в JSON.
THRESHOLD_TOLERANCE = 1e-6

#: Допуск сравнения долей и координат.
SHARE_TOLERANCE = 1e-6

#: Текстовые маски нечисел, которые в пакете запрещены в любом регистре.
_BAD_TOKENS = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nat", "none", "null"}

# Геометрию зон считает соседний модуль. Пока его нет — проверка 6 частично пропускается.
try:  # pragma: no cover - ветка зависит от наличия чужого модуля
    from src.procurement.geometry import (  # type: ignore
        area_km2 as geometry_area_km2,
        area_km2_raw as geometry_area_km2_raw,
        validate_polygon as geometry_validate_polygon,
    )

    GEOMETRY_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    geometry_area_km2 = None  # type: ignore[assignment]
    geometry_area_km2_raw = None  # type: ignore[assignment]
    geometry_validate_polygon = None  # type: ignore[assignment]
    GEOMETRY_ERROR = f"{type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# Результат проверки
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Итог одной проверки: номер, название, статус и подробности."""

    number: int
    title: str
    status: str
    details: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAIL


class _Check:
    """Накопитель замечаний одной проверки.

    Ошибка перевешивает пропуск: если что-то удалось проверить и оно неверно,
    проверка проваливается, даже когда другая её часть была пропущена.
    """

    def __init__(self, number: int, title: str) -> None:
        self.number = number
        self.title = title
        self._problems: list[str] = []
        self._skips: list[str] = []
        self._notes: list[str] = []

    def fail(self, message: str) -> None:
        self._problems.append(message)

    def skip(self, message: str) -> None:
        self._skips.append(message)

    def note(self, message: str) -> None:
        self._notes.append(message)

    @property
    def has_problems(self) -> bool:
        """Нашлось ли уже расхождение — чтобы не дописывать успокоительных заметок."""
        return bool(self._problems)

    def result(self) -> CheckResult:
        if self._problems:
            status = STATUS_FAIL
            details = list(self._problems) + [f"пропущено: {s}" for s in self._skips] + self._notes
        elif self._skips:
            status = STATUS_SKIP
            details = list(self._skips) + self._notes
        else:
            status = STATUS_OK
            details = list(self._notes)
        return CheckResult(self.number, self.title, status, details)


# ─────────────────────────────────────────────────────────────────────────────
# Чтение файлов пакета
# ─────────────────────────────────────────────────────────────────────────────


class _BundleReadError(Exception):
    """Файл пакета не читается так, как обещано контрактом."""


def _reject_constant(token: str) -> Any:
    raise _BundleReadError(f"в JSON встретился запрещённый литерал {token}")


def _read_json(path: Path) -> Any:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise _BundleReadError("файл начинается с BOM, ожидается чистый UTF-8")
    try:
        text = raw.decode(CSV_ENCODING)
    except UnicodeDecodeError as exc:
        raise _BundleReadError(f"файл не в UTF-8: {exc}") from exc
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise _BundleReadError(f"некорректный JSON: {exc}") from exc


@dataclass
class _Table:
    """Прочитанный CSV: заголовок и строки словарями строк."""

    header: list[str]
    rows: list[dict[str, str]]


def _read_csv(path: Path) -> _Table:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise _BundleReadError("файл начинается с BOM, ожидается чистый UTF-8 без метки порядка")
    try:
        text = raw.decode(CSV_ENCODING)
    except UnicodeDecodeError as exc:
        raise _BundleReadError(f"файл не в UTF-8: {exc}") from exc
    lines = text.splitlines()
    if not lines:
        raise _BundleReadError("файл пуст")
    head = lines[0]
    if CSV_DELIMITER not in head and (";" in head or "\t" in head):
        raise _BundleReadError("разделитель не запятая: в заголовке найдены ';' или табуляция")
    reader = csv.reader(lines, delimiter=CSV_DELIMITER)
    records = list(reader)
    header = records[0]
    rows: list[dict[str, str]] = []
    for number, record in enumerate(records[1:], start=1):
        if not record:
            continue
        if len(record) != len(header):
            raise _BundleReadError(
                f"строка {number}: колонок {len(record)}, а в заголовке {len(header)}"
            )
        rows.append({name: value for name, value in zip(header, record)})
    return _Table(header=header, rows=rows)


@dataclass
class _Bundle:
    """Прочитанное содержимое пакета плюс ошибки чтения."""

    run_dir: Path
    paths: BundlePaths
    tables: dict[str, _Table] = field(default_factory=dict)
    documents: dict[str, Any] = field(default_factory=dict)
    read_errors: dict[str, str] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def table(self, name: str) -> _Table | None:
        return self.tables.get(name)

    def document(self, name: str) -> Any:
        return self.documents.get(name)


#: Ожидаемые обязательные колонки каждого CSV пакета.
_CSV_COLUMNS: Mapping[str, Sequence[str]] = {
    "assets.csv": COLUMNS_ASSETS,
    "asset_loss.csv": COLUMNS_ASSET_LOSS,
    "procurement_plan.csv": COLUMNS_PROCUREMENT,
    "strategy_comparison.csv": COLUMNS_STRATEGY_COMPARISON,
    "sensitivity.csv": COLUMNS_SENSITIVITY,
}

_JSON_FILES = ("assets.geojson", "candidate_orders.geojson", "strategy_plans.json", "run_metadata.json", "source_manifest.json")


def _load(run_dir: Path) -> _Bundle:
    """Читает всё, что читается. Нечитаемое попадает в ``read_errors``."""
    bundle = _Bundle(run_dir=run_dir, paths=BundlePaths.for_run(run_dir))
    for name in BUNDLE_FILES:
        path = run_dir / name
        if not path.is_file():
            bundle.missing.append(name)
            continue
        if path.stat().st_size == 0:
            bundle.read_errors[name] = "файл пустой"
            continue
        try:
            if name in _CSV_COLUMNS:
                bundle.tables[name] = _read_csv(path)
            elif name in _JSON_FILES:
                bundle.documents[name] = _read_json(path)
        except _BundleReadError as exc:
            bundle.read_errors[name] = str(exc)
        except Exception as exc:  # неожиданное, но отчёт должен дойти до конца
            bundle.read_errors[name] = f"{type(exc).__name__}: {exc}"
    return bundle


# ─────────────────────────────────────────────────────────────────────────────
# Мелкие помощники разбора значений
# ─────────────────────────────────────────────────────────────────────────────


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _decimal(value: str | None) -> Decimal | None:
    """Строка CSV в Decimal. ``None``, если значение пустое или не число."""
    if _is_blank(value):
        return None
    try:
        parsed = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        return None
    return parsed


def _float(value: Any) -> float | None:
    if _is_blank(value):
        return None
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if _is_blank(value):
        return None
    text = str(value).strip().lower()
    if text in ("true", "1"):
        return True
    if text in ("false", "0"):
        return False
    return None


def _iter_numbers(data: Any, where: str) -> Iterable[tuple[str, float]]:
    """Все числа структуры JSON с путём до них — для поиска NaN и Inf."""
    if isinstance(data, bool):
        return
    if isinstance(data, float):
        yield where, data
    elif isinstance(data, Mapping):
        for key, value in data.items():
            yield from _iter_numbers(value, f"{where}.{key}")
    elif isinstance(data, (list, tuple)):
        for index, value in enumerate(data):
            yield from _iter_numbers(value, f"{where}[{index}]")


def _features(document: Any) -> list[Mapping[str, Any]]:
    if isinstance(document, Mapping) and isinstance(document.get("features"), list):
        return [f for f in document["features"] if isinstance(f, Mapping)]
    return []


def _properties(feature: Mapping[str, Any]) -> Mapping[str, Any]:
    props = feature.get("properties")
    return props if isinstance(props, Mapping) else {}


def _open_raster(path: Path):
    return rasterio.open(path)


# ─────────────────────────────────────────────────────────────────────────────
# Проверка 1. Состав пакета
# ─────────────────────────────────────────────────────────────────────────────


def _check_files(bundle: _Bundle) -> CheckResult:
    check = _Check(1, "Все 12 файлов на месте, кодировки и разделители верные")
    for name in bundle.missing:
        check.fail(f"{name}: файла нет в каталоге запуска")
    for name, error in bundle.read_errors.items():
        check.fail(f"{name}: {error}")

    for name, columns in _CSV_COLUMNS.items():
        table = bundle.table(name)
        if table is None:
            continue
        absent = [column for column in columns if column not in table.header]
        if absent:
            check.fail(f"{name}: нет обязательных колонок {absent}")
            continue
        if list(table.header[: len(columns)]) != list(columns):
            check.note(f"{name}: порядок обязательных колонок отличается от контрактного")
        if not table.rows:
            if name == "procurement_plan.csv":
                check.note("procurement_plan.csv: платных позиций нет, стратегии B и C пусты")
            else:
                check.fail(f"{name}: нет ни одной строки данных")

    for name in ("assets.geojson", "candidate_orders.geojson"):
        document = bundle.document(name)
        if document is None:
            continue
        if not isinstance(document, Mapping) or document.get("type") != "FeatureCollection":
            check.fail(f"{name}: ожидается FeatureCollection")

    for name, dtype, nodata in (
        ("flood_probability.tif", "float32", PROB_NODATA),
        ("flood_mask.tif", "uint8", MASK_NODATA),
    ):
        path = bundle.run_dir / name
        if not path.is_file():
            continue
        try:
            with _open_raster(path) as src:
                if src.count != 1:
                    check.fail(f"{name}: каналов {src.count}, ожидается 1")
                if src.dtypes[0] != dtype:
                    check.fail(f"{name}: тип {src.dtypes[0]}, ожидается {dtype}")
                if src.nodata is None or float(src.nodata) != float(nodata):
                    check.fail(f"{name}: nodata {src.nodata}, ожидается {nodata}")
        except Exception as exc:
            check.fail(f"{name}: не открывается как GeoTIFF ({type(exc).__name__}: {exc})")
    return check.result()


# ─────────────────────────────────────────────────────────────────────────────
# Проверка 2. NaN, Inf и пустые поля
# ─────────────────────────────────────────────────────────────────────────────


def _check_no_nan(bundle: _Bundle) -> CheckResult:
    check = _Check(2, "Нет NaN и Inf, пустое там, где должно быть пустым")

    for name, table in bundle.tables.items():
        for index, row in enumerate(table.rows, start=1):
            for column, value in row.items():
                text = (value or "").strip()
                if text.lower() in _BAD_TOKENS:
                    check.fail(f"{name}, строка {index}, поле {column}: запрещённое значение {text!r}")
                    continue
                if not text:
                    continue
                try:
                    number = float(text)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    check.fail(f"{name}, строка {index}, поле {column}: нечисло {text!r}")

    for name, document in bundle.documents.items():
        for where, number in _iter_numbers(document, name):
            if not math.isfinite(number):
                check.fail(f"{where}: нечисло {number!r}")

    for name in ("flood_probability.tif", "flood_mask.tif"):
        path = bundle.run_dir / name
        if not path.is_file():
            continue
        try:
            with _open_raster(path) as src:
                data = src.read(1)
        except Exception as exc:
            check.fail(f"{name}: не читается ({type(exc).__name__}: {exc})")
            continue
        if name == "flood_probability.tif":
            valid = data != np.float32(PROB_NODATA)
            if not np.isfinite(data[valid]).all():
                check.fail("flood_probability.tif: NaN или Inf среди валидных пикселей")
            elif valid.any():
                low, high = float(data[valid].min()), float(data[valid].max())
                if low < 0.0 or high > 1.0:
                    check.fail(f"flood_probability.tif: вероятность вне [0;1], диапазон [{low}; {high}]")
        else:
            extra = sorted(set(int(v) for v in np.unique(data)) - {MASK_DRY, MASK_WATER, MASK_NODATA})
            if extra:
                check.fail(f"flood_mask.tif: посторонние значения {extra}, допустимы 0, 1 и 255")

    loss = bundle.table("asset_loss.csv")
    if loss is not None:
        blank_columns = ("p_flood", "expected_loss_rub", "rank")
        for index, row in enumerate(loss.rows, start=1):
            status = (row.get("status") or "").strip()
            if status not in ASSET_STATUSES:
                check.fail(f"asset_loss.csv, строка {index}: статус {status!r} не из {list(ASSET_STATUSES)}")
                continue
            if status == ASSET_STATUS_OK:
                for column in blank_columns:
                    if _is_blank(row.get(column)):
                        check.fail(f"asset_loss.csv, строка {index}: при статусе ok поле {column} пустое")
            else:
                for column in blank_columns:
                    if not _is_blank(row.get(column)):
                        check.fail(
                            f"asset_loss.csv, строка {index}: при статусе {status} поле {column} "
                            f"должно быть пустым, а там {row.get(column)!r}"
                        )

    comparison = bundle.table("strategy_comparison.csv")
    if comparison is not None:
        for index, row in enumerate(comparison.rows, start=1):
            if _is_blank(row.get("coverage_share")) and _is_blank(row.get("uncertainty_basis")):
                check.note(
                    f"strategy_comparison.csv, строка {index}: coverage_share пуст, "
                    "причина пустоты в файле не указана"
                )
    return check.result()


# ─────────────────────────────────────────────────────────────────────────────
# Проверка 3. Маска и порог
# ─────────────────────────────────────────────────────────────────────────────


def _threshold(bundle: _Bundle) -> float | None:
    meta = bundle.document("run_metadata.json")
    if meta is None:
        return None
    return _float(find_nested(meta, THRESHOLD_KEYS))


def _check_mask_threshold(bundle: _Bundle) -> CheckResult:
    check = _Check(3, "Маска соответствует порогу, nodata растров согласованы")
    prob_path = bundle.run_dir / "flood_probability.tif"
    mask_path = bundle.run_dir / "flood_mask.tif"
    if not prob_path.is_file() or not mask_path.is_file():
        check.skip("нет одного из растров, сравнивать нечего")
        return check.result()
    threshold = _threshold(bundle)
    if threshold is None:
        check.fail(
            "в run_metadata.json не найден порог бинаризации "
            f"(искали поля {list(THRESHOLD_KEYS)}), проверить соответствие маски нечем"
        )
        return check.result()
    try:
        with _open_raster(prob_path) as src:
            prob = src.read(1)
        with _open_raster(mask_path) as src:
            mask = src.read(1)
    except Exception as exc:
        check.fail(f"растры не читаются: {type(exc).__name__}: {exc}")
        return check.result()
    if prob.shape != mask.shape:
        check.fail(f"размеры растров различаются: {prob.shape} и {mask.shape}")
        return check.result()

    prob_nodata = prob == np.float32(PROB_NODATA)
    mask_nodata = mask == MASK_NODATA
    mismatch_nodata = prob_nodata != mask_nodata
    if mismatch_nodata.any():
        rows, cols = np.nonzero(mismatch_nodata)
        check.fail(
            f"nodata не согласованы в {int(mismatch_nodata.sum())} пикселях, "
            f"первый — строка {int(rows[0])}, столбец {int(cols[0])}"
        )

    valid = (~prob_nodata) & (~mask_nodata)
    expected_water = prob.astype(np.float64) >= (threshold - THRESHOLD_TOLERANCE)
    actual_water = mask == MASK_WATER
    wrong = valid & (expected_water != actual_water)
    if wrong.any():
        near_boundary = np.abs(prob.astype(np.float64) - threshold) <= THRESHOLD_TOLERANCE
        wrong = wrong & (~near_boundary)
    if wrong.any():
        rows, cols = np.nonzero(wrong)
        row, col = int(rows[0]), int(cols[0])
        check.fail(
            f"маска не совпадает с правилом p >= {threshold} в {int(wrong.sum())} валидных пикселях, "
            f"первый — строка {row}, столбец {col}: p={float(prob[row, col])}, mask={int(mask[row, col])}"
        )
    else:
        check.note(f"порог {threshold}, валидных пикселей {int(valid.sum())}, воды {int((valid & actual_water).sum())}")
    return check.result()


# ─────────────────────────────────────────────────────────────────────────────
# Проверка 4. Сетка растров и исходный чип
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_source(bundle: _Bundle, raw: str) -> Path | None:
    """Путь к исходному чипу: абсолютный, от каталога запуска или от корня проекта."""
    candidate = Path(str(raw))
    variants = [candidate, bundle.run_dir / candidate, Path.cwd() / candidate]
    project_root = Path(__file__).resolve().parents[1]
    variants.append(project_root / candidate)
    for variant in variants:
        if variant.is_file():
            return variant
    return None


def _check_grid(bundle: _Bundle) -> CheckResult:
    check = _Check(4, "Растры совпадают с исходным чипом по CRS, transform и размеру")
    prob_path = bundle.run_dir / "flood_probability.tif"
    mask_path = bundle.run_dir / "flood_mask.tif"
    if not prob_path.is_file() or not mask_path.is_file():
        check.skip("нет одного из растров")
        return check.result()
    try:
        with _open_raster(prob_path) as src:
            prob_grid = (src.crs, src.transform, src.width, src.height)
        with _open_raster(mask_path) as src:
            mask_grid = (src.crs, src.transform, src.width, src.height)
    except Exception as exc:
        check.fail(f"растры не читаются: {type(exc).__name__}: {exc}")
        return check.result()

    labels = ("CRS", "transform", "ширина", "высота")
    for label, left, right in zip(labels, prob_grid, mask_grid):
        if left != right:
            check.fail(f"flood_probability.tif и flood_mask.tif различаются по {label}: {left} и {right}")

    meta = bundle.document("run_metadata.json")
    if meta is None:
        check.skip("run_metadata.json не прочитан, путь к исходному чипу неизвестен")
        return check.result()
    raw_path = find_nested(meta, SOURCE_CHIP_KEYS)
    if _is_blank(raw_path):
        check.skip(
            "в run_metadata.json нет пути к исходному S1-чипу "
            f"(искали поля {list(SOURCE_CHIP_KEYS)}), сверка с источником невозможна"
        )
        return check.result()
    source = _resolve_source(bundle, str(raw_path))
    if source is None:
        check.skip(f"исходный чип {raw_path!r} недоступен на этой машине, сверка с источником невозможна")
        return check.result()
    try:
        with _open_raster(source) as src:
            source_grid = (src.crs, src.transform, src.width, src.height)
    except Exception as exc:
        check.skip(f"исходный чип {source} не читается ({type(exc).__name__}: {exc})")
        return check.result()
    for label, left, right in zip(labels, prob_grid, source_grid):
        if left != right:
            check.fail(f"растры пакета расходятся с исходным чипом по {label}: {left} и {right}")
    if not check.has_problems:
        check.note(f"сверено с исходным чипом {source}")
    return check.result()


# ─────────────────────────────────────────────────────────────────────────────
# Проверка 5. Портфель объектов
# ─────────────────────────────────────────────────────────────────────────────


def _check_assets(bundle: _Bundle) -> CheckResult:
    check = _Check(5, "Ровно 10 объектов, значения из таблицы, координаты сходятся")
    geo = bundle.document("assets.geojson")
    table = bundle.table("assets.csv")
    if geo is None or table is None:
        check.skip("assets.geojson или assets.csv не прочитаны")
        return check.result()

    features = _features(geo)
    if len(features) != ASSET_COUNT:
        check.fail(f"assets.geojson: объектов {len(features)}, должно быть {ASSET_COUNT}")
    if len(table.rows) != ASSET_COUNT:
        check.fail(f"assets.csv: строк {len(table.rows)}, должно быть {ASSET_COUNT}")

    geo_points: dict[str, tuple[float, float]] = {}
    geo_classes: list[str] = []
    for index, feature in enumerate(features, start=1):
        props = _properties(feature)
        asset_id = str(props.get("asset_id", "")).strip()
        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping) or geometry.get("type") != "Point":
            check.fail(f"assets.geojson, объект {index}: геометрия должна быть Point")
            continue
        coords = geometry.get("coordinates")
        if not isinstance(coords, (list, tuple)) or len(coords) < 2:
            check.fail(f"assets.geojson, объект {index}: координаты должны быть [долгота, широта]")
            continue
        lon, lat = _float(coords[0]), _float(coords[1])
        if lon is None or lat is None:
            check.fail(f"assets.geojson, объект {index}: координаты не числа")
            continue
        if not asset_id:
            check.fail(f"assets.geojson, объект {index}: пустой asset_id")
            continue
        if asset_id in geo_points:
            check.fail(f"assets.geojson: asset_id {asset_id!r} повторяется")
        geo_points[asset_id] = (lon, lat)
        geo_classes.append(str(props.get("asset_class", "")).strip())

    csv_points: dict[str, tuple[float, float]] = {}
    csv_classes: list[str] = []
    for index, row in enumerate(table.rows, start=1):
        asset_id = (row.get("asset_id") or "").strip()
        if not asset_id:
            check.fail(f"assets.csv, строка {index}: пустой asset_id")
            continue
        if asset_id in csv_points:
            check.fail(f"assets.csv: asset_id {asset_id!r} повторяется")
        lon, lat = _float(row.get("longitude")), _float(row.get("latitude"))
        if lon is None or lat is None:
            check.fail(f"assets.csv, строка {index}: координаты не читаются")
            continue
        csv_points[asset_id] = (lon, lat)
        asset_class = (row.get("asset_class") or "").strip()
        csv_classes.append(asset_class)
        known = ASSET_TYPE_BY_CLASS.get(asset_class)
        if known is None:
            check.fail(f"assets.csv, строка {index}: тип {asset_class!r} не из таблицы кейса")
            continue
        value = _decimal(row.get("asset_value_rub"))
        vulnerability = _decimal(row.get("vulnerability_coef"))
        if value is None or value != Decimal(known.value_rub):
            check.fail(
                f"assets.csv, строка {index}: V={row.get('asset_value_rub')!r} "
                f"вместо {known.value_rub} для типа {asset_class}"
            )
        if vulnerability is None or vulnerability != Decimal(str(known.vulnerability)):
            check.fail(
                f"assets.csv, строка {index}: q={row.get('vulnerability_coef')!r} "
                f"вместо {known.vulnerability} для типа {asset_class}"
            )

    expected_classes = sorted(ASSET_TYPE_BY_CLASS)
    if sorted(csv_classes) != expected_classes:
        check.fail(
            "assets.csv: состав портфеля не совпадает с таблицей кейса, "
            f"должно быть ровно по одному объекту каждого из {len(expected_classes)} типов"
        )
    if geo_classes and sorted(geo_classes) != expected_classes:
        check.fail("assets.geojson: состав портфеля не совпадает с таблицей кейса")

    if set(geo_points) != set(csv_points):
        only_geo = sorted(set(geo_points) - set(csv_points))
        only_csv = sorted(set(csv_points) - set(geo_points))
        check.fail(f"составы файлов различаются: только в GeoJSON {only_geo}, только в CSV {only_csv}")
    for asset_id in sorted(set(geo_points) & set(csv_points)):
        (geo_lon, geo_lat), (csv_lon, csv_lat) = geo_points[asset_id], csv_points[asset_id]
        if geo_lon != csv_lon or geo_lat != csv_lat:
            check.fail(
                f"{asset_id}: координаты расходятся, GeoJSON ({geo_lon}, {geo_lat}), "
                f"CSV ({csv_lon}, {csv_lat})"
            )

    meta = bundle.document("run_metadata.json")
    meta_chip = str(find_nested(meta, ("chip_id",)) or "").strip() if meta is not None else ""
    if meta_chip:
        for index, row in enumerate(table.rows, start=1):
            if (row.get("chip_id") or "").strip() != meta_chip:
                check.fail(
                    f"assets.csv, строка {index}: chip_id {row.get('chip_id')!r} "
                    f"не совпадает с паспортом запуска {meta_chip!r}"
                )
                break

    prob_path = bundle.run_dir / "flood_probability.tif"
    if not prob_path.is_file():
        check.skip("нет flood_probability.tif, попадание точек в растр не проверено")
        return check.result()
    try:
        with _open_raster(prob_path) as src:
            left, bottom, right, top = src.bounds
    except Exception as exc:
        check.skip(f"растр не читается ({type(exc).__name__}: {exc}), попадание точек не проверено")
        return check.result()
    for asset_id, (lon, lat) in sorted(csv_points.items()):
        if not (left <= lon <= right and bottom <= lat <= top):
            check.fail(
                f"{asset_id}: точка ({lon}, {lat}) вне растра "
                f"[{left}; {right}] × [{bottom}; {top}]"
            )
    return check.result()


# ─────────────────────────────────────────────────────────────────────────────
# Проверка 6. Геометрия кандидатных зон
# ─────────────────────────────────────────────────────────────────────────────


def _polygon_violations(polygon: Any) -> list[str]:
    """Нарушения требований к зоне заказа по мнению src.procurement.geometry.

    Соседний модуль возвращает список нарушений, но валидатор не обязан знать
    об этом наверняка: исключение тоже трактуется как нарушение.
    """
    if geometry_validate_polygon is None:
        return []
    try:
        reported = geometry_validate_polygon(polygon)
    except Exception as exc:
        return [f"{type(exc).__name__}: {exc}"]
    if not reported:
        return []
    if isinstance(reported, (list, tuple, set)):
        return [str(item) for item in reported]
    return [str(reported)]


def _check_candidates(bundle: _Bundle) -> CheckResult:
    check = _Check(6, "Зоны заказа: валидный полигон без вырезов, площадь и уникальность id")
    geo = bundle.document("candidate_orders.geojson")
    if geo is None:
        check.skip("candidate_orders.geojson не прочитан")
        return check.result()
    features = _features(geo)
    if not features:
        check.fail("candidate_orders.geojson: каталог зон пуст")
        return check.result()

    seen: set[str] = set()
    for index, feature in enumerate(features, start=1):
        props = _properties(feature)
        candidate_id = str(props.get("candidate_id", "")).strip()
        where = f"зона {index}" + (f" ({candidate_id})" if candidate_id else "")
        if not candidate_id:
            check.fail(f"{where}: пустой candidate_id")
        elif candidate_id in seen:
            check.fail(f"candidate_id {candidate_id!r} повторяется — идентификаторы зон должны быть уникальны")
        else:
            seen.add(candidate_id)
        missing = [name for name in PROPERTIES_CANDIDATE if _is_blank(props.get(name))]
        if missing:
            check.fail(f"{where}: не заполнены свойства {missing}")

        geometry = feature.get("geometry")
        if not isinstance(geometry, Mapping) or geometry.get("type") != "Polygon":
            check.fail(f"{where}: геометрия должна быть Polygon")
            continue
        rings = geometry.get("coordinates")
        if not isinstance(rings, (list, tuple)) or not rings:
            check.fail(f"{where}: у полигона нет колец координат")
            continue
        if len(rings) > 1:
            check.fail(f"{where}: полигон с вырезами ({len(rings) - 1} внутренних колец), вырезы запрещены")
            continue
        try:
            polygon = shapely_shape(geometry)
        except Exception as exc:
            check.fail(f"{where}: геометрия не разбирается ({type(exc).__name__}: {exc})")
            continue
        if polygon.is_empty:
            check.fail(f"{where}: пустая геометрия")
            continue
        if not polygon.is_valid:
            check.fail(f"{where}: невалидный полигон — {explain_validity(polygon)}")
            continue
        if list(polygon.interiors):
            check.fail(f"{where}: у полигона есть внутренние вырезы")
            continue

        if geometry_area_km2 is None:
            continue
        try:
            raw_area = float(geometry_area_km2_raw(polygon))
            rounded = Decimal(str(geometry_area_km2(polygon)))
        except Exception as exc:
            check.fail(f"{where}: площадь не считается ({type(exc).__name__}: {exc})")
            continue
        violations = _polygon_violations(polygon)
        for violation in violations:
            check.fail(f"{where}: {violation}")
        if not violations and raw_area < MIN_ORDER_AREA_KM2:
            check.fail(
                f"{where}: площадь {raw_area} км² меньше минимальной {MIN_ORDER_AREA_KM2} км² "
                "(проверяется до округления)"
            )
        written = _decimal(props.get("area_km2"))
        if written is None:
            check.fail(f"{where}: area_km2 не число")
        elif written != rounded:
            check.fail(f"{where}: записанная площадь {written} км², вычисленная {rounded} км²")

    if geometry_area_km2 is None:
        check.skip(
            "модуль src.procurement.geometry недоступен "
            f"({GEOMETRY_ERROR}), площади зон не сверены — проверены только структура полигонов "
            "и уникальность candidate_id"
        )
    return check.result()


# ─────────────────────────────────────────────────────────────────────────────
# Проверка 7. Согласованность планов
# ─────────────────────────────────────────────────────────────────────────────


def _procurement_by_strategy(table: _Table) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for row in table.rows:
        result.setdefault((row.get("strategy") or "").strip(), []).append(row)
    return result


def _check_plans(bundle: _Bundle) -> CheckResult:
    check = _Check(7, "strategy_plans.json, procurement_plan.csv и каталог зон согласованы")
    plans = bundle.document("strategy_plans.json")
    procurement = bundle.table("procurement_plan.csv")
    catalogue = bundle.document("candidate_orders.geojson")
    if plans is None or procurement is None or catalogue is None:
        check.skip("один из трёх файлов не прочитан")
        return check.result()
    if not isinstance(plans, Mapping):
        check.fail("strategy_plans.json: ожидается объект {\"A\": [], \"B\": [...], \"C\": [...]}")
        return check.result()

    known_ids = {
        str(_properties(feature).get("candidate_id", "")).strip()
        for feature in _features(catalogue)
    }
    known_ids.discard("")
    by_strategy = _procurement_by_strategy(procurement)
    for strategy in by_strategy:
        if strategy not in STRATEGIES:
            check.fail(f"procurement_plan.csv: неизвестная стратегия {strategy!r}")
    if by_strategy.get(STRATEGY_OPEN):
        check.fail("procurement_plan.csv: у стратегии A не может быть платных позиций")

    catalogue_props = {
        str(_properties(f).get("candidate_id", "")).strip(): _properties(f)
        for f in _features(catalogue)
    }

    for strategy in STRATEGIES:
        planned = plans.get(strategy)
        if planned is None:
            check.fail(f"strategy_plans.json: нет ключа {strategy!r}")
            continue
        if not isinstance(planned, list):
            check.fail(f"strategy_plans.json: план стратегии {strategy} должен быть списком")
            continue
        ids = [str(item).strip() for item in planned]
        if len(set(ids)) != len(ids):
            check.fail(f"strategy_plans.json: в плане стратегии {strategy} повторяются candidate_id")
        if strategy == STRATEGY_OPEN and ids:
            check.fail("strategy_plans.json: стратегия A работает на открытых данных, её список должен быть пустым")
        unknown = sorted(set(ids) - known_ids)
        if unknown:
            check.fail(f"стратегия {strategy}: зон {unknown} нет в candidate_orders.geojson")
        rows = by_strategy.get(strategy, [])
        row_ids = [(row.get("candidate_id") or "").strip() for row in rows]
        if len(set(row_ids)) != len(row_ids):
            check.fail(f"procurement_plan.csv: у стратегии {strategy} повторяется пара (strategy, candidate_id)")
        if set(ids) != set(row_ids):
            only_plan = sorted(set(ids) - set(row_ids))
            only_rows = sorted(set(row_ids) - set(ids))
            check.fail(
                f"стратегия {strategy}: план и строки закупки расходятся, "
                f"только в плане {only_plan}, только в CSV {only_rows}"
            )
        for row in rows:
            candidate_id = (row.get("candidate_id") or "").strip()
            props = catalogue_props.get(candidate_id)
            if props is None:
                continue
            plan_area = _decimal(row.get("area_km2"))
            catalogue_area = _decimal(str(props.get("area_km2")))
            if plan_area is not None and catalogue_area is not None and plan_area != catalogue_area:
                check.fail(
                    f"стратегия {strategy}, зона {candidate_id}: площадь в закупке {plan_area} км², "
                    f"в каталоге {catalogue_area} км²"
                )
            plan_rate = _decimal(row.get("base_rate_rub_km2"))
            catalogue_rate = _decimal(str(props.get("base_rate_rub_km2")))
            if plan_rate is not None and catalogue_rate is not None and plan_rate != catalogue_rate:
                check.fail(
                    f"стратегия {strategy}, зона {candidate_id}: ставка в закупке {plan_rate}, "
                    f"в каталоге {catalogue_rate}"
                )
            for column in ("processing_level", "usage_type"):
                plan_value = (row.get(column) or "").strip()
                catalogue_value = str(props.get(column, "")).strip()
                if plan_value and catalogue_value and plan_value != catalogue_value:
                    check.note(
                        f"стратегия {strategy}, зона {candidate_id}: {column} в закупке {plan_value!r}, "
                        f"в каталоге {catalogue_value!r}"
                    )
    return check.result()


# ─────────────────────────────────────────────────────────────────────────────
# Проверка 8. Суммы и бюджет
# ─────────────────────────────────────────────────────────────────────────────


def _comparison_rows(table: _Table) -> dict[str, dict[str, str]]:
    return {(row.get("strategy") or "").strip(): row for row in table.rows}


def _check_costs(bundle: _Bundle) -> CheckResult:
    check = _Check(8, "Суммы закупки равны сравнению, бюджет соблюдён")
    procurement = bundle.table("procurement_plan.csv")
    comparison = bundle.table("strategy_comparison.csv")
    if procurement is None or comparison is None:
        check.skip("procurement_plan.csv или strategy_comparison.csv не прочитаны")
        return check.result()

    rows = _comparison_rows(comparison)
    if len(rows) != len(comparison.rows):
        check.fail("strategy_comparison.csv: стратегия встречается больше одного раза")
    for strategy in STRATEGIES:
        if strategy not in rows:
            check.fail(f"strategy_comparison.csv: нет строки стратегии {strategy}")

    sums: dict[str, Decimal] = {strategy: Decimal("0") for strategy in STRATEGIES}
    for index, row in enumerate(procurement.rows, start=1):
        strategy = (row.get("strategy") or "").strip()
        cost = _decimal(row.get("cost_rub"))
        if cost is None:
            check.fail(f"procurement_plan.csv, строка {index}: cost_rub не число")
            continue
        if strategy in sums:
            sums[strategy] += cost

    for strategy, row in rows.items():
        if strategy not in STRATEGIES:
            check.fail(f"strategy_comparison.csv: неизвестная стратегия {strategy!r}")
            continue
        data_cost = _decimal(row.get("data_cost_rub"))
        other_cost = _decimal(row.get("other_cost_rub"))
        decision_cost = _decimal(row.get("decision_cost_rub"))
        budget = _decimal(row.get("budget_rub"))
        feasible = _boolean(row.get("budget_feasible"))
        if data_cost is None:
            check.fail(f"стратегия {strategy}: data_cost_rub не число")
        elif quantize_money(data_cost) != quantize_money(sums[strategy]):
            check.fail(
                f"стратегия {strategy}: сумма cost_rub по закупке {quantize_money(sums[strategy])} руб., "
                f"в сравнении data_cost_rub {quantize_money(data_cost)} руб."
            )
        if strategy == STRATEGY_OPEN and data_cost is not None and data_cost != 0:
            check.fail(f"стратегия A закупает только открытые данные, её data_cost_rub должен быть 0, а он {data_cost}")
        if data_cost is not None and other_cost is not None and decision_cost is not None:
            if quantize_money(decision_cost) != quantize_money(data_cost + other_cost):
                check.fail(
                    f"стратегия {strategy}: decision_cost_rub {decision_cost} не равен "
                    f"data_cost_rub + other_cost_rub = {data_cost + other_cost}"
                )
        if decision_cost is None or budget is None:
            check.fail(f"стратегия {strategy}: decision_cost_rub или budget_rub не число")
            continue
        if feasible is None:
            check.fail(f"стратегия {strategy}: budget_feasible должен быть true или false")
        over_budget = decision_cost > budget
        if strategy == STRATEGY_SELECTIVE:
            if over_budget:
                check.fail(
                    f"стратегия C вышла за бюджет: {decision_cost} руб. против {budget} руб., "
                    "выборочная закупка обязана укладываться в лимит"
                )
            if feasible is False:
                check.fail("стратегия C помечена budget_feasible=false, хотя обязана быть выполнимой")
        if strategy == STRATEGY_BROAD and over_budget and feasible is not False:
            check.fail(
                f"стратегия B вышла за бюджет ({decision_cost} руб. против {budget} руб.), "
                "значит она контрфактическая и budget_feasible должен быть false"
            )
        if not over_budget and feasible is False:
            check.note(f"стратегия {strategy}: budget_feasible=false при укладывании в бюджет")
    return check.result()


# ─────────────────────────────────────────────────────────────────────────────
# Проверка 9. Охват ожидаемого ущерба
# ─────────────────────────────────────────────────────────────────────────────


def _covered_ids(raw: str) -> list[str]:
    """Необязательная колонка со списком покрытых объектов: JSON-список или через ';'."""
    text = str(raw).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return [str(item).strip() for item in parsed] if isinstance(parsed, list) else []
    return [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]


def _check_coverage(bundle: _Bundle) -> CheckResult:
    check = _Check(9, "Охват ожидаемого ущерба уникален по asset_id, двойного учёта нет")
    loss = bundle.table("asset_loss.csv")
    comparison = bundle.table("strategy_comparison.csv")
    if loss is None or comparison is None:
        check.skip("asset_loss.csv или strategy_comparison.csv не прочитаны")
        return check.result()

    losses: dict[str, Decimal] = {}
    for index, row in enumerate(loss.rows, start=1):
        asset_id = (row.get("asset_id") or "").strip()
        if not asset_id:
            check.fail(f"asset_loss.csv, строка {index}: пустой asset_id")
            continue
        if asset_id in losses:
            check.fail(f"asset_loss.csv: объект {asset_id!r} встречается больше одного раза")
        value = _decimal(row.get("expected_loss_rub"))
        losses[asset_id] = value if value is not None else Decimal("0")

    ranks = [
        _decimal(row.get("rank"))
        for row in loss.rows
        if (row.get("status") or "").strip() == ASSET_STATUS_OK
    ]
    clean_ranks = [int(r) for r in ranks if r is not None]
    if len(set(clean_ranks)) != len(clean_ranks):
        check.fail("asset_loss.csv: ранги повторяются, порядок ущерба неоднозначен")

    total = sum(losses.values(), Decimal("0"))
    for row in comparison.rows:
        strategy = (row.get("strategy") or "").strip()
        covered = _decimal(row.get("covered_expected_loss_rub"))
        share = _decimal(row.get("coverage_share"))
        if covered is None:
            check.fail(f"стратегия {strategy}: covered_expected_loss_rub не число")
            continue
        if covered < 0:
            check.fail(f"стратегия {strategy}: охваченный ущерб отрицателен")
        if covered > total + Decimal("0.01"):
            check.fail(
                f"стратегия {strategy}: охвачено {covered} руб. при суммарном ожидаемом ущербе "
                f"{total} руб. — значит, какой-то объект учтён дважды"
            )
        if total == 0:
            if share is not None:
                check.fail("суммарный ожидаемый ущерб нулевой, coverage_share обязан быть пустым")
        elif share is None:
            check.note(f"стратегия {strategy}: coverage_share пуст при ненулевом знаменателе")
        else:
            expected = covered / total
            if abs(float(share) - float(expected)) > SHARE_TOLERANCE:
                check.fail(
                    f"стратегия {strategy}: coverage_share {share}, а охват/итог даёт {expected}"
                )

        raw_ids = row.get("covered_asset_ids")
        if _is_blank(raw_ids):
            continue
        ids = _covered_ids(raw_ids or "")
        if len(set(ids)) != len(ids):
            check.fail(f"стратегия {strategy}: в списке покрытых объектов есть повторы — это двойной учёт")
        unknown = sorted(set(ids) - set(losses))
        if unknown:
            check.fail(f"стратегия {strategy}: объектов {unknown} нет в asset_loss.csv")
        listed = sum((losses.get(a, Decimal("0")) for a in set(ids)), Decimal("0"))
        if quantize_money(listed) != quantize_money(covered):
            check.fail(
                f"стратегия {strategy}: сумма ущерба перечисленных объектов {quantize_money(listed)} руб., "
                f"а covered_expected_loss_rub {quantize_money(covered)} руб."
            )

    if not any("covered_asset_ids" in row for row in comparison.rows):
        check.note(
            "списка покрытых asset_id в strategy_comparison.csv нет: проверено, что охват "
            "не превышает суммарный ожидаемый ущерб и согласован с coverage_share"
        )
    return check.result()


# ─────────────────────────────────────────────────────────────────────────────
# Проверка 10. Пересчёт цены позиции
# ─────────────────────────────────────────────────────────────────────────────


def _check_pricing(bundle: _Bundle) -> CheckResult:
    check = _Check(10, "Цена каждой позиции пересчитывается из её же множителей")
    procurement = bundle.table("procurement_plan.csv")
    if procurement is None:
        check.skip("procurement_plan.csv не прочитан")
        return check.result()
    if not procurement.rows:
        check.note("платных позиций нет, пересчитывать нечего")
        return check.result()

    for index, row in enumerate(procurement.rows, start=1):
        where = f"procurement_plan.csv, строка {index} ({row.get('strategy')}/{row.get('candidate_id')})"
        numbers: dict[str, Decimal] = {}
        broken = False
        for column in (
            "area_km2",
            "base_rate_rub_km2",
            "processing_coef",
            "usage_coef",
            "freshness_coef",
            "discount_coef",
            "unit_price_rub_km2",
            "cost_rub",
        ):
            value = _decimal(row.get(column))
            if value is None:
                check.fail(f"{where}: поле {column} не число")
                broken = True
            else:
                numbers[column] = value
        if broken:
            continue

        level = (row.get("processing_level") or "").strip()
        expected_processing = PROCESSING_COEF.get(level)
        if expected_processing is None:
            check.fail(f"{where}: уровень обработки {level!r} не из {list(PROCESSING_COEF)}")
        elif numbers["processing_coef"] != expected_processing:
            check.fail(
                f"{where}: processing_coef {numbers['processing_coef']} не соответствует "
                f"уровню {level} (должно быть {expected_processing})"
            )
        usage = (row.get("usage_type") or "").strip()
        expected_usage = USAGE_COEF.get(usage)
        if expected_usage is None:
            check.fail(f"{where}: условия использования {usage!r} не из {list(USAGE_COEF)}")
        elif numbers["usage_coef"] != expected_usage:
            check.fail(
                f"{where}: usage_coef {numbers['usage_coef']} не соответствует "
                f"условиям {usage} (должно быть {expected_usage})"
            )
        if numbers["freshness_coef"] not in set(FRESHNESS_COEF.values()):
            check.fail(
                f"{where}: freshness_coef {numbers['freshness_coef']} не из "
                f"{sorted(str(v) for v in FRESHNESS_COEF.values())}"
            )
        if not (Decimal("0") < numbers["discount_coef"] <= Decimal("1")):
            check.fail(f"{where}: discount_coef {numbers['discount_coef']} вне (0; 1]")

        expected_unit = quantize_money(
            numbers["base_rate_rub_km2"]
            * numbers["processing_coef"]
            * numbers["usage_coef"]
            * numbers["freshness_coef"]
            * numbers["discount_coef"]
        )
        if quantize_money(numbers["unit_price_rub_km2"]) != expected_unit:
            check.fail(
                f"{where}: unit_price_rub_km2 {numbers['unit_price_rub_km2']}, "
                f"а произведение множителей даёт {expected_unit}"
            )
        expected_cost = quantize_money(numbers["unit_price_rub_km2"] * numbers["area_km2"])
        if quantize_money(numbers["cost_rub"]) != expected_cost:
            check.fail(
                f"{where}: cost_rub {numbers['cost_rub']}, а unit_price × area_km2 даёт {expected_cost}"
            )
        formula = (row.get("formula") or "").strip()
        if formula and formula != PRICE_FORMULA:
            check.note(f"{where}: формула {formula!r} отличается от контрактной {PRICE_FORMULA!r}")
        edition = (row.get("legal_edition") or "").strip()
        if edition and edition != LEGAL_EDITION:
            check.note(f"{where}: редакция {edition!r} отличается от контрактной {LEGAL_EDITION!r}")
    return check.result()


# ─────────────────────────────────────────────────────────────────────────────
# Прогон и отчёт
# ─────────────────────────────────────────────────────────────────────────────

_CHECKS = (
    _check_files,
    _check_no_nan,
    _check_mask_threshold,
    _check_grid,
    _check_assets,
    _check_candidates,
    _check_plans,
    _check_costs,
    _check_coverage,
    _check_pricing,
)


def validate(run_dir: Path | str) -> list[CheckResult]:
    """Прогоняет все десять проверок пакета и возвращает их результаты."""
    root = Path(run_dir)
    if not root.is_dir():
        return [
            CheckResult(
                number=1,
                title="Все 12 файлов на месте, кодировки и разделители верные",
                status=STATUS_FAIL,
                details=[f"каталог запуска {root} не найден"],
            )
        ]
    bundle = _load(root)
    return [check(bundle) for check in _CHECKS]


def format_report(run_dir: Path | str, results: Sequence[CheckResult]) -> str:
    """Читаемый отчёт: по строке на проверку, подробности с отступом, итог в конце."""
    width = 72
    lines = [f"Проверка пакета: {Path(run_dir)}", ""]
    for result in results:
        head = f"{result.number:2d}. {result.title} "
        lines.append(f"{head.ljust(width, '.')} {result.status}")
        for detail in result.details:
            lines.append(f"      - {detail}")
    failed = sum(1 for r in results if r.status == STATUS_FAIL)
    skipped = sum(1 for r in results if r.status == STATUS_SKIP)
    passed = sum(1 for r in results if r.status == STATUS_OK)
    lines.append("")
    lines.append(f"Итог: пройдено {passed}, пропущено {skipped}, с ошибками {failed} из {len(results)}.")
    if failed:
        lines.append("Пакет к сдаче не готов: сначала исправить ошибки выше.")
    elif skipped:
        lines.append("Ошибок нет. Часть проверок пропущена — в отчёте указано, чего именно не хватило.")
    else:
        lines.append("Пакет полный и согласованный, все проверки пройдены.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.validate_bundle",
        description="Проверка пакета сдачи Водополь: десять проверок обязательных артефактов.",
    )
    parser.add_argument("run_dir", help="каталог запуска, например outputs/20260926-India_900498-b250000")
    args = parser.parse_args(argv)
    results = validate(args.run_dir)
    report = format_report(args.run_dir, results)
    print(report)
    return 1 if any(result.failed for result in results) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
