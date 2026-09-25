"""Запись обязательных файлов пакета сдачи FloodValue.

На каждый артефакт — своя функция. Ничего не считаем: сюда приходят готовые значения,
здесь они только раскладываются по файлам в договорённом формате.

Общие правила, которые соблюдают все функции модуля:

  * CSV — UTF-8 без BOM, разделитель запятая, десятичная точка, перевод строки ``\\n``,
    отсутствующее значение — пустая строка (не ноль и не строка ``NaN``);
  * JSON и GeoJSON — ``ensure_ascii=False``, ``allow_nan=False``, отступ 2;
  * ``Decimal`` пишется обычным числом без экспоненты;
  * NaN и Inf не попадают в пакет никогда: они ловятся при сериализации и роняют
    запись с понятным сообщением.

Имена файлов и колонок берутся только из :mod:`src.contracts`.
"""

from __future__ import annotations

import csv
import json
import math
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import rasterio

from src.contracts import (
    ASSET_COUNT,
    ASSET_STATUS_OK,
    ASSET_STATUSES,
    COLUMNS_ASSET_LOSS,
    COLUMNS_ASSETS,
    COLUMNS_PROCUREMENT,
    COLUMNS_SENSITIVITY,
    COLUMNS_STRATEGY_COMPARISON,
    CSV_DELIMITER,
    CSV_EMPTY,
    CSV_ENCODING,
    MASK_DRY,
    MASK_NODATA,
    MASK_WATER,
    PROB_NODATA,
    PROPERTIES_CANDIDATE,
    STRATEGIES,
    STRATEGY_OPEN,
)

__all__ = [
    "BundleWriteError",
    "format_value",
    "find_nested",
    "write_csv",
    "write_json",
    "write_probability_tif",
    "write_mask_tif",
    "write_assets",
    "write_asset_loss",
    "write_candidates",
    "write_procurement_plan",
    "write_strategy_plans",
    "write_strategy_comparison",
    "write_sensitivity",
    "write_run_metadata",
    "write_source_manifest",
]


class BundleWriteError(ValueError):
    """Пакет собрать нельзя: данные нарушают контракт выходных файлов."""


# ─────────────────────────────────────────────────────────────────────────────
# Числа и строки
# ─────────────────────────────────────────────────────────────────────────────

#: Маркер, которым ``Decimal`` временно подменяется в JSON, чтобы уйти в файл
#: обычным числом без кавычек и без экспоненты.
_NUM_OPEN = "@@floodvalue-number:"
_NUM_CLOSE = "@@"
_NUM_RE = re.compile(r'"' + re.escape(_NUM_OPEN) + r'([^"]*?)' + re.escape(_NUM_CLOSE) + r'"')
_PLAIN_NUMBER_RE = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$")


def _unwrap(value: Any) -> Any:
    """Разворачивает скаляры numpy в обычные типы Python."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _check_finite(value: float, where: str) -> float:
    """Роняет запись, если число не конечное. NaN и Inf в пакете запрещены."""
    if math.isnan(value):
        raise BundleWriteError(f"NaN недопустим в пакете: {where}")
    if math.isinf(value):
        raise BundleWriteError(f"Inf недопустим в пакете: {where}")
    return value


def _check_finite_decimal(value: Decimal, where: str) -> Decimal:
    if value.is_nan():
        raise BundleWriteError(f"NaN недопустим в пакете: {where}")
    if value.is_infinite():
        raise BundleWriteError(f"Inf недопустим в пакете: {where}")
    return value


def _decimal_text(value: Decimal) -> str:
    """Decimal в обычную запись без экспоненты: ``1E+3`` → ``1000``."""
    return format(value, "f")


def format_value(value: Any, where: str = "значение") -> str:
    """Строковое представление значения для CSV.

    ``None`` — пустая строка, булево — ``true``/``false`` строчными,
    ``Decimal`` и ``float`` — без экспоненты, NaN и Inf роняют запись.
    """
    value = _unwrap(value)
    if value is None:
        return CSV_EMPTY
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, Decimal):
        return _decimal_text(_check_finite_decimal(value, where))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        _check_finite(value, where)
        text = repr(value)
        if "e" in text or "E" in text:
            text = _decimal_text(Decimal(text))
        return text
    raise BundleWriteError(f"неподдерживаемый тип значения {type(value).__name__}: {where}")


def find_nested(data: Any, names: Sequence[str]) -> Any:
    """Первое значение по одному из имён ключей на любой глубине словаря.

    Нужна и писателю (проверить, что порог в паспорте есть), и валидатору
    (достать порог и путь к исходному чипу), поэтому живёт здесь одна.
    Возвращает ``None``, если ни одного такого ключа нет.
    """
    wanted = tuple(names)
    if isinstance(data, Mapping):
        for name in wanted:
            if name in data and data[name] is not None:
                return data[name]
        for nested in data.values():
            found = find_nested(nested, wanted)
            if found is not None:
                return found
    elif isinstance(data, (list, tuple)):
        for nested in data:
            found = find_nested(nested, wanted)
            if found is not None:
                return found
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Общие писатели CSV и JSON
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _columns(required: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Обязательные колонки в контрактном порядке плюс дополнительные — в конец."""
    columns = list(required)
    seen = set(columns)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return columns


def write_csv(
    path: Path | str,
    required_columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """CSV пакета: UTF-8 без BOM, запятая, ``\\n``, пустая строка вместо пропуска."""
    path = _ensure_parent(Path(path))
    columns = _columns(required_columns, rows)
    with path.open("w", encoding=CSV_ENCODING, newline="") as handle:
        writer = csv.writer(handle, delimiter=CSV_DELIMITER, lineterminator="\n")
        writer.writerow(columns)
        for index, row in enumerate(rows):
            cells = [
                format_value(row.get(column), f"{path.name}, строка {index + 1}, поле {column}")
                for column in columns
            ]
            writer.writerow(cells)
    return path


def _sanitize(value: Any, where: str) -> Any:
    """Готовит структуру к json.dump: ловит NaN/Inf, разворачивает Decimal и геометрии."""
    value = _unwrap(value)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if _NUM_OPEN in value:
            raise BundleWriteError(f"служебный маркер чисел попал в текст: {where}")
        return value
    if isinstance(value, Decimal):
        text = _decimal_text(_check_finite_decimal(value, where))
        if not _PLAIN_NUMBER_RE.match(text):
            raise BundleWriteError(f"Decimal не приводится к обычному числу: {where}")
        return f"{_NUM_OPEN}{text}{_NUM_CLOSE}"
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _check_finite(value, where)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise BundleWriteError(f"ключ JSON должен быть строкой: {where}")
            result[key] = _sanitize(nested, f"{where}.{key}")
        return result
    if isinstance(value, np.ndarray):
        return [_sanitize(item, f"{where}[{i}]") for i, item in enumerate(value.tolist())]
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, f"{where}[{i}]") for i, item in enumerate(value)]
    geo = getattr(value, "__geo_interface__", None)
    if geo is not None:
        return _sanitize(geo, where)
    raise BundleWriteError(f"неподдерживаемый тип значения {type(value).__name__}: {where}")


def write_json(path: Path | str, data: Any) -> Path:
    """JSON пакета: UTF-8, ensure_ascii=False, allow_nan=False, отступ 2."""
    path = _ensure_parent(Path(path))
    payload = _sanitize(data, path.name)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
    text = _NUM_RE.sub(lambda m: m.group(1), text)
    path.write_text(text + "\n", encoding=CSV_ENCODING, newline="\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Растры
# ─────────────────────────────────────────────────────────────────────────────


def _raster_profile(
    profile: Mapping[str, Any],
    *,
    dtype: str,
    nodata: float | int,
    shape: tuple[int, int],
) -> dict[str, Any]:
    """Профиль записи: CRS, transform и сетка — как у исходного чипа."""
    if profile is None:
        raise BundleWriteError("профиль исходного чипа обязателен: нужны crs и transform")
    result = {key: value for key, value in dict(profile).items() if value is not None}
    for key in ("crs", "transform"):
        if result.get(key) is None:
            raise BundleWriteError(f"в профиле исходного чипа нет поля {key!r}")
    height, width = shape
    for key, expected in (("height", height), ("width", width)):
        actual = result.get(key)
        if actual is not None and int(actual) != expected:
            raise BundleWriteError(
                f"размер растра не совпадает с профилем чипа: {key}={actual}, а массив даёт {expected}"
            )
    result.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        nodata=nodata,
        height=height,
        width=width,
    )
    return result


def _as_2d(array: Any, name: str) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise BundleWriteError(f"{name}: ожидается один канал, получено {data.shape}")
    return data


def write_probability_tif(
    path: Path | str,
    prob: np.ndarray,
    profile: Mapping[str, Any],
) -> Path:
    """``flood_probability.tif``: Float32, один канал, p ∈ [0;1], nodata = −9999."""
    path = _ensure_parent(Path(path))
    data = _as_2d(prob, "flood_probability").astype(np.float32, copy=True)
    nodata = np.float32(PROB_NODATA)
    valid = data != nodata
    if not np.isfinite(data[valid]).all():
        raise BundleWriteError(
            "flood_probability.tif: NaN или Inf среди валидных пикселей, "
            "пропуски обозначаются значением nodata -9999"
        )
    out_of_range = valid & ((data < 0.0) | (data > 1.0))
    if out_of_range.any():
        bad = data[out_of_range]
        raise BundleWriteError(
            "flood_probability.tif: вероятность вне [0;1] в "
            f"{int(out_of_range.sum())} пикселях, например {float(bad.flat[0])}"
        )
    with rasterio.open(
        path, "w", **_raster_profile(profile, dtype="float32", nodata=PROB_NODATA, shape=data.shape)
    ) as dst:
        dst.write(data, 1)
    return path


def write_mask_tif(
    path: Path | str,
    mask: np.ndarray,
    profile: Mapping[str, Any],
) -> Path:
    """``flood_mask.tif``: UInt8 на той же сетке, 1 вода, 0 не вода, 255 nodata."""
    path = _ensure_parent(Path(path))
    source = _as_2d(mask, "flood_mask")
    if np.issubdtype(source.dtype, np.floating) and not np.isfinite(source).all():
        raise BundleWriteError("flood_mask.tif: NaN или Inf в маске, пропуски обозначаются 255")
    data = source.astype(np.uint8, copy=True)
    allowed = {MASK_DRY, MASK_WATER, MASK_NODATA}
    present = set(int(v) for v in np.unique(data))
    if not present <= allowed:
        raise BundleWriteError(
            f"flood_mask.tif: допустимы только {sorted(allowed)}, найдены {sorted(present - allowed)}"
        )
    with rasterio.open(
        path, "w", **_raster_profile(profile, dtype="uint8", nodata=MASK_NODATA, shape=data.shape)
    ) as dst:
        dst.write(data, 1)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Объекты
# ─────────────────────────────────────────────────────────────────────────────


def _feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


def write_assets(
    path_geojson: Path | str,
    path_csv: Path | str,
    assets: Sequence[Mapping[str, Any]],
) -> tuple[Path, Path]:
    """``assets.geojson`` и ``assets.csv``: ровно 10 точек, координаты идентичны.

    Порядок координат в GeoJSON — [долгота, широта]. В оба файла уходит одно и то же
    текстовое представление числа, поэтому расхождение координат невозможно физически.
    """
    rows = [dict(asset) for asset in assets]
    if len(rows) != ASSET_COUNT:
        raise BundleWriteError(f"объектов должно быть ровно {ASSET_COUNT}, передано {len(rows)}")
    features: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        where = f"assets, объект {index + 1}"
        for column in COLUMNS_ASSETS:
            if row.get(column) is None or row.get(column) == CSV_EMPTY:
                raise BundleWriteError(f"{where}: поле {column} обязательно и не может быть пустым")
        lon_text = format_value(row["longitude"], f"{where}, longitude")
        lat_text = format_value(row["latitude"], f"{where}, latitude")
        properties = {
            key: value for key, value in row.items() if key not in ("longitude", "latitude")
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [Decimal(lon_text), Decimal(lat_text)]},
                "properties": properties,
            }
        )
        csv_row = dict(row)
        csv_row["longitude"] = lon_text
        csv_row["latitude"] = lat_text
        csv_rows.append(csv_row)
    geojson_path = write_json(path_geojson, _feature_collection(features))
    csv_path = write_csv(path_csv, COLUMNS_ASSETS, csv_rows)
    return geojson_path, csv_path


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def write_asset_loss(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    """``asset_loss.csv``: по строке на объект.

    Для статусов ``partial`` и ``no_data`` поля ``p_flood``, ``expected_loss_rub`` и
    ``rank`` остаются пустыми — ноль там означал бы измеренное отсутствие ущерба.
    """
    prepared = [dict(row) for row in rows]
    blank_for_status = ("p_flood", "expected_loss_rub", "rank")
    for index, row in enumerate(prepared):
        where = f"asset_loss.csv, строка {index + 1}"
        status = row.get("status")
        if status not in ASSET_STATUSES:
            raise BundleWriteError(f"{where}: статус {status!r} не из {list(ASSET_STATUSES)}")
        if status == ASSET_STATUS_OK:
            for column in blank_for_status:
                if _is_empty(row.get(column)):
                    raise BundleWriteError(f"{where}: при статусе ok поле {column} обязательно")
        else:
            for column in blank_for_status:
                if not _is_empty(row.get(column)):
                    raise BundleWriteError(
                        f"{where}: при статусе {status} поле {column} должно быть пустым, "
                        f"а не {row.get(column)!r}"
                    )
                row[column] = None
    return write_csv(path, COLUMNS_ASSET_LOSS, prepared)


# ─────────────────────────────────────────────────────────────────────────────
# Зоны заказа, закупка и стратегии
# ─────────────────────────────────────────────────────────────────────────────


def _as_geometry(value: Any, where: str) -> Any:
    """Геометрия кандидата: mapping GeoJSON или объект shapely."""
    if value is None:
        raise BundleWriteError(f"{where}: геометрия обязательна")
    if isinstance(value, Mapping):
        return value
    geo = getattr(value, "__geo_interface__", None)
    if geo is None:
        raise BundleWriteError(f"{where}: геометрия не похожа ни на GeoJSON, ни на shapely")
    return geo


def write_candidates(path: Path | str, features: Sequence[Mapping[str, Any]]) -> Path:
    """``candidate_orders.geojson``: каталог всех предложенных зон, не только выбранных.

    На вход принимается либо готовый Feature (``geometry`` + ``properties``), либо
    плоский словарь, где геометрия лежит в ``geometry``, а остальное — свойства.
    """
    collection: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(features):
        where = f"candidate_orders.geojson, зона {index + 1}"
        geometry = _as_geometry(item.get("geometry"), where)
        if "properties" in item and isinstance(item["properties"], Mapping):
            properties = dict(item["properties"])
        else:
            properties = {
                key: value for key, value in item.items() if key not in ("geometry", "type")
            }
        missing = [name for name in PROPERTIES_CANDIDATE if properties.get(name) is None]
        if missing:
            raise BundleWriteError(f"{where}: нет обязательных свойств {missing}")
        candidate_id = str(properties["candidate_id"])
        if candidate_id in seen:
            raise BundleWriteError(f"{where}: candidate_id {candidate_id!r} уже встречался")
        seen.add(candidate_id)
        properties["guaranteed_purchase"] = bool(properties["guaranteed_purchase"])
        ordered = {name: properties.pop(name) for name in PROPERTIES_CANDIDATE}
        ordered.update(properties)
        collection.append({"type": "Feature", "geometry": geometry, "properties": ordered})
    return write_json(path, _feature_collection(collection))


def write_procurement_plan(path: Path | str, positions: Sequence[Mapping[str, Any]]) -> Path:
    """``procurement_plan.csv``: только платные заказы B и C, для A строк нет.

    Пара ``(strategy, candidate_id)`` уникальна, ``guaranteed_purchase`` пишется
    строчными ``true``/``false``.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, position in enumerate(positions):
        where = f"procurement_plan.csv, строка {index + 1}"
        row = dict(position)
        strategy = str(row.get("strategy", ""))
        if strategy not in STRATEGIES:
            raise BundleWriteError(f"{where}: стратегия {strategy!r} не из {list(STRATEGIES)}")
        if strategy == STRATEGY_OPEN:
            raise BundleWriteError(
                f"{where}: стратегия A работает на открытых данных, платных позиций у неё нет"
            )
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            raise BundleWriteError(f"{where}: пустой candidate_id")
        key = (strategy, candidate_id)
        if key in seen:
            raise BundleWriteError(f"{where}: пара (strategy, candidate_id) {key} повторяется")
        seen.add(key)
        missing = [name for name in COLUMNS_PROCUREMENT if _is_empty(row.get(name))]
        if missing:
            raise BundleWriteError(f"{where}: не заполнены обязательные поля {missing}")
        row["guaranteed_purchase"] = bool(row["guaranteed_purchase"])
        rows.append(row)
    return write_csv(path, COLUMNS_PROCUREMENT, rows)


def write_strategy_plans(path: Path | str, plans: Mapping[str, Sequence[str]]) -> Path:
    """``strategy_plans.json``: ``{"A": [], "B": [...], "C": [...]}``."""
    result: dict[str, list[str]] = {}
    for strategy in STRATEGIES:
        chosen = plans.get(strategy)
        if chosen is None:
            raise BundleWriteError(f"strategy_plans.json: нет списка для стратегии {strategy}")
        ids = [str(item) for item in chosen]
        if len(set(ids)) != len(ids):
            raise BundleWriteError(
                f"strategy_plans.json: в плане стратегии {strategy} повторяются candidate_id"
            )
        if strategy == STRATEGY_OPEN and ids:
            raise BundleWriteError(
                "strategy_plans.json: стратегия A не закупает данные, её список должен быть пустым"
            )
        result[strategy] = ids
    return write_json(path, result)


def write_strategy_comparison(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    """``strategy_comparison.csv``: по строке на стратегию."""
    prepared = [dict(row) for row in rows]
    for index, row in enumerate(prepared):
        where = f"strategy_comparison.csv, строка {index + 1}"
        if row.get("strategy") not in STRATEGIES:
            raise BundleWriteError(f"{where}: стратегия {row.get('strategy')!r} не из {list(STRATEGIES)}")
        if row.get("budget_feasible") is not None:
            row["budget_feasible"] = bool(row["budget_feasible"])
    return write_csv(path, COLUMNS_STRATEGY_COMPARISON, prepared)


def write_sensitivity(path: Path | str, rows: Sequence[Mapping[str, Any]]) -> Path:
    """``sensitivity.csv``: одна строка — один вариант параметров."""
    prepared: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        where = f"sensitivity.csv, строка {index + 1}"
        item = dict(row)
        changed = item.get("changed_inputs_json")
        if isinstance(changed, Mapping):
            item["changed_inputs_json"] = json.dumps(
                _sanitize(changed, where), ensure_ascii=False, allow_nan=False, sort_keys=True
            )
        prepared.append(item)
    return write_csv(path, COLUMNS_SENSITIVITY, prepared)


# ─────────────────────────────────────────────────────────────────────────────
# Паспорт запуска и манифест источников
# ─────────────────────────────────────────────────────────────────────────────

#: Имена, под которыми валидатор ищет порог бинаризации в паспорте запуска.
THRESHOLD_KEYS: Sequence[str] = ("threshold", "mask_threshold", "decision_threshold", "p_threshold")

#: Имена, под которыми валидатор ищет путь к исходному S1-чипу.
SOURCE_CHIP_KEYS: Sequence[str] = (
    "source_chip_path",
    "chip_path",
    "s1_path",
    "s1_file",
    "s1_hand_path",
)


def write_run_metadata(path: Path | str, meta: Mapping[str, Any]) -> Path:
    """``run_metadata.json``: паспорт комплекта.

    Состав паспорта свободный, но три вещи обязательны: ``run_id``, ``chip_id`` и
    порог бинаризации — без порога проверка «маска ⟺ p ≥ threshold» невозможна.
    """
    data = dict(meta)
    for key in ("run_id", "chip_id"):
        if _is_empty(data.get(key)):
            raise BundleWriteError(f"run_metadata.json: поле {key} обязательно")
    if find_nested(data, THRESHOLD_KEYS) is None:
        raise BundleWriteError(
            "run_metadata.json: не найден порог бинаризации "
            f"(одно из полей {list(THRESHOLD_KEYS)}), без него пакет непроверяем"
        )
    return write_json(path, data)


def write_source_manifest(path: Path | str, entries: Iterable[Mapping[str, Any]]) -> Path:
    """``source_manifest.json``: все исходные данные с лицензией и ограничениями."""
    items = [dict(entry) for entry in entries]
    if not items:
        raise BundleWriteError("source_manifest.json: манифест источников не может быть пустым")
    return write_json(path, {"sources": items})
